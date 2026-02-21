#!/usr/bin/env python3
"""Train MAPPO on a Hotelling spatial-competition environment (JAX)."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import jax

from spatial_competition_jax.marl.config import Config
from spatial_competition_jax.marl.mappo.evaluation import evaluate_ego_policy, evaluate_policy
from spatial_competition_jax.marl.mappo.mappo import MAPPO, linear_anneal
from spatial_competition_jax.marl.mappo.networks import (
    DiscreteActorCritic,
    EgoActorCritic,
    EgoConv1dFactoredDiscreteActorCritic,
    EgoConv2dActorCritic,
    EgoDiscreteActorCritic,
    EgoFactoredDiscreteActorCritic,
    SharedActorCritic,
)
from spatial_competition_jax.marl.mappo.policy import (
    ContinuousPolicy,
    DiscretePolicy,
    EgoContinuousPolicy,
    EgoDiscretePolicy,
    EgoFactoredDiscretePolicy,
    PolicyAdapter,
)
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper
from spatial_competition_jax.marl.utils.checkpoints import save_checkpoint
from spatial_competition_jax.marl.utils.device import resolve_device
from spatial_competition_jax.marl.utils.logging import Logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train MAPPO (JAX) on Hotelling env")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/hotelling_1d.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--log-dir", type=str, default=None, help="Override log directory")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "JAX device to use, e.g. 'cpu', 'gpu', 'gpu:0', 'gpu:1', 'tpu'. "
            "Defaults to JAX default (first available accelerator)."
        ),
    )

    return parser.parse_args()


def build_policy(config: Config, wrapper: TrainingWrapper) -> PolicyAdapter:
    """Build the appropriate PolicyAdapter from config.

    Handles all combos: {global, egocentric} × {continuous, discrete}
    and the Conv1D variant when ``obs_type == "conv_bin"``.
    """
    hidden_dims = tuple(config.train.hidden_dims)
    ego = config.train.observation_mode == "egocentric"
    discrete = config.env.action_type == "discrete"
    conv_bin = config.train.obs_type == "conv_bin"

    ind_heads = config.train.independent_heads and config.train.independent

    if ego and discrete and conv_bin:
        # Conv1D network for spatial grid observations
        scalar_dim = wrapper.dimensions + 1
        if wrapper.include_quality:
            scalar_dim += 1
        net = EgoConv1dFactoredDiscreteActorCritic(
            num_location_bins=wrapper.num_location_bins,
            num_price_bins=wrapper.num_price_bins,
            spatial_resolution=wrapper.space_resolution,
            num_grid_channels=wrapper._conv_grid_channels,
            num_scalar_features=scalar_dim,
            mlp_hidden_dims=hidden_dims,
            independent_heads=ind_heads,
            num_agents=wrapper.num_agents,
        )
        return EgoFactoredDiscretePolicy(net, num_agents=wrapper.num_agents)

    if ego and not discrete and conv_bin:
        # Conv2D network for 2-D spatial grid observations (continuous actions)
        gp = wrapper.space_resolution + 1  # grid points per dimension
        scalar_dim = wrapper.dimensions + 1
        if wrapper.include_quality:
            scalar_dim += 1
        net = EgoConv2dActorCritic(
            movement_dim=wrapper.movement_dim,
            bounded_dim=wrapper.bounded_dim,
            spatial_resolution=gp,
            num_grid_channels=wrapper._conv_grid_channels,
            num_scalar_features=scalar_dim,
            mlp_hidden_dims=hidden_dims,
            independent_heads=ind_heads,
            num_agents=wrapper.num_agents,
        )
        return EgoContinuousPolicy(net, num_agents=wrapper.num_agents)

    if ego and discrete:
        net = EgoFactoredDiscreteActorCritic(
            num_location_bins=wrapper.num_location_bins,
            num_price_bins=wrapper.num_price_bins,
            hidden_dims=hidden_dims,
        )
        return EgoFactoredDiscretePolicy(net, num_agents=wrapper.num_agents)
    if ego:
        net = EgoActorCritic(movement_dim=wrapper.movement_dim, bounded_dim=wrapper.bounded_dim, hidden_dims=hidden_dims)
        return EgoContinuousPolicy(net, num_agents=wrapper.num_agents)
    if discrete:
        net = DiscreteActorCritic(num_actions=wrapper.num_actions, num_agents=wrapper.num_agents, hidden_dims=hidden_dims)
        return DiscretePolicy(net)

    net = SharedActorCritic(movement_dim=wrapper.movement_dim, bounded_dim=wrapper.bounded_dim, num_agents=wrapper.num_agents, hidden_dims=hidden_dims)
    return ContinuousPolicy(net)


def main() -> None:
    """Run the training loop."""
    args = parse_args()

    # ── device selection ───────────────────────────────────────────────
    device = resolve_device(args.device)

    # ── config ─────────────────────────────────────────────────────────
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = Config.from_yaml(config_path)
    else:
        print(f"Config not found: {config_path}, using defaults")
        config = Config()

    if args.seed is not None:
        config.train.seed = args.seed
    if args.log_dir is not None:
        config.train.log_dir = args.log_dir
    if args.no_tensorboard:
        config.train.use_tensorboard = False

    # ── experiment name ────────────────────────────────────────────────
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        action_tag = "disc" if config.env.action_type == "discrete" else "cont"
        experiment_name = f"hotelling_{config.env.dimensions}d_{action_tag}_{timestamp}"
    else:
        experiment_name = args.experiment_name

    # ── logger ─────────────────────────────────────────────────────────
    logger = Logger(
        log_dir=config.train.log_dir,
        use_tensorboard=config.train.use_tensorboard,
        experiment_name=experiment_name,
    )

    print("=" * 60)
    print("MAPPO Training (JAX)")
    print("=" * 60)
    print(f"Experiment:  {experiment_name}")
    print(f"Device:      {device}")
    print(f"Backend:     {jax.default_backend()}")
    print(f"Action type: {config.env.action_type}")
    print(f"Num envs:    {config.train.num_envs}")
    print(f"Rollout:     {config.train.rollout_length}")
    print(f"Updates:     {config.train.total_updates}")
    print("=" * 60)

    # All array creation and JIT compilation is scoped to *device*.
    with jax.default_device(device):
        _train(args, config, logger, experiment_name)


def _train(
    args: argparse.Namespace,
    config: Config,
    logger: Logger,
    experiment_name: str,
) -> None:
    """Inner training body – runs inside ``jax.default_device``."""
    # ── environment wrapper ────────────────────────────────────────────
    wrapper = TrainingWrapper(
        num_sellers=config.env.num_sellers,
        max_buyers=config.env.max_buyers,
        dimensions=config.env.dimensions,
        space_resolution=config.env.space_resolution,
        max_price=config.env.max_price,
        max_quality=config.env.max_quality,
        max_step_size=config.env.max_step_size,
        production_cost_factor=config.env.production_cost_factor,
        movement_cost=config.env.movement_cost,
        transport_cost=config.env.transport_cost,
        transportation_cost_norm=config.env.transportation_cost_norm,
        transport_cost_exponent=config.env.transport_cost_exponent,
        quality_taste=config.env.quality_taste,
        include_quality=config.env.include_quality,
        include_buyer_valuation=config.env.include_buyer_valuation,
        buyer_value=config.env.buyer_value,
        despawn_no_purchase=config.env.despawn_no_purchase,
        new_buyers_per_step=config.env.new_buyers_per_step,
        max_env_steps=config.env.max_env_steps,
        buyer_choice_temperature=config.env.buyer_choice_temperature,
        blob_sigma=config.train.blob_sigma,
        action_type=config.env.action_type,
        num_location_bins=config.env.num_location_bins,
        num_price_bins=config.env.num_price_bins,
        obs_type=config.train.obs_type,
        buyer_distribution=config.env.buyer_distribution,
        buyer_dist_means=config.env.buyer_dist_means,
        buyer_dist_stds=config.env.buyer_dist_stds,
        buyer_dist_weights=config.env.buyer_dist_weights,
    )

    # ── independent PPO: append agent IDs to ego obs ────────────────
    if config.train.independent and config.train.observation_mode == "egocentric":
        wrapper.enable_agent_id()
        print(f"Independent: True (agent ID appended to obs)")

    # ── policy adapter ────────────────────────────────────────────────
    use_ego = config.train.observation_mode == "egocentric"
    policy = build_policy(config, wrapper)

    print(f"Obs mode:    {'egocentric' if use_ego else 'global (per-agent heads)'}")
    print(f"Obs dim:     {wrapper.obs_dim}")
    if config.env.action_type == "discrete":
        print(f"Num actions: {wrapper.num_actions} "
              f"({config.env.num_location_bins} loc × {config.env.num_price_bins} price)")
    else:
        print(f"Action dim:  {wrapper.action_dim}")
    print(f"Num agents:  {wrapper.num_agents}")

    # ── MAPPO agent ────────────────────────────────────────────────────
    agent = MAPPO(
        wrapper=wrapper,
        policy=policy,
        num_envs=config.train.num_envs,
        rollout_length=config.train.rollout_length,
        gamma=config.train.gamma,
        gae_lambda=config.train.gae_lambda,
        clip_epsilon=config.train.clip_epsilon,
        value_coef=config.train.value_coef,
        entropy_coef=config.train.entropy_coef,
        learning_rate=config.train.learning_rate,
        max_grad_norm=config.train.max_grad_norm,
        ppo_epochs=config.train.ppo_epochs,
        num_minibatches=config.train.num_minibatches,
        target_kl=config.train.target_kl,
        seed=config.train.seed,
    )

    # Verify all initial arrays landed on the chosen device.
    _check_device = jax.tree.leaves(agent.train_state.params)[0]
    print(f"Params on:   {_check_device.devices()}")

    # ── annealing setup ────────────────────────────────────────────────
    use_temp_anneal = (
        config.env.buyer_choice_temperature is not None and config.train.buyer_choice_temp_start is not None
    )
    if use_temp_anneal:
        temp_start = config.train.buyer_choice_temp_start
        assert temp_start is not None  # for type checker
        temp_end = config.train.buyer_choice_temp_end
        temp_anneal_frac = config.train.buyer_choice_temp_anneal_frac
        print(f"Temperature annealing: {temp_start} → {temp_end} over {temp_anneal_frac * 100:.0f}% of training")

    use_entropy_anneal = config.train.entropy_coef_start is not None
    if use_entropy_anneal:
        ent_start = config.train.entropy_coef_start
        assert ent_start is not None  # for type checker
        ent_end = config.train.entropy_coef_end
        ent_anneal_frac = config.train.entropy_coef_anneal_frac
        print(f"Entropy coef annealing: {ent_start} → {ent_end} over {ent_anneal_frac * 100:.0f}% of training")

    # ── training loop ──────────────────────────────────────────────────
    total_timesteps = 0
    best_eval_reward = float("-inf")
    update = 0

    print("\nStarting training …")
    print("-" * 60)

    try:
        for update in range(1, config.train.total_updates + 1):
            # Compute current temperature
            if use_temp_anneal:
                assert temp_start is not None  # for type checker
                temperature: float | None = linear_anneal(
                    update,
                    config.train.total_updates,
                    temp_start,
                    temp_end,
                    temp_anneal_frac,
                )
            else:
                temperature = None

            # Compute current entropy coefficient
            if use_entropy_anneal:
                assert ent_start is not None  # for type checker
                cur_entropy_coef: float | None = linear_anneal(
                    update,
                    config.train.total_updates,
                    ent_start,
                    ent_end,
                    ent_anneal_frac,
                )
            else:
                cur_entropy_coef = None

            # Collect rollout
            transitions, advantages, returns, rollout_stats = agent.collect_rollout(
                temperature=temperature,
            )
            total_timesteps += config.train.rollout_length * config.train.num_envs * wrapper.num_agents

            # PPO update
            update_stats = agent.update(
                transitions,
                advantages,
                returns,
                entropy_coef=cur_entropy_coef,
            )

            # Logging
            logger.set_step(update)

            if update % config.train.log_interval == 0:
                metrics = {
                    **rollout_stats,
                    **update_stats,
                    "timesteps": total_timesteps,
                }
                if temperature is not None:
                    metrics["temperature"] = temperature
                logger.log_metrics(metrics, prefix="train")

                print_metrics: dict[str, float] = {
                    "reward": rollout_stats["mean_reward"],
                    "policy_loss": update_stats["policy_loss"],
                    "value_loss": update_stats["value_loss"],
                    "entropy": update_stats["entropy"],
                }
                if temperature is not None:
                    print_metrics["temp"] = temperature
                if cur_entropy_coef is not None:
                    print_metrics["ent_coef"] = cur_entropy_coef
                logger.print_metrics(
                    print_metrics,
                    step=update,
                    prefix="Train",
                )

            # Evaluation
            if update % config.train.eval_interval == 0:
                if use_ego:
                    eval_stats = evaluate_ego_policy(
                        network=agent.network,
                        params=agent.params,
                        wrapper=wrapper,
                        num_episodes=config.train.eval_episodes,
                        deterministic=config.train.deterministic_eval,
                        temperature=config.train.buyer_choice_temp_end if use_temp_anneal else None,
                        is_discrete=config.env.action_type == "discrete",
                        is_factored=config.env.action_type == "discrete",
                    )
                else:
                    eval_stats = evaluate_policy(
                        policy=policy,
                        params=agent.params,
                        wrapper=wrapper,
                        num_episodes=config.train.eval_episodes,
                        deterministic=config.train.deterministic_eval,
                        temperature=config.train.buyer_choice_temp_end if use_temp_anneal else None,
                    )
                logger.log_metrics(eval_stats, prefix="eval")

                eval_log: dict[str, float] = {
                    "reward": eval_stats["eval_reward_mean"],
                }
                if "eval_seller_distance" in eval_stats:
                    eval_log["distance"] = eval_stats["eval_seller_distance"]
                eval_log["price"] = eval_stats["eval_price_mean"]
                logger.print_metrics(eval_log, step=update, prefix="Eval")

                if eval_stats["eval_reward_mean"] > best_eval_reward:
                    best_eval_reward = eval_stats["eval_reward_mean"]
                    save_checkpoint(
                        path=logger.experiment_dir / "best_model.pkl",
                        step=update,
                        params=agent.params,
                        opt_state=agent.opt_state,
                        metrics=eval_stats,
                    )
                    print(f"  → New best model (reward: {best_eval_reward:.2f})")

            # Periodic checkpoint
            if update % config.train.save_interval == 0:
                save_checkpoint(
                    path=logger.experiment_dir / f"checkpoint_{update}.pkl",
                    step=update,
                    params=agent.params,
                    opt_state=agent.opt_state,
                )

    except KeyboardInterrupt:
        print("\n\nTraining interrupted.")

    # ── final save ─────────────────────────────────────────────────────
    print("-" * 60)
    save_checkpoint(
        path=logger.experiment_dir / "final_model.pkl",
        step=update,
        params=agent.params,
        opt_state=agent.opt_state,
    )

    # ── final evaluation ───────────────────────────────────────────────
    print("Final evaluation …")
    if use_ego:
        final_eval = evaluate_ego_policy(
            network=agent.network,
            params=agent.params,
            wrapper=wrapper,
            num_episodes=20,
            deterministic=True,
            temperature=config.train.buyer_choice_temp_end if use_temp_anneal else None,
            is_discrete=config.env.action_type == "discrete",
            is_factored=config.env.action_type == "discrete",
        )
    else:
        final_eval = evaluate_policy(
            policy=policy,
            params=agent.params,
            wrapper=wrapper,
            num_episodes=20,
            deterministic=True,
            temperature=config.train.buyer_choice_temp_end if use_temp_anneal else None,
        )
    print(f"  Reward:   {final_eval['eval_reward_mean']:.2f} ± {final_eval['eval_reward_std']:.2f}")
    print(f"  Position: {final_eval['eval_position_mean']:.3f}")
    print(f"  Price:    {final_eval['eval_price_mean']:.2f}")
    if "eval_seller_distance" in final_eval:
        print(f"  Distance: {final_eval['eval_seller_distance']:.3f}")

    logger.close()

    print("=" * 60)
    print(f"Training complete! Results: {logger.experiment_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
