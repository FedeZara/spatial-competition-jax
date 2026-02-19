#!/usr/bin/env python3
"""Evaluate trained MAPPO agents (JAX)."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

from spatial_competition_jax.marl.config import Config
from spatial_competition_jax.marl.mappo.evaluation import evaluate_ego_policy, evaluate_policy
from spatial_competition_jax.marl.mappo.networks import (
    DiscreteActorCritic,
    EgoActorCritic,
    EgoConv1dFactoredDiscreteActorCritic,
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
from spatial_competition_jax.marl.utils.checkpoints import load_checkpoint
from spatial_competition_jax.marl.utils.device import resolve_device


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained MAPPO agent (JAX)")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pkl checkpoint")
    parser.add_argument("--config", type=str, default="configs/hotelling_1d.yaml", help="Config YAML")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
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


def build_wrapper(config: Config) -> TrainingWrapper:
    """Build a TrainingWrapper from config."""
    return TrainingWrapper(
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
        new_buyers_per_step=config.env.new_buyers_per_step,
        max_env_steps=config.env.max_env_steps,
        buyer_choice_temperature=config.env.buyer_choice_temperature,
        blob_sigma=config.train.blob_sigma,
        action_type=config.env.action_type,
        num_location_bins=config.env.num_location_bins,
        num_price_bins=config.env.num_price_bins,
        obs_type=config.train.obs_type,
    )


def build_policy(config: Config, wrapper: TrainingWrapper) -> PolicyAdapter:
    """Build the appropriate PolicyAdapter from config.

    Handles all combos: {global, egocentric} × {continuous, discrete}
    and the Conv1D variant when ``obs_type == "conv_bin"``.
    """
    from typing import Any

    hidden_dims = tuple(config.train.hidden_dims)
    ego = config.train.observation_mode == "egocentric"
    discrete = config.env.action_type == "discrete"
    conv_bin = config.train.obs_type == "conv_bin"

    net: Any  # Flax module — concrete type varies by branch

    if ego and discrete and conv_bin:
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
        )
        return EgoFactoredDiscretePolicy(net, num_agents=wrapper.num_agents)

    if ego and discrete:
        net = EgoFactoredDiscreteActorCritic(
            num_location_bins=wrapper.num_location_bins,
            num_price_bins=wrapper.num_price_bins,
            hidden_dims=hidden_dims,
        )
        return EgoFactoredDiscretePolicy(net, num_agents=wrapper.num_agents)

    if ego:
        net = EgoActorCritic(
            movement_dim=wrapper.movement_dim,
            bounded_dim=wrapper.bounded_dim,
            hidden_dims=hidden_dims,
        )
        return EgoContinuousPolicy(net, num_agents=wrapper.num_agents)

    if discrete:
        net = DiscreteActorCritic(
            num_actions=wrapper.num_actions,
            num_agents=wrapper.num_agents,
            hidden_dims=hidden_dims,
        )
        return DiscretePolicy(net)

    net = SharedActorCritic(
        movement_dim=wrapper.movement_dim,
        bounded_dim=wrapper.bounded_dim,
        num_agents=wrapper.num_agents,
        hidden_dims=hidden_dims,
    )
    return ContinuousPolicy(net)


def main() -> None:
    """Run evaluation."""
    args = parse_args()

    # ── device selection ───────────────────────────────────────────────
    device = resolve_device(args.device)

    print("=" * 60)
    print("MAPPO Evaluation (JAX)")
    print("=" * 60)
    print(f"Device: {device}")

    # ── checkpoint ─────────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    print(f"Loading: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)
    print(f"Step: {checkpoint['step']}")
    if "metrics" in checkpoint:
        print(f"Saved metrics: {checkpoint['metrics']}")

    # ── config ─────────────────────────────────────────────────────────
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = Config.from_yaml(config_path)
    else:
        print(f"Config not found: {config_path}, using defaults")
        config = Config()

    use_ego = config.train.observation_mode == "egocentric"
    is_discrete = config.env.action_type == "discrete"

    with jax.default_device(device):
        # ── wrapper ────────────────────────────────────────────────────
        wrapper = build_wrapper(config)

        if config.train.independent and use_ego:
            wrapper.enable_agent_id()

        # ── policy ────────────────────────────────────────────────────
        policy = build_policy(config, wrapper)
        params = jax.device_put(checkpoint["params"], device)

        # ── evaluate ──────────────────────────────────────────────────
        key = jax.random.PRNGKey(args.seed)

        print("-" * 60)
        print(f"Running {args.num_episodes} episodes …")
        print(f"Deterministic: {not args.stochastic}")
        print(f"Observation mode: {'egocentric' if use_ego else 'global'}")
        print(f"Action type: {'discrete' if is_discrete else 'continuous'}")
        print("-" * 60)

        # Use near-deterministic temperature for evaluation when softmax
        # buyer choice is enabled.
        eval_temperature: float | None = None
        if config.env.buyer_choice_temperature is not None:
            eval_temperature = config.train.buyer_choice_temp_end

        if use_ego:
            results = evaluate_ego_policy(
                network=policy.network,  # type: ignore[attr-defined]
                params=params,
                wrapper=wrapper,
                num_episodes=args.num_episodes,
                deterministic=not args.stochastic,
                key=key,
                temperature=eval_temperature,
                is_discrete=is_discrete,
                is_factored=is_discrete,  # all ego+discrete use factored now
            )
        else:
            results = evaluate_policy(
                policy=policy,
                params=params,
                wrapper=wrapper,
                num_episodes=args.num_episodes,
                deterministic=not args.stochastic,
                key=key,
                temperature=eval_temperature,
            )

    # ── results ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Reward:   {results['eval_reward_mean']:8.2f} ± {results['eval_reward_std']:.2f}")
    print(f"  Length:   {results['eval_length_mean']:.1f}")
    print(f"  Position: {results['eval_position_mean']:.3f}")
    print(f"  Price:    {results['eval_price_mean']:.2f}")
    if "eval_seller_distance" in results:
        print(f"  Distance: {results['eval_seller_distance']:.3f}")

    # Hotelling analysis
    print()
    if "eval_seller_distance" in results:
        d = results["eval_seller_distance"]
        if d < 0.1:
            print("  → Minimum differentiation (sellers clustered)")
        elif d > 0.6:
            print("  → Maximum differentiation (sellers spread)")
        else:
            print("  → Intermediate differentiation")

    print("=" * 60)


if __name__ == "__main__":
    main()
