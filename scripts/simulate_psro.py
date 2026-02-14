#!/usr/bin/env python3
"""Run a visual simulation of two PSRO single-agent policies.

Loads a PSRO checkpoint (.pkl) and pits two policies from the
population against each other with Pygame rendering.

Usage::

    poetry run python scripts/simulate_psro.py \
        --checkpoint results/psro_1d_*/psro_iter_5.pkl \
        --config configs/psro_hotelling_1d.yaml

    # Specific policy indices (default: Nash meta-strategy sampling)
    poetry run python scripts/simulate_psro.py \
        --checkpoint results/psro_1d_*/psro_final.pkl \
        --config configs/psro_hotelling_1d.yaml \
        --policy0 2 --policy1 4

    # Sample from meta-strategy each episode
    poetry run python scripts/simulate_psro.py \
        --checkpoint results/psro_1d_*/psro_final.pkl \
        --config configs/psro_hotelling_1d.yaml \
        --sample-nash

Controls:
    Space       Pause / resume
    Escape      Deselect entity
    Mouse       Click sellers / buyers for details
    Slider      Adjust playback speed
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from spatial_competition_jax.marl.config import Config
from spatial_competition_jax.marl.mappo.networks import (
    SharedActorCritic,
    deterministic_actions,
)
from spatial_competition_jax.marl.psro.state_utils import permute_agent_state
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper
from spatial_competition_jax.marl.utils.device import resolve_device
from spatial_competition_jax.renderer import SpatialCompetitionRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate two PSRO policies against each other",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to PSRO .pkl checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default="configs/psro_hotelling_1d.yaml",
        help="Config YAML",
    )
    parser.add_argument(
        "--policy0", type=int, default=None,
        help="Population index for agent 0 (default: last trained)",
    )
    parser.add_argument(
        "--policy1", type=int, default=None,
        help="Population index for agent 1 (default: second-to-last)",
    )
    parser.add_argument(
        "--sample-nash", action="store_true",
        help="Sample both policies from the meta-strategy each episode",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=5,
        help="Number of episodes to simulate",
    )
    parser.add_argument(
        "--delay", type=float, default=0.15,
        help="Base delay between steps in seconds",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default=None,
        help="JAX device, e.g. 'cpu', 'gpu', 'gpu:0'",
    )
    return parser.parse_args()


def _build_wrapper(config: Config) -> TrainingWrapper:
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
    )


def _step_psro_policies(
    network: SharedActorCritic,
    wrapper: TrainingWrapper,
    params0: Any,
    params1: Any,
    global_state: jnp.ndarray,
) -> jnp.ndarray:
    """Compute deterministic actions for two single-agent PSRO policies.

    Policy 0 sees the original state; policy 1 sees the permuted state
    (so it thinks it is agent 0).

    Returns:
        actions: ``(2, action_dim)``
    """
    state_batch = global_state[None, ...]  # (1, state_dim)

    # Agent 0
    actions0, _ = deterministic_actions(network, params0, state_batch)
    agent0_action = actions0[0, 0]  # (action_dim,)

    # Agent 1 (permuted view)
    permuted = permute_agent_state(state_batch, wrapper)
    actions1, _ = deterministic_actions(network, params1, permuted)
    agent1_action = actions1[0, 0]  # (action_dim,)

    return jnp.stack([agent0_action, agent1_action], axis=0)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    # ── Load checkpoint ───────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    print(f"Loading: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        ckpt: dict[str, Any] = pickle.load(f)  # noqa: S301

    population = ckpt["population"]
    meta_strategy = ckpt.get("meta_strategy")
    payoff_matrix = ckpt.get("payoff_matrix")
    K = len(population)

    print(f"Population size: {K}")
    if meta_strategy is not None:
        print(f"Meta-strategy:   {np.array2string(np.asarray(meta_strategy), precision=3)}")
    if payoff_matrix is not None:
        print(f"Payoff matrix shape: {np.asarray(payoff_matrix).shape}")

    # ── Config ────────────────────────────────────────────────────────
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = Config.from_yaml(config_path)
    else:
        print(f"Config not found: {config_path}, using defaults")
        config = Config()

    with jax.default_device(device):
        # ── Wrapper & network ─────────────────────────────────────────
        wrapper = _build_wrapper(config)
        network = SharedActorCritic(
            movement_dim=wrapper.movement_dim,
            bounded_dim=wrapper.bounded_dim,
            num_agents=1,
            hidden_dims=tuple(config.train.hidden_dims),
        )

        # ── Renderer ──────────────────────────────────────────────────
        renderer = SpatialCompetitionRenderer(
            wrapper.env,
            max_env_steps=config.env.max_env_steps,
        )

        # ── Temperature ───────────────────────────────────────────────
        eval_temperature: float | None = None
        if config.env.buyer_choice_temperature is not None:
            eval_temperature = config.train.buyer_choice_temp_end

        key = jax.random.PRNGKey(args.seed)
        rng = np.random.default_rng(args.seed)
        max_steps = config.env.max_env_steps
        running = True

        print("=" * 60)
        print("PSRO Simulation (JAX + Pygame)")
        print("=" * 60)
        print(f"Device:       {device}")
        print(f"Episodes:     {args.num_episodes}")
        print(f"Max steps:    {max_steps}")
        print(f"Base delay:   {args.delay}s")
        print("-" * 60)
        print("Controls: Space=pause, click entities for info, slider=speed")
        print("-" * 60)

        for episode in range(args.num_episodes):
            if not running:
                break

            # ── Select policies for this episode ──────────────────────
            if args.sample_nash and meta_strategy is not None:
                ms = np.asarray(meta_strategy, dtype=np.float64)
                ms = np.maximum(ms, 0.0)
                ms /= ms.sum()
                idx0 = int(rng.choice(K, p=ms))
                idx1 = int(rng.choice(K, p=ms))
            else:
                idx0 = args.policy0 if args.policy0 is not None else K - 1
                idx1 = args.policy1 if args.policy1 is not None else max(K - 2, 0)

            params0 = jax.device_put(population[idx0], device)
            params1 = jax.device_put(population[idx1], device)

            print(f"\n▶ Episode {episode + 1}/{args.num_episodes}  "
                  f"(agent 0 = policy {idx0}, agent 1 = policy {idx1})")

            # ── Reset ─────────────────────────────────────────────────
            key, reset_key = jax.random.split(key)
            global_state, env_state = wrapper.reset(reset_key)
            cumulative_rewards = {"seller_0": 0.0, "seller_1": 0.0}

            running = renderer.render_and_wait(
                env_state,
                base_delay=args.delay,
                current_step=0,
                cumulative_rewards=cumulative_rewards,
            )
            if not running:
                break

            # ── Episode loop ──────────────────────────────────────────
            for step in range(1, max_steps + 1):
                actions = _step_psro_policies(
                    network, wrapper, params0, params1, global_state,
                )

                key, step_key = jax.random.split(key)
                global_state, env_state, rewards, dones = wrapper.step(
                    step_key, env_state, actions,
                    temperature=eval_temperature,
                )

                rewards_np = np.asarray(rewards)
                cumulative_rewards["seller_0"] += float(rewards_np[0])
                cumulative_rewards["seller_1"] += float(rewards_np[1])

                running = renderer.render_and_wait(
                    env_state,
                    base_delay=args.delay,
                    current_step=step,
                    cumulative_rewards=cumulative_rewards,
                )
                if not running:
                    break
                if bool(dones[0]):
                    break

            # ── Episode summary ───────────────────────────────────────
            total = sum(cumulative_rewards.values())
            print(f"  Total reward: {total:.2f}")
            for agent_id in range(2):
                name = f"seller_{agent_id}"
                policy_idx = idx0 if agent_id == 0 else idx1
                pos = (
                    np.asarray(env_state.seller_positions[agent_id], dtype=np.float64)
                    / wrapper.space_resolution
                )
                price = float(np.asarray(env_state.seller_prices[agent_id]))
                print(f"    {name} (policy {policy_idx}): "
                      f"reward={cumulative_rewards[name]:.2f}  "
                      f"pos={pos}  price={price:.2f}")

    renderer.close()
    print("\n" + "=" * 60)
    print("Simulation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
