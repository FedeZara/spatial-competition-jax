#!/usr/bin/env python3
"""Run a visual simulation of trained MAPPO agents with Pygame rendering.

Usage::

    poetry run python scripts/simulate.py \\
        --checkpoint results/<experiment>/best_model.pkl \\
        --config configs/hotelling_1d.yaml

Controls:
    Space       Pause / resume
    Escape      Deselect entity
    Mouse       Click sellers / buyers for details
    Slider      Adjust playback speed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from spatial_competition_jax.marl.config import Config
from spatial_competition_jax.marl.mappo.networks import (
    SharedActorCritic,
    deterministic_actions,
    sample_actions,
)
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper
from spatial_competition_jax.marl.utils.checkpoints import load_checkpoint
from spatial_competition_jax.marl.utils.device import resolve_device
from spatial_competition_jax.renderer import SpatialCompetitionRenderer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a visual simulation of trained MAPPO agents",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pkl checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hotelling_1d.yaml",
        help="Config YAML",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes to simulate",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help="Base delay between steps in seconds",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="JAX device, e.g. 'cpu', 'gpu', 'gpu:0'",
    )

    return parser.parse_args()


def main() -> None:
    """Run the visual simulation."""
    args = parse_args()

    # ── device selection ───────────────────────────────────────────────
    device = resolve_device(args.device)

    print("=" * 60)
    print("MAPPO Simulation (JAX + Pygame)")
    print("=" * 60)
    print(f"Device: {device}")

    # ── checkpoint ─────────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    print(f"Loading: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)
    print(f"Step: {checkpoint['step']}")

    # ── config ─────────────────────────────────────────────────────────
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = Config.from_yaml(config_path)
    else:
        print(f"Config not found: {config_path}, using defaults")
        config = Config()

    with jax.default_device(device):
        # ── wrapper ────────────────────────────────────────────────────
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
            new_buyers_per_step=config.env.new_buyers_per_step,
            max_env_steps=config.env.max_env_steps,
            buyer_choice_temperature=config.env.buyer_choice_temperature,
            blob_sigma=config.train.blob_sigma,
        )

        # ── network ───────────────────────────────────────────────────
        network = SharedActorCritic(
            movement_dim=wrapper.movement_dim,
            bounded_dim=wrapper.bounded_dim,
            num_agents=wrapper.num_agents,
            hidden_dims=tuple(config.train.hidden_dims),
        )
        params = jax.device_put(checkpoint["params"], device)

        # ── renderer ──────────────────────────────────────────────────
        renderer = SpatialCompetitionRenderer(
            wrapper.env,
            max_env_steps=config.env.max_env_steps,
        )

        # ── temperature ───────────────────────────────────────────────
        eval_temperature: float | None = None
        if config.env.buyer_choice_temperature is not None:
            eval_temperature = config.train.buyer_choice_temp_end

        deterministic = not args.stochastic
        key = jax.random.PRNGKey(args.seed)
        max_steps = config.env.max_env_steps
        running = True

        print(f"\nDeterministic: {deterministic}")
        print(f"Episodes:      {args.num_episodes}")
        print(f"Max steps:     {max_steps}")
        print(f"Base delay:    {args.delay}s")
        print("-" * 60)
        print("Controls: Space=pause, click entities for info, slider=speed")
        print("-" * 60)

        for episode in range(args.num_episodes):
            if not running:
                break

            key, reset_key = jax.random.split(key)
            global_state, env_state = wrapper.reset(reset_key)

            cumulative_rewards: dict[str, float] = {f"seller_{i}": 0.0 for i in range(wrapper.num_agents)}

            print(f"\n▶ Episode {episode + 1}/{args.num_episodes}")

            # Render initial state
            running = renderer.render_and_wait(
                env_state,
                base_delay=args.delay,
                current_step=0,
                cumulative_rewards=cumulative_rewards,
            )
            if not running:
                break

            for step in range(1, max_steps + 1):
                key, action_key = jax.random.split(key)

                state_batch = global_state[None, ...]  # add batch dim

                if deterministic:
                    actions, _ = deterministic_actions(network, params, state_batch)
                else:
                    actions, _, _ = sample_actions(network, params, state_batch, action_key)

                actions = actions[0]  # remove batch dim → (A, action_dim)

                key, step_key = jax.random.split(key)
                global_state, env_state, rewards, dones = wrapper.step(
                    step_key,
                    env_state,
                    actions,
                    temperature=eval_temperature,
                )

                # Update cumulative rewards
                rewards_np = np.asarray(rewards)
                for i in range(wrapper.num_agents):
                    cumulative_rewards[f"seller_{i}"] += float(rewards_np[i])

                # Render and wait
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

            # Print episode summary
            total = sum(cumulative_rewards.values())
            print(f"  Total reward: {total:.2f}")
            for name, rew in cumulative_rewards.items():
                pos = (
                    np.asarray(
                        env_state.seller_positions[int(name.split("_")[1])],
                        dtype=np.float64,
                    )
                    / wrapper.space_resolution
                )
                price = float(np.asarray(env_state.seller_prices[int(name.split("_")[1])]))
                print(f"    {name}: reward={rew:.2f}  pos={pos}  price={price:.2f}")

    renderer.close()
    print("\n" + "=" * 60)
    print("Simulation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
