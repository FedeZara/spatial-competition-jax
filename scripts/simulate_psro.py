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
from spatial_competition_jax.marl.psro.psro import _build_single_agent_policy, _build_wrapper
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
    # Prefer config embedded in checkpoint (guarantees matching arch),
    # fall back to CLI --config path.
    if "config" in ckpt and ckpt["config"] is not None:
        config = ckpt["config"]
        print("Config loaded from checkpoint (architecture guaranteed to match)")
    else:
        config_path = Path(__file__).parent.parent / args.config
        if config_path.exists():
            config = Config.from_yaml(config_path)
            print(f"Config loaded from {config_path}")
        else:
            print(f"Config not found: {config_path}, using defaults")
            config = Config()

    egocentric = config.train.observation_mode == "egocentric"

    with jax.default_device(device):
        # ── Wrapper, policy & renderer ────────────────────────────────
        wrapper = _build_wrapper(config)
        policy = _build_single_agent_policy(config, wrapper)

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
        print(f"Obs mode:     {'egocentric' if egocentric else 'global'}")
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

            if egocentric:
                ego_obs, env_state = wrapper.reset_ego(reset_key)
            else:
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
                    policy, wrapper, params0, params1,
                    ego_obs if egocentric else global_state,
                    egocentric=egocentric,
                )

                key, step_key = jax.random.split(key)

                if egocentric:
                    ego_obs, env_state, rewards, dones = wrapper.step_ego(
                        step_key, env_state, actions,
                        temperature=eval_temperature,
                    )
                else:
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


def _step_psro_policies(
    policy: Any,
    wrapper: Any,
    params0: Any,
    params1: Any,
    obs: jnp.ndarray,
    *,
    egocentric: bool = True,
) -> jnp.ndarray:
    """Compute deterministic actions for two single-agent PSRO policies.

    Returns:
        actions: ``(2, action_dim)``
    """
    if egocentric:
        # obs: (A, obs_dim) — agent 0 and agent 1's ego obs
        obs0 = obs[0:1][None, ...]  # (1, 1, obs_dim)
        obs1 = obs[1:2][None, ...]  # (1, 1, obs_dim)

        a0, _ = policy.deterministic(params0, obs0)
        a1, _ = policy.deterministic(params1, obs1)
        return jnp.concatenate([a0[0, 0:1], a1[0, 0:1]], axis=0)  # (2, action_dim)
    else:
        from spatial_competition_jax.marl.psro.state_utils import permute_agent_state

        # obs: (state_dim,)
        state_batch = obs[None, ...]  # (1, state_dim)

        a0, _ = policy.deterministic(params0, state_batch)
        agent0_action = a0[0, 0]  # (action_dim,)

        permuted = permute_agent_state(state_batch, wrapper)
        a1, _ = policy.deterministic(params1, permuted)
        agent1_action = a1[0, 0]  # (action_dim,)

        return jnp.stack([agent0_action, agent1_action], axis=0)


if __name__ == "__main__":
    main()
