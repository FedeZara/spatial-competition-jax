"""Policy evaluation utilities for MAPPO."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from spatial_competition_jax.marl.mappo.networks import (
    EgoActorCritic,
    EgoDiscreteActorCritic,
    ego_2d_factored_discrete_deterministic,
    ego_2d_factored_discrete_sample,
    ego_deterministic_actions,
    ego_discrete_deterministic,
    ego_discrete_sample,
    ego_factored_discrete_deterministic,
    ego_factored_discrete_sample,
    ego_sample_actions,
)
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper

if TYPE_CHECKING:
    from spatial_competition_jax.marl.mappo.policy import PolicyAdapter

# ---------------------------------------------------------------------------
# JIT-compiled, vmapped evaluation kernel
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(0, 1, 4, 5))
def _eval_episodes_jit(
    policy: PolicyAdapter,
    wrapper: TrainingWrapper,
    params: Any,
    keys: jnp.ndarray,
    deterministic: bool,
    use_temp: bool,
    temperature: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run evaluation episodes in parallel.

    Returns:
        ``(total_rewards, final_positions, final_prices, total_sales)``
        with shapes ``(E, A)``, ``(E, S, D)``, ``(E, S)``, ``(E, S)``.
    """
    max_steps = wrapper.env.max_env_steps

    def run_one(key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        reset_key, step_key = jax.random.split(key)
        global_state, env_state = wrapper.reset(reset_key)

        def scan_fn(
            carry: tuple[Any, jnp.ndarray, jnp.ndarray],
            _: None,
        ) -> tuple[tuple[Any, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
            env_state, global_state, sk = carry
            sk, action_key = jax.random.split(sk)

            state_batch = global_state[None, ...]

            if deterministic:
                actions, _ = policy.deterministic(params, state_batch)
            else:
                actions, _, _ = policy.sample(params, state_batch, action_key)
            actions = actions[0]

            if use_temp:
                next_gs, next_es, rewards, _dones = wrapper.step(
                    sk, env_state, actions, temperature=temperature,
                )
            else:
                next_gs, next_es, rewards, _dones = wrapper.step(
                    sk, env_state, actions,
                )

            return (next_es, next_gs, sk), (rewards, next_es.seller_running_sales)

        (final_env_state, _, _), (all_rewards, all_sales) = jax.lax.scan(
            scan_fn,
            (env_state, global_state, step_key),
            None,
            length=max_steps,
        )

        total_reward = all_rewards.sum(axis=0)  # (A,)
        total_sales = all_sales.sum(axis=0)  # (A,)
        positions = (
            final_env_state.seller_positions.astype(jnp.float32)
            / wrapper.space_resolution
        )
        prices = final_env_state.seller_prices
        qualities = final_env_state.seller_qualities

        return total_reward, positions, prices, total_sales, qualities

    return jax.vmap(run_one)(keys)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_policy(
    policy: PolicyAdapter,
    params: dict,
    wrapper: TrainingWrapper,
    num_episodes: int = 10,
    deterministic: bool = True,
    key: jnp.ndarray | None = None,
    temperature: float | None = None,
) -> dict[str, float]:
    """Evaluate a trained policy with rich per-agent metrics."""
    if key is None:
        key = jax.random.PRNGKey(0)

    use_temp = wrapper.env.buyer_choice_temperature is not None
    temp_arr = jnp.float32(temperature if temperature is not None else 0.0)
    keys = jax.random.split(key, num_episodes)

    total_rewards, all_positions, all_prices, total_sales, all_qualities = _eval_episodes_jit(
        policy, wrapper, params, keys, deterministic, use_temp, temp_arr,
    )

    return _build_eval_results(
        total_rewards, all_positions, all_prices, total_sales, wrapper, all_qualities
    )


# ---------------------------------------------------------------------------
# Egocentric evaluation
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(0, 1, 4, 5, 6, 7, 8))
def _eval_ego_episodes_jit(
    network: EgoActorCritic | EgoDiscreteActorCritic,
    wrapper: TrainingWrapper,
    params: Any,
    keys: jnp.ndarray,
    deterministic: bool,
    use_temp: bool,
    is_discrete: bool,
    is_factored: bool,
    is_2d_factored: bool,
    temperature: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Egocentric evaluation with per-agent sales tracking."""
    max_steps = wrapper.env.max_env_steps

    def run_one(key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        reset_key, step_key = jax.random.split(key)
        obs_all, env_state = wrapper.reset_ego(reset_key)

        def scan_fn(
            carry: tuple[Any, jnp.ndarray, jnp.ndarray],
            _: None,
        ) -> tuple[tuple[Any, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
            env_state, obs_all, sk = carry
            sk, action_key = jax.random.split(sk)

            if is_2d_factored:
                if deterministic:
                    actions, _ = ego_2d_factored_discrete_deterministic(network, params, obs_all)  # type: ignore[arg-type]
                else:
                    actions, _, _ = ego_2d_factored_discrete_sample(network, params, obs_all, action_key)  # type: ignore[arg-type]
            elif is_discrete and is_factored:
                if deterministic:
                    actions, _ = ego_factored_discrete_deterministic(network, params, obs_all)  # type: ignore[arg-type]
                else:
                    actions, _, _ = ego_factored_discrete_sample(network, params, obs_all, action_key)  # type: ignore[arg-type]
            elif is_discrete:
                if deterministic:
                    actions, _ = ego_discrete_deterministic(network, params, obs_all)  # type: ignore[arg-type]
                else:
                    actions, _, _ = ego_discrete_sample(network, params, obs_all, action_key)  # type: ignore[arg-type]
            else:
                if deterministic:
                    actions, _ = ego_deterministic_actions(network, params, obs_all)  # type: ignore[arg-type]
                else:
                    actions, _, _ = ego_sample_actions(network, params, obs_all, action_key)  # type: ignore[arg-type]

            if use_temp:
                next_obs, next_es, rewards, _dones = wrapper.step_ego(
                    sk, env_state, actions, temperature=temperature,
                )
            else:
                next_obs, next_es, rewards, _dones = wrapper.step_ego(
                    sk, env_state, actions,
                )

            return (next_es, next_obs, sk), (rewards, next_es.seller_running_sales)

        (final_env_state, _, _), (all_rewards, all_sales) = jax.lax.scan(
            scan_fn,
            (env_state, obs_all, step_key),
            None,
            length=max_steps,
        )

        total_reward = all_rewards.sum(axis=0)
        total_sales = all_sales.sum(axis=0)
        positions = (
            final_env_state.seller_positions.astype(jnp.float32)
            / wrapper.space_resolution
        )
        prices = final_env_state.seller_prices
        qualities = final_env_state.seller_qualities
        return total_reward, positions, prices, total_sales, qualities

    return jax.vmap(run_one)(keys)


def evaluate_ego_policy(
    network: EgoActorCritic | EgoDiscreteActorCritic,
    params: dict,
    wrapper: TrainingWrapper,
    num_episodes: int = 10,
    deterministic: bool = True,
    key: jnp.ndarray | None = None,
    temperature: float | None = None,
    is_discrete: bool = False,
    is_factored: bool = False,
    is_2d_factored: bool = False,
) -> dict[str, float]:
    """Evaluate a trained egocentric policy with rich per-agent metrics."""
    if key is None:
        key = jax.random.PRNGKey(0)

    use_temp = wrapper.env.buyer_choice_temperature is not None
    temp_arr = jnp.float32(temperature if temperature is not None else 0.0)
    keys = jax.random.split(key, num_episodes)

    total_rewards, all_positions, all_prices, total_sales, all_qualities = _eval_ego_episodes_jit(
        network, wrapper, params, keys, deterministic, use_temp,
        is_discrete, is_factored, is_2d_factored, temp_arr,
    )

    return _build_eval_results(
        total_rewards, all_positions, all_prices, total_sales, wrapper, all_qualities
    )


# ---------------------------------------------------------------------------
# Shared results builder
# ---------------------------------------------------------------------------


def _build_eval_results(
    total_rewards: jnp.ndarray,
    all_positions: jnp.ndarray,
    all_prices: jnp.ndarray,
    total_sales: jnp.ndarray,
    wrapper: TrainingWrapper,
    all_qualities: jnp.ndarray | None = None,
) -> dict[str, float]:
    """Build a comprehensive results dict from raw eval arrays.

    Args:
        total_rewards: ``(E, A)`` per-agent total rewards.
        all_positions: ``(E, A, D)`` final positions.
        all_prices: ``(E, A)`` final prices.
        total_sales: ``(E, A)`` total sales per agent.
        wrapper: Training wrapper (for num_agents, dimensions).
        all_qualities: ``(E, A)`` final qualities (when include_quality).

    Returns:
        Dictionary of evaluation metrics.
    """
    A = wrapper.num_agents
    D = wrapper.dimensions

    # ── Aggregate rewards ─────────────────────────────────────────────
    episode_rewards = total_rewards.sum(axis=-1)  # (E,)

    results: dict[str, float] = {
        "eval_reward_mean": float(episode_rewards.mean()),
        "eval_reward_std": float(episode_rewards.std()),
        "eval_length_mean": float(wrapper.env.max_env_steps),
    }

    # ── Per-agent rewards ─────────────────────────────────────────────
    for a in range(A):
        results[f"eval_reward_agent_{a}"] = float(total_rewards[:, a].mean())

    # ── Per-agent positions ───────────────────────────────────────────
    results["eval_position_mean"] = float(all_positions[..., 0].mean())
    for a in range(A):
        for d in range(D):
            dim_label = ["x", "y", "z"][d] if d < 3 else str(d)
            results[f"eval_position_agent_{a}_{dim_label}"] = float(
                all_positions[:, a, d].mean()
            )

    # ── Per-agent prices ──────────────────────────────────────────────
    results["eval_price_mean"] = float(all_prices.mean())
    results["eval_price_spread"] = float(all_prices.mean(axis=0).std())
    for a in range(A):
        results[f"eval_price_agent_{a}"] = float(all_prices[:, a].mean())

    # ── Per-agent qualities (when include_quality) ─────────────────────
    if wrapper.include_quality and all_qualities is not None:
        results["eval_quality_mean"] = float(all_qualities.mean())
        results["eval_quality_spread"] = float(all_qualities.mean(axis=0).std())
        for a in range(A):
            results[f"eval_quality_agent_{a}"] = float(all_qualities[:, a].mean())

    # ── Sales / market share ──────────────────────────────────────────
    total_sales_all = total_sales.sum(axis=-1)  # (E,)
    results["eval_total_sales"] = float(total_sales_all.mean())
    for a in range(A):
        agent_sales = total_sales[:, a]  # (E,)
        results[f"eval_sales_agent_{a}"] = float(agent_sales.mean())
        share = agent_sales / jnp.maximum(total_sales_all, 1e-8)
        results[f"eval_market_share_agent_{a}"] = float(share.mean())

    # ── Seller distance (2-agent only) ────────────────────────────────
    if A == 2:
        dist = jnp.linalg.norm(
            all_positions[:, 0] - all_positions[:, 1], axis=-1
        )
        results["eval_seller_distance"] = float(dist.mean())

    return results
