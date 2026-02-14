"""Policy evaluation utilities for MAPPO."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run all evaluation episodes in parallel.

    Uses ``jax.lax.scan`` for the inner step loop and ``jax.vmap``
    across episodes so the entire evaluation is a single fused kernel.

    Args:
        policy: ``PolicyAdapter`` (static).
        wrapper: ``TrainingWrapper`` (static).
        params: Network parameters.
        keys: ``(num_episodes, 2)`` PRNG keys.
        deterministic: Whether to use deterministic actions (static).
        use_temp: Whether temperature is used by the env (static).
        temperature: Buyer-choice temperature (scalar array).

    Returns:
        ``(total_rewards, final_positions, final_prices)`` with shapes
        ``(E, A)``, ``(E, S, D)``, ``(E, S)`` where *E* = num_episodes.
    """
    max_steps = wrapper.env.max_env_steps

    def run_one(key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        reset_key, step_key = jax.random.split(key)
        global_state, env_state = wrapper.reset(reset_key)

        def scan_fn(
            carry: tuple[Any, jnp.ndarray, jnp.ndarray],
            _: None,
        ) -> tuple[tuple[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
            env_state, global_state, sk = carry
            sk, action_key = jax.random.split(sk)

            state_batch = global_state[None, ...]  # add batch dim

            if deterministic:
                actions, _ = policy.deterministic(params, state_batch)
            else:
                actions, _, _ = policy.sample(params, state_batch, action_key)
            actions = actions[0]  # remove batch dim → (A, action_dim)

            if use_temp:
                next_gs, next_es, rewards, _dones = wrapper.step(
                    sk, env_state, actions, temperature=temperature,
                )
            else:
                next_gs, next_es, rewards, _dones = wrapper.step(
                    sk, env_state, actions,
                )

            return (next_es, next_gs, sk), rewards

        (final_env_state, _, _), all_rewards = jax.lax.scan(
            scan_fn,
            (env_state, global_state, step_key),
            None,
            length=max_steps,
        )

        # all_rewards: (max_steps, num_agents) → (num_agents,)
        total_reward = all_rewards.sum(axis=0)

        positions = (
            final_env_state.seller_positions.astype(jnp.float32)
            / wrapper.space_resolution
        )
        prices = final_env_state.seller_prices

        return total_reward, positions, prices

    # Vectorise across episodes
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
    """Evaluate a trained policy.

    Runs *num_episodes* episodes in parallel (vmapped) with
    ``jax.lax.scan`` for the inner step loop — fully JIT-compiled.

    Args:
        policy: ``PolicyAdapter`` instance.
        params: Network parameters.
        wrapper: :class:`TrainingWrapper` for the environment.
        num_episodes: Number of evaluation episodes.
        deterministic: If *True*, use mean actions.
        key: PRNG key (required for stochastic evaluation).
        temperature: Optional buyer-choice temperature override.
            Defaults to ``None`` which uses the env's static setting.

    Returns:
        Dictionary of evaluation statistics.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    use_temp = wrapper.env.buyer_choice_temperature is not None
    temp_arr = jnp.float32(temperature if temperature is not None else 0.0)
    keys = jax.random.split(key, num_episodes)

    total_rewards, all_positions, all_prices = _eval_episodes_jit(
        policy, wrapper, params, keys, deterministic, use_temp, temp_arr,
    )

    # total_rewards: (num_episodes, num_agents)
    episode_rewards = total_rewards.sum(axis=-1)  # (num_episodes,)

    results: dict[str, float] = {
        "eval_reward_mean": float(episode_rewards.mean()),
        "eval_reward_std": float(episode_rewards.std()),
        "eval_length_mean": float(wrapper.env.max_env_steps),
        "eval_position_mean": float(all_positions[..., 0].mean()),
        "eval_price_mean": float(all_prices.mean()),
    }

    if wrapper.num_agents == 2:
        # all_positions: (E, 2, D) → distance between agent 0 and 1
        distances = jnp.abs(all_positions[:, 0, 0] - all_positions[:, 1, 0])
        results["eval_seller_distance"] = float(distances.mean())

    return results
