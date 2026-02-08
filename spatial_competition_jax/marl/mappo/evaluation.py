"""Policy evaluation utilities for MAPPO."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from spatial_competition_jax.marl.mappo.networks import (
    SharedActorCritic,
    deterministic_actions,
    sample_actions,
)
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper


def evaluate_policy(
    network: SharedActorCritic,
    params: dict,
    wrapper: TrainingWrapper,
    num_episodes: int = 10,
    deterministic: bool = True,
    key: jnp.ndarray | None = None,
    temperature: float | None = None,
) -> dict[str, float]:
    """Evaluate a trained policy.

    Runs single-environment episodes (not vmapped) for
    deterministic evaluation.

    Args:
        network: ``SharedActorCritic`` module.
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

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    final_positions: list[float] = []
    final_prices: list[float] = []

    for _ in range(num_episodes):
        key, reset_key, step_key = jax.random.split(key, 3)

        global_state, env_state = wrapper.reset(reset_key)

        episode_reward = jnp.zeros(wrapper.num_agents)
        episode_length = 0
        done = False

        while not done:
            step_key, action_key = jax.random.split(step_key)

            state_batch = global_state[None, ...]  # add batch dim

            if deterministic:
                actions, _ = deterministic_actions(network, params, state_batch)
            else:
                actions, _, _ = sample_actions(network, params, state_batch, action_key)

            actions = actions[0]  # remove batch dim → (A, action_dim)

            global_state, env_state, rewards, dones = wrapper.step(
                step_key,
                env_state,
                actions,
                temperature=temperature,
            )

            episode_reward = episode_reward + rewards
            episode_length += 1
            done = bool(dones[0])

        episode_rewards.append(float(episode_reward.sum()))
        episode_lengths.append(episode_length)

        # Extract final seller info
        positions = env_state.seller_positions.astype(jnp.float32) / wrapper.space_resolution
        prices = env_state.seller_prices

        for i in range(wrapper.num_agents):
            final_positions.append(float(positions[i, 0]))
            final_prices.append(float(prices[i]))

    results: dict[str, float] = {
        "eval_reward_mean": float(np.mean(episode_rewards)),
        "eval_reward_std": float(np.std(episode_rewards)),
        "eval_length_mean": float(np.mean(episode_lengths)),
        "eval_position_mean": float(np.mean(final_positions)),
        "eval_price_mean": float(np.mean(final_prices)),
    }

    if wrapper.num_agents == 2:
        distances = [abs(final_positions[i] - final_positions[i + 1]) for i in range(0, len(final_positions), 2)]
        results["eval_seller_distance"] = float(np.mean(distances))

    return results
