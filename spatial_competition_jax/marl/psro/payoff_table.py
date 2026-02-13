"""Payoff-matrix construction for symmetric PSRO.

Cross-evaluates every pair of policies in the population to build the
empirical normal-form payoff matrix.  Uses ``jax.lax.scan`` for the
inner step loop and ``jax.vmap`` across episodes so the entire
evaluation of a single matchup is a single fused JIT kernel.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from spatial_competition_jax.marl.mappo.networks import (
    SharedActorCritic,
    deterministic_actions,
)
from spatial_competition_jax.marl.psro.state_utils import permute_agent_state
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper


# ---------------------------------------------------------------------------
# JIT kernel: evaluate one matchup over multiple episodes
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(0, 1, 4))
def _evaluate_matchup_jit(
    network: SharedActorCritic,
    wrapper: TrainingWrapper,
    params_row: Any,
    params_col: Any,
    use_temp: bool,
    temperature: jnp.ndarray,
    keys: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a single policy matchup over parallel episodes.

    Policy *row* plays as agent 0 with the original state.
    Policy *col* plays as agent 1 with the permuted state (sees itself
    as agent 0).

    Both policies use deterministic (mode) actions for evaluation
    consistency.

    Args:
        network: ``SharedActorCritic`` with ``num_agents=1`` (static).
        wrapper: ``TrainingWrapper`` (static).
        params_row: Params for the row (agent 0) policy.
        params_col: Params for the column (agent 1) policy.
        use_temp: Whether buyer-choice temperature is active (static).
        temperature: Scalar buyer-choice temperature.
        keys: ``(num_episodes,)`` PRNG keys.

    Returns:
        Per-agent total rewards ``(num_episodes, 2)``.
    """
    max_steps = wrapper.env.max_env_steps

    def run_one(key: jnp.ndarray) -> jnp.ndarray:
        reset_key, step_key = jax.random.split(key)
        global_state, env_state = wrapper.reset(reset_key)

        def scan_fn(
            carry: tuple[Any, jnp.ndarray, jnp.ndarray],
            _: None,
        ) -> tuple[tuple[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
            env_state, global_state, sk = carry
            sk, action_key = jax.random.split(sk)

            state_batch = global_state[None, ...]  # (1, state_dim)

            # Agent 0 (row policy) – original state
            row_actions, _ = deterministic_actions(
                network, params_row, state_batch,
            )
            agent0_action = row_actions[0, 0]  # (action_dim,)

            # Agent 1 (col policy) – permuted state
            permuted = permute_agent_state(state_batch, wrapper)
            col_actions, _ = deterministic_actions(
                network, params_col, permuted,
            )
            agent1_action = col_actions[0, 0]  # (action_dim,)

            # Combine actions: (2, action_dim)
            combined = jnp.stack([agent0_action, agent1_action], axis=0)

            if use_temp:
                next_gs, next_es, rewards, _dones = wrapper.step(
                    sk, env_state, combined, temperature=temperature,
                )
            else:
                next_gs, next_es, rewards, _dones = wrapper.step(
                    sk, env_state, combined,
                )

            return (next_es, next_gs, sk), rewards

        (_, _, _), all_rewards = jax.lax.scan(
            scan_fn,
            (env_state, global_state, step_key),
            None,
            length=max_steps,
        )

        # all_rewards: (max_steps, 2) → (2,)
        return all_rewards.sum(axis=0)

    # Vectorise across episodes
    return jax.vmap(run_one)(keys)


# ---------------------------------------------------------------------------
# PayoffTable
# ---------------------------------------------------------------------------


class PayoffTable:
    """Incrementally-built payoff matrix for symmetric PSRO.

    Stores a ``K x K`` matrix where ``U[i, j]`` is the mean total
    reward of agent 0 when policy *i* plays agent 0 and policy *j*
    plays agent 1.

    For a symmetric game: ``U_agent1[i, j] = U_agent0[j, i]``.
    """

    def __init__(
        self,
        network: SharedActorCritic,
        wrapper: TrainingWrapper,
        num_eval_episodes: int = 50,
        temperature: float | None = None,
        seed: int = 0,
    ) -> None:
        self.network = network
        self.wrapper = wrapper
        self.num_eval_episodes = num_eval_episodes
        self.use_temp = wrapper.env.buyer_choice_temperature is not None
        self.temperature = jnp.float32(
            temperature if temperature is not None else 0.0
        )
        self.seed = seed

        # Payoff matrix (row player = agent 0)
        self._matrix: np.ndarray | None = None
        self._size: int = 0  # current population size

    @property
    def matrix(self) -> np.ndarray:
        """Current payoff matrix ``(K, K)``."""
        assert self._matrix is not None, "Payoff table is empty."
        return self._matrix

    def update(self, population: list[Any]) -> np.ndarray:
        """Incrementally evaluate new entries after population growth.

        Only evaluates matchups involving the newly-added policies
        (the last ``len(population) - self._size`` entries) against
        *all* existing policies, avoiding re-evaluation of already
        computed entries.

        Args:
            population: List of param pytrees (length K).

        Returns:
            Updated ``(K, K)`` payoff matrix.
        """
        K = len(population)
        old_K = self._size

        # Grow matrix
        new_matrix = np.zeros((K, K), dtype=np.float64)
        if self._matrix is not None and old_K > 0:
            new_matrix[:old_K, :old_K] = self._matrix[:old_K, :old_K]

        key = jax.random.PRNGKey(self.seed)

        # Evaluate new rows (new policy as agent 0 vs all as agent 1)
        for i in range(old_K, K):
            for j in range(K):
                key, subkey = jax.random.split(key)
                keys = jax.random.split(subkey, self.num_eval_episodes)

                rewards = _evaluate_matchup_jit(
                    self.network,
                    self.wrapper,
                    population[i],
                    population[j],
                    self.use_temp,
                    self.temperature,
                    keys,
                )
                # rewards: (num_episodes, 2)
                new_matrix[i, j] = float(rewards[:, 0].mean())

        # Evaluate new columns (existing policies as agent 0 vs new as agent 1)
        for i in range(old_K):
            for j in range(old_K, K):
                key, subkey = jax.random.split(key)
                keys = jax.random.split(subkey, self.num_eval_episodes)

                rewards = _evaluate_matchup_jit(
                    self.network,
                    self.wrapper,
                    population[i],
                    population[j],
                    self.use_temp,
                    self.temperature,
                    keys,
                )
                new_matrix[i, j] = float(rewards[:, 0].mean())

        self._matrix = new_matrix
        self._size = K
        return new_matrix
