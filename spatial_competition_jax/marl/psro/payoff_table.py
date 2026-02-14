"""Payoff-matrix construction for symmetric PSRO.

Cross-evaluates every pair of policies in the population to build the
empirical normal-form payoff matrix.  Uses ``jax.lax.scan`` for the
inner step loop and ``jax.vmap`` across episodes so the entire
evaluation of a single matchup is a single fused JIT kernel.

Supports both egocentric and global observation modes.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from spatial_competition_jax.marl.psro.state_utils import permute_agent_state
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper

if TYPE_CHECKING:
    from spatial_competition_jax.marl.mappo.policy import PolicyAdapter


# ---------------------------------------------------------------------------
# JIT kernels: evaluate one matchup over multiple episodes
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(0, 1, 4))
def _evaluate_matchup_ego_jit(
    policy: PolicyAdapter,
    wrapper: TrainingWrapper,
    params_row: Any,
    params_col: Any,
    use_temp: bool,
    temperature: jnp.ndarray,
    keys: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a matchup using **egocentric** observations.

    Policy *row* plays as agent 0 — gets ``ego_obs[0]``.
    Policy *col* plays as agent 1 — gets ``ego_obs[1]`` (already
    structured from agent 1's perspective, so it "thinks" it is
    agent 0).

    Both policies use deterministic (mode) actions.

    Returns:
        Per-agent total rewards ``(num_episodes, 2)``.
    """
    max_steps = wrapper.env.max_env_steps

    def run_one(key: jnp.ndarray) -> jnp.ndarray:
        reset_key, step_key = jax.random.split(key)
        ego_obs, env_state = wrapper.reset_ego(reset_key)  # (A, obs_dim)

        def scan_fn(
            carry: tuple[Any, jnp.ndarray, jnp.ndarray],
            _: None,
        ) -> tuple[tuple[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
            env_state, ego_obs, sk = carry
            sk, _ = jax.random.split(sk)

            # Agent 0 (row) — deterministic from ego_obs[0]
            row_obs = ego_obs[0]  # (obs_dim,)
            row_actions, _ = policy.deterministic(
                params_row, row_obs[None, None, ...],  # (1, 1, obs_dim)
            )
            a0 = row_actions[0, 0]  # (action_dim,)

            # Agent 1 (col) — deterministic from ego_obs[1]
            col_obs = ego_obs[1]  # (obs_dim,)
            col_actions, _ = policy.deterministic(
                params_col, col_obs[None, None, ...],
            )
            a1 = col_actions[0, 0]  # (action_dim,)

            combined = jnp.stack([a0, a1], axis=0)  # (2, action_dim)

            if use_temp:
                next_obs, next_es, rewards, _dones = wrapper.step_ego(
                    sk, env_state, combined, temperature=temperature,
                )
            else:
                next_obs, next_es, rewards, _dones = wrapper.step_ego(
                    sk, env_state, combined,
                )

            return (next_es, next_obs, sk), rewards

        (_, _, _), all_rewards = jax.lax.scan(
            scan_fn, (env_state, ego_obs, step_key), None, length=max_steps,
        )
        return all_rewards.sum(axis=0)  # (2,)

    return jax.vmap(run_one)(keys)  # (num_episodes, 2)


@functools.partial(jax.jit, static_argnums=(0, 1, 4))
def _evaluate_matchup_global_jit(
    policy: PolicyAdapter,
    wrapper: TrainingWrapper,
    params_row: Any,
    params_col: Any,
    use_temp: bool,
    temperature: jnp.ndarray,
    keys: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a matchup using **global** observations.

    Policy *row* plays as agent 0 — original state.
    Policy *col* plays as agent 1 — permuted state (sees itself
    as agent 0).

    Both policies use deterministic (mode) actions.

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
            env_state, gs, sk = carry
            sk, _ = jax.random.split(sk)

            state_batch = gs[None, ...]  # (1, state_dim)

            # Agent 0 (row) — deterministic, take agent 0 action
            row_actions, _ = policy.deterministic(params_row, state_batch)
            a0 = row_actions[0, 0]  # (action_dim,)

            # Agent 1 (col) — deterministic on permuted state, take agent 0 action
            permuted = permute_agent_state(state_batch, wrapper)
            col_actions, _ = policy.deterministic(params_col, permuted)
            a1 = col_actions[0, 0]  # (action_dim,)

            combined = jnp.stack([a0, a1], axis=0)  # (2, action_dim)

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
            scan_fn, (env_state, global_state, step_key), None, length=max_steps,
        )
        return all_rewards.sum(axis=0)

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
        policy: PolicyAdapter,
        wrapper: TrainingWrapper,
        *,
        egocentric: bool = True,
        num_eval_episodes: int = 50,
        temperature: float | None = None,
        seed: int = 0,
    ) -> None:
        self.policy = policy
        self.wrapper = wrapper
        self.egocentric = egocentric
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

        # Determine which pairs need evaluation
        new_pairs: set[tuple[int, int]] = set()
        for i in range(old_K, K):
            for j in range(K):
                new_pairs.add((i, j))
                new_pairs.add((j, i))
        for i in range(old_K):
            for j in range(old_K):
                new_pairs.discard((i, j))

        # Select the appropriate JIT kernel
        eval_fn = (
            _evaluate_matchup_ego_jit if self.egocentric
            else _evaluate_matchup_global_jit
        )

        # Evaluate only unordered pairs (i, j) with i <= j
        evaluated: set[tuple[int, int]] = set()
        for i, j in sorted(new_pairs):
            if (i, j) in evaluated:
                continue

            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, self.num_eval_episodes)

            # Policy i as agent 0, policy j as agent 1
            rewards = eval_fn(
                self.policy,
                self.wrapper,
                population[i],
                population[j],
                self.use_temp,
                self.temperature,
                keys,
            )
            # rewards: (num_episodes, 2)
            val_ij = float(rewards[:, 0].mean())
            new_matrix[i, j] = val_ij if np.isfinite(val_ij) else 0.0

            if i != j:
                val_ji = float(rewards[:, 1].mean())
                new_matrix[j, i] = val_ji if np.isfinite(val_ji) else 0.0
                evaluated.add((j, i))

            evaluated.add((i, j))

        self._matrix = new_matrix
        self._size = K
        return new_matrix
