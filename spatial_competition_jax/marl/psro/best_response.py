"""Best-response trainer for symmetric PSRO.

Trains a single policy against opponents sampled from the PSRO
population according to the meta-strategy mixture.  Uses the same
:class:`PolicyAdapter` abstraction as :class:`MAPPO`, but with:

* Two separate param sets: trainee (updated) vs. frozen opponent.
* Only agent 0's rewards/values used for GAE / PPO.
* Opponent params traced as a JIT argument (no recompilation when the
  opponent changes between rollouts).

All PPO improvements (log-ratio clamping, clipped value loss,
KL-based early stopping) are applied.

Supports both egocentric and global observation modes.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from spatial_competition_jax.marl.mappo.buffer import (
    RolloutBatch,
    Transition,
    compute_gae,
    normalize_advantages,
)
from spatial_competition_jax.marl.mappo.networks import symlog
from spatial_competition_jax.marl.psro.state_utils import permute_agent_state
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper

if TYPE_CHECKING:
    from spatial_competition_jax.marl.mappo.policy import PolicyAdapter


class BestResponseTrainer:
    """Train a best-response policy against a population mixture.

    The trainee always plays as agent 0.  At the start of each
    rollout an opponent is sampled from the population according to
    ``meta_strategy`` and plays as agent 1.

    In **egocentric** mode, the opponent receives ``ego_obs[1]``
    (which already has its own features first), so no state permutation
    is needed.

    In **global** mode, the opponent sees the permuted global state
    (features swapped so it "thinks" it is agent 0).

    The opponent uses **deterministic** (mode) actions; stochasticity
    comes from *which* opponent is drawn at each rollout.
    """

    def __init__(
        self,
        wrapper: TrainingWrapper,
        policy: PolicyAdapter,
        population: list[Any],
        meta_strategy: np.ndarray,
        *,
        num_envs: int = 16,
        rollout_length: int = 512,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.005,
        learning_rate: float = 3e-4,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 6,
        num_minibatches: int = 8,
        target_kl: float | None = None,
        seed: int = 42,
        warmstart_params: Any | None = None,
    ) -> None:
        self.wrapper = wrapper
        self.policy = policy
        self.population = list(population)
        self.meta_strategy = np.array(meta_strategy, dtype=np.float64)
        self.num_envs = num_envs
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.target_kl = target_kl if target_kl is not None else float("inf")

        key = jax.random.PRNGKey(seed)
        key, init_key, reset_key = jax.random.split(key, 3)

        # ── initialise or warm-start parameters ──────────────────────
        if warmstart_params is not None:
            params = warmstart_params
        else:
            if policy.per_agent_obs:
                dummy = jnp.zeros(wrapper.obs_dim)
            else:
                dummy = jnp.zeros(wrapper.state_dim)
            params = policy.init(init_key, dummy)

        # ── optimizer ─────────────────────────────────────────────────
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate),
        )
        self.train_state: TrainState = TrainState.create(
            apply_fn=None,
            params=params,
            tx=tx,
        )

        # ── initial environment states ────────────────────────────────
        reset_keys = jax.random.split(reset_key, num_envs)
        if policy.per_agent_obs:
            self._obs, self.env_states = jax.vmap(wrapper.reset_ego)(reset_keys)
        else:
            self._obs, self.env_states = jax.vmap(wrapper.reset)(reset_keys)

        self.key = key

        # NumPy RNG for opponent sampling (outside JIT)
        self._np_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_rollout(
        self,
        temperature: float | None = None,
    ) -> tuple[Transition, jnp.ndarray, jnp.ndarray, dict[str, float]]:
        """Collect one rollout against a sampled opponent.

        Returns:
            ``(transitions, advantages, returns, stats)``
        """
        self.key, subkey = jax.random.split(self.key)

        # Sample opponent from population using meta-strategy
        opponent_idx = self._np_rng.choice(
            len(self.population), p=self.meta_strategy,
        )
        opponent_params = self.population[opponent_idx]

        temp_arr = jnp.float32(temperature if temperature is not None else 0.0)

        transitions, advantages, returns, env_states, obs = (
            self._collect_rollout(
                self.train_state,
                self.env_states,
                self._obs,
                subkey,
                temp_arr,
                opponent_params,
            )
        )
        self.env_states = env_states
        self._obs = obs

        stats = {
            "mean_reward": float(transitions.rewards.mean()),
            "std_reward": float(transitions.rewards.std()),
            "mean_value": float(transitions.values.mean()),
            "mean_return": float(returns.mean()),
            "total_reward": float(transitions.rewards.sum()),
            "opponent_idx": float(opponent_idx),
        }

        return transitions, advantages, returns, stats

    def update(
        self,
        transitions: Transition,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        entropy_coef: float | None = None,
    ) -> dict[str, float]:
        """Run PPO update epochs on single-agent data.

        Returns:
            Dictionary of averaged training metrics.
        """
        self.key, subkey = jax.random.split(self.key)
        ent_coef = jnp.float32(
            entropy_coef if entropy_coef is not None else self.entropy_coef
        )
        self.train_state, metrics = self._update(
            self.train_state, transitions, advantages, returns, subkey, ent_coef,
        )
        return {k: float(v) for k, v in metrics.items()}

    def update_opponents(
        self,
        population: list[Any],
        meta_strategy: np.ndarray,
    ) -> None:
        """Replace the opponent population and meta-strategy."""
        self.population = list(population)
        self.meta_strategy = np.array(meta_strategy, dtype=np.float64)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def params(self) -> Any:
        """Current trainee network parameters."""
        return self.train_state.params

    @property
    def opt_state(self) -> Any:
        """Current optimiser state."""
        return self.train_state.opt_state

    # ------------------------------------------------------------------
    # JIT-compiled rollout collection
    # ------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0,))
    def _collect_rollout(
        self,
        train_state: TrainState,
        env_states: Any,
        obs: jnp.ndarray,
        key: jnp.ndarray,
        temperature: jnp.ndarray,
        opponent_params: Any,
    ) -> Any:
        """Collect a rollout with the trainee as agent 0 and a frozen
        opponent as agent 1."""
        _use_temp = self.wrapper.env.buyer_choice_temperature is not None
        _ego = self.policy.per_agent_obs

        def scan_fn(carry: Any, _: Any) -> Any:
            env_states, obs, key = carry
            key, step_key, sample_key = jax.random.split(key, 3)

            if _ego:
                # Egocentric: obs shape (E, A, obs_dim)
                # Trainee (agent 0): obs[:, 0, :] → (E, obs_dim)
                # Opponent (agent 1): obs[:, 1, :] → (E, obs_dim)
                trainee_obs = obs[:, 0:1, :]  # (E, 1, obs_dim)
                opponent_obs = obs[:, 1:2, :]  # (E, 1, obs_dim)
            else:
                # Global: obs shape (E, state_dim)
                trainee_obs = obs  # (E, state_dim)
                opponent_obs = permute_agent_state(obs, self.wrapper)

            # ── Trainee forward (stochastic) ──────────────────────
            actions, log_probs, values = self.policy.sample(
                train_state.params, trainee_obs, sample_key,
            )
            # actions: (E, A_out, action_dim), log_probs: (E, A_out), values: (E, A_out)

            if _ego:
                # A_out = 1, extract the single agent's action
                trainee_action = actions[:, 0, :]  # (E, action_dim)
                log_probs = log_probs[:, 0]  # (E,)
                values = values[:, 0]  # (E,)
            else:
                # Global mode with num_agents=1: A_out = 1
                trainee_action = actions[:, 0, :]  # (E, action_dim)
                log_probs = log_probs[:, 0]  # (E,)
                values = values[:, 0]  # (E,)

            # ── Opponent forward (deterministic) ──────────────────
            opp_actions, _ = self.policy.deterministic(
                opponent_params, opponent_obs,
            )

            if _ego:
                opp_action = opp_actions[:, 0, :]  # (E, action_dim)
            else:
                opp_action = opp_actions[:, 0, :]  # (E, action_dim)

            # ── Combine and step ──────────────────────────────────
            combined = jnp.stack([trainee_action, opp_action], axis=1)  # (E, 2, action_dim)

            step_keys = jax.random.split(step_key, self.num_envs)
            if _ego:
                if _use_temp:
                    next_obs, next_es, rewards, dones = jax.vmap(
                        lambda k, s, a: self.wrapper.step_autoreset_ego(
                            k, s, a, temperature=temperature,
                        ),
                    )(step_keys, env_states, combined)
                else:
                    next_obs, next_es, rewards, dones = jax.vmap(
                        self.wrapper.step_autoreset_ego,
                    )(step_keys, env_states, combined)
            else:
                if _use_temp:
                    next_obs, next_es, rewards, dones = jax.vmap(
                        lambda k, s, a: self.wrapper.step_autoreset(
                            k, s, a, temperature=temperature,
                        ),
                    )(step_keys, env_states, combined)
                else:
                    next_obs, next_es, rewards, dones = jax.vmap(
                        self.wrapper.step_autoreset,
                    )(step_keys, env_states, combined)

            # Keep only agent 0's reward & done signal
            agent0_rewards = rewards[:, 0]  # (E,)
            agent0_dones = dones[:, 0]  # (E,)

            # Store trainee's observation (not the full multi-agent obs)
            if _ego:
                stored_obs = obs[:, 0, :]  # (E, obs_dim)
            else:
                stored_obs = obs  # (E, state_dim)

            transition = Transition(
                states=stored_obs,
                actions=trainee_action[:, None, :],  # (E, 1, action_dim)
                log_probs=log_probs[:, None],  # (E, 1)
                values=values[:, None],  # (E, 1)
                rewards=agent0_rewards[:, None],  # (E, 1)
                dones=agent0_dones[:, None],  # (E, 1)
            )

            return (next_es, next_obs, key), transition

        (final_es, final_obs, _), transitions = jax.lax.scan(
            scan_fn, (env_states, obs, key), None, length=self.rollout_length,
        )

        # Bootstrap value for GAE
        if _ego:
            bootstrap_obs = final_obs[:, 0:1, :]  # (E, 1, obs_dim)
        else:
            bootstrap_obs = final_obs  # (E, state_dim)
        bootstrap_values = self.policy.value(train_state.params, bootstrap_obs)
        if _ego:
            bootstrap_values = bootstrap_values[:, 0:1]  # (E, 1)
        else:
            bootstrap_values = bootstrap_values[:, 0:1]  # (E, 1)

        # GAE (single-agent: A=1)
        advantages, returns = compute_gae(
            transitions.rewards,
            transitions.values,
            transitions.dones,
            bootstrap_values,
            self.gamma,
            self.gae_lambda,
        )

        advantages = normalize_advantages(advantages, per_agent=True)
        returns = symlog(returns)

        return transitions, advantages, returns, final_es, final_obs

    # ------------------------------------------------------------------
    # JIT-compiled PPO update
    # ------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update(
        self,
        train_state: TrainState,
        transitions: Transition,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        key: jnp.ndarray,
        entropy_coef: jnp.ndarray,
    ) -> tuple[TrainState, dict[str, jnp.ndarray]]:
        """PPO update on single-agent data (A=1)."""
        _target_kl = self.target_kl

        def loss_fn(
            params: Any, batch: RolloutBatch,
        ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
            new_lp, entropy, values = self.policy.evaluate(
                params, batch.states, batch.actions,
            )

            # PPO clipped objective with log-ratio clamping
            log_ratio = new_lp - batch.log_probs
            log_ratio = jnp.clip(log_ratio, -2.0, 2.0)
            ratio = jnp.exp(log_ratio)
            surr1 = ratio * batch.advantages
            surr2 = (
                jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * batch.advantages
            )
            policy_loss = -jnp.minimum(surr1, surr2).mean()

            approx_kl = ((ratio - 1) - log_ratio).mean()

            # Clipped value loss in symlog space
            old_v_sym = symlog(batch.values)
            v_clipped = old_v_sym + jnp.clip(
                values - old_v_sym, -self.clip_epsilon, self.clip_epsilon,
            )
            vl_u = (values - batch.returns) ** 2
            vl_c = (v_clipped - batch.returns) ** 2
            value_loss = jnp.maximum(vl_u, vl_c).mean() * 0.5

            entropy_loss = -entropy.mean()
            total_loss = (
                policy_loss
                + self.value_coef * value_loss
                + entropy_coef * entropy_loss
            )

            metrics = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy.mean(),
                "entropy_coef": entropy_coef,
                "approx_kl": approx_kl,
                "clip_fraction": (
                    jnp.abs(ratio - 1) > self.clip_epsilon
                ).astype(jnp.float32).mean(),
            }
            return total_loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        # KL-based early stopping (zeros grads when KL exceeds threshold)
        def minibatch_step(
            carry: tuple[TrainState, jnp.ndarray],
            batch: RolloutBatch,
        ) -> tuple[tuple[TrainState, jnp.ndarray], dict[str, jnp.ndarray]]:
            ts, kl_ex = carry
            (_, met), grads = grad_fn(ts.params, batch)
            grads = jax.tree.map(
                lambda g: jnp.where(kl_ex, jnp.zeros_like(g), g), grads,
            )
            ts = ts.apply_gradients(grads=grads)
            kl_ex = kl_ex | (met["approx_kl"] > _target_kl)
            return (ts, kl_ex), met

        def epoch_step(
            carry: tuple[TrainState, jnp.ndarray],
            epoch_key: jnp.ndarray,
        ) -> tuple[tuple[TrainState, jnp.ndarray], dict[str, jnp.ndarray]]:
            ts, kl_ex = carry
            batches = _make_br_minibatches(
                epoch_key, transitions, advantages, returns,
                self.num_minibatches,
            )
            (ts, kl_ex), met = jax.lax.scan(
                minibatch_step, (ts, kl_ex), batches,
            )
            return (ts, kl_ex), met

        epoch_keys = jax.random.split(key, self.ppo_epochs)
        (train_state, _), all_metrics = jax.lax.scan(
            epoch_step, (train_state, jnp.bool_(False)), epoch_keys,
        )

        return train_state, jax.tree.map(lambda x: x.mean(), all_metrics)


# ---------------------------------------------------------------------------
# Minibatch creation for best-response (A=1, flatten T×E)
# ---------------------------------------------------------------------------


def _make_br_minibatches(
    key: jnp.ndarray,
    transitions: Transition,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    num_minibatches: int,
) -> RolloutBatch:
    """Flatten ``(T, E, 1)`` into independent samples."""
    T, E = transitions.rewards.shape[:2]
    N = T * E
    B = N // num_minibatches

    flat_s = transitions.states.reshape(N, -1)
    flat_a = transitions.actions.reshape(N, -1)
    flat_lp = transitions.log_probs.reshape(N)
    flat_v = transitions.values.reshape(N)
    flat_adv = advantages.reshape(N)
    flat_ret = returns.reshape(N)

    perm = jax.random.permutation(key, N)
    perm = perm[: B * num_minibatches].reshape(num_minibatches, B)

    return RolloutBatch(
        states=flat_s[perm],
        actions=flat_a[perm],
        log_probs=flat_lp[perm],
        values=flat_v[perm],
        advantages=flat_adv[perm],
        returns=flat_ret[perm],
    )
