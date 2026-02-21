"""MAPPO algorithm implementation in JAX.

Fully network-agnostic: all distribution-specific logic lives behind
the :class:`PolicyAdapter` protocol.  MAPPO only calls
``policy.sample``, ``policy.evaluate``, ``policy.value``.

The single flag ``policy.per_agent_obs`` tells MAPPO whether to use
egocentric env stepping (``reset_ego`` / ``step_autoreset_ego``) and
``T×E×A`` minibatching, or global stepping and ``T×E`` minibatching.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from spatial_competition_jax.marl.mappo.buffer import (
    RolloutBatch,
    Transition,
    compute_gae,
    make_minibatches,
    normalize_advantages,
)
from spatial_competition_jax.marl.mappo.networks import symlog
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper

if TYPE_CHECKING:
    from spatial_competition_jax.marl.mappo.policy import PolicyAdapter


def linear_anneal(
    update: int,
    total_updates: int,
    start: float,
    end: float,
    anneal_frac: float = 0.8,
) -> float:
    """Linearly anneal a value from *start* to *end*."""
    anneal_updates = max(int(total_updates * anneal_frac), 1)
    frac = min(update / anneal_updates, 1.0)
    return start + (end - start) * frac


# Backward-compatible alias
compute_temperature = linear_anneal


class MAPPO:
    """MAPPO with a pluggable :class:`PolicyAdapter`.

    The policy adapter abstracts all observation and action logic.
    MAPPO only uses ``policy.per_agent_obs`` to choose:
    - ``reset_ego`` / ``step_autoreset_ego`` vs ``reset`` / ``step_autoreset``
    - ``T×E×A`` vs ``T×E`` minibatching

    All PPO improvements (log-ratio clamping, clipped value loss,
    KL-based early stopping) are applied universally.
    """

    def __init__(
        self,
        wrapper: TrainingWrapper,
        policy: PolicyAdapter,
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
    ) -> None:
        self.wrapper = wrapper
        self.policy = policy
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

        # ── Init params ──────────────────────────────────────────────
        if policy.per_agent_obs:
            dummy = jnp.zeros(wrapper.obs_dim)
        else:
            dummy = jnp.zeros(wrapper.state_dim)
        params = policy.init(init_key, dummy)

        # ── Optimizer ─────────────────────────────────────────────────
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate),
        )
        self.train_state: TrainState = TrainState.create(
            apply_fn=None,
            params=params,
            tx=tx,
        )

        # ── Initial env states ────────────────────────────────────────
        reset_keys = jax.random.split(reset_key, num_envs)
        if policy.per_agent_obs:
            self._obs, self.env_states = jax.vmap(wrapper.reset_ego)(reset_keys)
        else:
            self._obs, self.env_states = jax.vmap(wrapper.reset)(reset_keys)

        self.key = key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_rollout(
        self,
        temperature: float | None = None,
    ) -> tuple[Transition, jnp.ndarray, jnp.ndarray, dict[str, float]]:
        self.key, subkey = jax.random.split(self.key)
        temp_arr = jnp.float32(temperature if temperature is not None else 0.0)

        transitions, advantages, returns, env_states, obs = self._collect_rollout(
            self.train_state, self.env_states, self._obs, subkey, temp_arr,
        )
        self.env_states = env_states
        self._obs = obs

        return transitions, advantages, returns, {
            "mean_reward": float(transitions.rewards.mean()),
            "std_reward": float(transitions.rewards.std()),
            "mean_value": float(transitions.values.mean()),
            "mean_return": float(returns.mean()),
            "total_reward": float(transitions.rewards.sum()),
        }

    def update(
        self,
        transitions: Transition,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        entropy_coef: float | None = None,
    ) -> dict[str, float]:
        self.key, subkey = jax.random.split(self.key)
        ent_coef = jnp.float32(entropy_coef if entropy_coef is not None else self.entropy_coef)
        self.train_state, metrics = self._update(
            self.train_state, transitions, advantages, returns, subkey, ent_coef,
        )
        return {k: float(v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def params(self) -> Any:
        return self.train_state.params

    @property
    def opt_state(self) -> Any:
        return self.train_state.opt_state

    @property
    def network(self) -> Any:
        """Underlying network (for ego evaluation). May be None for global."""
        return getattr(self.policy, "network", None)

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
    ) -> Any:
        _use_temp = self.wrapper.env.buyer_choice_temperature is not None
        _ego = self.policy.per_agent_obs

        def scan_fn(carry: Any, _: Any) -> Any:
            env_states, obs, key = carry
            key, step_key, sample_key = jax.random.split(key, 3)

            actions, log_probs, values = self.policy.sample(
                train_state.params, obs, sample_key,
            )

            step_keys = jax.random.split(step_key, self.num_envs)
            if _ego:
                if _use_temp:
                    next_obs, next_es, rewards, dones = jax.vmap(
                        lambda k, s, a: self.wrapper.step_autoreset_ego(
                            k, s, a, temperature=temperature,
                        ),
                    )(step_keys, env_states, actions)
                else:
                    next_obs, next_es, rewards, dones = jax.vmap(
                        self.wrapper.step_autoreset_ego,
                    )(step_keys, env_states, actions)
            else:
                if _use_temp:
                    next_obs, next_es, rewards, dones = jax.vmap(
                        lambda k, s, a: self.wrapper.step_autoreset(
                            k, s, a, temperature=temperature,
                        ),
                    )(step_keys, env_states, actions)
                else:
                    next_obs, next_es, rewards, dones = jax.vmap(
                        self.wrapper.step_autoreset,
                    )(step_keys, env_states, actions)

            transition = Transition(
                states=obs, actions=actions, log_probs=log_probs,
                values=values, rewards=rewards, dones=dones,
            )
            return (next_es, next_obs, key), transition

        (final_es, final_obs, _), transitions = jax.lax.scan(
            scan_fn, (env_states, obs, key), None, length=self.rollout_length,
        )

        bootstrap = self.policy.value(train_state.params, final_obs)

        advantages, returns = compute_gae(
            transitions.rewards, transitions.values, transitions.dones,
            bootstrap, self.gamma, self.gae_lambda,
        )

        # Always normalise per-agent so each agent gets equally
        # strong gradient signal regardless of absolute reward level.
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
        _target_kl = self.target_kl
        _ego = self.policy.per_agent_obs

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

            # Clipped value loss
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
            if _ego:
                batches = _make_ego_minibatches(
                    epoch_key, transitions, advantages, returns,
                    self.num_minibatches,
                )
            else:
                batches = make_minibatches(
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
# Ego-specific minibatch creation (flattens T × E × A)
# ---------------------------------------------------------------------------


def _make_ego_minibatches(
    key: jnp.ndarray,
    transitions: Transition,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    num_minibatches: int,
) -> RolloutBatch:
    """Flatten ``(T, E, A)`` into independent samples for ego mode."""
    T, E, A = transitions.rewards.shape[:3]
    N = T * E * A
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
        states=flat_s[perm], actions=flat_a[perm], log_probs=flat_lp[perm],
        values=flat_v[perm], advantages=flat_adv[perm], returns=flat_ret[perm],
    )
