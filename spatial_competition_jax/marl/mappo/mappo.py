"""MAPPO algorithm implementation in JAX.

Network-agnostic: all distribution-specific logic lives behind a
:class:`PolicyAdapter` (see :mod:`policy`).  The algorithm itself only
calls ``policy.sample``, ``policy.evaluate``, ``policy.value``.

Environment vectorisation uses ``jax.vmap``; rollout collection and
PPO updates use ``jax.lax.scan``.
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
    """Linearly anneal a value from *start* to *end*.

    Interpolates over the first ``anneal_frac * total_updates``
    updates, then holds at *end* for the remainder.

    Args:
        update: Current update number (1-based).
        total_updates: Total number of training updates.
        start: Initial value.
        end: Final value.
        anneal_frac: Fraction of training over which to anneal.

    Returns:
        Annealed value for the current update.
    """
    anneal_updates = max(int(total_updates * anneal_frac), 1)
    frac = min(update / anneal_updates, 1.0)
    return start + (end - start) * frac


# Backward-compatible alias
compute_temperature = linear_anneal


class MAPPO:
    """MAPPO with a pluggable actor-critic policy.

    All heavy computation (rollout collection, PPO update) is
    JIT-compiled.  Environment vectorisation happens through
    ``jax.vmap`` – no sub-processes required.
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

        key = jax.random.PRNGKey(seed)
        key, init_key, reset_key = jax.random.split(key, 3)

        # ── initialise parameters ─────────────────────────────────────
        dummy_state = jnp.zeros(wrapper.state_dim)
        params = policy.init(init_key, dummy_state)

        # ── optimizer ─────────────────────────────────────────────────
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate),
        )

        self.train_state: TrainState = TrainState.create(
            apply_fn=None,  # not used – we go through policy adapter
            params=params,
            tx=tx,
        )

        # ── initial environment states ────────────────────────────────
        reset_keys = jax.random.split(reset_key, num_envs)
        self.global_states, self.env_states = jax.vmap(wrapper.reset)(reset_keys)

        self.key = key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_rollout(
        self,
        temperature: float | None = None,
    ) -> tuple[Transition, jnp.ndarray, jnp.ndarray, dict[str, float]]:
        """Collect one rollout and compute GAE.

        Args:
            temperature: Optional buyer-choice temperature override
                for this rollout (used for annealing schedules).

        Returns:
            ``(transitions, advantages, returns, stats)``
        """
        self.key, subkey = jax.random.split(self.key)

        temp_arr = jnp.float32(temperature if temperature is not None else 0.0)

        transitions, advantages, returns, env_states, global_states = self._collect_rollout(
            self.train_state,
            self.env_states,
            self.global_states,
            subkey,
            temp_arr,
        )
        self.env_states = env_states
        self.global_states = global_states

        stats = {
            "mean_reward": float(transitions.rewards.mean()),
            "std_reward": float(transitions.rewards.std()),
            "mean_value": float(transitions.values.mean()),
            "mean_return": float(returns.mean()),
            "total_reward": float(transitions.rewards.sum()),
        }

        return transitions, advantages, returns, stats

    def update(
        self,
        transitions: Transition,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        entropy_coef: float | None = None,
    ) -> dict[str, float]:
        """Run PPO update epochs.

        Returns:
            Dictionary of averaged training metrics.
        """
        self.key, subkey = jax.random.split(self.key)

        ent_coef = jnp.float32(entropy_coef if entropy_coef is not None else self.entropy_coef)

        self.train_state, metrics = self._update(
            self.train_state,
            transitions,
            advantages,
            returns,
            subkey,
            ent_coef,
        )

        return {k: float(v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def params(self) -> Any:
        """Current network parameters."""
        return self.train_state.params

    @property
    def opt_state(self) -> Any:
        """Current optimiser state."""
        return self.train_state.opt_state

    # ------------------------------------------------------------------
    # JIT-compiled internals
    # ------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0,))
    def _collect_rollout(
        self,
        train_state: TrainState,
        env_states: Any,
        global_states: jnp.ndarray,
        key: jnp.ndarray,
        temperature: jnp.ndarray,
    ) -> Any:
        """Collect a rollout of *rollout_length* steps using ``jax.lax.scan``."""
        _use_temp = self.wrapper.env.buyer_choice_temperature is not None

        def scan_fn(carry: Any, _: Any) -> Any:
            env_states, global_states, key = carry
            key, step_key, sample_key = jax.random.split(key, 3)

            # ── Policy forward: sample actions ────────────────────
            actions, log_probs, values = self.policy.sample(
                train_state.params, global_states, sample_key,
            )

            # ── Step all envs (vmapped) ───────────────────────────
            step_keys = jax.random.split(step_key, self.num_envs)
            if _use_temp:
                next_global_states, next_env_states, rewards, dones = jax.vmap(
                    lambda k, s, a: self.wrapper.step_autoreset(k, s, a, temperature=temperature),
                )(step_keys, env_states, actions)
            else:
                next_global_states, next_env_states, rewards, dones = jax.vmap(
                    self.wrapper.step_autoreset,
                )(step_keys, env_states, actions)

            transition = Transition(
                states=global_states,
                actions=actions,
                log_probs=log_probs,
                values=values,
                rewards=rewards,
                dones=dones,
            )

            return (next_env_states, next_global_states, key), transition

        (final_env_states, final_global_states, _), transitions = jax.lax.scan(
            scan_fn,
            (env_states, global_states, key),
            None,
            length=self.rollout_length,
        )

        # Bootstrap value
        bootstrap_values = self.policy.value(
            train_state.params, final_global_states,
        )

        # GAE
        advantages, returns = compute_gae(
            transitions.rewards,
            transitions.values,
            transitions.dones,
            bootstrap_values,
            self.gamma,
            self.gae_lambda,
        )

        # Normalise advantages (per agent)
        advantages = normalize_advantages(advantages, per_agent=True)
        # Compress returns to symlog space to match critic output scale
        returns = symlog(returns)

        return transitions, advantages, returns, final_env_states, final_global_states

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
        """Run *ppo_epochs* of PPO updates using ``jax.lax.scan``."""

        def loss_fn(params: Any, batch: RolloutBatch) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
            # ── Policy evaluate: log-probs, entropy, values ───────
            new_log_probs, entropy, values = self.policy.evaluate(
                params, batch.states, batch.actions,
            )

            # PPO clipped objective
            ratio = jnp.exp(new_log_probs - batch.log_probs)
            surr1 = ratio * batch.advantages
            surr2 = jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch.advantages
            policy_loss = -jnp.minimum(surr1, surr2).mean()

            # Value loss
            value_loss = ((values - batch.returns) ** 2).mean()

            # Entropy bonus
            entropy_loss = -entropy.mean()

            total_loss = policy_loss + self.value_coef * value_loss + entropy_coef * entropy_loss

            metrics = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy.mean(),
                "entropy_coef": entropy_coef,
                "approx_kl": ((ratio - 1) - jnp.log(ratio)).mean(),
                "clip_fraction": (jnp.abs(ratio - 1) > self.clip_epsilon).astype(jnp.float32).mean(),
            }
            return total_loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        def minibatch_step(train_state: TrainState, batch: RolloutBatch) -> tuple[TrainState, dict[str, jnp.ndarray]]:
            (_, metrics), grads = grad_fn(train_state.params, batch)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, metrics

        def epoch_step(train_state: TrainState, epoch_key: jnp.ndarray) -> tuple[TrainState, dict[str, jnp.ndarray]]:
            batches = make_minibatches(
                epoch_key,
                transitions,
                advantages,
                returns,
                self.num_minibatches,
            )
            train_state, metrics = jax.lax.scan(minibatch_step, train_state, batches)
            return train_state, metrics

        epoch_keys = jax.random.split(key, self.ppo_epochs)
        train_state, all_metrics = jax.lax.scan(epoch_step, train_state, epoch_keys)

        # Average over epochs × minibatches
        avg_metrics = jax.tree.map(lambda x: x.mean(), all_metrics)

        return train_state, avg_metrics
