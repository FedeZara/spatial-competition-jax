"""Best-response trainer for symmetric PSRO.

Trains a single-agent policy (``num_agents=1``) against opponents
sampled from the PSRO population according to the meta-strategy
mixture.  Modelled after :class:`MAPPO` but with key differences:

* The trainee network is ``SharedActorCritic(num_agents=1)``.
* During rollout collection the opponent's actions come from a frozen
  policy selected from the population.
* Only agent 0's rewards and values are used for GAE / PPO.
* Opponent params are a traced JIT argument (no recompilation when the
  opponent changes between rollouts).
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from spatial_competition_jax.marl.mappo.buffer import (
    RolloutBatch,
    Transition,
    compute_gae,
    make_minibatches,
    normalize_advantages,
)
from spatial_competition_jax.marl.mappo.networks import (
    EPS,
    SharedActorCritic,
    _entropy_beta,
    _entropy_gaussian,
    _log_prob_beta,
    _log_prob_tanh_normal,
    symexp,
    symlog,
)
from spatial_competition_jax.marl.psro.state_utils import permute_agent_state
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper


class BestResponseTrainer:
    """Train a best-response policy against a population mixture.

    The trainee always plays as agent 0.  At the start of each
    rollout an opponent is sampled from the population according to
    ``meta_strategy`` and plays as agent 1 (seeing the permuted state
    so it "thinks" it is agent 0 from its own perspective).

    The opponent uses **deterministic** (mode) actions; stochasticity
    comes from *which* opponent is drawn at each rollout.
    """

    def __init__(
        self,
        wrapper: TrainingWrapper,
        population: list[Any],
        meta_strategy: np.ndarray,
        *,
        hidden_dims: list[int] | None = None,
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
        warmstart_params: Any | None = None,
    ) -> None:
        self.wrapper = wrapper
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

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # ── single-agent network ──────────────────────────────────────
        self.network = SharedActorCritic(
            movement_dim=wrapper.movement_dim,
            bounded_dim=wrapper.bounded_dim,
            num_agents=1,
            hidden_dims=tuple(hidden_dims),
        )

        key = jax.random.PRNGKey(seed)
        key, init_key, reset_key = jax.random.split(key, 3)

        if warmstart_params is not None:
            params = warmstart_params
        else:
            dummy_state = jnp.zeros(wrapper.state_dim)
            params = self.network.init(init_key, dummy_state)

        # ── optimizer ─────────────────────────────────────────────────
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate),
        )

        self.train_state: TrainState = TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=tx,
        )

        # ── initial environment states ────────────────────────────────
        reset_keys = jax.random.split(reset_key, num_envs)
        self.global_states, self.env_states = jax.vmap(wrapper.reset)(reset_keys)

        self.key = key

        # NumPy RNG for opponent sampling (outside JIT)
        self._np_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_rollout(
        self,
        temperature: float | None = None,
        entropy_coef: float | None = None,
    ) -> tuple[Transition, jnp.ndarray, jnp.ndarray, dict[str, float]]:
        """Collect one rollout against a sampled opponent.

        Args:
            temperature: Optional buyer-choice temperature override.
            entropy_coef: Unused here (kept for API parity); pass to
                :meth:`update` instead.

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

        transitions, advantages, returns, env_states, global_states = (
            self._collect_rollout(
                self.train_state,
                self.env_states,
                self.global_states,
                subkey,
                temp_arr,
                opponent_params,
            )
        )
        self.env_states = env_states
        self.global_states = global_states

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
            self.train_state,
            transitions,
            advantages,
            returns,
            subkey,
            ent_coef,
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
        opponent_params: Any,
    ) -> Any:
        """Collect a rollout with the trainee as agent 0 and a frozen
        opponent as agent 1."""
        _use_temp = self.wrapper.env.buyer_choice_temperature is not None

        def scan_fn(carry: Any, _: Any) -> Any:
            env_states, global_states, key = carry
            key, step_key, k_gauss, k_beta = jax.random.split(key, 4)

            # ── Trainee forward (agent 0, original state) ─────────
            (
                gauss_means,
                gauss_log_stds,
                beta_alphas,
                beta_betas,
                values_symlog,
            ) = self.network.apply(train_state.params, global_states)  # type: ignore[misc]

            values = symexp(values_symlog)  # (E, 1)
            gauss_stds = jnp.exp(gauss_log_stds)

            # Sample trainee movement (tanh-squashed Gaussian)
            raw_movement = (
                gauss_means
                + gauss_stds * jax.random.normal(k_gauss, gauss_means.shape)
            )
            movement_actions = jnp.tanh(raw_movement)
            lp_move = _log_prob_tanh_normal(
                gauss_means, gauss_stds, raw_movement, movement_actions,
            )

            # Sample trainee bounded (Beta)
            bounded_actions = jax.random.beta(k_beta, beta_alphas, beta_betas)
            bounded_actions = jnp.clip(bounded_actions, EPS, 1.0 - EPS)
            lp_bounded = _log_prob_beta(beta_alphas, beta_betas, bounded_actions)

            # Trainee action: (E, 1, action_dim)
            trainee_actions = jnp.concatenate(
                [movement_actions, bounded_actions], axis=-1,
            )
            log_probs = lp_move + lp_bounded  # (E, 1)

            # ── Opponent forward (agent 1, permuted state) ────────
            permuted = permute_agent_state(global_states, self.wrapper)

            (
                opp_gauss_means,
                _,
                opp_beta_alphas,
                opp_beta_betas,
                _,
            ) = self.network.apply(opponent_params, permuted)  # type: ignore[misc]

            # Deterministic opponent actions
            opp_movement = jnp.tanh(opp_gauss_means)
            opp_bounded = (opp_beta_alphas - 1.0) / (
                opp_beta_alphas + opp_beta_betas - 2.0
            )
            opp_actions = jnp.concatenate(
                [opp_movement, opp_bounded], axis=-1,
            )  # (E, 1, action_dim)

            # ── Combine and step ──────────────────────────────────
            # (E, 2, action_dim) – agent 0 = trainee, agent 1 = opponent
            combined = jnp.concatenate(
                [trainee_actions, opp_actions], axis=-2,
            )

            step_keys = jax.random.split(step_key, self.num_envs)
            if _use_temp:
                next_global_states, next_env_states, rewards, dones = jax.vmap(
                    lambda k, s, a: self.wrapper.step_autoreset(
                        k, s, a, temperature=temperature,
                    ),
                )(step_keys, env_states, combined)
            else:
                next_global_states, next_env_states, rewards, dones = jax.vmap(
                    self.wrapper.step_autoreset,
                )(step_keys, env_states, combined)

            # rewards, dones: (E, 2) → keep only agent 0
            agent0_rewards = rewards[:, 0:1]  # (E, 1)
            agent0_dones = dones[:, 0:1]  # (E, 1)

            transition = Transition(
                states=global_states,
                actions=trainee_actions,
                log_probs=log_probs,
                values=values,
                rewards=agent0_rewards,
                dones=agent0_dones,
            )

            return (next_env_states, next_global_states, key), transition

        (final_env_states, final_global_states, _), transitions = (
            jax.lax.scan(
                scan_fn,
                (env_states, global_states, key),
                None,
                length=self.rollout_length,
            )
        )

        # Bootstrap value for GAE
        _, _, _, _, bootstrap_symlog = self.network.apply(  # type: ignore[misc]
            train_state.params,
            final_global_states,
        )
        bootstrap_values = symexp(bootstrap_symlog)  # (E, 1)

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

        return (
            transitions,
            advantages,
            returns,
            final_env_states,
            final_global_states,
        )

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
        _move_dim = self.network.movement_dim

        def loss_fn(
            params: Any, batch: RolloutBatch,
        ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
            (
                gauss_means,
                gauss_log_stds,
                beta_alphas,
                beta_betas,
                values,
            ) = self.network.apply(params, batch.states)  # type: ignore[misc]
            gauss_stds = jnp.exp(gauss_log_stds)

            movement_actions = batch.actions[..., :_move_dim]
            bounded_actions = batch.actions[..., _move_dim:]

            clipped_move = jnp.clip(
                movement_actions, -1.0 + EPS, 1.0 - EPS,
            )
            raw_movement = jnp.arctanh(clipped_move)
            lp_move = _log_prob_tanh_normal(
                gauss_means, gauss_stds, raw_movement, movement_actions,
            )
            lp_bounded = _log_prob_beta(
                beta_alphas, beta_betas, bounded_actions,
            )

            new_log_probs = lp_move + lp_bounded

            entropy = _entropy_gaussian(gauss_log_stds) + _entropy_beta(
                beta_alphas, beta_betas,
            )

            # PPO clipped objective
            ratio = jnp.exp(new_log_probs - batch.log_probs)
            ratio = jnp.clip(ratio, 1e-4, 10.0)  # prevent loss spikes from stale data
            surr1 = ratio * batch.advantages
            surr2 = (
                jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * batch.advantages
            )
            policy_loss = -jnp.minimum(surr1, surr2).mean()

            value_loss = ((values - batch.returns) ** 2).mean()
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
                "approx_kl": ((ratio - 1) - jnp.log(ratio)).mean(),
                "clip_fraction": (
                    jnp.abs(ratio - 1) > self.clip_epsilon
                ).astype(jnp.float32).mean(),
            }
            return total_loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        def minibatch_step(
            train_state: TrainState, batch: RolloutBatch,
        ) -> tuple[TrainState, dict[str, jnp.ndarray]]:
            (_, metrics), grads = grad_fn(train_state.params, batch)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, metrics

        def epoch_step(
            train_state: TrainState, epoch_key: jnp.ndarray,
        ) -> tuple[TrainState, dict[str, jnp.ndarray]]:
            batches = make_minibatches(
                epoch_key,
                transitions,
                advantages,
                returns,
                self.num_minibatches,
            )
            train_state, metrics = jax.lax.scan(
                minibatch_step, train_state, batches,
            )
            return train_state, metrics

        epoch_keys = jax.random.split(key, self.ppo_epochs)
        train_state, all_metrics = jax.lax.scan(
            epoch_step, train_state, epoch_keys,
        )

        avg_metrics = jax.tree.map(lambda x: x.mean(), all_metrics)
        return train_state, avg_metrics
