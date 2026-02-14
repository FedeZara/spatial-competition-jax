"""Policy adapters that abstract distribution-specific logic.

``MAPPO`` calls only the methods defined by the :class:`PolicyAdapter`
protocol.  Two concrete implementations are provided:

``ContinuousPolicy``
    Wraps :class:`SharedActorCritic` (tanh-Gaussian movement + Beta
    price).

``DiscretePolicy``
    Wraps :class:`DiscreteActorCritic` (joint Categorical over
    location × price bins).
"""

from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp

from spatial_competition_jax.marl.mappo.networks import (
    EPS,
    DiscreteActorCritic,
    SharedActorCritic,
    _entropy_beta,
    _entropy_gaussian,
    _log_prob_beta,
    _log_prob_tanh_normal,
    symexp,
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class PolicyAdapter(Protocol):
    """Interface that every policy network must implement.

    All array shapes use the conventions:
    - ``A`` = number of agents
    - ``...`` = arbitrary leading batch dimensions
    """

    num_agents: int

    def init(
        self, key: jnp.ndarray, dummy_state: jnp.ndarray,
    ) -> Any:
        """Initialise network parameters."""
        ...

    def sample(
        self, params: Any, states: jnp.ndarray, key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample stochastic actions.

        Returns:
            ``(actions, log_probs, values)``
            - actions:   ``(..., A, action_dim)``
            - log_probs: ``(..., A)``
            - values:    ``(..., A)``
        """
        ...

    def evaluate(
        self, params: Any, states: jnp.ndarray, actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Evaluate log-prob and entropy for stored actions.

        Returns:
            ``(log_probs, entropy, values)``
            - log_probs: ``(..., A)``
            - entropy:   ``(..., A)``
            - values:    ``(..., A)``
        """
        ...

    def deterministic(
        self, params: Any, states: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return deterministic (mode) actions.

        Returns:
            ``(actions, values)``
            - actions: ``(..., A, action_dim)``
            - values:  ``(..., A)``
        """
        ...

    def value(
        self, params: Any, states: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute value estimates only.

        Returns:
            ``values``  shape ``(..., A)``
        """
        ...


# ---------------------------------------------------------------------------
# Continuous (Gaussian + Beta)
# ---------------------------------------------------------------------------


class ContinuousPolicy:
    """Adapter for :class:`SharedActorCritic` (continuous actions).

    Movement dimensions use a tanh-squashed Gaussian; bounded
    dimensions (price / quality) use a Beta distribution.
    """

    def __init__(self, network: SharedActorCritic) -> None:
        self.network = network
        self.num_agents = network.num_agents

    # -- init ---------------------------------------------------------------

    def init(
        self, key: jnp.ndarray, dummy_state: jnp.ndarray,
    ) -> Any:
        return self.network.init(key, dummy_state)

    # -- sample -------------------------------------------------------------

    def sample(
        self, params: Any, states: jnp.ndarray, key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        gauss_means, gauss_log_stds, beta_alphas, beta_betas, values_symlog = (
            self.network.apply(params, states)  # type: ignore[misc]
        )
        values = symexp(values_symlog)
        gauss_stds = jnp.exp(gauss_log_stds)

        k_gauss, k_beta = jax.random.split(key)

        # Movement (tanh-squashed Gaussian)
        raw = gauss_means + gauss_stds * jax.random.normal(
            k_gauss, gauss_means.shape,
        )
        movement = jnp.tanh(raw)
        lp_move = _log_prob_tanh_normal(gauss_means, gauss_stds, raw, movement)

        # Bounded (Beta)
        bounded = jax.random.beta(k_beta, beta_alphas, beta_betas)
        bounded = jnp.clip(bounded, EPS, 1.0 - EPS)
        lp_bounded = _log_prob_beta(beta_alphas, beta_betas, bounded)

        actions = jnp.concatenate([movement, bounded], axis=-1)
        log_probs = lp_move + lp_bounded

        return actions, log_probs, values

    # -- evaluate -----------------------------------------------------------

    def evaluate(
        self, params: Any, states: jnp.ndarray, actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        gauss_means, gauss_log_stds, beta_alphas, beta_betas, values = (
            self.network.apply(params, states)  # type: ignore[misc]
        )
        gauss_stds = jnp.exp(gauss_log_stds)

        move_dim = self.network.movement_dim
        movement_actions = actions[..., :move_dim]
        bounded_actions = actions[..., move_dim:]

        # Movement log-prob (recover pre-tanh value)
        clipped = jnp.clip(movement_actions, -1.0 + EPS, 1.0 - EPS)
        raw = jnp.arctanh(clipped)
        lp_move = _log_prob_tanh_normal(
            gauss_means, gauss_stds, raw, movement_actions,
        )

        # Bounded log-prob
        lp_bounded = _log_prob_beta(beta_alphas, beta_betas, bounded_actions)

        log_probs = lp_move + lp_bounded
        entropy = (
            _entropy_gaussian(gauss_log_stds)
            + _entropy_beta(beta_alphas, beta_betas)
        )

        return log_probs, entropy, values

    # -- deterministic ------------------------------------------------------

    def deterministic(
        self, params: Any, states: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        gauss_means, _, beta_alphas, beta_betas, values_symlog = (
            self.network.apply(params, states)  # type: ignore[misc]
        )
        values = symexp(values_symlog)

        movement = jnp.tanh(gauss_means)
        bounded = (beta_alphas - 1.0) / (beta_alphas + beta_betas - 2.0)
        actions = jnp.concatenate([movement, bounded], axis=-1)

        return actions, values

    # -- value --------------------------------------------------------------

    def value(
        self, params: Any, states: jnp.ndarray,
    ) -> jnp.ndarray:
        _, _, _, _, values_symlog = self.network.apply(params, states)  # type: ignore[misc]
        return symexp(values_symlog)


# ---------------------------------------------------------------------------
# Discrete (joint Categorical)
# ---------------------------------------------------------------------------


class DiscretePolicy:
    """Adapter for :class:`DiscreteActorCritic` (discrete actions).

    Each agent samples a single action index from a Categorical
    distribution over ``num_actions`` options.  The index is stored
    as a float32 scalar (shape ``(..., A, 1)``) so that the existing
    buffer / minibatch code works unchanged.
    """

    def __init__(self, network: DiscreteActorCritic) -> None:
        self.network = network
        self.num_agents = network.num_agents
        self.num_actions = network.num_actions

    # -- init ---------------------------------------------------------------

    def init(
        self, key: jnp.ndarray, dummy_state: jnp.ndarray,
    ) -> Any:
        return self.network.init(key, dummy_state)

    # -- sample -------------------------------------------------------------

    def sample(
        self, params: Any, states: jnp.ndarray, key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        logits, values_symlog = self.network.apply(params, states)  # type: ignore[misc]
        values = symexp(values_symlog)

        # logits: (..., A, num_actions)
        # Sample one index per agent
        flat_logits = logits.reshape(-1, self.num_actions)
        flat_indices = jax.random.categorical(key, flat_logits)  # (N,)
        indices = flat_indices.reshape(logits.shape[:-1])  # (..., A)

        # Log-prob of chosen actions
        log_probs = _categorical_log_prob(logits, indices)

        # Store as (..., A, 1) float32
        actions = indices[..., None].astype(jnp.float32)

        return actions, log_probs, values

    # -- evaluate -----------------------------------------------------------

    def evaluate(
        self, params: Any, states: jnp.ndarray, actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        logits, values = self.network.apply(params, states)  # type: ignore[misc]

        indices = actions[..., 0].astype(jnp.int32)  # (..., A)

        log_probs = _categorical_log_prob(logits, indices)
        entropy = _categorical_entropy(logits)

        return log_probs, entropy, values

    # -- deterministic ------------------------------------------------------

    def deterministic(
        self, params: Any, states: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        logits, values_symlog = self.network.apply(params, states)  # type: ignore[misc]
        values = symexp(values_symlog)

        indices = jnp.argmax(logits, axis=-1)  # (..., A)
        actions = indices[..., None].astype(jnp.float32)

        return actions, values

    # -- value --------------------------------------------------------------

    def value(
        self, params: Any, states: jnp.ndarray,
    ) -> jnp.ndarray:
        _, values_symlog = self.network.apply(params, states)  # type: ignore[misc]
        return symexp(values_symlog)


# ---------------------------------------------------------------------------
# Categorical helpers
# ---------------------------------------------------------------------------


def _categorical_log_prob(
    logits: jnp.ndarray,
    indices: jnp.ndarray,
) -> jnp.ndarray:
    """Log-probability of chosen indices under a Categorical.

    Args:
        logits: ``(..., A, num_actions)``
        indices: ``(..., A)`` integer action indices.

    Returns:
        ``(..., A)``
    """
    log_probs_all = jax.nn.log_softmax(logits, axis=-1)  # (..., A, N)
    return jnp.take_along_axis(
        log_probs_all, indices[..., None], axis=-1,
    ).squeeze(-1)


def _categorical_entropy(logits: jnp.ndarray) -> jnp.ndarray:
    """Entropy of a Categorical distribution.

    Args:
        logits: ``(..., A, num_actions)``

    Returns:
        ``(..., A)``
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jax.nn.softmax(logits, axis=-1)
    return -(probs * log_probs).sum(axis=-1)
