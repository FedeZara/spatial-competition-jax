"""Policy adapters that abstract distribution-specific logic.

``MAPPO`` calls only the methods defined by the :class:`PolicyAdapter`
protocol.  Four concrete implementations are provided:

**Global observation** (one state vector, per-agent heads):
- ``ContinuousPolicy``  – tanh-Gaussian movement + Beta price
- ``DiscretePolicy``     – joint Categorical

**Egocentric observation** (per-agent obs, shared single-agent net):
- ``EgoContinuousPolicy``  – same distributions, ego reshape
- ``EgoDiscretePolicy``    – Categorical, ego reshape
"""

from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp

from spatial_competition_jax.marl.mappo.networks import (
    EPS,
    DiscreteActorCritic,
    EgoActorCritic,
    EgoDiscreteActorCritic,
    SharedActorCritic,
    _categorical_entropy,
    _categorical_log_prob,
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
    """Interface that every policy must implement.

    Shape conventions:
    - ``E`` = environments,  ``A`` = agents
    - Global mode:  states ``(E, state_dim)``, outputs ``(E, A, ...)``
    - Ego mode:     states ``(E, A, obs_dim)``, outputs ``(E, A, ...)``
    """

    num_agents: int
    per_agent_obs: bool  # True → (E, A, obs_dim); False → (E, state_dim)

    def init(self, key: jnp.ndarray, dummy: jnp.ndarray) -> Any: ...

    def sample(
        self, params: Any, states: jnp.ndarray, key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Returns ``(actions, log_probs, values)`` — shapes ``(E, A, ...)``."""
        ...

    def evaluate(
        self, params: Any, states: jnp.ndarray, actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Returns ``(log_probs, entropy, values)`` — shapes ``(E, A)``."""
        ...

    def deterministic(
        self, params: Any, states: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Returns ``(actions, values)``."""
        ...

    def value(self, params: Any, states: jnp.ndarray) -> jnp.ndarray:
        """Returns ``values`` — shape ``(E, A)``."""
        ...


# ---------------------------------------------------------------------------
# Global: Continuous (Gaussian + Beta)
# ---------------------------------------------------------------------------


class ContinuousPolicy:
    """Global-state adapter for :class:`SharedActorCritic`."""

    per_agent_obs = False

    def __init__(self, network: SharedActorCritic) -> None:
        self.network = network
        self.num_agents = network.num_agents

    def init(self, key: jnp.ndarray, dummy: jnp.ndarray) -> Any:
        return self.network.init(key, dummy)

    def sample(
        self, params: Any, states: jnp.ndarray, key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        gm, gls, ba, bb, v_sym = self.network.apply(params, states)  # type: ignore[misc]
        values = symexp(v_sym)
        gs = jnp.exp(gls)
        k_g, k_b = jax.random.split(key)
        raw = gm + gs * jax.random.normal(k_g, gm.shape)
        movement = jnp.tanh(raw)
        lp_m = _log_prob_tanh_normal(gm, gs, raw, movement)
        bounded = jax.random.beta(k_b, ba, bb)
        bounded = jnp.clip(bounded, EPS, 1.0 - EPS)
        lp_b = _log_prob_beta(ba, bb, bounded)
        return jnp.concatenate([movement, bounded], axis=-1), lp_m + lp_b, values

    def evaluate(
        self, params: Any, states: jnp.ndarray, actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        gm, gls, ba, bb, values = self.network.apply(params, states)  # type: ignore[misc]
        gs = jnp.exp(gls)
        d = self.network.movement_dim
        m_a, b_a = actions[..., :d], actions[..., d:]
        raw = jnp.arctanh(jnp.clip(m_a, -1.0 + EPS, 1.0 - EPS))
        lp_m = _log_prob_tanh_normal(gm, gs, raw, m_a)
        lp_b = _log_prob_beta(ba, bb, b_a)
        entropy = _entropy_gaussian(gls) + _entropy_beta(ba, bb)
        return lp_m + lp_b, entropy, values

    def deterministic(
        self, params: Any, states: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        gm, _, ba, bb, v_sym = self.network.apply(params, states)  # type: ignore[misc]
        movement = jnp.tanh(gm)
        bounded = (ba - 1.0) / (ba + bb - 2.0)
        return jnp.concatenate([movement, bounded], axis=-1), symexp(v_sym)

    def value(self, params: Any, states: jnp.ndarray) -> jnp.ndarray:
        _, _, _, _, v_sym = self.network.apply(params, states)  # type: ignore[misc]
        return symexp(v_sym)


# ---------------------------------------------------------------------------
# Global: Discrete (joint Categorical)
# ---------------------------------------------------------------------------


class DiscretePolicy:
    """Global-state adapter for :class:`DiscreteActorCritic`."""

    per_agent_obs = False

    def __init__(self, network: DiscreteActorCritic) -> None:
        self.network = network
        self.num_agents = network.num_agents
        self.num_actions = network.num_actions

    def init(self, key: jnp.ndarray, dummy: jnp.ndarray) -> Any:
        return self.network.init(key, dummy)

    def sample(
        self, params: Any, states: jnp.ndarray, key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        logits, v_sym = self.network.apply(params, states)  # type: ignore[misc]
        values = symexp(v_sym)
        flat = logits.reshape(-1, self.num_actions)
        idx = jax.random.categorical(key, flat).reshape(logits.shape[:-1])
        lp = _categorical_log_prob(logits, idx)
        return idx[..., None].astype(jnp.float32), lp, values

    def evaluate(
        self, params: Any, states: jnp.ndarray, actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        logits, values = self.network.apply(params, states)  # type: ignore[misc]
        idx = actions[..., 0].astype(jnp.int32)
        return _categorical_log_prob(logits, idx), _categorical_entropy(logits), values

    def deterministic(
        self, params: Any, states: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        logits, v_sym = self.network.apply(params, states)  # type: ignore[misc]
        idx = jnp.argmax(logits, axis=-1)
        return idx[..., None].astype(jnp.float32), symexp(v_sym)

    def value(self, params: Any, states: jnp.ndarray) -> jnp.ndarray:
        _, v_sym = self.network.apply(params, states)  # type: ignore[misc]
        return symexp(v_sym)


# ---------------------------------------------------------------------------
# Egocentric: Continuous (Gaussian + Beta)
# ---------------------------------------------------------------------------


class EgoContinuousPolicy:
    """Egocentric adapter for :class:`EgoActorCritic`.

    Accepts ``(E, A, obs_dim)`` states, flattens to ``(E*A, obs_dim)``
    for the network forward pass, then reshapes outputs back to
    ``(E, A, ...)``.
    """

    per_agent_obs = True

    def __init__(self, network: EgoActorCritic, num_agents: int) -> None:
        self.network = network
        self.num_agents = num_agents

    def init(self, key: jnp.ndarray, dummy: jnp.ndarray) -> Any:
        return self.network.init(key, dummy)

    def _forward(self, params: Any, states: jnp.ndarray) -> tuple:
        """Forward pass with automatic reshape.

        Accepts ``(E, A, obs_dim)`` during rollout collection or
        ``(B, obs_dim)`` during PPO updates (already flat from minibatching).
        """
        if states.ndim == 3:
            E, A = states.shape[0], states.shape[1]
            flat = states.reshape(E * A, -1)
            gm, gls, ba, bb, v_sym = self.network.apply(params, flat)  # type: ignore[misc]
            return (
                gm.reshape(E, A, -1), gls.reshape(E, A, -1),
                ba.reshape(E, A, -1), bb.reshape(E, A, -1),
                v_sym.reshape(E, A),
            )
        # Already flat (B, obs_dim) — apply directly
        return self.network.apply(params, states)  # type: ignore[misc]

    def sample(
        self, params: Any, states: jnp.ndarray, key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        gm, gls, ba, bb, v_sym = self._forward(params, states)
        values = symexp(v_sym)
        gs = jnp.exp(gls)
        k_g, k_b = jax.random.split(key)
        raw = gm + gs * jax.random.normal(k_g, gm.shape)
        movement = jnp.tanh(raw)
        lp_m = _log_prob_tanh_normal(gm, gs, raw, movement)
        bounded = jax.random.beta(k_b, ba, bb)
        bounded = jnp.clip(bounded, EPS, 1.0 - EPS)
        lp_b = _log_prob_beta(ba, bb, bounded)
        return jnp.concatenate([movement, bounded], axis=-1), lp_m + lp_b, values

    def evaluate(
        self, params: Any, states: jnp.ndarray, actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        gm, gls, ba, bb, values = self._forward(params, states)
        gs = jnp.exp(gls)
        d = self.network.movement_dim
        m_a, b_a = actions[..., :d], actions[..., d:]
        raw = jnp.arctanh(jnp.clip(m_a, -1.0 + EPS, 1.0 - EPS))
        lp_m = _log_prob_tanh_normal(gm, gs, raw, m_a)
        lp_b = _log_prob_beta(ba, bb, b_a)
        entropy = _entropy_gaussian(gls) + _entropy_beta(ba, bb)
        return lp_m + lp_b, entropy, values

    def deterministic(
        self, params: Any, states: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        gm, _, ba, bb, v_sym = self._forward(params, states)
        movement = jnp.tanh(gm)
        bounded = (ba - 1.0) / (ba + bb - 2.0)
        return jnp.concatenate([movement, bounded], axis=-1), symexp(v_sym)

    def value(self, params: Any, states: jnp.ndarray) -> jnp.ndarray:
        _, _, _, _, v_sym = self._forward(params, states)
        return symexp(v_sym)


# ---------------------------------------------------------------------------
# Egocentric: Discrete (joint Categorical)
# ---------------------------------------------------------------------------


class EgoDiscretePolicy:
    """Egocentric adapter for :class:`EgoDiscreteActorCritic`.

    Accepts ``(E, A, obs_dim)`` states, flattens/reshapes internally.
    """

    per_agent_obs = True

    def __init__(self, network: EgoDiscreteActorCritic, num_agents: int) -> None:
        self.network = network
        self.num_agents = num_agents
        self.num_actions = network.num_actions

    def init(self, key: jnp.ndarray, dummy: jnp.ndarray) -> Any:
        return self.network.init(key, dummy)

    def _forward(self, params: Any, states: jnp.ndarray) -> tuple:
        if states.ndim == 3:
            E, A = states.shape[0], states.shape[1]
            flat = states.reshape(E * A, -1)
            logits, v_sym = self.network.apply(params, flat)  # type: ignore[misc]
            return logits.reshape(E, A, -1), v_sym.reshape(E, A)
        return self.network.apply(params, states)  # type: ignore[misc]

    def sample(
        self, params: Any, states: jnp.ndarray, key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        logits, v_sym = self._forward(params, states)
        values = symexp(v_sym)
        E, A, N = logits.shape
        flat = logits.reshape(E * A, N)
        idx = jax.random.categorical(key, flat).reshape(E, A)
        lp = _categorical_log_prob(logits, idx)
        return idx[..., None].astype(jnp.float32), lp, values

    def evaluate(
        self, params: Any, states: jnp.ndarray, actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        logits, values = self._forward(params, states)
        idx = actions[..., 0].astype(jnp.int32)
        return _categorical_log_prob(logits, idx), _categorical_entropy(logits), values

    def deterministic(
        self, params: Any, states: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        logits, v_sym = self._forward(params, states)
        idx = jnp.argmax(logits, axis=-1)
        return idx[..., None].astype(jnp.float32), symexp(v_sym)

    def value(self, params: Any, states: jnp.ndarray) -> jnp.ndarray:
        _, v_sym = self._forward(params, states)
        return symexp(v_sym)
