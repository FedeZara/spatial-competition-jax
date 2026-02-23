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
    EgoFactoredDiscreteActorCritic,
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
        bounded = ba / (ba + bb)
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
        bounded = ba / (ba + bb)
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


# ---------------------------------------------------------------------------
# Egocentric: Factored Discrete (separate Location + Price categoricals)
# ---------------------------------------------------------------------------


class EgoFactoredDiscretePolicy:
    """Egocentric adapter for :class:`EgoFactoredDiscreteActorCritic`.

    Samples location, price, and (optionally) quality **independently**
    from factored Categoricals.  Each dimension has its own entropy term
    so the PPO entropy bonus keeps location exploration alive even after
    prices converge.

    The joint action index is passed to the environment exactly like
    :class:`EgoDiscretePolicy`.
    """

    per_agent_obs = True

    def __init__(
        self, network: EgoFactoredDiscreteActorCritic, num_agents: int,
    ) -> None:
        self.network = network
        self.num_agents = num_agents
        self.num_actions = network.num_actions
        self._n_loc = network.num_location_bins
        self._n_price = network.num_price_bins
        self._n_quality = getattr(network, "num_quality_bins", 0)
        self._has_quality = self._n_quality > 0
        self._n_q = max(self._n_quality, 1)

    def init(self, key: jnp.ndarray, dummy: jnp.ndarray) -> Any:
        return self.network.init(key, dummy)

    def _forward(self, params: Any, states: jnp.ndarray):
        """Return ``(loc_logits, price_logits, [quality_logits,] value_symlog)``."""
        if states.ndim == 3:
            E, A = states.shape[0], states.shape[1]
            flat = states.reshape(E * A, -1)
            result = self.network.apply(params, flat)  # type: ignore[misc]
            if self._has_quality:
                loc_l, price_l, qual_l, v_sym = result
                return (
                    loc_l.reshape(E, A, -1),
                    price_l.reshape(E, A, -1),
                    qual_l.reshape(E, A, -1),
                    v_sym.reshape(E, A),
                )
            loc_l, price_l, v_sym = result
            return (
                loc_l.reshape(E, A, -1),
                price_l.reshape(E, A, -1),
                v_sym.reshape(E, A),
            )
        return self.network.apply(params, states)  # type: ignore[misc]

    # -- sampling ----------------------------------------------------------

    def sample(
        self, params: Any, states: jnp.ndarray, key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        result = self._forward(params, states)
        if self._has_quality:
            loc_logits, price_logits, qual_logits, v_sym = result
        else:
            loc_logits, price_logits, v_sym = result
        values = symexp(v_sym)

        E, A, _ = loc_logits.shape

        if self._has_quality:
            k_l, k_p, k_q = jax.random.split(key, 3)
        else:
            k_l, k_p = jax.random.split(key)

        loc_idx = jax.random.categorical(
            k_l, loc_logits.reshape(E * A, -1),
        ).reshape(E, A)
        price_idx = jax.random.categorical(
            k_p, price_logits.reshape(E * A, -1),
        ).reshape(E, A)

        lp = (
            _categorical_log_prob(loc_logits, loc_idx)
            + _categorical_log_prob(price_logits, price_idx)
        )

        if self._has_quality:
            qual_idx = jax.random.categorical(
                k_q, qual_logits.reshape(E * A, -1),
            ).reshape(E, A)
            joint_idx = (loc_idx * (self._n_price * self._n_q)
                         + price_idx * self._n_q + qual_idx)
            lp = lp + _categorical_log_prob(qual_logits, qual_idx)
        else:
            joint_idx = loc_idx * self._n_price + price_idx

        return joint_idx[..., None].astype(jnp.float32), lp, values

    # -- evaluate (PPO update) ---------------------------------------------

    def evaluate(
        self, params: Any, states: jnp.ndarray, actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        result = self._forward(params, states)
        if self._has_quality:
            loc_logits, price_logits, qual_logits, values = result
        else:
            loc_logits, price_logits, values = result

        joint = actions[..., 0].astype(jnp.int32)

        if self._has_quality:
            loc_idx = joint // (self._n_price * self._n_q)
            remainder = joint % (self._n_price * self._n_q)
            price_idx = remainder // self._n_q
            qual_idx = remainder % self._n_q
            lp = (
                _categorical_log_prob(loc_logits, loc_idx)
                + _categorical_log_prob(price_logits, price_idx)
                + _categorical_log_prob(qual_logits, qual_idx)
            )
            entropy = (
                _categorical_entropy(loc_logits)
                + _categorical_entropy(price_logits)
                + _categorical_entropy(qual_logits)
            )
        else:
            loc_idx = joint // self._n_price
            price_idx = joint % self._n_price
            lp = (
                _categorical_log_prob(loc_logits, loc_idx)
                + _categorical_log_prob(price_logits, price_idx)
            )
            entropy = (
                _categorical_entropy(loc_logits)
                + _categorical_entropy(price_logits)
            )
        return lp, entropy, values

    # -- deterministic -----------------------------------------------------

    def deterministic(
        self, params: Any, states: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        result = self._forward(params, states)
        if self._has_quality:
            loc_logits, price_logits, qual_logits, v_sym = result
        else:
            loc_logits, price_logits, v_sym = result

        loc_idx = jnp.argmax(loc_logits, axis=-1)
        price_idx = jnp.argmax(price_logits, axis=-1)

        if self._has_quality:
            qual_idx = jnp.argmax(qual_logits, axis=-1)
            joint_idx = (loc_idx * (self._n_price * self._n_q)
                         + price_idx * self._n_q + qual_idx)
        else:
            joint_idx = loc_idx * self._n_price + price_idx

        return joint_idx[..., None].astype(jnp.float32), symexp(v_sym)

    # -- value only --------------------------------------------------------

    def value(self, params: Any, states: jnp.ndarray) -> jnp.ndarray:
        result = self._forward(params, states)
        v_sym = result[-1]
        return symexp(v_sym)


# ---------------------------------------------------------------------------
# Egocentric: 2D Factored Discrete (loc_x × loc_y × price)
# ---------------------------------------------------------------------------


class Ego2dFactoredDiscretePolicy:
    """Egocentric adapter for :class:`EgoConv2dFactoredDiscreteActorCritic`.

    Independent categoricals: ``loc_x × loc_y × price (× quality)``.
    """

    per_agent_obs = True

    def __init__(
        self,
        network: Any,  # EgoConv2dFactoredDiscreteActorCritic
        num_agents: int,
    ) -> None:
        self.network = network
        self.num_agents = num_agents
        self._n_loc = network.num_location_bins
        self._n_price = network.num_price_bins
        self._n_quality = getattr(network, "num_quality_bins", 0)
        self._has_quality = self._n_quality > 0
        self._n_q = max(self._n_quality, 1)

    def init(self, key: jnp.ndarray, dummy: jnp.ndarray) -> Any:
        return self.network.init(key, dummy)

    def _forward(self, params: Any, states: jnp.ndarray):
        """Return ``(loc_x_logits, loc_y_logits, price_logits,
        [quality_logits,] value_symlog)``."""
        if states.ndim == 3:
            E, A = states.shape[0], states.shape[1]
            flat = states.reshape(E * A, -1)
            result = self.network.apply(params, flat)  # type: ignore[misc]
            if self._has_quality:
                lx, ly, pr, ql, v = result
                return (
                    lx.reshape(E, A, -1), ly.reshape(E, A, -1),
                    pr.reshape(E, A, -1), ql.reshape(E, A, -1),
                    v.reshape(E, A),
                )
            lx, ly, pr, v = result
            return (
                lx.reshape(E, A, -1), ly.reshape(E, A, -1),
                pr.reshape(E, A, -1), v.reshape(E, A),
            )
        return self.network.apply(params, states)  # type: ignore[misc]

    def sample(
        self, params: Any, states: jnp.ndarray, key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        result = self._forward(params, states)
        if self._has_quality:
            lx_lg, ly_lg, p_lg, q_lg, v_sym = result
        else:
            lx_lg, ly_lg, p_lg, v_sym = result
        values = symexp(v_sym)

        E, A, _ = lx_lg.shape
        flat_n = E * A

        if self._has_quality:
            k_x, k_y, k_p, k_q = jax.random.split(key, 4)
        else:
            k_x, k_y, k_p = jax.random.split(key, 3)

        loc_x = jax.random.categorical(k_x, lx_lg.reshape(flat_n, -1)).reshape(E, A)
        loc_y = jax.random.categorical(k_y, ly_lg.reshape(flat_n, -1)).reshape(E, A)
        price = jax.random.categorical(k_p, p_lg.reshape(flat_n, -1)).reshape(E, A)

        lp = (
            _categorical_log_prob(lx_lg, loc_x)
            + _categorical_log_prob(ly_lg, loc_y)
            + _categorical_log_prob(p_lg, price)
        )

        if self._has_quality:
            qual = jax.random.categorical(k_q, q_lg.reshape(flat_n, -1)).reshape(E, A)
            joint = (loc_x * (self._n_loc * self._n_price * self._n_q)
                     + loc_y * (self._n_price * self._n_q)
                     + price * self._n_q + qual)
            lp = lp + _categorical_log_prob(q_lg, qual)
        else:
            joint = loc_x * (self._n_loc * self._n_price) + loc_y * self._n_price + price

        return joint[..., None].astype(jnp.float32), lp, values

    def evaluate(
        self, params: Any, states: jnp.ndarray, actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        result = self._forward(params, states)
        if self._has_quality:
            lx_lg, ly_lg, p_lg, q_lg, values = result
        else:
            lx_lg, ly_lg, p_lg, values = result

        joint = actions[..., 0].astype(jnp.int32)

        if self._has_quality:
            loc_x = joint // (self._n_loc * self._n_price * self._n_q)
            rem1 = joint % (self._n_loc * self._n_price * self._n_q)
            loc_y = rem1 // (self._n_price * self._n_q)
            rem2 = rem1 % (self._n_price * self._n_q)
            price = rem2 // self._n_q
            qual = rem2 % self._n_q
            lp = (
                _categorical_log_prob(lx_lg, loc_x)
                + _categorical_log_prob(ly_lg, loc_y)
                + _categorical_log_prob(p_lg, price)
                + _categorical_log_prob(q_lg, qual)
            )
            entropy = (
                _categorical_entropy(lx_lg)
                + _categorical_entropy(ly_lg)
                + _categorical_entropy(p_lg)
                + _categorical_entropy(q_lg)
            )
        else:
            loc_x = joint // (self._n_loc * self._n_price)
            rem = joint % (self._n_loc * self._n_price)
            loc_y = rem // self._n_price
            price = rem % self._n_price
            lp = (
                _categorical_log_prob(lx_lg, loc_x)
                + _categorical_log_prob(ly_lg, loc_y)
                + _categorical_log_prob(p_lg, price)
            )
            entropy = (
                _categorical_entropy(lx_lg)
                + _categorical_entropy(ly_lg)
                + _categorical_entropy(p_lg)
            )
        return lp, entropy, values

    def deterministic(
        self, params: Any, states: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        result = self._forward(params, states)
        if self._has_quality:
            lx_lg, ly_lg, p_lg, q_lg, v_sym = result
        else:
            lx_lg, ly_lg, p_lg, v_sym = result

        loc_x = jnp.argmax(lx_lg, axis=-1)
        loc_y = jnp.argmax(ly_lg, axis=-1)
        price = jnp.argmax(p_lg, axis=-1)

        if self._has_quality:
            qual = jnp.argmax(q_lg, axis=-1)
            joint = (loc_x * (self._n_loc * self._n_price * self._n_q)
                     + loc_y * (self._n_price * self._n_q)
                     + price * self._n_q + qual)
        else:
            joint = loc_x * (self._n_loc * self._n_price) + loc_y * self._n_price + price

        return joint[..., None].astype(jnp.float32), symexp(v_sym)

    def value(self, params: Any, states: jnp.ndarray) -> jnp.ndarray:
        result = self._forward(params, states)
        v_sym = result[-1]
        return symexp(v_sym)
