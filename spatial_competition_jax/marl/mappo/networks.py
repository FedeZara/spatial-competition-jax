"""Actor-Critic networks implemented in Flax linen.

Two architectures are provided:

``SharedActorCritic``
    Continuous actions via tanh-squashed Gaussian (movement) and
    Beta distributions (price / quality).

``DiscreteActorCritic``
    Discrete actions via a joint Categorical over
    ``num_actions = num_location_bins * num_price_bins`` options.
"""

from __future__ import annotations

from typing import Any, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import orthogonal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPS = 1e-6

Dtype = Any  # jnp.float32, jnp.bfloat16, …


# ---------------------------------------------------------------------------
# Beta-distribution helpers
# ---------------------------------------------------------------------------


def _lbeta(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Log of the Beta function: log B(a, b)."""
    return jax.lax.lgamma(a) + jax.lax.lgamma(b) - jax.lax.lgamma(a + b)


def _log_prob_beta(
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Log-probability of a Beta distribution, summed over the last axis.

    Args:
        alpha, beta: ``(..., bounded_dim)``  (both > 0)
        x: ``(..., bounded_dim)`` in ``(0, 1)``

    Returns:
        ``(...,)``
    """
    x = jnp.clip(x, EPS, 1.0 - EPS)
    log_prob = (alpha - 1.0) * jnp.log(x) + (beta - 1.0) * jnp.log(1.0 - x) - _lbeta(alpha, beta)
    return log_prob.sum(axis=-1)


def _entropy_beta(
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
) -> jnp.ndarray:
    """Differential entropy of a Beta distribution, summed over the last axis.

    Args:
        alpha, beta: ``(..., bounded_dim)``

    Returns:
        ``(...,)``
    """
    digamma = jax.scipy.special.digamma
    ent = (
        _lbeta(alpha, beta)
        - (alpha - 1.0) * digamma(alpha)
        - (beta - 1.0) * digamma(beta)
        + (alpha + beta - 2.0) * digamma(alpha + beta)
    )
    return ent.sum(axis=-1)


# ---------------------------------------------------------------------------
# Symlog / Symexp value transforms (DreamerV3-style)
# ---------------------------------------------------------------------------


def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric logarithmic compression: sign(x) * ln(1 + |x|)."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse of :func:`symlog`: sign(x) * (exp(|x|) - 1)."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


# ---------------------------------------------------------------------------
# Tanh-Normal helpers
# ---------------------------------------------------------------------------


def _log_prob_tanh_normal(
    means: jnp.ndarray,
    stds: jnp.ndarray,
    raw_actions: jnp.ndarray,
    squashed_actions: jnp.ndarray,
) -> jnp.ndarray:
    """Log-probability of a tanh-squashed Normal, summed over the last axis.

    Args:
        means, stds: ``(..., movement_dim)``
        raw_actions: Pre-tanh values ``(..., movement_dim)``
        squashed_actions: Post-tanh values ``(..., movement_dim)``

    Returns:
        ``(...,)``
    """
    var = stds**2
    log_prob_raw = -0.5 * ((raw_actions - means) ** 2 / var + jnp.log(var) + jnp.log(2 * jnp.pi))
    log_det_jacobian = jnp.log(1 - squashed_actions**2 + EPS)
    return (log_prob_raw - log_det_jacobian).sum(axis=-1)


def _entropy_gaussian(log_stds: jnp.ndarray) -> jnp.ndarray:
    """Gaussian entropy (pre-squashing), summed over the last axis.

    Args:
        log_stds: ``(..., movement_dim)``

    Returns:
        ``(...,)``
    """
    return (0.5 * (1.0 + jnp.log(2 * jnp.pi) + 2 * log_stds)).sum(axis=-1)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class SharedActorCritic(nn.Module):
    """Unified Actor-Critic with shared backbone and per-agent heads.

    Movement dimensions use a tanh-squashed Gaussian; bounded
    dimensions (price and optionally quality) use a Beta distribution.

    Mixed-precision support
    ~~~~~~~~~~~~~~~~~~~~~~~
    Set ``dtype=jnp.bfloat16`` (or ``jnp.float16``) and keep
    ``param_dtype=jnp.float32`` to run all Dense matmuls in reduced
    precision while storing and optimising parameters in full
    precision.  Outputs are always promoted back to float32 so that
    downstream numerics (log-probs, GAE, loss) stay precise.
    """

    movement_dim: int
    bounded_dim: int
    num_agents: int
    hidden_dims: Sequence[int] = (256, 256)
    dtype: Dtype = jnp.bfloat16
    param_dtype: Dtype = jnp.float32
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @property
    def action_dim(self) -> int:
        """Total action dimension (movement + bounded)."""
        return self.movement_dim + self.bounded_dim

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            state: ``(..., state_dim)``

        Returns:
            ``(gauss_means, gauss_log_stds, beta_alphas, beta_betas, values)``
            always float32, with shapes:
            - gauss_means:    ``(..., A, movement_dim)``
            - gauss_log_stds: ``(..., A, movement_dim)``
            - beta_alphas:    ``(..., A, bounded_dim)``
            - beta_betas:     ``(..., A, bounded_dim)``
            - values:         ``(..., A)``
        """
        # Cast input to computation dtype (e.g. bfloat16)
        x = state.astype(self.dtype)

        # Shared backbone
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(
                dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(jnp.sqrt(2)),
                name=f"backbone_{i}",
            )(x)
            x = nn.relu(x)

        features = x

        # Per-agent actor heads
        gauss_means_list = []
        gauss_log_stds_list = []
        beta_alphas_list = []
        beta_betas_list = []

        for i in range(self.num_agents):
            # --- Gaussian head (movement) ---
            g_mean = nn.Dense(
                self.movement_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(0.01),
                name=f"gauss_mean_{i}",
            )(features)
            g_log_std = nn.Dense(
                self.movement_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(0.01),
                name=f"gauss_log_std_{i}",
            )(features)
            g_log_std = jnp.clip(g_log_std, self.log_std_min, self.log_std_max)
            gauss_means_list.append(g_mean)
            gauss_log_stds_list.append(g_log_std)

            # --- Beta head (price, and optionally quality) ---
            b_alpha_raw = nn.Dense(
                self.bounded_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(0.01),
                name=f"beta_alpha_{i}",
            )(features)
            b_beta_raw = nn.Dense(
                self.bounded_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(0.01),
                name=f"beta_beta_{i}",
            )(features)
            # Clamp raws before softplus to avoid bfloat16 overflow
            b_alpha_raw = jnp.clip(b_alpha_raw, -20.0, 20.0)
            b_beta_raw = jnp.clip(b_beta_raw, -20.0, 20.0)
            # softplus + 1 ensures alpha, beta > 1  (unimodal)
            b_alpha = nn.softplus(b_alpha_raw) + 1.0
            b_beta = nn.softplus(b_beta_raw) + 1.0
            beta_alphas_list.append(b_alpha)
            beta_betas_list.append(b_beta)

        gauss_means_arr = jnp.stack(gauss_means_list, axis=-2)
        gauss_log_stds_arr = jnp.stack(gauss_log_stds_list, axis=-2)
        beta_alphas_arr = jnp.stack(beta_alphas_list, axis=-2)
        beta_betas_arr = jnp.stack(beta_betas_list, axis=-2)

        # Critic head
        values = nn.Dense(
            self.num_agents,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=orthogonal(1.0),
            name="critic",
        )(features)

        # Promote outputs to float32 for numerically-sensitive
        # downstream ops (log-probs, GAE, PPO loss).
        return (
            gauss_means_arr.astype(jnp.float32),
            gauss_log_stds_arr.astype(jnp.float32),
            beta_alphas_arr.astype(jnp.float32),
            beta_betas_arr.astype(jnp.float32),
            values.astype(jnp.float32),
        )


# ---------------------------------------------------------------------------
# Action-sampling utilities (pure functions)
# ---------------------------------------------------------------------------


def sample_actions(
    network: SharedActorCritic,
    params: dict,
    state: jnp.ndarray,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample stochastic actions.

    Movement dims are sampled from a tanh-squashed Gaussian;
    bounded dims (price/quality) are sampled from a Beta distribution.

    Args:
        network: ``SharedActorCritic`` module.
        params: Network parameters.
        state: ``(..., state_dim)``
        key: PRNG key.

    Returns:
        ``(actions, log_probs, values)`` with shapes
        ``(..., A, action_dim)``, ``(..., A)``, ``(..., A)``.
        Movement entries are in ``[-1, 1]``, bounded entries in ``(0, 1)``.
    """
    gauss_means, gauss_log_stds, beta_alphas, beta_betas, values = network.apply(params, state)  # type: ignore[misc]
    gauss_stds = jnp.exp(gauss_log_stds)

    k_gauss, k_beta = jax.random.split(key)

    # --- Gaussian (movement) ---
    raw_movement = gauss_means + gauss_stds * jax.random.normal(k_gauss, gauss_means.shape)
    movement_actions = jnp.tanh(raw_movement)
    log_prob_movement = _log_prob_tanh_normal(gauss_means, gauss_stds, raw_movement, movement_actions)

    # --- Beta (price / quality) ---
    bounded_actions = jax.random.beta(k_beta, beta_alphas, beta_betas)
    # Clamp to avoid exact 0 / 1 which break log-prob
    bounded_actions = jnp.clip(bounded_actions, EPS, 1.0 - EPS)
    log_prob_bounded = _log_prob_beta(beta_alphas, beta_betas, bounded_actions)

    # --- concatenate ---
    actions = jnp.concatenate([movement_actions, bounded_actions], axis=-1)
    log_probs = log_prob_movement + log_prob_bounded

    return actions, log_probs, values


def deterministic_actions(
    network: SharedActorCritic,
    params: dict,
    state: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return deterministic actions.

    Movement: ``tanh(mean)``.
    Bounded: Beta mode ``(alpha - 1) / (alpha + beta - 2)``
    (well-defined because alpha, beta > 1).

    Returns:
        ``(actions, values)`` with shapes
        ``(..., A, action_dim)``, ``(..., A)``.
    """
    gauss_means, _, beta_alphas, beta_betas, values = network.apply(params, state)  # type: ignore[misc]

    movement_actions = jnp.tanh(gauss_means)
    bounded_actions = (beta_alphas - 1.0) / (beta_alphas + beta_betas - 2.0)

    actions = jnp.concatenate([movement_actions, bounded_actions], axis=-1)
    return actions, values


# ---------------------------------------------------------------------------
# Discrete Actor-Critic (joint Categorical)
# ---------------------------------------------------------------------------


class DiscreteActorCritic(nn.Module):
    """Actor-Critic with a joint Categorical over discrete actions.

    Each agent independently picks one of ``num_actions`` options per
    step, where each option encodes a (location, price) pair.

    Architecture::

        Global State -> [Shared Backbone] -> Features
                                |
                    +-----------+-----------+
                    |                       |
              [Per-Agent Logit Heads]  [Critic Head]
              (A, num_actions)         (A,)

    ``__call__`` returns ``(logits, values)`` both in float32.
    """

    num_actions: int
    num_agents: int
    hidden_dims: Sequence[int] = (256, 256)
    dtype: Dtype = jnp.bfloat16
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self, state: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            state: ``(..., state_dim)``

        Returns:
            ``(logits, values)`` with shapes:
            - logits: ``(..., A, num_actions)``
            - values: ``(..., A)``
        """
        x = state.astype(self.dtype)

        # Shared backbone
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(
                dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(jnp.sqrt(2)),
                name=f"backbone_{i}",
            )(x)
            x = nn.relu(x)

        features = x

        # Per-agent logit heads
        logits_list = []
        for i in range(self.num_agents):
            logits_i = nn.Dense(
                self.num_actions,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(0.01),
                name=f"logits_{i}",
            )(features)
            logits_list.append(logits_i)

        logits = jnp.stack(logits_list, axis=-2)  # (..., A, num_actions)

        # Critic head
        values = nn.Dense(
            self.num_agents,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=orthogonal(1.0),
            name="critic",
        )(features)

        return (
            logits.astype(jnp.float32),
            values.astype(jnp.float32),
        )
