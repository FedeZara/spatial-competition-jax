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
    Bounded: Beta mean ``alpha / (alpha + beta)``.

    Returns:
        ``(actions, values)`` with shapes
        ``(..., A, action_dim)``, ``(..., A)``.
    """
    gauss_means, _, beta_alphas, beta_betas, values = network.apply(params, state)  # type: ignore[misc]

    movement_actions = jnp.tanh(gauss_means)
    bounded_actions = beta_alphas / (beta_alphas + beta_betas)

    actions = jnp.concatenate([movement_actions, bounded_actions], axis=-1)
    return actions, values


# ---------------------------------------------------------------------------
# Egocentric Actor-Critic (single-agent, shared across agents)
# ---------------------------------------------------------------------------


class EgoActorCritic(nn.Module):
    """Actor-Critic for per-agent egocentric observations.

    Takes a single agent's observation and produces that agent's
    action distribution parameters and value estimate.  The same
    network is applied to every agent (weight sharing → symmetric
    strategies).

    Movement dimensions use a tanh-squashed Gaussian; bounded
    dimensions (price and optionally quality) use a Beta distribution.
    """

    movement_dim: int
    bounded_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    dtype: Dtype = jnp.bfloat16
    param_dtype: Dtype = jnp.float32
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @property
    def action_dim(self) -> int:
        return self.movement_dim + self.bounded_dim

    @nn.compact
    def __call__(
        self, obs: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            obs: ``(..., obs_dim)`` — a single agent's observation.

        Returns:
            ``(gauss_means, gauss_log_stds, beta_alphas, beta_betas, values)``
            always float32, with shapes:
            - gauss_means:    ``(..., movement_dim)``
            - gauss_log_stds: ``(..., movement_dim)``
            - beta_alphas:    ``(..., bounded_dim)``
            - beta_betas:     ``(..., bounded_dim)``
            - values:         ``(...,)``
        """
        x = obs.astype(self.dtype)

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

        # --- Gaussian head (movement) ---
        gauss_mean = nn.Dense(
            self.movement_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=orthogonal(0.01),
            name="gauss_mean",
        )(features)
        gauss_log_std = nn.Dense(
            self.movement_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=orthogonal(0.01),
            name="gauss_log_std",
        )(features)
        gauss_log_std = jnp.clip(gauss_log_std, self.log_std_min, self.log_std_max)

        # --- Beta head (price, and optionally quality) ---
        beta_alpha_raw = nn.Dense(
            self.bounded_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=orthogonal(0.01),
            name="beta_alpha",
        )(features)
        beta_beta_raw = nn.Dense(
            self.bounded_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=orthogonal(0.01),
            name="beta_beta",
        )(features)
        beta_alpha_raw = jnp.clip(beta_alpha_raw, -20.0, 20.0)
        beta_beta_raw = jnp.clip(beta_beta_raw, -20.0, 20.0)
        beta_alpha = nn.softplus(beta_alpha_raw) + 1.0
        beta_beta = nn.softplus(beta_beta_raw) + 1.0

        # --- Critic head (scalar value) ---
        value = nn.Dense(
            1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=orthogonal(1.0),
            name="critic",
        )(features).squeeze(-1)

        return (
            gauss_mean.astype(jnp.float32),
            gauss_log_std.astype(jnp.float32),
            beta_alpha.astype(jnp.float32),
            beta_beta.astype(jnp.float32),
            value.astype(jnp.float32),
        )


def ego_sample_actions(
    network: EgoActorCritic,
    params: dict,
    obs: jnp.ndarray,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample stochastic actions for per-agent egocentric observations.

    Args:
        network: ``EgoActorCritic`` module.
        params: Network parameters.
        obs: ``(..., obs_dim)`` — per-agent observations.
        key: PRNG key.

    Returns:
        ``(actions, log_probs, values)`` with shapes
        ``(..., action_dim)``, ``(...,)``, ``(...,)``.
    """
    gauss_means, gauss_log_stds, beta_alphas, beta_betas, values = network.apply(params, obs)  # type: ignore[misc]
    gauss_stds = jnp.exp(gauss_log_stds)

    k_gauss, k_beta = jax.random.split(key)

    raw_movement = gauss_means + gauss_stds * jax.random.normal(k_gauss, gauss_means.shape)
    movement_actions = jnp.tanh(raw_movement)
    log_prob_movement = _log_prob_tanh_normal(gauss_means, gauss_stds, raw_movement, movement_actions)

    bounded_actions = jax.random.beta(k_beta, beta_alphas, beta_betas)
    bounded_actions = jnp.clip(bounded_actions, EPS, 1.0 - EPS)
    log_prob_bounded = _log_prob_beta(beta_alphas, beta_betas, bounded_actions)

    actions = jnp.concatenate([movement_actions, bounded_actions], axis=-1)
    log_probs = log_prob_movement + log_prob_bounded

    return actions, log_probs, values


def ego_deterministic_actions(
    network: EgoActorCritic,
    params: dict,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return deterministic actions for per-agent egocentric observations.

    Returns:
        ``(actions, values)`` with shapes ``(..., action_dim)``, ``(...,)``.
    """
    gauss_means, _, beta_alphas, beta_betas, values = network.apply(params, obs)  # type: ignore[misc]

    movement_actions = jnp.tanh(gauss_means)
    bounded_actions = beta_alphas / (beta_alphas + beta_betas)

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


# ---------------------------------------------------------------------------
# Egocentric Discrete Actor-Critic (single-agent Categorical)
# ---------------------------------------------------------------------------


class EgoDiscreteActorCritic(nn.Module):
    """Egocentric Actor-Critic with a joint Categorical over discrete actions.

    Takes a single agent's egocentric observation and produces logits
    over ``num_actions`` options plus a scalar value.  The same network
    is applied to every agent (weight sharing → symmetric strategies).

    ``__call__`` returns ``(logits, value)`` both in float32.
    """

    num_actions: int
    hidden_dims: Sequence[int] = (256, 256)
    dtype: Dtype = jnp.bfloat16
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self, obs: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            obs: ``(..., obs_dim)`` — a single agent's observation.

        Returns:
            ``(logits, value)`` with shapes:
            - logits: ``(..., num_actions)``
            - value:  ``(...,)``
        """
        x = obs.astype(self.dtype)

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

        logits = nn.Dense(
            self.num_actions,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=orthogonal(0.01),
            name="logits",
        )(features)

        value = nn.Dense(
            1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=orthogonal(1.0),
            name="critic",
        )(features).squeeze(-1)

        return (
            logits.astype(jnp.float32),
            value.astype(jnp.float32),
        )


def _categorical_log_prob(
    logits: jnp.ndarray,
    indices: jnp.ndarray,
) -> jnp.ndarray:
    """Log-prob of chosen indices under a Categorical. ``(...,)``."""
    log_probs_all = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(
        log_probs_all, indices[..., None], axis=-1,
    ).squeeze(-1)


def _categorical_entropy(logits: jnp.ndarray) -> jnp.ndarray:
    """Entropy of a Categorical. ``(...,)``."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jax.nn.softmax(logits, axis=-1)
    return -(probs * log_probs).sum(axis=-1)


# ---------------------------------------------------------------------------
# Factored Discrete Actor-Critic (separate Location + Price categoricals)
# ---------------------------------------------------------------------------


class EgoFactoredDiscreteActorCritic(nn.Module):
    """Egocentric Actor-Critic with *factored* location + price (+ quality)
    categoricals.

    Instead of a single Categorical over the joint action space, this
    network produces **independent** sets of logits for each factor:

    - ``loc_logits``     ``(..., num_location_bins)``
    - ``price_logits``   ``(..., num_price_bins)``
    - ``quality_logits`` ``(..., num_quality_bins)``  *(only when
      ``num_quality_bins > 0``)*

    Each dimension gets its own entropy bonus during PPO training, which
    prevents the common failure mode where the policy concentrates on
    centre locations while only exploring prices.

    The downstream environment interface is unchanged — the policy adapter
    recombines them into a joint index for ``map_discrete_actions``.
    """

    num_location_bins: int
    num_price_bins: int
    num_quality_bins: int = 0
    hidden_dims: Sequence[int] = (256, 256)
    dtype: Dtype = jnp.bfloat16
    param_dtype: Dtype = jnp.float32

    @property
    def num_actions(self) -> int:
        """Total joint action count (for compatibility)."""
        n = self.num_location_bins * self.num_price_bins
        if self.num_quality_bins > 0:
            n *= self.num_quality_bins
        return n

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        """Forward pass.

        Args:
            obs: ``(..., obs_dim)`` — a single agent's observation.

        Returns:
            ``(loc_logits, price_logits, value)`` when quality is
            disabled, or ``(loc_logits, price_logits, quality_logits,
            value)`` when ``num_quality_bins > 0``.
        """
        x = obs.astype(self.dtype)

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

        loc_logits = nn.Dense(
            self.num_location_bins,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=orthogonal(0.01),
            name="loc_logits",
        )(features)

        price_logits = nn.Dense(
            self.num_price_bins,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=orthogonal(0.01),
            name="price_logits",
        )(features)

        value = nn.Dense(
            1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=orthogonal(1.0),
            name="critic",
        )(features).squeeze(-1)

        if self.num_quality_bins > 0:
            quality_logits = nn.Dense(
                self.num_quality_bins,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(0.01),
                name="quality_logits",
            )(features)
            return (
                loc_logits.astype(jnp.float32),
                price_logits.astype(jnp.float32),
                quality_logits.astype(jnp.float32),
                value.astype(jnp.float32),
            )

        return (
            loc_logits.astype(jnp.float32),
            price_logits.astype(jnp.float32),
            value.astype(jnp.float32),
        )


def ego_discrete_sample(
    network: EgoDiscreteActorCritic,
    params: dict,
    obs: jnp.ndarray,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample discrete actions for per-agent egocentric observations.

    Args:
        obs: ``(..., obs_dim)``

    Returns:
        ``(actions, log_probs, values)`` with shapes
        ``(..., 1)``, ``(...,)``, ``(...,)``.
        Actions are float32 bin indices.
    """
    logits, values_symlog = network.apply(params, obs)  # type: ignore[misc]
    values = symexp(values_symlog)

    flat_logits = logits.reshape(-1, network.num_actions)
    flat_indices = jax.random.categorical(key, flat_logits)
    indices = flat_indices.reshape(logits.shape[:-1])

    log_probs = _categorical_log_prob(logits, indices)
    actions = indices[..., None].astype(jnp.float32)

    return actions, log_probs, values


def ego_discrete_deterministic(
    network: EgoDiscreteActorCritic,
    params: dict,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Deterministic discrete actions (argmax).

    Returns:
        ``(actions, values)`` with shapes ``(..., 1)``, ``(...,)``.
    """
    logits, values_symlog = network.apply(params, obs)  # type: ignore[misc]
    values = symexp(values_symlog)

    indices = jnp.argmax(logits, axis=-1)
    actions = indices[..., None].astype(jnp.float32)

    return actions, values


# ---------------------------------------------------------------------------
# Factored discrete sampling helpers
# ---------------------------------------------------------------------------


def ego_factored_discrete_sample(
    network: EgoFactoredDiscreteActorCritic,
    params: dict,
    obs: jnp.ndarray,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample factored discrete actions (location + price [+ quality]).

    Returns:
        ``(actions, log_probs, values)`` with shapes
        ``(..., 1)``, ``(...,)``, ``(...,)``.
    """
    has_quality = network.num_quality_bins > 0
    result = network.apply(params, obs)  # type: ignore[misc]

    if has_quality:
        loc_logits, price_logits, quality_logits, v_sym = result
    else:
        loc_logits, price_logits, v_sym = result
    values = symexp(v_sym)

    n_p = network.num_price_bins
    n_q = max(network.num_quality_bins, 1)

    if has_quality:
        k_l, k_p, k_q = jax.random.split(key, 3)
    else:
        k_l, k_p = jax.random.split(key)

    flat_loc = loc_logits.reshape(-1, network.num_location_bins)
    loc_idx = jax.random.categorical(k_l, flat_loc).reshape(loc_logits.shape[:-1])

    flat_price = price_logits.reshape(-1, n_p)
    price_idx = jax.random.categorical(k_p, flat_price).reshape(price_logits.shape[:-1])

    log_probs = (
        _categorical_log_prob(loc_logits, loc_idx)
        + _categorical_log_prob(price_logits, price_idx)
    )

    if has_quality:
        flat_qual = quality_logits.reshape(-1, network.num_quality_bins)
        qual_idx = jax.random.categorical(k_q, flat_qual).reshape(quality_logits.shape[:-1])
        joint_idx = loc_idx * (n_p * n_q) + price_idx * n_q + qual_idx
        log_probs = log_probs + _categorical_log_prob(quality_logits, qual_idx)
    else:
        joint_idx = loc_idx * n_p + price_idx

    return joint_idx[..., None].astype(jnp.float32), log_probs, values


def ego_factored_discrete_deterministic(
    network: EgoFactoredDiscreteActorCritic,
    params: dict,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Deterministic factored discrete actions (argmax per dimension).

    Returns:
        ``(actions, values)`` with shapes ``(..., 1)``, ``(...,)``.
    """
    has_quality = network.num_quality_bins > 0
    result = network.apply(params, obs)  # type: ignore[misc]

    if has_quality:
        loc_logits, price_logits, quality_logits, v_sym = result
    else:
        loc_logits, price_logits, v_sym = result
    values = symexp(v_sym)

    n_p = network.num_price_bins
    n_q = max(network.num_quality_bins, 1)

    loc_idx = jnp.argmax(loc_logits, axis=-1)
    price_idx = jnp.argmax(price_logits, axis=-1)

    if has_quality:
        qual_idx = jnp.argmax(quality_logits, axis=-1)
        joint_idx = loc_idx * (n_p * n_q) + price_idx * n_q + qual_idx
    else:
        joint_idx = loc_idx * n_p + price_idx

    return joint_idx[..., None].astype(jnp.float32), values


# ---------------------------------------------------------------------------
# Conv2D Egocentric Actor-Critic (2-D spatial, continuous actions)
# ---------------------------------------------------------------------------


class EgoConv2dActorCritic(nn.Module):
    """Conv2D-based Actor-Critic for 2-D spatial observations.

    Designed for the ``obs_type="conv_bin"`` observation layout with
    ``dimensions=2``::

        obs = [ grid_channels (C × R²) , scalar_features (S) ]

    Grid channels (4 × R² by default, Gaussian-smoothed blobs):
        0 – self position blob
        1 – other sellers blob
        2 – buyer density blob (normalised)
        3 – seller avg-price blob

    Scalar features (appended after global average pooling):
        own_position / R  (2 floats)
        own_price / max   (1 float)
        (own_quality / max if applicable)

    Architecture::

        Grid (R, R, C) → Conv2D(32,5,s2) → Conv2D(64,3,s2)
                        → Conv2D(128,3,s2) → GlobalAvgPool
                                                  ↓
                                          Concat(scalars) → MLP → Heads

    When ``independent_heads=True``, per-agent actor heads (Gaussian
    movement + Beta price) are created and selected via the one-hot
    agent ID.  The Conv + MLP backbone and critic remain shared.

    Output interface matches :class:`EgoActorCritic`::

        (gauss_means, gauss_log_stds, beta_alphas, beta_betas, value)

    so the existing :class:`EgoContinuousPolicy` adapter works unchanged.
    """

    movement_dim: int = 2
    bounded_dim: int = 1
    spatial_resolution: int = 51  # R = space_resolution + 1 (grid points)
    num_grid_channels: int = 4
    num_scalar_features: int = 3  # D + 1 (pos_x, pos_y, price)
    conv_features: Sequence[int] = (32, 64, 128)
    conv_kernel_sizes: Sequence[int] = (5, 3, 3)
    conv_strides: Sequence[int] = (2, 2, 2)
    mlp_hidden_dims: Sequence[int] = (256, 256)
    independent_heads: bool = False
    num_agents: int = 2  # only used when independent_heads=True
    dtype: Dtype = jnp.bfloat16
    param_dtype: Dtype = jnp.float32
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @property
    def action_dim(self) -> int:
        return self.movement_dim + self.bounded_dim

    @nn.compact
    def __call__(
        self, obs: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            obs: ``(..., obs_dim)`` where
                 ``obs_dim = num_grid_channels * spatial_resolution²
                             + num_scalar_features
                             (+ num_agents if independent)``.

        Returns:
            ``(gauss_means, gauss_log_stds, beta_alphas, beta_betas, value)``
            always float32, with shapes:
            - gauss_means:    ``(..., movement_dim)``
            - gauss_log_stds: ``(..., movement_dim)``
            - beta_alphas:    ``(..., bounded_dim)``
            - beta_betas:     ``(..., bounded_dim)``
            - value:          ``(...,)``
        """
        R = self.spatial_resolution
        C = self.num_grid_channels
        grid_size = C * R * R

        # ── Split grid and scalar parts ──────────────────────────────
        grid_flat = obs[..., :grid_size]
        scalars = obs[..., grid_size:]  # (..., S) or (..., S + A)

        # ── Reshape grid: (..., C*R*R) → (..., R, R, C) ─────────────
        batch_shape = grid_flat.shape[:-1]
        grid = grid_flat.reshape(*batch_shape, C, R, R)
        # (…, C, R, R) → (…, R, R, C)  (channels-last for nn.Conv)
        grid = jnp.moveaxis(grid, -3, -1)

        x = grid.astype(self.dtype)

        # ── Conv2D trunk (strided) ───────────────────────────────────
        for i, (feats, ks, stride) in enumerate(
            zip(self.conv_features, self.conv_kernel_sizes, self.conv_strides),
        ):
            x = nn.Conv(
                features=feats,
                kernel_size=(ks, ks),
                strides=(stride, stride),
                padding="SAME",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(jnp.sqrt(2)),
                name=f"conv_{i}",
            )(x)
            x = nn.relu(x)

        # ── Global Average Pooling ───────────────────────────────────
        # (..., H, W, F) → (..., F)
        x = x.mean(axis=(-3, -2))

        # ── Concat with scalar features ──────────────────────────────
        scalars_cast = scalars.astype(self.dtype)
        x = jnp.concatenate([x, scalars_cast], axis=-1)

        # ── MLP trunk ────────────────────────────────────────────────
        for i, dim in enumerate(self.mlp_hidden_dims):
            x = nn.Dense(
                dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(jnp.sqrt(2)),
                name=f"mlp_{i}",
            )(x)
            x = nn.relu(x)

        features = x

        if self.independent_heads:
            A = self.num_agents
            agent_id = obs[..., -A:].astype(self.dtype)  # (..., A)

            # ── Per-agent Gaussian heads (movement) ──────────────────
            stacked_mean = jnp.stack([
                nn.Dense(
                    self.movement_dim,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    kernel_init=orthogonal(0.01),
                    name=f"gauss_mean_{a}",
                )(features)
                for a in range(A)
            ], axis=-2)  # (..., A, movement_dim)
            gauss_mean = jnp.einsum("...a,...ab->...b", agent_id, stacked_mean)

            stacked_log_std = jnp.stack([
                nn.Dense(
                    self.movement_dim,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    kernel_init=orthogonal(0.01),
                    name=f"gauss_log_std_{a}",
                )(features)
                for a in range(A)
            ], axis=-2)
            gauss_log_std = jnp.einsum("...a,...ab->...b", agent_id, stacked_log_std)
            gauss_log_std = jnp.clip(gauss_log_std, self.log_std_min, self.log_std_max)

            # ── Per-agent Beta heads (price) ─────────────────────────
            stacked_alpha = jnp.stack([
                nn.Dense(
                    self.bounded_dim,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    kernel_init=orthogonal(0.01),
                    name=f"beta_alpha_{a}",
                )(features)
                for a in range(A)
            ], axis=-2)
            beta_alpha_raw = jnp.einsum("...a,...ab->...b", agent_id, stacked_alpha)

            stacked_beta = jnp.stack([
                nn.Dense(
                    self.bounded_dim,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    kernel_init=orthogonal(0.01),
                    name=f"beta_beta_{a}",
                )(features)
                for a in range(A)
            ], axis=-2)
            beta_beta_raw = jnp.einsum("...a,...ab->...b", agent_id, stacked_beta)

            beta_alpha_raw = jnp.clip(beta_alpha_raw, -20.0, 20.0)
            beta_beta_raw = jnp.clip(beta_beta_raw, -20.0, 20.0)
            beta_alpha = nn.softplus(beta_alpha_raw) + 1.0
            beta_beta = nn.softplus(beta_beta_raw) + 1.0
        else:
            # ── Shared Gaussian head (movement) ──────────────────────
            gauss_mean = nn.Dense(
                self.movement_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(0.01),
                name="gauss_mean",
            )(features)
            gauss_log_std = nn.Dense(
                self.movement_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(0.01),
                name="gauss_log_std",
            )(features)
            gauss_log_std = jnp.clip(gauss_log_std, self.log_std_min, self.log_std_max)

            # ── Shared Beta head (price) ─────────────────────────────
            beta_alpha_raw = nn.Dense(
                self.bounded_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(0.01),
                name="beta_alpha",
            )(features)
            beta_beta_raw = nn.Dense(
                self.bounded_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(0.01),
                name="beta_beta",
            )(features)
            beta_alpha_raw = jnp.clip(beta_alpha_raw, -20.0, 20.0)
            beta_beta_raw = jnp.clip(beta_beta_raw, -20.0, 20.0)
            beta_alpha = nn.softplus(beta_alpha_raw) + 1.0
            beta_beta = nn.softplus(beta_beta_raw) + 1.0

        # ── Critic head (always shared) ──────────────────────────────
        value = nn.Dense(
            1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=orthogonal(1.0),
            name="critic",
        )(features).squeeze(-1)

        return (
            gauss_mean.astype(jnp.float32),
            gauss_log_std.astype(jnp.float32),
            beta_alpha.astype(jnp.float32),
            beta_beta.astype(jnp.float32),
            value.astype(jnp.float32),
        )


# ---------------------------------------------------------------------------
# Conv1D Factored Discrete Actor-Critic (1-D spatial)
# ---------------------------------------------------------------------------


class EgoConv1dFactoredDiscreteActorCritic(nn.Module):
    """Conv1D-based Actor-Critic for 1-D spatial observations.

    Designed for the ``obs_type="conv_bin"`` observation layout::

        obs = [ grid_channels (C × R) , scalar_features (S) ]

    Grid channels (4 × R by default):
        0 – self position (binary one-hot)
        1 – other sellers (count per bin)
        2 – buyer presence (binary per bin)
        3 – seller avg-price blob (Gaussian-smoothed)

    Scalar features (appended after flatten):
        own_position / R  (D floats)
        own_price / max   (1 float)
        (own_quality / max if applicable)

    Architecture::

        Grid (R, C) → Conv1D stack → Flatten
                                       ↓
                               Concat(scalars) → MLP → Factored heads

    When ``independent_heads=True``, per-agent actor heads are created
    and selected via the one-hot agent ID appended to the observation.
    The Conv + MLP backbone and critic head remain shared.

    Output interface is identical to
    :class:`EgoFactoredDiscreteActorCritic`: ``(loc_logits, price_logits,
    value)`` all in float32, so the same :class:`EgoFactoredDiscretePolicy`
    adapter works unchanged.
    """

    num_location_bins: int
    num_price_bins: int
    spatial_resolution: int
    num_quality_bins: int = 0
    num_grid_channels: int = 4
    num_scalar_features: int = 2  # D + 1 (position + price); set at construction
    conv_features: Sequence[int] = (16,)
    conv_kernel_size: int = 3
    mlp_hidden_dims: Sequence[int] = (128, 128)
    independent_heads: bool = False
    num_agents: int = 2  # only used when independent_heads=True
    dtype: Dtype = jnp.bfloat16
    param_dtype: Dtype = jnp.float32

    @property
    def num_actions(self) -> int:
        """Total joint action count (for compatibility)."""
        n = self.num_location_bins * self.num_price_bins
        if self.num_quality_bins > 0:
            n *= self.num_quality_bins
        return n

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        """Forward pass.

        Args:
            obs: ``(..., obs_dim)`` where
                 ``obs_dim = num_grid_channels * spatial_resolution
                             + num_scalar_features
                             (+ num_agents if independent)``.

        Returns:
            ``(loc_logits, price_logits, value)`` or
            ``(loc_logits, price_logits, quality_logits, value)``
            when ``num_quality_bins > 0``.
        """
        R = self.spatial_resolution
        C = self.num_grid_channels
        grid_size = C * R

        # ── Split grid and scalar parts ──────────────────────────────
        grid_flat = obs[..., :grid_size]
        scalars = obs[..., grid_size:]  # (..., S) or (..., S + A)

        # ── Reshape grid: (..., C*R) → (..., R, C) ──────────────────
        batch_shape = grid_flat.shape[:-1]
        grid = grid_flat.reshape(*batch_shape, C, R)
        grid = jnp.swapaxes(grid, -2, -1)  # (..., R, C)

        x = grid.astype(self.dtype)

        # ── Conv1D trunk ─────────────────────────────────────────────
        for i, feats in enumerate(self.conv_features):
            x = nn.Conv(
                features=feats,
                kernel_size=(self.conv_kernel_size,),
                padding="SAME",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(jnp.sqrt(2)),
                name=f"conv_{i}",
            )(x)
            x = nn.relu(x)

        # Flatten spatial: (..., R, feats) → (..., R * feats)
        x = x.reshape(*batch_shape, -1)

        # ── Concat with scalar features ──────────────────────────────
        scalars_cast = scalars.astype(self.dtype)
        x = jnp.concatenate([x, scalars_cast], axis=-1)

        # ── MLP trunk ────────────────────────────────────────────────
        for i, dim in enumerate(self.mlp_hidden_dims):
            x = nn.Dense(
                dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=orthogonal(jnp.sqrt(2)),
                name=f"mlp_{i}",
            )(x)
            x = nn.relu(x)

        features = x

        # ── Factored action heads ────────────────────────────────────
        def _make_head(name: str, dim: int) -> jnp.ndarray:
            return nn.Dense(
                dim, dtype=self.dtype, param_dtype=self.param_dtype,
                kernel_init=orthogonal(0.01), name=name,
            )(features)

        if self.independent_heads:
            A = self.num_agents
            agent_id = obs[..., -A:].astype(self.dtype)  # (..., A)

            def _per_agent_head(base: str, dim: int) -> jnp.ndarray:
                stacked = jnp.stack([
                    nn.Dense(
                        dim, dtype=self.dtype, param_dtype=self.param_dtype,
                        kernel_init=orthogonal(0.01), name=f"{base}_{a}",
                    )(features)
                    for a in range(A)
                ], axis=-2)
                return jnp.einsum("...a,...ab->...b", agent_id, stacked)

            loc_logits = _per_agent_head("loc_logits", self.num_location_bins)
            price_logits = _per_agent_head("price_logits", self.num_price_bins)
            quality_logits = (
                _per_agent_head("quality_logits", self.num_quality_bins)
                if self.num_quality_bins > 0 else None
            )
        else:
            loc_logits = _make_head("loc_logits", self.num_location_bins)
            price_logits = _make_head("price_logits", self.num_price_bins)
            quality_logits = (
                _make_head("quality_logits", self.num_quality_bins)
                if self.num_quality_bins > 0 else None
            )

        # ── Critic head (shared) ─────────────────────────────────────
        value = nn.Dense(
            1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=orthogonal(1.0),
            name="critic",
        )(features).squeeze(-1)

        if quality_logits is not None:
            return (
                loc_logits.astype(jnp.float32),
                price_logits.astype(jnp.float32),
                quality_logits.astype(jnp.float32),
                value.astype(jnp.float32),
            )

        return (
            loc_logits.astype(jnp.float32),
            price_logits.astype(jnp.float32),
            value.astype(jnp.float32),
        )


# ---------------------------------------------------------------------------
# Conv2D Factored Discrete Actor-Critic (2-D spatial, 3-way factored)
# ---------------------------------------------------------------------------


class EgoConv2dFactoredDiscreteActorCritic(nn.Module):
    """Conv2D + factored discrete heads: loc_x × loc_y × price (× quality).

    Same Conv2D backbone as :class:`EgoConv2dActorCritic`.
    Returns ``(loc_x_logits, loc_y_logits, price_logits, value)`` or
    ``(loc_x_logits, loc_y_logits, price_logits, quality_logits, value)``
    when ``num_quality_bins > 0``.
    """

    num_location_bins: int = 11
    num_price_bins: int = 11
    num_quality_bins: int = 0
    spatial_resolution: int = 51
    num_grid_channels: int = 4
    num_scalar_features: int = 3
    conv_features: Sequence[int] = (32, 64, 128)
    conv_kernel_sizes: Sequence[int] = (5, 3, 3)
    conv_strides: Sequence[int] = (2, 2, 2)
    mlp_hidden_dims: Sequence[int] = (256, 256)
    independent_heads: bool = False
    num_agents: int = 2
    dtype: Dtype = jnp.bfloat16
    param_dtype: Dtype = jnp.float32

    @property
    def num_actions(self) -> int:
        n = self.num_location_bins ** 2 * self.num_price_bins
        if self.num_quality_bins > 0:
            n *= self.num_quality_bins
        return n

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        R = self.spatial_resolution
        C = self.num_grid_channels
        grid_size = C * R * R

        grid_flat = obs[..., :grid_size]
        scalars = obs[..., grid_size:]
        batch_shape = grid_flat.shape[:-1]
        grid = grid_flat.reshape(*batch_shape, C, R, R)
        grid = jnp.moveaxis(grid, -3, -1)
        x = grid.astype(self.dtype)

        for i, (feats, ks, stride) in enumerate(
            zip(self.conv_features, self.conv_kernel_sizes, self.conv_strides),
        ):
            x = nn.Conv(
                features=feats, kernel_size=(ks, ks), strides=(stride, stride),
                padding="SAME", dtype=self.dtype, param_dtype=self.param_dtype,
                kernel_init=orthogonal(jnp.sqrt(2)), name=f"conv_{i}",
            )(x)
            x = nn.relu(x)

        x = x.mean(axis=(-3, -2))  # GlobalAvgPool
        x = jnp.concatenate([x, scalars.astype(self.dtype)], axis=-1)

        for i, dim in enumerate(self.mlp_hidden_dims):
            x = nn.Dense(
                dim, dtype=self.dtype, param_dtype=self.param_dtype,
                kernel_init=orthogonal(jnp.sqrt(2)), name=f"mlp_{i}",
            )(x)
            x = nn.relu(x)
        features = x

        def _head(name: str, dim: int) -> jnp.ndarray:
            return nn.Dense(
                dim, dtype=self.dtype, param_dtype=self.param_dtype,
                kernel_init=orthogonal(0.01), name=name,
            )(features)

        if self.independent_heads:
            A = self.num_agents
            aid = obs[..., -A:].astype(self.dtype)

            def _pa(base: str, dim: int) -> jnp.ndarray:
                st = jnp.stack([
                    nn.Dense(dim, dtype=self.dtype, param_dtype=self.param_dtype,
                             kernel_init=orthogonal(0.01), name=f"{base}_{a}")(features)
                    for a in range(A)
                ], axis=-2)
                return jnp.einsum("...a,...ab->...b", aid, st)

            lx = _pa("loc_x", self.num_location_bins)
            ly = _pa("loc_y", self.num_location_bins)
            pr = _pa("price", self.num_price_bins)
            ql = _pa("quality", self.num_quality_bins) if self.num_quality_bins > 0 else None
        else:
            lx = _head("loc_x", self.num_location_bins)
            ly = _head("loc_y", self.num_location_bins)
            pr = _head("price", self.num_price_bins)
            ql = _head("quality", self.num_quality_bins) if self.num_quality_bins > 0 else None

        val = nn.Dense(
            1, dtype=self.dtype, param_dtype=self.param_dtype,
            kernel_init=orthogonal(1.0), name="critic",
        )(features).squeeze(-1)

        if ql is not None:
            return (
                lx.astype(jnp.float32), ly.astype(jnp.float32),
                pr.astype(jnp.float32), ql.astype(jnp.float32),
                val.astype(jnp.float32),
            )

        return lx.astype(jnp.float32), ly.astype(jnp.float32), pr.astype(jnp.float32), val.astype(jnp.float32)


def ego_2d_factored_discrete_sample(
    network: EgoConv2dFactoredDiscreteActorCritic,
    params: dict, obs: jnp.ndarray, key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    has_quality = network.num_quality_bins > 0
    result = network.apply(params, obs)  # type: ignore[misc]

    if has_quality:
        lx_lg, ly_lg, p_lg, q_lg, v_sym = result
    else:
        lx_lg, ly_lg, p_lg, v_sym = result
    values = symexp(v_sym)

    n_l, n_p = network.num_location_bins, network.num_price_bins
    n_q = max(network.num_quality_bins, 1)

    if has_quality:
        k_x, k_y, k_p, k_q = jax.random.split(key, 4)
    else:
        k_x, k_y, k_p = jax.random.split(key, 3)

    lx = jax.random.categorical(k_x, lx_lg.reshape(-1, n_l)).reshape(lx_lg.shape[:-1])
    ly = jax.random.categorical(k_y, ly_lg.reshape(-1, n_l)).reshape(ly_lg.shape[:-1])
    pi = jax.random.categorical(k_p, p_lg.reshape(-1, n_p)).reshape(p_lg.shape[:-1])

    lp = (_categorical_log_prob(lx_lg, lx)
          + _categorical_log_prob(ly_lg, ly)
          + _categorical_log_prob(p_lg, pi))

    if has_quality:
        qi = jax.random.categorical(k_q, q_lg.reshape(-1, n_q)).reshape(q_lg.shape[:-1])
        joint = lx * (n_l * n_p * n_q) + ly * (n_p * n_q) + pi * n_q + qi
        lp = lp + _categorical_log_prob(q_lg, qi)
    else:
        joint = lx * (n_l * n_p) + ly * n_p + pi

    return joint[..., None].astype(jnp.float32), lp, values


def ego_2d_factored_discrete_deterministic(
    network: EgoConv2dFactoredDiscreteActorCritic,
    params: dict, obs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    has_quality = network.num_quality_bins > 0
    result = network.apply(params, obs)  # type: ignore[misc]

    if has_quality:
        lx_lg, ly_lg, p_lg, q_lg, v_sym = result
    else:
        lx_lg, ly_lg, p_lg, v_sym = result
    values = symexp(v_sym)

    n_l, n_p = network.num_location_bins, network.num_price_bins
    n_q = max(network.num_quality_bins, 1)

    lx, ly, pi = jnp.argmax(lx_lg, -1), jnp.argmax(ly_lg, -1), jnp.argmax(p_lg, -1)

    if has_quality:
        qi = jnp.argmax(q_lg, -1)
        joint = lx * (n_l * n_p * n_q) + ly * (n_p * n_q) + pi * n_q + qi
    else:
        joint = lx * (n_l * n_p) + ly * n_p + pi

    return joint[..., None].astype(jnp.float32), values
