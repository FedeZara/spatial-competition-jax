"""Rollout-buffer utilities for MAPPO with GAE computation.

All operations use pure JAX arrays.  GAE is computed via
``jax.lax.scan`` (reverse).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class Transition(NamedTuple):
    """One step of rollout data (batched over environments).

    Leading dimension conventions:
        - Stored per-step: ``(E, ...)``
        - After ``jax.lax.scan``: ``(T, E, ...)``
    """

    states: jnp.ndarray  # (E, state_dim)
    actions: jnp.ndarray  # (E, A, action_dim)
    log_probs: jnp.ndarray  # (E, A)
    values: jnp.ndarray  # (E, A)
    rewards: jnp.ndarray  # (E, A)
    dones: jnp.ndarray  # (E, A)


class RolloutBatch(NamedTuple):
    """Minibatch of rollout data for PPO training.

    ``B`` = minibatch size, ``A`` = num agents.
    """

    states: jnp.ndarray  # (B, state_dim)
    actions: jnp.ndarray  # (B, A, action_dim)
    log_probs: jnp.ndarray  # (B, A)
    values: jnp.ndarray  # (B, A)
    advantages: jnp.ndarray  # (B, A)
    returns: jnp.ndarray  # (B, A)


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generalised Advantage Estimation via reverse ``jax.lax.scan``.

    Args:
        rewards: ``(T, E, A)``
        values: ``(T, E, A)``
        dones: ``(T, E, A)``
        last_value: ``(E, A)`` bootstrap value at step *T*.
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        ``(advantages, returns)`` each ``(T, E, A)``.
    """

    def _gae_step(carry: Any, transition: Any) -> Any:
        gae, next_value = carry
        reward, value, done = transition
        non_terminal = 1.0 - done
        delta = reward + gamma * next_value * non_terminal - value
        gae = delta + gamma * gae_lambda * non_terminal * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _gae_step,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, dones),
        reverse=True,
    )

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Advantage normalisation
# ---------------------------------------------------------------------------


def normalize_advantages(
    advantages: jnp.ndarray,
    per_agent: bool = True,
) -> jnp.ndarray:
    """Zero-mean, unit-variance normalisation of advantages.

    Args:
        advantages: ``(T, E, A)``
        per_agent: Normalise independently per agent dimension.

    Returns:
        Normalised advantages ``(T, E, A)``.
    """
    if per_agent:
        mean = advantages.mean(axis=(0, 1), keepdims=True)
        std = advantages.std(axis=(0, 1), keepdims=True) + 1e-8
    else:
        mean = advantages.mean()
        std = advantages.std() + 1e-8

    return (advantages - mean) / std


# ---------------------------------------------------------------------------
# Minibatch creation
# ---------------------------------------------------------------------------


def make_minibatches(
    key: jnp.ndarray,
    transitions: Transition,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    num_minibatches: int,
) -> RolloutBatch:
    """Shuffle and split rollout data into minibatches.

    Flattens ``(T, E)`` into a single sample dimension, keeping the
    agent axis intact.

    Args:
        key: PRNG key for shuffling.
        transitions: Stacked transitions ``(T, E, ...)``.
        advantages: ``(T, E, A)``
        returns: ``(T, E, A)``
        num_minibatches: Number of minibatches.

    Returns:
        A single :class:`RolloutBatch` whose arrays have leading shape
        ``(num_minibatches, batch_size, ...)``, ready for
        ``jax.lax.scan``.
    """
    T, E = transitions.rewards.shape[:2]
    num_samples = T * E
    batch_size = num_samples // num_minibatches

    # Flatten (T, E) -> (T*E)
    flat_states = transitions.states.reshape(num_samples, -1)
    flat_actions = transitions.actions.reshape(num_samples, *transitions.actions.shape[2:])
    flat_log_probs = transitions.log_probs.reshape(num_samples, -1)
    flat_values = transitions.values.reshape(num_samples, -1)
    flat_advantages = advantages.reshape(num_samples, -1)
    flat_returns = returns.reshape(num_samples, -1)

    # Random permutation
    perm = jax.random.permutation(key, num_samples)
    perm = perm[: batch_size * num_minibatches].reshape(num_minibatches, batch_size)

    return RolloutBatch(
        states=flat_states[perm],
        actions=flat_actions[perm],
        log_probs=flat_log_probs[perm],
        values=flat_values[perm],
        advantages=flat_advantages[perm],
        returns=flat_returns[perm],
    )
