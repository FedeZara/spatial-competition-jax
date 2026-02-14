"""State-permutation utilities for symmetric PSRO.

In a symmetric game the two agents are interchangeable.  Each PSRO
policy always plays "as agent 0".  When a policy needs to act as
agent 1, we permute the observation so the policy sees itself
in the agent-0 slot.

**Egocentric mode** (``observation_mode="egocentric"``):
    Permutation is unnecessary — the wrapper already builds each
    agent's observation with its own features first.  Agent 1's
    egocentric obs already looks like "I am agent 0" from its
    perspective.

**Global mode** (``observation_mode="global"``):
    The global-state layout (from :class:`TrainingWrapper`) is::

        [ blob_channels (permutation-invariant) | agent_0_feats | agent_1_feats | … ]

    The blob channels aggregate across all sellers and are therefore
    symmetric.  Only the per-agent scalar features at the tail need
    swapping.
"""

from __future__ import annotations

import jax.numpy as jnp

from spatial_competition_jax.marl.training_wrapper import TrainingWrapper


def permute_agent_state(
    state: jnp.ndarray,
    wrapper: TrainingWrapper,
) -> jnp.ndarray:
    """Swap agent 0 and agent 1 features in the global-state vector.

    Works for arbitrary leading batch dimensions (``(..., state_dim)``).

    Args:
        state: Global-state array of shape ``(..., state_dim)``.
        wrapper: The :class:`TrainingWrapper` used to build the state
            (provides ``_blob_dim``, ``_per_agent_dim``, ``num_agents``).

    Returns:
        Permuted state with the same shape, where agent 0 and agent 1
        per-agent features are swapped.
    """
    blob_dim = wrapper._blob_dim
    per_agent_dim = wrapper._per_agent_dim
    num_agents = wrapper.num_agents

    # Blob channels are permutation-invariant — keep as-is.
    blob = state[..., :blob_dim]

    # Per-agent feature block: shape (..., num_agents * per_agent_dim)
    per_agent_flat = state[..., blob_dim:]

    # Reshape to (..., num_agents, per_agent_dim), reverse agent order,
    # then flatten back.
    original_shape = per_agent_flat.shape
    per_agent = per_agent_flat.reshape(
        *original_shape[:-1], num_agents, per_agent_dim
    )

    # Reverse the agent axis (works for any num_agents, but for 2 agents
    # this is simply a swap of indices 0 and 1).
    per_agent_swapped = jnp.flip(per_agent, axis=-2)

    per_agent_flat_swapped = per_agent_swapped.reshape(original_shape)

    return jnp.concatenate([blob, per_agent_flat_swapped], axis=-1)
