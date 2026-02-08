"""Observation building for the JAX spatial competition environment.

All observations are built using flat-index scatter: positions are converted
to flat indices via ``sum(pos[i] * R^(D-1-i))``, then values are scattered
into a 1D array and reshaped to the N-dimensional grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from spatial_competition_jax.env import EnvState, SpatialCompetitionEnv


# ---------------------------------------------------------------------------
# Flat-index helpers
# ---------------------------------------------------------------------------


def _positions_to_flat(positions: jnp.ndarray, strides: jnp.ndarray) -> jnp.ndarray:
    """Convert (N, D) int32 positions to (N,) flat indices."""
    return jnp.sum(positions * strides[None, :], axis=-1)


def _position_to_flat(position: jnp.ndarray, strides: jnp.ndarray) -> jnp.ndarray:
    """Convert (D,) int32 position to a scalar flat index."""
    return jnp.sum(position * strides)


# ---------------------------------------------------------------------------
# Distance helper (buyer -> single seller)
# ---------------------------------------------------------------------------


def _compute_buyer_distances(
    env: SpatialCompetitionEnv,
    buyer_positions: jnp.ndarray,
    seller_pos: jnp.ndarray,
) -> jnp.ndarray:
    """Compute distances from every buyer to one seller.

    Args:
        env: Environment instance (for topology / norm config).
        buyer_positions: (B, D) int32.
        seller_pos: (D,) int32.

    Returns:
        (B,) float32 distances in space coordinates.
    """
    buyer_space = buyer_positions.astype(jnp.float32) / env.space_resolution
    seller_space = seller_pos.astype(jnp.float32) / env.space_resolution

    diff = buyer_space - seller_space[None, :]  # (B, D)

    if env.topology == 1:  # TORUS
        abs_diff = jnp.abs(diff)
        abs_diff = jnp.minimum(abs_diff, 1.0 - abs_diff)
    else:  # RECTANGLE
        abs_diff = jnp.abs(diff)

    p = env.transportation_cost_norm
    if p == float("inf"):
        return jnp.max(abs_diff, axis=-1)
    elif p == 1.0:
        return jnp.sum(abs_diff, axis=-1)
    elif p == 2.0:
        return jnp.sqrt(jnp.sum(abs_diff**2, axis=-1))
    else:
        return jnp.sum(abs_diff**p, axis=-1) ** (1.0 / p)


# ---------------------------------------------------------------------------
# Per-agent grid builders
# ---------------------------------------------------------------------------


def _build_local_view(
    env: SpatialCompetitionEnv,
    state: EnvState,
    agent_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Build 3-channel local view for one agent.

    Channels:
        0 – self position (binary)
        1 – other seller count
        2 – valid buyers (binary)

    Returns:
        (3, R, …, R) uint8 grid.
    """
    strides = jnp.array(env.strides, dtype=jnp.int32)
    tc = env.total_cells

    # Channel 0: self
    self_flat = _position_to_flat(state.seller_positions[agent_idx], strides)
    ch0 = jnp.zeros(tc, dtype=jnp.float32).at[self_flat].set(1.0)

    # Channel 1: other sellers
    all_flat = _positions_to_flat(state.seller_positions, strides)  # (S,)
    other_mask = (jnp.arange(env.num_sellers) != agent_idx).astype(jnp.float32)
    ch1 = jnp.zeros(tc, dtype=jnp.float32).at[all_flat].add(other_mask)

    # Channel 2: valid buyers
    buyer_flat = _positions_to_flat(state.buyer_positions, strides)  # (B,)
    ch2 = jnp.zeros(tc, dtype=jnp.float32).at[buyer_flat].add(state.buyer_valid.astype(jnp.float32))
    ch2 = jnp.clip(ch2, 0.0, 1.0)

    grid = jnp.stack([ch0, ch1, ch2], axis=0)  # (3, tc)
    return grid.reshape((3,) + env.grid_shape).astype(jnp.uint8)


def _build_buyers_grid(
    env: SpatialCompetitionEnv,
    state: EnvState,
    agent_idx: jnp.ndarray,  # noqa: ARG001 – kept for vmap API consistency
) -> jnp.ndarray:
    """Build 3-channel buyer attribute grid.

    Channels (averaged when multiple buyers overlap):
        0 – inner valuation  (``buyer_values``)
        1 – distance factor  (``buyer_distance_factors``)
        2 – quality factor   (``buyer_quality_tastes``)

    Empty cells are 0.  Buyer presence is already in ``local_view`` ch-2.

    Returns:
        (3, R, …, R) float32 grid.
    """
    strides = jnp.array(env.strides, dtype=jnp.int32)
    tc = env.total_cells
    buyer_flat = _positions_to_flat(state.buyer_positions, strides)  # (B,)
    valid = state.buyer_valid.astype(jnp.float32)  # (B,)

    # Scatter-add counts for averaging
    counts = jnp.zeros(tc, dtype=jnp.float32).at[buyer_flat].add(valid)

    def _avg_channel(values: jnp.ndarray) -> jnp.ndarray:
        masked = jnp.where(state.buyer_valid, values, 0.0)
        summed = jnp.zeros(tc, dtype=jnp.float32).at[buyer_flat].add(masked)
        return jnp.where(counts > 0, summed / counts, 0.0)

    ch0 = _avg_channel(state.buyer_values)
    ch1 = _avg_channel(state.buyer_distance_factors)
    ch2 = _avg_channel(state.buyer_quality_tastes)

    grid = jnp.stack([ch0, ch1, ch2], axis=0)  # (3, tc)
    return grid.reshape((3,) + env.grid_shape)


def _build_sellers_price_grid(
    env: SpatialCompetitionEnv,
    state: EnvState,
    agent_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Build other-sellers' price grid for one agent.

    Each cell shows the **minimum** price among other sellers at that
    position (i.e. the most competitive rival), or 0 if no other
    seller is present.

    Returns:
        (R, …, R) float32 grid.
    """
    strides = jnp.array(env.strides, dtype=jnp.int32)
    seller_flat = _positions_to_flat(state.seller_positions, strides)

    other_mask = jnp.arange(env.num_sellers) != agent_idx
    vals = jnp.where(other_mask, state.seller_prices, jnp.inf)

    grid = jnp.full(env.total_cells, jnp.inf, dtype=jnp.float32)
    grid = grid.at[seller_flat].min(vals)

    # Replace +inf (no seller present) with 0
    grid = jnp.where(grid == jnp.inf, 0.0, grid)

    return grid.reshape(env.grid_shape)


def _build_sellers_quality_grid(
    env: SpatialCompetitionEnv,
    state: EnvState,
    agent_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Build other-sellers' quality grid for one agent.

    Returns:
        (R, …, R) float32 grid.
    """
    strides = jnp.array(env.strides, dtype=jnp.int32)
    seller_flat = _positions_to_flat(state.seller_positions, strides)

    other_mask = jnp.arange(env.num_sellers) != agent_idx
    vals = jnp.where(other_mask, state.seller_qualities, 0.0)

    grid = jnp.zeros(env.total_cells, dtype=jnp.float32)
    grid = grid.at[seller_flat].max(vals)

    return grid.reshape(env.grid_shape)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_observations(
    env: SpatialCompetitionEnv,
    state: EnvState,
) -> dict[str, jnp.ndarray]:
    """Build observations for **all** agents.

    Returns a dict of arrays where each value has leading dimension
    ``num_sellers``.
    """
    obs: dict[str, jnp.ndarray] = {}

    # Own state (no vmap needed – already stacked)
    obs["own_position"] = state.seller_positions.astype(jnp.float32) / env.space_resolution
    obs["own_price"] = state.seller_prices

    if env.include_quality:
        obs["own_quality"] = state.seller_qualities

    # Grid-based observations (vmap over agent index)
    def _per_agent_grids(agent_idx: jnp.ndarray) -> dict[str, jnp.ndarray]:
        result: dict[str, jnp.ndarray] = {
            "local_view": _build_local_view(env, state, agent_idx),
        }
        if env.information_level >= 1:  # LIMITED or COMPLETE
            result["buyers"] = _build_buyers_grid(env, state, agent_idx)
        if env.information_level >= 2:  # COMPLETE
            result["sellers_price"] = _build_sellers_price_grid(env, state, agent_idx)
            result["sellers_quality"] = _build_sellers_quality_grid(env, state, agent_idx)
        return result

    grid_obs = jax.vmap(_per_agent_grids)(jnp.arange(env.num_sellers))
    obs.update(grid_obs)

    return obs
