"""Training wrapper for SpatialCompetitionEnv.

Provides flat observations, action mapping, and vectorised operations
via ``jax.vmap`` for MAPPO training.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spatial_competition_jax.env import (
    EnvState,
    SpatialCompetitionEnv,
    make_constant_sampler,
)

# ---------------------------------------------------------------------------
# Gaussian blob helpers
# ---------------------------------------------------------------------------


def _gaussian_blob_channel(
    positions: jnp.ndarray,
    values: jnp.ndarray,
    map_size: int,
    sigma: float,
    dimensions: int,
) -> jnp.ndarray:
    """Render a Gaussian-blob spatial map from agent positions.

    Args:
        positions: ``(N, D)`` float in ``[0, 1]`` (normalised coords).
        values: ``(N,)`` per-agent scalar weights.
        map_size: Resolution of the output grid per dimension.
        sigma: Blob width in **grid-cell** units.
        dimensions: Spatial dimensionality (1 or 2).

    Returns:
        ``(map_size,)`` for 1-D or ``(map_size, map_size)`` for 2-D.
    """
    # Convert sigma from grid-cell units to [0, 1] space
    sigma_norm = sigma / map_size

    coords = jnp.linspace(0.0, 1.0, map_size)  # (M,)

    if dimensions == 1:
        # positions[:, 0] → (N,)  ;  coords → (M,)
        # dist → (N, M)
        dist_sq = (positions[:, 0:1] - coords[None, :]) ** 2
        gauss = jnp.exp(-dist_sq / (2.0 * sigma_norm**2))  # (N, M)
        weighted = gauss * values[:, None]  # (N, M)
        return weighted.sum(axis=0)  # (M,)
    else:
        # 2-D case
        gx, gy = jnp.meshgrid(coords, coords, indexing="ij")
        grid = jnp.stack([gx, gy], axis=-1)  # (M, M, 2)
        pos = positions[:, None, None, :]  # (N, 1, 1, 2)
        dist_sq = jnp.sum((grid[None, ...] - pos) ** 2, axis=-1)  # (N, M, M)
        gauss = jnp.exp(-dist_sq / (2.0 * sigma_norm**2))  # (N, M, M)
        weighted = gauss * values[:, None, None]  # (N, M, M)
        return weighted.sum(axis=0)  # (M, M)


class TrainingWrapper:
    """Wrap :class:`SpatialCompetitionEnv` for MAPPO training.

    Provides:
    - Flat perfect-information observations
    - Action mapping from ``[-1, 1]`` to environment bounds
    - Vectorised ``reset`` / ``step`` via ``jax.vmap``
    - Automatic episode reset on ``done``
    """

    def __init__(
        self,
        num_sellers: int = 2,
        max_buyers: int = 200,
        dimensions: int = 1,
        space_resolution: int = 100,
        max_price: float = 10.0,
        max_quality: float = 5.0,
        max_step_size: float = 0.1,
        production_cost_factor: float = 0.5,
        movement_cost: float = 0.0,
        transport_cost: float = 2.0,
        transportation_cost_norm: float = 2.0,
        quality_taste: float = 0.0,
        include_quality: bool = False,
        new_buyers_per_step: int = 50,
        max_env_steps: int = 100,
        buyer_choice_temperature: float | None = None,
        blob_sigma: float = 1.5,
    ) -> None:
        self.env = SpatialCompetitionEnv(
            num_sellers=num_sellers,
            max_buyers=max_buyers,
            dimensions=dimensions,
            space_resolution=space_resolution,
            max_price=max_price,
            max_quality=max_quality,
            max_step_size=max_step_size,
            production_cost_factor=production_cost_factor,
            movement_cost=movement_cost,
            transportation_cost_norm=transportation_cost_norm,
            include_quality=include_quality,
            new_buyers_per_step=new_buyers_per_step,
            max_env_steps=max_env_steps,
            buyer_choice_temperature=buyer_choice_temperature,
            buyer_distance_factor_sampler=make_constant_sampler(transport_cost),
            buyer_quality_taste_sampler=make_constant_sampler(quality_taste),
        )

        self.num_agents = num_sellers
        self.dimensions = dimensions
        self.max_price = max_price
        self.max_step_size = max_step_size
        self.space_resolution = space_resolution
        self.include_quality = include_quality
        self.max_quality = max_quality
        self.blob_sigma = blob_sigma

        # Observation / state dimensions
        self._per_agent_dim = dimensions + 1  # position + price
        if include_quality:
            self._per_agent_dim += 1

        # Spatial blob channels: presence + avg_price (+ avg_quality)
        n_blob_channels = 2  # presence, avg_price
        if include_quality:
            n_blob_channels += 1  # avg_quality
        cells_per_channel = space_resolution**dimensions
        self._blob_dim = n_blob_channels * cells_per_channel

        self.state_dim = self._blob_dim + self._per_agent_dim * num_sellers
        self.obs_dim = self.state_dim  # perfect information

        # Action dimensions (split by distribution family)
        self.movement_dim = dimensions  # Gaussian (tanh-squashed)
        self.bounded_dim = 1  # Beta: price
        if include_quality:
            self.bounded_dim += 1  # Beta: + quality
        self.action_dim = self.movement_dim + self.bounded_dim

    # ------------------------------------------------------------------
    # Observation / action helpers
    # ------------------------------------------------------------------

    def extract_global_state(self, state: EnvState) -> jnp.ndarray:
        """Build a flat global-state vector from *state*.

        Layout::

            [ seller_presence_map, seller_avg_price_map,
              (seller_avg_quality_map,)
              pos0/R, price0/max, (quality0/max,) pos1/R, … ]

        Returns:
            Array of shape ``(state_dim,)``.
        """
        positions = state.seller_positions.astype(jnp.float32) / self.space_resolution
        norm_prices = state.seller_prices / self.max_price

        # --- Gaussian blob spatial channels (sellers only) ---
        ones = jnp.ones(self.num_agents)
        presence = _gaussian_blob_channel(
            positions,
            ones,
            self.space_resolution,
            self.blob_sigma,
            self.dimensions,
        )
        price_weighted = _gaussian_blob_channel(
            positions,
            norm_prices,
            self.space_resolution,
            self.blob_sigma,
            self.dimensions,
        )
        avg_price = price_weighted / (presence + 1e-8)

        channels = [presence.ravel(), avg_price.ravel()]

        if self.include_quality:
            norm_qualities = state.seller_qualities / self.max_quality
            quality_weighted = _gaussian_blob_channel(
                positions,
                norm_qualities,
                self.space_resolution,
                self.blob_sigma,
                self.dimensions,
            )
            avg_quality = quality_weighted / (presence + 1e-8)
            channels.append(avg_quality.ravel())

        # --- Per-agent scalar features ---
        prices_col = norm_prices[:, None]
        if self.include_quality:
            qualities_col = norm_qualities[:, None]  # type: ignore[possibly-undefined]
            per_agent = jnp.concatenate([positions, prices_col, qualities_col], axis=-1)
        else:
            per_agent = jnp.concatenate([positions, prices_col], axis=-1)

        channels.append(per_agent.ravel())
        return jnp.concatenate(channels)

    def map_actions(self, actions: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Map network actions to the env action dict.

        Movement dims (indices ``[:D]``) are in ``[-1, 1]`` (tanh Gaussian)
        and get scaled by ``max_step_size``.
        Bounded dims (indices ``[D:]``) are in ``(0, 1)`` (Beta) and get
        scaled by their respective maximum values.

        Args:
            actions: ``(num_agents, action_dim)``.

        Returns:
            Dict with ``'movement'``, ``'price'``, (optionally ``'quality'``).
        """
        D = self.dimensions
        movement = actions[:, :D] * self.max_step_size
        price = actions[:, D] * self.max_price

        result: dict[str, jnp.ndarray] = {"movement": movement, "price": price}

        if self.include_quality:
            quality = actions[:, D + 1] * self.max_quality
            result["quality"] = quality

        return result

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def reset(self, key: jnp.ndarray) -> tuple[jnp.ndarray, EnvState]:
        """Reset the environment.

        Returns:
            ``(global_state, env_state)``
        """
        _, env_state = self.env.reset(key)
        global_state = self.extract_global_state(env_state)
        return global_state, env_state

    def step(
        self,
        key: jnp.ndarray,
        env_state: EnvState,
        actions: jnp.ndarray,
        temperature: jnp.ndarray | float | None = None,
    ) -> tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray]:
        """Step the environment.

        Calls the four env phases directly so that
        ``build_observations`` is never computed (we only need the
        flat global state extracted from :class:`EnvState`).

        Args:
            key: PRNG key.
            env_state: Current ``EnvState``.
            actions: ``(num_agents, action_dim)``.
            temperature: Optional dynamic buyer-choice temperature
                override (for annealing during training).

        Returns:
            ``(global_state, new_env_state, rewards, dones)``
        """
        env_actions = self.map_actions(actions)
        k_spawn, k_sales = jax.random.split(key)

        env_state = self.env.step_remove_purchased(env_state)
        env_state = self.env.step_spawn_buyers(k_spawn, env_state)
        env_state = self.env.step_apply_actions(env_state, env_actions)

        # Inline the sales / reward logic without building grid obs
        running_sales, bought_from = self.env._process_sales(
            k_sales,
            env_state.seller_positions,
            env_state.seller_prices,
            env_state.seller_qualities,
            env_state.buyer_positions,
            env_state.buyer_valid,
            env_state.buyer_values,
            env_state.buyer_quality_tastes,
            env_state.buyer_distance_factors,
            temperature=temperature,
        )

        # Rewards
        revenue = running_sales * env_state.seller_prices
        if self.env.include_quality:
            prod_cost = self.env.production_cost_factor * env_state.seller_qualities**2
        else:
            prod_cost = jnp.float32(0.0)
        move_cost = self.env.movement_cost * env_state.seller_last_movement
        rewards = revenue - prod_cost - move_cost

        new_step = env_state.step + 1
        new_env_state = env_state.replace(  # type: ignore[attr-defined]
            seller_running_sales=running_sales,
            buyer_purchased_from=bought_from,
            step=new_step,
        )

        done = new_step >= self.env.max_env_steps
        dones = jnp.full(self.env.num_sellers, done)

        global_state = self.extract_global_state(new_env_state)
        return global_state, new_env_state, rewards, dones

    def step_autoreset(
        self,
        key: jnp.ndarray,
        env_state: EnvState,
        actions: jnp.ndarray,
        temperature: jnp.ndarray | float | None = None,
    ) -> tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray]:
        """Step with automatic reset when the episode is done.

        Args:
            temperature: Optional dynamic buyer-choice temperature
                override (for annealing during training).

        Returns:
            ``(global_state, env_state, rewards, dones)``
            When done, the returned ``env_state`` is already reset.
        """
        k_step, k_reset = jax.random.split(key)

        global_state, new_env_state, rewards, dones = self.step(
            k_step,
            env_state,
            actions,
            temperature=temperature,
        )

        done = dones[0]  # all agents share done status

        reset_global_state, reset_env_state = self.reset(k_reset)

        final_env_state = jax.tree.map(
            lambda r, s: jnp.where(done, r, s),
            reset_env_state,
            new_env_state,
        )
        final_global_state = jnp.where(done, reset_global_state, global_state)

        return final_global_state, final_env_state, rewards, dones
