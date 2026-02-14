"""Training wrapper for SpatialCompetitionEnv.

Provides flat observations, action mapping, and vectorised operations
via ``jax.vmap`` for MAPPO training.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

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
        transport_cost_exponent: float = 1.0,
        quality_taste: float = 0.0,
        include_quality: bool = False,
        new_buyers_per_step: int = 50,
        max_env_steps: int = 100,
        buyer_choice_temperature: float | None = None,
        blob_sigma: float = 1.5,
        # Discrete action space
        action_type: str = "continuous",
        num_location_bins: int = 10,
        num_price_bins: int = 10,
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
            transport_cost_exponent=transport_cost_exponent,
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

        # Discrete action space config
        self.action_type = action_type
        self.num_location_bins = num_location_bins
        self.num_price_bins = num_price_bins
        self.num_actions = num_location_bins * num_price_bins  # joint categorical size

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

        # Precompute ego-centric reordering indices.
        # _ego_indices[i] = [i, (i+1)%A, (i+2)%A, …]
        # So agent i always sees its own features first.
        self._ego_indices = jnp.array(
            np.stack([np.roll(np.arange(num_sellers), -i) for i in range(num_sellers)])
        )  # (A, A)

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

    def extract_all_agent_obs(self, state: EnvState) -> jnp.ndarray:
        """Build egocentric observations for all agents.

        Each agent's observation has the same dimension but with its
        own features placed first in the per-agent section.

        Layout (per agent)::

            [ blob_channels,
              my_pos/R, my_price/max, (my_quality/max,)
              other0_pos/R, other0_price/max, (other0_quality/max,)
              … ]

        Returns:
            Array of shape ``(A, obs_dim)``.
        """
        positions = state.seller_positions.astype(jnp.float32) / self.space_resolution
        norm_prices = state.seller_prices / self.max_price

        # --- Gaussian blob spatial channels (sellers only) ---
        ones = jnp.ones(self.num_agents)
        presence = _gaussian_blob_channel(
            positions, ones, self.space_resolution, self.blob_sigma, self.dimensions,
        )
        price_weighted = _gaussian_blob_channel(
            positions, norm_prices, self.space_resolution, self.blob_sigma, self.dimensions,
        )
        avg_price = price_weighted / (presence + 1e-8)

        channels = [presence.ravel(), avg_price.ravel()]

        if self.include_quality:
            norm_qualities = state.seller_qualities / self.max_quality
            quality_weighted = _gaussian_blob_channel(
                positions, norm_qualities, self.space_resolution, self.blob_sigma, self.dimensions,
            )
            avg_quality = quality_weighted / (presence + 1e-8)
            channels.append(avg_quality.ravel())

        blob_vec = jnp.concatenate(channels)  # (blob_dim,)

        # --- Per-agent scalar features ---
        prices_col = norm_prices[:, None]
        if self.include_quality:
            qualities_col = norm_qualities[:, None]  # type: ignore[possibly-undefined]
            per_agent = jnp.concatenate([positions, prices_col, qualities_col], axis=-1)
        else:
            per_agent = jnp.concatenate([positions, prices_col], axis=-1)
        # per_agent: (A, per_agent_dim)

        # --- Ego-centric reordering ---
        # _ego_indices: (A, A) — each row reorders agents so "self" is first
        ego_per_agent = per_agent[self._ego_indices]  # (A, A, per_agent_dim)
        ego_flat = ego_per_agent.reshape(self.num_agents, -1)  # (A, A*per_agent_dim)

        # Broadcast blob channels to all agents (shared)
        blob_broadcast = jnp.broadcast_to(blob_vec, (self.num_agents, blob_vec.shape[0]))

        return jnp.concatenate([blob_broadcast, ego_flat], axis=-1)  # (A, obs_dim)

    def map_actions(self, actions: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Map continuous network actions to the env action dict.

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

    def map_discrete_actions(
        self,
        actions: jnp.ndarray,
        seller_positions: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """Map discrete action indices to the env action dict.

        Each action is a joint index in ``[0, num_location_bins * num_price_bins)``.
        Decodes into ``(loc_bin, price_bin)`` and converts to
        absolute position / price.

        Location bin *i* maps directly to grid position *i* (no
        rounding to bin centre).  With ``num_location_bins == space_resolution``
        this gives exact 1-to-1 coverage.

        Movement is computed as the delta from the current position to
        the target grid position.

        Args:
            actions: ``(num_agents, 1)`` float32 bin indices.
            seller_positions: ``(num_agents, D)`` int32 current positions.

        Returns:
            Dict with ``'movement'``, ``'price'``.
        """
        idx = actions[:, 0].astype(jnp.int32)  # (A,)
        loc_bin = idx // self.num_price_bins
        price_bin = idx % self.num_price_bins

        # Target position: bin i maps to grid position i * (res / n_loc)
        n_loc = self.num_location_bins
        target_pos = (
            loc_bin.astype(jnp.float32)
            / jnp.maximum(n_loc - 1, 1)
            * (self.space_resolution - 1)
        ).astype(jnp.int32)
        target_pos = jnp.clip(target_pos, 0, self.space_resolution - 1)

        # Movement delta (in normalised [0, 1] coords)
        current_norm = seller_positions.astype(jnp.float32) / self.space_resolution
        target_norm = target_pos[:, None].astype(jnp.float32) / self.space_resolution
        movement = target_norm - current_norm  # (A, D)

        # Price: evenly spaced bins in [0, max_price]
        n_price = self.num_price_bins
        price = (
            price_bin.astype(jnp.float32) / jnp.maximum(n_price - 1, 1)
        ) * self.max_price

        return {"movement": movement, "price": price}

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
            actions: ``(num_agents, action_dim)`` for continuous,
                or ``(num_agents, 1)`` for discrete.
            temperature: Optional dynamic buyer-choice temperature
                override (for annealing during training).

        Returns:
            ``(global_state, new_env_state, rewards, dones)``
        """
        if self.action_type == "discrete":
            env_actions = self.map_discrete_actions(
                actions, env_state.seller_positions,
            )
        else:
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

    # ------------------------------------------------------------------
    # Egocentric Reset / Step (return (A, obs_dim) observations)
    # ------------------------------------------------------------------

    def reset_ego(self, key: jnp.ndarray) -> tuple[jnp.ndarray, EnvState]:
        """Reset and return egocentric observations.

        Returns:
            ``(obs, env_state)`` where ``obs`` has shape ``(A, obs_dim)``.
        """
        _, env_state = self.env.reset(key)
        obs = self.extract_all_agent_obs(env_state)
        return obs, env_state

    def step_ego(
        self,
        key: jnp.ndarray,
        env_state: EnvState,
        actions: jnp.ndarray,
        temperature: jnp.ndarray | float | None = None,
    ) -> tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray]:
        """Step and return egocentric observations.

        Dispatches to continuous or discrete action mapping based on
        ``self.action_type``.

        Returns:
            ``(obs, new_env_state, rewards, dones)``
            where ``obs`` has shape ``(A, obs_dim)``.
        """
        if self.action_type == "discrete":
            env_actions = self.map_discrete_actions(
                actions, env_state.seller_positions,
            )
        else:
            env_actions = self.map_actions(actions)
        k_spawn, k_sales = jax.random.split(key)

        env_state = self.env.step_remove_purchased(env_state)
        env_state = self.env.step_spawn_buyers(k_spawn, env_state)
        env_state = self.env.step_apply_actions(env_state, env_actions)

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

        obs = self.extract_all_agent_obs(new_env_state)
        return obs, new_env_state, rewards, dones

    def step_autoreset_ego(
        self,
        key: jnp.ndarray,
        env_state: EnvState,
        actions: jnp.ndarray,
        temperature: jnp.ndarray | float | None = None,
    ) -> tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray]:
        """Step with auto-reset, returning egocentric observations.

        Returns:
            ``(obs, env_state, rewards, dones)``
        """
        k_step, k_reset = jax.random.split(key)

        obs, new_env_state, rewards, dones = self.step_ego(
            k_step, env_state, actions, temperature=temperature,
        )

        done = dones[0]

        reset_obs, reset_env_state = self.reset_ego(k_reset)

        final_env_state = jax.tree.map(
            lambda r, s: jnp.where(done, r, s),
            reset_env_state,
            new_env_state,
        )
        final_obs = jnp.where(done, reset_obs, obs)

        return final_obs, final_env_state, rewards, dones
