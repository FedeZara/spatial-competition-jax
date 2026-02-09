"""JAX-native Spatial Competition Environment.

All environment logic is expressed as pure functions over JAX arrays.
``jax.jit(env.step)`` and ``jax.vmap(env.step)`` work out of the box.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import struct

from spatial_competition_jax.observations import build_observations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOPOLOGY_RECTANGLE = 0
TOPOLOGY_TORUS = 1

INFO_PRIVATE = 0
INFO_LIMITED = 1
INFO_COMPLETE = 2


# ---------------------------------------------------------------------------
# EnvState – mutable per-step state (JAX pytree)
# ---------------------------------------------------------------------------


@struct.dataclass
class EnvState:
    """Immutable JAX pytree holding all mutable environment state."""

    # Sellers – fixed-size arrays (S = num_sellers, D = dimensions)
    seller_positions: jnp.ndarray  # (S, D) int32  – tensor coords
    seller_prices: jnp.ndarray  # (S,)   float32
    seller_qualities: jnp.ndarray  # (S,)   float32
    seller_running_sales: jnp.ndarray  # (S,)   float32
    seller_last_movement: jnp.ndarray  # (S,)   float32 – movement norm

    # Buyers – fixed-size arrays with valid mask (B = max_buyers)
    buyer_positions: jnp.ndarray  # (B, D) int32  – tensor coords
    buyer_valid: jnp.ndarray  # (B,)   bool
    buyer_values: jnp.ndarray  # (B,)   float32
    buyer_quality_tastes: jnp.ndarray  # (B,)   float32
    buyer_distance_factors: jnp.ndarray  # (B,)   float32
    buyer_purchased_from: jnp.ndarray  # (B,)   int32  – seller idx, -1 = none

    # Step counter
    step: jnp.ndarray  # ()     int32


# ---------------------------------------------------------------------------
# Default sampler factories
# ---------------------------------------------------------------------------


def uniform_position_sampler(
    key: jnp.ndarray,
    dims: int,
    space_resolution: int,
) -> jnp.ndarray:
    """Uniform position sampler over ``[0, space_resolution)``."""
    return jax.random.randint(key, (dims,), 0, space_resolution)


def make_constant_sampler(value: float) -> Callable:
    """Return a sampler that always yields *value*."""

    def sampler(key: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(value, dtype=jnp.float32)

    return sampler


def make_uniform_sampler(low: float, high: float) -> Callable:
    """Return a sampler drawing from ``Uniform(low, high)``."""

    def sampler(key: jnp.ndarray) -> jnp.ndarray:
        return jax.random.uniform(key, dtype=jnp.float32, minval=low, maxval=high)

    return sampler


def make_normal_sampler(
    mean: float,
    std: float,
    min_val: float = 0.0,
    max_val: float = float("inf"),
) -> Callable:
    """Return a sampler drawing from clipped ``Normal(mean, std)``."""

    def sampler(key: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(
            mean + std * jax.random.normal(key, dtype=jnp.float32),
            min_val,
            max_val,
        )

    return sampler


def make_normal_position_sampler(mean: jnp.ndarray, std: float) -> Callable:
    """Return a position sampler centred on *mean* (space coords 0–1)."""

    def sampler(
        key: jnp.ndarray,
        dims: int,
        space_resolution: int,
    ) -> jnp.ndarray:
        raw = mean + std * jax.random.normal(key, (dims,), dtype=jnp.float32)
        raw = jnp.clip(raw, 0.0, 1.0 - 1.0 / space_resolution)
        return (raw * space_resolution).astype(jnp.int32)

    return sampler


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SpatialCompetitionEnv:
    """JAX-native spatial competition environment.

    All methods are pure functions over JAX arrays.  Wrap with
    ``jax.jit`` / ``jax.vmap`` for GPU acceleration::

        jit_reset = jax.jit(env.reset)
        jit_step  = jax.jit(env.step)
        vmap_step = jax.vmap(env.step)
    """

    def __init__(
        self,
        num_sellers: int = 3,
        max_buyers: int = 200,
        dimensions: int = 2,
        space_resolution: int = 100,
        max_price: float = 10.0,
        max_quality: float = 5.0,
        max_step_size: float = 0.1,
        production_cost_factor: float = 0.5,
        movement_cost: float = 0.1,
        transportation_cost_norm: float = 2.0,
        transport_cost_exponent: float = 1.0,
        topology: int = TOPOLOGY_RECTANGLE,
        information_level: int = INFO_COMPLETE,
        include_quality: bool = False,
        include_buyer_valuation: bool = False,
        new_buyers_per_step: int = 50,
        max_env_steps: int = 100,
        buyer_choice_temperature: float | None = None,
        # Optional custom samplers
        seller_position_sampler: Callable | None = None,
        seller_price_sampler: Callable | None = None,
        seller_quality_sampler: Callable | None = None,
        buyer_position_sampler: Callable | None = None,
        buyer_value_sampler: Callable | None = None,
        buyer_quality_taste_sampler: Callable | None = None,
        buyer_distance_factor_sampler: Callable | None = None,
    ) -> None:
        # ── store configuration ──
        self.num_sellers = num_sellers
        self.max_buyers = max_buyers
        self.dimensions = dimensions
        self.space_resolution = space_resolution
        self.max_price = max_price
        self.max_quality = max_quality
        self.max_step_size = max_step_size
        self.production_cost_factor = production_cost_factor
        self.movement_cost = movement_cost
        self.transportation_cost_norm = transportation_cost_norm
        self.transport_cost_exponent = transport_cost_exponent
        self.topology = topology
        self.information_level = information_level
        self.include_quality = include_quality
        self.include_buyer_valuation = include_buyer_valuation
        self.new_buyers_per_step = new_buyers_per_step
        self.max_env_steps = max_env_steps
        self.buyer_choice_temperature = buyer_choice_temperature

        # ── precomputed grid metadata ──
        self.grid_shape: tuple[int, ...] = tuple([space_resolution] * dimensions)
        self.total_cells: int = space_resolution**dimensions
        self.strides: tuple[int, ...] = tuple(space_resolution ** (dimensions - 1 - d) for d in range(dimensions))

        # ── samplers (defaults if not provided) ──
        self.seller_position_sampler = seller_position_sampler or uniform_position_sampler
        self.seller_price_sampler = seller_price_sampler or make_constant_sampler(max_price / 2)
        self.seller_quality_sampler = seller_quality_sampler or make_constant_sampler(max_quality / 2)
        self.buyer_position_sampler = buyer_position_sampler or uniform_position_sampler
        self.buyer_value_sampler = buyer_value_sampler or make_constant_sampler(max_price * 2)
        self.buyer_quality_taste_sampler = buyer_quality_taste_sampler or make_constant_sampler(1.0)
        self.buyer_distance_factor_sampler = buyer_distance_factor_sampler or make_constant_sampler(1.0)

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def _apply_topology(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Clamp / wrap positions according to the configured topology."""
        if self.topology == TOPOLOGY_RECTANGLE:
            return jnp.clip(positions, 0, self.space_resolution - 1)
        else:  # TORUS
            return positions % self.space_resolution

    def _compute_distances_pairwise(
        self,
        pos_a: jnp.ndarray,
        pos_b: jnp.ndarray,
    ) -> jnp.ndarray:
        """Pairwise distances.  ``pos_a``: (A, D), ``pos_b``: (B, D) → (A, B)."""
        a = pos_a.astype(jnp.float32) / self.space_resolution
        b = pos_b.astype(jnp.float32) / self.space_resolution

        diff = a[:, None, :] - b[None, :, :]  # (A, B, D)

        if self.topology == TOPOLOGY_TORUS:
            abs_diff = jnp.abs(diff)
            abs_diff = jnp.minimum(abs_diff, 1.0 - abs_diff)
        else:
            abs_diff = jnp.abs(diff)

        p = self.transportation_cost_norm
        if p == float("inf"):
            return jnp.max(abs_diff, axis=-1)
        elif p == 1.0:
            return jnp.sum(abs_diff, axis=-1)
        elif p == 2.0:
            return jnp.sqrt(jnp.sum(abs_diff**2, axis=-1))
        else:
            return jnp.sum(abs_diff**p, axis=-1) ** (1.0 / p)

    # ------------------------------------------------------------------
    # Buyer spawning
    # ------------------------------------------------------------------

    def _spawn_buyers(self, key: jnp.ndarray, state: EnvState) -> EnvState:
        """Fill the first ``new_buyers_per_step`` empty buyer slots."""
        empty = ~state.buyer_valid
        cumsum = jnp.cumsum(empty.astype(jnp.int32))
        spawn_mask = empty & (cumsum <= self.new_buyers_per_step)

        k_pos, k_val, k_qt, k_df = jax.random.split(key, 4)

        # Sample attributes for *all* slots (wasteful but JIT-friendly)
        new_positions = jax.vmap(lambda k: self.buyer_position_sampler(k, self.dimensions, self.space_resolution))(
            jax.random.split(k_pos, self.max_buyers)
        )

        if self.include_buyer_valuation:
            new_values = jnp.clip(
                jax.vmap(self.buyer_value_sampler)(jax.random.split(k_val, self.max_buyers)),
                0.0,
                jnp.inf,
            )
        else:
            new_values = jnp.zeros(self.max_buyers, dtype=jnp.float32)

        if self.include_quality:
            new_qt = jnp.clip(
                jax.vmap(self.buyer_quality_taste_sampler)(jax.random.split(k_qt, self.max_buyers)),
                0.0,
                jnp.inf,
            )
        else:
            new_qt = jnp.zeros(self.max_buyers, dtype=jnp.float32)

        new_df = jnp.clip(
            jax.vmap(self.buyer_distance_factor_sampler)(jax.random.split(k_df, self.max_buyers)),
            0.0,
            jnp.inf,
        )

        return state.replace(  # type: ignore[attr-defined, no-any-return]
            buyer_positions=jnp.where(spawn_mask[:, None], new_positions, state.buyer_positions),
            buyer_valid=state.buyer_valid | spawn_mask,
            buyer_values=jnp.where(spawn_mask, new_values, state.buyer_values),
            buyer_quality_tastes=jnp.where(spawn_mask, new_qt, state.buyer_quality_tastes),
            buyer_distance_factors=jnp.where(spawn_mask, new_df, state.buyer_distance_factors),
        )

    # ------------------------------------------------------------------
    # Sales processing
    # ------------------------------------------------------------------

    def _process_sales(
        self,
        key: jnp.ndarray,
        seller_positions: jnp.ndarray,
        seller_prices: jnp.ndarray,
        seller_qualities: jnp.ndarray,
        buyer_positions: jnp.ndarray,
        buyer_valid: jnp.ndarray,
        buyer_values: jnp.ndarray,
        buyer_quality_tastes: jnp.ndarray,
        buyer_distance_factors: jnp.ndarray,
        temperature: jnp.ndarray | float | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Vectorised sales processing.

        When ``buyer_choice_temperature`` is ``None``, uses hard argmax
        (deterministic, the classical model).  Otherwise, computes a
        softmax distribution over sellers for each buyer and uses the
        *expected* (fractional) allocation as ``seller_running_sales``.
        ``buyer_purchased_from`` still records the argmax seller for
        lifecycle tracking (buyer removal).

        Args:
            temperature: Dynamic temperature override.  When the env is
                configured for softmax (``buyer_choice_temperature`` is
                not ``None``), this value overrides the static one,
                enabling temperature annealing during training.

        Returns:
            ``(seller_running_sales (S,) float32,
              buyer_purchased_from (B,) int32)``
        """
        # Distances: (B, S)
        distances = self._compute_distances_pairwise(buyer_positions, seller_positions)

        # Utility matrix (B, S)
        # Apply transport cost exponent (1 = linear, 2 = quadratic à la d'Aspremont)
        transport_distances = distances ** self.transport_cost_exponent
        utility = (
            buyer_values[:, None]
            - buyer_distance_factors[:, None] * transport_distances
            + buyer_quality_tastes[:, None] * seller_qualities[None, :]
            - seller_prices[None, :]
        )

        # Small noise for random tie-breaking (used by hard path)
        noise = jax.random.uniform(key, utility.shape, dtype=jnp.float32) * 1e-6
        utility = utility + noise

        # Mask out invalid buyers
        utility = jnp.where(buyer_valid[:, None], utility, -jnp.inf)

        # Best seller per buyer (always needed for buyer lifecycle)
        best_seller = jnp.argmax(utility, axis=-1)  # (B,)
        best_utility = jnp.take_along_axis(utility, best_seller[:, None], axis=-1).squeeze(-1)

        # Buy decision
        if self.include_buyer_valuation:
            buys = buyer_valid & (best_utility > 0)
        else:
            buys = buyer_valid  # always buy

        buyer_purchased_from = jnp.where(buys, best_seller, jnp.int32(-1))

        if self.buyer_choice_temperature is not None:
            # ── Soft expected allocation ──
            # Use dynamic temperature if provided, else fall back to
            # the static one.  Clamp to a small epsilon to avoid NaN.
            t = temperature if temperature is not None else self.buyer_choice_temperature
            t = jnp.maximum(jnp.asarray(t, dtype=jnp.float32), 1e-8)

            # Compute softmax probabilities; for invalid buyers use
            # uniform logits (0) so softmax is well-defined, then zero
            # them out via the buys mask.
            logits = utility / t
            logits = jnp.where(buyer_valid[:, None], logits, 0.0)
            probs = jax.nn.softmax(logits, axis=-1)  # (B, S)

            # Expected sales: sum of probabilities weighted by buy mask
            sales = jnp.sum(
                probs * buys[:, None].astype(jnp.float32),
                axis=0,
            )  # (S,) float32
        else:
            # ── Hard argmax allocation (classical) ──
            one_hot = jax.nn.one_hot(best_seller, self.num_sellers)  # (B, S)
            sales = jnp.sum(
                one_hot * buys[:, None].astype(jnp.float32),
                axis=0,
            )  # (S,) float32

        return sales, buyer_purchased_from

    # ------------------------------------------------------------------
    # Step phases (can be called individually for phased rendering)
    # ------------------------------------------------------------------

    def step_remove_purchased(self, state: EnvState) -> EnvState:
        """Phase 1: Remove buyers who purchased in the previous step.

        Clears the ``buyer_purchased_from`` field and flips the ``valid``
        mask for buyers that made a purchase.
        """
        purchased = state.buyer_purchased_from >= 0
        return state.replace(  # type: ignore[attr-defined, no-any-return]
            buyer_valid=state.buyer_valid & ~purchased,
            buyer_purchased_from=jnp.full(self.max_buyers, -1, dtype=jnp.int32),
        )

    def step_spawn_buyers(self, key: jnp.ndarray, state: EnvState) -> EnvState:
        """Phase 2: Spawn new buyers into empty slots."""
        return self._spawn_buyers(key, state)

    def step_apply_actions(
        self,
        state: EnvState,
        actions: dict[str, jnp.ndarray],
    ) -> EnvState:
        """Phase 3: Apply seller actions (movement, price, quality).

        Updates seller positions (with topology), prices, qualities,
        and records the movement norm for reward computation.
        """
        movement = actions["movement"]  # (S, D) float32
        price = actions["price"]  # (S,) float32

        # Clip movement norm
        move_norm = jnp.linalg.norm(movement, axis=-1, keepdims=True)  # (S, 1)
        movement = jnp.where(
            move_norm > self.max_step_size,
            movement * self.max_step_size / jnp.maximum(move_norm, 1e-8),
            movement,
        )
        last_movement = jnp.linalg.norm(movement, axis=-1)  # (S,)

        # Convert to tensor coords & apply topology
        move_tensor = jnp.round(movement * self.space_resolution).astype(jnp.int32)
        new_positions = self._apply_topology(state.seller_positions + move_tensor)

        new_prices = jnp.clip(price, 0.0, self.max_price)
        if self.include_quality:
            new_qualities = jnp.clip(actions["quality"], 0.0, self.max_quality)
        else:
            new_qualities = state.seller_qualities

        return state.replace(  # type: ignore[attr-defined, no-any-return]
            seller_positions=new_positions,
            seller_prices=new_prices,
            seller_qualities=new_qualities,
            seller_last_movement=last_movement,
        )

    def step_process_sales(
        self,
        key: jnp.ndarray,
        state: EnvState,
    ) -> tuple[dict[str, jnp.ndarray], EnvState, jnp.ndarray, jnp.ndarray, dict[str, Any]]:
        """Phase 4: Process sales, compute rewards, build observations.

        Returns the same ``(obs, state, rewards, dones, info)`` tuple as
        :meth:`step`.
        """
        running_sales, bought_from = self._process_sales(
            key,
            state.seller_positions,
            state.seller_prices,
            state.seller_qualities,
            state.buyer_positions,
            state.buyer_valid,
            state.buyer_values,
            state.buyer_quality_tastes,
            state.buyer_distance_factors,
        )

        # Rewards
        revenue = running_sales * state.seller_prices
        if self.include_quality:
            prod_cost = self.production_cost_factor * state.seller_qualities**2
        else:
            prod_cost = jnp.float32(0.0)
        move_cost = self.movement_cost * state.seller_last_movement
        rewards = revenue - prod_cost - move_cost

        # Update state
        new_step = state.step + 1
        new_state = state.replace(  # type: ignore[attr-defined]
            seller_running_sales=running_sales,
            buyer_purchased_from=bought_from,
            step=new_step,
        )

        # Observations & dones
        obs = build_observations(self, new_state)
        done = new_step >= self.max_env_steps
        dones = jnp.full(self.num_sellers, done)
        info: dict[str, Any] = {}

        return obs, new_state, rewards, dones, info

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, key: jnp.ndarray) -> tuple[dict[str, jnp.ndarray], EnvState]:
        """Reset the environment.

        Args:
            key: JAX PRNG key.

        Returns:
            ``(observations, state)``
        """
        k_sell, k_buy, key = jax.random.split(key, 3)
        k_pos, k_price, k_qual = jax.random.split(k_sell, 3)

        # ── sellers ──
        seller_positions = jax.vmap(lambda k: self.seller_position_sampler(k, self.dimensions, self.space_resolution))(
            jax.random.split(k_pos, self.num_sellers)
        )

        seller_prices = jnp.clip(
            jax.vmap(self.seller_price_sampler)(jax.random.split(k_price, self.num_sellers)),
            0.0,
            self.max_price,
        )

        if self.include_quality:
            seller_qualities = jnp.clip(
                jax.vmap(self.seller_quality_sampler)(jax.random.split(k_qual, self.num_sellers)),
                0.0,
                self.max_quality,
            )
        else:
            seller_qualities = jnp.zeros(self.num_sellers, dtype=jnp.float32)

        # ── initial (empty) buyer arrays ──
        state = EnvState(
            seller_positions=seller_positions,
            seller_prices=seller_prices,
            seller_qualities=seller_qualities,
            seller_running_sales=jnp.zeros(self.num_sellers, dtype=jnp.float32),
            seller_last_movement=jnp.zeros(self.num_sellers, dtype=jnp.float32),
            buyer_positions=jnp.zeros((self.max_buyers, self.dimensions), dtype=jnp.int32),
            buyer_valid=jnp.zeros(self.max_buyers, dtype=jnp.bool_),
            buyer_values=jnp.zeros(self.max_buyers, dtype=jnp.float32),
            buyer_quality_tastes=jnp.zeros(self.max_buyers, dtype=jnp.float32),
            buyer_distance_factors=jnp.zeros(self.max_buyers, dtype=jnp.float32),
            buyer_purchased_from=jnp.full(self.max_buyers, -1, dtype=jnp.int32),
            step=jnp.int32(0),
        )

        # Spawn initial buyers
        state = self._spawn_buyers(k_buy, state)

        obs = build_observations(self, state)
        return obs, state

    def step(
        self,
        key: jnp.ndarray,
        state: EnvState,
        actions: dict[str, jnp.ndarray],
    ) -> tuple[dict[str, jnp.ndarray], EnvState, jnp.ndarray, jnp.ndarray, dict[str, Any]]:
        """Execute one environment step (all 4 phases at once).

        For phased rendering, call the individual phase methods instead:
        :meth:`step_remove_purchased`, :meth:`step_spawn_buyers`,
        :meth:`step_apply_actions`, :meth:`step_process_sales`.

        Args:
            key: JAX PRNG key.
            state: Current ``EnvState``.
            actions: Dict with ``'movement'`` (S, D), ``'price'`` (S,),
                and optionally ``'quality'`` (S,).

        Returns:
            ``(observations, new_state, rewards, dones, info)``
        """
        k_spawn, k_sales = jax.random.split(key)

        state = self.step_remove_purchased(state)
        state = self.step_spawn_buyers(k_spawn, state)
        state = self.step_apply_actions(state, actions)
        return self.step_process_sales(k_sales, state)
