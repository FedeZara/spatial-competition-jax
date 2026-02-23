"""Pygame-based renderer for the JAX Spatial Competition environment.

This module lives **outside** the JIT boundary.  All JAX arrays are
converted to NumPy before drawing.

Usage::

    renderer = SpatialCompetitionRenderer(env)
    # inside training loop …
    renderer.render(state, current_step=step, cumulative_rewards=cum_rew)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from spatial_competition_jax.env import EnvState, SpatialCompetitionEnv

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]


@dataclass
class EntityInfo:
    """Lightweight handle for a hovered / selected entity."""

    entity_type: str  # "seller" or "buyer"
    entity_idx: int
    screen_pos: tuple[int, int]
    color: tuple[int, int, int]


class SpatialCompetitionRenderer:
    """Pygame renderer that reads from :class:`EnvState` arrays."""

    def __init__(
        self,
        env: SpatialCompetitionEnv,
        max_env_steps: int | None = None,
        light_mode: bool = False,
    ) -> None:
        if pygame is None:
            raise ImportError("pygame is required for rendering.  Install with:  pip install pygame")

        self._env = env
        self._max_env_steps = max_env_steps or env.max_env_steps
        self._light_mode = light_mode

        # Pygame state (lazy init)
        self._pygame_initialized = False
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._small_font: pygame.font.Font | None = None
        self._window_size = (900, 650)
        self._seller_colors = self._generate_seller_colors()

        # Per-frame numpy caches (extracted from JAX EnvState)
        self._cached_seller_positions: np.ndarray | None = None
        self._cached_seller_prices: np.ndarray | None = None
        self._cached_seller_qualities: np.ndarray | None = None
        self._cached_seller_running_sales: np.ndarray | None = None
        self._cached_buyer_positions: np.ndarray | None = None
        self._cached_buyer_valid: np.ndarray | None = None
        self._cached_buyer_purchased_from: np.ndarray | None = None

        # Cumulative rewards (optional, passed by caller)
        self._cumulative_rewards: dict[str, float] = {}

        # Interactive state
        self._paused = False
        self._speed_multiplier = 1.0
        self._selected_entity: EntityInfo | None = None
        self._hovered_entity: EntityInfo | None = None

        # Entity screen positions for hit detection (updated each frame)
        self._seller_screen_positions: list[tuple[int, int, int, int]] = []  # (x, y, radius, idx)
        self._buyer_screen_positions: list[tuple[int, int, int, int]] = []

        # UI element positions
        self._pause_button_rect: pygame.Rect | None = None
        self._slider_rect: pygame.Rect | None = None
        self._slider_dragging = False

        # Leaderboard click areas (updated each frame)
        self._leaderboard_items: list[tuple[pygame.Rect, int, tuple[int, int, int]]] = []

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------

    @property
    def _theme(self) -> dict[str, tuple[int, int, int]]:
        """Color theme — dark or light."""
        if self._light_mode:
            return {
                "bg":             (255, 255, 255),
                "text":           (30, 30, 30),
                "text_secondary": (80, 80, 80),
                "text_muted":     (120, 120, 120),
                "axis_line":      (60, 60, 60),
                "grid_line":      (200, 200, 200),
                "grid_bg":        (245, 245, 250),
                "grid_border":    (120, 120, 140),
                "tick_label":     (60, 60, 60),
                "buyer_neutral":  (170, 170, 180),
                "panel_bg":       (240, 240, 245),
                "panel_border":   (180, 180, 190),
                "button_bg":      (220, 220, 230),
                "button_border":  (160, 160, 170),
                "button_active":  (180, 220, 180),
                "slider_bg":      (210, 210, 220),
                "slider_handle":  (100, 100, 120),
                "bar_bg":         (230, 230, 240),
                "info_box_bg":    (245, 245, 250),
                "highlight":      (0, 0, 0),
                "speed_max":      (180, 120, 0),
                "paused_text":    (180, 120, 0),
            }
        return {
            "bg":             (30, 30, 40),
            "text":           (200, 200, 200),
            "text_secondary": (150, 150, 150),
            "text_muted":     (120, 120, 140),
            "axis_line":      (100, 100, 120),
            "grid_line":      (60, 60, 75),
            "grid_bg":        (40, 40, 55),
            "grid_border":    (80, 80, 100),
            "tick_label":     (150, 150, 150),
            "buyer_neutral":  (80, 80, 90),
            "panel_bg":       (45, 45, 60),
            "panel_border":   (100, 100, 120),
            "button_bg":      (60, 60, 80),
            "button_border":  (100, 100, 120),
            "button_active":  (80, 120, 80),
            "slider_bg":      (50, 50, 65),
            "slider_handle":  (120, 120, 150),
            "bar_bg":         (40, 40, 55),
            "info_box_bg":    (50, 50, 65),
            "highlight":      (255, 255, 255),
            "speed_max":      (255, 200, 100),
            "paused_text":    (255, 200, 100),
        }

    def _generate_seller_colors(self) -> list[tuple[int, int, int]]:
        """Generate distinct colors for each seller using HSV color space."""
        num_sellers = self._env.num_sellers
        colors: list[tuple[int, int, int]] = []
        for i in range(num_sellers):
            hue = (i * 360 / num_sellers) % 360 if num_sellers > 0 else 0
            # Convert HSV to RGB — brighter for light mode
            h = hue / 60
            saturation = 0.85 if self._light_mode else 0.8
            value = 0.75 if self._light_mode else 0.65
            c = value * saturation
            x = c * (1 - abs(h % 2 - 1))
            m = value - c

            r: float
            g: float
            b: float
            if h < 1:
                r, g, b = c, x, 0.0
            elif h < 2:
                r, g, b = x, c, 0.0
            elif h < 3:
                r, g, b = 0.0, c, x
            elif h < 4:
                r, g, b = 0.0, x, c
            elif h < 5:
                r, g, b = x, 0.0, c
            else:
                r, g, b = c, 0.0, x
            colors.append((int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)))
        return colors

    def _get_buyer_color(self, buyer_idx: int) -> tuple[int, int, int]:
        """Get the color for a buyer based on which seller they purchased from."""
        assert self._cached_buyer_purchased_from is not None
        seller_idx = int(self._cached_buyer_purchased_from[buyer_idx])
        if seller_idx < 0:
            return self._theme["buyer_neutral"]
        return self._seller_colors[seller_idx % len(self._seller_colors)]

    # ------------------------------------------------------------------
    # Pygame lifecycle
    # ------------------------------------------------------------------

    def _init_pygame(self) -> None:
        """Initialize Pygame for rendering."""
        if self._pygame_initialized:
            return
        pygame.init()
        pygame.display.set_caption("Spatial Competition (JAX)")
        self._screen = pygame.display.set_mode(self._window_size)
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 14)
        self._small_font = pygame.font.SysFont("monospace", 12)
        self._pygame_initialized = True

    def close(self) -> None:
        """Close the renderer and cleanup Pygame resources."""
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False
            self._screen = self._clock = self._font = self._small_font = None

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_events(self) -> bool:
        """Handle Pygame events. Returns False if window was closed."""
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
            self._process_event(event, mouse_pos)
        return True

    def _process_event(self, event: pygame.event.Event, mouse_pos: tuple[int, int]) -> None:
        """Process a single Pygame event."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self._paused = not self._paused
            elif event.key == pygame.K_ESCAPE:
                self._selected_entity = None
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_click(mouse_pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._slider_dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self._slider_dragging:
                self._update_slider_from_mouse(mouse_pos)
            else:
                self._update_hover(mouse_pos)

    def _handle_click(self, mouse_pos: tuple[int, int]) -> None:
        """Handle mouse click at the given position."""
        # Check pause button
        if self._pause_button_rect and self._pause_button_rect.collidepoint(mouse_pos):
            self._paused = not self._paused
            return

        # Check slider
        if self._slider_rect and self._slider_rect.collidepoint(mouse_pos):
            self._slider_dragging = True
            self._update_slider_from_mouse(mouse_pos)
            return

        # Check sellers
        for screen_x, screen_y, radius, seller_idx in self._seller_screen_positions:
            if (mouse_pos[0] - screen_x) ** 2 + (mouse_pos[1] - screen_y) ** 2 <= radius**2:
                self._selected_entity = EntityInfo(
                    "seller", seller_idx, (screen_x, screen_y), self._seller_colors[seller_idx]
                )
                return

        # Check buyers
        for screen_x, screen_y, radius, buyer_idx in self._buyer_screen_positions:
            if (mouse_pos[0] - screen_x) ** 2 + (mouse_pos[1] - screen_y) ** 2 <= (radius + 2) ** 2:
                self._selected_entity = EntityInfo(
                    "buyer", buyer_idx, (screen_x, screen_y), self._get_buyer_color(buyer_idx)
                )
                return

        # Check leaderboard items
        for rect, seller_idx, color in self._leaderboard_items:
            if rect.collidepoint(mouse_pos):
                # Find the seller's screen position for the highlight
                screen_pos = (0, 0)
                for sx, sy, _, idx in self._seller_screen_positions:
                    if idx == seller_idx:
                        screen_pos = (sx, sy)
                        break
                self._selected_entity = EntityInfo("seller", seller_idx, screen_pos, color)
                return

        # Clicked empty space - deselect
        self._selected_entity = None

    def _update_slider_from_mouse(self, mouse_pos: tuple[int, int]) -> None:
        """Update speed multiplier based on mouse position on slider."""
        if self._slider_rect is None:
            return

        # Calculate position within slider (0 to 1)
        rel_x = max(0.0, min(1.0, (mouse_pos[0] - self._slider_rect.x) / self._slider_rect.width))

        # Map to speed range: 0.25x to 4x (logarithmic scale), with MAX at far right
        if rel_x >= 0.95:
            self._speed_multiplier = float("inf")
        else:
            self._speed_multiplier = 0.25 * (16 ** (rel_x / 0.95))

    def _update_hover(self, mouse_pos: tuple[int, int]) -> None:
        """Update hovered entity based on mouse position."""
        # Check sellers first (they're on top)
        for screen_x, screen_y, radius, seller_idx in self._seller_screen_positions:
            if (mouse_pos[0] - screen_x) ** 2 + (mouse_pos[1] - screen_y) ** 2 <= radius**2:
                self._hovered_entity = EntityInfo(
                    "seller", seller_idx, (screen_x, screen_y), self._seller_colors[seller_idx]
                )
                return

        # Check buyers
        for screen_x, screen_y, radius, buyer_idx in self._buyer_screen_positions:
            if (mouse_pos[0] - screen_x) ** 2 + (mouse_pos[1] - screen_y) ** 2 <= (radius + 2) ** 2:
                self._hovered_entity = EntityInfo(
                    "buyer", buyer_idx, (screen_x, screen_y), self._get_buyer_color(buyer_idx)
                )
                return

        self._hovered_entity = None

    # ------------------------------------------------------------------
    # State caching
    # ------------------------------------------------------------------

    def _cache_state(self, state: EnvState) -> None:
        """Pull numpy copies of the JAX state for the current frame."""
        self._cached_seller_positions = np.asarray(state.seller_positions)
        self._cached_seller_prices = np.asarray(state.seller_prices)
        self._cached_seller_qualities = np.asarray(state.seller_qualities)
        self._cached_seller_running_sales = np.asarray(state.seller_running_sales)
        self._cached_buyer_positions = np.asarray(state.buyer_positions)
        self._cached_buyer_valid = np.asarray(state.buyer_valid)
        self._cached_buyer_purchased_from = np.asarray(state.buyer_purchased_from)

    def _seller_space_coords(self, seller_idx: int) -> np.ndarray[Any, Any]:
        """Get normalised [0, 1] space coordinates for a seller."""
        assert self._cached_seller_positions is not None
        result: np.ndarray[Any, Any] = (
            self._cached_seller_positions[seller_idx].astype(np.float64) / self._env.space_resolution
        )
        return result

    # ------------------------------------------------------------------
    # Public render
    # ------------------------------------------------------------------

    def render(
        self,
        state: EnvState,
        current_step: int = 0,
        cumulative_rewards: dict[str, float] | None = None,
    ) -> bool:
        """Render one frame.  Returns *False* if the window was closed."""
        self._init_pygame()
        assert self._screen is not None and self._font is not None and self._clock is not None

        self._cache_state(state)
        self._cumulative_rewards = cumulative_rewards or {}

        # Handle Pygame events (uses entity positions from previous frame for hit detection)
        if not self._handle_events():
            return False

        # Clear entity positions for this frame (after event handling so clicks work)
        self._seller_screen_positions = []
        self._buyer_screen_positions = []

        self._screen.fill(self._theme["bg"])
        width, height = self._window_size
        margin = 60

        dimensions = self._env.dimensions
        if dimensions == 1:
            self._render_1d(width, height, margin)
        elif dimensions == 2:
            self._render_2d(width, height, margin)
        else:
            # For dimensions > 2, show stats in the main area
            self._render_high_dim(width, height, margin, dimensions)

        # Draw UI controls
        self._draw_controls(width, current_step)

        # Draw leaderboard (top 10 sellers)
        self._draw_leaderboard(width, height)

        # Clear selection if buyer entity no longer valid
        if self._selected_entity:
            if self._selected_entity.entity_type == "buyer":
                assert self._cached_buyer_valid is not None
                if not self._cached_buyer_valid[self._selected_entity.entity_idx]:
                    self._selected_entity = None

        # Draw highlight around hovered entity
        if self._hovered_entity:
            self._draw_hover_highlight(self._hovered_entity)
            self._draw_tooltip(self._hovered_entity)

        # Draw highlight around selected entity
        if self._selected_entity:
            self._draw_hover_highlight(self._selected_entity)
            self._draw_detail_panel(self._selected_entity)

        pygame.display.flip()
        self._clock.tick(60)  # 60 FPS for smooth interaction
        return True

    def capture_frame(self) -> np.ndarray:
        """Capture the current Pygame surface as an RGB numpy array (H, W, 3)."""
        if self._screen is None:
            raise RuntimeError("Renderer not initialized; call render() first")
        # pygame.surfarray gives (W, H, 3) — transpose to (H, W, 3)
        return np.transpose(pygame.surfarray.array3d(self._screen), (1, 0, 2)).copy()

    def render_and_wait(
        self,
        state: EnvState,
        base_delay: float,
        current_step: int = 0,
        cumulative_rewards: dict[str, float] | None = None,
    ) -> bool:
        """Render and wait *base_delay* seconds (adjusted by speed).

        This method renders the current state and waits for the specified delay,
        continuously processing events and re-rendering to keep the UI interactive.
        The delay is adjusted by the speed multiplier, and pausing freezes the timer.
        """
        import time

        start_time = time.time()
        while True:
            if not self.render(state, current_step, cumulative_rewards):
                return False

            # If paused, reset timer and keep looping
            if self._paused:
                start_time = time.time()
                time.sleep(0.016)  # ~60fps while paused
                continue

            # MAX speed: skip delay entirely (just render one frame)
            if self._speed_multiplier == float("inf"):
                break

            # Calculate elapsed time and check if delay is complete
            elapsed = time.time() - start_time
            adjusted_delay = base_delay / self._speed_multiplier
            if elapsed >= adjusted_delay:
                break

            # Small sleep to prevent busy-waiting
            remaining = adjusted_delay - elapsed
            time.sleep(min(0.016, remaining))  # ~60fps or remaining time
        return True

    def step_and_render(
        self,
        key: "jnp.ndarray",
        state: "EnvState",
        actions: dict[str, "jnp.ndarray"],
        base_delay: float,
        current_step: int = 0,
        cumulative_rewards: dict[str, float] | None = None,
        record_phase_frames: list[np.ndarray] | None = None,
    ) -> tuple["EnvState", "jnp.ndarray", "jnp.ndarray", bool]:
        """Step through all 4 env phases, rendering after each one.

        When *record_phase_frames* is provided, a screenshot is captured and
        appended after each phase (4 frames per step).

        Returns ``(new_state, rewards, dones, alive)`` where *alive* is
        ``False`` when the window was closed.
        """
        import jax

        env = self._env
        k_spawn, k_sales = jax.random.split(key)
        kw = dict(current_step=current_step, cumulative_rewards=cumulative_rewards)

        # Phase 1 – remove previous buyers
        state = env.step_remove_purchased(state)
        if not self.render_and_wait(state, base_delay, **kw):
            return state, None, None, False  # type: ignore[return-value]
        if record_phase_frames is not None:
            record_phase_frames.append(self.capture_frame())

        # Phase 2 – spawn new buyers
        state = env.step_spawn_buyers(k_spawn, state)
        if not self.render_and_wait(state, base_delay, **kw):
            return state, None, None, False  # type: ignore[return-value]
        if record_phase_frames is not None:
            record_phase_frames.append(self.capture_frame())

        # Phase 3 – apply seller actions
        state = env.step_apply_actions(state, actions)
        if not self.render_and_wait(state, base_delay, **kw):
            return state, None, None, False  # type: ignore[return-value]
        if record_phase_frames is not None:
            record_phase_frames.append(self.capture_frame())

        # Phase 4 – process sales
        obs, state, rewards, dones, info = env.step_process_sales(k_sales, state)
        if not self.render_and_wait(state, base_delay, **kw):
            return state, rewards, dones, False
        if record_phase_frames is not None:
            record_phase_frames.append(self.capture_frame())

        return state, rewards, dones, True

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_controls(self, width: int, current_step: int) -> None:
        """Draw UI controls (step counter, pause button, speed slider)."""
        assert self._screen is not None and self._font is not None
        assert self._cached_buyer_valid is not None

        t = self._theme
        # Step counter
        step_text = self._font.render(f"Step: {current_step}/{self._max_env_steps}", True, t["text"])
        self._screen.blit(step_text, (10, 10))

        # Sellers and buyers count
        num_buyers = int(np.sum(self._cached_buyer_valid))
        counts_text = self._font.render(
            f"Sellers: {self._env.num_sellers}  Buyers: {num_buyers}", True, t["text_secondary"]
        )
        self._screen.blit(counts_text, (10, 28))

        # Speed label and value
        speed_label = self._font.render("Speed:", True, t["text_secondary"])
        if self._speed_multiplier == float("inf"):
            speed_value = self._font.render("MAX", True, t["speed_max"])
        else:
            speed_value = self._font.render(f"{self._speed_multiplier:.1f}x", True, t["text"])

        # Position elements from right edge
        button_width, button_height = 70, 25
        slider_width, slider_height = 80, 10

        # Pause button (rightmost)
        button_x = width - button_width - 10
        button_y = 10
        self._pause_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        button_color = t["button_active"] if self._paused else t["button_bg"]
        pygame.draw.rect(self._screen, button_color, self._pause_button_rect, border_radius=4)
        pygame.draw.rect(self._screen, t["button_border"], self._pause_button_rect, 1, border_radius=4)

        pause_text = self._font.render("Resume" if self._paused else "Pause", True, t["text"])
        text_x = button_x + (button_width - pause_text.get_width()) // 2
        text_y = button_y + (button_height - pause_text.get_height()) // 2
        self._screen.blit(pause_text, (text_x, text_y))

        # Speed value (left of button)
        value_x = button_x - speed_value.get_width() - 10
        self._screen.blit(speed_value, (value_x, 13))

        # Slider (left of value)
        slider_x = value_x - slider_width - 10
        slider_y = 17
        self._slider_rect = pygame.Rect(slider_x, slider_y, slider_width, slider_height)
        pygame.draw.rect(self._screen, t["slider_bg"], self._slider_rect, border_radius=3)

        # Slider handle position (logarithmic: 0.25 -> 0, 1.0 -> ~0.47, 4.0 -> 0.95, MAX -> 1.0)
        if self._speed_multiplier == float("inf"):
            handle_pos = 1.0
        else:
            handle_pos = float(np.log(self._speed_multiplier / 0.25) / np.log(16) * 0.95)
        handle_x = int(slider_x + handle_pos * slider_width)
        handle_rect = pygame.Rect(handle_x - 4, slider_y - 2, 8, slider_height + 4)
        pygame.draw.rect(self._screen, t["slider_handle"], handle_rect, border_radius=2)

        # Speed label (left of slider)
        label_x = slider_x - speed_label.get_width() - 8
        self._screen.blit(speed_label, (label_x, 13))

        # Paused indicator
        if self._paused:
            paused_text = self._font.render("PAUSED (Space to resume)", True, t["paused_text"])
            self._screen.blit(paused_text, (width // 2 - paused_text.get_width() // 2, 10))

    # ---- 1-D rendering ----

    def _render_1d(self, width: int, height: int, margin: int) -> None:
        """Render 1D environment as a horizontal line with entities."""
        assert self._screen is not None and self._font is not None
        assert (
            self._cached_seller_positions is not None
            and self._cached_seller_prices is not None
            and self._cached_buyer_positions is not None
            and self._cached_buyer_valid is not None
        )

        # Draw the market line (add extra left margin for leaderboard)
        legend_width = 220
        line_y = height // 2
        line_start = legend_width + margin
        line_end = width - margin
        line_length = line_end - line_start

        t = self._theme
        # Draw axis line
        pygame.draw.line(self._screen, t["axis_line"], (line_start, line_y), (line_end, line_y), 2)

        # Draw axis markers
        for i in range(11):
            tick_x = line_start + (line_length * i) // 10
            pygame.draw.line(self._screen, t["axis_line"], (tick_x, line_y - 5), (tick_x, line_y + 5), 1)
            label = self._font.render(f"{i / 10:.1f}", True, t["tick_label"])
            self._screen.blit(label, (tick_x - label.get_width() // 2, line_y + 15))

        # Draw buyers as small dots below the line, colored by purchased seller
        buyer_radius = 3
        for buyer_idx in range(self._env.max_buyers):
            if not self._cached_buyer_valid[buyer_idx]:
                continue
            x_pos = self._cached_buyer_positions[buyer_idx, 0] / self._env.space_resolution
            screen_x = int(line_start + x_pos * line_length)
            screen_y = line_y + 40
            buyer_color = self._get_buyer_color(buyer_idx)
            pygame.draw.circle(self._screen, buyer_color, (screen_x, screen_y), buyer_radius)
            self._buyer_screen_positions.append((screen_x, screen_y, buyer_radius, buyer_idx))

        # Draw sellers as larger circles above the line with labels
        seller_radius = 12
        for seller_idx in range(self._env.num_sellers):
            x_pos = self._cached_seller_positions[seller_idx, 0] / self._env.space_resolution
            screen_x = int(line_start + x_pos * line_length)
            screen_y = line_y - 30
            color = self._seller_colors[seller_idx]

            # Draw seller circle
            pygame.draw.circle(self._screen, color, (screen_x, screen_y), seller_radius)
            self._seller_screen_positions.append((screen_x, screen_y, seller_radius, seller_idx))

            # Draw seller label
            label = self._font.render(f"seller_{seller_idx}", True, color)
            self._screen.blit(label, (screen_x - label.get_width() // 2, line_y - 85))

            # Draw price (and optionally quality)
            info_text = f"p={float(self._cached_seller_prices[seller_idx]):.1f}"
            if self._env.include_quality:
                assert self._cached_seller_qualities is not None
                info_text += f" q={float(self._cached_seller_qualities[seller_idx]):.1f}"
            price_label = self._font.render(info_text, True, t["text"])
            self._screen.blit(price_label, (screen_x - price_label.get_width() // 2, line_y - 70))

    # ---- 2-D rendering ----

    def _render_2d(self, width: int, height: int, margin: int) -> None:
        """Render 2D environment as a grid with entities."""
        assert self._screen is not None and self._font is not None
        assert (
            self._cached_seller_positions is not None
            and self._cached_seller_prices is not None
            and self._cached_buyer_positions is not None
            and self._cached_buyer_valid is not None
        )

        # Calculate drawing area (square to maintain aspect ratio)
        legend_width = 220
        available_width = width - legend_width
        draw_size = min(available_width, height) - 2 * margin
        offset_x = legend_width + (available_width - draw_size) // 2
        offset_y = (height - draw_size) // 2

        t = self._theme
        # Draw grid background
        pygame.draw.rect(self._screen, t["grid_bg"], pygame.Rect(offset_x, offset_y, draw_size, draw_size))
        pygame.draw.rect(self._screen, t["grid_border"], pygame.Rect(offset_x, offset_y, draw_size, draw_size), 2)

        # Draw grid lines
        for i in range(11):
            grid_x = offset_x + (draw_size * i) // 10
            pygame.draw.line(self._screen, t["grid_line"], (grid_x, offset_y), (grid_x, offset_y + draw_size), 1)
            grid_y = offset_y + (draw_size * i) // 10
            pygame.draw.line(self._screen, t["grid_line"], (offset_x, grid_y), (offset_x + draw_size, grid_y), 1)

        # Draw sellers first (as larger circles with labels)
        seller_radius = 10
        for seller_idx in range(self._env.num_sellers):
            coords = self._seller_space_coords(seller_idx)
            screen_x = int(offset_x + coords[0] * draw_size)
            screen_y = int(offset_y + (1 - coords[1]) * draw_size)  # Flip Y for screen coords
            color = self._seller_colors[seller_idx]

            # Draw seller circle
            pygame.draw.circle(self._screen, color, (screen_x, screen_y), seller_radius)
            self._seller_screen_positions.append((screen_x, screen_y, seller_radius, seller_idx))

            # Draw seller label (ID, price, and optionally quality)
            label = self._font.render(f"seller_{seller_idx}", True, color)
            self._screen.blit(label, (screen_x + 15, screen_y - 25))

            info_text = f"p={float(self._cached_seller_prices[seller_idx]):.1f}"
            if self._env.include_quality:
                assert self._cached_seller_qualities is not None
                info_text += f" q={float(self._cached_seller_qualities[seller_idx]):.1f}"
            price_label = self._font.render(info_text, True, t["text"])
            self._screen.blit(price_label, (screen_x + 15, screen_y - 10))

        # Draw buyers on top (as small dots, colored by purchased seller)
        buyer_radius = 3
        for buyer_idx in range(self._env.max_buyers):
            if not self._cached_buyer_valid[buyer_idx]:
                continue
            coords = self._cached_buyer_positions[buyer_idx].astype(np.float64) / self._env.space_resolution
            screen_x = int(offset_x + coords[0] * draw_size)
            screen_y = int(offset_y + (1 - coords[1]) * draw_size)  # Flip Y for screen coords
            buyer_color = self._get_buyer_color(buyer_idx)
            pygame.draw.circle(self._screen, buyer_color, (screen_x, screen_y), buyer_radius)
            self._buyer_screen_positions.append((screen_x, screen_y, buyer_radius, buyer_idx))

    # ---- High-dim rendering ----

    def _render_high_dim(self, width: int, height: int, margin: int, dimensions: int) -> None:
        """Render high-dimensional environment (>2D) with stats display."""
        assert self._screen is not None and self._font is not None and self._small_font is not None
        assert self._cached_buyer_valid is not None and self._cached_seller_running_sales is not None

        # Left margin for leaderboard
        legend_width = 220
        content_x = legend_width + margin
        content_width = width - legend_width - 2 * margin

        t = self._theme
        # Title
        title = self._font.render(f"{dimensions}D Spatial Competition", True, t["text"])
        self._screen.blit(title, (content_x + (content_width - title.get_width()) // 2, 60))

        subtitle = self._small_font.render("(See leaderboard for rankings)", True, t["text_muted"])
        self._screen.blit(subtitle, (content_x + (content_width - subtitle.get_width()) // 2, 85))

        # Competition stats
        num_buyers = int(np.sum(self._cached_buyer_valid))
        total_sales = float(np.sum(self._cached_seller_running_sales))
        total_reward = sum(self._cumulative_rewards.values()) if self._cumulative_rewards else 0.0

        stats = [
            ("Dimensions", str(dimensions)),
            ("Active Sellers", str(self._env.num_sellers)),
            ("Active Buyers", str(num_buyers)),
            ("Running Sales", f"{total_sales:.1f}"),
            ("Total Reward", f"{total_reward:.1f}"),
        ]

        # Draw stats box
        box_width = min(300, content_width - 40)
        box_x = content_x + (content_width - box_width) // 2
        box_y = 120
        box_height = len(stats) * 28 + 20
        pygame.draw.rect(self._screen, t["panel_bg"], (box_x, box_y, box_width, box_height), border_radius=8)
        pygame.draw.rect(self._screen, t["panel_border"], (box_x, box_y, box_width, box_height), 2, border_radius=8)

        # Draw stats
        for i, (label, value) in enumerate(stats):
            row_y = box_y + 12 + i * 28
            label_text = self._font.render(label, True, t["text_secondary"])
            value_text = self._font.render(value, True, t["text"])
            self._screen.blit(label_text, (box_x + 15, row_y))
            self._screen.blit(value_text, (box_x + box_width - value_text.get_width() - 15, row_y))

        # Hint at bottom
        hint = self._small_font.render("Click leaderboard items for details", True, t["text_muted"])
        self._screen.blit(hint, (content_x + (content_width - hint.get_width()) // 2, height - 40))

    # ---- Leaderboard ----

    def _draw_leaderboard(self, width: int, height: int) -> None:  # noqa: ARG002
        """Draw a leaderboard chart showing top 10 sellers by cumulative reward."""
        assert self._screen is not None and self._font is not None
        assert self._cached_seller_prices is not None

        # Get sellers sorted by cumulative reward
        seller_data: list[tuple[int, float, tuple[int, int, int]]] = []
        for seller_idx in range(self._env.num_sellers):
            reward = self._cumulative_rewards.get(f"seller_{seller_idx}", 0.0)
            seller_data.append((seller_idx, reward, self._seller_colors[seller_idx]))

        # Sort by reward descending and take top 10
        seller_data.sort(key=lambda x: x[1], reverse=True)
        top_sellers = seller_data[:10]

        # Clear and rebuild leaderboard click areas
        self._leaderboard_items = []
        if not top_sellers:
            return

        # Chart dimensions
        chart_x = 10
        chart_y = 80
        chart_width = 200
        bar_height = 26
        bar_spacing = 4

        t = self._theme
        # Draw chart title
        title = self._font.render("Top Sellers (by reward)", True, t["text_secondary"])
        self._screen.blit(title, (chart_x, chart_y - 20))

        # Find max reward for scaling (guard against NaN)
        rewards_valid = [e[1] for e in top_sellers if not (np.isnan(e[1]) or np.isinf(e[1]))]
        max_reward = max(rewards_valid) if rewards_valid else 1.0
        if max_reward <= 0:
            max_reward = 1.0

        # Draw bars
        for i, (seller_idx, reward, color) in enumerate(top_sellers):
            row_y = chart_y + i * (bar_height + bar_spacing)

            # Bar background
            bg_rect = pygame.Rect(chart_x, row_y, chart_width, bar_height)
            pygame.draw.rect(self._screen, t["bar_bg"], bg_rect, border_radius=3)
            self._leaderboard_items.append((bg_rect, seller_idx, color))

            # Highlight if this seller is selected
            is_selected = (
                self._selected_entity is not None
                and self._selected_entity.entity_type == "seller"
                and self._selected_entity.entity_idx == seller_idx
            )
            if is_selected:
                pygame.draw.rect(self._screen, t["highlight"], bg_rect, 2, border_radius=3)

            # Bar fill (proportional to reward)
            ratio = (reward / max_reward) if max_reward > 0 else 0.0
            bar_width = int(ratio * (chart_width - 4)) if not (np.isnan(ratio) or np.isinf(ratio)) else 0
            if bar_width > 0:
                bar_rect = pygame.Rect(chart_x + 2, row_y + 2, bar_width, bar_height - 4)
                pygame.draw.rect(self._screen, color, bar_rect, border_radius=2)

            # Seller label and reward (vertically centered)
            label_color = t["bg"] if not self._light_mode else (255, 255, 255)
            label = self._font.render(f"seller_{seller_idx}", True, label_color)
            label_y = row_y + (bar_height - label.get_height()) // 2
            self._screen.blit(label, (chart_x + 5, label_y))

            reward_text = self._font.render(f"{reward:.0f}", True, t["text"])
            reward_y = row_y + (bar_height - reward_text.get_height()) // 2
            self._screen.blit(reward_text, (chart_x + chart_width - reward_text.get_width() - 5, reward_y))

    # ---- Tooltip / highlight ----

    def _draw_hover_highlight(self, entity_info: EntityInfo) -> None:
        """Draw a white border highlight around the hovered or selected entity."""
        assert self._screen is not None
        current_pos = self._find_entity_screen_pos(entity_info)
        if current_pos is None:
            return
        radius = 6 if entity_info.entity_type == "buyer" else 14
        pygame.draw.circle(self._screen, self._theme["highlight"], current_pos, radius, 2)

    def _find_entity_screen_pos(self, entity_info: EntityInfo) -> tuple[int, int] | None:
        """Find the current screen position of an entity, or None if not found."""
        positions = (
            self._seller_screen_positions if entity_info.entity_type == "seller" else self._buyer_screen_positions
        )
        for screen_x, screen_y, _, idx in positions:
            if idx == entity_info.entity_idx:
                return (screen_x, screen_y)
        return None

    def _format_position(self, coords: np.ndarray) -> str | None:
        """Format position coordinates. Returns None if dimensions > 3."""
        if len(coords) > 3:
            return None
        return "(" + ", ".join(f"{c:.2f}" for c in coords) + ")"

    def _draw_tooltip(self, entity_info: EntityInfo) -> None:
        """Draw a tooltip near the hovered entity."""
        assert self._screen is not None and self._small_font is not None
        lines: list[str] = []

        if entity_info.entity_type == "seller":
            assert (
                self._cached_seller_positions is not None
                and self._cached_seller_prices is not None
                and self._cached_seller_running_sales is not None
            )
            seller_idx = entity_info.entity_idx
            lines.append(f"seller_{seller_idx}")
            pos_str = self._format_position(self._seller_space_coords(seller_idx))
            if pos_str:
                lines.append(f"Position: {pos_str}")
            lines.append(f"Price: {float(self._cached_seller_prices[seller_idx]):.2f}")
            if self._env.include_quality:
                assert self._cached_seller_qualities is not None
                lines.append(f"Quality: {float(self._cached_seller_qualities[seller_idx]):.2f}")
            lines.append(f"Sales: {float(self._cached_seller_running_sales[seller_idx]):.1f}")
        else:
            assert self._cached_buyer_positions is not None and self._cached_buyer_purchased_from is not None
            buyer_idx = entity_info.entity_idx
            lines.append("Buyer")
            coords = self._cached_buyer_positions[buyer_idx].astype(np.float64) / self._env.space_resolution
            pos_str = self._format_position(coords)
            if pos_str:
                lines.append(f"Position: {pos_str}")
            purchased_from = int(self._cached_buyer_purchased_from[buyer_idx])
            if purchased_from >= 0:
                lines.append(f"Bought from: seller_{purchased_from}")

        # Use original screen position (box stays in place while highlight follows entity)
        self._draw_info_box(entity_info.screen_pos, lines, entity_info.color)

    def _draw_detail_panel(self, entity_info: EntityInfo) -> None:
        """Draw a detailed info panel for the selected entity."""
        assert self._screen is not None and self._font is not None
        lines: list[str] = []

        if entity_info.entity_type == "seller":
            assert (
                self._cached_seller_positions is not None
                and self._cached_seller_prices is not None
                and self._cached_seller_running_sales is not None
            )
            seller_idx = entity_info.entity_idx
            lines.append(f"=== seller_{seller_idx} ===")
            lines.append("")
            pos_str = self._format_position(self._seller_space_coords(seller_idx))
            if pos_str:
                lines.append(f"Position: {pos_str}")
            lines.append(f"Price: {float(self._cached_seller_prices[seller_idx]):.3f}")
            if self._env.include_quality:
                assert self._cached_seller_qualities is not None
                lines.append(f"Quality: {float(self._cached_seller_qualities[seller_idx]):.3f}")
            lines.append("")
            lines.append(f"Running sales: {float(self._cached_seller_running_sales[seller_idx]):.1f}")
            reward = self._cumulative_rewards.get(f"seller_{seller_idx}", 0.0)
            lines.append(f"Cumulative reward: {reward:.2f}")
        else:
            assert self._cached_buyer_positions is not None and self._cached_buyer_purchased_from is not None
            buyer_idx = entity_info.entity_idx
            lines.append("=== Buyer ===")
            lines.append("")
            coords = self._cached_buyer_positions[buyer_idx].astype(np.float64) / self._env.space_resolution
            pos_str = self._format_position(coords)
            if pos_str:
                lines.append(f"Position: {pos_str}")
            purchased_from = int(self._cached_buyer_purchased_from[buyer_idx])
            if purchased_from >= 0:
                lines.append(f"Purchased from: seller_{purchased_from}")
            else:
                lines.append("Has not purchased")

        # Draw panel on the bottom left
        _, window_height = self._window_size
        line_height = 18
        padding = 15

        # Calculate panel dimensions based on content
        max_text_width = max(self._font.size(line)[0] for line in lines) if lines else 100
        panel_width = max_text_width + padding * 2
        panel_height = len(lines) * line_height + padding * 2
        panel_x = 10
        panel_y = window_height - panel_height - 10

        # Draw panel background
        t = self._theme
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self._screen, t["panel_bg"], panel_rect, border_radius=5)
        pygame.draw.rect(self._screen, entity_info.color, panel_rect, 2, border_radius=5)

        # Draw lines
        for i, line in enumerate(lines):
            text_color = entity_info.color if i == 0 else t["text"]
            text = self._font.render(line, True, text_color)
            self._screen.blit(text, (panel_x + padding, panel_y + padding + i * line_height))

    def _draw_info_box(
        self,
        anchor_pos: tuple[int, int],
        lines: list[str],
        border_color: tuple[int, int, int],
    ) -> None:
        """Draw an info box near the given anchor position."""
        assert self._screen is not None and self._small_font is not None
        if not lines:
            return

        # Calculate box dimensions
        line_height = 16
        padding = 8
        max_width = max(self._small_font.size(line)[0] for line in lines)
        box_width = max_width + padding * 2
        box_height = len(lines) * line_height + padding * 2

        # Position box (offset from anchor, keep on screen)
        window_width, window_height = self._window_size
        box_x = anchor_pos[0] + 20
        box_y = anchor_pos[1] - box_height // 2

        # Keep on screen
        if box_x + box_width > window_width - 10:
            box_x = anchor_pos[0] - box_width - 20
        box_y = max(10, min(box_y, window_height - box_height - 10))

        # Draw box
        t = self._theme
        box_rect = pygame.Rect(box_x, box_y, box_width, box_height)
        pygame.draw.rect(self._screen, t["info_box_bg"], box_rect, border_radius=4)
        pygame.draw.rect(self._screen, border_color, box_rect, 1, border_radius=4)

        # Draw text
        for i, line in enumerate(lines):
            text = self._small_font.render(line, True, t["text"])
            self._screen.blit(text, (box_x + padding, box_y + padding + i * line_height))
