#!/usr/bin/env python3
"""Live demo of the spatial competition environment with random agents.

No checkpoint needed — sellers take random actions. Opens a Pygame window
to watch the simulation in real time. When recording, captures all four
phases per step (remove purchased, spawn buyers, apply actions, process sales).

Usage::

    # Just watch the demo
    poetry run python scripts/demo.py

    # Watch + record a GIF (4 frames per step, one per phase)
    poetry run python scripts/demo.py --record

    # Record to a specific file
    poetry run python scripts/demo.py --record --output my_demo.gif

    # Also save MP4
    poetry run python scripts/demo.py --record --mp4

    # 1D demo
    poetry run python scripts/demo.py --dimensions 1

    # Customize
    poetry run python scripts/demo.py --sellers 4 --steps 120 --fps 30
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from spatial_competition_jax import (
    INFO_COMPLETE,
    SpatialCompetitionEnv,
)
from spatial_competition_jax.env import (
    make_mixture_position_sampler,
    make_uniform_sampler,
)
from spatial_competition_jax.renderer import SpatialCompetitionRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live demo with random agents")
    parser.add_argument("--dimensions", type=int, default=2, help="1 or 2 (default: 2)")
    parser.add_argument("--sellers", type=int, default=4, help="Number of sellers (default: 4)")
    parser.add_argument("--buyers-per-step", type=int, default=80, help="New buyers per step")
    parser.add_argument("--max-buyers", type=int, default=300, help="Max concurrent buyers")
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps to run")
    parser.add_argument("--fps", type=int, default=8, help="Target frames per second")
    parser.add_argument("--light", action="store_true", help="Light-mode rendering")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--record", action="store_true", help="Record frames and save as GIF")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: results/demo.gif)")
    parser.add_argument("--mp4", action="store_true", help="Also save as MP4 (requires imageio[ffmpeg])")
    return parser.parse_args()


# ── Recording helpers ────────────────────────────────────────────────────


def save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    """Save frames as an animated GIF using Pillow."""
    from PIL import Image

    images = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / fps)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Saved GIF: {path}  ({len(frames)} frames, {size_mb:.1f} MB)")


def save_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    """Save frames as MP4 using imageio-ffmpeg."""
    try:
        import imageio.v3 as iio
    except ImportError:
        print("  ⚠ imageio not found — skipping MP4. Install with: pip install imageio[ffmpeg]")
        return
    iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264")
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Saved MP4: {path}  ({len(frames)} frames, {size_mb:.1f} MB)")


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    # Ensure Ctrl+C actually kills the process even with pygame active
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    args = parse_args()
    recording = args.record or args.output is not None or args.mp4

    # ── Build environment ───────────────────────────────────────────────
    buyer_position_sampler = None
    if args.dimensions == 2:
        # Use a 4-component mixture of Gaussians — looks more interesting
        try:
            buyer_position_sampler = make_mixture_position_sampler(
                means=[[0.25, 0.75], [0.75, 0.25], [0.25, 0.25], [0.75, 0.75]],
                stds=[0.12, 0.12, 0.12, 0.12],
                weights=[0.25, 0.25, 0.25, 0.25],
            )
        except Exception:
            pass  # Fall back to uniform

    # Low buyer values + high distance sensitivity → many buyers can't afford to buy
    # utility = value - dist_factor * dist + quality_taste * quality - price
    buyer_value_sampler = make_uniform_sampler(0.0, 4.0)
    buyer_distance_factor_sampler = make_uniform_sampler(2.0, 6.0)

    env = SpatialCompetitionEnv(
        num_sellers=args.sellers,
        max_buyers=args.max_buyers,
        dimensions=args.dimensions,
        space_resolution=100,
        max_price=10.0,
        max_quality=5.0,
        max_step_size=0.05,
        information_level=INFO_COMPLETE,
        include_quality=True,
        include_buyer_valuation=True,
        new_buyers_per_step=args.buyers_per_step,
        max_env_steps=args.steps,
        movement_cost=0.1,
        production_cost_factor=0.1,
        transportation_cost_norm=2.0,
        despawn_no_purchase=True,
        buyer_position_sampler=buyer_position_sampler,
        buyer_value_sampler=buyer_value_sampler,
        buyer_distance_factor_sampler=buyer_distance_factor_sampler,
    )

    renderer = SpatialCompetitionRenderer(env, max_env_steps=args.steps, light_mode=args.light)

    key = jax.random.PRNGKey(args.seed)
    obs, state = env.reset(key)

    cumulative_rewards = {f"seller_{i}": 0.0 for i in range(env.num_sellers)}
    frame_delay = 1.0 / args.fps
    frames: list[np.ndarray] = []

    print("=" * 60)
    print("Live demo" + ("  +  recording" if recording else "") + "  (close the window to quit)")
    print("=" * 60)
    print(f"  Dimensions:  {args.dimensions}D")
    print(f"  Sellers:     {args.sellers}")
    print(f"  Steps:       {args.steps}")
    print(f"  FPS:         {args.fps}")
    print(f"  Light mode:  {args.light}")
    print(f"  Recording:   {recording} (4 frames/step, one per phase)")
    print()

    # Render initial frame
    if not renderer.render_and_wait(state, frame_delay, current_step=0, cumulative_rewards=cumulative_rewards):
        renderer.close()
        return
    if recording:
        frames.append(renderer.capture_frame())

    try:
        for step_i in range(1, args.steps + 1):
            key, k_act, k_step = jax.random.split(key, 3)

            # ── Random actions ──
            movement = jax.random.uniform(
                k_act,
                (env.num_sellers, env.dimensions),
                minval=-env.max_step_size,
                maxval=env.max_step_size,
            )
            price = jax.random.uniform(
                k_act,
                (env.num_sellers,),
                minval=1.0,
                maxval=env.max_price - 1.0,
            )
            quality = jax.random.uniform(
                k_act,
                (env.num_sellers,),
                minval=1.0,
                maxval=env.max_quality - 1.0,
            )
            actions = {"movement": movement, "price": price, "quality": quality}

            # ── Step all 4 phases, rendering each one (renderer captures per-phase frames) ──
            phase_frames: list[np.ndarray] = [] if recording else []
            state, rewards, dones, alive = renderer.step_and_render(
                k_step,
                state,
                actions,
                base_delay=frame_delay,
                current_step=step_i,
                cumulative_rewards=cumulative_rewards,
                record_phase_frames=phase_frames if recording else None,
            )
            if not alive:
                break

            if recording and phase_frames:
                frames.extend(phase_frames)

            for i in range(env.num_sellers):
                cumulative_rewards[f"seller_{i}"] += float(rewards[i])

            print(f"  Step {step_i:>3d}/{args.steps}", end="\r")

            if bool(jnp.all(dones)):
                print(f"\n  Environment done at step {step_i}")
                break
    finally:
        renderer.close()
        print()

    # ── Save output (if recording) ───────────────────────────────────────
    if recording and frames:
        if args.output:
            out_path = Path(args.output)
        else:
            out_dir = Path(__file__).resolve().parent.parent / "results"
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / "demo.gif"

        save_gif(frames, out_path, args.fps)

        if args.mp4:
            save_mp4(frames, out_path.with_suffix(".mp4"), args.fps)

    # ── Final summary ───────────────────────────────────────────────────
    print(f"\nFinal cumulative rewards:")
    for name, rew in sorted(cumulative_rewards.items()):
        print(f"  {name}: {rew:+.2f}")
    print()


if __name__ == "__main__":
    main()
