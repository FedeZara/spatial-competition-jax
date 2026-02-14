#!/usr/bin/env python3
"""Run the symmetric PSRO loop for Hotelling spatial competition (JAX).

Finds approximate Nash equilibria by iteratively building a policy
population, solving the meta-game, and training best-response oracles.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import jax

from spatial_competition_jax.marl.config import Config
from spatial_competition_jax.marl.psro.psro import PSROLoop
from spatial_competition_jax.marl.utils.device import resolve_device
from spatial_competition_jax.marl.utils.logging import Logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run symmetric PSRO on Hotelling spatial competition (JAX)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/psro_hotelling_1d.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Override number of PSRO iterations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Override log directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "JAX device to use, e.g. 'cpu', 'gpu', 'gpu:0', 'tpu'. "
            "Defaults to JAX default (first available accelerator)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the PSRO loop."""
    args = parse_args()

    # ── device selection ──────────────────────────────────────────────
    device = resolve_device(args.device)

    # ── config ────────────────────────────────────────────────────────
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = Config.from_yaml(config_path)
    else:
        print(f"Config not found: {config_path}, using defaults")
        config = Config()

    if args.seed is not None:
        config.train.seed = args.seed
    if args.log_dir is not None:
        config.train.log_dir = args.log_dir
    if args.no_tensorboard:
        config.train.use_tensorboard = False

    # ── experiment name ───────────────────────────────────────────────
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        action_tag = "disc" if config.env.action_type == "discrete" else "cont"
        obs_tag = "ego" if config.train.observation_mode == "egocentric" else "glob"
        experiment_name = f"psro_{config.env.dimensions}d_{action_tag}_{obs_tag}_{timestamp}"
    else:
        experiment_name = args.experiment_name

    # ── logger ────────────────────────────────────────────────────────
    logger = Logger(
        log_dir=config.train.log_dir,
        use_tensorboard=config.train.use_tensorboard,
        experiment_name=experiment_name,
    )

    print("=" * 60)
    print("Symmetric PSRO (JAX)")
    print("=" * 60)
    print(f"Experiment:    {experiment_name}")
    print(f"Device:        {device}")
    print(f"Backend:       {jax.default_backend()}")
    print(f"Obs mode:      {config.train.observation_mode}")
    print(f"Action type:   {config.env.action_type}")
    print(f"PSRO iters:    {args.num_iterations or config.psro.num_psro_iterations}")
    print(f"BR updates:    {config.psro.num_br_updates}")
    print(f"Eval episodes: {config.psro.num_eval_episodes}")
    print(f"Num envs:      {config.train.num_envs}")
    print(f"Rollout:       {config.train.rollout_length}")
    print("=" * 60)

    with jax.default_device(device):
        psro = PSROLoop(config=config, logger=logger)

        try:
            results = psro.run(num_iterations=args.num_iterations)
        except KeyboardInterrupt:
            print("\n\nPSRO interrupted — saving checkpoint …")
            results = {
                "population": psro.population,
                "meta_strategy": psro.meta_strategy,
                "payoff_matrix": (
                    psro.payoff_table.matrix
                    if psro.payoff_table._size > 0
                    else None
                ),
            }

            # Save interrupted checkpoint
            import pickle

            ckpt_path = logger.experiment_dir / "psro_interrupted.pkl"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ckpt_path, "wb") as f:
                pickle.dump(
                    {
                        "population": psro.population,
                        "meta_strategy": psro.meta_strategy,
                        "payoff_matrix": results["payoff_matrix"],
                        "exploitability_history": [],
                    },
                    f,
                )
            print(f"Saved: {ckpt_path}")
            print(f"Population size: {len(psro.population)}")

    # ── summary ───────────────────────────────────────────────────────
    if "exploitability_history" in results:
        hist = results["exploitability_history"]
        if hist:
            print(f"\nExploitability trajectory: "
                  f"{[f'{x:.4f}' for x in hist]}")

    logger.close()


if __name__ == "__main__":
    main()
