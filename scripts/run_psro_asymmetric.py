#!/usr/bin/env python3
"""Run the asymmetric (two-population) PSRO loop (JAX).

Each player maintains a separate population of policies.  Best
responses are trained for both players at each iteration.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import jax

from spatial_competition_jax.marl.config import Config
from spatial_competition_jax.marl.psro.psro_asymmetric import AsymmetricPSROLoop
from spatial_competition_jax.marl.utils.device import resolve_device
from spatial_competition_jax.marl.utils.logging import Logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run asymmetric PSRO on Hotelling spatial competition (JAX)",
    )
    parser.add_argument("--config", type=str, default="configs/psro_hotelling_1d_discrete.yaml")
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

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

    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        action_tag = "disc" if config.env.action_type == "discrete" else "cont"
        experiment_name = f"psro_asym_{config.env.dimensions}d_{action_tag}_{timestamp}"
    else:
        experiment_name = args.experiment_name

    logger = Logger(
        log_dir=config.train.log_dir,
        use_tensorboard=config.train.use_tensorboard,
        experiment_name=experiment_name,
    )

    print("=" * 60)
    print("Asymmetric PSRO (JAX) — Two Populations")
    print("=" * 60)
    print(f"Experiment:    {experiment_name}")
    print(f"Device:        {device}")
    print(f"Backend:       {jax.default_backend()}")
    print(f"Obs mode:      {config.train.observation_mode}")
    print(f"Action type:   {config.env.action_type}")
    print(f"PSRO iters:    {args.num_iterations or config.psro.num_psro_iterations}")
    print(f"BR updates:    {config.psro.num_br_updates}")
    print("=" * 60)

    with jax.default_device(device):
        psro = AsymmetricPSROLoop(config=config, logger=logger)

        try:
            results = psro.run(num_iterations=args.num_iterations)
        except KeyboardInterrupt:
            print("\n\nPSRO interrupted — saving checkpoint …")
            psro._save_checkpoint(0, [], final=True)
            results = {
                "pop0": psro.pop0,
                "pop1": psro.pop1,
                "sigma0": psro.sigma0,
                "sigma1": psro.sigma1,
            }

    if "exploitability_history" in results:
        hist = results["exploitability_history"]
        if hist:
            print(f"\nExploitability: {[f'{x:.4f}' for x in hist]}")

    logger.close()


if __name__ == "__main__":
    main()
