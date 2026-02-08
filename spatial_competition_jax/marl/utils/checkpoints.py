"""Checkpoint saving and loading utilities using pickle."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_checkpoint(
    path: str | Path,
    step: int,
    params: Any,
    opt_state: Any | None = None,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint.

    Args:
        path: Path to save checkpoint to.
        step: Current training step.
        params: Network parameters (pytree).
        opt_state: Optimizer state (pytree).
        config: Optional configuration dictionary.
        metrics: Optional metrics dictionary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "step": step,
        "params": params,
    }

    if opt_state is not None:
        checkpoint["opt_state"] = opt_state
    if config is not None:
        checkpoint["config"] = config
    if metrics is not None:
        checkpoint["metrics"] = metrics

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Path to checkpoint file.

    Returns:
        Dictionary containing checkpoint data with keys:
        - step: Training step
        - params: Network parameters
        - opt_state: Optimizer state (if saved)
        - config: Optional configuration
        - metrics: Optional metrics
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, "rb") as f:
        result: dict[str, Any] = pickle.load(f)  # noqa: S301
        return result
