"""Logging utilities for training."""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


class Logger:
    """Simple logger with optional TensorBoard support."""

    def __init__(
        self,
        log_dir: str | Path,
        use_tensorboard: bool = True,
        experiment_name: str | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.use_tensorboard = use_tensorboard

        # Create experiment directory
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = None
        if use_tensorboard:
            try:
                from tensorboardX import SummaryWriter

                self.writer = SummaryWriter(logdir=str(self.experiment_dir / "tensorboard"))
            except ImportError:
                print("tensorboardX not available, logging to console only")
                self.use_tensorboard = False

        # Metric buffers for averaging
        self._buffers: dict[str, list[float]] = defaultdict(list)
        self._step = 0

    def log_scalar(self, tag: str, value: float, step: int | None = None) -> None:
        """Log a scalar value."""
        if step is None:
            step = self._step

        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], step: int | None = None) -> None:
        """Log multiple scalars under a main tag."""
        if step is None:
            step = self._step

        for tag, value in tag_scalar_dict.items():
            self.log_scalar(f"{main_tag}/{tag}", value, step)

    def buffer_scalar(self, tag: str, value: float) -> None:
        """Buffer a scalar value for later averaging."""
        self._buffers[tag].append(value)

    def flush_buffers(self, step: int | None = None) -> dict[str, float]:
        """Flush buffered scalars by computing means and logging."""
        if step is None:
            step = self._step

        means: dict[str, float] = {}
        for tag, values in self._buffers.items():
            if values:
                mean = sum(values) / len(values)
                means[tag] = mean
                self.log_scalar(tag, mean, step)

        self._buffers.clear()
        return means

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None, prefix: str = "") -> None:
        """Log a dictionary of metrics."""
        if step is None:
            step = self._step

        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            if isinstance(value, (int, float)):
                self.log_scalar(tag, value, step)
            elif isinstance(value, dict):
                self.log_metrics(value, step, prefix=tag)

    def print_metrics(self, metrics: dict[str, Any], step: int, prefix: str = "Train") -> None:
        """Print metrics to console."""
        parts = [f"[{prefix}] Step {step}"]
        for key, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{key}: {value:.4f}")
            else:
                parts.append(f"{key}: {value}")
        print(" | ".join(parts), file=sys.stdout)

    def set_step(self, step: int) -> None:
        """Set the current step."""
        self._step = step

    def close(self) -> None:
        """Close the logger."""
        if self.writer is not None:
            self.writer.close()
