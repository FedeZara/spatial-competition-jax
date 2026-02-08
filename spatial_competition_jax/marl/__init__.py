"""MARL training module for spatial competition."""

from spatial_competition_jax.marl.config import Config, EnvConfig, TrainConfig
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper

__all__ = ["Config", "EnvConfig", "TrainConfig", "TrainingWrapper"]
