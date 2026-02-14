"""MARL training module for spatial competition."""

from spatial_competition_jax.marl.config import Config, EnvConfig, PSROConfig, TrainConfig
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper

__all__ = ["Config", "EnvConfig", "PSROConfig", "TrainConfig", "TrainingWrapper"]
