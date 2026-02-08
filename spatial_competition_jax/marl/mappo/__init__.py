"""MAPPO algorithm implementation."""

from spatial_competition_jax.marl.mappo.mappo import MAPPO, compute_temperature, linear_anneal
from spatial_competition_jax.marl.mappo.networks import SharedActorCritic

__all__ = ["MAPPO", "SharedActorCritic", "compute_temperature", "linear_anneal"]
