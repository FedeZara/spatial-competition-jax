"""MAPPO algorithm implementation."""

from spatial_competition_jax.marl.mappo.mappo import MAPPO, compute_temperature, linear_anneal
from spatial_competition_jax.marl.mappo.networks import DiscreteActorCritic, SharedActorCritic
from spatial_competition_jax.marl.mappo.policy import ContinuousPolicy, DiscretePolicy, PolicyAdapter

__all__ = [
    "MAPPO",
    "ContinuousPolicy",
    "DiscreteActorCritic",
    "DiscretePolicy",
    "PolicyAdapter",
    "SharedActorCritic",
    "compute_temperature",
    "linear_anneal",
]
