"""MAPPO algorithm implementation."""

from spatial_competition_jax.marl.mappo.mappo import MAPPO, compute_temperature, linear_anneal
from spatial_competition_jax.marl.mappo.networks import (
    DiscreteActorCritic,
    EgoConv1dFactoredDiscreteActorCritic,
    EgoFactoredDiscreteActorCritic,
    SharedActorCritic,
)
from spatial_competition_jax.marl.mappo.policy import (
    ContinuousPolicy,
    DiscretePolicy,
    EgoFactoredDiscretePolicy,
    PolicyAdapter,
)

__all__ = [
    "MAPPO",
    "ContinuousPolicy",
    "DiscreteActorCritic",
    "DiscretePolicy",
    "EgoConv1dFactoredDiscreteActorCritic",
    "EgoFactoredDiscreteActorCritic",
    "EgoFactoredDiscretePolicy",
    "PolicyAdapter",
    "SharedActorCritic",
    "compute_temperature",
    "linear_anneal",
]
