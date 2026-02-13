"""Policy Space Response Oracles (PSRO) for spatial competition."""

from spatial_competition_jax.marl.config import PSROConfig
from spatial_competition_jax.marl.psro.best_response import BestResponseTrainer
from spatial_competition_jax.marl.psro.meta_solver import (
    compute_exploitability,
    projected_replicator_dynamics,
)
from spatial_competition_jax.marl.psro.payoff_table import PayoffTable
from spatial_competition_jax.marl.psro.psro import PSROLoop
from spatial_competition_jax.marl.psro.state_utils import permute_agent_state

__all__ = [
    "BestResponseTrainer",
    "PSROConfig",
    "PSROLoop",
    "PayoffTable",
    "compute_exploitability",
    "permute_agent_state",
    "projected_replicator_dynamics",
]
