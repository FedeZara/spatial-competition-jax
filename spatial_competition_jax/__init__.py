"""JAX-native Spatial Competition Environment."""

from spatial_competition_jax.env import (
    INFO_COMPLETE,
    INFO_LIMITED,
    INFO_PRIVATE,
    TOPOLOGY_RECTANGLE,
    TOPOLOGY_TORUS,
    EnvState,
    SpatialCompetitionEnv,
    make_constant_sampler,
    make_normal_position_sampler,
    make_normal_sampler,
    make_uniform_sampler,
    uniform_position_sampler,
)
from spatial_competition_jax.wrappers import JaxMARLWrapper

__all__ = [
    # Core
    "SpatialCompetitionEnv",
    "EnvState",
    # Wrapper
    "JaxMARLWrapper",
    # Constants
    "TOPOLOGY_RECTANGLE",
    "TOPOLOGY_TORUS",
    "INFO_PRIVATE",
    "INFO_LIMITED",
    "INFO_COMPLETE",
    # Sampler factories
    "uniform_position_sampler",
    "make_constant_sampler",
    "make_uniform_sampler",
    "make_normal_sampler",
    "make_normal_position_sampler",
]


def __getattr__(name: str) -> type:
    if name == "SpatialCompetitionRenderer":
        from spatial_competition_jax.renderer import SpatialCompetitionRenderer

        return SpatialCompetitionRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
