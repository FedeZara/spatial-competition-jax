"""Shared policy builder used by all training / evaluation scripts.

Single source of truth for the mapping:

    (config, wrapper) → PolicyAdapter
"""

from __future__ import annotations

from spatial_competition_jax.marl.config import Config
from spatial_competition_jax.marl.mappo.networks import (
    DiscreteActorCritic,
    EgoActorCritic,
    EgoConv1dFactoredDiscreteActorCritic,
    EgoConv2dActorCritic,
    EgoConv2dFactoredDiscreteActorCritic,
    EgoDiscreteActorCritic,
    EgoFactoredDiscreteActorCritic,
    SharedActorCritic,
)
from spatial_competition_jax.marl.mappo.policy import (
    ContinuousPolicy,
    DiscretePolicy,
    Ego2dFactoredDiscretePolicy,
    EgoContinuousPolicy,
    EgoDiscretePolicy,
    EgoFactoredDiscretePolicy,
    PolicyAdapter,
)
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper


def build_policy(config: Config, wrapper: TrainingWrapper) -> PolicyAdapter:
    """Build the appropriate PolicyAdapter from config.

    Handles all combos::

        {global, egocentric} × {continuous, discrete}
        × {blob, bin, conv_bin} × {1D, 2D}

    including per-agent heads when ``independent_heads=True``.
    """
    hidden_dims = tuple(config.train.hidden_dims)
    ego = config.train.observation_mode == "egocentric"
    discrete = config.env.action_type == "discrete"
    conv_bin = config.train.obs_type == "conv_bin"

    ind_heads = config.train.independent_heads and config.train.independent

    # ── 2D discrete + conv_bin → factored (loc_x × loc_y × price [× quality])
    if ego and discrete and conv_bin and wrapper.dimensions == 2:
        gp = wrapper.space_resolution + 1
        scalar_dim = wrapper.dimensions + 1
        if wrapper.include_quality:
            scalar_dim += 1
        net = EgoConv2dFactoredDiscreteActorCritic(
            num_location_bins=wrapper.num_location_bins,
            num_price_bins=wrapper.num_price_bins,
            num_quality_bins=wrapper.num_quality_bins,
            spatial_resolution=gp,
            num_grid_channels=wrapper._conv_grid_channels,
            num_scalar_features=scalar_dim,
            mlp_hidden_dims=hidden_dims,
            independent_heads=ind_heads,
            num_agents=wrapper.num_agents,
        )
        return Ego2dFactoredDiscretePolicy(net, num_agents=wrapper.num_agents)

    # ── 1D discrete + conv_bin → factored (loc × price [× quality])
    if ego and discrete and conv_bin:
        gp = wrapper.space_resolution + 1
        scalar_dim = wrapper.dimensions + 1
        if wrapper.include_quality:
            scalar_dim += 1
        net = EgoConv1dFactoredDiscreteActorCritic(
            num_location_bins=wrapper.num_location_bins,
            num_price_bins=wrapper.num_price_bins,
            num_quality_bins=wrapper.num_quality_bins,
            spatial_resolution=gp,
            num_grid_channels=wrapper._conv_grid_channels,
            num_scalar_features=scalar_dim,
            mlp_hidden_dims=hidden_dims,
            independent_heads=ind_heads,
            num_agents=wrapper.num_agents,
        )
        return EgoFactoredDiscretePolicy(net, num_agents=wrapper.num_agents)

    # ── 2D continuous + conv_bin
    if ego and not discrete and conv_bin:
        gp = wrapper.space_resolution + 1
        scalar_dim = wrapper.dimensions + 1
        if wrapper.include_quality:
            scalar_dim += 1
        net = EgoConv2dActorCritic(
            movement_dim=wrapper.movement_dim,
            bounded_dim=wrapper.bounded_dim,
            spatial_resolution=gp,
            num_grid_channels=wrapper._conv_grid_channels,
            num_scalar_features=scalar_dim,
            mlp_hidden_dims=hidden_dims,
            independent_heads=ind_heads,
            num_agents=wrapper.num_agents,
        )
        return EgoContinuousPolicy(net, num_agents=wrapper.num_agents)

    # ── Ego discrete (MLP, no conv)
    if ego and discrete:
        net = EgoFactoredDiscreteActorCritic(
            num_location_bins=wrapper.num_location_bins,
            num_price_bins=wrapper.num_price_bins,
            num_quality_bins=wrapper.num_quality_bins,
            hidden_dims=hidden_dims,
        )
        return EgoFactoredDiscretePolicy(net, num_agents=wrapper.num_agents)

    # ── Ego continuous (MLP, no conv)
    if ego:
        net = EgoActorCritic(
            movement_dim=wrapper.movement_dim,
            bounded_dim=wrapper.bounded_dim,
            hidden_dims=hidden_dims,
        )
        return EgoContinuousPolicy(net, num_agents=wrapper.num_agents)

    # ── Global discrete
    if discrete:
        net = DiscreteActorCritic(
            num_actions=wrapper.num_actions,
            num_agents=wrapper.num_agents,
            hidden_dims=hidden_dims,
        )
        return DiscretePolicy(net)

    # ── Global continuous
    net = SharedActorCritic(
        movement_dim=wrapper.movement_dim,
        bounded_dim=wrapper.bounded_dim,
        num_agents=wrapper.num_agents,
        hidden_dims=hidden_dims,
    )
    return ContinuousPolicy(net)
