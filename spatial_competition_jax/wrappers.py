"""JaxMARL-compatible wrapper for the spatial competition environment.

Converts between the stacked-array interface of
:class:`SpatialCompetitionEnv` and JaxMARL's per-agent dictionary
interface.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from spatial_competition_jax.env import EnvState, SpatialCompetitionEnv


class JaxMARLWrapper:
    """Thin adapter that makes the environment pluggable into JaxMARL's
    MAPPO (or any other) training loop.

    JaxMARL expects:

    * ``reset(key) → (obs_dict, state)``
    * ``step(key, state, actions_dict) → (obs_dict, state, rewards_dict, dones_dict, info)``

    where every ``*_dict`` is ``{agent_name: value}``.
    """

    def __init__(self, env: SpatialCompetitionEnv) -> None:
        self._env = env
        self.num_agents: int = env.num_sellers
        self.agents: list[str] = [f"seller_{i}" for i in range(env.num_sellers)]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(
        self,
        key: jnp.ndarray,
    ) -> tuple[dict[str, Any], EnvState]:
        """Reset and return per-agent observation dicts."""
        obs, state = self._env.reset(key)
        obs_dict = {agent: jax.tree.map(lambda x: x[i], obs) for i, agent in enumerate(self.agents)}
        return obs_dict, state

    def step(
        self,
        key: jnp.ndarray,
        state: EnvState,
        actions: dict[str, dict[str, jnp.ndarray]],
    ) -> tuple[dict[str, Any], EnvState, dict[str, jnp.ndarray], dict[str, jnp.ndarray], dict]:
        """Step with per-agent action dicts; return per-agent output dicts."""
        # Stack per-agent actions into arrays
        stacked: dict[str, jnp.ndarray] = {
            "movement": jnp.stack([actions[a]["movement"] for a in self.agents]),
            "price": jnp.stack([actions[a]["price"] for a in self.agents]),
        }
        if self._env.include_quality:
            stacked["quality"] = jnp.stack([actions[a]["quality"] for a in self.agents])

        obs, state, rewards, dones, info = self._env.step(key, state, stacked)

        # Split back into per-agent dicts
        obs_dict = {agent: jax.tree.map(lambda x: x[i], obs) for i, agent in enumerate(self.agents)}
        rewards_dict: dict[str, jnp.ndarray] = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        dones_dict: dict[str, jnp.ndarray] = {agent: dones[i] for i, agent in enumerate(self.agents)}
        dones_dict["__all__"] = jnp.all(dones)

        return obs_dict, state, rewards_dict, dones_dict, info

    # ------------------------------------------------------------------
    # Space descriptions
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> dict[str, Any]:  # noqa: ARG002
        """Return a dict describing the observation space for *agent*."""
        D = self._env.dimensions
        R = self._env.space_resolution
        grid_shape = tuple([R] * D)

        space: dict[str, Any] = {
            "own_position": {"shape": (D,), "dtype": "float32", "low": 0.0, "high": 1.0},
            "own_price": {"shape": (), "dtype": "float32", "low": 0.0, "high": self._env.max_price},
            "local_view": {"shape": (3,) + grid_shape, "dtype": "uint8", "low": 0, "high": 1},
        }
        if self._env.include_quality:
            space["own_quality"] = {
                "shape": (),
                "dtype": "float32",
                "low": 0.0,
                "high": self._env.max_quality,
            }
        if self._env.information_level >= 1:
            space["buyers"] = {"shape": (3,) + grid_shape, "dtype": "float32"}
        if self._env.information_level >= 2:
            space["sellers_price"] = {"shape": grid_shape, "dtype": "float32"}
            space["sellers_quality"] = {"shape": grid_shape, "dtype": "float32"}

        return space

    def action_space(self, agent: str) -> dict[str, Any]:  # noqa: ARG002
        """Return a dict describing the action space for *agent*."""
        D = self._env.dimensions
        space: dict[str, Any] = {
            "movement": {
                "shape": (D,),
                "dtype": "float32",
                "low": -1.0,
                "high": 1.0,
            },
            "price": {
                "shape": (),
                "dtype": "float32",
                "low": 0.0,
                "high": self._env.max_price,
            },
        }
        if self._env.include_quality:
            space["quality"] = {
                "shape": (),
                "dtype": "float32",
                "low": 0.0,
                "high": self._env.max_quality,
            }
        return space
