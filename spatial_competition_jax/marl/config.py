"""Configuration management for MAPPO and PSRO training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    """Environment configuration."""

    dimensions: int = 1
    num_sellers: int = 2
    max_buyers: int = 200
    max_price: float = 10.0
    max_quality: float = 5.0
    max_step_size: float = 0.02
    production_cost_factor: float = 0.1
    movement_cost: float = 0.0
    transport_cost: float = 2.0
    transportation_cost_norm: float = 2.0
    transport_cost_exponent: float = 1.0
    quality_taste: float = 0.0
    include_quality: bool = False
    new_buyers_per_step: int = 50
    max_env_steps: int = 200
    space_resolution: int = 100
    buyer_choice_temperature: float | None = None

    # Discrete action space (used when action_type="discrete")
    action_type: str = "continuous"  # "continuous" or "discrete"
    num_location_bins: int = 10
    num_price_bins: int = 10


@dataclass
class TrainConfig:
    """Training configuration."""

    # Training
    num_envs: int = 16
    rollout_length: int = 512
    total_updates: int = 2000
    seed: int = 42

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 6
    num_minibatches: int = 8
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    target_kl: float | None = None  # KL-based early stopping (None = disabled)

    # Observation mode: "global" (old per-agent heads) or "egocentric"
    observation_mode: str = "global"

    # Independent PPO: append one-hot agent ID to ego observations.
    # Breaks weight sharing so each agent can learn a different strategy.
    independent: bool = False

    # Optimization
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5

    # Network architecture
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])

    # Observation type: "blob" (Gaussian-smoothed spatial maps) or
    # "bin" (raw bin/grid observations from the environment).
    obs_type: str = "blob"

    # Gaussian blob observation encoding (only used when obs_type="blob")
    blob_sigma: float = 1.5

    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 200
    use_tensorboard: bool = True
    log_dir: str = "results"

    # Evaluation
    eval_episodes: int = 10
    deterministic_eval: bool = True

    # Entropy coefficient decay
    entropy_coef_start: float | None = None
    entropy_coef_end: float = 0.0
    entropy_coef_anneal_frac: float = 1.0

    # Buyer-choice temperature annealing (used when env has softmax)
    buyer_choice_temp_start: float | None = None
    buyer_choice_temp_end: float = 0.001
    buyer_choice_temp_anneal_frac: float = 0.8


@dataclass
class PSROConfig:
    """PSRO-specific configuration.

    Controls the outer PSRO loop: how many iterations to run, how many
    PPO updates per best-response oracle, payoff-matrix evaluation
    budget, and warm-starting behaviour.
    """

    # Outer loop
    num_psro_iterations: int = 10
    num_br_updates: int = 50000
    num_eval_episodes: int = 50
    num_initial_policies: int = 1

    # Warm-starting best-response from population
    warmstart_br: bool = True

    # BR opponent mixing: blend Nash meta-strategy with uniform.
    #   opponent_mix = alpha * uniform + (1 - alpha) * nash
    # Anneals from br_mix_alpha_start → br_mix_alpha_end over the
    # first br_mix_alpha_anneal_frac fraction of PSRO iterations.
    br_mix_alpha_start: float = 0.5
    br_mix_alpha_end: float = 0.1
    br_mix_alpha_anneal_frac: float = 0.7

    # BR early stopping: stop when mean reward hasn't improved by
    # br_early_stop_delta over the last br_patience evaluation windows.
    # Set br_patience = 0 to disable early stopping.
    br_patience: int = 20
    br_early_stop_delta: float = 0.05

    # Logging
    log_interval: int = 50
    save_interval: int = 1

    # Evaluation temperature override (for softmax buyer choice)
    eval_temperature: float | None = None


@dataclass
class Config:
    """Complete configuration."""

    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    psro: PSROConfig = field(default_factory=PSROConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        # Handle parent config inheritance
        if "_parent" in data:
            parent_path = path.parent / data.pop("_parent")
            parent_config = cls.from_yaml(parent_path)
            parent_dict = _config_to_dict(parent_config)
            _deep_update(parent_dict, data)
            data = parent_dict

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create configuration from dictionary."""
        env_data = {}
        train_data = {}
        psro_data = {}

        env_fields = {f.name for f in EnvConfig.__dataclass_fields__.values()}
        train_fields = {f.name for f in TrainConfig.__dataclass_fields__.values()}
        psro_fields = {f.name for f in PSROConfig.__dataclass_fields__.values()}

        for key, value in data.items():
            if key in env_fields:
                env_data[key] = value
            elif key in train_fields:
                train_data[key] = value
            elif key in psro_fields:
                psro_data[key] = value

        return cls(
            env=EnvConfig(**env_data),
            train=TrainConfig(**train_data),
            psro=PSROConfig(**psro_data),
        )


def _config_to_dict(config: Config) -> dict[str, Any]:
    """Convert config to flat dictionary."""
    result: dict[str, Any] = {}
    for key, value in config.env.__dict__.items():
        result[key] = value
    for key, value in config.train.__dict__.items():
        result[key] = value
    for key, value in config.psro.__dict__.items():
        result[key] = value
    return result


def _deep_update(base: dict, update: dict) -> None:
    """Deep update base dict with update dict."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
