"""Symmetric PSRO loop for finding Nash equilibria.

Orchestrates the outer PSRO loop: payoff-matrix construction,
meta-game solution via Projected Replicator Dynamics, and
best-response training.

Supports both **egocentric** and **global** observation modes via the
:class:`PolicyAdapter` abstraction.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from spatial_competition_jax.marl.config import Config
from spatial_competition_jax.marl.mappo.mappo import linear_anneal
from spatial_competition_jax.marl.mappo.networks import (
    EgoActorCritic,
    EgoConv1dFactoredDiscreteActorCritic,
    EgoConv2dActorCritic,
    EgoDiscreteActorCritic,
    EgoFactoredDiscreteActorCritic,
    SharedActorCritic,
    DiscreteActorCritic,
)
from spatial_competition_jax.marl.mappo.policy import (
    ContinuousPolicy,
    DiscretePolicy,
    EgoContinuousPolicy,
    EgoDiscretePolicy,
    EgoFactoredDiscretePolicy,
)
from spatial_competition_jax.marl.psro.best_response import BestResponseTrainer
from spatial_competition_jax.marl.psro.meta_solver import (
    compute_exploitability,
    solve_meta_game,
)
from spatial_competition_jax.marl.psro.payoff_table import PayoffTable
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper
from spatial_competition_jax.marl.utils.logging import Logger

if TYPE_CHECKING:
    from spatial_competition_jax.marl.mappo.policy import PolicyAdapter


class PSROLoop:
    """Symmetric PSRO loop for 2-player spatial competition.

    Maintains a single population of policies (each parameterised for
    a single agent).  At each iteration:

    1. Updates the payoff matrix (incremental).
    2. Solves the symmetric meta-game via PRD.
    3. Logs exploitability.
    4. Trains a best-response policy against the meta-strategy.
    5. Adds the new policy to the population.
    6. Saves a checkpoint.

    The ``observation_mode`` in ``config.train`` determines whether
    egocentric or global observations are used.
    """

    def __init__(
        self,
        config: Config,
        logger: Logger,
        wrapper: TrainingWrapper | None = None,
    ) -> None:
        self.config = config
        self.logger = logger

        # ── environment wrapper ───────────────────────────────────────
        if wrapper is not None:
            self.wrapper = wrapper
        else:
            self.wrapper = _build_wrapper(config)

        # ── observation mode ──────────────────────────────────────────
        self.egocentric = config.train.observation_mode == "egocentric"
        self.discrete = config.env.action_type == "discrete"

        # ── single-agent policy adapter ──────────────────────────────
        # PSRO policies always have num_agents=1 (single agent that
        # plays as "agent 0" from its own perspective).
        self.policy = _build_single_agent_policy(config, self.wrapper)

        # ── population & meta-strategy ────────────────────────────────
        self.population: list[Any] = []
        self.meta_strategy: np.ndarray = np.array([], dtype=np.float64)

        # ── payoff table ──────────────────────────────────────────────
        eval_temp = config.psro.eval_temperature
        if eval_temp is None and config.env.buyer_choice_temperature is not None:
            eval_temp = config.train.buyer_choice_temp_end
        self.payoff_table = PayoffTable(
            policy=self.policy,
            wrapper=self.wrapper,
            egocentric=self.egocentric,
            num_eval_episodes=config.psro.num_eval_episodes,
            temperature=eval_temp,
            seed=config.train.seed,
        )

        # ── annealing flags ───────────────────────────────────────────
        self._use_temp_anneal = (
            config.env.buyer_choice_temperature is not None
            and config.train.buyer_choice_temp_start is not None
        )
        self._use_entropy_anneal = config.train.entropy_coef_start is not None

    # ------------------------------------------------------------------
    # Population seeding
    # ------------------------------------------------------------------

    def seed_population(
        self,
        seed: int | None = None,
        num_seeds: int = 1,
    ) -> None:
        """Add initial random policies to the population.

        Args:
            seed: Base random seed.
            num_seeds: Number of diverse random policies to add.
                More seeds give the meta-game a richer starting
                point, helping PRD find mixed strategies earlier.
        """
        if seed is None:
            seed = self.config.train.seed

        if self.egocentric:
            dummy = jnp.zeros(self.wrapper.obs_dim)
        else:
            dummy = jnp.zeros(self.wrapper.state_dim)

        for i in range(num_seeds):
            key = jax.random.PRNGKey(seed + i)
            params = self.policy.init(key, dummy)
            self.population.append(params)

        K = len(self.population)
        self.meta_strategy = np.ones(K, dtype=np.float64) / K
        print(f"Seeded population with {num_seeds} random policies (base seed={seed})")

    def add_pretrained(self, checkpoint_path: str | Path) -> None:
        """Add a pretrained single-agent checkpoint to the population."""
        path = Path(checkpoint_path)
        with open(path, "rb") as f:
            ckpt: dict[str, Any] = pickle.load(f)  # noqa: S301
        params = ckpt["params"]
        self.population.append(params)
        K = len(self.population)
        self.meta_strategy = np.ones(K, dtype=np.float64) / K
        print(f"Loaded pretrained policy from {path} (population size: {K})")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, num_iterations: int | None = None) -> dict[str, Any]:
        """Run the full PSRO loop.

        Args:
            num_iterations: Number of PSRO iterations.  Defaults to
                ``config.psro.num_psro_iterations``.

        Returns:
            Dictionary with final results: population params,
            meta-strategy, payoff matrix, exploitability history.
        """
        if num_iterations is None:
            num_iterations = self.config.psro.num_psro_iterations

        if not self.population:
            self.seed_population(
                num_seeds=self.config.psro.num_initial_policies,
            )

        exploitability_history: list[float] = []

        print("=" * 60)
        print("PSRO Loop")
        print("=" * 60)
        print(f"Iterations:       {num_iterations}")
        print(f"BR updates/iter:  {self.config.psro.num_br_updates}")
        print(f"Eval episodes:    {self.config.psro.num_eval_episodes}")
        print(f"Population size:  {len(self.population)}")
        print(f"Obs mode:         {'egocentric' if self.egocentric else 'global'}")
        print(f"Action type:      {self.config.env.action_type}")
        print("=" * 60)

        for iteration in range(1, num_iterations + 1):
            print(f"\n{'─' * 60}")
            print(f"PSRO Iteration {iteration}/{num_iterations}")
            print(f"{'─' * 60}")

            # ── 1. Update payoff matrix ───────────────────────────────
            print("  Building payoff matrix …")
            payoff_matrix = self.payoff_table.update(self.population)

            # ── 2. Solve meta-game ────────────────────────────────────
            self.meta_strategy = solve_meta_game(payoff_matrix)
            exploit = compute_exploitability(payoff_matrix, self.meta_strategy)
            exploitability_history.append(exploit)

            meta_entropy = -float(np.sum(
                self.meta_strategy * np.log(self.meta_strategy + 1e-12)
            ))
            diversity = _population_diversity(self.population)
            print(f"  σ: {_fmt_nonzero(self.meta_strategy)}")
            print(f"  exploit={exploit:.6f}  ent={meta_entropy:.4f}  div={diversity:.4f}")

            self.logger.set_step(iteration)
            self.logger.log_metrics(
                {
                    "exploitability": exploit,
                    "population_size": len(self.population),
                    "meta_entropy": meta_entropy,
                    "population_diversity": diversity,
                },
                prefix="psro",
            )

            # ── 3. Train best response ────────────────────────────────
            print("  Training best response …")
            new_params = self._train_best_response(iteration)

            # ── Guard: reject NaN / unstable parameters ─────────────
            if _params_contain_nan(new_params):
                print("  ⚠ BR produced unstable params — skipping this iteration.")
                continue

            # ── 4. Add to population ──────────────────────────────────
            self.population.append(new_params)
            K = len(self.population)
            # Extend meta-strategy with a small initial weight for the
            # new policy; it will be properly recomputed by PRD next
            # iteration.
            ext = np.append(self.meta_strategy, 1.0 / K)
            self.meta_strategy = ext / ext.sum()
            print(f"  Population size: {K}")

            # ── 5. Save checkpoint ────────────────────────────────────
            if iteration % self.config.psro.psro_save_interval == 0:
                self._save_checkpoint(iteration, exploitability_history)

        # ── Final solve after adding the last policy ──────────────────
        print(f"\n{'─' * 60}")
        print("Final payoff matrix & meta-solve")
        print(f"{'─' * 60}")
        payoff_matrix = self.payoff_table.update(self.population)
        self.meta_strategy = solve_meta_game(payoff_matrix)
        final_exploit = compute_exploitability(payoff_matrix, self.meta_strategy)
        exploitability_history.append(final_exploit)

        print(f"  σ: {_fmt_nonzero(self.meta_strategy)}")
        print(f"  Exploitability: {final_exploit:.6f}")

        self._save_checkpoint(num_iterations, exploitability_history, final=True)

        print("\n" + "=" * 60)
        print("PSRO Complete")
        print("=" * 60)
        print(f"  Final population size: {len(self.population)}")
        print(f"  Final exploitability:  {final_exploit:.6f}")
        print(f"  Results: {self.logger.experiment_dir}")
        print("=" * 60)

        return {
            "population": self.population,
            "meta_strategy": self.meta_strategy,
            "payoff_matrix": payoff_matrix,
            "exploitability_history": exploitability_history,
        }

    # ------------------------------------------------------------------
    # Best-response training
    # ------------------------------------------------------------------

    def _train_best_response(
        self, psro_iteration: int,
    ) -> Any:
        """Train a best-response policy and return its params."""
        cfg = self.config

        # Opponent mixture: blend Nash meta-strategy with uniform.
        #   mix = alpha * uniform + (1 - alpha) * nash
        # alpha anneals from br_mix_alpha_start → br_mix_alpha_end.
        K = len(self.population)
        num_iters = cfg.psro.num_psro_iterations
        alpha = linear_anneal(
            psro_iteration,
            num_iters,
            cfg.psro.br_mix_alpha_start,
            cfg.psro.br_mix_alpha_end,
            cfg.psro.br_mix_alpha_anneal_frac,
        )
        uniform = np.ones(K, dtype=np.float64) / K
        # Extend or truncate meta_strategy to match current population
        nash = self.meta_strategy
        if len(nash) != K:
            nash = np.ones(K, dtype=np.float64) / K
        br_strategy = alpha * uniform + (1.0 - alpha) * nash
        br_strategy /= br_strategy.sum()  # ensure normalization

        print(f"    α = {alpha:.3f}  (uniform weight in opponent mix)")

        # Warm-start from the policy with highest meta-strategy weight
        warmstart_params = None
        if cfg.psro.warmstart_br and K > 0:
            best_idx = int(np.argmax(self.meta_strategy))
            warmstart_params = self.population[best_idx]

        trainer = BestResponseTrainer(
            wrapper=self.wrapper,
            policy=self.policy,
            population=self.population,
            meta_strategy=br_strategy,
            num_envs=cfg.train.num_envs,
            rollout_length=cfg.train.rollout_length,
            gamma=cfg.train.gamma,
            gae_lambda=cfg.train.gae_lambda,
            clip_epsilon=cfg.train.clip_epsilon,
            value_coef=cfg.train.value_coef,
            entropy_coef=cfg.train.entropy_coef,
            learning_rate=cfg.train.learning_rate,
            max_grad_norm=cfg.train.max_grad_norm,
            ppo_epochs=cfg.train.ppo_epochs,
            num_minibatches=cfg.train.num_minibatches,
            target_kl=cfg.train.target_kl,
            seed=cfg.train.seed + psro_iteration * 1000,
            warmstart_params=warmstart_params,
        )

        num_updates = cfg.psro.num_br_updates
        log_interval = cfg.psro.psro_log_interval

        # ── Early stopping state ──────────────────────────────────────
        # Uses a rolling mean over recent log windows.  Converged when
        # the rolling mean hasn't improved by more than `delta` for
        # `patience` consecutive windows.
        patience = cfg.psro.br_patience
        delta = cfg.psro.br_early_stop_delta
        use_early_stop = patience > 0
        _es_window = max(patience // 2, 5)  # rolling window size
        _es_history: list[float] = []
        best_rolling_mean = -float("inf")
        stale_count = 0
        final_update = num_updates

        for update in range(1, num_updates + 1):
            # Annealing
            temperature: float | None = None
            if self._use_temp_anneal:
                assert cfg.train.buyer_choice_temp_start is not None
                temperature = linear_anneal(
                    update,
                    num_updates,
                    cfg.train.buyer_choice_temp_start,
                    cfg.train.buyer_choice_temp_end,
                    cfg.train.buyer_choice_temp_anneal_frac,
                )

            cur_entropy_coef: float | None = None
            if self._use_entropy_anneal:
                assert cfg.train.entropy_coef_start is not None
                cur_entropy_coef = linear_anneal(
                    update,
                    num_updates,
                    cfg.train.entropy_coef_start,
                    cfg.train.entropy_coef_end,
                    cfg.train.entropy_coef_anneal_frac,
                )

            # Collect + update
            transitions, advantages, returns, rollout_stats = (
                trainer.collect_rollout(temperature=temperature)
            )
            update_stats = trainer.update(
                transitions, advantages, returns,
                entropy_coef=cur_entropy_coef,
            )

            if update % log_interval == 0:
                global_step = (psro_iteration - 1) * num_updates + update
                self.logger.set_step(global_step)
                metrics = {**rollout_stats, **update_stats}
                if temperature is not None:
                    metrics["temperature"] = temperature
                self.logger.log_metrics(metrics, prefix="br")

                mean_rew = rollout_stats["mean_reward"]
                vloss = update_stats.get("value_loss", 0.0)
                kl = update_stats.get("approx_kl", 0.0)
                clip_f = update_stats.get("clip_fraction", 0.0)
                ent_c = update_stats.get("entropy_coef", 0.0)
                print(
                    f"    [BR {update}/{num_updates}] "
                    f"rew: {mean_rew:.2f} | "
                    f"ploss: {update_stats['policy_loss']:.4f} | "
                    f"vloss: {vloss:.2f} | "
                    f"ent: {update_stats['entropy']:.2f} | "
                    f"kl: {kl:.4f} | "
                    f"clip: {clip_f:.3f} | "
                    f"ec: {ent_c:.4f}"
                )

                # ── Early stopping check (rolling mean) ────────────────
                if use_early_stop:
                    _es_history.append(mean_rew)
                    if len(_es_history) >= _es_window:
                        rolling = sum(_es_history[-_es_window:]) / _es_window
                        if rolling > best_rolling_mean + delta:
                            best_rolling_mean = rolling
                            stale_count = 0
                        else:
                            stale_count += 1

                        if stale_count >= patience:
                            print(
                                f"    ✓ BR converged (rolling mean {rolling:.2f} "
                                f"stable for {patience} windows = "
                                f"{patience * log_interval} updates)"
                            )
                            final_update = update
                            break

        if final_update == num_updates and use_early_stop:
            rolling = (
                sum(_es_history[-_es_window:]) / min(len(_es_history), _es_window)
                if _es_history else 0.0
            )
            print(
                f"    ⚠ BR hit max updates ({num_updates}). "
                f"Rolling mean: {rolling:.2f}"
            )

        return trainer.params

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        iteration: int,
        exploitability_history: list[float],
        final: bool = False,
    ) -> None:
        """Save PSRO checkpoint."""
        name = "psro_final.pkl" if final else f"psro_iter_{iteration}.pkl"
        path = self.logger.experiment_dir / name

        data: dict[str, Any] = {
            "iteration": iteration,
            "population": self.population,
            "meta_strategy": self.meta_strategy,
            "payoff_matrix": (
                self.payoff_table.matrix if self.payoff_table._size > 0 else None
            ),
            "exploitability_history": exploitability_history,
            "config": self.config,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"  Saved checkpoint: {path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_wrapper(config: Config) -> TrainingWrapper:
    """Build a TrainingWrapper from config."""
    return TrainingWrapper(
        num_sellers=config.env.num_sellers,
        max_buyers=config.env.max_buyers,
        dimensions=config.env.dimensions,
        space_resolution=config.env.space_resolution,
        max_price=config.env.max_price,
        max_quality=config.env.max_quality,
        max_step_size=config.env.max_step_size,
        production_cost_factor=config.env.production_cost_factor,
        movement_cost=config.env.movement_cost,
        transport_cost=config.env.transport_cost,
        transportation_cost_norm=config.env.transportation_cost_norm,
        transport_cost_exponent=config.env.transport_cost_exponent,
        quality_taste=config.env.quality_taste,
        include_quality=config.env.include_quality,
        include_buyer_valuation=config.env.include_buyer_valuation,
        buyer_value=config.env.buyer_value,
        despawn_no_purchase=config.env.despawn_no_purchase,
        new_buyers_per_step=config.env.new_buyers_per_step,
        max_env_steps=config.env.max_env_steps,
        buyer_choice_temperature=config.env.buyer_choice_temperature,
        blob_sigma=config.train.blob_sigma,
        action_type=config.env.action_type,
        num_location_bins=config.env.num_location_bins,
        num_price_bins=config.env.num_price_bins,
        num_quality_bins=config.env.num_quality_bins,
        obs_type=config.train.obs_type,
        buyer_distribution=config.env.buyer_distribution,
        buyer_dist_means=config.env.buyer_dist_means,
        buyer_dist_stds=config.env.buyer_dist_stds,
        buyer_dist_weights=config.env.buyer_dist_weights,
    )


def _build_single_agent_policy(
    config: Config,
    wrapper: TrainingWrapper,
) -> PolicyAdapter:
    """Build a single-agent :class:`PolicyAdapter` for PSRO.

    PSRO policies always operate as a single agent (num_agents=1 for
    global mode, or a single ego network).  The same weights can be
    evaluated from either agent's perspective by feeding the
    appropriate observation.
    """
    hidden_dims = tuple(config.train.hidden_dims)
    ego = config.train.observation_mode == "egocentric"
    discrete = config.env.action_type == "discrete"
    conv_bin = config.train.obs_type == "conv_bin"

    if ego and discrete and conv_bin:
        scalar_dim = config.env.dimensions + 1
        if config.env.include_quality:
            scalar_dim += 1
        conv_net = EgoConv1dFactoredDiscreteActorCritic(
            num_location_bins=wrapper.num_location_bins,
            num_price_bins=wrapper.num_price_bins,
            num_quality_bins=wrapper.num_quality_bins,
            spatial_resolution=wrapper.space_resolution,
            num_grid_channels=wrapper._conv_grid_channels,
            num_scalar_features=scalar_dim,
            mlp_hidden_dims=hidden_dims,
        )
        return EgoFactoredDiscretePolicy(conv_net, num_agents=1)

    if ego and not discrete and conv_bin:
        gp = config.env.space_resolution + 1
        scalar_dim = config.env.dimensions + 1
        if config.env.include_quality:
            scalar_dim += 1
        conv2d_net = EgoConv2dActorCritic(
            movement_dim=wrapper.movement_dim,
            bounded_dim=wrapper.bounded_dim,
            spatial_resolution=gp,
            num_grid_channels=wrapper._conv_grid_channels,
            num_scalar_features=scalar_dim,
            mlp_hidden_dims=hidden_dims,
        )
        return EgoContinuousPolicy(conv2d_net, num_agents=1)

    if ego and discrete:
        ego_fac_net = EgoFactoredDiscreteActorCritic(
            num_location_bins=wrapper.num_location_bins,
            num_price_bins=wrapper.num_price_bins,
            num_quality_bins=wrapper.num_quality_bins,
            hidden_dims=hidden_dims,
        )
        return EgoFactoredDiscretePolicy(ego_fac_net, num_agents=1)

    if ego:
        ego_cont_net = EgoActorCritic(
            movement_dim=wrapper.movement_dim,
            bounded_dim=wrapper.bounded_dim,
            hidden_dims=hidden_dims,
        )
        return EgoContinuousPolicy(ego_cont_net, num_agents=1)

    if discrete:
        disc_net = DiscreteActorCritic(
            num_actions=wrapper.num_actions,
            num_agents=1,
            hidden_dims=hidden_dims,
        )
        return DiscretePolicy(disc_net)

    cont_net = SharedActorCritic(
        movement_dim=wrapper.movement_dim,
        bounded_dim=wrapper.bounded_dim,
        num_agents=1,
        hidden_dims=hidden_dims,
    )
    return ContinuousPolicy(cont_net)


def _params_contain_nan(params: Any) -> bool:
    """Check if any parameter leaf contains NaN or Inf."""
    for leaf in jax.tree.leaves(params):
        if jnp.any(jnp.isnan(leaf)).item() or jnp.any(jnp.isinf(leaf)).item():
            return True
    return False


def _fmt_vec(v: np.ndarray, precision: int = 3) -> str:
    """Format a probability vector for printing."""
    parts = [f"{x:.{precision}f}" for x in v]
    return "[" + ", ".join(parts) + "]"


def _fmt_nonzero(v: np.ndarray, threshold: float = 0.01) -> str:
    """Format only the significant entries of a probability vector.

    Shows ``index:weight`` for entries above *threshold*.
    Example: ``{0: 0.45, 3: 0.30, 7: 0.25}`` (K=10)
    """
    parts = [f"{i}:{x:.3f}" for i, x in enumerate(v) if x >= threshold]
    return "{" + ", ".join(parts) + f"}} (K={len(v)})"


def _population_diversity(population: list[Any]) -> float:
    """Mean pairwise L2 distance between flattened parameter vectors.

    Returns 0.0 for populations with fewer than 2 policies.
    """
    K = len(population)
    if K < 2:
        return 0.0

    flat = [
        jnp.concatenate([p.ravel() for p in jax.tree.leaves(params)])
        for params in population
    ]

    total = 0.0
    count = 0
    for i in range(K):
        for j in range(i + 1, K):
            total += float(jnp.linalg.norm(flat[i] - flat[j]))
            count += 1
    return total / count


def _print_matrix(m: np.ndarray, indent: int = 4) -> None:
    """Pretty-print a small matrix."""
    prefix = " " * indent
    for i in range(m.shape[0]):
        row = " ".join(f"{m[i, j]:8.2f}" for j in range(m.shape[1]))
        print(f"{prefix}[{row}]")
