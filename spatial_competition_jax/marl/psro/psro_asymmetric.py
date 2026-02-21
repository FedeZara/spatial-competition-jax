"""Asymmetric (two-population) PSRO loop.

Each player maintains its own population of policies and meta-strategy.
At each iteration:

1. Update the bimatrix payoff table (pop0 × pop1).
2. Solve the bimatrix meta-game for (σ0, σ1).
3. Train a best-response for **each** player against the opponent's
   meta-strategy.
4. Add the new BRs to their respective populations.

The environment is egocentric, so a policy trained "as agent 0" can be
evaluated from either perspective — the observation already permutes
features to the agent's viewpoint.
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
from spatial_competition_jax.marl.psro.best_response import BestResponseTrainer
from spatial_competition_jax.marl.psro.meta_solver import (
    compute_exploitability_bimatrix,
    solve_bimatrix_game,
)
from spatial_competition_jax.marl.psro.payoff_table import (
    _evaluate_matchup_ego_jit,
    _evaluate_matchup_global_jit,
)
from spatial_competition_jax.marl.psro.psro import (
    _build_single_agent_policy,
    _build_wrapper,
    _fmt_nonzero,
    _params_contain_nan,
    _population_diversity,
)
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper
from spatial_competition_jax.marl.utils.logging import Logger

if TYPE_CHECKING:
    from spatial_competition_jax.marl.mappo.policy import PolicyAdapter


# ---------------------------------------------------------------------------
# Asymmetric payoff table (K0 × K1)
# ---------------------------------------------------------------------------


class AsymmetricPayoffTable:
    """Payoff table for two separate populations.

    Stores two matrices ``U0[i, j]`` and ``U1[i, j]`` where:
    - ``U0[i, j]`` = player 0's reward when pop0[i] vs pop1[j]
    - ``U1[i, j]`` = player 1's reward in the same matchup
    """

    def __init__(
        self,
        policy: PolicyAdapter,
        wrapper: TrainingWrapper,
        *,
        egocentric: bool = True,
        num_eval_episodes: int = 50,
        temperature: float | None = None,
        seed: int = 0,
    ) -> None:
        self.policy = policy
        self.wrapper = wrapper
        self.egocentric = egocentric
        self.num_eval_episodes = num_eval_episodes
        self.use_temp = wrapper.env.buyer_choice_temperature is not None
        self.temperature = jnp.float32(
            temperature if temperature is not None else 0.0
        )
        self.seed = seed

        self._U0: np.ndarray | None = None
        self._U1: np.ndarray | None = None
        self._size0: int = 0
        self._size1: int = 0

    @property
    def U0(self) -> np.ndarray:
        assert self._U0 is not None
        return self._U0

    @property
    def U1(self) -> np.ndarray:
        assert self._U1 is not None
        return self._U1

    def update(
        self,
        pop0: list[Any],
        pop1: list[Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Incrementally evaluate new matchups.

        Returns:
            ``(U0, U1)`` — payoff matrices of shape ``(K0, K1)``.
        """
        K0, K1 = len(pop0), len(pop1)
        old_K0, old_K1 = self._size0, self._size1

        new_U0 = np.zeros((K0, K1), dtype=np.float64)
        new_U1 = np.zeros((K0, K1), dtype=np.float64)

        if self._U0 is not None:
            r0 = min(old_K0, K0)
            c0 = min(old_K1, K1)
            new_U0[:r0, :c0] = self._U0[:r0, :c0]
            new_U1[:r0, :c0] = self._U1[:r0, :c0]  # type: ignore[index]

        eval_fn = (
            _evaluate_matchup_ego_jit if self.egocentric
            else _evaluate_matchup_global_jit
        )

        key = jax.random.PRNGKey(self.seed)

        # Determine which (i, j) pairs need evaluation
        for i in range(K0):
            for j in range(K1):
                if i < old_K0 and j < old_K1:
                    continue  # already evaluated

                key, subkey = jax.random.split(key)
                keys = jax.random.split(subkey, self.num_eval_episodes)

                rewards = eval_fn(
                    self.policy,
                    self.wrapper,
                    pop0[i],   # agent 0
                    pop1[j],   # agent 1
                    self.use_temp,
                    self.temperature,
                    keys,
                )
                # rewards: (num_episodes, 2)
                v0 = float(rewards[:, 0].mean())
                v1 = float(rewards[:, 1].mean())
                new_U0[i, j] = v0 if np.isfinite(v0) else 0.0
                new_U1[i, j] = v1 if np.isfinite(v1) else 0.0

        self._U0 = new_U0
        self._U1 = new_U1
        self._size0 = K0
        self._size1 = K1
        return new_U0, new_U1


# ---------------------------------------------------------------------------
# Asymmetric PSRO loop
# ---------------------------------------------------------------------------


class AsymmetricPSROLoop:
    """Two-population PSRO for 2-player spatial competition.

    Each player has its own population of policies.  At each iteration
    a best-response is trained for **both** players against the
    opponent's meta-strategy, then added to the respective population.
    """

    def __init__(
        self,
        config: Config,
        logger: Logger,
        wrapper: TrainingWrapper | None = None,
    ) -> None:
        self.config = config
        self.logger = logger

        if wrapper is not None:
            self.wrapper = wrapper
        else:
            self.wrapper = _build_wrapper(config)

        self.egocentric = config.train.observation_mode == "egocentric"
        self.discrete = config.env.action_type == "discrete"

        # Single-agent policy (used by both populations)
        self.policy = _build_single_agent_policy(config, self.wrapper)

        # Two populations and meta-strategies
        self.pop0: list[Any] = []
        self.pop1: list[Any] = []
        self.sigma0: np.ndarray = np.array([], dtype=np.float64)
        self.sigma1: np.ndarray = np.array([], dtype=np.float64)

        # Payoff table
        eval_temp = config.psro.eval_temperature
        if eval_temp is None and config.env.buyer_choice_temperature is not None:
            eval_temp = config.train.buyer_choice_temp_end
        self.payoff_table = AsymmetricPayoffTable(
            policy=self.policy,
            wrapper=self.wrapper,
            egocentric=self.egocentric,
            num_eval_episodes=config.psro.num_eval_episodes,
            temperature=eval_temp,
            seed=config.train.seed,
        )

        self._use_temp_anneal = (
            config.env.buyer_choice_temperature is not None
            and config.train.buyer_choice_temp_start is not None
        )
        self._use_entropy_anneal = config.train.entropy_coef_start is not None

    # ------------------------------------------------------------------
    # Population seeding
    # ------------------------------------------------------------------

    def seed_populations(
        self,
        seed: int | None = None,
        num_seeds: int = 1,
    ) -> None:
        """Add initial random policies to both populations."""
        if seed is None:
            seed = self.config.train.seed

        if self.egocentric:
            dummy = jnp.zeros(self.wrapper.obs_dim)
        else:
            dummy = jnp.zeros(self.wrapper.state_dim)

        for i in range(num_seeds):
            key0 = jax.random.PRNGKey(seed + i)
            key1 = jax.random.PRNGKey(seed + 10000 + i)
            self.pop0.append(self.policy.init(key0, dummy))
            self.pop1.append(self.policy.init(key1, dummy))

        self.sigma0 = np.ones(len(self.pop0), dtype=np.float64) / len(self.pop0)
        self.sigma1 = np.ones(len(self.pop1), dtype=np.float64) / len(self.pop1)
        print(f"Seeded both populations with {num_seeds} random policies")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, num_iterations: int | None = None) -> dict[str, Any]:
        """Run the asymmetric PSRO loop."""
        if num_iterations is None:
            num_iterations = self.config.psro.num_psro_iterations

        if not self.pop0:
            self.seed_populations(
                num_seeds=self.config.psro.num_initial_policies,
            )

        exploitability_history: list[float] = []

        print("=" * 60)
        print("Asymmetric PSRO Loop (two populations)")
        print("=" * 60)
        print(f"Iterations:       {num_iterations}")
        print(f"BR updates/iter:  {self.config.psro.num_br_updates}")
        print(f"Eval episodes:    {self.config.psro.num_eval_episodes}")
        print(f"Pop0 size:        {len(self.pop0)}")
        print(f"Pop1 size:        {len(self.pop1)}")
        print(f"Obs mode:         {'egocentric' if self.egocentric else 'global'}")
        print("=" * 60)

        for iteration in range(1, num_iterations + 1):
            print(f"\n{'─' * 60}")
            print(f"PSRO Iteration {iteration}/{num_iterations}")
            print(f"{'─' * 60}")

            # ── 1. Update payoff table ────────────────────────────────
            print("  Building payoff table …")
            U0, U1 = self.payoff_table.update(self.pop0, self.pop1)
            # ── 2. Solve bimatrix meta-game ───────────────────────────
            self.sigma0, self.sigma1 = solve_bimatrix_game(U0, U1)
            exploit = compute_exploitability_bimatrix(
                U0, U1, self.sigma0, self.sigma1,
            )
            exploitability_history.append(exploit)

            meta_ent0 = -float(np.sum(self.sigma0 * np.log(self.sigma0 + 1e-12)))
            meta_ent1 = -float(np.sum(self.sigma1 * np.log(self.sigma1 + 1e-12)))
            div0 = _population_diversity(self.pop0)
            div1 = _population_diversity(self.pop1)
            print(f"  σ0: {_fmt_nonzero(self.sigma0)}")
            print(f"  σ1: {_fmt_nonzero(self.sigma1)}")
            print(f"  exploit={exploit:.6f}  ent0={meta_ent0:.4f}  ent1={meta_ent1:.4f}")
            print(f"  div0={div0:.4f}  div1={div1:.4f}")

            self.logger.set_step(iteration)
            self.logger.log_metrics(
                {
                    "exploitability": exploit,
                    "pop0_size": len(self.pop0),
                    "pop1_size": len(self.pop1),
                    "meta_entropy_p0": meta_ent0,
                    "meta_entropy_p1": meta_ent1,
                    "pop_diversity_p0": div0,
                    "pop_diversity_p1": div1,
                },
                prefix="psro",
            )

            # ── 3. Train best responses for both players ──────────────
            print("  Training BR for player 0 …")
            br0 = self._train_best_response(
                iteration, player=0,
                opponent_pop=self.pop1,
                opponent_sigma=self.sigma1,
            )

            print("  Training BR for player 1 …")
            br1 = self._train_best_response(
                iteration, player=1,
                opponent_pop=self.pop0,
                opponent_sigma=self.sigma0,
            )

            # ── 4. Add to populations ─────────────────────────────────
            if br0 is not None and not _params_contain_nan(br0):
                self.pop0.append(br0)
                ext = np.append(self.sigma0, 1.0 / len(self.pop0))
                self.sigma0 = ext / ext.sum()
            else:
                print("  ⚠ Player 0 BR unstable — skipped.")

            if br1 is not None and not _params_contain_nan(br1):
                self.pop1.append(br1)
                ext = np.append(self.sigma1, 1.0 / len(self.pop1))
                self.sigma1 = ext / ext.sum()
            else:
                print("  ⚠ Player 1 BR unstable — skipped.")

            print(f"  Pop sizes: {len(self.pop0)} / {len(self.pop1)}")

            # ── 5. Save checkpoint ────────────────────────────────────
            if iteration % self.config.psro.psro_save_interval == 0:
                self._save_checkpoint(iteration, exploitability_history)

        # ── Final solve ───────────────────────────────────────────────
        print(f"\n{'─' * 60}")
        print("Final payoff table & meta-solve")
        print(f"{'─' * 60}")
        U0, U1 = self.payoff_table.update(self.pop0, self.pop1)
        self.sigma0, self.sigma1 = solve_bimatrix_game(U0, U1)
        final_exploit = compute_exploitability_bimatrix(
            U0, U1, self.sigma0, self.sigma1,
        )
        exploitability_history.append(final_exploit)

        print(f"  σ0: {_fmt_nonzero(self.sigma0)}")
        print(f"  σ1: {_fmt_nonzero(self.sigma1)}")
        print(f"  Exploitability: {final_exploit:.6f}")

        self._save_checkpoint(num_iterations, exploitability_history, final=True)

        print("\n" + "=" * 60)
        print("Asymmetric PSRO Complete")
        print("=" * 60)
        print(f"  Pop0 size:            {len(self.pop0)}")
        print(f"  Pop1 size:            {len(self.pop1)}")
        print(f"  Final exploitability: {final_exploit:.6f}")
        print(f"  Results: {self.logger.experiment_dir}")
        print("=" * 60)

        return {
            "pop0": self.pop0,
            "pop1": self.pop1,
            "sigma0": self.sigma0,
            "sigma1": self.sigma1,
            "U0": U0,
            "U1": U1,
            "exploitability_history": exploitability_history,
        }

    # ------------------------------------------------------------------
    # Best-response training
    # ------------------------------------------------------------------

    def _train_best_response(
        self,
        psro_iteration: int,
        player: int,
        opponent_pop: list[Any],
        opponent_sigma: np.ndarray,
    ) -> Any:
        """Train a best-response for *player* against *opponent_pop*.

        The trainee always plays as agent 0 in the code.  For player 1's
        BR, the opponent (from pop0) plays as agent 1.  Since egocentric
        observations handle perspective automatically, the trained
        policy works from either player's viewpoint.
        """
        cfg = self.config
        K = len(opponent_pop)
        num_iters = cfg.psro.num_psro_iterations

        alpha = linear_anneal(
            psro_iteration, num_iters,
            cfg.psro.br_mix_alpha_start,
            cfg.psro.br_mix_alpha_end,
            cfg.psro.br_mix_alpha_anneal_frac,
        )
        uniform = np.ones(K, dtype=np.float64) / K
        opp_sigma = opponent_sigma
        if len(opp_sigma) != K:
            opp_sigma = uniform
        br_strategy = alpha * uniform + (1.0 - alpha) * opp_sigma
        br_strategy /= br_strategy.sum()

        print(f"    Player {player} | α={alpha:.3f}")

        warmstart_params = None
        if cfg.psro.warmstart_br and K > 0:
            best_idx = int(np.argmax(opp_sigma))
            warmstart_params = opponent_pop[best_idx]

        # Offset seed by player to get different initializations
        seed = cfg.train.seed + psro_iteration * 1000 + player * 500

        trainer = BestResponseTrainer(
            wrapper=self.wrapper,
            policy=self.policy,
            population=opponent_pop,
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
            seed=seed,
            warmstart_params=warmstart_params,
        )

        num_updates = cfg.psro.num_br_updates
        log_interval = cfg.psro.psro_log_interval

        patience = cfg.psro.br_patience
        delta = cfg.psro.br_early_stop_delta
        use_early_stop = patience > 0
        best_reward = -float("inf")
        stale_count = 0
        final_update = num_updates

        for update in range(1, num_updates + 1):
            temperature: float | None = None
            if self._use_temp_anneal:
                assert cfg.train.buyer_choice_temp_start is not None
                temperature = linear_anneal(
                    update, num_updates,
                    cfg.train.buyer_choice_temp_start,
                    cfg.train.buyer_choice_temp_end,
                    cfg.train.buyer_choice_temp_anneal_frac,
                )

            cur_entropy_coef: float | None = None
            if self._use_entropy_anneal:
                assert cfg.train.entropy_coef_start is not None
                cur_entropy_coef = linear_anneal(
                    update, num_updates,
                    cfg.train.entropy_coef_start,
                    cfg.train.entropy_coef_end,
                    cfg.train.entropy_coef_anneal_frac,
                )

            transitions, advantages, returns, rollout_stats = (
                trainer.collect_rollout(temperature=temperature)
            )
            update_stats = trainer.update(
                transitions, advantages, returns,
                entropy_coef=cur_entropy_coef,
            )

            if update % log_interval == 0:
                prefix = f"br_p{player}"
                global_step = (psro_iteration - 1) * num_updates + update
                self.logger.set_step(global_step)
                self.logger.log_metrics(
                    {**rollout_stats, **update_stats}, prefix=prefix,
                )

                mean_rew = rollout_stats["mean_reward"]
                print(
                    f"    [P{player} BR {update}/{num_updates}] "
                    f"rew: {mean_rew:.2f} | "
                    f"ploss: {update_stats['policy_loss']:.4f} | "
                    f"ent: {update_stats['entropy']:.2f}"
                )

                if use_early_stop:
                    if mean_rew > best_reward + delta:
                        best_reward = mean_rew
                        stale_count = 0
                    else:
                        stale_count += 1
                    if stale_count >= patience:
                        print(
                            f"    ✓ P{player} BR converged "
                            f"(best: {best_reward:.4f})"
                        )
                        final_update = update
                        break

        if final_update == num_updates and use_early_stop:
            print(
                f"    ⚠ P{player} BR hit max updates "
                f"(best: {best_reward:.4f})"
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
        name = "psro_asym_final.pkl" if final else f"psro_asym_iter_{iteration}.pkl"
        path = self.logger.experiment_dir / name

        data: dict[str, Any] = {
            "iteration": iteration,
            "pop0": self.pop0,
            "pop1": self.pop1,
            "sigma0": self.sigma0,
            "sigma1": self.sigma1,
            "U0": self.payoff_table._U0,
            "U1": self.payoff_table._U1,
            "exploitability_history": exploitability_history,
            "config": self.config,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"  Saved checkpoint: {path}")
