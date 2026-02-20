#!/usr/bin/env python3
"""Diagnostic script: run a short training and print detailed sanity checks."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from spatial_competition_jax.marl.config import Config
from spatial_competition_jax.marl.mappo.mappo import MAPPO, linear_anneal
from spatial_competition_jax.marl.mappo.networks import (
    EPS,
    SharedActorCritic,
    _entropy_beta,
    _entropy_gaussian,
    _log_prob_beta,
    _log_prob_tanh_normal,
    deterministic_actions,
    sample_actions,
    symexp,
    symlog,
)
from spatial_competition_jax.marl.training_wrapper import TrainingWrapper

# ── Helpers ───────────────────────────────────────────────────────────
PASS = "✅"
WARN = "⚠️ "
FAIL = "🔴"


def _stats(name: str, arr: jnp.ndarray, expected_range=None) -> str:
    """Format array statistics."""
    a = np.asarray(arr).astype(np.float64)
    n_nan = int(np.isnan(a).sum())
    n_inf = int(np.isinf(a).sum())
    lines = [
        f"  {name}: shape={arr.shape} dtype={arr.dtype}",
        f"    min={float(a.min()):.6f}  max={float(a.max()):.6f}  "
        f"mean={float(a.mean()):.6f}  std={float(a.std()):.6f}",
    ]
    status = PASS
    if n_nan > 0:
        lines.append(f"    {FAIL} NaN count: {n_nan}")
        status = FAIL
    if n_inf > 0:
        lines.append(f"    {FAIL} Inf count: {n_inf}")
        status = FAIL
    if expected_range is not None:
        lo, hi = expected_range
        if float(a.min()) < lo - 1e-6 or float(a.max()) > hi + 1e-6:
            lines.append(
                f"    {WARN}Expected range [{lo}, {hi}], "
                f"got [{float(a.min()):.4f}, {float(a.max()):.4f}]"
            )
            status = WARN
    if status == PASS:
        lines[0] = f"  {PASS} {name}: shape={arr.shape} dtype={arr.dtype}"
    return "\n".join(lines)


def _check_nan(name: str, arr: jnp.ndarray) -> bool:
    a = np.asarray(arr)
    has_nan = bool(np.isnan(a).any())
    has_inf = bool(np.isinf(a).any())
    if has_nan or has_inf:
        print(f"  {FAIL} {name} has {'NaN' if has_nan else ''}{'&' if has_nan and has_inf else ''}{'Inf' if has_inf else ''}")
        return False
    return True


def _grad_norm(grads) -> float:
    leaves = jax.tree.leaves(grads)
    return float(jnp.sqrt(sum(jnp.sum(g**2) for g in leaves)))


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  TRAINING DIAGNOSTICS")
    print("=" * 70)

    # ── 1. Load config ────────────────────────────────────────────────
    config_path = Path(__file__).parent.parent / "configs" / "hotelling_1d.yaml"
    config = Config.from_yaml(config_path)
    print(f"\nConfig loaded from {config_path}")
    print(f"  dimensions={config.env.dimensions}, sellers={config.env.num_sellers}")
    print(f"  space_res={config.env.space_resolution}, max_price={config.env.max_price}")
    print(f"  max_step_size={config.env.max_step_size}, transport_cost={config.env.transport_cost}")
    print(f"  include_quality={config.env.include_quality}")
    print(f"  buyer_choice_temperature={config.env.buyer_choice_temperature}")
    print(f"  new_buyers_per_step={config.env.new_buyers_per_step}")

    # ── 2. Create wrapper ─────────────────────────────────────────────
    wrapper = TrainingWrapper(
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
        buyer_distribution=config.env.buyer_distribution,
        buyer_dist_means=config.env.buyer_dist_means,
        buyer_dist_stds=config.env.buyer_dist_stds,
        buyer_dist_weights=config.env.buyer_dist_weights,
    )

    print(f"\n{'─'*70}")
    print(f"WRAPPER DIMENSIONS")
    print(f"  state_dim    = {wrapper.state_dim}")
    print(f"  movement_dim = {wrapper.movement_dim}")
    print(f"  bounded_dim  = {wrapper.bounded_dim}")
    print(f"  action_dim   = {wrapper.action_dim}")
    print(f"  num_agents   = {wrapper.num_agents}")

    # ── 3. Reset env & check initial state ────────────────────────────
    print(f"\n{'─'*70}")
    print("INITIAL STATE CHECK")
    key = jax.random.PRNGKey(42)
    key, reset_key = jax.random.split(key)
    global_state, env_state = wrapper.reset(reset_key)

    print(_stats("global_state", global_state))
    print(f"\n  env_state.seller_positions = {env_state.seller_positions}")
    pos_norm = env_state.seller_positions.astype(jnp.float32) / wrapper.space_resolution
    print(f"  normalized positions       = {pos_norm}")
    print(f"  env_state.seller_prices    = {env_state.seller_prices}")
    print(f"  env_state.step             = {env_state.step}")

    # ── 4. Network forward pass ───────────────────────────────────────
    print(f"\n{'─'*70}")
    print("NETWORK FORWARD PASS (single state)")

    network = SharedActorCritic(
        movement_dim=wrapper.movement_dim,
        bounded_dim=wrapper.bounded_dim,
        num_agents=wrapper.num_agents,
        hidden_dims=(256, 256),
    )
    key, init_key = jax.random.split(key)
    params = network.init(init_key, global_state)

    # Count parameters
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"  Total parameters: {n_params:,}")

    gauss_means, gauss_log_stds, beta_alphas, beta_betas, values = network.apply(params, global_state)

    print(_stats("gauss_means", gauss_means, expected_range=(-5, 5)))
    print(_stats("gauss_log_stds", gauss_log_stds, expected_range=(-5, 2)))
    print(_stats("beta_alphas", beta_alphas, expected_range=(1, 22)))
    print(_stats("beta_betas", beta_betas, expected_range=(1, 22)))
    print(_stats("values (symlog)", values))

    # ── 5. Sample actions ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("SAMPLE ACTIONS (stochastic)")
    key, sample_key = jax.random.split(key)
    actions, log_probs, vals = sample_actions(network, params, global_state, sample_key)

    print(_stats("actions", actions))
    print(f"  movement actions[:, :D]:  {actions[:, :wrapper.movement_dim]}")
    print(f"  bounded  actions[:, D:]:  {actions[:, wrapper.movement_dim:]}")
    print(_stats("log_probs", log_probs))
    print(_stats("values", vals))

    # ── 6. Deterministic actions ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print("DETERMINISTIC ACTIONS")
    det_actions, det_vals = deterministic_actions(network, params, global_state)
    print(_stats("det_actions", det_actions))
    print(f"  movement: {det_actions[:, :wrapper.movement_dim]}")
    print(f"  bounded:  {det_actions[:, wrapper.movement_dim:]}")
    _check_nan("det_actions", det_actions)
    _check_nan("det_vals", det_vals)

    # ── 7. Map actions → env actions ──────────────────────────────────
    print(f"\n{'─'*70}")
    print("ACTION MAPPING")
    env_actions = wrapper.map_actions(actions)
    for k, v in env_actions.items():
        print(_stats(f"env_actions['{k}']", v))

    # ── 8. Step env ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("ENVIRONMENT STEP")
    key, step_key = jax.random.split(key)
    use_temp = wrapper.env.buyer_choice_temperature is not None
    if use_temp:
        temp = jnp.float32(1.0)
        next_gs, next_es, rewards, dones = wrapper.step(step_key, env_state, actions, temperature=temp)
    else:
        next_gs, next_es, rewards, dones = wrapper.step(step_key, env_state, actions)

    print(_stats("next_global_state", next_gs))
    print(_stats("rewards", rewards))
    print(_stats("dones", dones))
    print(f"  rewards per agent: {rewards}")

    # ── 9. Run N env steps to check reward scale ──────────────────────
    print(f"\n{'─'*70}")
    print("REWARD TRAJECTORY (10 random-action steps)")
    gs = global_state
    es = env_state
    all_rewards = []
    for i in range(10):
        key, act_key, step_key2 = jax.random.split(key, 3)
        a, _, _ = sample_actions(network, params, gs, act_key)
        if use_temp:
            gs, es, r, d = wrapper.step(step_key2, es, a, temperature=jnp.float32(1.0))
        else:
            gs, es, r, d = wrapper.step(step_key2, es, a)
        all_rewards.append(np.asarray(r))
        if i < 5 or i == 9:
            print(f"  step {i+1:2d}: reward={r}, done={d[0]}")

    all_rewards = np.stack(all_rewards)
    print(f"\n  Reward summary over 10 steps:")
    print(f"    mean={all_rewards.mean():.4f}  std={all_rewards.std():.4f}")
    print(f"    min={all_rewards.min():.4f}   max={all_rewards.max():.4f}")

    # ── 10. Full MAPPO training loop (50 updates) ─────────────────────
    print(f"\n{'='*70}")
    print("MAPPO TRAINING LOOP (50 updates)")
    print("=" * 70)

    # Use fewer envs for speed
    NUM_ENVS = 16
    ROLLOUT_LEN = 64

    agent = MAPPO(
        wrapper=wrapper,
        num_envs=NUM_ENVS,
        rollout_length=ROLLOUT_LEN,
        hidden_dims=config.train.hidden_dims,
        gamma=config.train.gamma,
        gae_lambda=config.train.gae_lambda,
        clip_epsilon=config.train.clip_epsilon,
        value_coef=config.train.value_coef,
        entropy_coef=config.train.entropy_coef,
        learning_rate=config.train.learning_rate,
        max_grad_norm=config.train.max_grad_norm,
        ppo_epochs=config.train.ppo_epochs,
        num_minibatches=config.train.num_minibatches,
        seed=config.train.seed,
    )

    # Temperature annealing setup
    use_temp_anneal = (
        config.env.buyer_choice_temperature is not None
        and config.train.buyer_choice_temp_start is not None
    )

    N_UPDATES = 50
    # Print detailed info on these updates, compact on the rest
    DETAIL_UPDATES = {1, 2, 3, 5, 10, 20, 30, 40, 50}

    # Header for compact output
    print(f"\n  {'upd':>4s} | {'reward':>8s} | {'pol_loss':>9s} | {'val_loss':>9s} | {'entropy':>8s} | {'kl':>7s} | {'clip%':>6s} | {'grad_n':>7s} | {'val_mean':>9s} | {'beta_mode':>10s} | {'price':>6s}")
    print(f"  {'─'*4:s}-+-{'─'*8:s}-+-{'─'*9:s}-+-{'─'*9:s}-+-{'─'*8:s}-+-{'─'*7:s}-+-{'─'*6:s}-+-{'─'*7:s}-+-{'─'*9:s}-+-{'─'*10:s}-+-{'─'*6:s}")

    from spatial_competition_jax.marl.mappo.buffer import make_minibatches

    for update in range(1, N_UPDATES + 1):
        verbose = update in DETAIL_UPDATES

        # Compute temperature
        if use_temp_anneal:
            temperature_val = linear_anneal(
                update, config.train.total_updates,
                config.train.buyer_choice_temp_start,
                config.train.buyer_choice_temp_end,
                config.train.buyer_choice_temp_anneal_frac,
            )
        else:
            temperature_val = None

        # Compute entropy coef
        if config.train.entropy_coef_start is not None:
            cur_ent_coef = linear_anneal(
                update, config.train.total_updates,
                config.train.entropy_coef_start,
                config.train.entropy_coef_end,
                config.train.entropy_coef_anneal_frac,
            )
        else:
            cur_ent_coef = None

        # ── Collect rollout ───────────────────────────────────────────
        transitions, advantages, returns, rollout_stats = agent.collect_rollout(
            temperature=temperature_val,
        )

        # ── PPO Update ────────────────────────────────────────────────
        update_stats = agent.update(
            transitions, advantages, returns,
            entropy_coef=cur_ent_coef,
        )

        # ── Gradient norm (sampled) ───────────────────────────────────
        mb_key = jax.random.PRNGKey(update)
        batches = make_minibatches(mb_key, transitions, advantages, returns, agent.num_minibatches)
        first_batch = jax.tree.map(lambda x: x[0], batches)

        def loss_fn(params):
            gm2, gls2, ba2, bb2, v2 = agent.network.apply(params, first_batch.states)
            gs2 = jnp.exp(gls2)
            m_act = first_batch.actions[..., :wrapper.movement_dim]
            b_act = first_batch.actions[..., wrapper.movement_dim:]
            clipped_m = jnp.clip(m_act, -1.0 + EPS, 1.0 - EPS)
            raw_m = jnp.arctanh(clipped_m)
            lp_m = _log_prob_tanh_normal(gm2, gs2, raw_m, m_act)
            lp_b = _log_prob_beta(ba2, bb2, b_act)
            new_lp = lp_m + lp_b
            ratio = jnp.exp(new_lp - first_batch.log_probs)
            surr1 = ratio * first_batch.advantages
            surr2 = jnp.clip(ratio, 0.8, 1.2) * first_batch.advantages
            p_loss = -jnp.minimum(surr1, surr2).mean()
            v_loss = ((v2 - first_batch.returns) ** 2).mean()
            ent = _entropy_gaussian(gls2) + _entropy_beta(ba2, bb2)
            e_loss = -ent.mean()
            total = p_loss + 0.5 * v_loss + 0.01 * e_loss
            return total, {
                "ratio_min": ratio.min(), "ratio_max": ratio.max(), "ratio_mean": ratio.mean(),
            }

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, aux), grads = grad_fn(agent.params)
        gnorm = _grad_norm(grads)

        # ── Distribution snapshot ─────────────────────────────────────
        sample_state = transitions.states[0, 0]
        gm, gls, ba, bb, v = agent.network.apply(agent.params, sample_state)
        beta_mode = (ba - 1.0) / (ba + bb - 2.0)
        price_mode = float(beta_mode.mean()) * wrapper.max_price

        # ── Compact one-liner ─────────────────────────────────────────
        p_sign = PASS if update_stats["policy_loss"] <= 0 else WARN
        nan_flag = FAIL if any(np.isnan(v) for v in update_stats.values()) else ""
        print(
            f"  {update:4d} | {rollout_stats['mean_reward']:8.2f} | "
            f"{p_sign}{update_stats['policy_loss']:+9.5f} | "
            f"{update_stats['value_loss']:9.5f} | "
            f"{update_stats['entropy']:8.4f} | "
            f"{update_stats['approx_kl']:7.4f} | "
            f"{update_stats['clip_fraction']*100:5.1f}% | "
            f"{gnorm:7.2f} | "
            f"{float(transitions.values.mean()):9.2f} | "
            f"{np.asarray(beta_mode).flatten().round(3)} | "
            f"{price_mode:6.2f} {nan_flag}"
        )

        # ── Verbose detail on selected updates ────────────────────────
        if verbose:
            print(f"\n    --- Detailed (update {update}) ---")
            print(f"    Rewards: mean={rollout_stats['mean_reward']:.2f}  std={rollout_stats['std_reward']:.2f}")
            print(f"    Values:  mean={float(transitions.values.mean()):.2f}  min={float(transitions.values.min()):.2f}  max={float(transitions.values.max()):.2f}")
            returns_real = np.asarray(symexp(returns))
            print(f"    Returns (real): mean={returns_real.mean():.2f}  min={returns_real.min():.2f}  max={returns_real.max():.2f}")
            print(f"    Advantages: mean={float(advantages.mean()):.4f}  std={float(advantages.std()):.4f}  min={float(advantages.min()):.2f}  max={float(advantages.max()):.2f}")

            move_act = transitions.actions[..., :wrapper.movement_dim]
            bound_act = transitions.actions[..., wrapper.movement_dim:]
            print(f"    Movement: mean={float(move_act.mean()):.4f}  std={float(move_act.std()):.4f}")
            print(f"    Bounded:  mean={float(bound_act.mean()):.4f}  std={float(bound_act.std()):.4f}")

            print(f"    Gauss means:    {np.asarray(gm).flatten().round(4)}")
            print(f"    Gauss stds:     {np.asarray(jnp.exp(gls)).flatten().round(4)}")
            print(f"    Beta α:         {np.asarray(ba).flatten().round(4)}")
            print(f"    Beta β:         {np.asarray(bb).flatten().round(4)}")
            print(f"    Beta mode:      {np.asarray(beta_mode).flatten().round(4)} → price ≈ {price_mode:.2f}")
            print(f"    Values (symlog):{np.asarray(v).flatten().round(4)}")
            print(f"    Values (real):  {np.asarray(symexp(v)).flatten().round(4)}")
            print(f"    Ratio: [{float(aux['ratio_min']):.3f}, {float(aux['ratio_max']):.3f}]  mean={float(aux['ratio_mean']):.4f}")
            print(f"    Grad norm: {gnorm:.4f}")

            # NaN check
            all_finite = True
            for name, arr in [("states", transitions.states), ("actions", transitions.actions),
                               ("log_probs", transitions.log_probs), ("values", transitions.values),
                               ("rewards", transitions.rewards), ("advantages", advantages), ("returns", returns)]:
                if not _check_nan(f"    {name}", arr):
                    all_finite = False
            if all_finite:
                print(f"    {PASS} All arrays finite")
            print()

    # ── 11. Deterministic eval after training ─────────────────────────
    print(f"\n{'='*70}")
    print(f"DETERMINISTIC EVAL (after {N_UPDATES} updates)")
    print("=" * 70)
    from spatial_competition_jax.marl.mappo.evaluation import evaluate_policy

    eval_stats = evaluate_policy(
        network=agent.network,
        params=agent.params,
        wrapper=wrapper,
        num_episodes=5,
        deterministic=True,
        temperature=config.train.buyer_choice_temp_end if use_temp_anneal else None,
    )
    for k, v in eval_stats.items():
        status = FAIL if np.isnan(v) else PASS
        print(f"  {status} {k}: {v:.4f}")

    # ── 12. Check param values ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("PARAMETER HEALTH (after training)")
    param_leaves = jax.tree.leaves(agent.params)
    for i, p in enumerate(param_leaves):
        a = np.asarray(p)
        has_issue = np.isnan(a).any() or np.isinf(a).any()
        if has_issue:
            print(f"  {FAIL} param[{i}]: shape={p.shape} has NaN/Inf!")
        elif np.abs(a).max() > 100:
            print(f"  {WARN}param[{i}]: shape={p.shape} max_abs={np.abs(a).max():.2f} (large)")

    all_ok = all(
        not (np.isnan(np.asarray(p)).any() or np.isinf(np.asarray(p)).any())
        for p in param_leaves
    )
    if all_ok:
        print(f"  {PASS} All {len(param_leaves)} parameter arrays are finite")

    print(f"\n{'='*70}")
    print("  DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
