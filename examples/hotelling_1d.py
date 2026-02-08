"""Hotelling 1-D demo with rendering.

Two sellers compete on a unit line segment.  Buyers are uniformly
distributed and choose the seller that maximises their utility
(value − distance − price).  Sellers take random actions each step.

Controls:
    Space   – pause / resume
    Click   – select entity for detail panel
    Slider  – adjust playback speed
    Esc     – deselect
"""

import jax
import jax.numpy as jnp

from spatial_competition_jax import (
    INFO_COMPLETE,
    TOPOLOGY_RECTANGLE,
    SpatialCompetitionEnv,
    SpatialCompetitionRenderer,
)


def main() -> None:
    # ── environment ──
    env = SpatialCompetitionEnv(
        num_sellers=2,
        max_buyers=100,
        dimensions=1,
        space_resolution=100,
        max_price=10.0,
        max_step_size=0.05,
        topology=TOPOLOGY_RECTANGLE,
        information_level=INFO_COMPLETE,
        include_quality=False,
        include_buyer_valuation=False,
        new_buyers_per_step=30,
        max_env_steps=500,
        movement_cost=0.1,
        production_cost_factor=0.0,
        transportation_cost_norm=2.0,
    )

    renderer = SpatialCompetitionRenderer(env)

    # ── JIT-compile individual phases ──
    jit_remove = jax.jit(env.step_remove_purchased)
    jit_spawn = jax.jit(env.step_spawn_buyers)
    jit_actions = jax.jit(env.step_apply_actions)
    jit_sales = jax.jit(env.step_process_sales)

    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)

    cumulative_rewards = {f"seller_{i}": 0.0 for i in range(env.num_sellers)}

    print("Running Hotelling 1-D demo  (close window to quit)")
    print(f"  sellers={env.num_sellers}  buyers/step={env.new_buyers_per_step}  max_steps={env.max_env_steps}")

    phase_delay = 0.15  # delay between each sub-step render

    for step_i in range(env.max_env_steps):
        # ── random actions ──
        key, k_act, k_spawn, k_sales = jax.random.split(key, 4)

        movement = jax.random.uniform(
            k_act,
            (env.num_sellers, env.dimensions),
            minval=-env.max_step_size,
            maxval=env.max_step_size,
        )
        price = jax.random.uniform(
            k_act,
            (env.num_sellers,),
            minval=1.0,
            maxval=env.max_price - 1.0,
        )
        actions = {"movement": movement, "price": price}

        render_kw = dict(current_step=step_i + 1, cumulative_rewards=cumulative_rewards)

        # ── Phase 1: Remove purchased buyers ──
        state = jit_remove(state)
        if not renderer.render_and_wait(state, base_delay=phase_delay, **render_kw):
            break

        # ── Phase 2: Spawn new buyers ──
        state = jit_spawn(k_spawn, state)
        if not renderer.render_and_wait(state, base_delay=phase_delay, **render_kw):
            break

        # ── Phase 3: Apply seller actions ──
        state = jit_actions(state, actions)
        if not renderer.render_and_wait(state, base_delay=phase_delay, **render_kw):
            break

        # ── Phase 4: Process sales ──
        obs, state, rewards, dones, info = jit_sales(k_sales, state)

        # Track cumulative rewards
        for i in range(env.num_sellers):
            cumulative_rewards[f"seller_{i}"] += float(rewards[i])

        if not renderer.render_and_wait(state, base_delay=phase_delay, **render_kw):
            break

        if bool(jnp.all(dones)):
            renderer.render_and_wait(state, base_delay=2.0, **render_kw)
            break

    renderer.close()
    print("\nFinal cumulative rewards:")
    for name, rew in sorted(cumulative_rewards.items()):
        print(f"  {name}: {rew:+.2f}")


if __name__ == "__main__":
    main()
