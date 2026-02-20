"""Basic tests: shapes, JIT, vmap."""

import jax
import jax.numpy as jnp

from spatial_competition_jax import (
    INFO_COMPLETE,
    INFO_LIMITED,
    INFO_PRIVATE,
    JaxMARLWrapper,
    SpatialCompetitionEnv,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_env(**kw: object) -> SpatialCompetitionEnv:
    defaults: dict[str, object] = dict(
        num_sellers=3,
        max_buyers=50,
        dimensions=2,
        space_resolution=20,
        max_env_steps=10,
    )
    defaults.update(kw)
    return SpatialCompetitionEnv(**defaults)  # type: ignore[arg-type]


def _zero_actions(env: SpatialCompetitionEnv, batch_shape: tuple[int, ...] = ()) -> dict[str, jnp.ndarray]:
    s = env.num_sellers
    d = env.dimensions
    actions = {
        "movement": jnp.zeros(batch_shape + (s, d)),
        "price": jnp.full(batch_shape + (s,), 5.0),
    }
    if env.include_quality:
        actions["quality"] = jnp.full(batch_shape + (s,), 2.5)
    return actions


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestResetShapes:
    def test_basic(self) -> None:
        env = _make_env()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        assert state.seller_positions.shape == (3, 2)
        assert state.seller_prices.shape == (3,)
        assert state.seller_qualities.shape == (3,)
        assert state.buyer_positions.shape == (50, 2)
        assert state.buyer_valid.shape == (50,)
        assert obs["own_position"].shape == (3, 2)
        assert obs["own_price"].shape == (3,)
        assert obs["local_view"].shape == (3, 3, 21, 21)

    def test_complete_info_grids(self) -> None:
        env = _make_env(information_level=INFO_COMPLETE)
        obs, _ = env.reset(jax.random.PRNGKey(1))
        assert "buyers" in obs
        assert "sellers_price" in obs
        assert "sellers_quality" in obs
        assert obs["buyers"].shape == (3, 3, 21, 21)
        assert obs["sellers_price"].shape == (3, 21, 21)

    def test_limited_info_grids(self) -> None:
        env = _make_env(information_level=INFO_LIMITED)
        obs, _ = env.reset(jax.random.PRNGKey(2))
        assert "buyers" in obs
        assert "sellers_price" not in obs

    def test_private_info_grids(self) -> None:
        env = _make_env(information_level=INFO_PRIVATE)
        obs, _ = env.reset(jax.random.PRNGKey(3))
        assert "buyers" not in obs
        assert "sellers_price" not in obs

    def test_include_quality(self) -> None:
        env = _make_env(include_quality=True)
        obs, _ = env.reset(jax.random.PRNGKey(4))
        assert "own_quality" in obs

    def test_no_quality(self) -> None:
        env = _make_env(include_quality=False)
        obs, _ = env.reset(jax.random.PRNGKey(5))
        assert "own_quality" not in obs

    def test_1d(self) -> None:
        env = _make_env(dimensions=1)
        obs, state = env.reset(jax.random.PRNGKey(6))
        assert state.seller_positions.shape == (3, 1)
        assert obs["local_view"].shape == (3, 3, 21)

    def test_3d(self) -> None:
        env = _make_env(dimensions=3, space_resolution=5, max_buyers=10)
        obs, state = env.reset(jax.random.PRNGKey(7))
        assert state.seller_positions.shape == (3, 3)
        assert obs["local_view"].shape == (3, 3, 6, 6, 6)


class TestStep:
    def test_shapes(self) -> None:
        env = _make_env()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        key, k = jax.random.split(key)
        actions = _zero_actions(env)
        obs, state, rewards, dones, info = env.step(k, state, actions)

        assert rewards.shape == (3,)
        assert dones.shape == (3,)
        assert int(state.step) == 1

    def test_truncation_at_max_steps(self) -> None:
        env = _make_env(max_env_steps=2)
        key = jax.random.PRNGKey(10)
        _, state = env.reset(key)

        actions = _zero_actions(env)
        for i in range(2):
            key, k = jax.random.split(key)
            _, state, _, dones, _ = env.step(k, state, actions)

        assert bool(jnp.all(dones))

    def test_with_quality(self) -> None:
        env = _make_env(include_quality=True)
        key = jax.random.PRNGKey(20)
        _, state = env.reset(key)
        actions = _zero_actions(env)
        key, k = jax.random.split(key)
        obs, state, rewards, dones, _ = env.step(k, state, actions)
        assert "own_quality" in obs

    def test_with_buyer_valuation(self) -> None:
        env = _make_env(include_buyer_valuation=True)
        key = jax.random.PRNGKey(30)
        _, state = env.reset(key)
        actions = _zero_actions(env)
        key, k = jax.random.split(key)
        _, state, rewards, _, _ = env.step(k, state, actions)
        assert rewards.shape == (3,)

    def test_softmax_buyer_choice(self) -> None:
        """With buyer_choice_temperature, sales should be fractional."""
        env = _make_env(buyer_choice_temperature=1.0)
        key = jax.random.PRNGKey(50)
        _, state = env.reset(key)
        actions = _zero_actions(env)
        key, k = jax.random.split(key)
        _, state, rewards, _, _ = env.step(k, state, actions)

        # Sales are now float32, not int32
        assert state.seller_running_sales.dtype == jnp.float32
        assert rewards.shape == (3,)

        # With multiple sellers, fractional sales should generally be non-integer
        total_sales = jnp.sum(state.seller_running_sales)
        num_buyers = jnp.sum(state.buyer_valid | (state.buyer_purchased_from >= 0))
        # Total expected sales should be close to number of buying buyers
        assert total_sales > 0

    def test_softmax_low_temperature_approaches_hard(self) -> None:
        """Very low temperature should approximate hard argmax."""
        key = jax.random.PRNGKey(60)

        env_hard = _make_env(buyer_choice_temperature=None)
        _, state_hard = env_hard.reset(key)
        actions = _zero_actions(env_hard)
        k = jax.random.split(key)[1]
        _, state_hard, _, _, _ = env_hard.step(k, state_hard, actions)

        env_soft = _make_env(buyer_choice_temperature=0.001)
        _, state_soft = env_soft.reset(key)
        actions = _zero_actions(env_soft)
        _, state_soft, _, _, _ = env_soft.step(k, state_soft, actions)

        # With very low temperature, sales should be close to hard allocation
        assert jnp.allclose(state_hard.seller_running_sales, state_soft.seller_running_sales, atol=0.1)

    def test_softmax_jit(self) -> None:
        """Softmax buyer choice should work under JIT."""
        env = _make_env(
            num_sellers=2,
            max_buyers=20,
            dimensions=1,
            space_resolution=10,
            buyer_choice_temperature=1.0,
        )
        key = jax.random.PRNGKey(70)
        jit_reset = jax.jit(env.reset)
        _, state = jit_reset(key)

        jit_step = jax.jit(env.step)
        key, k = jax.random.split(key)
        actions = _zero_actions(env)
        _, state, rewards, _, _ = jit_step(k, state, actions)
        assert rewards.shape == (2,)
        assert state.seller_running_sales.dtype == jnp.float32


class TestJIT:
    def test_jit_reset_step(self) -> None:
        env = _make_env(num_sellers=2, max_buyers=20, dimensions=1, space_resolution=10)
        key = jax.random.PRNGKey(42)

        jit_reset = jax.jit(env.reset)
        obs, state = jit_reset(key)

        jit_step = jax.jit(env.step)
        key, k = jax.random.split(key)
        actions = _zero_actions(env)
        obs, state, rewards, dones, _ = jit_step(k, state, actions)

        assert int(state.step) == 1
        assert rewards.shape == (2,)


class TestVmap:
    def test_vmap_reset_step(self) -> None:
        env = _make_env(num_sellers=2, max_buyers=10, dimensions=1, space_resolution=10)

        num_envs = 4
        keys = jax.random.split(jax.random.PRNGKey(0), num_envs)

        vmap_reset = jax.vmap(env.reset)
        obs, states = vmap_reset(keys)
        assert states.seller_positions.shape == (num_envs, 2, 1)

        vmap_step = jax.vmap(env.step)
        step_keys = jax.random.split(jax.random.PRNGKey(1), num_envs)
        actions = _zero_actions(env, batch_shape=(num_envs,))
        obs, states, rewards, dones, _ = vmap_step(step_keys, states, actions)
        assert rewards.shape == (num_envs, 2)
        assert dones.shape == (num_envs, 2)


class TestJaxMARLWrapper:
    def test_reset_step(self) -> None:
        env = _make_env(num_sellers=2, max_buyers=20)
        wrapper = JaxMARLWrapper(env)

        key = jax.random.PRNGKey(99)
        obs_dict, state = wrapper.reset(key)

        assert "seller_0" in obs_dict
        assert "seller_1" in obs_dict
        assert obs_dict["seller_0"]["own_position"].shape == (2,)

        key, k = jax.random.split(key)
        actions_dict = {
            agent: {
                "movement": jnp.zeros(2),
                "price": jnp.float32(5.0),
            }
            for agent in wrapper.agents
        }
        obs_dict, state, rew, dones, _ = wrapper.step(k, state, actions_dict)
        assert "seller_0" in rew
        assert "__all__" in dones
