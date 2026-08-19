from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax import Events, load_tables

from strategic_v5.dynamic_arena_v2 import (
    dynamic_two_policy_step_v2,
    make_dynamic_paired_arena_rollout_runtime_gates_v2,
    make_dynamic_paired_arena_rollout_v2,
)
from strategic_v5.dynamic_rollout_v2 import dynamic_full_step_v2
from strategic_v5.learned_v2 import initialize_full_learned_params_v2
from strategic_v5.rollout_v2 import initialize_full_learned_carry_v2


def test_two_policy_identical_params_matches_single_policy_one_step():
    seeds = jnp.asarray([11, 12], dtype=jnp.int32)
    events = Events(
        weed_spawn=jnp.zeros((2, 30, 200), dtype=jnp.bool_),
        shop_choice=jnp.zeros((2, 30, 201), dtype=jnp.int8),
    )
    carry = initialize_full_learned_carry_v2(seeds, events, jax.random.key(5))
    params = initialize_full_learned_params_v2(jax.random.key(6))
    expected_kernel = jax.jit(
        lambda one_carry, tables, one_params: dynamic_full_step_v2(
            one_carry, tables, one_params, deterministic=True
        )
    )
    actual_kernel = jax.jit(
        lambda one_carry, tables, one_params: dynamic_two_policy_step_v2(
            one_carry, tables, one_params, one_params, deterministic=True
        )
    )
    expected, expected_trace = expected_kernel(carry, load_tables(), params)
    actual, (invalid, overflow) = actual_kernel(carry, load_tables(), params)
    jax.block_until_ready((expected, actual))
    comparisons = jax.tree.leaves(
        jax.tree.map(
            lambda left, right: jnp.array_equal(left, right),
            expected.environment_state,
            actual.environment_state,
        )
    )
    assert all(bool(value) for value in comparisons)
    assert jnp.array_equal(expected_trace.invalid_market_orders, invalid)
    assert jnp.array_equal(expected_trace.overflow_market_orders, overflow)


def test_explicit_equal_player_gates_match_legacy_gate():
    seeds = jnp.asarray([21, 22], dtype=jnp.int32)
    events = Events(
        weed_spawn=jnp.zeros((2, 30, 200), dtype=jnp.bool_),
        shop_choice=jnp.zeros((2, 30, 201), dtype=jnp.int8),
    )
    carry = initialize_full_learned_carry_v2(seeds, events, jax.random.key(15))
    params = initialize_full_learned_params_v2(jax.random.key(16))
    tables = load_tables()
    legacy = jax.jit(
        lambda one_carry: dynamic_two_policy_step_v2(
            one_carry,
            tables,
            params,
            params,
            deterministic=True,
            allow_nonpositive_econ=True,
        )
    )(carry)
    explicit = jax.jit(
        lambda one_carry: dynamic_two_policy_step_v2(
            one_carry,
            tables,
            params,
            params,
            deterministic=True,
            allow_nonpositive_econ=False,
            player0_allow_nonpositive_econ=True,
            player1_allow_nonpositive_econ=True,
        )
    )(carry)
    jax.block_until_ready((legacy, explicit))
    comparisons = jax.tree.leaves(
        jax.tree.map(lambda left, right: jnp.array_equal(left, right), legacy, explicit)
    )
    assert all(bool(value) for value in comparisons)


def test_mixed_player_gates_execute_one_step():
    seeds = jnp.asarray([31, 32], dtype=jnp.int32)
    events = Events(
        weed_spawn=jnp.zeros((2, 30, 200), dtype=jnp.bool_),
        shop_choice=jnp.zeros((2, 30, 201), dtype=jnp.int8),
    )
    carry = initialize_full_learned_carry_v2(seeds, events, jax.random.key(25))
    params = initialize_full_learned_params_v2(jax.random.key(26))
    following, (invalid, overflow) = jax.jit(
        lambda one_carry: dynamic_two_policy_step_v2(
            one_carry,
            load_tables(),
            params,
            params,
            deterministic=True,
            player0_allow_nonpositive_econ=False,
            player1_allow_nonpositive_econ=True,
        )
    )(carry)
    jax.block_until_ready(following.environment_state.step)
    assert jnp.array_equal(following.environment_state.step, jnp.ones((2,), dtype=jnp.int16))
    assert invalid.shape == (2, 2)
    assert overflow.shape == (2, 2)


def test_runtime_gate_arena_matches_static_gate_arena():
    seeds = jnp.asarray([41, 42], dtype=jnp.int32)
    events = Events(
        weed_spawn=jnp.zeros((2, 30, 200), dtype=jnp.bool_),
        shop_choice=jnp.zeros((2, 30, 201), dtype=jnp.int8),
    )
    carry = initialize_full_learned_carry_v2(seeds, events, jax.random.key(35))
    params = initialize_full_learned_params_v2(jax.random.key(36))
    tables = load_tables()
    static = jax.jit(
        make_dynamic_paired_arena_rollout_v2(
            rollout_steps=2,
            deterministic=True,
            player0_allow_nonpositive_econ=False,
            player1_allow_nonpositive_econ=True,
        )
    )(carry, tables, params, params)
    runtime = jax.jit(
        make_dynamic_paired_arena_rollout_runtime_gates_v2(
            rollout_steps=2,
            deterministic=True,
        )
    )(
        carry,
        tables,
        params,
        params,
        jnp.asarray(False),
        jnp.asarray(True),
    )
    jax.block_until_ready((static, runtime))
    comparisons = jax.tree.leaves(
        jax.tree.map(lambda left, right: jnp.array_equal(left, right), static, runtime)
    )
    assert all(bool(value) for value in comparisons)
