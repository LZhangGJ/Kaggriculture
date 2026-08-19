from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import load_event_bank, load_tables
from strategic_v5 import (
    build_full_core_candidates_v1,
    build_full_econ_features_v1,
    compile_full_core_action_bundle_v1,
    evaluate_full_core_feasibility_v1,
    h_execute_selected_step_v2,
    initialize_full_learned_params_v1,
    make_h_learned_decision_kernel_v2,
    make_h_rule_decision_kernel_v2,
    staged_h_hybrid_step_v2,
)
from strategic_v5.h_arena import (
    _rule_decision_v1,
    h_hybrid_step_v1,
    initialize_h_hybrid_carry_v1,
)
from strategic_v5.learned_v1 import _player_decision_v1
from strategic_v5.lifecycle import clear_invalidated_full_core_tasks_v1


TABLES = load_tables()


def _carry(batch_size: int = 2):
    event_ids, events = load_event_bank()
    events = jax.tree.map(lambda value: value[:batch_size], events)
    return initialize_h_hybrid_carry_v1(
        jnp.asarray(event_ids[:batch_size], dtype=jnp.int32),
        events,
        jax.random.key(2026082700),
    )


def _assert_tree_exact(left, right) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    assert len(left_leaves) == len(right_leaves)
    for left_value, right_value in zip(left_leaves, right_leaves, strict=True):
        if jax.dtypes.issubdtype(left_value.dtype, jax.dtypes.prng_key):
            left_value = jax.random.key_data(left_value)
            right_value = jax.random.key_data(right_value)
        assert np.array_equal(np.asarray(left_value), np.asarray(right_value))


def test_dynamic_risk_flags_are_exactly_equal_to_static_variants() -> None:
    carry = _carry()
    states = carry.environment_state
    for player in (0, 1):
        controller = clear_invalidated_full_core_tasks_v1(
            states,
            carry.player0_controller if player == 0 else carry.player1_controller,
            player,
        )
        candidates = build_full_core_candidates_v1(
            states, controller, TABLES, player
        )
        feasibility = evaluate_full_core_feasibility_v1(
            states, candidates, TABLES, player
        )
        for expected, scenario in ((True, True), (True, False), (False, False)):
            static = build_full_econ_features_v1(
                states,
                candidates,
                feasibility,
                TABLES,
                player,
                include_expected_risk=expected,
                include_scenario_risk=scenario,
            )
            flags = jnp.asarray([expected, scenario], dtype=jnp.bool_)
            dynamic = build_full_econ_features_v1(
                states,
                candidates,
                feasibility,
                TABLES,
                player,
                include_expected_risk=flags[0],
                include_scenario_risk=flags[1],
            )
            _assert_tree_exact(static, dynamic)


def _reference_selection(carry, params, learned_player: int):
    states = carry.environment_state
    next_key, learned_key = jax.random.split(carry.rng)
    controller0 = clear_invalidated_full_core_tasks_v1(
        states, carry.player0_controller, 0
    )
    controller1 = clear_invalidated_full_core_tasks_v1(
        states, carry.player1_controller, 1
    )
    if learned_player == 0:
        learned, _, _, _, _ = _player_decision_v1(
            states, controller0, TABLES, params, 0, learned_key, True
        )
        rule = _rule_decision_v1(
            states,
            controller1,
            TABLES,
            1,
            include_expected_risk=True,
            include_scenario_risk=True,
        )
        return (
            learned.controller,
            rule.controller,
            learned.internal_resource_conflict,
            rule.internal_resource_conflict,
            next_key,
        )
    rule = _rule_decision_v1(
        states,
        controller0,
        TABLES,
        0,
        include_expected_risk=True,
        include_scenario_risk=True,
    )
    learned, _, _, _, _ = _player_decision_v1(
        states, controller1, TABLES, params, 1, learned_key, True
    )
    return (
        rule.controller,
        learned.controller,
        rule.internal_resource_conflict,
        learned.internal_resource_conflict,
        next_key,
    )


def test_staged_selections_actions_and_full_step_match_v1_for_both_seats() -> None:
    carry = _carry()
    params = initialize_full_learned_params_v1(jax.random.key(2026082701))
    flags = jnp.asarray([True, True], dtype=jnp.bool_)
    for learned_player in (0, 1):
        selected0, selected1, conflict0, conflict1, next_key = _reference_selection(
            carry, params, learned_player
        )
        learned = make_h_learned_decision_kernel_v2(learned_player)(
            carry, TABLES, params
        )
        rule = make_h_rule_decision_kernel_v2(1 - learned_player)(
            carry, TABLES, flags
        )
        if learned_player == 0:
            staged0, staged_conflict0 = (
                learned.controller,
                learned.internal_resource_conflict,
            )
            staged1, staged_conflict1 = (
                rule.controller,
                rule.internal_resource_conflict,
            )
        else:
            staged0, staged_conflict0 = (
                rule.controller,
                rule.internal_resource_conflict,
            )
            staged1, staged_conflict1 = (
                learned.controller,
                learned.internal_resource_conflict,
            )
        _assert_tree_exact(selected0, staged0)
        _assert_tree_exact(selected1, staged1)
        _assert_tree_exact(conflict0, staged_conflict0)
        _assert_tree_exact(conflict1, staged_conflict1)
        _assert_tree_exact(next_key, learned.next_key)

        reference_action = compile_full_core_action_bundle_v1(
            carry.environment_state, selected0, selected1
        ).action
        staged_action = compile_full_core_action_bundle_v1(
            carry.environment_state, staged0, staged1
        ).action
        _assert_tree_exact(reference_action, staged_action)

        reference_step = h_hybrid_step_v1(
            carry,
            TABLES,
            params,
            learned_player=learned_player,
            decision_interval=1,
            include_expected_risk=True,
            include_scenario_risk=True,
        )
        staged_step = h_execute_selected_step_v2(
            carry,
            TABLES,
            staged0,
            staged1,
            staged_conflict0,
            staged_conflict1,
            learned.next_key,
        )
        direct_composition = staged_h_hybrid_step_v2(
            carry,
            TABLES,
            params,
            flags,
            learned_player=learned_player,
        )
        _assert_tree_exact(reference_step, staged_step)
        _assert_tree_exact(reference_step, direct_composition)
