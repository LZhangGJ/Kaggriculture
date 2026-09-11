from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import MAX_UNITS
from kaggriculture_jax.state import reset
from strategic_v5.constants import TaskStatusV1

from project_route_search_v2.constants import (
    MAX_MARKET_ORDERS,
    MAX_PROJECTS_V2,
    MAX_ROUTE_STOPS,
    MAX_ROUTE_CARD_STOPS_V3,
    ProjectStatusV2,
    SCHEMA_VERSION_V2,
)
from project_route_search_v2.lifecycle import (
    controller_equal_per_batch_v2,
    initialize_project_controller_v2,
    reconcile_project_controller_v2,
    reset_project_controller_v2,
)
from project_route_search_v2.scan_carry import (
    advance_controller_scan_carry_v2,
    initialize_controller_scan_carry_v2,
    poison_controller_for_reset_test_v2,
    scan_controller_state_sequence_v2,
)


def _states(batch_size: int = 3):
    return jax.vmap(reset)(jnp.arange(100, 100 + batch_size, dtype=jnp.int32))


def test_reset_factory_has_fixed_shapes_and_empty_sentinels() -> None:
    controller = reset_project_controller_v2(3)
    assert controller.schema_version.shape == (3,)
    np.testing.assert_array_equal(controller.schema_version, SCHEMA_VERSION_V2)
    assert controller.projects.project_id.shape == (3, MAX_PROJECTS_V2)
    assert controller.unit_tasks.status.shape == (3, MAX_UNITS)
    assert controller.unit_plans.route_obligation_ids.shape == (
        3,
        MAX_UNITS,
        MAX_ROUTE_STOPS,
    )
    assert controller.market_tasks.status.shape == (3, MAX_MARKET_ORDERS)
    assert controller.route_cards.target_ids.shape == (
        3,
        MAX_UNITS,
        MAX_ROUTE_CARD_STOPS_V3,
    )
    np.testing.assert_array_equal(controller.route_cards.status, 0)
    np.testing.assert_array_equal(controller.route_cards.target_ids, -1)
    np.testing.assert_array_equal(controller.projects.project_id, -1)
    np.testing.assert_array_equal(controller.projects.status, ProjectStatusV2.UNUSED)
    np.testing.assert_array_equal(controller.unit_tasks.status, TaskStatusV1.EMPTY)
    np.testing.assert_array_equal(controller.tile_project_id, -1)
    np.testing.assert_array_equal(controller.last_money, -1)
    np.testing.assert_array_equal(controller.project_cap_hits, 0)


def test_initialize_snapshots_exact_current_player_state() -> None:
    states = _states()
    controller = initialize_project_controller_v2(states, player=1)
    np.testing.assert_array_equal(controller.last_money, states.money[:, 1])
    np.testing.assert_array_equal(controller.last_shed, states.shed[:, 1])
    np.testing.assert_array_equal(controller.last_seeds, states.seeds[:, 1])
    np.testing.assert_array_equal(controller.last_tile_kind, states.tile_kind[:, 1])
    np.testing.assert_array_equal(controller.last_tile_yield, states.tile_yield[:, 1])


def test_reconcile_reports_deltas_and_clears_terminal_records() -> None:
    states = _states(2)
    controller = initialize_project_controller_v2(states, player=0)
    projects = controller.projects._replace(
        project_id=controller.projects.project_id.at[:, 0].set(7),
        status=controller.projects.status.at[:, 0].set(ProjectStatusV2.COMPLETE),
    )
    unit_tasks = controller.unit_tasks._replace(
        status=controller.unit_tasks.status.at[:, 0].set(TaskStatusV1.DONE)
    )
    market_tasks = controller.market_tasks._replace(
        status=controller.market_tasks.status.at[:, 0].set(TaskStatusV1.FAILED)
    )
    unit_plans = controller.unit_plans._replace(
        primary_project_id=controller.unit_plans.primary_project_id.at[:, 1].set(7)
    )
    controller = controller._replace(
        projects=projects,
        unit_tasks=unit_tasks,
        market_tasks=market_tasks,
        unit_plans=unit_plans,
        tile_project_id=controller.tile_project_id.at[:, 0, 0].set(7),
    )
    next_states = states._replace(
        money=states.money.at[:, 0].add(25),
        shed=states.shed.at[:, 0, 0].set(3),
        seeds=states.seeds.at[:, 0, 0].set(2),
        tile_kind=states.tile_kind.at[:, 0, 0, 0].set(2),
        tile_yield=states.tile_yield.at[:, 0, 0, 0].set(5),
    )
    updated, diagnostics = jax.jit(
        lambda state, ctrl: reconcile_project_controller_v2(
            state, ctrl, player=0
        )
    )(next_states, controller)
    np.testing.assert_array_equal(diagnostics.snapshot_was_valid, True)
    np.testing.assert_array_equal(diagnostics.money_delta, 25)
    np.testing.assert_array_equal(diagnostics.shed_delta[:, 0], 3)
    np.testing.assert_array_equal(diagnostics.seed_delta[:, 0], 2)
    np.testing.assert_array_equal(diagnostics.tile_kind_changed_count, 1)
    np.testing.assert_array_equal(diagnostics.tile_yield_changed_count, 1)
    np.testing.assert_array_equal(diagnostics.cleared_project_count, 1)
    np.testing.assert_array_equal(diagnostics.cleared_unit_task_count, 1)
    np.testing.assert_array_equal(diagnostics.cleared_market_task_count, 1)
    np.testing.assert_array_equal(diagnostics.cleared_unit_plan_count, 1)
    np.testing.assert_array_equal(updated.projects.project_id[:, 0], -1)
    np.testing.assert_array_equal(updated.tile_project_id[:, 0, 0], -1)
    np.testing.assert_array_equal(updated.unit_tasks.status[:, 0], TaskStatusV1.EMPTY)
    np.testing.assert_array_equal(updated.market_tasks.status[:, 0], TaskStatusV1.EMPTY)
    np.testing.assert_array_equal(updated.unit_plans.primary_project_id[:, 1], -1)
    np.testing.assert_array_equal(updated.last_money, next_states.money[:, 0])


def test_reconcile_with_invalid_snapshot_does_not_report_false_deltas() -> None:
    states = _states(2)
    controller = reset_project_controller_v2(2)
    updated, diagnostics = reconcile_project_controller_v2(states, controller, 0)
    np.testing.assert_array_equal(diagnostics.snapshot_was_valid, False)
    np.testing.assert_array_equal(diagnostics.money_delta, 0)
    np.testing.assert_array_equal(diagnostics.shed_delta, 0)
    np.testing.assert_array_equal(diagnostics.seed_delta, 0)
    np.testing.assert_array_equal(updated.last_money, states.money[:, 0])


def test_episode_boundary_resets_only_selected_batch_lanes() -> None:
    states = _states(2)
    carry = initialize_controller_scan_carry_v2(states, player=0)
    carry = carry._replace(
        controller=poison_controller_for_reset_test_v2(carry.controller)
    )
    episode_start = jnp.asarray([True, False])
    next_carry, diagnostics = jax.jit(
        lambda value, state, start: advance_controller_scan_carry_v2(
            value, state, start, player=0
        )
    )(carry, states, episode_start)
    fresh = initialize_project_controller_v2(states, player=0)
    equal = controller_equal_per_batch_v2(next_carry.controller, fresh)
    np.testing.assert_array_equal(equal, [True, False])
    np.testing.assert_array_equal(diagnostics.episode_reset, episode_start)
    np.testing.assert_array_equal(diagnostics.cross_episode_contamination, False)


def test_state_and_controller_are_valid_lax_scan_carry() -> None:
    initial_states = _states(2)
    initial_carry = initialize_controller_scan_carry_v2(initial_states, player=0)
    state1 = initial_states._replace(
        money=initial_states.money.at[:, 0].add(10)
    )
    state2 = initial_states._replace(
        money=initial_states.money.at[:, 0].add(20)
    )
    sequence = jax.tree.map(lambda left, right: jnp.stack((left, right)), state1, state2)
    starts = jnp.asarray([[False, False], [True, False]])
    run = jax.jit(
        lambda carry, states, flags: scan_controller_state_sequence_v2(
            carry, states, flags, player=0
        )
    )
    final_carry, diagnostics = run(initial_carry, sequence, starts)
    np.testing.assert_array_equal(final_carry.environment_state.money, state2.money)
    np.testing.assert_array_equal(final_carry.controller.last_money, state2.money[:, 0])
    assert diagnostics.episode_reset.shape == (2, 2)
    np.testing.assert_array_equal(diagnostics.cross_episode_contamination, False)
