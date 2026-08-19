"""Episode-safe lifecycle and state reconciliation for the V2 controller."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    BOARD_SIZE,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_CROPS,
    NUM_SHED_ITEMS,
)
from kaggriculture_jax.types import State
from strategic_v5.constants import TaskStatusV1
from strategic_v5.schema import MarketTaskStateV1, UnitTaskStateV1

from .constants import (
    MAX_PROJECTS_V2,
    MAX_ROUTE_CARD_STOPS_V3,
    MAX_ROUTE_STOPS,
    ProjectStatusV2,
    SCHEMA_VERSION_V2,
)
from .schema import (
    ControllerReconcileDiagnosticsV2,
    ProjectControllerStateV2,
    ProjectStateV2,
    RouteCardStateV3,
    UnitPlanStateV2,
)


def _checked_batch_size(batch_size: int) -> int:
    value = int(batch_size)
    if value <= 0:
        raise ValueError("batch_size must be positive")
    return value


def empty_project_state_v2(batch_size: int) -> ProjectStateV2:
    batch = _checked_batch_size(batch_size)
    shape = (batch, MAX_PROJECTS_V2)
    i8 = lambda value=0: jnp.full(shape, value, dtype=jnp.int8)
    i16 = lambda value=0: jnp.full(shape, value, dtype=jnp.int16)
    i32 = lambda value=0: jnp.full(shape, value, dtype=jnp.int32)
    return ProjectStateV2(
        project_id=i16(-1),
        project_type=i8(),
        item_id=i8(-1),
        status=i8(ProjectStatusV2.UNUSED),
        phase=i8(),
        target_count=i16(),
        committed_count=i16(),
        active_count=i16(),
        completed_count=i16(),
        start_step=i16(-1),
        last_progress_step=i16(-1),
        stop_step=i16(-1),
        latest_bank_step=i16(-1),
        cash_budget=i32(),
        cash_spent=i32(),
        expected_bank_value=i32(),
        layout_policy_id=i8(-1),
        priority_class=i8(),
        failure_code=i8(),
    )


def empty_unit_tasks_v2(batch_size: int) -> UnitTaskStateV1:
    batch = _checked_batch_size(batch_size)
    shape = (batch, MAX_UNITS)
    i8 = lambda value=0: jnp.full(shape, value, dtype=jnp.int8)
    i16 = lambda value=0: jnp.full(shape, value, dtype=jnp.int16)
    return UnitTaskStateV1(
        task_type=i8(),
        owner_unit=jnp.broadcast_to(
            jnp.arange(MAX_UNITS, dtype=jnp.int8), shape
        ),
        target_id=i16(-1),
        target_x=i8(-1),
        target_y=i8(-1),
        item_id=i8(-1),
        quantity=i16(),
        phase=i8(),
        start_step=i16(-1),
        last_progress_step=i16(-1),
        expected_finish_step=i16(-1),
        deadline_step=i16(-1),
        status=i8(TaskStatusV1.EMPTY),
        failure_code=i8(),
    )


def empty_market_tasks_v2(batch_size: int) -> MarketTaskStateV1:
    batch = _checked_batch_size(batch_size)
    shape = (batch, MAX_MARKET_ORDERS)
    i8 = lambda value=0: jnp.full(shape, value, dtype=jnp.int8)
    i16 = lambda value=0: jnp.full(shape, value, dtype=jnp.int16)
    return MarketTaskStateV1(
        task_type=i8(),
        item_id=i8(-1),
        quantity=i16(),
        start_step=i16(-1),
        deadline_step=i16(-1),
        status=i8(TaskStatusV1.EMPTY),
        failure_code=i8(),
    )


def empty_unit_plans_v2(batch_size: int) -> UnitPlanStateV2:
    batch = _checked_batch_size(batch_size)
    unit_shape = (batch, MAX_UNITS)
    route_shape = (batch, MAX_UNITS, MAX_ROUTE_STOPS)
    i8 = lambda value=0: jnp.full(unit_shape, value, dtype=jnp.int8)
    i16 = lambda value=0: jnp.full(unit_shape, value, dtype=jnp.int16)
    return UnitPlanStateV2(
        home_region=i8(-1),
        primary_project_id=i16(-1),
        active_obligation_id=i16(-1),
        route_obligation_ids=jnp.full(route_shape, -1, dtype=jnp.int16),
        route_length=i8(),
        route_cursor=i8(),
        commitment_until_step=i16(-1),
        last_switch_step=i16(-1),
        switches_today=i8(),
        last_target_x=i8(-1),
        last_target_y=i8(-1),
    )


def empty_route_cards_v3(batch_size: int) -> RouteCardStateV3:
    batch = _checked_batch_size(batch_size)
    unit_shape = (batch, MAX_UNITS)
    stop_shape = (batch, MAX_UNITS, MAX_ROUTE_CARD_STOPS_V3)
    i8 = lambda value=0: jnp.full(unit_shape, value, dtype=jnp.int8)
    i16 = lambda value=0: jnp.full(unit_shape, value, dtype=jnp.int16)
    return RouteCardStateV3(
        card_type=i8(),
        status=i8(),
        target_ids=jnp.full(stop_shape, -1, dtype=jnp.int16),
        target_items=jnp.full(stop_shape, -1, dtype=jnp.int8),
        action_masks=jnp.zeros(stop_shape, dtype=jnp.uint8),
        route_length=i8(),
        route_cursor=i8(),
        pickup_item=i8(-1),
        pickup_quantity=i16(),
        return_mode=i8(),
        start_step=i16(-1),
        deadline_step=i16(-1),
        last_progress_step=i16(-1),
        failure_code=i8(),
    )


def reset_project_controller_v2(batch_size: int) -> ProjectControllerStateV2:
    """Return a fully empty batched controller with invalid state snapshots."""

    batch = _checked_batch_size(batch_size)
    return ProjectControllerStateV2(
        schema_version=jnp.full((batch,), SCHEMA_VERSION_V2, dtype=jnp.int16),
        projects=empty_project_state_v2(batch),
        unit_tasks=empty_unit_tasks_v2(batch),
        unit_plans=empty_unit_plans_v2(batch),
        route_cards=empty_route_cards_v3(batch),
        market_tasks=empty_market_tasks_v2(batch),
        tile_project_id=jnp.full(
            (batch, BOARD_SIZE, BOARD_SIZE), -1, dtype=jnp.int16
        ),
        route_phase=jnp.zeros((batch,), dtype=jnp.int8),
        liquidation_mode=jnp.zeros((batch,), dtype=jnp.bool_),
        last_money=jnp.full((batch,), -1, dtype=jnp.int32),
        last_shed=jnp.full((batch, NUM_SHED_ITEMS), -1, dtype=jnp.int16),
        last_seeds=jnp.full((batch, NUM_CROPS), -1, dtype=jnp.int16),
        last_tile_kind=jnp.full(
            (batch, BOARD_SIZE, BOARD_SIZE), -1, dtype=jnp.int8
        ),
        last_tile_yield=jnp.full(
            (batch, BOARD_SIZE, BOARD_SIZE), -1, dtype=jnp.int16
        ),
        project_cap_hits=jnp.zeros((batch,), dtype=jnp.int32),
        obligation_cap_hits=jnp.zeros((batch,), dtype=jnp.int32),
        unexplained_effect_failures=jnp.zeros((batch,), dtype=jnp.int32),
    )


def clear_project_controller_v2(
    controller: ProjectControllerStateV2,
) -> ProjectControllerStateV2:
    """Discard every per-episode value, including snapshots and counters."""

    return reset_project_controller_v2(controller.schema_version.shape[0])


def snapshot_project_controller_v2(
    states: State,
    controller: ProjectControllerStateV2,
    player: int,
) -> ProjectControllerStateV2:
    """Copy exact public own-farm state into the controller's previous-step view."""

    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    return controller._replace(
        last_money=states.money[:, player].astype(jnp.int32),
        last_shed=states.shed[:, player].astype(jnp.int16),
        last_seeds=states.seeds[:, player].astype(jnp.int16),
        last_tile_kind=states.tile_kind[:, player].astype(jnp.int8),
        last_tile_yield=states.tile_yield[:, player].astype(jnp.int16),
    )


def initialize_project_controller_v2(
    states: State,
    player: int,
) -> ProjectControllerStateV2:
    """Create an empty controller whose snapshots match a new episode state."""

    controller = reset_project_controller_v2(states.step.shape[0])
    return snapshot_project_controller_v2(states, controller, player)


def _masked_replace(record, empty, mask: jax.Array):
    return jax.tree.map(
        lambda value, replacement: jnp.where(
            mask.reshape(mask.shape + (1,) * (value.ndim - mask.ndim)),
            replacement,
            value,
        ),
        record,
        empty,
    )


def clear_finished_controller_records_v2(
    states: State,
    controller: ProjectControllerStateV2,
    player: int,
) -> tuple[ProjectControllerStateV2, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Clear terminal projects/tasks and inactive hand plans without touching counters."""

    batch = states.step.shape[0]
    project_finished = controller.projects.status >= ProjectStatusV2.COMPLETE
    unit_finished = controller.unit_tasks.status >= TaskStatusV1.DONE
    market_finished = controller.market_tasks.status >= TaskStatusV1.DONE
    inactive_units = ~states.unit_active[:, player]
    unit_nonempty = controller.unit_tasks.status != TaskStatusV1.EMPTY
    plan_nonempty = (
        (controller.unit_plans.primary_project_id >= 0)
        | (controller.unit_plans.active_obligation_id >= 0)
        | (controller.unit_plans.route_length > 0)
        | (controller.unit_plans.commitment_until_step >= 0)
        | (controller.unit_plans.last_switch_step >= 0)
    )

    empty_projects = empty_project_state_v2(batch)
    empty_units = empty_unit_tasks_v2(batch)
    empty_market = empty_market_tasks_v2(batch)
    empty_plans = empty_unit_plans_v2(batch)
    empty_routes = empty_route_cards_v3(batch)

    projects = _masked_replace(controller.projects, empty_projects, project_finished)
    unit_clear = unit_finished | (inactive_units & unit_nonempty)
    plan_clear = inactive_units & plan_nonempty
    unit_tasks = _masked_replace(controller.unit_tasks, empty_units, unit_clear)
    market_tasks = _masked_replace(controller.market_tasks, empty_market, market_finished)
    unit_plans = _masked_replace(controller.unit_plans, empty_plans, plan_clear)
    # Route status 3 is WAIT_AUTO_BANK and must survive until the official
    # day-end transition.  DONE/FAILED are 4/5.
    route_finished = controller.route_cards.status >= 4
    route_nonempty = controller.route_cards.status != 0
    route_clear = route_finished | (inactive_units & route_nonempty)
    route_cards = _masked_replace(
        controller.route_cards, empty_routes, route_clear
    )

    cleared_ids = jnp.where(
        project_finished, controller.projects.project_id, jnp.int16(-2)
    )
    tile_matches = (
        controller.tile_project_id[..., None]
        == cleared_ids[:, None, None, :]
    )
    tile_project_id = jnp.where(
        jnp.any(tile_matches, axis=-1),
        jnp.int16(-1),
        controller.tile_project_id,
    )
    updated = controller._replace(
        projects=projects,
        unit_tasks=unit_tasks,
        unit_plans=unit_plans,
        route_cards=route_cards,
        market_tasks=market_tasks,
        tile_project_id=tile_project_id,
    )
    counts = (
        jnp.sum(project_finished, axis=-1, dtype=jnp.int32),
        jnp.sum(unit_clear, axis=-1, dtype=jnp.int32),
        jnp.sum(market_finished, axis=-1, dtype=jnp.int32),
        jnp.sum(plan_clear, axis=-1, dtype=jnp.int32),
    )
    return updated, counts


def reconcile_project_controller_v2(
    states: State,
    controller: ProjectControllerStateV2,
    player: int,
) -> tuple[ProjectControllerStateV2, ControllerReconcileDiagnosticsV2]:
    """Clear invalid episode-local records and refresh exact previous-state snapshots.

    M1 reports observable deltas only. It does not infer action success or advance
    project phases; those require the M2/M3 task state machines.
    """

    valid = controller.last_money >= 0
    current_money = states.money[:, player].astype(jnp.int32)
    current_shed = states.shed[:, player].astype(jnp.int16)
    current_seeds = states.seeds[:, player].astype(jnp.int16)
    current_kind = states.tile_kind[:, player].astype(jnp.int8)
    current_yield = states.tile_yield[:, player].astype(jnp.int16)
    expanded_valid = valid[:, None]
    tile_valid = valid[:, None, None]

    updated, counts = clear_finished_controller_records_v2(
        states, controller, player
    )
    diagnostics = ControllerReconcileDiagnosticsV2(
        snapshot_was_valid=valid,
        money_delta=jnp.where(valid, current_money - controller.last_money, 0),
        shed_delta=jnp.where(
            expanded_valid, current_shed - controller.last_shed, 0
        ).astype(jnp.int16),
        seed_delta=jnp.where(
            expanded_valid, current_seeds - controller.last_seeds, 0
        ).astype(jnp.int16),
        tile_kind_changed_count=jnp.sum(
            tile_valid & (current_kind != controller.last_tile_kind),
            axis=(1, 2),
            dtype=jnp.int32,
        ),
        tile_yield_changed_count=jnp.sum(
            tile_valid & (current_yield != controller.last_tile_yield),
            axis=(1, 2),
            dtype=jnp.int32,
        ),
        cleared_project_count=counts[0],
        cleared_unit_task_count=counts[1],
        cleared_market_task_count=counts[2],
        cleared_unit_plan_count=counts[3],
    )
    return snapshot_project_controller_v2(states, updated, player), diagnostics


def controller_equal_per_batch_v2(
    left: ProjectControllerStateV2,
    right: ProjectControllerStateV2,
) -> jax.Array:
    """Exact equality across every controller leaf, reduced per batch lane."""

    comparisons = []
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True
    ):
        equal = left_leaf == right_leaf
        axes = tuple(range(1, equal.ndim))
        comparisons.append(jnp.all(equal, axis=axes) if axes else equal)
    return jnp.all(jnp.stack(comparisons, axis=0), axis=0)


__all__ = [
    "clear_finished_controller_records_v2",
    "clear_project_controller_v2",
    "controller_equal_per_batch_v2",
    "empty_market_tasks_v2",
    "empty_project_state_v2",
    "empty_route_cards_v3",
    "empty_unit_plans_v2",
    "empty_unit_tasks_v2",
    "initialize_project_controller_v2",
    "reconcile_project_controller_v2",
    "reset_project_controller_v2",
    "snapshot_project_controller_v2",
]
