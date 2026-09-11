"""Persistent E2 executor/compiler and effect checks."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    BOARD_SIZE,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    MarketOp,
    UnitOp,
)
from kaggriculture_jax.types import Action, State

from .constants import FailureCodeV1, TaskPhaseV1, TaskStatusV1, TaskTypeV1
from .e2_core import E2PlayerActionV1, FERTILIZER_ITEM
from .geometry import movement_op_toward_v1, nearest_shed_access_v1
from .schema import (
    CompileDiagnosticsV1,
    ControllerStateV1,
    EffectDiagnosticsV1,
)


class E2ActionBundleV1(NamedTuple):
    action: Action
    diagnostics: CompileDiagnosticsV1
    player0: E2PlayerActionV1
    player1: E2PlayerActionV1


def compile_e2_player_action_v1(
    states: State, controller: ControllerStateV1, player: int
) -> E2PlayerActionV1:
    batch_size = states.step.shape[0]
    tasks = controller.unit_tasks
    task_active = tasks.status == TaskStatusV1.ACTIVE
    apply_task = tasks.task_type == TaskTypeV1.APPLY_FERTILIZER
    owned_apply = task_active & apply_task
    apply = owned_apply & states.unit_active[:, player]
    position = states.unit_pos[:, player]
    target = jnp.stack(
        (
            jnp.clip(tasks.target_x, 0, BOARD_SIZE - 1),
            jnp.clip(tasks.target_y, 0, BOARD_SIZE - 1),
        ),
        axis=-1,
    ).astype(jnp.int8)
    at_target = jnp.all(position == target, axis=-1)
    depot, _ = nearest_shed_access_v1(position)
    at_depot = jnp.all(position == depot, axis=-1)
    carried = states.unit_inventory[:, player, :, FERTILIZER_ITEM] > 0
    shed_has = states.shed[:, player, FERTILIZER_ITEM] > 0
    toward_target = movement_op_toward_v1(position, target)
    toward_depot = movement_op_toward_v1(position, depot)
    normal_intended = jnp.where(
        carried,
        jnp.where(at_target, UnitOp.FERTILIZE, toward_target),
        jnp.where(
            shed_has[:, None],
            jnp.where(at_depot, UnitOp.PICKUP, toward_depot),
            UnitOp.PASS,
        ),
    ).astype(jnp.int8)
    last_turn = ((states.step + 1) % 24) == 0
    useful_on_last_turn = carried & at_target
    intended = jnp.where(
        last_turn[:, None] & (~useful_on_last_turn), UnitOp.PASS, normal_intended
    ).astype(jnp.int8)
    unit_op = jnp.where(apply, intended, UnitOp.PASS).astype(jnp.int8)
    unit_item = jnp.where(
        apply & ((unit_op == UnitOp.PICKUP) | (unit_op == UnitOp.FERTILIZE)),
        FERTILIZER_ITEM,
        -1,
    ).astype(jnp.int8)
    unit_amount = jnp.ones((batch_size, MAX_UNITS), dtype=jnp.int32)
    unit_slot = jnp.arange(MAX_UNITS, dtype=jnp.int16)[None, :]
    unit_count = jnp.maximum(
        1,
        jnp.max(jnp.where(states.unit_active[:, player], unit_slot + 1, 0), axis=-1),
    ).astype(jnp.int8)

    market_active = controller.market_tasks.status == TaskStatusV1.ACTIVE
    market_task = controller.market_tasks.task_type
    land = market_active & (market_task == TaskTypeV1.BUY_LAND)
    buy_fertilizer = (
        market_active
        & (market_task == TaskTypeV1.BUY_PRODUCT)
        & (controller.market_tasks.item_id == FERTILIZER_ITEM)
    )
    market_op = jnp.where(
        land,
        MarketOp.BUY_LAND,
        jnp.where(buy_fertilizer, MarketOp.BUY_PRODUCT, MarketOp.NONE),
    ).astype(jnp.int8)
    market_item = jnp.where(
        buy_fertilizer, FERTILIZER_ITEM, -1
    ).astype(jnp.int8)
    market_amount = jnp.where(
        market_active,
        jnp.maximum(controller.market_tasks.quantity, 1),
        0,
    ).astype(jnp.int32)
    market_slot = jnp.arange(MAX_MARKET_ORDERS, dtype=jnp.int8)[None, :]
    market_count = jnp.max(
        jnp.where(market_op != MarketOp.NONE, market_slot + 1, 0), axis=-1
    ).astype(jnp.int8)

    valid_unit = (unit_op >= UnitOp.PASS) & (unit_op <= UnitOp.CARE)
    valid_market = (market_op >= MarketOp.NONE) & (market_op <= MarketOp.SELL)
    deliberate_day_boundary_wait = apply & last_turn[:, None] & (~useful_on_last_turn)
    unexpected_pass = apply & (unit_op == UnitOp.PASS) & (~deliberate_day_boundary_wait)
    diagnostics = CompileDiagnosticsV1(
        invalid_raw_action_count=(
            jnp.sum(~valid_unit, axis=-1, dtype=jnp.int32)
            + jnp.sum(~valid_market, axis=-1, dtype=jnp.int32)
        ),
        unexpected_pass_count=jnp.sum(unexpected_pass, axis=-1, dtype=jnp.int32),
        land_orders=jnp.sum(land, axis=-1, dtype=jnp.int32),
        fertilizer_buy_orders=jnp.sum(buy_fertilizer, axis=-1, dtype=jnp.int32),
        fertilizer_pickup_actions=jnp.sum(
            unit_op == UnitOp.PICKUP, axis=-1, dtype=jnp.int32
        ),
        fertilizer_apply_actions=jnp.sum(
            unit_op == UnitOp.FERTILIZE, axis=-1, dtype=jnp.int32
        ),
    )
    return E2PlayerActionV1(
        unit_op=unit_op,
        unit_item=unit_item,
        unit_amount=unit_amount,
        unit_count=unit_count,
        market_op=market_op,
        market_item=market_item,
        market_amount=market_amount,
        market_count=market_count,
        diagnostics=diagnostics,
    )


def compile_e2_action_bundle_v1(
    states: State,
    player0_controller: ControllerStateV1,
    player1_controller: ControllerStateV1,
) -> E2ActionBundleV1:
    player0 = compile_e2_player_action_v1(states, player0_controller, 0)
    player1 = compile_e2_player_action_v1(states, player1_controller, 1)
    action = Action(
        unit_op=jnp.stack((player0.unit_op, player1.unit_op), axis=1),
        unit_item=jnp.stack((player0.unit_item, player1.unit_item), axis=1),
        unit_amount=jnp.stack((player0.unit_amount, player1.unit_amount), axis=1),
        unit_count=jnp.stack((player0.unit_count, player1.unit_count), axis=1),
        market_op=jnp.stack((player0.market_op, player1.market_op), axis=1),
        market_item=jnp.stack((player0.market_item, player1.market_item), axis=1),
        market_amount=jnp.stack((player0.market_amount, player1.market_amount), axis=1),
        market_count=jnp.stack((player0.market_count, player1.market_count), axis=1),
    )
    diagnostics = jax.tree.map(
        lambda left, right: jnp.stack((left, right), axis=1),
        player0.diagnostics,
        player1.diagnostics,
    )
    return E2ActionBundleV1(action, diagnostics, player0, player1)


def update_e2_controller_from_effects_v1(
    states: State,
    next_states: State,
    controller: ControllerStateV1,
    player_action: E2PlayerActionV1,
    player: int,
) -> tuple[ControllerStateV1, EffectDiagnosticsV1]:
    """Advance E2 tasks only after their official state effect is visible."""

    tasks = controller.unit_tasks
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    x = jnp.clip(tasks.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(tasks.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    task_active = tasks.status == TaskStatusV1.ACTIVE
    apply_task = tasks.task_type == TaskTypeV1.APPLY_FERTILIZER
    owned_apply = task_active & apply_task
    apply = owned_apply & states.unit_active[:, player]
    op = player_action.unit_op
    owner_inactive = owned_apply & (~states.unit_active[:, player])
    day_end = (next_states.step % 24) == 0
    moving = apply & (
        (op == UnitOp.NORTH)
        | (op == UnitOp.SOUTH)
        | (op == UnitOp.EAST)
        | (op == UnitOp.WEST)
    )
    predicted = states.unit_pos[:, player].astype(jnp.int16) + jnp.stack(
        (
            (op == UnitOp.EAST).astype(jnp.int16)
            - (op == UnitOp.WEST).astype(jnp.int16),
            (op == UnitOp.SOUTH).astype(jnp.int16)
            - (op == UnitOp.NORTH).astype(jnp.int16),
        ),
        axis=-1,
    )
    move_success = moving & (
        jnp.all(next_states.unit_pos[:, player] == predicted, axis=-1)
        | day_end[:, None]
    )
    pre_inventory = states.unit_inventory[:, player, :, FERTILIZER_ITEM]
    post_inventory = next_states.unit_inventory[:, player, :, FERTILIZER_ITEM]
    pickup_attempt = apply & (op == UnitOp.PICKUP)
    pickup_success = pickup_attempt & (post_inventory > pre_inventory)
    apply_attempt = apply & (op == UnitOp.FERTILIZE)
    pre_until = states.tile_fertilized_until[batch, player, y, x].astype(jnp.int16)
    post_until = next_states.tile_fertilized_until[batch, player, y, x].astype(jnp.int16)
    expected_until = jnp.maximum(
        pre_until, (states.step[:, None] // 24).astype(jnp.int16) + 2
    )
    apply_success = (
        apply_attempt
        & (pre_inventory > post_inventory)
        & (post_until == expected_until)
    )
    no_resource = (
        apply
        & (op == UnitOp.PASS)
        & (pre_inventory <= 0)
        & (states.shed[:, player, FERTILIZER_ITEM, None] <= 0)
    )
    expected_effect = pickup_attempt | apply_attempt
    effect_success = pickup_success | apply_success
    mismatch = expected_effect & (~effect_success)
    deadline = (
        owned_apply
        & (states.step[:, None] >= tasks.deadline_step)
        & (~(move_success | pickup_success | apply_success))
    )
    progress = move_success | pickup_success | apply_success
    failed = owner_inactive | no_resource | mismatch | deadline
    status = jnp.where(
        apply_success,
        TaskStatusV1.DONE,
        jnp.where(failed, TaskStatusV1.FAILED, tasks.status),
    ).astype(jnp.int8)
    failure = jnp.where(
        owner_inactive,
        FailureCodeV1.OWNER_INACTIVE,
        jnp.where(
            no_resource,
            FailureCodeV1.RESOURCE_UNAVAILABLE,
            jnp.where(
                deadline,
                FailureCodeV1.DEADLINE_MISSED,
                jnp.where(mismatch, FailureCodeV1.EFFECT_MISMATCH, tasks.failure_code),
            ),
        ),
    ).astype(jnp.int8)
    phase = jnp.where(
        pickup_success,
        TaskPhaseV1.MOVE_TO_TARGET,
        jnp.where(apply_attempt, TaskPhaseV1.OPERATE, tasks.phase),
    ).astype(jnp.int8)
    unit_tasks = tasks._replace(
        phase=phase,
        last_progress_step=jnp.where(
            progress, next_states.step[:, None], tasks.last_progress_step
        ).astype(jnp.int16),
        status=status,
        failure_code=failure,
    )

    market_tasks = controller.market_tasks
    market_active = market_tasks.status == TaskStatusV1.ACTIVE
    land_attempt = market_active & (market_tasks.task_type == TaskTypeV1.BUY_LAND)
    buy_attempt = (
        market_active
        & (market_tasks.task_type == TaskTypeV1.BUY_PRODUCT)
        & (market_tasks.item_id == FERTILIZER_ITEM)
    )
    land_effect = next_states.unlocked_count[:, player] > states.unlocked_count[:, player]
    successful_pickups = jnp.sum(pickup_success, axis=-1, dtype=jnp.int16)
    expected_shed_after_pickup = (
        states.shed[:, player, FERTILIZER_ITEM].astype(jnp.int16) - successful_pickups
    )
    bought_units = jnp.maximum(
        next_states.shed[:, player, FERTILIZER_ITEM].astype(jnp.int16)
        - expected_shed_after_pickup,
        0,
    )
    buy_effect = bought_units > 0
    land_success = land_attempt & land_effect[:, None]
    buy_success = buy_attempt & buy_effect[:, None]
    market_expected = land_attempt | buy_attempt
    market_success = land_success | buy_success
    market_deadline = market_expected & (
        states.step[:, None] >= market_tasks.deadline_step
    ) & (~market_success)
    market_mismatch = market_expected & (~market_success) & (~market_deadline)
    market_status = jnp.where(
        market_success,
        TaskStatusV1.DONE,
        jnp.where(
            market_deadline | market_mismatch,
            TaskStatusV1.FAILED,
            market_tasks.status,
        ),
    ).astype(jnp.int8)
    market_failure = jnp.where(
        market_deadline,
        FailureCodeV1.DEADLINE_MISSED,
        jnp.where(
            market_mismatch,
            FailureCodeV1.EFFECT_MISMATCH,
            market_tasks.failure_code,
        ),
    ).astype(jnp.int8)
    market_tasks = market_tasks._replace(
        status=market_status, failure_code=market_failure
    )
    updated = controller._replace(
        unit_tasks=unit_tasks, market_tasks=market_tasks
    )
    diagnostics = EffectDiagnosticsV1(
        effect_mismatch_count=(
            jnp.sum(mismatch, axis=-1, dtype=jnp.int32)
            + jnp.sum(market_mismatch, axis=-1, dtype=jnp.int32)
        ),
        owner_inactive_count=jnp.sum(owner_inactive, axis=-1, dtype=jnp.int32),
        deadline_missed_count=(
            jnp.sum(deadline, axis=-1, dtype=jnp.int32)
            + jnp.sum(market_deadline, axis=-1, dtype=jnp.int32)
        ),
        resource_unavailable_count=jnp.sum(no_resource, axis=-1, dtype=jnp.int32),
        land_purchase_success_count=jnp.sum(
            land_success, axis=-1, dtype=jnp.int32
        ),
        fertilizer_buy_success_count=jnp.sum(
            buy_success, axis=-1, dtype=jnp.int32
        ),
        fertilizer_pickup_success_count=jnp.sum(
            pickup_success, axis=-1, dtype=jnp.int32
        ),
        fertilizer_apply_success_count=jnp.sum(
            apply_success, axis=-1, dtype=jnp.int32
        ),
    )
    return updated, diagnostics
