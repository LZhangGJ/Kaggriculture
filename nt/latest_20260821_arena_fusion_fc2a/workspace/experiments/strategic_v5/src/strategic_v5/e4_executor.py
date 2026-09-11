"""E4 unified Full-core compiler and effect-checked persistent executor."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    BOARD_SIZE,
    CROP_FIRST_YIELD_DAY,
    FLAG_WATERED,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_CROPS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    TURNS_PER_DAY,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.types import Action, State

from .constants import FailureCodeV1, TaskPhaseV1, TaskStatusV1, TaskTypeV1
from .e2_core import E2PlayerActionV1
from .e2_executor import compile_e2_player_action_v1, update_e2_controller_from_effects_v1
from .e3_executor import (
    E3PlayerActionV1,
    compile_e3_player_action_v1,
    update_e3_controller_from_effects_v1,
)
from .geometry import movement_op_toward_v1, nearest_shed_access_v1
from .schema import ControllerStateV1


_CROP_FIRST = jnp.asarray(CROP_FIRST_YIELD_DAY, dtype=jnp.int16)


class CropCompileDiagnosticsV1(NamedTuple):
    invalid_raw_action_count: jax.Array
    unexpected_pass_count: jax.Array
    plant_actions: jax.Array
    water_actions: jax.Array
    harvest_actions: jax.Array
    clear_actions: jax.Array
    pickup_actions: jax.Array
    deposit_actions: jax.Array
    seed_buy_orders: jax.Array
    hire_orders: jax.Array
    sell_orders: jax.Array
    terminal_sell_orders: jax.Array


class CropEffectDiagnosticsV1(NamedTuple):
    effect_mismatch_count: jax.Array
    owner_inactive_count: jax.Array
    deadline_missed_count: jax.Array
    resource_unavailable_count: jax.Array
    plant_success_count: jax.Array
    water_success_count: jax.Array
    harvest_success_count: jax.Array
    clear_success_count: jax.Array
    pickup_success_count: jax.Array
    deposit_success_count: jax.Array
    seed_purchase_success_count: jax.Array
    hire_success_count: jax.Array
    sold_product_units: jax.Array


class CropPlayerActionV1(NamedTuple):
    unit_op: jax.Array
    unit_item: jax.Array
    unit_amount: jax.Array
    unit_count: jax.Array
    market_op: jax.Array
    market_item: jax.Array
    market_amount: jax.Array
    market_count: jax.Array
    diagnostics: CropCompileDiagnosticsV1


class FullCoreCompileDiagnosticsV1(NamedTuple):
    invalid_raw_action_count: jax.Array
    unexpected_pass_count: jax.Array
    unit_compiler_overlap_count: jax.Array
    market_compiler_overlap_count: jax.Array


class FullCoreEffectDiagnosticsV1(NamedTuple):
    effect_mismatch_count: jax.Array
    owner_inactive_count: jax.Array
    deadline_missed_count: jax.Array
    resource_unavailable_count: jax.Array
    e2: object
    e3: object
    crop_inventory: CropEffectDiagnosticsV1


class E4PlayerActionV1(NamedTuple):
    unit_op: jax.Array
    unit_item: jax.Array
    unit_amount: jax.Array
    unit_count: jax.Array
    market_op: jax.Array
    market_item: jax.Array
    market_amount: jax.Array
    market_count: jax.Array
    diagnostics: FullCoreCompileDiagnosticsV1
    e2: E2PlayerActionV1
    e3: E3PlayerActionV1
    crop_inventory: CropPlayerActionV1


class E4ActionBundleV1(NamedTuple):
    action: Action
    diagnostics: FullCoreCompileDiagnosticsV1
    player0: E4PlayerActionV1
    player1: E4PlayerActionV1


def compile_crop_inventory_player_action_v1(
    states: State, controller: ControllerStateV1, player: int
) -> CropPlayerActionV1:
    """Compile crop, recovery, worker, explicit sell, and terminal tasks."""

    batch_size = states.step.shape[0]
    tasks = controller.unit_tasks
    task_record_active = tasks.status == TaskStatusV1.ACTIVE
    owner_active = states.unit_active[:, player]
    active = task_record_active & owner_active
    task = tasks.task_type
    crop_task = active & (task == TaskTypeV1.CROP_PRODUCTION)
    water_task = active & (task == TaskTypeV1.WATER_CROP)
    clear_task = active & (task == TaskTypeV1.CLEAR_OR_REMOVE_TILE)
    recovery_task = active & (
        (task == TaskTypeV1.SAFE_RECOVERY) | (task == TaskTypeV1.SHED_DEPOSIT)
    )
    pickup_task = active & (task == TaskTypeV1.SHED_PICKUP)
    supported = crop_task | water_task | clear_task | recovery_task | pickup_task
    position = states.unit_pos[:, player]
    target = jnp.stack(
        (
            jnp.clip(tasks.target_x, 0, BOARD_SIZE - 1),
            jnp.clip(tasks.target_y, 0, BOARD_SIZE - 1),
        ),
        axis=-1,
    ).astype(jnp.int8)
    at_target = jnp.all(position == target, axis=-1)
    toward_target = movement_op_toward_v1(position, target)
    depot, _ = nearest_shed_access_v1(position)
    at_depot = jnp.all(position == depot, axis=-1)
    toward_depot = movement_op_toward_v1(position, depot)
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    x = jnp.clip(tasks.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(tasks.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    tile_kind = states.tile_kind[batch, player, y, x]
    tile_crop = states.tile_crop[batch, player, y, x]
    tile_yield = states.tile_yield[batch, player, y, x]
    tile_flags = states.tile_flags[batch, player, y, x]
    tile_max_lifespan = states.tile_max_lifespan[batch, player, y, x].astype(
        jnp.int16
    )
    tile_origin = states.tile_origin_day[batch, player, y, x].astype(jnp.int16)
    safe_crop = jnp.clip(tile_crop.astype(jnp.int32), 0, NUM_CROPS - 1)
    current_day = (states.step[:, None] // TURNS_PER_DAY).astype(jnp.int16)
    mature = current_day - tile_origin >= _CROP_FIRST[safe_crop]
    returning = crop_task & (tasks.phase == TaskPhaseV1.MOVE_TO_DEPOT)
    crop_at_target = jnp.where(
        returning,
        UnitOp.DROP,
        jnp.where(
            tile_kind == TileKind.EMPTY,
            UnitOp.PLANT,
            jnp.where(
                (tile_kind == TileKind.PLANT) & mature & (tile_yield > 0),
                UnitOp.HARVEST,
                jnp.where(
                    (tile_kind == TileKind.PLANT)
                    & (tile_crop == tasks.item_id)
                    & ((tile_flags & jnp.uint8(FLAG_WATERED)) == 0),
                    UnitOp.WATER,
                    UnitOp.PASS,
                ),
            ),
        ),
    )
    crop_intended = jnp.where(
        returning,
        jnp.where(at_depot, UnitOp.DROP, toward_depot),
        jnp.where(at_target, crop_at_target, toward_target),
    )
    water_intended = jnp.where(at_target, UnitOp.WATER, toward_target)
    clear_intended = jnp.where(at_target, UnitOp.DIG, toward_target)
    recovery_intended = jnp.where(at_depot, UnitOp.DROP, toward_depot)
    pickup_intended = jnp.where(at_depot, UnitOp.PICKUP, toward_depot)
    intended = jnp.where(
        crop_task,
        crop_intended,
        jnp.where(
            water_task,
            water_intended,
            jnp.where(
                clear_task,
                clear_intended,
                jnp.where(recovery_task, recovery_intended, pickup_intended),
            ),
        ),
    ).astype(jnp.int8)
    last_turn = ((states.step + 1) % TURNS_PER_DAY) == 0
    terminal_operation = (
        (intended == UnitOp.WATER)
        | (intended == UnitOp.DIG)
        | (intended == UnitOp.DROP)
        | (intended == UnitOp.PICKUP)
    )
    intended = jnp.where(
        last_turn[:, None] & (~terminal_operation), UnitOp.PASS, intended
    ).astype(jnp.int8)
    unit_op = jnp.where(supported, intended, UnitOp.PASS).astype(jnp.int8)
    unit_item = jnp.where(
        (unit_op == UnitOp.PLANT) | (unit_op == UnitOp.PICKUP), tasks.item_id, -1
    ).astype(jnp.int8)
    unit_amount = jnp.where(
        unit_op == UnitOp.PICKUP, jnp.maximum(tasks.quantity, 1), 1
    ).astype(jnp.int32)

    # Official 1.32.7 evaluates the whole simultaneous plant demand against
    # seeds available *before* the market phase.  If demand exceeds stock it
    # blocks every plant of that crop.  Deterministically release only the
    # first N workers covered by current stock; the remaining persistent tasks
    # wait for the next step, after seed purchases have settled.
    seed_wait = jnp.zeros_like(unit_op, dtype=jnp.bool_)
    for crop_id in range(NUM_CROPS):
        wants_crop = (unit_op == UnitOp.PLANT) & (unit_item == crop_id)
        rank = jnp.cumsum(wants_crop.astype(jnp.int16), axis=1)
        seed_wait = seed_wait | (
            wants_crop
            & (rank > states.seeds[:, player, crop_id][:, None].astype(jnp.int16))
        )

    # Do not dispatch two crop mutations to the same tile in one synchronized
    # unit phase.  The second operation would observe the first operation's
    # effect and silently no-op (for example WATER + WATER).
    mutating = (
        (unit_op == UnitOp.PLANT)
        | (unit_op == UnitOp.WATER)
        | (unit_op == UnitOp.HARVEST)
        | (unit_op == UnitOp.DIG)
    )
    same_target = tasks.target_id[:, :, None] == tasks.target_id[:, None, :]
    earlier = jnp.tril(
        jnp.ones((MAX_UNITS, MAX_UNITS), dtype=jnp.bool_), k=-1
    )
    duplicate_tile_wait = mutating & jnp.any(
        same_target & mutating[:, None, :] & earlier[None, :, :], axis=2
    )
    expiring_water_wait = (
        (unit_op == UnitOp.WATER)
        & (tile_max_lifespan >= 0)
        & (states.step[:, None].astype(jnp.int16) >= tile_max_lifespan)
    )
    coordinated_wait = seed_wait | duplicate_tile_wait | expiring_water_wait
    unit_op = jnp.where(coordinated_wait, UnitOp.PASS, unit_op).astype(jnp.int8)
    unit_item = jnp.where(coordinated_wait, -1, unit_item).astype(jnp.int8)
    unit_amount = jnp.where(coordinated_wait, 1, unit_amount).astype(jnp.int32)
    unit_slot = jnp.arange(MAX_UNITS, dtype=jnp.int16)[None, :]
    unit_count = jnp.maximum(
        1,
        jnp.max(jnp.where(owner_active, unit_slot + 1, 0), axis=-1),
    ).astype(jnp.int8)

    market = controller.market_tasks
    market_active = market.status == TaskStatusV1.ACTIVE
    seed = market_active & (market.task_type == TaskTypeV1.CROP_PRODUCTION)
    hire = market_active & (market.task_type == TaskTypeV1.HIRE_WORKER)
    sell = market_active & (market.task_type == TaskTypeV1.SELL_INVENTORY)
    terminal_sell = market_active & (
        market.task_type == TaskTypeV1.TERMINAL_LIQUIDATION
    )
    market_op = jnp.where(
        seed,
        MarketOp.BUY_SEED,
        jnp.where(
            hire,
            MarketOp.HIRE,
            jnp.where(sell | terminal_sell, MarketOp.SELL, MarketOp.NONE),
        ),
    ).astype(jnp.int8)
    market_item = jnp.where(seed | sell | terminal_sell, market.item_id, -1).astype(jnp.int8)
    market_amount = jnp.where(
        seed | hire | sell | terminal_sell, jnp.maximum(market.quantity, 1), 0
    ).astype(jnp.int32)
    market_slot = jnp.arange(MAX_MARKET_ORDERS, dtype=jnp.int8)[None, :]
    market_count = jnp.max(
        jnp.where(market_op != MarketOp.NONE, market_slot + 1, 0), axis=-1
    ).astype(jnp.int8)

    deliberate_wait = supported & (
        (last_turn[:, None] & (~terminal_operation)) | coordinated_wait
    )
    completed_crop = (
        crop_task
        & (tile_kind == TileKind.PLANT)
        & (tile_crop == tasks.item_id)
        & ((tile_flags & jnp.uint8(FLAG_WATERED)) != 0)
        & (~returning)
        & (~mature | (tile_yield <= 0))
    )
    unexpected_pass = supported & (unit_op == UnitOp.PASS) & (~deliberate_wait) & (~completed_crop)
    valid_unit = (unit_op >= UnitOp.PASS) & (unit_op <= UnitOp.CARE)
    valid_market = (market_op >= MarketOp.NONE) & (market_op <= MarketOp.SELL)
    diagnostics = CropCompileDiagnosticsV1(
        invalid_raw_action_count=(
            jnp.sum(~valid_unit, axis=-1, dtype=jnp.int32)
            + jnp.sum(~valid_market, axis=-1, dtype=jnp.int32)
        ),
        unexpected_pass_count=jnp.sum(unexpected_pass, axis=-1, dtype=jnp.int32),
        plant_actions=jnp.sum(unit_op == UnitOp.PLANT, axis=-1, dtype=jnp.int32),
        water_actions=jnp.sum(unit_op == UnitOp.WATER, axis=-1, dtype=jnp.int32),
        harvest_actions=jnp.sum(unit_op == UnitOp.HARVEST, axis=-1, dtype=jnp.int32),
        clear_actions=jnp.sum(unit_op == UnitOp.DIG, axis=-1, dtype=jnp.int32),
        pickup_actions=jnp.sum(unit_op == UnitOp.PICKUP, axis=-1, dtype=jnp.int32),
        deposit_actions=jnp.sum(unit_op == UnitOp.DROP, axis=-1, dtype=jnp.int32),
        seed_buy_orders=jnp.sum(market_op == MarketOp.BUY_SEED, axis=-1, dtype=jnp.int32),
        hire_orders=jnp.sum(market_op == MarketOp.HIRE, axis=-1, dtype=jnp.int32),
        sell_orders=jnp.sum(
            (market_op == MarketOp.SELL) & sell, axis=-1, dtype=jnp.int32
        ),
        terminal_sell_orders=jnp.sum(
            (market_op == MarketOp.SELL) & terminal_sell,
            axis=-1,
            dtype=jnp.int32,
        ),
    )
    return CropPlayerActionV1(
        unit_op,
        unit_item,
        unit_amount,
        unit_count,
        market_op,
        market_item,
        market_amount,
        market_count,
        diagnostics,
    )


def _merge_module_actions(
    e2: E2PlayerActionV1,
    e3: E3PlayerActionV1,
    crop: CropPlayerActionV1,
) -> tuple[jax.Array, ...]:
    e2_unit = e2.unit_op != UnitOp.PASS
    e3_unit = e3.unit_op != UnitOp.PASS
    crop_unit = crop.unit_op != UnitOp.PASS
    unit_overlap = (
        e2_unit.astype(jnp.int8) + e3_unit.astype(jnp.int8) + crop_unit.astype(jnp.int8)
    ) > 1
    unit_op = jnp.where(e2_unit, e2.unit_op, jnp.where(e3_unit, e3.unit_op, crop.unit_op))
    unit_item = jnp.where(e2_unit, e2.unit_item, jnp.where(e3_unit, e3.unit_item, crop.unit_item))
    unit_amount = jnp.where(
        e2_unit, e2.unit_amount, jnp.where(e3_unit, e3.unit_amount, crop.unit_amount)
    )

    e2_market = e2.market_op != MarketOp.NONE
    e3_market = e3.market_op != MarketOp.NONE
    crop_market = crop.market_op != MarketOp.NONE
    market_overlap = (
        e2_market.astype(jnp.int8)
        + e3_market.astype(jnp.int8)
        + crop_market.astype(jnp.int8)
    ) > 1
    market_op = jnp.where(
        e2_market, e2.market_op, jnp.where(e3_market, e3.market_op, crop.market_op)
    )
    market_item = jnp.where(
        e2_market, e2.market_item, jnp.where(e3_market, e3.market_item, crop.market_item)
    )
    market_amount = jnp.where(
        e2_market,
        e2.market_amount,
        jnp.where(e3_market, e3.market_amount, crop.market_amount),
    )
    # Product purchases are partial-fill (buy up to N).  Execute exact/atomic
    # commitments first so an early partial fill cannot consume cash reserved
    # for a later seed, animal, land or hire card.  Stable sorting preserves
    # the selected order inside both groups and compacts NONE slots to the end.
    market_sort_key = jnp.where(
        market_op == MarketOp.NONE,
        2,
        jnp.where(market_op == MarketOp.BUY_PRODUCT, 1, 0),
    )
    sorted_market_order = jnp.argsort(market_sort_key, axis=-1, stable=True)
    identity_market_order = jnp.broadcast_to(
        jnp.arange(MAX_MARKET_ORDERS, dtype=jnp.int32)[None, :],
        sorted_market_order.shape,
    )
    has_partial_fill = jnp.any(market_op == MarketOp.BUY_PRODUCT, axis=-1)
    market_order = jnp.where(
        has_partial_fill[:, None], sorted_market_order, identity_market_order
    )
    market_op = jnp.take_along_axis(market_op, market_order, axis=-1)
    market_item = jnp.take_along_axis(market_item, market_order, axis=-1)
    market_amount = jnp.take_along_axis(market_amount, market_order, axis=-1)
    compact_count = jnp.sum(
        market_op != MarketOp.NONE, axis=-1, dtype=jnp.int8
    ).astype(jnp.int8)
    market_slot = jnp.arange(MAX_MARKET_ORDERS, dtype=jnp.int8)[None, :]
    sparse_count = jnp.max(
        jnp.where(market_op != MarketOp.NONE, market_slot + 1, 0), axis=-1
    ).astype(jnp.int8)
    market_count = jnp.where(has_partial_fill, compact_count, sparse_count)
    unit_count = jnp.maximum(jnp.maximum(e2.unit_count, e3.unit_count), crop.unit_count)
    return (
        unit_op.astype(jnp.int8),
        unit_item.astype(jnp.int8),
        unit_amount.astype(jnp.int32),
        unit_count.astype(jnp.int8),
        market_op.astype(jnp.int8),
        market_item.astype(jnp.int8),
        market_amount.astype(jnp.int32),
        market_count,
        jnp.sum(unit_overlap, axis=-1, dtype=jnp.int32),
        jnp.sum(market_overlap, axis=-1, dtype=jnp.int32),
    )


def compile_full_core_player_action_v1(
    states: State, controller: ControllerStateV1, player: int
) -> E4PlayerActionV1:
    e2 = compile_e2_player_action_v1(states, controller, player)
    e3 = compile_e3_player_action_v1(
        states, controller, player, auto_sell_products=False
    )
    crop = compile_crop_inventory_player_action_v1(states, controller, player)
    (
        unit_op,
        unit_item,
        unit_amount,
        unit_count,
        market_op,
        market_item,
        market_amount,
        market_count,
        unit_overlap,
        market_overlap,
    ) = _merge_module_actions(e2, e3, crop)
    diagnostics = FullCoreCompileDiagnosticsV1(
        invalid_raw_action_count=(
            e2.diagnostics.invalid_raw_action_count
            + e3.diagnostics.invalid_raw_action_count
            + crop.diagnostics.invalid_raw_action_count
        ),
        unexpected_pass_count=(
            e2.diagnostics.unexpected_pass_count
            + e3.diagnostics.unexpected_pass_count
            + crop.diagnostics.unexpected_pass_count
        ),
        unit_compiler_overlap_count=unit_overlap,
        market_compiler_overlap_count=market_overlap,
    )
    return E4PlayerActionV1(
        unit_op,
        unit_item,
        unit_amount,
        unit_count,
        market_op,
        market_item,
        market_amount,
        market_count,
        diagnostics,
        e2,
        e3,
        crop,
    )


def compile_full_core_action_bundle_v1(
    states: State,
    player0_controller: ControllerStateV1,
    player1_controller: ControllerStateV1,
) -> E4ActionBundleV1:
    player0 = compile_full_core_player_action_v1(states, player0_controller, 0)
    player1 = compile_full_core_player_action_v1(states, player1_controller, 1)
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
    return E4ActionBundleV1(action, diagnostics, player0, player1)


def update_crop_inventory_controller_from_effects_v1(
    states: State,
    next_states: State,
    controller: ControllerStateV1,
    player_action: CropPlayerActionV1,
    player: int,
    *,
    all_unit_op: jax.Array | None = None,
) -> tuple[ControllerStateV1, CropEffectDiagnosticsV1]:
    tasks = controller.unit_tasks
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    x = jnp.clip(tasks.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(tasks.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    task_record_active = tasks.status == TaskStatusV1.ACTIVE
    task = tasks.task_type
    supported_type = (
        (task == TaskTypeV1.CROP_PRODUCTION)
        | (task == TaskTypeV1.WATER_CROP)
        | (task == TaskTypeV1.CLEAR_OR_REMOVE_TILE)
        | (task == TaskTypeV1.SAFE_RECOVERY)
        | (task == TaskTypeV1.SHED_DEPOSIT)
        | (task == TaskTypeV1.SHED_PICKUP)
    )
    supported = task_record_active & supported_type
    op = player_action.unit_op
    pre_kind = states.tile_kind[batch, player, y, x]
    post_kind = next_states.tile_kind[batch, player, y, x]
    pre_crop = states.tile_crop[batch, player, y, x]
    post_crop = next_states.tile_crop[batch, player, y, x]
    pre_flags = states.tile_flags[batch, player, y, x]
    post_flags = next_states.tile_flags[batch, player, y, x]
    pre_yield = states.tile_yield[batch, player, y, x]
    post_yield = next_states.tile_yield[batch, player, y, x]
    day_end = (next_states.step % TURNS_PER_DAY) == 0
    moving = supported & (
        (op == UnitOp.NORTH)
        | (op == UnitOp.SOUTH)
        | (op == UnitOp.EAST)
        | (op == UnitOp.WEST)
    )
    predicted = states.unit_pos[:, player].astype(jnp.int16) + jnp.stack(
        (
            (op == UnitOp.EAST).astype(jnp.int16) - (op == UnitOp.WEST).astype(jnp.int16),
            (op == UnitOp.SOUTH).astype(jnp.int16) - (op == UnitOp.NORTH).astype(jnp.int16),
        ),
        axis=-1,
    )
    move_success = moving & (
        jnp.all(next_states.unit_pos[:, player] == predicted, axis=-1) | day_end[:, None]
    )
    plant_attempt = supported & (op == UnitOp.PLANT)
    plant_success = plant_attempt & (post_kind == TileKind.PLANT) & (post_crop == tasks.item_id)
    water_attempt = supported & (op == UnitOp.WATER)
    water_success = water_attempt & jnp.where(
        day_end[:, None],
        pre_kind == TileKind.PLANT,
        (post_flags & jnp.uint8(FLAG_WATERED)) != 0,
    )
    harvest_attempt = supported & (op == UnitOp.HARVEST)
    harvest_success = harvest_attempt & (pre_yield > 0) & (
        (post_yield < pre_yield) | (post_kind != TileKind.PLANT)
    )
    clear_attempt = supported & (op == UnitOp.DIG)
    clear_success = clear_attempt & (pre_kind != TileKind.EMPTY) & jnp.where(
        day_end[:, None], True, post_kind == TileKind.EMPTY
    )
    pickup_attempt = supported & (op == UnitOp.PICKUP)
    safe_item = jnp.clip(tasks.item_id.astype(jnp.int32), 0, NUM_SHED_ITEMS - 1)
    unit = jnp.arange(MAX_UNITS, dtype=jnp.int32)[None, :]
    pre_item = states.unit_inventory[batch, player, unit, safe_item]
    post_item = next_states.unit_inventory[batch, player, unit, safe_item]
    pickup_success = pickup_attempt & (post_item > pre_item)
    drop_attempt = supported & (op == UnitOp.DROP)
    pre_inventory_total = jnp.sum(
        states.unit_inventory[:, player].astype(jnp.int32), axis=-1
    )
    post_inventory_total = jnp.sum(
        next_states.unit_inventory[:, player].astype(jnp.int32), axis=-1
    )
    drop_progress = drop_attempt & (post_inventory_total < pre_inventory_total)
    drop_success = drop_attempt & (pre_inventory_total > 0) & (post_inventory_total == 0)
    crop_returning = supported & (task == TaskTypeV1.CROP_PRODUCTION) & (
        tasks.phase == TaskPhaseV1.MOVE_TO_DEPOT
    )
    auto_drop_success = (
        crop_returning
        & day_end[:, None]
        & (pre_inventory_total > 0)
        & (post_inventory_total == 0)
    )
    harvest_auto_deposit = harvest_success & day_end[:, None] & (post_inventory_total == 0)
    completed_crop_pass = (
        supported
        & (task == TaskTypeV1.CROP_PRODUCTION)
        & (op == UnitOp.PASS)
        & (pre_kind == TileKind.PLANT)
        & (pre_crop == tasks.item_id)
        & ((pre_flags & jnp.uint8(FLAG_WATERED)) != 0)
    )
    expected_effect = plant_attempt | water_attempt | harvest_attempt | clear_attempt | pickup_attempt | drop_attempt
    effect_success = plant_success | water_success | harvest_success | clear_success | pickup_success | drop_progress
    shed_full = drop_attempt & (pre_inventory_total > 0) & (~drop_progress)
    mismatch = expected_effect & (~effect_success) & (~shed_full)
    owner_inactive = supported & (~states.unit_active[:, player])
    deadline_progress = (
        move_success
        | effect_success
        | auto_drop_success
        | harvest_auto_deposit
        | completed_crop_pass
    )
    safe_task_crop = jnp.clip(tasks.item_id.astype(jnp.int32), 0, NUM_CROPS - 1)
    waiting_for_seed = (
        supported
        & (task == TaskTypeV1.CROP_PRODUCTION)
        & (op == UnitOp.PASS)
        & (pre_kind == TileKind.EMPTY)
        & (states.seeds[batch, player, safe_task_crop] <= 0)
    )
    deadline = (
        supported
        & (states.step[:, None] >= tasks.deadline_step)
        & (~deadline_progress)
        & (~waiting_for_seed)
    )
    failed = owner_inactive | mismatch | shed_full | deadline
    depot, _ = nearest_shed_access_v1(
        jnp.stack((x, y), axis=-1).astype(jnp.int8)
    )
    harvest_next = harvest_success & (~harvest_auto_deposit)
    next_target_x = jnp.where(harvest_next, depot[..., 0], tasks.target_x)
    next_target_y = jnp.where(harvest_next, depot[..., 1], tasks.target_y)
    next_target_id = jnp.where(
        harvest_next,
        depot[..., 1].astype(jnp.int16) * BOARD_SIZE + depot[..., 0].astype(jnp.int16),
        tasks.target_id,
    )
    done = (
        water_success
        | clear_success
        | pickup_success
        | drop_success
        | auto_drop_success
        | harvest_auto_deposit
        | completed_crop_pass
    )
    phase = jnp.where(
        plant_success,
        TaskPhaseV1.OPERATE,
        jnp.where(
            harvest_next,
            TaskPhaseV1.MOVE_TO_DEPOT,
            jnp.where(pickup_success, TaskPhaseV1.OPERATE, tasks.phase),
        ),
    ).astype(jnp.int8)
    progress = move_success | effect_success | auto_drop_success | harvest_auto_deposit | completed_crop_pass
    status = jnp.where(
        done,
        TaskStatusV1.DONE,
        jnp.where(failed, TaskStatusV1.FAILED, tasks.status),
    ).astype(jnp.int8)
    failure = jnp.where(
        owner_inactive,
        FailureCodeV1.OWNER_INACTIVE,
        jnp.where(
            shed_full,
            FailureCodeV1.RESOURCE_UNAVAILABLE,
            jnp.where(
                deadline,
                FailureCodeV1.DEADLINE_MISSED,
                jnp.where(mismatch, FailureCodeV1.EFFECT_MISMATCH, tasks.failure_code),
            ),
        ),
    ).astype(jnp.int8)
    unit_tasks = tasks._replace(
        target_x=next_target_x.astype(jnp.int8),
        target_y=next_target_y.astype(jnp.int8),
        target_id=next_target_id.astype(jnp.int16),
        phase=phase,
        last_progress_step=jnp.where(progress, next_states.step[:, None], tasks.last_progress_step).astype(jnp.int16),
        status=status,
        failure_code=failure,
    )

    market = controller.market_tasks
    market_active = market.status == TaskStatusV1.ACTIVE
    seed_attempt = (
        market_active
        & (market.task_type == TaskTypeV1.CROP_PRODUCTION)
        & (player_action.market_op == MarketOp.BUY_SEED)
    )
    hire_attempt = (
        market_active
        & (market.task_type == TaskTypeV1.HIRE_WORKER)
        & (player_action.market_op == MarketOp.HIRE)
    )
    sell_attempt = market_active & (
        (market.task_type == TaskTypeV1.SELL_INVENTORY)
        | (market.task_type == TaskTypeV1.TERMINAL_LIQUIDATION)
    ) & (player_action.market_op == MarketOp.SELL)
    attempted_market = seed_attempt | hire_attempt | sell_attempt
    market_item = jnp.clip(market.item_id.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    plant_by_crop = jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int16)
    plant_crop = jnp.clip(tasks.item_id.astype(jnp.int32), 0, NUM_CROPS - 1)
    plant_by_crop = plant_by_crop.at[batch, plant_crop].add(plant_success.astype(jnp.int16))
    seed_pre = states.seeds[batch, player, jnp.clip(market.item_id, 0, NUM_CROPS - 1)]
    seed_post = next_states.seeds[batch, player, jnp.clip(market.item_id, 0, NUM_CROPS - 1)]
    expected_seed_without_buy = seed_pre - plant_by_crop[batch, jnp.clip(market.item_id, 0, NUM_CROPS - 1)]
    seed_success = seed_attempt & ((seed_post - expected_seed_without_buy) >= market.quantity)
    hire_delta = next_states.hires_today[:, player].astype(jnp.int16) - states.hires_today[:, player].astype(jnp.int16)
    hire_success = hire_attempt & (hire_delta[:, None] > 0)
    all_drop_attempt = (
        player_action.unit_op == UnitOp.DROP
        if all_unit_op is None
        else all_unit_op == UnitOp.DROP
    )
    # Every active unit, including the farmer, deposits its carried inventory
    # at day end.  Account for that official transition when reconstructing
    # same-step market fills; otherwise a successful SELL can look unchanged
    # because the sold units are immediately replaced by automatic deposits.
    auto_deposit_unit = (
        day_end[:, None]
        & states.unit_active[:, player]
        & (pre_inventory_total > 0)
        & (post_inventory_total == 0)
    )
    dropped_by_item = jnp.sum(
        states.unit_inventory[:, player, :, :NUM_PRODUCTS].astype(jnp.int32)
        * (all_drop_attempt | auto_deposit_unit)[..., None],
        axis=1,
    )
    pre_shed = states.shed[batch, player, market_item].astype(jnp.int32)
    post_shed = next_states.shed[batch, player, market_item].astype(jnp.int32)
    sold_units = jnp.maximum(
        pre_shed + dropped_by_item[batch, market_item] - post_shed, 0
    )
    sell_success = sell_attempt & (sold_units > 0)
    sell_unavailable = sell_attempt & (
        (pre_shed + dropped_by_item[batch, market_item]) <= 0
    )
    market_success = seed_success | hire_success | sell_success
    market_deadline = attempted_market & (states.step[:, None] >= market.deadline_step) & (~market_success)
    market_mismatch = attempted_market & (~market_success) & (~market_deadline) & (~sell_unavailable)
    terminal_empty_sell = next_states.done[:, None] & sell_unavailable
    market_status = jnp.where(
        market_success | terminal_empty_sell,
        TaskStatusV1.DONE,
        jnp.where(market_deadline | market_mismatch, TaskStatusV1.FAILED, market.status),
    ).astype(jnp.int8)
    market_failure = jnp.where(
        terminal_empty_sell,
        FailureCodeV1.NONE,
        jnp.where(
        market_deadline,
        FailureCodeV1.DEADLINE_MISSED,
        jnp.where(market_mismatch, FailureCodeV1.EFFECT_MISMATCH, market.failure_code),
        ),
    ).astype(jnp.int8)
    updated = controller._replace(
        unit_tasks=unit_tasks,
        market_tasks=market._replace(status=market_status, failure_code=market_failure),
    )
    diagnostics = CropEffectDiagnosticsV1(
        effect_mismatch_count=(
            jnp.sum(mismatch, axis=-1, dtype=jnp.int32)
            + jnp.sum(market_mismatch, axis=-1, dtype=jnp.int32)
        ),
        owner_inactive_count=jnp.sum(owner_inactive, axis=-1, dtype=jnp.int32),
        deadline_missed_count=(
            jnp.sum(deadline, axis=-1, dtype=jnp.int32)
            + jnp.sum(market_deadline, axis=-1, dtype=jnp.int32)
        ),
        resource_unavailable_count=(
            jnp.sum(shed_full, axis=-1, dtype=jnp.int32)
            + jnp.sum(sell_unavailable, axis=-1, dtype=jnp.int32)
        ),
        plant_success_count=jnp.sum(plant_success, axis=-1, dtype=jnp.int32),
        water_success_count=jnp.sum(water_success, axis=-1, dtype=jnp.int32),
        harvest_success_count=jnp.sum(harvest_success, axis=-1, dtype=jnp.int32),
        clear_success_count=jnp.sum(clear_success, axis=-1, dtype=jnp.int32),
        pickup_success_count=jnp.sum(pickup_success, axis=-1, dtype=jnp.int32),
        deposit_success_count=jnp.sum(
            drop_success | auto_drop_success | harvest_auto_deposit,
            axis=-1,
            dtype=jnp.int32,
        ),
        seed_purchase_success_count=jnp.sum(seed_success, axis=-1, dtype=jnp.int32),
        hire_success_count=jnp.sum(hire_success, axis=-1, dtype=jnp.int32),
        sold_product_units=jnp.sum(
            jnp.where(sell_attempt, sold_units, 0), axis=-1, dtype=jnp.int32
        ),
    )
    return updated, diagnostics


def update_full_core_controller_from_effects_v1(
    states: State,
    next_states: State,
    controller: ControllerStateV1,
    player_action: E4PlayerActionV1,
    player: int,
) -> tuple[ControllerStateV1, FullCoreEffectDiagnosticsV1]:
    after_e2, e2_diag = update_e2_controller_from_effects_v1(
        states, next_states, controller, player_action.e2, player
    )
    after_e3, e3_diag = update_e3_controller_from_effects_v1(
        states, next_states, after_e2, player_action.e3, player
    )
    updated, crop_diag = update_crop_inventory_controller_from_effects_v1(
        states,
        next_states,
        after_e3,
        player_action.crop_inventory,
        player,
        all_unit_op=player_action.unit_op,
    )
    diagnostics = FullCoreEffectDiagnosticsV1(
        effect_mismatch_count=(
            e2_diag.effect_mismatch_count
            + e3_diag.effect_mismatch_count
            + crop_diag.effect_mismatch_count
        ),
        owner_inactive_count=(
            e2_diag.owner_inactive_count
            + e3_diag.owner_inactive_count
            + crop_diag.owner_inactive_count
        ),
        deadline_missed_count=(
            e2_diag.deadline_missed_count
            + e3_diag.deadline_missed_count
            + crop_diag.deadline_missed_count
        ),
        resource_unavailable_count=(
            e2_diag.resource_unavailable_count
            + e3_diag.resource_unavailable_count
            + crop_diag.resource_unavailable_count
        ),
        e2=e2_diag,
        e3=e3_diag,
        crop_inventory=crop_diag,
    )
    return updated, diagnostics
