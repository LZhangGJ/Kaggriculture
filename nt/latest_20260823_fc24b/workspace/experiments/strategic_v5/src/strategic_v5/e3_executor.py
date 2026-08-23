"""E3 persistent animal executor, compiler, and effect checker."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_PRODUCT,
    ANIMAL_STRUCTURE,
    BOARD_SIZE,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.types import Action, State

from .constants import FailureCodeV1, TaskPhaseV1, TaskStatusV1, TaskTypeV1
from .geometry import movement_op_toward_v1, nearest_shed_access_v1
from .schema import ControllerStateV1


WHEAT_ITEM = 0
FERTILIZER_ITEM = 8
_ANIMAL_STRUCTURE = jnp.asarray(ANIMAL_STRUCTURE, dtype=jnp.int8)
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int8)


class AnimalCompileDiagnosticsV1(NamedTuple):
    invalid_raw_action_count: jax.Array
    unexpected_pass_count: jax.Array
    build_actions: jax.Array
    pickup_actions: jax.Array
    place_actions: jax.Array
    feed_actions: jax.Array
    care_actions: jax.Array
    harvest_actions: jax.Array
    collect_fertilizer_actions: jax.Array
    animal_buy_orders: jax.Array
    wheat_buy_orders: jax.Array
    product_sell_orders: jax.Array


class AnimalEffectDiagnosticsV1(NamedTuple):
    effect_mismatch_count: jax.Array
    owner_inactive_count: jax.Array
    deadline_missed_count: jax.Array
    resource_unavailable_count: jax.Array
    build_success_count: jax.Array
    pickup_success_count: jax.Array
    place_success_count: jax.Array
    feed_success_count: jax.Array
    care_success_count: jax.Array
    harvest_success_count: jax.Array
    collect_fertilizer_success_count: jax.Array
    deposit_success_count: jax.Array
    animal_purchase_success_count: jax.Array
    wheat_purchase_success_count: jax.Array
    sold_product_units: jax.Array


class E3PlayerActionV1(NamedTuple):
    unit_op: jax.Array
    unit_item: jax.Array
    unit_amount: jax.Array
    unit_count: jax.Array
    market_op: jax.Array
    market_item: jax.Array
    market_amount: jax.Array
    market_count: jax.Array
    diagnostics: AnimalCompileDiagnosticsV1


class E3ActionBundleV1(NamedTuple):
    action: Action
    diagnostics: AnimalCompileDiagnosticsV1
    player0: E3PlayerActionV1
    player1: E3PlayerActionV1


def compile_e3_player_action_v1(
    states: State,
    controller: ControllerStateV1,
    player: int,
    *,
    auto_sell_products: bool = True,
) -> E3PlayerActionV1:
    batch_size = states.step.shape[0]
    tasks = controller.unit_tasks
    task_active = tasks.status == TaskStatusV1.ACTIVE
    owner_active = states.unit_active[:, player]
    active = task_active & owner_active
    task = tasks.task_type
    supported_type = (
        (task == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
        | (task == TaskTypeV1.ANIMAL_PLACE)
        | (task == TaskTypeV1.ANIMAL_FEED)
        | (task == TaskTypeV1.ANIMAL_CARE)
        | (task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
        | (task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
    )
    owned_supported = task_active & supported_type
    build = active & (task == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
    place = active & (task == TaskTypeV1.ANIMAL_PLACE)
    feed = active & (task == TaskTypeV1.ANIMAL_FEED)
    care = active & (task == TaskTypeV1.ANIMAL_CARE)
    harvest = active & (task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
    collect = active & (task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
    supported = build | place | feed | care | harvest | collect
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
    toward_target = movement_op_toward_v1(position, target)
    toward_depot = movement_op_toward_v1(position, depot)
    safe_item = jnp.clip(tasks.item_id.astype(jnp.int32), 0, NUM_SHED_ITEMS - 1)
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    unit = jnp.arange(MAX_UNITS, dtype=jnp.int32)[None, :]
    carried = states.unit_inventory[batch, player, unit, safe_item] > 0
    shed_has = states.shed[batch, player, safe_item] > 0
    returning = (harvest | collect) & (tasks.phase == TaskPhaseV1.MOVE_TO_DEPOT)

    animal_id_for_build = jnp.clip(tasks.item_id.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    build_op = jnp.where(
        _ANIMAL_STRUCTURE[animal_id_for_build] == TileKind.COOP,
        UnitOp.BUILD_COOP,
        UnitOp.BUILD_PASTURE,
    )
    build_intended = jnp.where(at_target, build_op, toward_target)
    place_intended = jnp.where(
        carried,
        jnp.where(at_target, UnitOp.PLACE, toward_target),
        jnp.where(shed_has, jnp.where(at_depot, UnitOp.PICKUP, toward_depot), UnitOp.PASS),
    )
    feed_intended = jnp.where(
        carried,
        jnp.where(at_target, UnitOp.FEED, toward_target),
        jnp.where(shed_has, jnp.where(at_depot, UnitOp.PICKUP, toward_depot), UnitOp.PASS),
    )
    care_intended = jnp.where(at_target, UnitOp.CARE, toward_target)
    harvest_intended = jnp.where(
        returning,
        jnp.where(at_depot, UnitOp.DROP, toward_depot),
        jnp.where(at_target, UnitOp.HARVEST, toward_target),
    )
    collect_intended = jnp.where(
        returning,
        jnp.where(at_depot, UnitOp.DROP, toward_depot),
        jnp.where(at_target, UnitOp.COLLECT_FERTILIZER, toward_target),
    )
    intended = jnp.where(
        build,
        build_intended,
        jnp.where(
            place,
            place_intended,
            jnp.where(
                feed,
                feed_intended,
                jnp.where(care, care_intended, jnp.where(harvest, harvest_intended, collect_intended)),
            ),
        ),
    ).astype(jnp.int8)
    last_turn = ((states.step + 1) % 24) == 0
    terminal_operation = (
        ((build_op == intended) & build)
        | ((intended == UnitOp.PLACE) & place)
        | ((intended == UnitOp.FEED) & feed)
        | ((intended == UnitOp.CARE) & care)
        | ((intended == UnitOp.DROP) & (harvest | collect))
    )
    intended = jnp.where(
        last_turn[:, None] & (~terminal_operation), UnitOp.PASS, intended
    ).astype(jnp.int8)
    unit_op = jnp.where(supported, intended, UnitOp.PASS).astype(jnp.int8)
    unit_item = jnp.where(
        (unit_op == UnitOp.PICKUP) | (unit_op == UnitOp.PLACE), tasks.item_id, -1
    ).astype(jnp.int8)
    # FEED tasks may reserve a bounded pickup batch.  The executor still feeds
    # one animal per atomic FEED action; only the preceding shed pickup uses
    # the task quantity, allowing the same worker to continue to later feed
    # tasks without an unnecessary depot round trip.
    unit_amount = jnp.where(
        (unit_op == UnitOp.PICKUP) & feed,
        jnp.maximum(tasks.quantity.astype(jnp.int32), 1),
        1,
    ).astype(jnp.int32)
    unit_slot = jnp.arange(MAX_UNITS, dtype=jnp.int16)[None, :]
    unit_count = jnp.maximum(
        1,
        jnp.max(jnp.where(states.unit_active[:, player], unit_slot + 1, 0), axis=-1),
    ).astype(jnp.int8)

    market_tasks = controller.market_tasks
    market_active = market_tasks.status == TaskStatusV1.ACTIVE
    animal_buy = market_active & (market_tasks.task_type == TaskTypeV1.ANIMAL_PURCHASE)
    wheat_buy = (
        market_active
        & (market_tasks.task_type == TaskTypeV1.BUY_PRODUCT)
        & (market_tasks.item_id == WHEAT_ITEM)
    )
    market_op = jnp.where(
        animal_buy,
        MarketOp.BUY_ANIMAL,
        jnp.where(wheat_buy, MarketOp.BUY_PRODUCT, MarketOp.NONE),
    ).astype(jnp.int8)
    market_item = jnp.where(animal_buy | wheat_buy, market_tasks.item_id, -1).astype(jnp.int8)
    market_amount = jnp.where(
        animal_buy | wheat_buy, jnp.maximum(market_tasks.quantity, 1), 0
    ).astype(jnp.int32)

    dropping = unit_op == UnitOp.DROP
    drop_products = jnp.sum(
        states.unit_inventory[:, player, :, :NUM_PRODUCTS].astype(jnp.int32)
        * dropping[..., None],
        axis=1,
    )
    sell_items = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int8)
    sell_amount = states.shed[:, player, sell_items].astype(jnp.int32) + drop_products[:, sell_items]
    sell_active = sell_amount > 0
    if auto_sell_products:
        sell_slots = jnp.arange(3, dtype=jnp.int32) + (MAX_MARKET_ORDERS - 3)
        market_op = market_op.at[:, sell_slots].set(
            jnp.where(sell_active, MarketOp.SELL, MarketOp.NONE).astype(jnp.int8)
        )
        market_item = market_item.at[:, sell_slots].set(
            jnp.broadcast_to(sell_items, (batch_size, 3))
        )
        market_amount = market_amount.at[:, sell_slots].set(sell_amount)
    else:
        sell_active = jnp.zeros_like(sell_active)
    market_slot = jnp.arange(MAX_MARKET_ORDERS, dtype=jnp.int8)[None, :]
    market_count = jnp.max(
        jnp.where(market_op != MarketOp.NONE, market_slot + 1, 0), axis=-1
    ).astype(jnp.int8)

    deliberate_wait = supported & last_turn[:, None] & (~terminal_operation)
    unexpected_pass = supported & (unit_op == UnitOp.PASS) & (~deliberate_wait)
    valid_unit = (unit_op >= UnitOp.PASS) & (unit_op <= UnitOp.CARE)
    valid_market = (market_op >= MarketOp.NONE) & (market_op <= MarketOp.SELL)
    diagnostics = AnimalCompileDiagnosticsV1(
        invalid_raw_action_count=(
            jnp.sum(~valid_unit, axis=-1, dtype=jnp.int32)
            + jnp.sum(~valid_market, axis=-1, dtype=jnp.int32)
        ),
        unexpected_pass_count=jnp.sum(unexpected_pass, axis=-1, dtype=jnp.int32),
        build_actions=jnp.sum(
            (unit_op == UnitOp.BUILD_COOP) | (unit_op == UnitOp.BUILD_PASTURE),
            axis=-1,
            dtype=jnp.int32,
        ),
        pickup_actions=jnp.sum(unit_op == UnitOp.PICKUP, axis=-1, dtype=jnp.int32),
        place_actions=jnp.sum(unit_op == UnitOp.PLACE, axis=-1, dtype=jnp.int32),
        feed_actions=jnp.sum(unit_op == UnitOp.FEED, axis=-1, dtype=jnp.int32),
        care_actions=jnp.sum(unit_op == UnitOp.CARE, axis=-1, dtype=jnp.int32),
        harvest_actions=jnp.sum(unit_op == UnitOp.HARVEST, axis=-1, dtype=jnp.int32),
        collect_fertilizer_actions=jnp.sum(
            unit_op == UnitOp.COLLECT_FERTILIZER, axis=-1, dtype=jnp.int32
        ),
        animal_buy_orders=jnp.sum(animal_buy, axis=-1, dtype=jnp.int32),
        wheat_buy_orders=jnp.sum(wheat_buy, axis=-1, dtype=jnp.int32),
        product_sell_orders=jnp.sum(sell_active, axis=-1, dtype=jnp.int32),
    )
    return E3PlayerActionV1(
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


def compile_e3_action_bundle_v1(
    states: State,
    player0_controller: ControllerStateV1,
    player1_controller: ControllerStateV1,
) -> E3ActionBundleV1:
    player0 = compile_e3_player_action_v1(states, player0_controller, 0)
    player1 = compile_e3_player_action_v1(states, player1_controller, 1)
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
    return E3ActionBundleV1(action, diagnostics, player0, player1)


def update_e3_controller_from_effects_v1(
    states: State,
    next_states: State,
    controller: ControllerStateV1,
    player_action: E3PlayerActionV1,
    player: int,
) -> tuple[ControllerStateV1, AnimalEffectDiagnosticsV1]:
    tasks = controller.unit_tasks
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    unit = jnp.arange(MAX_UNITS, dtype=jnp.int32)[None, :]
    x = jnp.clip(tasks.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(tasks.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    task = tasks.task_type
    task_active = tasks.status == TaskStatusV1.ACTIVE
    supported_type = (
        (task == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
        | (task == TaskTypeV1.ANIMAL_PLACE)
        | (task == TaskTypeV1.ANIMAL_FEED)
        | (task == TaskTypeV1.ANIMAL_CARE)
        | (task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
        | (task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
    )
    owned_supported = task_active & supported_type
    active = owned_supported & states.unit_active[:, player]
    build = active & (task == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
    place = active & (task == TaskTypeV1.ANIMAL_PLACE)
    feed = active & (task == TaskTypeV1.ANIMAL_FEED)
    care = active & (task == TaskTypeV1.ANIMAL_CARE)
    harvest = active & (task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
    collect = active & (task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
    supported = build | place | feed | care | harvest | collect
    op = player_action.unit_op
    day_end = (next_states.step % 24) == 0
    owner_inactive = owned_supported & (~states.unit_active[:, player])
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
        jnp.all(next_states.unit_pos[:, player] == predicted, axis=-1)
        | day_end[:, None]
    )
    safe_item = jnp.clip(tasks.item_id.astype(jnp.int32), 0, NUM_SHED_ITEMS - 1)
    pre_inventory = states.unit_inventory[batch, player, unit, safe_item]
    post_inventory = next_states.unit_inventory[batch, player, unit, safe_item]
    pre_kind = states.tile_kind[batch, player, y, x]
    post_kind = next_states.tile_kind[batch, player, y, x]
    pre_animal = states.tile_animal[batch, player, y, x]
    post_animal = next_states.tile_animal[batch, player, y, x]
    pre_flags = states.tile_flags[batch, player, y, x]
    post_flags = next_states.tile_flags[batch, player, y, x]
    pre_yield = states.tile_yield[batch, player, y, x]
    post_yield = next_states.tile_yield[batch, player, y, x]

    pickup_attempt = supported & (op == UnitOp.PICKUP)
    pickup_success = pickup_attempt & (post_inventory > pre_inventory)
    build_attempt = build & (
        (op == UnitOp.BUILD_COOP) | (op == UnitOp.BUILD_PASTURE)
    )
    build_success = build_attempt & (pre_kind == TileKind.EMPTY) & (
        (post_kind == TileKind.COOP) | (post_kind == TileKind.PASTURE)
    )
    animal_id = jnp.clip(tasks.item_id.astype(jnp.int32) - NUM_PRODUCTS, 0, NUM_ANIMALS - 1)
    place_attempt = place & (op == UnitOp.PLACE)
    place_success = place_attempt & (post_animal == animal_id) & (pre_animal < 0)
    feed_attempt = feed & (op == UnitOp.FEED)
    feed_success = feed_attempt & (
        jnp.where(
            day_end[:, None],
            (pre_animal >= 0)
            & (post_animal == pre_animal)
            & (pre_inventory > 0),
            (post_flags & jnp.uint8(FLAG_FED)) != 0,
        )
    )
    care_attempt = care & (op == UnitOp.CARE)
    care_success = care_attempt & (
        jnp.where(
            day_end[:, None],
            (pre_animal >= 0)
            & (
                (post_animal == pre_animal)
                | (
                    (post_animal < 0)
                    & (
                        states.tile_neglect[batch, player, y, x].astype(jnp.int16)
                        + 1
                        >= 2
                    )
                )
            ),
            (post_flags & jnp.uint8(FLAG_CARED)) != 0,
        )
    )
    harvest_attempt = harvest & (op == UnitOp.HARVEST)
    harvest_success = harvest_attempt & (pre_yield > 0) & (post_yield < pre_yield)
    collect_attempt = collect & (op == UnitOp.COLLECT_FERTILIZER)
    collect_success = collect_attempt & (
        (pre_flags & jnp.uint8(FLAG_FERTILIZER_AVAILABLE)) != 0
    ) & ((post_flags & jnp.uint8(FLAG_FERTILIZER_AVAILABLE)) == 0)
    drop_attempt = (harvest | collect) & (op == UnitOp.DROP)
    pre_total = jnp.sum(states.unit_inventory[:, player].astype(jnp.int32), axis=-1)
    post_total = jnp.sum(next_states.unit_inventory[:, player].astype(jnp.int32), axis=-1)
    drop_success = drop_attempt & (pre_total > 0) & (post_total == 0)
    expected_effect = (
        pickup_attempt
        | build_attempt
        | place_attempt
        | feed_attempt
        | care_attempt
        | harvest_attempt
        | collect_attempt
        | drop_attempt
    )
    effect_success = (
        pickup_success
        | build_success
        | place_success
        | feed_success
        | care_success
        | harvest_success
        | collect_success
        | drop_success
    )
    mismatch = expected_effect & (~effect_success)
    item_task = place | feed
    no_resource = (
        item_task
        & (op == UnitOp.PASS)
        & (pre_inventory <= 0)
        & (states.shed[batch, player, safe_item] <= 0)
    )
    deadline = (
        owned_supported
        & (states.step[:, None] >= tasks.deadline_step)
        & (~(move_success | effect_success))
    )
    failed = owner_inactive | mismatch | no_resource | deadline
    done = build_success | place_success | feed_success | care_success | drop_success
    phase = jnp.where(
        harvest_success | collect_success,
        TaskPhaseV1.MOVE_TO_DEPOT,
        jnp.where(pickup_success, TaskPhaseV1.MOVE_TO_TARGET, tasks.phase),
    ).astype(jnp.int8)
    progress = move_success | effect_success
    unit_status = jnp.where(
        done,
        TaskStatusV1.DONE,
        jnp.where(failed, TaskStatusV1.FAILED, tasks.status),
    ).astype(jnp.int8)
    unit_failure = jnp.where(
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
    unit_tasks = tasks._replace(
        phase=phase,
        last_progress_step=jnp.where(progress, next_states.step[:, None], tasks.last_progress_step),
        status=unit_status,
        failure_code=unit_failure,
    )

    market_tasks = controller.market_tasks
    market_active = market_tasks.status == TaskStatusV1.ACTIVE
    purchase_attempt = market_active & (market_tasks.task_type == TaskTypeV1.ANIMAL_PURCHASE)
    wheat_attempt = (
        market_active
        & (market_tasks.task_type == TaskTypeV1.BUY_PRODUCT)
        & (market_tasks.item_id == WHEAT_ITEM)
    )
    market_item = jnp.clip(market_tasks.item_id.astype(jnp.int32), 0, NUM_SHED_ITEMS - 1)
    successful_pickup_by_item = jnp.zeros(
        (batch_size, NUM_SHED_ITEMS), dtype=jnp.int16
    )
    successful_pickup_units = jnp.maximum(
        post_inventory.astype(jnp.int16) - pre_inventory.astype(jnp.int16), 0
    )
    successful_pickup_by_item = successful_pickup_by_item.at[
        batch, safe_item
    ].add(
        jnp.where(pickup_success, successful_pickup_units, 0).astype(jnp.int16)
    )
    expected_without_buy = (
        states.shed[batch, player, market_item]
        - successful_pickup_by_item[batch, market_item]
    )
    observed_gain = next_states.shed[batch, player, market_item] - expected_without_buy
    purchase_success = purchase_attempt & (observed_gain >= market_tasks.quantity)
    # BUY_PRODUCT is an official partial-fill order (buy up to quantity), so
    # any positive fill is a successful execution of the task card.
    wheat_success = wheat_attempt & (observed_gain > 0)
    market_success = purchase_success | wheat_success
    market_deadline = (purchase_attempt | wheat_attempt) & (
        states.step[:, None] >= market_tasks.deadline_step
    ) & (~market_success)
    market_mismatch = (purchase_attempt | wheat_attempt) & (~market_success) & (~market_deadline)
    market_status = jnp.where(
        market_success,
        TaskStatusV1.DONE,
        jnp.where(market_deadline | market_mismatch, TaskStatusV1.FAILED, market_tasks.status),
    ).astype(jnp.int8)
    market_failure = jnp.where(
        market_deadline,
        FailureCodeV1.DEADLINE_MISSED,
        jnp.where(market_mismatch, FailureCodeV1.EFFECT_MISMATCH, market_tasks.failure_code),
    ).astype(jnp.int8)
    market_tasks = market_tasks._replace(status=market_status, failure_code=market_failure)

    product_ids = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int32)
    pre_product_shed = jnp.take(
        states.shed[:, player], product_ids, axis=-1
    ).astype(jnp.int32)
    dropped_products = jnp.sum(
        jnp.take(
            states.unit_inventory[:, player], product_ids, axis=-1
        ).astype(jnp.int32)
        * drop_attempt[..., None],
        axis=1,
    )
    post_product_shed = jnp.take(
        next_states.shed[:, player], product_ids, axis=-1
    ).astype(jnp.int32)
    sold = jnp.sum(
        jnp.maximum(
            pre_product_shed + dropped_products - post_product_shed,
            0,
        ),
        axis=-1,
        dtype=jnp.int32,
    )
    updated = controller._replace(unit_tasks=unit_tasks, market_tasks=market_tasks)
    diagnostics = AnimalEffectDiagnosticsV1(
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
        build_success_count=jnp.sum(build_success, axis=-1, dtype=jnp.int32),
        pickup_success_count=jnp.sum(pickup_success, axis=-1, dtype=jnp.int32),
        place_success_count=jnp.sum(place_success, axis=-1, dtype=jnp.int32),
        feed_success_count=jnp.sum(feed_success, axis=-1, dtype=jnp.int32),
        care_success_count=jnp.sum(care_success, axis=-1, dtype=jnp.int32),
        harvest_success_count=jnp.sum(harvest_success, axis=-1, dtype=jnp.int32),
        collect_fertilizer_success_count=jnp.sum(
            collect_success, axis=-1, dtype=jnp.int32
        ),
        deposit_success_count=jnp.sum(drop_success, axis=-1, dtype=jnp.int32),
        animal_purchase_success_count=jnp.sum(
            market_success & purchase_attempt, axis=-1, dtype=jnp.int32
        ),
        wheat_purchase_success_count=jnp.sum(
            market_success & wheat_attempt, axis=-1, dtype=jnp.int32
        ),
        sold_product_units=sold,
    )
    return updated, diagnostics
