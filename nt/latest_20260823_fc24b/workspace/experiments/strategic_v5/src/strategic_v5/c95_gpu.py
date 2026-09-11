"""GPU-native port of the public C95 opponent.

C95 is a frozen 719-step choreography plus a small collection of public-state
controllers.  The functions here reproduce those controllers with fixed-shape
JAX state so the policy can participate in large batched arenas without Python
callbacks inside the rollout.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    MARKET_LUT_SIZE,
    MARKET_MIN_INVENTORY,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    PRODUCTS,
    SHED_ACCESS,
    TileKind,
    UnitOp,
    MarketOp,
)
from kaggriculture_jax.types import Action, State, StaticTables


PREMIUM_IDS = jnp.asarray(
    [PRODUCTS.index(name) for name in ("MELON", "STRAWBERRY", "MILK", "WOOL")],
    dtype=jnp.int8,
)
SELLABLE_IDS = jnp.asarray(
    [
        PRODUCTS.index(name)
        for name in (
            "STRAWBERRY",
            "MELON",
            "MILK",
            "WOOL",
            "EGG",
            "TOMATO",
            "CARROT",
            "WHEAT",
            "FERTILIZER",
        )
    ],
    dtype=jnp.int8,
)
WHEAT_ID = PRODUCTS.index("WHEAT")
FERTILIZER_ID = PRODUCTS.index("FERTILIZER")
_PREMIUM_WEIGHT = jnp.asarray([3.5, 2.0, 2.0, 3.2], dtype=jnp.float32)
_PREMIUM_BASE = jnp.asarray([250, 120, 160, 200], dtype=jnp.float32)
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)
_TERMINAL_SHED_ACCESS = jnp.asarray(((4, 4), (5, 4), (5, 5), (4, 5)), dtype=jnp.int16)
_BLOCKED_OPS = jnp.asarray(
    (UnitOp.BUILD_PASTURE, UnitOp.BUILD_COOP, UnitOp.PLANT, UnitOp.PLACE),
    dtype=jnp.int8,
)
_SAFE_BANK_OPS = jnp.asarray(
    (UnitOp.PASS, UnitOp.NORTH, UnitOp.SOUTH, UnitOp.EAST, UnitOp.WEST),
    dtype=jnp.int8,
)
_WEED_LAST_PLANNED_USE = jnp.asarray(
    (
        (599, 618, 621, 514, 587, 574, 642, 564, 569, 571),
        (594, 589, 643, 593, 586, 560, 620, 563, 568, 575),
        (573, 568, 563, 560, 14, 181, 566, 571, 637, 574),
        (612, 613, 608, 20, 9, 176, 198, 519, 610, 615),
        (618, 632, 129, 5, 4, 165, 165, 204, 594, 599),
        (617, 588, 583, 586, 581, -1, -1, -1, -1, -1),
        (610, 613, 608, 591, 630, -1, -1, -1, -1, -1),
        (636, 635, 634, 632, 635, -1, -1, -1, -1, -1),
        (254, 257, 639, 637, 640, -1, -1, -1, -1, -1),
        (263, 260, 257, 260, 259, -1, -1, -1, -1, -1),
    ),
    dtype=jnp.int16,
)
_LEXICAL_RANK = jnp.asarray(
    # CARROT, EGG, FERTILIZER, MELON, MILK, STRAWBERRY, TOMATO, WHEAT, WOOL
    [7, 0, 6, 5, 3, 8, 4, 2, 1],
    dtype=jnp.int8,
)


class C95PlayerCarryV1(NamedTuple):
    outer_confidence: jax.Array
    sbt_confidence: jax.Array
    c94_debt: jax.Array
    repair_op: jax.Array
    repair_item: jax.Array
    repair_amount: jax.Array
    repair_length: jax.Array


def initialize_c95_player_carry_v1(batch_size: int) -> C95PlayerCarryV1:
    queue = (batch_size, MAX_UNITS, 24)
    return C95PlayerCarryV1(
        outer_confidence=jnp.zeros((batch_size,), dtype=jnp.int8),
        sbt_confidence=jnp.zeros((batch_size,), dtype=jnp.int8),
        c94_debt=jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int16),
        repair_op=jnp.full(queue, UnitOp.PASS, dtype=jnp.int8),
        repair_item=jnp.full(queue, -1, dtype=jnp.int8),
        repair_amount=jnp.ones(queue, dtype=jnp.int32),
        repair_length=jnp.zeros((batch_size, MAX_UNITS), dtype=jnp.int8),
    )


def _raw_action(states: State, bank, skeleton_id: jax.Array) -> Action:
    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    return Action(
        bank.unit_op[skeleton_id, step],
        bank.unit_item[skeleton_id, step],
        bank.unit_amount[skeleton_id, step],
        bank.unit_count[skeleton_id, step],
        bank.market_op[skeleton_id, step],
        bank.market_item[skeleton_id, step],
        bank.market_amount[skeleton_id, step],
        bank.market_count[skeleton_id, step],
    )


def _profile_distance(states: State) -> jax.Array:
    kinds = states.tile_kind.astype(jnp.int32)
    crops = states.tile_crop.astype(jnp.int32)
    animals = states.tile_animal.astype(jnp.int32)
    count_columns = []
    # Exact alphabetical order used by sorted(counts) in the Python policy.
    count_columns.append(jnp.sum(crops == 1, axis=(2, 3)))  # CARROT
    count_columns.append(jnp.sum((animals < 0) & (crops < 0) & (kinds == TileKind.COOP), axis=(2, 3)))
    count_columns.append(jnp.sum(animals == 1, axis=(2, 3)))  # COW
    count_columns.append(jnp.sum(animals == 0, axis=(2, 3)))  # GOOSE
    count_columns.append(jnp.sum(crops == 4, axis=(2, 3)))  # MELON
    count_columns.append(jnp.sum((animals < 0) & (crops < 0) & (kinds == TileKind.PASTURE), axis=(2, 3)))
    count_columns.append(jnp.sum(animals == 2, axis=(2, 3)))  # SHEEP
    count_columns.append(jnp.sum(crops == 3, axis=(2, 3)))  # STRAWBERRY
    count_columns.append(jnp.sum(crops == 2, axis=(2, 3)))  # TOMATO
    count_columns.append(jnp.sum((animals < 0) & (crops < 0) & (kinds == TileKind.WEED), axis=(2, 3)))
    count_columns.append(jnp.sum(crops == 0, axis=(2, 3)))  # WHEAT
    counts = jnp.stack(count_columns, axis=-1).astype(jnp.int32)
    count_distance = jnp.sum(jnp.abs(counts[:, 0] - counts[:, 1]), axis=1)

    active = states.unit_active.astype(jnp.bool_)
    hands = jnp.sum(active, axis=2).astype(jnp.int32) - 1
    positions = states.unit_pos.astype(jnp.int32)
    pos_id = jnp.clip(positions[..., 1], 0, 9) * 10 + jnp.clip(positions[..., 0], 0, 9)
    position_hist = jax.nn.one_hot(pos_id, 100, dtype=jnp.int16) * active[..., None]
    position_hist = jnp.sum(position_hist, axis=2)
    positions_differ = jnp.any(position_hist[:, 0] != position_hist[:, 1], axis=1)
    return (
        jnp.abs(hands[:, 0] - hands[:, 1])
        + 3 * jnp.abs(states.unlocked_count[:, 0].astype(jnp.int32) - states.unlocked_count[:, 1].astype(jnp.int32))
        + count_distance
        + 2 * positions_differ.astype(jnp.int32)
    )


def _update_confidence(states: State, confidence: jax.Array, enabled: jax.Array) -> jax.Array:
    step = states.step.astype(jnp.int32)
    checkpoint = (step == 4) | (step == 24) | ((step >= 48) & (step % 24 == 0))
    update = enabled & checkpoint
    distance = _profile_distance(states)
    following = jnp.where(
        distance <= 1,
        jnp.minimum(confidence.astype(jnp.int32) + 1, 8),
        jnp.where(
            distance <= 4,
            jnp.maximum(confidence.astype(jnp.int32) - 1, 0),
            jnp.maximum(confidence.astype(jnp.int32) - 3, 0),
        ),
    ).astype(jnp.int8)
    return jnp.where(update, following, confidence).astype(jnp.int8)


def _compact_market(op, item, amount, count, keep):
    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    keep = keep & (slot < count[:, None])
    order = jnp.argsort(jnp.where(keep, slot, slot + MAX_MARKET_ORDERS), axis=1, stable=True)
    op = jnp.take_along_axis(op, order, axis=1)
    item = jnp.take_along_axis(item, order, axis=1)
    amount = jnp.take_along_axis(amount, order, axis=1)
    new_count = jnp.sum(keep, axis=1).astype(jnp.int8)
    active = slot < new_count[:, None]
    return (
        jnp.where(active, op, MarketOp.NONE).astype(jnp.int8),
        jnp.where(active, item, -1).astype(jnp.int8),
        jnp.where(active, amount, 0).astype(jnp.int32),
        new_count,
    )


def _append_market(op, item, amount, count, enabled, new_op, new_item, new_amount):
    batch = jnp.arange(op.shape[0])
    slot = jnp.clip(count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1)
    op = op.at[batch, slot].set(jnp.where(enabled, new_op, op[batch, slot]))
    item = item.at[batch, slot].set(jnp.where(enabled, new_item, item[batch, slot]))
    amount = amount.at[batch, slot].set(jnp.where(enabled, new_amount, amount[batch, slot]))
    return op, item, amount, (count + enabled.astype(jnp.int8)).astype(jnp.int8)


def _prepend_market(op, item, amount, count, enabled, new_op, new_item, new_amount):
    shifted_op = jnp.concatenate((jnp.full_like(op[:, :1], MarketOp.NONE), op[:, :-1]), axis=1)
    shifted_item = jnp.concatenate((jnp.full_like(item[:, :1], -1), item[:, :-1]), axis=1)
    shifted_amount = jnp.concatenate((jnp.zeros_like(amount[:, :1]), amount[:, :-1]), axis=1)
    op = jnp.where(enabled[:, None], shifted_op, op).at[:, 0].set(jnp.where(enabled, new_op, op[:, 0]))
    item = jnp.where(enabled[:, None], shifted_item, item).at[:, 0].set(jnp.where(enabled, new_item, item[:, 0]))
    amount = jnp.where(enabled[:, None], shifted_amount, amount).at[:, 0].set(jnp.where(enabled, new_amount, amount[:, 0]))
    count = jnp.where(enabled, jnp.minimum(count + 1, MAX_MARKET_ORDERS), count).astype(jnp.int8)
    return op, item, amount, count


def _sell_impact(states: State, tables: StaticTables, player: int, item, amount):
    safe_item = jnp.clip(item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    qty = jnp.maximum(amount.astype(jnp.int32), 0)
    held = jnp.take_along_axis(states.shed[:, player, :NUM_PRODUCTS], safe_item, axis=1).astype(jnp.int32)
    qty = jnp.where(held > 0, jnp.minimum(qty, held), qty)
    inventory = jnp.take_along_axis(states.market_inventory, safe_item, axis=1).astype(jnp.int32)
    quote = jnp.take_along_axis(states.market_price, safe_item, axis=1).astype(jnp.int32)
    lut = jnp.clip(inventory + qty - MARKET_MIN_INVENTORY, 0, MARKET_LUT_SIZE - 1)
    later = tables.market_price[safe_item, lut].astype(jnp.int32)
    return qty.astype(jnp.float32) * (quote - later).astype(jnp.float32)


def _sort_premium_first(states: State, tables: StaticTables, action: Action, player: int) -> Action:
    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    active = slot < action.market_count[:, None]
    is_premium = jnp.any(action.market_item[..., None] == PREMIUM_IDS[None, None, :], axis=-1)
    premium = active & (action.market_op == MarketOp.SELL) & is_premium
    score = jnp.where(
        premium,
        _sell_impact(states, tables, player, action.market_item, action.market_amount),
        -jnp.inf,
    )
    premium_order = jnp.argsort(-score, axis=1, stable=True)
    rest_order = jnp.argsort(jnp.where(active & ~premium, slot, slot + MAX_MARKET_ORDERS), axis=1, stable=True)
    premium_count = jnp.sum(premium, axis=1).astype(jnp.int32)
    out_slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    premium_index = jnp.take_along_axis(premium_order, jnp.minimum(out_slot, MAX_MARKET_ORDERS - 1), axis=1)
    rest_ordinal = jnp.maximum(out_slot - premium_count[:, None], 0)
    rest_index = jnp.take_along_axis(rest_order, jnp.minimum(rest_ordinal, MAX_MARKET_ORDERS - 1), axis=1)
    index = jnp.where(out_slot < premium_count[:, None], premium_index, rest_index)
    return action._replace(
        market_op=jnp.take_along_axis(action.market_op, index, axis=1),
        market_item=jnp.take_along_axis(action.market_item, index, axis=1),
        market_amount=jnp.take_along_axis(action.market_amount, index, axis=1),
    )


def _front_run(states: State, bank, skeleton_id, action: Action, confidence, player: int) -> Action:
    step = states.step.astype(jnp.int32)
    future = jnp.clip(step + 1, 0, 718)
    future_op = bank.market_op[skeleton_id, future]
    future_item = bank.market_item[skeleton_id, future]
    future_amount = bank.market_amount[skeleton_id, future]
    future_count = bank.market_count[skeleton_id, future]
    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    future_active = slot < future_count[:, None]
    active = slot < action.market_count[:, None]
    best_score = jnp.full(step.shape, -jnp.inf, dtype=jnp.float32)
    best_item = jnp.full(step.shape, -1, dtype=jnp.int8)
    best_qty = jnp.zeros(step.shape, dtype=jnp.int32)
    for ordinal, product_id in enumerate(PREMIUM_IDS.tolist()):
        planned = jnp.sum(
            jnp.where(future_active & (future_op == MarketOp.SELL) & (future_item == product_id), jnp.maximum(future_amount, 0), 0),
            axis=1,
        )
        already = jnp.sum(
            jnp.where(active & (action.market_op == MarketOp.SELL) & (action.market_item == product_id), jnp.maximum(action.market_amount, 0), 0),
            axis=1,
        )
        available = jnp.maximum(states.shed[:, player, product_id].astype(jnp.int32) - already, 0)
        qty = jnp.minimum(available, planned)
        price = states.market_price[:, product_id].astype(jnp.float32)
        score = price * qty.astype(jnp.float32) * _PREMIUM_WEIGHT[ordinal] + _PREMIUM_BASE[ordinal]
        valid = qty > 0
        # PREMIUM_IDS are in Python lexical ascending order. Equal score takes
        # the later item, matching max((priority, item, quantity)).
        take = valid & (score >= best_score)
        best_score = jnp.where(take, score, best_score)
        best_item = jnp.where(take, product_id, best_item).astype(jnp.int8)
        best_qty = jnp.where(take, qty, best_qty)
    enabled = (confidence >= 2) & (step < 718) & (action.market_count < MAX_MARKET_ORDERS) & (best_item >= 0)
    values = _append_market(
        action.market_op,
        action.market_item,
        action.market_amount,
        action.market_count,
        enabled,
        MarketOp.SELL,
        best_item,
        best_qty,
    )
    return action._replace(market_op=values[0], market_item=values[1], market_amount=values[2], market_count=values[3])


def _terminal_liquidation(states: State, action: Action, player: int) -> Action:
    op, item, amount, count = action.market_op, action.market_item, action.market_amount, action.market_count
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < count[:, None]
    terminal = states.step >= 680
    for product_id in SELLABLE_IDS.tolist():
        exists = jnp.any(active & (op == MarketOp.SELL) & (item == product_id), axis=1)
        qty = states.shed[:, player, product_id].astype(jnp.int32)
        append = terminal & (qty > 0) & (~exists) & (count < MAX_MARKET_ORDERS)
        op, item, amount, count = _append_market(op, item, amount, count, append, MarketOp.SELL, product_id, qty)
        active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < count[:, None]
    return action._replace(market_op=op, market_item=item, market_amount=amount, market_count=count)


def _move_toward(states: State, player: int, actor: int, target: jax.Array) -> jax.Array:
    position = states.unit_pos[:, player, actor].astype(jnp.int32)
    x, y = position[:, 0], position[:, 1]
    tx, ty = target[:, 0].astype(jnp.int32), target[:, 1].astype(jnp.int32)
    op = jnp.where(tx < x, UnitOp.WEST, jnp.where(tx > x, UnitOp.EAST, jnp.where(ty < y, UnitOp.NORTH, jnp.where(ty > y, UnitOp.SOUTH, UnitOp.PASS))))
    nx = x + (op == UnitOp.EAST).astype(jnp.int32) - (op == UnitOp.WEST).astype(jnp.int32)
    ny = y + (op == UnitOp.SOUTH).astype(jnp.int32) - (op == UnitOp.NORTH).astype(jnp.int32)
    unlocked = states.tile_kind[jnp.arange(x.shape[0]), player, jnp.clip(ny, 0, 9), jnp.clip(nx, 0, 9)] != TileKind.LOCKED
    return jnp.where(unlocked, op, UnitOp.PASS).astype(jnp.int8)


def _terminal_action(states: State, player: int) -> Action:
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size)
    unit_count = jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int8)
    unit_op = jnp.full((batch_size, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8)
    unit_item = jnp.full((batch_size, MAX_UNITS), -1, dtype=jnp.int8)
    unit_amount = jnp.ones((batch_size, MAX_UNITS), dtype=jnp.int32)
    available = states.tile_yield[:, player].reshape(batch_size, 100) > 0
    pending = jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int32)
    for actor in range(MAX_UNITS):
        present = actor < unit_count
        pos = states.unit_pos[:, player, actor].astype(jnp.int32)
        x, y = pos[:, 0], pos[:, 1]
        inventory = states.unit_inventory[:, player, actor].astype(jnp.int32)
        load = jnp.sum(jnp.maximum(inventory, 0), axis=1)
        at_shed = jnp.any(jnp.all(pos[:, None, :] == _TERMINAL_SHED_ACCESS[None, :, :], axis=2), axis=1)
        current_yield = states.tile_yield[batch, player, jnp.clip(y, 0, 9), jnp.clip(x, 0, 9)] > 0

        distance_shed = jnp.sum(jnp.abs(pos[:, None, :] - _TERMINAL_SHED_ACCESS[None, :, :]), axis=2)
        shed_index = jnp.argmin(distance_shed, axis=1)
        shed_target = _TERMINAL_SHED_ACCESS[shed_index]

        tile_index = jnp.arange(100)[None, :]
        tx = tile_index % 10
        ty = tile_index // 10
        distance = jnp.abs(tx - x[:, None]) + jnp.abs(ty - y[:, None])
        key = jnp.where(available, distance * 100 + tile_index, 1_000_000)
        target_index = jnp.argmin(key, axis=1)
        has_target = jnp.any(available, axis=1)
        target = jnp.stack((target_index % 10, target_index // 10), axis=1).astype(jnp.int16)

        move_shed = _move_toward(states, player, actor, shed_target)
        move_yield = _move_toward(states, player, actor, target)
        tile_kind = states.tile_kind[batch, player, jnp.clip(y, 0, 9), jnp.clip(x, 0, 9)]
        tile_flags = states.tile_flags[batch, player, jnp.clip(y, 0, 9), jnp.clip(x, 0, 9)]
        fertilizer_available = (tile_flags & 8) != 0
        chosen = jnp.where(
            (load > 0) & at_shed,
            UnitOp.DROP,
            jnp.where(
                current_yield,
                UnitOp.HARVEST,
                jnp.where(load > 0, move_shed, jnp.where(has_target, move_yield, jnp.where(fertilizer_available & (tile_kind != TileKind.LOCKED), UnitOp.COLLECT_FERTILIZER, UnitOp.PASS))),
            ),
        ).astype(jnp.int8)
        unit_op = unit_op.at[:, actor].set(jnp.where(present, chosen, UnitOp.PASS))
        drop = present & (chosen == UnitOp.DROP)
        pending = pending + jnp.where(drop[:, None], inventory[:, :NUM_PRODUCTS], 0)
        claim = present & (~((load > 0) & at_shed)) & (~current_yield) & (load == 0) & has_target
        available = available.at[batch, target_index].set(jnp.where(claim, False, available[batch, target_index]))
        current_index = jnp.clip(y, 0, 9) * 10 + jnp.clip(x, 0, 9)
        available = available.at[batch, current_index].set(jnp.where(present & current_yield, False, available[batch, current_index]))

    shed = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int32) + pending
    values = shed * states.market_price.astype(jnp.int32)
    lexical = _LEXICAL_RANK[None, :].astype(jnp.int32)
    valid_sellable = jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.bool_).at[:, SELLABLE_IDS].set(True)
    key = jnp.where(valid_sellable & (shed > 0), values.astype(jnp.int64) * 32 + lexical, -1)
    ranked = jnp.argsort(-key, axis=1, stable=True)
    market_op = jnp.full((batch_size, MAX_MARKET_ORDERS), MarketOp.NONE, dtype=jnp.int8)
    market_item = jnp.full((batch_size, MAX_MARKET_ORDERS), -1, dtype=jnp.int8)
    market_amount = jnp.zeros((batch_size, MAX_MARKET_ORDERS), dtype=jnp.int32)
    market_count = jnp.zeros((batch_size,), dtype=jnp.int8)
    for ordinal in range(NUM_PRODUCTS):
        product_id = ranked[:, ordinal]
        qty = jnp.take_along_axis(shed, product_id[:, None], axis=1)[:, 0]
        eligible = jnp.take_along_axis(valid_sellable, product_id[:, None], axis=1)[:, 0] & (qty > 0) & (market_count < 10)
        market_op, market_item, market_amount, market_count = _append_market(
            market_op, market_item, market_amount, market_count, eligible, MarketOp.SELL, product_id.astype(jnp.int8), qty
        )
    return Action(unit_op, unit_item, unit_amount, unit_count, market_op, market_item, market_amount, market_count)


def _base_controller(states: State, tables: StaticTables, bank, skeleton_id, confidence, player: int) -> Action:
    action = _raw_action(states, bank, skeleton_id)
    action = _front_run(states, bank, skeleton_id, action, confidence, player)
    action = _terminal_liquidation(states, action, player)
    action = _sort_premium_first(states, tables, action, player)
    terminal = states.step >= 717
    return Action(*(jnp.where(terminal.reshape((terminal.shape[0],) + (1,) * (left.ndim - 1)), left, right) for left, right in zip(_terminal_action(states, player), action, strict=True)))


def _sbt_bank(states: State, tables: StaticTables, action: Action, player: int) -> Action:
    step = states.step.astype(jnp.int32)
    enabled_step = (step >= 120) & (step < 680)
    unit_op = action.unit_op
    premium_pending = jnp.zeros((step.shape[0], NUM_PRODUCTS), dtype=jnp.int32)
    actual_count = jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int8)
    for actor in range(MAX_UNITS):
        present = actor < actual_count
        pos = states.unit_pos[:, player, actor].astype(jnp.int32)
        inventory = states.unit_inventory[:, player, actor].astype(jnp.int32)
        op = unit_op[:, actor]
        premium_value = jnp.sum(inventory[:, PREMIUM_IDS] * states.market_price[:, PREMIUM_IDS], axis=1)
        at_shed = jnp.any(jnp.all(pos[:, None, :] == _SHED_ACCESS[None, :, :], axis=2), axis=1)
        ordinary_drop = present & enabled_step & (op == UnitOp.DROP) & at_shed
        safe = jnp.any(op[:, None] == _SAFE_BANK_OPS[None, :], axis=1)
        distance = jnp.sum(jnp.abs(pos[:, None, :] - _SHED_ACCESS[None, :, :]), axis=2)
        target_index = jnp.argmin(distance * 100 + jnp.asarray((0, 1, 2, 3))[None, :], axis=1)
        target = _SHED_ACCESS[target_index]
        best_distance = jnp.take_along_axis(distance, target_index[:, None], axis=1)[:, 0]
        divert = present & enabled_step & (~ordinary_drop) & (premium_value >= 1500) & safe
        bank_drop = divert & (best_distance == 0)
        bank_move = divert & (best_distance > 0) & (best_distance <= 1)
        move = _move_toward(states, player, actor, target)
        chosen = jnp.where(bank_drop, UnitOp.DROP, jnp.where(bank_move, move, op)).astype(jnp.int8)
        unit_op = unit_op.at[:, actor].set(jnp.where(present, chosen, op))
        credit = ordinary_drop | bank_drop
        premium_pending = premium_pending + jnp.where(credit[:, None], inventory[:, :NUM_PRODUCTS], 0)

    action = action._replace(unit_op=unit_op, unit_count=actual_count)
    op, item, amount, count = action.market_op, action.market_item, action.market_amount, action.market_count
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < count[:, None]
    for product_id in PREMIUM_IDS.tolist():
        qty = premium_pending[:, product_id]
        price_ok = states.market_price[:, product_id].astype(jnp.float32) / float((250, 120, 160, 200)[PREMIUM_IDS.tolist().index(product_id)]) >= 0.25
        valid = enabled_step & (qty > 0) & price_ok
        locations = jnp.where(active & (op == MarketOp.SELL) & (item == product_id), jnp.arange(MAX_MARKET_ORDERS)[None, :], -1)
        location = jnp.max(locations, axis=1)
        merge = valid & (location >= 0)
        batch = jnp.arange(step.shape[0])
        safe_location = jnp.maximum(location, 0)
        amount = amount.at[batch, safe_location].set(jnp.where(merge, amount[batch, safe_location] + qty, amount[batch, safe_location]))
        append = valid & (location < 0) & (count < MAX_MARKET_ORDERS)
        op, item, amount, count = _append_market(op, item, amount, count, append, MarketOp.SELL, product_id, qty)
        active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < count[:, None]
    return _sort_premium_first(states, tables, action._replace(market_op=op, market_item=item, market_amount=amount, market_count=count), player)


def _guarded_weed(states: State, bank, skeleton_id, action: Action, player: int) -> Action:
    step = states.step.astype(jnp.int32)
    actual_count = jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int8)
    unit_count = jnp.maximum(action.unit_count, actual_count).astype(jnp.int8)
    unit_op, unit_item, unit_amount = action.unit_op, action.unit_item, action.unit_amount
    weed = (states.tile_kind[:, player] == TileKind.WEED) & (_WEED_LAST_PLANNED_USE[None, :, :] > step[:, None, None])
    claimed = jnp.zeros((step.shape[0], 100), dtype=jnp.bool_)
    day_end = ((step // 24) + 1) * 24
    turns_left = day_end - step
    future_offsets = jnp.arange(24)[None, :]
    future_steps = jnp.clip(step[:, None] + future_offsets, 0, 718)
    in_day = step[:, None] + future_offsets < day_end[:, None]
    for actor in range(MAX_UNITS):
        tail_op = bank.unit_op[skeleton_id[:, None], future_steps, actor]
        tail_free = jnp.all((~in_day) | (tail_op == UnitOp.PASS), axis=1)
        inventory_empty = jnp.sum(jnp.maximum(states.unit_inventory[:, player, actor].astype(jnp.int32), 0), axis=1) == 0
        present = actor < actual_count
        eligible_actor = present & (unit_op[:, actor] == UnitOp.PASS) & tail_free & inventory_empty & (step < 717)
        pos = states.unit_pos[:, player, actor].astype(jnp.int32)
        index = jnp.arange(100)[None, :]
        x, y = index % 10, index // 10
        distance = jnp.abs(x - pos[:, 0:1]) + jnp.abs(y - pos[:, 1:2])
        choices = weed.reshape(step.shape[0], 100) & (~claimed)
        key = jnp.where(choices, distance * 100 + index, 1_000_000)
        target_index = jnp.argmin(key, axis=1)
        target_distance = jnp.take_along_axis(distance, target_index[:, None], axis=1)[:, 0]
        has_target = jnp.any(choices, axis=1)
        take = eligible_actor & has_target & (target_distance + 1 <= turns_left)
        tx, ty = target_index % 10, target_index // 10
        chosen = jnp.where(tx < pos[:, 0], UnitOp.WEST, jnp.where(tx > pos[:, 0], UnitOp.EAST, jnp.where(ty < pos[:, 1], UnitOp.NORTH, jnp.where(ty > pos[:, 1], UnitOp.SOUTH, UnitOp.DIG)))).astype(jnp.int8)
        unit_op = unit_op.at[:, actor].set(jnp.where(take, chosen, unit_op[:, actor]))
        unit_item = unit_item.at[:, actor].set(jnp.where(take, -1, unit_item[:, actor]))
        unit_amount = unit_amount.at[:, actor].set(jnp.where(take, 1, unit_amount[:, actor]))
        claimed = claimed.at[jnp.arange(step.shape[0]), target_index].set(jnp.where(take, True, claimed[jnp.arange(step.shape[0]), target_index]))
    return action._replace(unit_op=unit_op, unit_item=unit_item, unit_amount=unit_amount, unit_count=unit_count)


def _repair_productive(states: State, action: Action, carry: C95PlayerCarryV1, player: int):
    step = states.step.astype(jnp.int32)
    reset = step % 24 == 0
    qop = jnp.where(reset[:, None, None], UnitOp.PASS, carry.repair_op)
    qitem = jnp.where(reset[:, None, None], -1, carry.repair_item)
    qamount = jnp.where(reset[:, None, None], 1, carry.repair_amount)
    qlen = jnp.where(reset[:, None], 0, carry.repair_length).astype(jnp.int8)
    unit_op, unit_item, unit_amount = action.unit_op, action.unit_item, action.unit_amount
    actual_count = jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int8)
    unit_count = jnp.maximum(action.unit_count, actual_count).astype(jnp.int8)
    batch = jnp.arange(step.shape[0])
    for actor in range(MAX_UNITS):
        scheduled_op = unit_op[:, actor]
        scheduled_item = unit_item[:, actor]
        scheduled_amount = unit_amount[:, actor]
        pending = qlen[:, actor] > 0
        first_op, first_item, first_amount = qop[:, actor, 0], qitem[:, actor, 0], qamount[:, actor, 0]
        shifted_op = jnp.concatenate((qop[:, actor, 1:], jnp.full_like(qop[:, actor, :1], UnitOp.PASS)), axis=1)
        shifted_item = jnp.concatenate((qitem[:, actor, 1:], jnp.full_like(qitem[:, actor, :1], -1)), axis=1)
        shifted_amount = jnp.concatenate((qamount[:, actor, 1:], jnp.ones_like(qamount[:, actor, :1])), axis=1)
        new_len = jnp.maximum(qlen[:, actor].astype(jnp.int32) - pending.astype(jnp.int32), 0)
        append_scheduled = pending & (scheduled_op != UnitOp.PASS) & (new_len < 24)
        append_slot = jnp.clip(new_len, 0, 23)
        shifted_op = shifted_op.at[batch, append_slot].set(jnp.where(append_scheduled, scheduled_op, shifted_op[batch, append_slot]))
        shifted_item = shifted_item.at[batch, append_slot].set(jnp.where(append_scheduled, scheduled_item, shifted_item[batch, append_slot]))
        shifted_amount = shifted_amount.at[batch, append_slot].set(jnp.where(append_scheduled, scheduled_amount, shifted_amount[batch, append_slot]))
        new_len = new_len + append_scheduled.astype(jnp.int32)

        pos = states.unit_pos[:, player, actor].astype(jnp.int32)
        tile = states.tile_kind[batch, player, jnp.clip(pos[:, 1], 0, 9), jnp.clip(pos[:, 0], 0, 9)]
        blocked = jnp.any(scheduled_op[:, None] == _BLOCKED_OPS[None, :], axis=1)
        trigger = (~pending) & (actor < actual_count) & blocked & (tile == TileKind.WEED) & (step < 717)
        shifted_op = shifted_op.at[batch, 0].set(jnp.where(trigger, scheduled_op, shifted_op[:, 0]))
        shifted_item = shifted_item.at[batch, 0].set(jnp.where(trigger, scheduled_item, shifted_item[:, 0]))
        shifted_amount = shifted_amount.at[batch, 0].set(jnp.where(trigger, scheduled_amount, shifted_amount[:, 0]))
        new_len = jnp.where(trigger, 1, new_len)
        output_op = jnp.where(pending, first_op, jnp.where(trigger, UnitOp.DIG, scheduled_op))
        output_item = jnp.where(pending, first_item, jnp.where(trigger, -1, scheduled_item))
        output_amount = jnp.where(pending, first_amount, jnp.where(trigger, 1, scheduled_amount))
        unit_op = unit_op.at[:, actor].set(output_op.astype(jnp.int8))
        unit_item = unit_item.at[:, actor].set(output_item.astype(jnp.int8))
        unit_amount = unit_amount.at[:, actor].set(output_amount.astype(jnp.int32))
        qop = qop.at[:, actor].set(shifted_op)
        qitem = qitem.at[:, actor].set(shifted_item)
        qamount = qamount.at[:, actor].set(shifted_amount)
        qlen = qlen.at[:, actor].set(new_len.astype(jnp.int8))
    following = carry._replace(repair_op=qop, repair_item=qitem, repair_amount=qamount, repair_length=qlen)
    return action._replace(unit_op=unit_op, unit_item=unit_item, unit_amount=unit_amount, unit_count=unit_count), following


def _c94(states: State, bank, skeleton_id, action: Action, carry: C95PlayerCarryV1, player: int):
    step = states.step.astype(jnp.int32)
    debt = carry.c94_debt.astype(jnp.int32)
    op, item, amount, count = action.market_op, action.market_item, action.market_amount, action.market_count
    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    active = slot < count[:, None]
    remove = jnp.zeros_like(active)
    leftover = debt
    for product_id in range(NUM_PRODUCTS):
        requested = jnp.where(active & (op == MarketOp.SELL) & (item == product_id), jnp.maximum(amount, 0), 0)
        prefix = jnp.cumsum(requested, axis=1) - requested
        reduction = jnp.minimum(requested, jnp.maximum(debt[:, product_id:product_id + 1] - prefix, 0))
        changed = active & (op == MarketOp.SELL) & (item == product_id) & (reduction > 0)
        amount = jnp.where(changed, amount - reduction, amount)
        remove = remove | (changed & (amount <= 0))
        leftover = leftover.at[:, product_id].set(jnp.maximum(debt[:, product_id] - jnp.sum(reduction, axis=1), 0))
    op, item, amount, count = _compact_market(op, item, amount, count, ~remove)

    # C94 feed5 opening: remove every original WHEAT buy, then prepend exactly 5.
    opening = step == 0
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < count[:, None]
    keep = ~(opening[:, None] & active & (op == MarketOp.BUY_PRODUCT) & (item == WHEAT_ID))
    op, item, amount, count = _compact_market(op, item, amount, count, keep)
    op, item, amount, count = _prepend_market(op, item, amount, count, opening, MarketOp.BUY_PRODUCT, WHEAT_ID, jnp.full(step.shape, 5, dtype=jnp.int32))

    future_step = jnp.clip(step + 1, 0, 718)
    future_op = bank.market_op[skeleton_id, future_step]
    future_item = bank.market_item[skeleton_id, future_step]
    future_amount = bank.market_amount[skeleton_id, future_step]
    future_count = bank.market_count[skeleton_id, future_step]
    future_active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < future_count[:, None]
    split_enabled = (step >= 120) & (step < 715)
    new_debt = leftover
    for product_id, cap in ((WHEAT_ID, 10), (FERTILIZER_ID, 5)):
        planned = jnp.sum(jnp.where(future_active & (future_op == MarketOp.SELL) & (future_item == product_id), jnp.maximum(future_amount, 0), 0), axis=1)
        active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < count[:, None]
        committed = jnp.sum(jnp.where(active & (op == MarketOp.SELL) & (item == product_id), jnp.maximum(amount, 0), 0), axis=1)
        available = jnp.maximum(states.shed[:, player, product_id].astype(jnp.int32) - committed, 0)
        qty = jnp.minimum(jnp.minimum(planned, cap), available)
        append = split_enabled & (planned > 0) & (qty > 0) & (count < MAX_MARKET_ORDERS)
        op, item, amount, count = _append_market(op, item, amount, count, append, MarketOp.SELL, product_id, qty)
        new_debt = new_debt.at[:, product_id].add(jnp.where(append, qty, 0))
    following = carry._replace(c94_debt=new_debt.astype(jnp.int16))
    return action._replace(market_op=op, market_item=item, market_amount=amount, market_count=count), following


def c95_player_action_v1(
    states: State,
    tables: StaticTables,
    bank,
    skeleton_id: jax.Array,
    carry: C95PlayerCarryV1,
    player: int,
) -> tuple[Action, C95PlayerCarryV1]:
    """Emit C95 actions for one seat; caller masks carry for non-C95 rows."""

    enabled = jnp.ones_like(states.step, dtype=jnp.bool_)
    outer_conf = _update_confidence(states, carry.outer_confidence, enabled)
    outer = _base_controller(states, tables, bank, skeleton_id, outer_conf, player)
    use_sbt = (states.step >= 408) & ((states.money[:, player] - states.money[:, 1 - player]) <= 250)
    sbt_conf = _update_confidence(states, carry.sbt_confidence, use_sbt)
    sbt = _sbt_bank(states, tables, _base_controller(states, tables, bank, skeleton_id, sbt_conf, player), player)
    action = Action(*(jnp.where(use_sbt.reshape((use_sbt.shape[0],) + (1,) * (left.ndim - 1)), left, right) for left, right in zip(sbt, outer, strict=True)))
    action = _guarded_weed(states, bank, skeleton_id, action, player)
    carry = carry._replace(outer_confidence=outer_conf, sbt_confidence=sbt_conf)
    action, carry = _repair_productive(states, action, carry, player)
    return _c94(states, bank, skeleton_id, action, carry, player)
