"""Exact GPU-native Boatlee V16-RC2 frozen-opponent policy.

The original submission is a 719-step trace plus five small state machines.
This module keeps those state machines explicit and emits the simulator's raw
``Action`` tensors without calling Python or the CPU inside a rollout.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import (
    ANIMALS,
    MARKET_LUT_SIZE,
    MARKET_MIN_INVENTORY,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_CROPS,
    NUM_PRODUCTS,
    PRODUCTS,
    SHED_ITEMS,
    SHOP_NAMES,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.types import Action, State, StaticTables


TRACE_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts/boatlee_v16/boatlee_v16_rc2_trace_v1.npz"
)
WOOL_ID = PRODUCTS.index("WOOL")
FERTILIZER_ID = PRODUCTS.index("FERTILIZER")
COW_ID = SHED_ITEMS.index("COW")
SHEEP_ID = SHED_ITEMS.index("SHEEP")
YARN_STORE_ID = SHOP_NAMES.index("YARN_STORE")
SMOOTHIE_SHOP_ID = SHOP_NAMES.index("SMOOTHIE_SHOP")
PIZZA_SHOP_ID = SHOP_NAMES.index("PIZZA_SHOP")


class BoatleeTraceV1(NamedTuple):
    unit_op: jax.Array
    unit_item: jax.Array
    unit_amount: jax.Array
    unit_count: jax.Array
    market_op: jax.Array
    market_item: jax.Array
    market_amount: jax.Array
    market_count: jax.Array
    future_fertilizer: jax.Array


class BoatleePlayerCarryV1(NamedTuple):
    yarn_known: jax.Array
    yarn_route: jax.Array
    wool_last_sale: jax.Array
    route_checks: jax.Array
    route_locked: jax.Array
    relay_due_step: jax.Array
    relay_due: jax.Array
    weed_active: jax.Array
    weed_start: jax.Array
    weed_intended_op: jax.Array
    weed_intended_item: jax.Array
    weed_intended_amount: jax.Array


class BoatleeCarryV1(NamedTuple):
    yarn_known: jax.Array
    yarn_route: jax.Array
    wool_last_sale: jax.Array
    route_checks: jax.Array
    route_locked: jax.Array
    relay_due_step: jax.Array
    relay_due: jax.Array
    weed_active: jax.Array
    weed_start: jax.Array
    weed_intended_op: jax.Array
    weed_intended_item: jax.Array
    weed_intended_amount: jax.Array


class BoatleePlayerActionV1(NamedTuple):
    unit_op: jax.Array
    unit_item: jax.Array
    unit_amount: jax.Array
    unit_count: jax.Array
    market_op: jax.Array
    market_item: jax.Array
    market_amount: jax.Array
    market_count: jax.Array


def load_boatlee_trace_v1(path: Path = TRACE_PATH) -> BoatleeTraceV1:
    with np.load(path, allow_pickle=False) as data:
        return BoatleeTraceV1(
            *(jnp.asarray(data[name]) for name in BoatleeTraceV1._fields)
        )


def initialize_boatlee_carry_v1(batch_size: int) -> BoatleeCarryV1:
    pair = (batch_size, 2)
    units = (batch_size, 2, MAX_UNITS)
    return BoatleeCarryV1(
        yarn_known=jnp.zeros(pair, dtype=jnp.bool_),
        yarn_route=jnp.zeros(pair, dtype=jnp.bool_),
        wool_last_sale=jnp.full(pair, -1000, dtype=jnp.int16),
        route_checks=jnp.zeros((batch_size, 2, 3), dtype=jnp.bool_),
        route_locked=jnp.zeros(pair, dtype=jnp.bool_),
        relay_due_step=jnp.full(pair, -1, dtype=jnp.int16),
        relay_due=jnp.zeros(pair, dtype=jnp.int16),
        weed_active=jnp.zeros(units, dtype=jnp.bool_),
        weed_start=jnp.full(units, -1, dtype=jnp.int16),
        weed_intended_op=jnp.full(units, UnitOp.PASS, dtype=jnp.int8),
        weed_intended_item=jnp.full(units, -1, dtype=jnp.int8),
        weed_intended_amount=jnp.ones(units, dtype=jnp.int32),
    )


def initialize_boatlee_player_carry_v1(batch_size: int) -> BoatleePlayerCarryV1:
    """Initialize one seat of the exact Boatlee controller.

    The original acceptance harness runs Boatlee in both seats and therefore
    uses :func:`initialize_boatlee_carry_v1`.  Mixed-opponent GPU arenas need
    only one seat, so exposing the same state without an artificial player
    axis avoids duplicating or approximating the controller.
    """

    units = (batch_size, MAX_UNITS)
    return BoatleePlayerCarryV1(
        yarn_known=jnp.zeros((batch_size,), dtype=jnp.bool_),
        yarn_route=jnp.zeros((batch_size,), dtype=jnp.bool_),
        wool_last_sale=jnp.full((batch_size,), -1000, dtype=jnp.int16),
        route_checks=jnp.zeros((batch_size, 3), dtype=jnp.bool_),
        route_locked=jnp.zeros((batch_size,), dtype=jnp.bool_),
        relay_due_step=jnp.full((batch_size,), -1, dtype=jnp.int16),
        relay_due=jnp.zeros((batch_size,), dtype=jnp.int16),
        weed_active=jnp.zeros(units, dtype=jnp.bool_),
        weed_start=jnp.full(units, -1, dtype=jnp.int16),
        weed_intended_op=jnp.full(units, UnitOp.PASS, dtype=jnp.int8),
        weed_intended_item=jnp.full(units, -1, dtype=jnp.int8),
        weed_intended_amount=jnp.ones(units, dtype=jnp.int32),
    )


def _player_carry(carry: BoatleeCarryV1, player: int) -> BoatleePlayerCarryV1:
    return BoatleePlayerCarryV1(*(value[:, player] for value in carry))


def _stack_player_carry(
    left: BoatleePlayerCarryV1, right: BoatleePlayerCarryV1
) -> BoatleeCarryV1:
    return BoatleeCarryV1(
        *(jnp.stack(values, axis=1) for values in zip(left, right, strict=True))
    )


def _tile_under_units(states: State, player: int) -> jax.Array:
    batch = jnp.arange(states.step.shape[0])[:, None]
    position = states.unit_pos[:, player].astype(jnp.int32)
    return states.tile_kind[
        batch,
        player,
        jnp.clip(position[..., 1], 0, 9),
        jnp.clip(position[..., 0], 0, 9),
    ]


def _weed_repair(
    states: State,
    player: int,
    trace: BoatleeTraceV1,
    carry: BoatleePlayerCarryV1,
    unit_op: jax.Array,
    unit_item: jax.Array,
    unit_amount: jax.Array,
    unit_count: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, BoatleePlayerCarryV1]:
    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    unit_index = jnp.arange(MAX_UNITS)[None, :]
    present = unit_index < unit_count[:, None]
    age = step[:, None] - carry.weed_start.astype(jnp.int32)
    existing = carry.weed_active & present & (age <= 9)
    replay_step = jnp.clip(step - 1, 0, 718)
    replay_op = trace.unit_op[replay_step]
    replay_item = trace.unit_item[replay_step]
    replay_amount = trace.unit_amount[replay_step]

    age_one = existing & (age == 1)
    replay = existing & (age >= 2) & (age <= 9)
    unit_op = jnp.where(age_one, carry.weed_intended_op, unit_op)
    unit_item = jnp.where(age_one, carry.weed_intended_item, unit_item)
    unit_amount = jnp.where(age_one, carry.weed_intended_amount, unit_amount)
    unit_op = jnp.where(replay, replay_op, unit_op)
    unit_item = jnp.where(replay, replay_item, unit_item)
    unit_amount = jnp.where(replay, replay_amount, unit_amount)

    trigger = (
        (~existing)
        & present
        & ((unit_op == UnitOp.BUILD_PASTURE) | (unit_op == UnitOp.PLANT))
        & (_tile_under_units(states, player) == TileKind.WEED)
    )
    intended_op, intended_item, intended_amount = unit_op, unit_item, unit_amount
    unit_op = jnp.where(trigger, UnitOp.DIG, unit_op).astype(jnp.int8)
    unit_item = jnp.where(trigger, -1, unit_item).astype(jnp.int8)
    unit_amount = jnp.where(trigger, 1, unit_amount).astype(jnp.int32)
    following = carry._replace(
        weed_active=existing | trigger,
        weed_start=jnp.where(trigger, step[:, None], carry.weed_start).astype(
            jnp.int16
        ),
        weed_intended_op=jnp.where(
            trigger, intended_op, carry.weed_intended_op
        ).astype(jnp.int8),
        weed_intended_item=jnp.where(
            trigger, intended_item, carry.weed_intended_item
        ).astype(jnp.int8),
        weed_intended_amount=jnp.where(
            trigger, intended_amount, carry.weed_intended_amount
        ).astype(jnp.int32),
    )
    return unit_op, unit_item, unit_amount, following


def _town_has(states: State, shop_id: int) -> jax.Array:
    active = jnp.arange(states.town_shops.shape[1])[None, :] < states.town_count[:, None]
    return jnp.any(active & (states.town_shops == shop_id), axis=1)


def _update_yarn_route(
    states: State, carry: BoatleePlayerCarryV1
) -> BoatleePlayerCarryV1:
    decide = (states.step >= 161) & (~carry.yarn_known)
    route = (
        _town_has(states, YARN_STORE_ID)
        & (~_town_has(states, SMOOTHIE_SHOP_ID))
        & (~_town_has(states, PIZZA_SHOP_ID))
    )
    return carry._replace(
        yarn_known=carry.yarn_known | decide,
        yarn_route=jnp.where(decide, route, carry.yarn_route),
    )


def _append_market(
    op: jax.Array,
    item: jax.Array,
    amount: jax.Array,
    count: jax.Array,
    append: jax.Array,
    new_op: int,
    new_item: int,
    new_amount: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    batch = jnp.arange(op.shape[0])
    slot = jnp.clip(count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1)
    old_op = op[batch, slot]
    old_item = item[batch, slot]
    old_amount = amount[batch, slot]
    op = op.at[batch, slot].set(jnp.where(append, new_op, old_op))
    item = item.at[batch, slot].set(jnp.where(append, new_item, old_item))
    amount = amount.at[batch, slot].set(jnp.where(append, new_amount, old_amount))
    return op, item, amount, (count + append.astype(count.dtype)).astype(jnp.int8)


_SHOP_DEMAND = jnp.asarray(
    (
        (1, 0, 0, 0, 0, 1, 0, 0, 0),
        (1, 0, 0, 1, 0, 1, 0, 0, 0),
        (1, 1, 1, 1, 0, 0, 0, 0, 0),
        (1, 0, 0, 1, 0, 0, 1, 0, 0),
        (0, 2, 0, 0, 0, 0, 0, 0, 0),
        (1, 0, 1, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 2, 0),
    ),
    dtype=jnp.float32,
)


def _rank_sell_slots(
    states: State,
    tables: StaticTables,
    op: jax.Array,
    item: jax.Array,
    amount: jax.Array,
    count: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    active = slot < count[:, None]
    sell = active & (op == MarketOp.SELL) & (item >= 0) & (item < NUM_PRODUCTS)
    safe_item = jnp.clip(item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    quantity = jnp.maximum(amount, 0).astype(jnp.int32)
    current_inventory = jnp.take_along_axis(
        states.market_inventory, safe_item, axis=1
    )
    current_quote = jnp.take_along_axis(states.market_price, safe_item, axis=1)
    later_lut = jnp.clip(
        current_inventory + quantity - MARKET_MIN_INVENTORY,
        0,
        MARKET_LUT_SIZE - 1,
    )
    later_quote = tables.market_price[safe_item, later_lut].astype(jnp.int32)
    impact = quantity.astype(jnp.float32) * jnp.maximum(
        current_quote - later_quote, 0
    ).astype(jnp.float32)

    town_active = (
        jnp.arange(states.town_shops.shape[1])[None, :]
        < states.town_count[:, None]
    )
    safe_shops = jnp.clip(states.town_shops.astype(jnp.int32), 0, len(SHOP_NAMES) - 1)
    demand_by_item = jnp.sum(
        _SHOP_DEMAND[safe_shops] * town_active[..., None], axis=1
    ) * 6.0
    demand_by_item = demand_by_item + jnp.asarray(
        [1.0] * (NUM_PRODUCTS - 1) + [0.0], dtype=jnp.float32
    )[None, :]
    demand = jnp.maximum(
        jnp.take_along_axis(demand_by_item, safe_item, axis=1), 0.25
    )
    excess = jnp.maximum(current_inventory + quantity - 10000, 0).astype(jnp.float32)
    urgency = jnp.minimum(1.0, (excess / demand) / 10.0)
    score = impact * (1.0 + 0.25 * urgency)
    score = jnp.where(sell, score, -jnp.inf)
    ranked_index = jnp.argsort(-score, axis=1, stable=True)
    ranked_op = jnp.take_along_axis(op, ranked_index, axis=1)
    ranked_item = jnp.take_along_axis(item, ranked_index, axis=1)
    ranked_amount = jnp.take_along_axis(amount, ranked_index, axis=1)
    sell_ordinal = jnp.maximum(jnp.cumsum(sell, axis=1) - 1, 0)
    return (
        jnp.where(sell, jnp.take_along_axis(ranked_op, sell_ordinal, axis=1), op),
        jnp.where(
            sell, jnp.take_along_axis(ranked_item, sell_ordinal, axis=1), item
        ),
        jnp.where(
            sell, jnp.take_along_axis(ranked_amount, sell_ordinal, axis=1), amount
        ),
    )


def _wool_controller(
    states: State,
    player: int,
    carry: BoatleePlayerCarryV1,
    op: jax.Array,
    item: jax.Array,
    amount: jax.Array,
    count: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, BoatleePlayerCarryV1]:
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < count[:, None]
    existing = jnp.any(
        active & (op == MarketOp.SELL) & (item == WOOL_ID), axis=1
    )
    wool = jnp.maximum(states.shed[:, player, WOOL_ID].astype(jnp.int32), 0)
    total = jnp.sum(jnp.maximum(states.shed[:, player].astype(jnp.int32), 0), axis=1)
    price = jnp.maximum(states.market_price[:, WOOL_ID].astype(jnp.int32), 0)
    step = states.step.astype(jnp.int32)
    gate = jnp.where(step < 480, 170, jnp.where(step < 600, 120, jnp.where(step < 672, 80, 1)))
    terminal = step >= 713
    pressure = total >= 78
    price_ok = price >= gate
    gap_ok = step - carry.wool_last_sale.astype(jnp.int32) >= 6
    append = (
        carry.yarn_route
        & (~existing)
        & (wool > 0)
        & (count < MAX_MARKET_ORDERS)
        & (terminal | pressure | (price_ok & gap_ok))
    )
    quantity = jnp.where(terminal, wool, jnp.minimum(wool, 16))
    quantity = jnp.where(
        pressure, jnp.minimum(wool, jnp.maximum(quantity, total - 66)), quantity
    )
    quantity = jnp.maximum(quantity, 1)
    op, item, amount, count = _append_market(
        op, item, amount, count, append, MarketOp.SELL, WOOL_ID, quantity
    )
    carry = carry._replace(
        wool_last_sale=jnp.where(append, step, carry.wool_last_sale).astype(jnp.int16)
    )
    return op, item, amount, count, carry


def _route_distance(states: State) -> jax.Array:
    crop = states.tile_crop.astype(jnp.int32)
    animal = states.tile_animal.astype(jnp.int32)
    crop_counts = jax.nn.one_hot(
        jnp.clip(crop, 0, NUM_CROPS - 1), NUM_CROPS, dtype=jnp.int32
    ) * (crop >= 0)[..., None]
    animal_counts = jax.nn.one_hot(
        jnp.clip(animal, 0, len(ANIMALS) - 1), len(ANIMALS), dtype=jnp.int32
    ) * (animal >= 0)[..., None]
    crop_counts = crop_counts.sum(axis=(2, 3))
    animal_counts = animal_counts.sum(axis=(2, 3))
    no_animal = animal < 0
    kinds = states.tile_kind.astype(jnp.int32)
    structure_counts = jnp.stack(
        (
            jnp.sum(no_animal & (kinds == TileKind.COOP), axis=(2, 3)),
            jnp.sum(no_animal & (kinds == TileKind.PASTURE), axis=(2, 3)),
            jnp.sum(kinds == TileKind.WEED, axis=(2, 3)),
        ),
        axis=-1,
    )
    signature = jnp.concatenate((crop_counts, animal_counts, structure_counts), axis=-1)
    count_distance = jnp.sum(jnp.abs(signature[:, 0] - signature[:, 1]), axis=1)
    hands = jnp.sum(states.unit_active, axis=2).astype(jnp.int32) - 1
    return (
        count_distance
        + jnp.abs(hands[:, 0] - hands[:, 1])
        + 3
        * jnp.abs(
            states.unlocked_count[:, 0].astype(jnp.int32)
            - states.unlocked_count[:, 1].astype(jnp.int32)
        )
    )


def _compact_market(
    op: jax.Array,
    item: jax.Array,
    amount: jax.Array,
    count: jax.Array,
    keep: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    keep = keep & (slot < count[:, None])
    key = jnp.where(keep, slot, slot + MAX_MARKET_ORDERS)
    order = jnp.argsort(key, axis=1, stable=True)
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


def _relay_controller(
    states: State,
    player: int,
    trace: BoatleeTraceV1,
    carry: BoatleePlayerCarryV1,
    op: jax.Array,
    item: jax.Array,
    amount: jax.Array,
    count: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, BoatleePlayerCarryV1]:
    step = states.step.astype(jnp.int32)
    distance_ok = _route_distance(states) <= 8
    checkpoints = jnp.asarray((216, 240, 264), dtype=jnp.int32)
    at_checkpoint = step[:, None] == checkpoints[None, :]
    checks = jnp.where(at_checkpoint, distance_ok[:, None], carry.route_checks)
    locked = jnp.where(step == 264, jnp.all(checks, axis=1), carry.route_locked)

    due_now = carry.relay_due_step.astype(jnp.int32) == step
    due_late = (carry.relay_due_step >= 0) & (carry.relay_due_step < step)
    due_quantity = jnp.where(due_now, carry.relay_due.astype(jnp.int32), 0)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < count[:, None]
    sell_fertilizer = (
        active & (op == MarketOp.SELL) & (item == FERTILIZER_ID)
    )
    requested = jnp.where(sell_fertilizer, jnp.maximum(amount, 0), 0)
    prefix = jnp.cumsum(requested, axis=1) - requested
    reduction = jnp.minimum(
        requested, jnp.maximum(due_quantity[:, None] - prefix, 0)
    )
    repaid_amount = amount - reduction
    remove = due_now[:, None] & sell_fertilizer & (repaid_amount <= 0)
    amount = jnp.where(due_now[:, None] & sell_fertilizer, repaid_amount, amount)
    op, item, amount, count = _compact_market(
        op, item, amount, count, ~remove
    )
    due = jnp.where(due_now | due_late, 0, carry.relay_due).astype(jnp.int16)
    due_step = jnp.where(
        due_now | due_late, -1, carry.relay_due_step
    ).astype(jnp.int16)

    safe_step = jnp.clip(step, 0, 718)
    target = trace.future_fertilizer[safe_step].astype(jnp.int32)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < count[:, None]
    reserved = jnp.sum(
        jnp.where(
            active & (op == MarketOp.SELL) & (item == FERTILIZER_ID),
            jnp.maximum(amount, 0),
            0,
        ),
        axis=1,
    )
    available = jnp.maximum(
        states.shed[:, player, FERTILIZER_ID].astype(jnp.int32) - reserved, 0
    )
    quantity = jnp.minimum(target, available)
    append = locked & (due == 0) & (target > 0) & (count < 10) & (quantity > 0)
    op, item, amount, count = _append_market(
        op,
        item,
        amount,
        count,
        append,
        MarketOp.SELL,
        FERTILIZER_ID,
        quantity,
    )
    due_step = jnp.where(append, step + 3, due_step).astype(jnp.int16)
    due = jnp.where(append, quantity, due).astype(jnp.int16)
    carry = carry._replace(
        route_checks=checks,
        route_locked=locked,
        relay_due_step=due_step,
        relay_due=due,
    )
    return op, item, amount, count, carry


def _player_action(
    states: State,
    tables: StaticTables,
    trace: BoatleeTraceV1,
    carry: BoatleePlayerCarryV1,
    player: int,
) -> tuple[BoatleePlayerActionV1, BoatleePlayerCarryV1]:
    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    unit_op = trace.unit_op[step]
    unit_item = trace.unit_item[step]
    unit_amount = trace.unit_amount[step]
    unit_count = jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int8)
    present = jnp.arange(MAX_UNITS)[None, :] < unit_count[:, None]
    unit_op = jnp.where(present, unit_op, UnitOp.PASS).astype(jnp.int8)
    unit_item = jnp.where(present, unit_item, -1).astype(jnp.int8)
    unit_amount = jnp.where(present, unit_amount, 1).astype(jnp.int32)
    unit_op, unit_item, unit_amount, carry = _weed_repair(
        states,
        player,
        trace,
        carry,
        unit_op,
        unit_item,
        unit_amount,
        unit_count,
    )
    carry = _update_yarn_route(states, carry)

    market_op = trace.market_op[step]
    market_item = trace.market_item[step]
    market_amount = trace.market_amount[step]
    market_count = trace.market_count[step]
    convert_market = carry.yarn_route[:, None] & (step[:, None] == 192)
    market_item = jnp.where(
        convert_market & (market_item == COW_ID), SHEEP_ID, market_item
    ).astype(jnp.int8)
    convert_unit = (
        carry.yarn_route
        & (step >= 193)
        & (step <= 212)
        & (unit_count > 5)
    )
    unit_item = unit_item.at[:, 5].set(
        jnp.where(
            convert_unit & (unit_item[:, 5] == COW_ID),
            SHEEP_ID,
            unit_item[:, 5],
        )
    )
    market_op, market_item, market_amount, market_count, carry = _wool_controller(
        states,
        player,
        carry,
        market_op,
        market_item,
        market_amount,
        market_count,
    )
    market_op, market_item, market_amount = _rank_sell_slots(
        states, tables, market_op, market_item, market_amount, market_count
    )
    market_op, market_item, market_amount, market_count, carry = _relay_controller(
        states,
        player,
        trace,
        carry,
        market_op,
        market_item,
        market_amount,
        market_count,
    )
    market_op, market_item, market_amount = _rank_sell_slots(
        states, tables, market_op, market_item, market_amount, market_count
    )
    return (
        BoatleePlayerActionV1(
            unit_op,
            unit_item,
            unit_amount,
            unit_count,
            market_op,
            market_item,
            market_amount,
            market_count,
        ),
        carry,
    )


def boatlee_player_action_v1(
    states: State,
    tables: StaticTables,
    trace: BoatleeTraceV1,
    carry: BoatleePlayerCarryV1,
    player: int,
) -> tuple[Action, BoatleePlayerCarryV1]:
    """Emit the exact Boatlee action for one selected seat."""

    action, following = _player_action(states, tables, trace, carry, player)
    return Action(*action), following


def boatlee_actions_v1(
    states: State,
    tables: StaticTables,
    trace: BoatleeTraceV1,
    carry: BoatleeCarryV1,
) -> tuple[Action, BoatleeCarryV1]:
    left_action, left_carry = _player_action(
        states, tables, trace, _player_carry(carry, 0), 0
    )
    right_action, right_carry = _player_action(
        states, tables, trace, _player_carry(carry, 1), 1
    )
    action = Action(
        *(jnp.stack(values, axis=1) for values in zip(left_action, right_action, strict=True))
    )
    return action, _stack_player_carry(left_carry, right_carry)
