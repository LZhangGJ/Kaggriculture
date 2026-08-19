"""GPU-native exact controller family for the public V20/K320/Tetsutani agents.

The three public submissions share the same eight production tapes and nearly
all runtime policy code.  This module keeps the shared state machines in one
JAX implementation and exposes a static ``mode`` switch for the three small
differences:

``MODE_BOATLEE``
    Live shop routing, one-step premium sale preemption.
``MODE_RAY_K320``
    Shop-route locking, adaptive 1/2/4-step preemption, persistent repayment,
    and removal of late unfinishable WHEAT/CARROT seed buys.
``MODE_TETSUTANI``
    Live shop routing and a one- or two-step premium queue lead selected by the
    first unlocked shop.

All mutable Python dictionaries from the source agents are represented by the
carry, so batched ``lax.scan`` rollouts do not call Python or the CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import (
    FLAG_FERTILIZER_AVAILABLE,
    MARKET_MIN_INVENTORY,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    PRODUCTS,
    SHED_ACCESS,
    SHED_CAPACITY,
    SHOP_NAMES,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.types import Action, State, StaticTables


MODE_BOATLEE = 0
MODE_RAY_K320 = 1
MODE_TETSUTANI = 2

_YARN = SHOP_NAMES.index("YARN_STORE")
_BAKERY = SHOP_NAMES.index("BAKERY")
_MILK_SUPPORT = jnp.asarray(
    [SHOP_NAMES.index(name) for name in ("PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP")],
    dtype=jnp.int8,
)
_PREMIUM = jnp.asarray(
    [PRODUCTS.index(name) for name in ("STRAWBERRY", "MELON", "MILK", "WOOL")],
    dtype=jnp.int8,
)
_COUNTER_ITEMS = jnp.asarray(
    [PRODUCTS.index(name) for name in ("MELON", "MILK", "STRAWBERRY", "WOOL")],
    dtype=jnp.int8,
)
_LIQUIDATION = jnp.asarray(
    [
        PRODUCTS.index(name)
        for name in (
            "CARROT",
            "EGG",
            "FERTILIZER",
            "MELON",
            "MILK",
            "STRAWBERRY",
            "TOMATO",
            "WHEAT",
            "WOOL",
        )
    ],
    dtype=jnp.int8,
)
_ROOM_PRIORITY = jnp.asarray(
    [
        PRODUCTS.index(name)
        for name in (
            "WOOL",
            "MILK",
            "EGG",
            "MELON",
            "STRAWBERRY",
            "TOMATO",
            "CARROT",
            "FERTILIZER",
            "WHEAT",
        )
    ],
    dtype=jnp.int8,
)
# CPython 3.12 iteration order for the source's four-element set.
_EVAC_ACCESS = jnp.asarray(((4, 4), (5, 4), (5, 5), (4, 5)), dtype=jnp.int16)
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)


class HighPotentialRuntimeTablesV1(NamedTuple):
    r5_market_op: jax.Array
    r5_market_item: jax.Array
    r5_market_amount: jax.Array
    r5_market_count: jax.Array
    md_market_op: jax.Array
    md_market_item: jax.Array
    md_market_amount: jax.Array
    md_market_count: jax.Array
    future_plant_before_day28: jax.Array
    agent_market_price_lut: jax.Array


def load_high_potential_runtime_tables_v1(
    path: Path,
) -> HighPotentialRuntimeTablesV1:
    with np.load(path, allow_pickle=False) as data:
        return HighPotentialRuntimeTablesV1(
            *(jnp.asarray(data[field]) for field in HighPotentialRuntimeTablesV1._fields)
        )


class HighPotentialV20CarryV1(NamedTuple):
    legacy_decided: jax.Array
    legacy_layout: jax.Array
    ray_route_locked: jax.Array
    ray_route_id: jax.Array
    due_step: jax.Array
    due: jax.Array
    weed_active: jax.Array
    weed_start: jax.Array
    weed_intended_op: jax.Array
    weed_intended_item: jax.Array
    weed_intended_amount: jax.Array
    r5_target: jax.Array
    md_target: jax.Array
    evac_day: jax.Array
    evac_actor: jax.Array
    evac_target: jax.Array


def initialize_high_potential_v20_carry_v1(batch_size: int) -> HighPotentialV20CarryV1:
    units = (batch_size, MAX_UNITS)
    return HighPotentialV20CarryV1(
        legacy_decided=jnp.zeros((batch_size,), dtype=jnp.bool_),
        legacy_layout=jnp.zeros((batch_size,), dtype=jnp.bool_),
        ray_route_locked=jnp.zeros((batch_size,), dtype=jnp.bool_),
        ray_route_id=jnp.ones((batch_size,), dtype=jnp.int8),
        due_step=jnp.full((batch_size,), -1, dtype=jnp.int16),
        due=jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int16),
        weed_active=jnp.zeros(units, dtype=jnp.bool_),
        weed_start=jnp.full(units, -1, dtype=jnp.int16),
        weed_intended_op=jnp.full(units, UnitOp.PASS, dtype=jnp.int8),
        weed_intended_item=jnp.full(units, -1, dtype=jnp.int8),
        weed_intended_amount=jnp.ones(units, dtype=jnp.int32),
        r5_target=jnp.zeros((batch_size,), dtype=jnp.bool_),
        md_target=jnp.zeros((batch_size,), dtype=jnp.bool_),
        evac_day=jnp.full((batch_size,), -1, dtype=jnp.int8),
        evac_actor=jnp.full((batch_size,), -1, dtype=jnp.int8),
        evac_target=jnp.zeros((batch_size, 2), dtype=jnp.int16),
    )


def _town_has_first(states: State, shop_id: int, count: int) -> jax.Array:
    slots = jnp.arange(states.town_shops.shape[1])[None, :]
    active = slots < jnp.minimum(states.town_count[:, None], count)
    return jnp.any(active & (states.town_shops == shop_id), axis=1)


def _shop_route(states: State) -> jax.Array:
    """Return current route IDs 0..4 with the public source's precedence."""

    first_yarn = (states.town_count > 0) & (states.town_shops[:, 0] == _YARN)
    yarn_two = _town_has_first(states, _YARN, 2)
    yarn_three = _town_has_first(states, _YARN, 3)
    first_three = jnp.arange(states.town_shops.shape[1])[None, :] < jnp.minimum(
        states.town_count[:, None], 3
    )
    milk_support = jnp.any(
        first_three[..., None]
        & (states.town_shops[..., None] == _MILK_SUPPORT[None, None, :]),
        axis=(1, 2),
    )
    return jnp.where(
        first_yarn,
        3,
        jnp.where(yarn_two, 4, jnp.where(yarn_three, 2, jnp.where(milk_support, 0, 1))),
    ).astype(jnp.int8)


def _opponent_counts(states: State, player: int):
    rival = 1 - player
    kinds = states.tile_kind[:, rival]
    crops = states.tile_crop[:, rival]
    animals = states.tile_animal[:, rival]
    wheat = jnp.sum(crops == PRODUCTS.index("WHEAT"), axis=(1, 2))
    melon = jnp.sum(crops == PRODUCTS.index("MELON"), axis=(1, 2))
    cows = jnp.sum(animals == 1, axis=(1, 2))
    sheep = jnp.sum(animals == 2, axis=(1, 2))
    empty_pasture = jnp.sum((kinds == TileKind.PASTURE) & (animals < 0), axis=(1, 2))
    return wheat, melon, cows, sheep, empty_pasture


def _select_route(
    states: State, carry: HighPotentialV20CarryV1, player: int, mode: int
) -> tuple[jax.Array, HighPotentialV20CarryV1]:
    step = states.step.astype(jnp.int32)
    current = _shop_route(states)
    wheat, melon, cows, sheep, empty_pasture = _opponent_counts(states, player)
    decision_window = (~carry.legacy_decided) & (step >= 24) & (step < 72)
    legacy_now = (
        (wheat == 5)
        & (melon == 5)
        & (cows == 1)
        & (sheep == 4)
        & (empty_pasture == 0)
        & (states.money[:, 1 - player] <= 12)
    )
    carry = carry._replace(
        legacy_decided=carry.legacy_decided | decision_window,
        legacy_layout=jnp.where(decision_window, legacy_now, carry.legacy_layout),
    )

    if mode == MODE_RAY_K320:
        yarn_program = (current == 3) | (current == 4)
        lock_now = (~carry.ray_route_locked) & (
            yarn_program | (states.town_count >= 3) | (step >= 216)
        )
        carry = carry._replace(
            ray_route_locked=carry.ray_route_locked | lock_now,
            ray_route_id=jnp.where(lock_now, current, carry.ray_route_id).astype(jnp.int8),
        )
        current = jnp.where(carry.ray_route_locked, carry.ray_route_id, current).astype(
            jnp.int8
        )

    # The two yarn legacy tapes are byte-identical to their current versions.
    legacy_id = jnp.where(
        current == 0,
        5,
        jnp.where(current == 1, 6, jnp.where(current == 2, 7, current)),
    )
    route = jnp.where(carry.legacy_layout, legacy_id, current).astype(jnp.int32)
    return route, carry


def _tile_under_units(states: State, player: int) -> jax.Array:
    batch = jnp.arange(states.step.shape[0])[:, None]
    pos = states.unit_pos[:, player].astype(jnp.int32)
    return states.tile_kind[
        batch, player, jnp.clip(pos[..., 1], 0, 9), jnp.clip(pos[..., 0], 0, 9)
    ]


def _raw_with_weed(
    states: State,
    bank,
    route: jax.Array,
    carry: HighPotentialV20CarryV1,
    player: int,
) -> tuple[Action, HighPotentialV20CarryV1]:
    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    previous = jnp.clip(step - 1, 0, 718)
    count = jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int8)
    present = jnp.arange(MAX_UNITS)[None, :] < count[:, None]
    op = jnp.where(present, bank.unit_op[route, step], UnitOp.PASS).astype(jnp.int8)
    item = jnp.where(present, bank.unit_item[route, step], -1).astype(jnp.int8)
    amount = jnp.where(present, bank.unit_amount[route, step], 1).astype(jnp.int32)

    age = step[:, None] - carry.weed_start.astype(jnp.int32)
    existing = carry.weed_active & present & (age <= 9)
    use_intended = existing & (age == 1)
    use_replay = existing & (age >= 2) & (age <= 9)
    op = jnp.where(use_intended, carry.weed_intended_op, op)
    item = jnp.where(use_intended, carry.weed_intended_item, item)
    amount = jnp.where(use_intended, carry.weed_intended_amount, amount)
    op = jnp.where(use_replay, bank.unit_op[route, previous], op)
    item = jnp.where(use_replay, bank.unit_item[route, previous], item)
    amount = jnp.where(use_replay, bank.unit_amount[route, previous], amount)
    trigger = (
        (~existing)
        & present
        & ((op == UnitOp.BUILD_PASTURE) | (op == UnitOp.PLANT))
        & (_tile_under_units(states, player) == TileKind.WEED)
    )
    intended_op, intended_item, intended_amount = op, item, amount
    op = jnp.where(trigger, UnitOp.DIG, op).astype(jnp.int8)
    item = jnp.where(trigger, -1, item).astype(jnp.int8)
    amount = jnp.where(trigger, 1, amount).astype(jnp.int32)
    carry = carry._replace(
        weed_active=existing | trigger,
        weed_start=jnp.where(trigger, step[:, None], carry.weed_start).astype(jnp.int16),
        weed_intended_op=jnp.where(trigger, intended_op, carry.weed_intended_op).astype(
            jnp.int8
        ),
        weed_intended_item=jnp.where(
            trigger, intended_item, carry.weed_intended_item
        ).astype(jnp.int8),
        weed_intended_amount=jnp.where(
            trigger, intended_amount, carry.weed_intended_amount
        ).astype(jnp.int32),
    )
    return Action(
        unit_op=op,
        unit_item=item,
        unit_amount=amount,
        unit_count=count,
        market_op=bank.market_op[route, step],
        market_item=bank.market_item[route, step],
        market_amount=bank.market_amount[route, step],
        market_count=bank.market_count[route, step],
    ), carry


def _compact_market(action: Action, keep: jax.Array) -> Action:
    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    keep = keep & (slot < action.market_count[:, None])
    order = jnp.argsort(
        jnp.where(keep, slot, slot + MAX_MARKET_ORDERS), axis=1, stable=True
    )
    op = jnp.take_along_axis(action.market_op, order, axis=1)
    item = jnp.take_along_axis(action.market_item, order, axis=1)
    amount = jnp.take_along_axis(action.market_amount, order, axis=1)
    count = jnp.sum(keep, axis=1).astype(jnp.int8)
    active = slot < count[:, None]
    return action._replace(
        market_op=jnp.where(active, op, MarketOp.NONE).astype(jnp.int8),
        market_item=jnp.where(active, item, -1).astype(jnp.int8),
        market_amount=jnp.where(active, amount, 0).astype(jnp.int32),
        market_count=count,
    )


def _append_or_merge_sale(
    action: Action, enabled: jax.Array, product: int, quantity: jax.Array
) -> tuple[Action, jax.Array]:
    slots = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    active = slots < action.market_count[:, None]
    matching = active & (action.market_op == MarketOp.SELL) & (action.market_item == product)
    location = jnp.min(jnp.where(matching, slots, MAX_MARKET_ORDERS), axis=1)
    merge = enabled & (location < MAX_MARKET_ORDERS)
    append = enabled & (~merge) & (action.market_count < MAX_MARKET_ORDERS)
    batch = jnp.arange(action.market_count.shape[0])
    merge_slot = jnp.clip(location, 0, MAX_MARKET_ORDERS - 1)
    amount = action.market_amount.at[batch, merge_slot].set(
        jnp.where(
            merge,
            action.market_amount[batch, merge_slot] + quantity,
            action.market_amount[batch, merge_slot],
        )
    )
    slot = jnp.clip(action.market_count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1)
    op = action.market_op.at[batch, slot].set(
        jnp.where(append, MarketOp.SELL, action.market_op[batch, slot])
    )
    item = action.market_item.at[batch, slot].set(
        jnp.where(append, product, action.market_item[batch, slot])
    )
    amount = amount.at[batch, slot].set(
        jnp.where(append, quantity, amount[batch, slot])
    )
    return action._replace(
        market_op=op,
        market_item=item,
        market_amount=amount,
        market_count=(action.market_count + append.astype(jnp.int8)).astype(jnp.int8),
    ), merge | append


def _planned_sales(action: Action) -> jax.Array:
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    valid = active & (action.market_op == MarketOp.SELL) & (action.market_item >= 0) & (
        action.market_item < NUM_PRODUCTS
    )
    safe = jnp.clip(action.market_item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    out = jnp.zeros((action.market_count.shape[0], NUM_PRODUCTS), dtype=jnp.int32)
    return out.at[jnp.arange(out.shape[0])[:, None], safe].add(
        jnp.where(valid, jnp.maximum(action.market_amount, 0), 0)
    )


def _rank_sell_slots_exact(
    states: State, runtime: HighPotentialRuntimeTablesV1, action: Action
) -> Action:
    """Exact public-agent ranking, including its frozen pre-1.32.7 price model."""

    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    active = slot < action.market_count[:, None]
    sell = active & (action.market_op == MarketOp.SELL) & (action.market_item >= 0) & (
        action.market_item < NUM_PRODUCTS
    )
    safe_item = jnp.clip(action.market_item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    quantity = jnp.maximum(action.market_amount, 0).astype(jnp.int32)
    inventory = jnp.take_along_axis(states.market_inventory, safe_item, axis=1)
    current_quote = jnp.take_along_axis(states.market_price, safe_item, axis=1)
    later_index = jnp.clip(
        inventory + quantity - MARKET_MIN_INVENTORY,
        0,
        runtime.agent_market_price_lut.shape[1] - 1,
    )
    later_quote = runtime.agent_market_price_lut[safe_item, later_index].astype(jnp.int32)
    impact = quantity.astype(jnp.float32) * jnp.maximum(
        current_quote - later_quote, 0
    ).astype(jnp.float32)
    shop_products = {
        "BAKERY": ("EGG", "WHEAT"),
        "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
        "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
        "YARN_STORE": ("WOOL",),
        "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
        "PET_CAFE": ("CARROT",),
        "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
        "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
    }
    demand_table = jnp.asarray(
        [
            [
                (2 if len(shop_products[shop]) == 1 else 1)
                if product in shop_products[shop]
                else 0
                for product in PRODUCTS
            ]
            for shop in SHOP_NAMES
        ],
        dtype=jnp.float32,
    )
    town_active = jnp.arange(states.town_shops.shape[1])[None, :] < states.town_count[:, None]
    safe_shops = jnp.clip(states.town_shops.astype(jnp.int32), 0, len(SHOP_NAMES) - 1)
    demand_by_item = jnp.sum(
        demand_table[safe_shops] * town_active[..., None], axis=1
    ) * 6.0
    demand_by_item = demand_by_item + jnp.asarray(
        [1.0] * (NUM_PRODUCTS - 1) + [0.0], dtype=jnp.float32
    )[None, :]
    demand = jnp.maximum(
        jnp.take_along_axis(demand_by_item, safe_item, axis=1), 0.25
    )
    excess = jnp.maximum(inventory + quantity - 10000, 0).astype(jnp.float32)
    urgency = jnp.minimum(1.0, (excess / demand) / 10.0)
    score = jnp.where(sell, impact * (1.0 + 0.25 * urgency), -jnp.inf)
    ranked_index = jnp.argsort(-score, axis=1, stable=True)
    ranked_op = jnp.take_along_axis(action.market_op, ranked_index, axis=1)
    ranked_item = jnp.take_along_axis(action.market_item, ranked_index, axis=1)
    ranked_amount = jnp.take_along_axis(action.market_amount, ranked_index, axis=1)
    ordinal = jnp.maximum(jnp.cumsum(sell, axis=1) - 1, 0)
    return action._replace(
        market_op=jnp.where(sell, jnp.take_along_axis(ranked_op, ordinal, axis=1), action.market_op),
        market_item=jnp.where(
            sell, jnp.take_along_axis(ranked_item, ordinal, axis=1), action.market_item
        ),
        market_amount=jnp.where(
            sell, jnp.take_along_axis(ranked_amount, ordinal, axis=1), action.market_amount
        ),
    )


def _pickup_reserve(action: Action) -> jax.Array:
    present = jnp.arange(MAX_UNITS)[None, :] < action.unit_count[:, None]
    valid = present & (action.unit_op == UnitOp.PICKUP) & (action.unit_item >= 0) & (
        action.unit_item < NUM_PRODUCTS
    )
    safe = jnp.clip(action.unit_item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    out = jnp.zeros((action.unit_count.shape[0], NUM_PRODUCTS), dtype=jnp.int32)
    return out.at[jnp.arange(out.shape[0])[:, None], safe].add(
        jnp.where(valid, jnp.maximum(action.unit_amount, 0), 0)
    )


def _move_toward(position: jax.Array, target: jax.Array) -> jax.Array:
    x, y = position[:, 0], position[:, 1]
    tx, ty = target[:, 0], target[:, 1]
    return jnp.where(
        x < tx,
        UnitOp.EAST,
        jnp.where(
            x > tx,
            UnitOp.WEST,
            jnp.where(y < ty, UnitOp.SOUTH, jnp.where(y > ty, UnitOp.NORTH, UnitOp.PASS)),
        ),
    ).astype(jnp.int8)


def _room_evac(
    states: State,
    action: Action,
    carry: HighPotentialV20CarryV1,
    player: int,
) -> tuple[Action, HighPotentialV20CarryV1]:
    step = states.step.astype(jnp.int32)
    hour = step % 24
    day = step // 24
    new_day = carry.evac_day.astype(jnp.int32) != day
    actor = jnp.where(new_day, -1, carry.evac_actor).astype(jnp.int8)
    target = carry.evac_target
    batch = jnp.arange(step.shape[0])
    total = jnp.sum(states.shed[:, player].astype(jnp.int32), axis=1) + jnp.sum(
        states.unit_inventory[:, player].astype(jnp.int32), axis=(1, 2)
    )

    choose_window = (step >= 648) & (hour == 21) & (actor < 0) & (total > 100)
    best_key = jnp.full(step.shape, 1_000_000, dtype=jnp.int32)
    best_actor = jnp.full(step.shape, -1, dtype=jnp.int8)
    best_target = jnp.zeros((step.shape[0], 2), dtype=jnp.int16)
    for unit in range(MAX_UNITS):
        present = unit < action.unit_count.astype(jnp.int32)
        saleable = jnp.sum(
            states.unit_inventory[:, player, unit, :NUM_PRODUCTS].astype(jnp.int32), axis=1
        )
        position = states.unit_pos[:, player, unit].astype(jnp.int16)
        distances = jnp.sum(jnp.abs(position[:, None, :] - _EVAC_ACCESS[None, :, :]), axis=2)
        access_index = jnp.argmin(distances, axis=1)
        unit_target = _EVAC_ACCESS[access_index]
        distance = jnp.min(distances, axis=1)
        eligible = (
            choose_window
            & present
            & (saleable > 0)
            & (action.unit_op[:, unit] == UnitOp.PASS)
            & (distance <= 2)
        )
        # Lexicographic source key: (distance, -saleable, actor, target).
        key = distance.astype(jnp.int32) * 100_000 - saleable * 100 + unit
        improve = eligible & (key < best_key)
        best_key = jnp.where(improve, key, best_key)
        best_actor = jnp.where(improve, unit, best_actor).astype(jnp.int8)
        best_target = jnp.where(improve[:, None], unit_target, best_target).astype(jnp.int16)
    selected = best_actor >= 0
    actor = jnp.where(selected, best_actor, actor).astype(jnp.int8)
    target = jnp.where(selected[:, None], best_target, target).astype(jnp.int16)

    active = (step >= 648) & (hour >= 21) & (actor >= 0)
    safe_actor = jnp.clip(actor.astype(jnp.int32), 0, MAX_UNITS - 1)
    position = states.unit_pos[batch, player, safe_actor].astype(jnp.int16)
    at_target = jnp.all(position == target, axis=1)
    move = active & (~at_target)
    drop = active & at_target & (hour == 23)
    old_op = action.unit_op[batch, safe_actor]
    op = action.unit_op.at[batch, safe_actor].set(
        jnp.where(move, _move_toward(position, target), jnp.where(drop, UnitOp.DROP, old_op))
    )
    action = action._replace(unit_op=op)

    existing = _planned_sales(action)
    needed = jnp.maximum(total - SHED_CAPACITY, 0)
    for product in _ROOM_PRIORITY.tolist():
        inventory = states.unit_inventory[batch, player, safe_actor, product].astype(jnp.int32)
        available = jnp.maximum(inventory - existing[:, product], 0)
        quantity = jnp.minimum(needed, available)
        action, changed = _append_or_merge_sale(
            action, drop & (quantity > 0), product, quantity
        )
        used = jnp.where(changed, quantity, 0)
        needed = jnp.maximum(needed - used, 0)
        existing = existing.at[:, product].add(used)
    carry = carry._replace(
        evac_day=day.astype(jnp.int8),
        evac_actor=actor,
        evac_target=target,
    )
    return action, carry


def _repay(
    action: Action,
    carry: HighPotentialV20CarryV1,
    step: jax.Array,
    mode: int,
) -> tuple[Action, HighPotentialV20CarryV1]:
    due = carry.due.astype(jnp.int32)
    if mode == MODE_RAY_K320:
        enabled = jnp.sum(due, axis=1) > 0
    else:
        enabled = carry.due_step.astype(jnp.int32) == step
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    keep = active.copy()
    amount = action.market_amount.astype(jnp.int32)
    remaining = due
    for slot in range(MAX_MARKET_ORDERS):
        is_sell = active[:, slot] & (action.market_op[:, slot] == MarketOp.SELL)
        item = jnp.clip(action.market_item[:, slot].astype(jnp.int32), 0, NUM_PRODUCTS - 1)
        owed = jnp.take_along_axis(remaining, item[:, None], axis=1)[:, 0]
        requested = jnp.maximum(amount[:, slot], 0)
        reduction = jnp.where(enabled & is_sell, jnp.minimum(requested, owed), 0)
        amount = amount.at[:, slot].add(-reduction)
        remaining = remaining.at[jnp.arange(step.shape[0]), item].add(-reduction)
        keep = keep.at[:, slot].set(~(enabled & is_sell & (amount[:, slot] <= 0)))
    action = _compact_market(action._replace(market_amount=amount), keep)
    if mode == MODE_RAY_K320:
        still_due = jnp.sum(remaining, axis=1) > 0
        due_step = jnp.where(still_due, step + 1, -1).astype(jnp.int16)
        due = jnp.where(still_due[:, None], remaining, 0).astype(jnp.int16)
    else:
        expired = (carry.due_step >= 0) & (carry.due_step.astype(jnp.int32) < step)
        clear = enabled | expired
        due_step = jnp.where(clear, -1, carry.due_step).astype(jnp.int16)
        due = jnp.where(clear[:, None], 0, carry.due).astype(jnp.int16)
    return action, carry._replace(due_step=due_step, due=due)


def _clone_distance(states: State) -> jax.Array:
    kinds, crops, animals = states.tile_kind, states.tile_crop, states.tile_animal
    columns = (
        jnp.sum(crops == 1, axis=(2, 3)),
        jnp.sum(kinds == TileKind.COOP, axis=(2, 3)),
        jnp.sum(animals == 1, axis=(2, 3)),
        jnp.sum(animals == 0, axis=(2, 3)),
        jnp.sum(crops == 4, axis=(2, 3)),
        jnp.sum(kinds == TileKind.PASTURE, axis=(2, 3)),
        jnp.sum(animals == 2, axis=(2, 3)),
        jnp.sum(crops == 3, axis=(2, 3)),
        jnp.sum(crops == 2, axis=(2, 3)),
        jnp.sum(kinds == TileKind.WEED, axis=(2, 3)),
        jnp.sum(crops == 0, axis=(2, 3)),
    )
    counts = jnp.stack(columns, axis=-1).astype(jnp.int32)
    hands = jnp.sum(states.unit_active, axis=2).astype(jnp.int32) - 1
    return (
        jnp.abs(hands[:, 0] - hands[:, 1])
        + 3
        * jnp.abs(
            states.unlocked_count[:, 0].astype(jnp.int32)
            - states.unlocked_count[:, 1].astype(jnp.int32)
        )
        + jnp.sum(jnp.abs(counts[:, 0] - counts[:, 1]), axis=1)
    )


def _projected_shed_without_pickups(states: State, action: Action, player: int) -> jax.Array:
    shed = states.shed[:, player].astype(jnp.int32)
    batch = jnp.arange(states.step.shape[0])
    positions = states.unit_pos[:, player].astype(jnp.int16)
    inventory = states.unit_inventory[:, player].astype(jnp.int32)
    order_keys = states.unit_inventory_order[:, player].astype(jnp.int32)
    for unit in range(MAX_UNITS):
        present = unit < action.unit_count.astype(jnp.int32)
        position = positions[:, unit]
        at_shed = jnp.any(
            jnp.all(position[:, None, :] == _SHED_ACCESS[None, :, :], axis=2), axis=1
        )
        drop = present & at_shed & (action.unit_op[:, unit] == UnitOp.DROP)
        actor_inventory = inventory[:, unit]
        item_order = jnp.argsort(
            jnp.where(order_keys[:, unit] >= 0, order_keys[:, unit], 32767),
            axis=1,
            stable=True,
        )
        for ordinal in range(NUM_SHED_ITEMS):
            product = item_order[:, ordinal].astype(jnp.int32)
            available = jnp.take_along_axis(
                actor_inventory, product[:, None], axis=1
            )[:, 0]
            room = jnp.maximum(SHED_CAPACITY - jnp.sum(shed, axis=1), 0)
            moved = jnp.where(drop, jnp.minimum(jnp.maximum(available, 0), room), 0)
            shed = shed.at[batch, product].add(moved)
            actor_inventory = actor_inventory.at[batch, product].add(-moved)
        place = (
            present
            & at_shed
            & (action.unit_op[:, unit] == UnitOp.PLACE)
            & (action.unit_item[:, unit] >= 0)
            & (action.unit_item[:, unit] < NUM_SHED_ITEMS)
        )
        safe_item = jnp.clip(action.unit_item[:, unit].astype(jnp.int32), 0, NUM_SHED_ITEMS - 1)
        # Animal PLACE onto its matching empty structure is not a shed deposit.
        x, y = jnp.clip(position[:, 0], 0, 9), jnp.clip(position[:, 1], 0, 9)
        animal = safe_item - NUM_PRODUCTS
        valid_animal = (animal >= 0) & (animal < 3)
        structure = jnp.asarray((TileKind.COOP, TileKind.PASTURE, TileKind.PASTURE), dtype=jnp.int8)[
            jnp.clip(animal, 0, 2)
        ]
        matching = (
            valid_animal
            & (states.tile_kind[batch, player, y, x] == structure)
            & (states.tile_animal[batch, player, y, x] < 0)
        )
        place = place & (~matching)
        available = inventory[batch, unit, safe_item]
        room = jnp.maximum(SHED_CAPACITY - jnp.sum(shed, axis=1), 0)
        moved = jnp.where(
            place,
            jnp.minimum(
                jnp.minimum(jnp.maximum(action.unit_amount[:, unit], 0), available), room
            ),
            0,
        )
        shed = shed.at[batch, safe_item].add(moved)
    return shed


def _future_premium(
    bank,
    route: jax.Array,
    step: jax.Array,
    states: State,
    carry: HighPotentialV20CarryV1,
    mode: int,
) -> tuple[jax.Array, jax.Array]:
    first = states.town_shops[:, 0]
    if mode == MODE_TETSUTANI:
        horizon = jnp.where(
            (states.town_count > 0) & ((first == _YARN) | (first == _BAKERY)), 1, 2
        ).astype(jnp.int32)
        cumulative = False
    elif mode == MODE_RAY_K320:
        horizon = jnp.where(
            carry.legacy_layout,
            4,
            jnp.where(_clone_distance(states) <= 6, 2, 1),
        ).astype(jnp.int32)
        cumulative = True
    else:
        horizon = jnp.ones(step.shape, dtype=jnp.int32)
        cumulative = False
    future = jnp.zeros((step.shape[0], NUM_PRODUCTS), dtype=jnp.int32)
    for ahead in range(1, 5):
        use = (ahead <= horizon) if cumulative else (ahead == horizon)
        future_step = jnp.clip(step + ahead, 0, 718)
        in_range = step + ahead < 719
        active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < bank.market_count[
            route, future_step
        ][:, None]
        for product in _PREMIUM.tolist():
            quantity = jnp.sum(
                jnp.where(
                    active
                    & (bank.market_op[route, future_step] == MarketOp.SELL)
                    & (bank.market_item[route, future_step] == product),
                    jnp.maximum(bank.market_amount[route, future_step], 0),
                    0,
                ),
                axis=1,
            )
            future = future.at[:, product].add(
                jnp.where(use & in_range, quantity, 0)
            )
    return future, horizon


def _preempt(
    states: State,
    bank,
    route: jax.Array,
    action: Action,
    carry: HighPotentialV20CarryV1,
    player: int,
    mode: int,
) -> tuple[Action, HighPotentialV20CarryV1]:
    step = states.step.astype(jnp.int32)
    eligible_row = (
        (step >= 120)
        & (step < 680)
        & (jnp.sum(carry.due.astype(jnp.int32), axis=1) == 0)
        & (_clone_distance(states) <= 6)
        & (action.market_count < MAX_MARKET_ORDERS)
    )
    future, horizon = _future_premium(bank, route, step, states, carry, mode)
    projected = _projected_shed_without_pickups(states, action, player)
    planned = _planned_sales(action)
    remaining = jnp.maximum(projected[:, :NUM_PRODUCTS] - planned, 0)
    shifted = jnp.zeros_like(carry.due, dtype=jnp.int32)
    for product in _PREMIUM.tolist():
        future_quantity = future[:, product]
        quantity = jnp.minimum(
            jnp.minimum(remaining[:, product], future_quantity),
            12,
        )
        enabled = eligible_row & (future_quantity >= 4) & (quantity > 0)
        # Source preemption always appends; it does not merge an existing sale.
        batch = jnp.arange(step.shape[0])
        slot = jnp.clip(action.market_count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1)
        append = enabled & (action.market_count < MAX_MARKET_ORDERS)
        action = action._replace(
            market_op=action.market_op.at[batch, slot].set(
                jnp.where(append, MarketOp.SELL, action.market_op[batch, slot])
            ),
            market_item=action.market_item.at[batch, slot].set(
                jnp.where(append, product, action.market_item[batch, slot])
            ),
            market_amount=action.market_amount.at[batch, slot].set(
                jnp.where(append, quantity, action.market_amount[batch, slot])
            ),
            market_count=(action.market_count + append.astype(jnp.int8)).astype(jnp.int8),
        )
        remaining = remaining.at[:, product].add(-jnp.where(append, quantity, 0))
        shifted = shifted.at[:, product].set(jnp.where(append, quantity, 0))
    changed = jnp.sum(shifted, axis=1) > 0
    carry = carry._replace(
        due_step=jnp.where(changed, step + horizon, carry.due_step).astype(jnp.int16),
        due=jnp.where(changed[:, None], shifted, carry.due).astype(jnp.int16),
    )
    return action, carry


def _town_demand_at(states: State, product: int, requested_step: jax.Array) -> jax.Array:
    demand = ((product != PRODUCTS.index("FERTILIZER")) & (requested_step % 24 == 0)).astype(
        jnp.int32
    )
    active = jnp.arange(states.town_shops.shape[1])[None, :] < states.town_count[:, None]
    # Same demand table as the source wrapper.
    shop_products = {
        "BAKERY": ("EGG", "WHEAT"),
        "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
        "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
        "YARN_STORE": ("WOOL",),
        "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
        "PET_CAFE": ("CARROT",),
        "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
        "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
    }
    weights = jnp.asarray(
        [
            (2 if len(shop_products[name]) == 1 else 1)
            if PRODUCTS[product] in shop_products[name]
            else 0
            for name in SHOP_NAMES
        ],
        dtype=jnp.int32,
    )
    shop = jnp.clip(states.town_shops.astype(jnp.int32), 0, len(SHOP_NAMES) - 1)
    increment = jnp.sum(jnp.where(active, weights[shop], 0), axis=1)
    return demand + jnp.where(requested_step % 4 == 0, increment, 0)


def _schedule_targets(
    op: jax.Array,
    item: jax.Array,
    amount: jax.Array,
    count: jax.Array,
    future_step: jax.Array,
) -> jax.Array:
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < count[future_step][:, None]
    safe = jnp.clip(item[future_step].astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    valid = active & (op[future_step] == MarketOp.SELL)
    out = jnp.zeros((future_step.shape[0], NUM_PRODUCTS), dtype=jnp.int32)
    return out.at[jnp.arange(out.shape[0])[:, None], safe].add(
        jnp.where(valid, jnp.maximum(amount[future_step], 0), 0)
    )


def _market_counters(
    states: State,
    action: Action,
    carry: HighPotentialV20CarryV1,
    runtime: HighPotentialRuntimeTablesV1,
    player: int,
) -> tuple[Action, HighPotentialV20CarryV1]:
    step = states.step.astype(jnp.int32)
    _, _, cows, sheep, _ = _opponent_counts(states, player)
    r5 = carry.r5_target | ((step >= 24) & (sheep >= 4) & (cows <= 3))
    md = carry.md_target | (
        (step >= 160)
        & (
            ((states.unlocked_count[:, 1 - player] >= 2) & (cows >= 4) & (sheep <= 2))
            | (cows >= 9)
        )
    )
    carry = carry._replace(r5_target=r5, md_target=md)
    planned = _planned_sales(action)
    reserve = _pickup_reserve(action)
    shed = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int32)

    future_r5 = jnp.clip(step + 3, 0, 718)
    r5_targets = _schedule_targets(
        runtime.r5_market_op,
        runtime.r5_market_item,
        runtime.r5_market_amount,
        runtime.r5_market_count,
        future_r5,
    )
    for product in _COUNTER_ITEMS.tolist():
        target = r5_targets[:, product]
        clean = (_town_demand_at(states, product, step) == 0) & (
            _town_demand_at(states, product, step + 1) == 0
        )
        available = jnp.maximum(shed[:, product] - planned[:, product] - reserve[:, product], 0)
        desired = jnp.maximum(1, jnp.rint(target.astype(jnp.float32) * 0.5).astype(jnp.int32))
        quantity = jnp.minimum(available, desired)
        enabled = r5 & (step + 3 < 719) & (target > 0) & clean & (quantity > 0)
        action, changed = _append_or_merge_sale(action, enabled, product, quantity)
        planned = planned.at[:, product].add(jnp.where(changed, quantity, 0))

    future_md = jnp.clip(step + 1, 0, 718)
    md_targets = _schedule_targets(
        runtime.md_market_op,
        runtime.md_market_item,
        runtime.md_market_amount,
        runtime.md_market_count,
        future_md,
    )
    for product in _COUNTER_ITEMS.tolist():
        target = md_targets[:, product]
        available = jnp.maximum(shed[:, product] - planned[:, product] - reserve[:, product], 0)
        quantity = jnp.minimum(available, jnp.maximum(1, target * 2))
        enabled = md & (step + 1 < 719) & (target > 0) & (quantity > 0)
        action, changed = _append_or_merge_sale(action, enabled, product, quantity)
        planned = planned.at[:, product].add(jnp.where(changed, quantity, 0))
    return action, carry


def _room_guard(states: State, action: Action, player: int) -> Action:
    step = states.step.astype(jnp.int32)
    enabled = step % 24 == 23
    batch = jnp.arange(step.shape[0])
    present = jnp.arange(MAX_UNITS)[None, :] < action.unit_count[:, None]
    pos = states.unit_pos[:, player].astype(jnp.int32)
    x, y = jnp.clip(pos[..., 0], 0, 9), jnp.clip(pos[..., 1], 0, 9)
    tile_yield = states.tile_yield[batch[:, None], player, y, x].astype(jnp.int32)
    tile_flags = states.tile_flags[batch[:, None], player, y, x].astype(jnp.int32)
    harvest = present & (action.unit_op == UnitOp.HARVEST)
    fertilizer = present & (action.unit_op == UnitOp.COLLECT_FERTILIZER) & (
        (tile_flags & FLAG_FERTILIZER_AVAILABLE) != 0
    )
    produced = jnp.sum(jnp.where(harvest, jnp.maximum(tile_yield, 0), 0), axis=1) + jnp.sum(
        fertilizer, axis=1
    )
    consumed = jnp.sum(
        present
        & (
            (action.unit_op == UnitOp.FEED)
            | (action.unit_op == UnitOp.FERTILIZE)
            | (
                (action.unit_op == UnitOp.PLACE)
                & (action.unit_item >= NUM_PRODUCTS)
                & (action.unit_item < NUM_SHED_ITEMS)
            )
        ),
        axis=1,
    )
    active_market = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    planned = _planned_sales(action)
    buys = jnp.sum(
        jnp.where(
            active_market
            & (
                (action.market_op == MarketOp.BUY_PRODUCT)
                | (action.market_op == MarketOp.BUY_ANIMAL)
            ),
            jnp.maximum(action.market_amount, 0),
            0,
        ),
        axis=1,
    )
    shed = states.shed[:, player].astype(jnp.int32)
    carried = jnp.sum(states.unit_inventory[:, player].astype(jnp.int32), axis=(1, 2))
    existing_sells = jnp.sum(
        jnp.minimum(shed[:, :NUM_PRODUCTS], planned), axis=1
    )
    needed = jnp.maximum(
        jnp.sum(shed, axis=1) + carried + produced - consumed + buys - existing_sells - 100,
        0,
    )
    for product in _ROOM_PRIORITY.tolist():
        available = jnp.maximum(shed[:, product] - planned[:, product], 0)
        quantity = jnp.minimum(needed, available)
        action, changed = _append_or_merge_sale(
            action, enabled & (quantity > 0), product, quantity
        )
        used = jnp.where(changed, quantity, 0)
        needed = jnp.maximum(needed - used, 0)
        planned = planned.at[:, product].add(used)
    return action


def _terminal_liquidation(states: State, action: Action, player: int) -> Action:
    step = states.step.astype(jnp.int32)
    planned = _planned_sales(action)
    shed = states.shed[:, player].astype(jnp.int32)
    for product in _LIQUIDATION.tolist():
        available = shed[:, product]
        quantity = jnp.where(step >= 718, available, jnp.maximum(available - planned[:, product], 0))
        batch = jnp.arange(step.shape[0])
        slot = jnp.clip(action.market_count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1)
        append = (step >= 716) & (quantity > 0) & (action.market_count < MAX_MARKET_ORDERS)
        action = action._replace(
            market_op=action.market_op.at[batch, slot].set(
                jnp.where(append, MarketOp.SELL, action.market_op[batch, slot])
            ),
            market_item=action.market_item.at[batch, slot].set(
                jnp.where(append, product, action.market_item[batch, slot])
            ),
            market_amount=action.market_amount.at[batch, slot].set(
                jnp.where(append, quantity, action.market_amount[batch, slot])
            ),
            market_count=(action.market_count + append.astype(jnp.int8)).astype(jnp.int8),
        )
    return action


def _trim_ray_seeds(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    route: jax.Array,
    action: Action,
    player: int,
) -> Action:
    step = states.step.astype(jnp.int32)
    for crop_ordinal, crop in enumerate(
        (PRODUCTS.index("WHEAT"), PRODUCTS.index("CARROT"))
    ):
        later = runtime.future_plant_before_day28[route, step, crop_ordinal].astype(
            jnp.int32
        )
        current = jnp.sum(
            (jnp.arange(MAX_UNITS)[None, :] < action.unit_count[:, None])
            & (action.unit_op == UnitOp.PLANT)
            & (action.unit_item == crop),
            axis=1,
        ).astype(jnp.int32)
        have = states.seeds[:, player, crop].astype(jnp.int32)
        slots = jnp.arange(MAX_MARKET_ORDERS)[None, :]
        active = slots < action.market_count[:, None]
        matching = active & (action.market_op == MarketOp.BUY_SEED) & (
            action.market_item == crop
        )
        remaining = jnp.maximum(current + later - have, 0)
        amount = action.market_amount.astype(jnp.int32)
        keep = active.copy()
        for slot in range(MAX_MARKET_ORDERS):
            quantity = jnp.minimum(jnp.maximum(amount[:, slot], 0), remaining)
            use = matching[:, slot]
            amount = amount.at[:, slot].set(jnp.where(use, quantity, amount[:, slot]))
            keep = keep.at[:, slot].set(~(use & (quantity <= 0)))
            remaining = jnp.maximum(remaining - jnp.where(use, quantity, 0), 0)
        action = _compact_market(action._replace(market_amount=amount), keep)
    return action


def high_potential_v20_player_action_v1(
    states: State,
    tables: StaticTables,
    bank,
    runtime: HighPotentialRuntimeTablesV1,
    carry: HighPotentialV20CarryV1,
    player: int,
    mode: int,
) -> tuple[Action, HighPotentialV20CarryV1]:
    """Emit the selected public agent action for one seat and a whole batch."""

    route, carry = _select_route(states, carry, player, mode)
    action, carry = _raw_with_weed(states, bank, route, carry, player)
    # Feed guard is present in the source but frozen off in all three agents.
    action, carry = _room_evac(states, action, carry, player)
    action, carry = _repay(action, carry, states.step.astype(jnp.int32), mode)
    action = _rank_sell_slots_exact(states, runtime, action)
    action, carry = _preempt(states, bank, route, action, carry, player, mode)
    action, carry = _market_counters(states, action, carry, runtime, player)
    action = _room_guard(states, action, player)
    action = _terminal_liquidation(states, action, player)
    if mode == MODE_RAY_K320:
        action = _trim_ray_seeds(states, runtime, route, action, player)
    return action, carry
