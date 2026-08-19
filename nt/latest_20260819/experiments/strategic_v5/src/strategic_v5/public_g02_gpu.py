"""Exact GPU controller for ``14_rank_agent_v17.py`` (public G02).

The public agent is not a fixed replay.  It repairs replay actions blocked by
weeds and, after detecting a near-mirror opponent, advances one premium-product
sale from the next four steps.  This module keeps those two state machines on
device so the official Python policy can be replaced in batched JAX arenas.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_STRUCTURE,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_ANIMALS,
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
from kaggriculture_jax.types import Action, State


_PREMIUM_NAMES = ("MELON", "MILK", "STRAWBERRY", "WOOL")
_PREMIUM_IDS = jnp.asarray([PRODUCTS.index(name) for name in _PREMIUM_NAMES], dtype=jnp.int8)
_PREMIUM_WEIGHTS = jnp.asarray((3.5, 2.0, 2.0, 3.2), dtype=jnp.float32)
_PREMIUM_BASE = jnp.asarray((250.0, 160.0, 120.0, 200.0), dtype=jnp.float32)
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)
_ANIMAL_STRUCTURE = jnp.asarray(ANIMAL_STRUCTURE, dtype=jnp.int8)

# unlocked_shops demand contribution used by the official Python wrapper.  The
# simulator keeps shop IDs in lexical order, so derive rows from names instead
# of relying on the order in the official configuration file.
_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
_SHOP_DEMAND = jnp.asarray(
    [
        [
            (2 if len(_SHOP_PRODUCTS[shop]) == 1 else 1)
            if product in _SHOP_PRODUCTS[shop]
            else 0
            for product in PRODUCTS
        ]
        for shop in SHOP_NAMES
    ],
    dtype=jnp.int16,
)


class PublicG02CarryV1(NamedTuple):
    clone_confidence: jax.Array
    due_step: jax.Array
    due: jax.Array
    repair_start: jax.Array
    repair_op: jax.Array
    repair_item: jax.Array
    repair_amount: jax.Array
    last_preempt: jax.Array
    debt_ring: jax.Array
    race_prev_inventory: jax.Array
    race_own_sells: jax.Array
    race_shop_demand: jax.Array
    race_scores: jax.Array
    race_events: jax.Array
    race_horizon: jax.Array
    race_last_step: jax.Array


def initialize_public_g02_carry_v1(batch_size: int) -> PublicG02CarryV1:
    units = (batch_size, MAX_UNITS)
    return PublicG02CarryV1(
        clone_confidence=jnp.zeros((batch_size,), dtype=jnp.int8),
        due_step=jnp.full((batch_size,), -1, dtype=jnp.int16),
        due=jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int16),
        repair_start=jnp.full(units, -1, dtype=jnp.int16),
        repair_op=jnp.full(units, UnitOp.PASS, dtype=jnp.int8),
        repair_item=jnp.full(units, -1, dtype=jnp.int8),
        repair_amount=jnp.ones(units, dtype=jnp.int32),
        last_preempt=jnp.full((batch_size,), -32768, dtype=jnp.int32),
        debt_ring=jnp.zeros((batch_size, 8, NUM_PRODUCTS), dtype=jnp.int16),
        race_prev_inventory=jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int32),
        race_own_sells=jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int16),
        race_shop_demand=jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int16),
        race_scores=jnp.zeros((batch_size, 6), dtype=jnp.float32),
        race_events=jnp.zeros((batch_size,), dtype=jnp.int16),
        race_horizon=jnp.full((batch_size,), 4, dtype=jnp.int8),
        race_last_step=jnp.full((batch_size,), -1, dtype=jnp.int16),
    )


def _raw_action(states: State, bank, skeleton_id: jax.Array, player: int) -> Action:
    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    actual_count = jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int8)
    present = jnp.arange(MAX_UNITS)[None, :] < actual_count[:, None]
    return Action(
        jnp.where(present, bank.unit_op[skeleton_id, step], UnitOp.PASS).astype(jnp.int8),
        jnp.where(present, bank.unit_item[skeleton_id, step], -1).astype(jnp.int8),
        jnp.where(present, bank.unit_amount[skeleton_id, step], 1).astype(jnp.int32),
        actual_count,
        bank.market_op[skeleton_id, step],
        bank.market_item[skeleton_id, step],
        bank.market_amount[skeleton_id, step],
        bank.market_count[skeleton_id, step],
    )


def _profile_distance(states: State) -> jax.Array:
    kinds = states.tile_kind.astype(jnp.int32)
    crops = states.tile_crop.astype(jnp.int32)
    animals = states.tile_animal.astype(jnp.int32)
    columns = (
        jnp.sum(crops == 1, axis=(2, 3)),  # CARROT
        jnp.sum((animals < 0) & (crops < 0) & (kinds == TileKind.COOP), axis=(2, 3)),
        jnp.sum(animals == 1, axis=(2, 3)),  # COW
        jnp.sum(animals == 0, axis=(2, 3)),  # GOOSE
        jnp.sum(crops == 4, axis=(2, 3)),  # MELON
        jnp.sum((animals < 0) & (crops < 0) & (kinds == TileKind.PASTURE), axis=(2, 3)),
        jnp.sum(animals == 2, axis=(2, 3)),  # SHEEP
        jnp.sum(crops == 3, axis=(2, 3)),  # STRAWBERRY
        jnp.sum(crops == 2, axis=(2, 3)),  # TOMATO
        jnp.sum((animals < 0) & (crops < 0) & (kinds == TileKind.WEED), axis=(2, 3)),
        jnp.sum(crops == 0, axis=(2, 3)),  # WHEAT
    )
    counts = jnp.stack(columns, axis=-1).astype(jnp.int32)
    active = states.unit_active.astype(jnp.bool_)
    hands = jnp.sum(active, axis=2).astype(jnp.int32) - 1
    positions = states.unit_pos.astype(jnp.int32)
    pos_id = jnp.clip(positions[..., 1], 0, 9) * 10 + jnp.clip(positions[..., 0], 0, 9)
    histogram = jnp.sum(jax.nn.one_hot(pos_id, 100, dtype=jnp.int16) * active[..., None], axis=2)
    position_differs = jnp.any(histogram[:, 0] != histogram[:, 1], axis=1)
    return (
        jnp.abs(hands[:, 0] - hands[:, 1])
        + 3 * jnp.abs(states.unlocked_count[:, 0].astype(jnp.int32) - states.unlocked_count[:, 1].astype(jnp.int32))
        + jnp.sum(jnp.abs(counts[:, 0] - counts[:, 1]), axis=1)
        + 2 * position_differs.astype(jnp.int32)
    )


def _update_clone(states: State, confidence: jax.Array) -> jax.Array:
    step = states.step.astype(jnp.int32)
    checkpoint = (step == 4) | (step == 24) | ((step >= 48) & (step % 24 == 0))
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
    reset = step == 0
    confidence = jnp.where(reset, 0, confidence).astype(jnp.int8)
    return jnp.where(checkpoint, following, confidence).astype(jnp.int8)


def _compact_market(action: Action, keep: jax.Array) -> Action:
    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    keep = keep & (slot < action.market_count[:, None])
    order = jnp.argsort(jnp.where(keep, slot, slot + MAX_MARKET_ORDERS), axis=1, stable=True)
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


def _append_sell(action: Action, enabled: jax.Array, item: jax.Array, amount: jax.Array) -> Action:
    batch = jnp.arange(action.market_count.shape[0])
    slot = jnp.clip(action.market_count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1)
    return action._replace(
        market_op=action.market_op.at[batch, slot].set(jnp.where(enabled, MarketOp.SELL, action.market_op[batch, slot])),
        market_item=action.market_item.at[batch, slot].set(jnp.where(enabled, item, action.market_item[batch, slot])),
        market_amount=action.market_amount.at[batch, slot].set(jnp.where(enabled, amount, action.market_amount[batch, slot])),
        market_count=(action.market_count + enabled.astype(jnp.int8)).astype(jnp.int8),
    )


def _repay(action: Action, step: jax.Array, due_step: jax.Array, due: jax.Array):
    due_now = due_step.astype(jnp.int32) == step.astype(jnp.int32)
    expired = (due_step >= 0) & (due_step.astype(jnp.int32) < step.astype(jnp.int32))
    amount = action.market_amount.astype(jnp.int32)
    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    active = slot < action.market_count[:, None]
    remove = jnp.zeros_like(active)
    for product_id in _PREMIUM_IDS.tolist():
        matching = active & (action.market_op == MarketOp.SELL) & (action.market_item == product_id)
        requested = jnp.where(matching, jnp.maximum(amount, 0), 0)
        prefix = jnp.cumsum(requested, axis=1) - requested
        reduction = jnp.minimum(requested, jnp.maximum(due[:, product_id:product_id + 1].astype(jnp.int32) - prefix, 0))
        amount = jnp.where(due_now[:, None] & matching, amount - reduction, amount)
        remove = remove | (due_now[:, None] & matching & (amount <= 0))
    action = _compact_market(action._replace(market_amount=amount), ~remove)
    clear = due_now | expired
    return (
        action,
        jnp.where(clear, -1, due_step).astype(jnp.int16),
        jnp.where(clear[:, None], 0, due).astype(jnp.int16),
    )


def _weed_repair(
    states: State,
    bank,
    skeleton_id,
    action: Action,
    carry: PublicG02CarryV1,
    player: int,
    replay_steps: int = 8,
):
    step = states.step.astype(jnp.int32)
    batch = jnp.arange(step.shape[0])
    start = carry.repair_start
    intended_op, intended_item, intended_amount = carry.repair_op, carry.repair_item, carry.repair_amount
    unit_op, unit_item, unit_amount = action.unit_op, action.unit_item, action.unit_amount
    actual_count = action.unit_count.astype(jnp.int32)
    previous = jnp.clip(step - 1, 0, 718)
    previous_count = bank.unit_count[skeleton_id, previous].astype(jnp.int32)
    for actor in range(MAX_UNITS):
        present = actor < actual_count
        # Python deletes a pending hand transaction immediately when that hand
        # disappears at day end.  Fixed JAX action slots still exist, so mask
        # them explicitly before replaying the queued action.
        missing = (start[:, actor] >= 0) & (~present)
        start = start.at[:, actor].set(jnp.where(missing, -1, start[:, actor]))
        active = (start[:, actor] >= 0) & present
        age = step - start[:, actor].astype(jnp.int32)
        use_intended = active & (age == 1)
        use_replay = active & (age >= 2) & (age <= 1 + replay_steps)
        previous_present = actor < previous_count
        replay_op = jnp.where(previous_present, bank.unit_op[skeleton_id, previous, actor], UnitOp.PASS)
        replay_item = jnp.where(previous_present, bank.unit_item[skeleton_id, previous, actor], -1)
        replay_amount = jnp.where(previous_present, bank.unit_amount[skeleton_id, previous, actor], 1)
        unit_op = unit_op.at[:, actor].set(jnp.where(use_intended, intended_op[:, actor], jnp.where(use_replay, replay_op, unit_op[:, actor])))
        unit_item = unit_item.at[:, actor].set(jnp.where(use_intended, intended_item[:, actor], jnp.where(use_replay, replay_item, unit_item[:, actor])))
        unit_amount = unit_amount.at[:, actor].set(jnp.where(use_intended, intended_amount[:, actor], jnp.where(use_replay, replay_amount, unit_amount[:, actor])))
        finished = active & (age > 1 + replay_steps)
        start = start.at[:, actor].set(jnp.where(finished, -1, start[:, actor]))

        position = states.unit_pos[:, player, actor].astype(jnp.int32)
        tile = states.tile_kind[batch, player, jnp.clip(position[:, 1], 0, 9), jnp.clip(position[:, 0], 0, 9)]
        trigger = (
            present
            & (start[:, actor] < 0)
            & ((unit_op[:, actor] == UnitOp.BUILD_PASTURE) | (unit_op[:, actor] == UnitOp.PLANT))
            & (tile == TileKind.WEED)
        )
        start = start.at[:, actor].set(jnp.where(trigger, step, start[:, actor]).astype(jnp.int16))
        intended_op = intended_op.at[:, actor].set(jnp.where(trigger, unit_op[:, actor], intended_op[:, actor]))
        intended_item = intended_item.at[:, actor].set(jnp.where(trigger, unit_item[:, actor], intended_item[:, actor]))
        intended_amount = intended_amount.at[:, actor].set(jnp.where(trigger, unit_amount[:, actor], intended_amount[:, actor]))
        unit_op = unit_op.at[:, actor].set(jnp.where(trigger, UnitOp.DIG, unit_op[:, actor]))
        unit_item = unit_item.at[:, actor].set(jnp.where(trigger, -1, unit_item[:, actor]))
        unit_amount = unit_amount.at[:, actor].set(jnp.where(trigger, 1, unit_amount[:, actor]))
    action = action._replace(unit_op=unit_op, unit_item=unit_item, unit_amount=unit_amount)
    return action, carry._replace(
        repair_start=start,
        repair_op=intended_op,
        repair_item=intended_item,
        repair_amount=intended_amount,
    )


def _project_shed(states: State, action: Action, player: int) -> jax.Array:
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size)
    shed = states.shed[:, player].astype(jnp.int32)
    inventory = states.unit_inventory[:, player].astype(jnp.int32)
    inventory_order = states.unit_inventory_order[:, player].astype(jnp.int32)
    positions = states.unit_pos[:, player].astype(jnp.int32)
    for actor in range(MAX_UNITS):
        present = actor < action.unit_count.astype(jnp.int32)
        position = positions[:, actor]
        at_shed = jnp.any(jnp.all(position[:, None, :] == _SHED_ACCESS[None, :, :], axis=2), axis=1)
        op = action.unit_op[:, actor]
        item = action.unit_item[:, actor].astype(jnp.int32)
        requested = jnp.maximum(action.unit_amount[:, actor], 0).astype(jnp.int32)
        safe_item = jnp.clip(item, 0, NUM_SHED_ITEMS - 1)

        pickup = present & at_shed & (op == UnitOp.PICKUP) & (item >= 0) & (item < NUM_SHED_ITEMS)
        take = jnp.where(pickup, jnp.minimum(requested, shed[batch, safe_item]), 0)
        shed = shed.at[batch, safe_item].add(-take)
        actor_inventory = inventory[:, actor].at[batch, safe_item].add(take)

        drop = present & at_shed & (op == UnitOp.DROP)
        order_key = jnp.where(inventory_order[:, actor] >= 0, inventory_order[:, actor], 32767)
        item_order = jnp.argsort(order_key, axis=1, stable=True)
        for ordinal in range(NUM_SHED_ITEMS):
            product = item_order[:, ordinal].astype(jnp.int32)
            available = jnp.take_along_axis(actor_inventory, product[:, None], axis=1)[:, 0]
            room = jnp.maximum(SHED_CAPACITY - jnp.sum(shed, axis=1), 0)
            moved = jnp.where(drop, jnp.minimum(jnp.maximum(available, 0), room), 0)
            shed = shed.at[batch, product].add(moved)
            actor_inventory = actor_inventory.at[batch, product].add(-moved)

        x = jnp.clip(position[:, 0], 0, 9)
        y = jnp.clip(position[:, 1], 0, 9)
        animal_id = item - NUM_PRODUCTS
        safe_animal = jnp.clip(animal_id, 0, NUM_ANIMALS - 1)
        matching_animal_place = (
            (animal_id >= 0)
            & (animal_id < NUM_ANIMALS)
            & (states.tile_kind[batch, player, y, x] == _ANIMAL_STRUCTURE[safe_animal])
            & (states.tile_animal[batch, player, y, x] < 0)
        )
        place = (
            present
            & at_shed
            & (op == UnitOp.PLACE)
            & (~matching_animal_place)
            & (item >= 0)
            & (item < NUM_SHED_ITEMS)
        )
        available = actor_inventory[batch, safe_item]
        room = jnp.maximum(SHED_CAPACITY - jnp.sum(shed, axis=1), 0)
        moved = jnp.where(place, jnp.minimum(jnp.minimum(requested, jnp.maximum(available, 0)), room), 0)
        shed = shed.at[batch, safe_item].add(moved)
    return shed


def _town_demand(states: State, product_id: int) -> jax.Array:
    step = states.step.astype(jnp.int32)
    day_tick = (step % 24 == 0).astype(jnp.int32)
    slot = jnp.arange(states.town_shops.shape[1])[None, :]
    active = slot < states.town_count[:, None]
    shop = jnp.clip(states.town_shops.astype(jnp.int32), 0, len(SHOP_NAMES) - 1)
    demand = jnp.sum(jnp.where(active, _SHOP_DEMAND[shop, product_id], 0), axis=1)
    return day_tick + jnp.where(step % 4 == 0, demand, 0)


def _original_front_run(states: State, bank, skeleton_id, action: Action, player: int, due_step, due):
    step = states.step.astype(jnp.int32)
    future = jnp.clip(step + 1, 0, 718)
    future_active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < bank.market_count[skeleton_id, future][:, None]
    current_active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    moved = jnp.zeros_like(due, dtype=jnp.int32)
    for product_id in _PREMIUM_IDS.tolist():
        target = jnp.sum(
            jnp.where(
                future_active & (bank.market_op[skeleton_id, future] == MarketOp.SELL) & (bank.market_item[skeleton_id, future] == product_id),
                jnp.maximum(bank.market_amount[skeleton_id, future], 0),
                0,
            ),
            axis=1,
        )
        pickup = jnp.sum(
            jnp.where(
                (jnp.arange(MAX_UNITS)[None, :] < action.unit_count[:, None])
                & (action.unit_op == UnitOp.PICKUP)
                & (action.unit_item == product_id),
                jnp.maximum(action.unit_amount, 0),
                0,
            ),
            axis=1,
        )
        already = jnp.sum(
            jnp.where(current_active & (action.market_op == MarketOp.SELL) & (action.market_item == product_id), jnp.maximum(action.market_amount, 0), 0),
            axis=1,
        )
        quantity = jnp.minimum(target, jnp.maximum(states.shed[:, player, product_id].astype(jnp.int32) - pickup - already, 0))
        eligible = (target > 0) & (_town_demand(states, product_id) == 0) & (quantity > 0)
        locations = jnp.where(current_active & (action.market_op == MarketOp.SELL) & (action.market_item == product_id), jnp.arange(MAX_MARKET_ORDERS)[None, :], -1)
        location = jnp.max(locations, axis=1)
        merge = eligible & (location >= 0)
        safe_location = jnp.maximum(location, 0)
        batch = jnp.arange(step.shape[0])
        amount = action.market_amount.at[batch, safe_location].set(
            jnp.where(merge, action.market_amount[batch, safe_location] + quantity, action.market_amount[batch, safe_location])
        )
        action = action._replace(market_amount=amount)
        append = eligible & (location < 0) & (action.market_count < MAX_MARKET_ORDERS)
        action = _append_sell(action, append, jnp.full_like(step, product_id, dtype=jnp.int8), quantity)
        moved = moved.at[:, product_id].set(jnp.where(merge | append, quantity, moved[:, product_id]))
        current_active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    changed = jnp.sum(moved, axis=1) > 0
    return (
        action,
        jnp.where(changed, step + 1, due_step).astype(jnp.int16),
        jnp.where(changed[:, None], moved, due).astype(jnp.int16),
    )


def _v17_front_run(states: State, bank, skeleton_id, rival_schedule, action: Action, player: int, confidence):
    step = states.step.astype(jnp.int32)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    projected = _project_shed(states, action, player)
    planned = jnp.zeros((step.shape[0], len(_PREMIUM_NAMES)), dtype=jnp.int32)
    first_distance = jnp.full_like(planned, 99)
    for distance in range(1, 5):
        future = jnp.clip(step + distance, 0, 718)
        in_range = step + distance < 719
        future_active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < bank.market_count[skeleton_id, future][:, None]
        for ordinal, product_id in enumerate(_PREMIUM_IDS.tolist()):
            trace_rows = jnp.where(
                future_active & (bank.market_op[skeleton_id, future] == MarketOp.SELL) & (bank.market_item[skeleton_id, future] == product_id),
                jnp.maximum(bank.market_amount[skeleton_id, future], 0),
                0,
            )
            trace_target = jnp.max(trace_rows, axis=1)
            schedule_target = rival_schedule[future, ordinal].astype(jnp.int32)
            target = jnp.where(in_range, jnp.maximum(trace_target, schedule_target), 0)
            first_distance = first_distance.at[:, ordinal].set(jnp.where((planned[:, ordinal] == 0) & (target > 0), distance, first_distance[:, ordinal]))
            planned = planned.at[:, ordinal].add(target)

    best_priority = jnp.full(step.shape, -jnp.inf, dtype=jnp.float32)
    best_item = jnp.full(step.shape, -1, dtype=jnp.int8)
    best_quantity = jnp.zeros(step.shape, dtype=jnp.int32)
    for ordinal, product_id in enumerate(_PREMIUM_IDS.tolist()):
        already = jnp.sum(
            jnp.where(active & (action.market_op == MarketOp.SELL) & (action.market_item == product_id), jnp.maximum(action.market_amount, 0), 0),
            axis=1,
        )
        quantity = jnp.minimum(planned[:, ordinal], jnp.maximum(projected[:, product_id] - already, 0))
        priority = (
            states.market_price[:, product_id].astype(jnp.float32) * quantity.astype(jnp.float32) * _PREMIUM_WEIGHTS[ordinal]
            + (5 - first_distance[:, ordinal]).astype(jnp.float32) * _PREMIUM_BASE[ordinal]
        )
        valid = (planned[:, ordinal] > 0) & (quantity > 0)
        # Python max((priority, item, quantity)) breaks equal priorities by the
        # lexicographically larger item.  The loop follows lexical item order,
        # so >= reproduces that tie break.
        take = valid & (priority >= best_priority)
        best_priority = jnp.where(take, priority, best_priority)
        best_item = jnp.where(take, product_id, best_item).astype(jnp.int8)
        best_quantity = jnp.where(take, quantity, best_quantity)
    append = (confidence >= 1) & (best_item >= 0) & (action.market_count < MAX_MARKET_ORDERS)
    return _append_sell(action, append, best_item, best_quantity)


def public_g02_player_action_v1(
    states: State,
    bank,
    skeleton_id: jax.Array,
    rival_schedule: jax.Array,
    carry: PublicG02CarryV1,
    player: int,
) -> tuple[Action, PublicG02CarryV1]:
    """Emit the exact V17 action for one player seat."""

    action = _raw_action(states, bank, skeleton_id, player)
    action, carry = _weed_repair(states, bank, skeleton_id, action, carry, player)
    step = states.step.astype(jnp.int32)
    action, due_step, due = _repay(action, step, carry.due_step, carry.due)
    confidence = _update_clone(states, carry.clone_confidence)
    original, original_due_step, original_due = _original_front_run(
        states, bank, skeleton_id, action, player, due_step, due
    )
    adaptive = _v17_front_run(states, bank, skeleton_id, rival_schedule, action, player, confidence)
    use_adaptive = confidence >= 1
    action = Action(
        *(
            jnp.where(
                use_adaptive.reshape((use_adaptive.shape[0],) + (1,) * (left.ndim - 1)),
                left,
                right,
            )
            for left, right in zip(adaptive, original, strict=True)
        )
    )
    return action, carry._replace(
        clone_confidence=confidence,
        due_step=jnp.where(use_adaptive, due_step, original_due_step).astype(jnp.int16),
        due=jnp.where(use_adaptive[:, None], due, original_due).astype(jnp.int16),
    )


def public_rc5_weed_player_action_v1(
    states: State,
    bank,
    skeleton_id: jax.Array,
    carry: PublicG02CarryV1,
    player: int,
) -> tuple[Action, PublicG02CarryV1]:
    """Frozen action tape plus the reusable RC5 weed-repair overlay.

    Public G04 embeds a modal fixed route and wraps only this RC5 controller;
    its RC5 market front-run functions are present in the embedded source but
    are never called by the final agent.
    """

    return _weed_repair(
        states,
        bank,
        skeleton_id,
        _raw_action(states, bank, skeleton_id, player),
        carry,
        player,
    )
