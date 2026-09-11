"""Coherent Hasegawa trace policy with atomic, state-conditioned recovery."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_STRUCTURE,
    BOARD_SIZE,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_WATERED,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    SHED_ACCESS,
    SHED_CAPACITY,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.types import Action, Events, State, StaticTables

from .trace_bank import HasegawaTraceBankV2


_ANIMAL_STRUCTURE = jnp.asarray(ANIMAL_STRUCTURE, dtype=jnp.int8)
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int8)
_MOVE_OP = jnp.asarray((UnitOp.NORTH, UnitOp.SOUTH, UnitOp.EAST, UnitOp.WEST), dtype=jnp.int8)
_LIQUIDATION_ORDER = jnp.asarray((1, 5, 8, 4, 6, 3, 2, 0, 7), dtype=jnp.int8)


class HasegawaCarryV2(NamedTuple):
    branch_id: jax.Array
    branch_locked: jax.Array
    branch_lock_step: jax.Array
    weed_active: jax.Array
    weed_start: jax.Array
    weed_intended_op: jax.Array
    weed_intended_item: jax.Array
    weed_intended_amount: jax.Array
    invalid_intent_total: jax.Array
    resync_total: jax.Array
    market_trim_total: jax.Array
    hard_counter_total: jax.Array


class HasegawaDiagnosticsV2(NamedTuple):
    branch_id: jax.Array
    branch_locked: jax.Array
    invalid_unit_intent_count: jax.Array
    resync_unit_count: jax.Array
    market_trim_count: jax.Array
    hard_counter_delta: jax.Array
    money_deviation_from_source: jax.Array


def initialize_hasegawa_carry_v2(batch_size: int) -> HasegawaCarryV2:
    units = (batch_size, MAX_UNITS)
    zeros = jnp.zeros((batch_size,), dtype=jnp.int32)
    return HasegawaCarryV2(
        branch_id=jnp.zeros((batch_size,), dtype=jnp.int8),
        branch_locked=jnp.zeros((batch_size,), dtype=jnp.bool_),
        branch_lock_step=jnp.full((batch_size,), -1, dtype=jnp.int16),
        weed_active=jnp.zeros(units, dtype=jnp.bool_),
        weed_start=jnp.full(units, -1, dtype=jnp.int16),
        weed_intended_op=jnp.full(units, UnitOp.PASS, dtype=jnp.int8),
        weed_intended_item=jnp.full(units, -1, dtype=jnp.int8),
        weed_intended_amount=jnp.ones(units, dtype=jnp.int32),
        invalid_intent_total=zeros,
        resync_total=zeros,
        market_trim_total=zeros,
        hard_counter_total=zeros,
    )


def select_hasegawa_branch_v2(states: State, carry: HasegawaCarryV2):
    """Lock exactly once using the first currently visible town shop."""

    first_shop = jnp.clip(states.town_shops[:, 0].astype(jnp.int16), 0, 7)
    can_lock = (~carry.branch_locked) & (states.town_count > 0)
    branch = jnp.where(can_lock, first_shop + 1, carry.branch_id).astype(jnp.int8)
    locked = carry.branch_locked | can_lock
    lock_step = jnp.where(can_lock, states.step, carry.branch_lock_step).astype(jnp.int16)
    return branch, locked, lock_step


def _tile_under_units(states: State, player: int):
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)[:, None]
    pos = states.unit_pos[:, player].astype(jnp.int32)
    x = jnp.clip(pos[..., 0], 0, BOARD_SIZE - 1)
    y = jnp.clip(pos[..., 1], 0, BOARD_SIZE - 1)
    return (
        states.tile_kind[batch, player, y, x],
        states.tile_crop[batch, player, y, x],
        states.tile_animal[batch, player, y, x],
        states.tile_yield[batch, player, y, x],
        states.tile_flags[batch, player, y, x],
    )


def _move_toward(position: jax.Array, target: jax.Array):
    x, y = position[..., 0], position[..., 1]
    tx, ty = target[..., 0], target[..., 1]
    return jnp.where(
        x < tx,
        UnitOp.EAST,
        jnp.where(x > tx, UnitOp.WEST, jnp.where(y < ty, UnitOp.SOUTH, jnp.where(y > ty, UnitOp.NORTH, UnitOp.PASS))),
    ).astype(jnp.int8)


def _apply_weed_repair(
    states: State,
    bank: HasegawaTraceBankV2,
    branch: jax.Array,
    carry: HasegawaCarryV2,
    op: jax.Array,
    item: jax.Array,
    amount: jax.Array,
    present: jax.Array,
    player: int,
):
    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    previous = jnp.clip(step - 1, 0, 718)
    age = step[:, None] - carry.weed_start.astype(jnp.int32)
    existing = carry.weed_active & present & (age <= 9)
    intended = existing & (age == 1)
    replay_lag = existing & (age >= 2) & (age <= 9)
    op = jnp.where(intended, carry.weed_intended_op, op)
    item = jnp.where(intended, carry.weed_intended_item, item)
    amount = jnp.where(intended, carry.weed_intended_amount, amount)
    op = jnp.where(replay_lag, bank.unit_op[branch, previous], op)
    item = jnp.where(replay_lag, bank.unit_item[branch, previous], item)
    amount = jnp.where(replay_lag, bank.unit_amount[branch, previous], amount)
    kind, _, _, _, _ = _tile_under_units(states, player)
    trigger = (
        (~existing)
        & present
        & ((op == UnitOp.BUILD_PASTURE) | (op == UnitOp.BUILD_COOP) | (op == UnitOp.PLANT))
        & (kind == TileKind.WEED)
    )
    intended_op, intended_item, intended_amount = op, item, amount
    op = jnp.where(trigger, UnitOp.DIG, op).astype(jnp.int8)
    item = jnp.where(trigger, -1, item).astype(jnp.int8)
    amount = jnp.where(trigger, 1, amount).astype(jnp.int32)
    carry = carry._replace(
        weed_active=existing | trigger,
        weed_start=jnp.where(trigger, step[:, None], carry.weed_start).astype(jnp.int16),
        weed_intended_op=jnp.where(trigger, intended_op, carry.weed_intended_op).astype(jnp.int8),
        weed_intended_item=jnp.where(trigger, intended_item, carry.weed_intended_item).astype(jnp.int8),
        weed_intended_amount=jnp.where(trigger, intended_amount, carry.weed_intended_amount).astype(jnp.int32),
    )
    return op, item, amount, carry


def _guard_and_recover_units(
    states: State,
    bank: HasegawaTraceBankV2,
    branch: jax.Array,
    op: jax.Array,
    item: jax.Array,
    amount: jax.Array,
    present: jax.Array,
    player: int,
):
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)[:, None]
    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    pos = states.unit_pos[:, player].astype(jnp.int8)
    kind, crop, animal, tile_yield, flags = _tile_under_units(states, player)
    inventory = states.unit_inventory[:, player].astype(jnp.int32)
    total_inventory = jnp.sum(inventory, axis=-1)
    safe_item = jnp.clip(item.astype(jnp.int32), 0, NUM_SHED_ITEMS - 1)
    carried_item = jnp.take_along_axis(inventory, safe_item[..., None], axis=-1)[..., 0]
    at_shed = jnp.any(jnp.all(pos[:, :, None, :] == _SHED_ACCESS[None, None, :, :], axis=-1), axis=-1)
    expected = bank.expected_unit_pos[branch, step]
    expected_active = bank.expected_unit_active[branch, step]
    mismatch = present & expected_active & jnp.any(pos != expected, axis=-1)
    movement = jnp.any(op[..., None] == _MOVE_OP[None, None, :], axis=-1)
    movement_valid = (
        ((op == UnitOp.NORTH) & (pos[..., 1] > 0))
        | ((op == UnitOp.SOUTH) & (pos[..., 1] < BOARD_SIZE - 1))
        | ((op == UnitOp.WEST) & (pos[..., 0] > 0))
        | ((op == UnitOp.EAST) & (pos[..., 0] < BOARD_SIZE - 1))
    )
    valid_item = (item >= 0) & (item < NUM_SHED_ITEMS)
    valid_crop = (item >= 0) & (item < NUM_CROPS)
    animal_id = item - NUM_PRODUCTS
    safe_animal = jnp.clip(animal_id.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    structure_ok = kind == _ANIMAL_STRUCTURE[safe_animal]
    place_animal = (
        valid_item
        & (animal_id >= 0)
        & (animal_id < NUM_ANIMALS)
        & structure_ok
        & (animal < 0)
        & (carried_item > 0)
    )
    # Official PLACE has a second meaning at a shed access tile: move a chosen
    # carried item into the shed.  Hasegawa uses this for same-turn
    # PLACE(product) -> SELL financing, so it must not be treated as an animal-
    # only operation.
    place_shed = at_shed & valid_item & (amount > 0) & (carried_item > 0)
    valid = (
        (op == UnitOp.PASS)
        | (movement & movement_valid)
        | ((op == UnitOp.DROP) & at_shed & (total_inventory > 0))
        | ((op == UnitOp.PICKUP) & at_shed & valid_item & (states.shed[:, player][batch, safe_item] > 0))
        | ((op == UnitOp.PLACE) & (place_animal | place_shed))
        | ((op == UnitOp.PLANT) & valid_crop & (kind == TileKind.EMPTY) & (states.seeds[:, player][batch, jnp.clip(item, 0, NUM_CROPS - 1)] > 0))
        | ((op == UnitOp.WATER) & (kind == TileKind.PLANT) & ((flags & FLAG_WATERED) == 0))
        | ((op == UnitOp.HARVEST) & ((kind == TileKind.PLANT) | (animal >= 0)) & (tile_yield > 0))
        | ((op == UnitOp.FERTILIZE) & (kind == TileKind.PLANT) & (inventory[..., NUM_PRODUCTS - 1] > 0))
        | ((op == UnitOp.DIG) & ((kind == TileKind.WEED) | (kind == TileKind.PLANT) | (kind == TileKind.COOP) | (kind == TileKind.PASTURE)))
        | ((op == UnitOp.BUILD_COOP) & (kind == TileKind.EMPTY))
        | ((op == UnitOp.BUILD_PASTURE) & (kind == TileKind.EMPTY))
        | ((op == UnitOp.FEED) & (animal >= 0) & ((flags & FLAG_FED) == 0) & (inventory[..., 0] > 0))
        | ((op == UnitOp.COLLECT_FERTILIZER) & (animal >= 0) & ((flags & FLAG_FERTILIZER_AVAILABLE) != 0))
        | ((op == UnitOp.CARE) & (animal >= 0) & ((flags & FLAG_CARED) == 0))
    )
    valid = valid & present
    resync = mismatch & ((op == UnitOp.PASS) | movement | (~valid))
    invalid = present & (~valid) & (~resync)
    move = _move_toward(pos, expected)
    local_op = jnp.where(
        (kind == TileKind.PLANT) & ((flags & FLAG_WATERED) == 0),
        UnitOp.WATER,
        jnp.where(
            (animal >= 0) & ((flags & FLAG_FED) == 0) & (inventory[..., 0] > 0),
            UnitOp.FEED,
            jnp.where(
                ((kind == TileKind.PLANT) | (animal >= 0)) & (tile_yield > 0),
                UnitOp.HARVEST,
                jnp.where(
                    (animal >= 0) & ((flags & FLAG_FERTILIZER_AVAILABLE) != 0),
                    UnitOp.COLLECT_FERTILIZER,
                    jnp.where(
                        (animal >= 0) & ((flags & FLAG_CARED) == 0),
                        UnitOp.CARE,
                        jnp.where(at_shed & (total_inventory > 0), UnitOp.DROP, UnitOp.PASS),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int8)
    op = jnp.where(resync, move, jnp.where(invalid, local_op, op)).astype(jnp.int8)
    item = jnp.where(resync | invalid, -1, item).astype(jnp.int8)
    amount = jnp.where(resync | invalid, 1, amount).astype(jnp.int32)
    return op, item, amount, jnp.sum(invalid, axis=1, dtype=jnp.int32), jnp.sum(resync, axis=1, dtype=jnp.int32)


def _compact_market(op, item, amount, keep):
    slot = jnp.arange(MAX_MARKET_ORDERS, dtype=jnp.int32)[None, :]
    order = jnp.argsort(jnp.where(keep, slot, slot + MAX_MARKET_ORDERS), axis=1, stable=True)
    count = jnp.sum(keep, axis=1, dtype=jnp.int8)
    active = slot < count[:, None]
    op = jnp.take_along_axis(op, order, axis=1)
    item = jnp.take_along_axis(item, order, axis=1)
    amount = jnp.take_along_axis(amount, order, axis=1)
    return (
        jnp.where(active, op, MarketOp.NONE).astype(jnp.int8),
        jnp.where(active, item, -1).astype(jnp.int8),
        jnp.where(active, amount, 0).astype(jnp.int32),
        count,
    )


def _sanitize_market(states: State, action: Action, player: int):
    batch_size = states.step.shape[0]
    slots = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    active = slots < action.market_count[:, None]
    op, item = action.market_op, action.market_item
    amount = jnp.maximum(action.market_amount.astype(jnp.int32), 0)
    shed = states.shed[:, player].astype(jnp.int32)
    # Unit actions settle before market orders.  In particular Hasegawa often
    # DROPs carried fertilizer/product and sells it in the same turn.  Guarding
    # against the pre-unit shed incorrectly deleted those financing sales.
    unit_present = jnp.arange(MAX_UNITS)[None, :] < action.unit_count[:, None]
    dropping = unit_present & (action.unit_op == UnitOp.DROP)
    shed = shed + jnp.sum(
        jnp.where(
            dropping[..., None],
            states.unit_inventory[:, player].astype(jnp.int32),
            0,
        ),
        axis=1,
        dtype=jnp.int32,
    )
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    at_shed = jnp.any(
        jnp.all(
            states.unit_pos[:, player, :, None, :] == _SHED_ACCESS[None, None, :, :],
            axis=-1,
        ),
        axis=-1,
    )
    for unit in range(MAX_UNITS):
        place = unit_present[:, unit] & (action.unit_op[:, unit] == UnitOp.PLACE) & at_shed[:, unit]
        safe_item = jnp.clip(action.unit_item[:, unit].astype(jnp.int32), 0, NUM_SHED_ITEMS - 1)
        available = states.unit_inventory[batch, player, unit, safe_item].astype(jnp.int32)
        quantity = jnp.minimum(jnp.maximum(action.unit_amount[:, unit], 0), available)
        shed = shed.at[batch, safe_item].add(jnp.where(place, quantity, 0))
    keep = jnp.zeros_like(active)
    for slot in range(MAX_MARKET_ORDERS):
        current_op = op[:, slot]
        current_item = item[:, slot].astype(jnp.int32)
        current_amount = amount[:, slot]
        safe_product = jnp.clip(current_item, 0, NUM_PRODUCTS - 1)
        sell = active[:, slot] & (current_op == MarketOp.SELL) & (current_item >= 0) & (current_item < NUM_PRODUCTS)
        sell_quantity = jnp.minimum(current_amount, shed[jnp.arange(batch_size), safe_product])
        buy_product = active[:, slot] & (current_op == MarketOp.BUY_PRODUCT) & (current_item >= 0) & (current_item < NUM_PRODUCTS) & (current_amount > 0)
        buy_animal = active[:, slot] & (current_op == MarketOp.BUY_ANIMAL) & (current_item >= NUM_PRODUCTS) & (current_item < NUM_SHED_ITEMS) & (current_amount > 0)
        seed = active[:, slot] & (current_op == MarketOp.BUY_SEED) & (current_item >= 0) & (current_item < NUM_CROPS) & (current_amount > 0)
        exact = active[:, slot] & ((current_op == MarketOp.HIRE) | (current_op == MarketOp.BUY_LAND))
        quantity = jnp.where(sell, sell_quantity, current_amount)
        valid = exact | seed | buy_product | buy_animal | (sell & (sell_quantity > 0))
        amount = amount.at[:, slot].set(quantity)
        keep = keep.at[:, slot].set(valid)
        shed = shed.at[jnp.arange(batch_size), safe_product].add(jnp.where(sell, -sell_quantity, 0))
    trimmed = jnp.sum(active & (~keep), axis=1, dtype=jnp.int32)
    op, item, amount, count = _compact_market(op, item, amount, keep)
    return action._replace(market_op=op, market_item=item, market_amount=amount, market_count=count), trimmed


def _terminal_liquidation(states: State, action: Action, player: int):
    step = states.step.astype(jnp.int32)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    safe = jnp.clip(action.market_item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    planned = jnp.zeros((states.step.shape[0], NUM_PRODUCTS), dtype=jnp.int32)
    planned = planned.at[jnp.arange(states.step.shape[0])[:, None], safe].add(
        jnp.where(active & (action.market_op == MarketOp.SELL), jnp.maximum(action.market_amount, 0), 0)
    )
    shed = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int32)
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)
    for product in _LIQUIDATION_ORDER.tolist():
        quantity = jnp.where(step >= 718, shed[:, product], jnp.maximum(shed[:, product] - planned[:, product], 0))
        append = (step >= 716) & (quantity > 0) & (action.market_count < MAX_MARKET_ORDERS)
        slot = jnp.clip(action.market_count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1)
        action = action._replace(
            market_op=action.market_op.at[batch, slot].set(jnp.where(append, MarketOp.SELL, action.market_op[batch, slot])),
            market_item=action.market_item.at[batch, slot].set(jnp.where(append, product, action.market_item[batch, slot])),
            market_amount=action.market_amount.at[batch, slot].set(jnp.where(append, quantity, action.market_amount[batch, slot])),
            market_count=(action.market_count + append.astype(jnp.int8)).astype(jnp.int8),
        )
    return action


def _pair(controlled: Action, external: Action, player: int):
    if player == 0:
        return Action(*(jnp.stack((left, right), axis=1) for left, right in zip(controlled, external, strict=True)))
    return Action(*(jnp.stack((left, right), axis=1) for left, right in zip(external, controlled, strict=True)))


def hasegawa_step_with_external_v2(
    states: State,
    carry: HasegawaCarryV2,
    bank: HasegawaTraceBankV2,
    external_action: Action,
    player: int,
    events: Events,
    tables: StaticTables,
):
    branch, locked, lock_step = select_hasegawa_branch_v2(states, carry)
    carry = carry._replace(branch_id=branch, branch_locked=locked, branch_lock_step=lock_step)
    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    count = jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int8)
    present = jnp.arange(MAX_UNITS)[None, :] < count[:, None]
    op = jnp.where(present, bank.unit_op[branch, step], UnitOp.PASS).astype(jnp.int8)
    item = jnp.where(present, bank.unit_item[branch, step], -1).astype(jnp.int8)
    amount = jnp.where(present, bank.unit_amount[branch, step], 1).astype(jnp.int32)
    op, item, amount, carry = _apply_weed_repair(states, bank, branch, carry, op, item, amount, present, player)
    op, item, amount, invalid, resync = _guard_and_recover_units(states, bank, branch, op, item, amount, present, player)
    action = Action(
        unit_op=op, unit_item=item, unit_amount=amount, unit_count=count,
        market_op=bank.market_op[branch, step], market_item=bank.market_item[branch, step],
        market_amount=bank.market_amount[branch, step], market_count=bank.market_count[branch, step],
    )
    action, trimmed = _sanitize_market(states, action, player)
    action = _terminal_liquidation(states, action, player)
    joint = _pair(action, external_action, player)
    next_state = batched_step_sync(states, joint, events, tables)
    hard_before = (
        states.hand_cap_hits[:, player]
        + states.market_loop_cap_hits
        + states.price_lut_oob
    )
    hard_after = (
        next_state.hand_cap_hits[:, player]
        + next_state.market_loop_cap_hits
        + next_state.price_lut_oob
    )
    hard_delta = jnp.maximum(hard_after - hard_before, 0).astype(jnp.int32)
    expected_money = bank.expected_money[branch, step].astype(jnp.int32)
    carry = carry._replace(
        invalid_intent_total=carry.invalid_intent_total + invalid,
        resync_total=carry.resync_total + resync,
        market_trim_total=carry.market_trim_total + trimmed,
        hard_counter_total=carry.hard_counter_total + hard_delta,
    )
    diagnostics = HasegawaDiagnosticsV2(
        branch_id=branch,
        branch_locked=locked,
        invalid_unit_intent_count=invalid,
        resync_unit_count=resync,
        market_trim_count=trimmed,
        hard_counter_delta=hard_delta,
        money_deviation_from_source=states.money[:, player].astype(jnp.int32) - expected_money,
    )
    return next_state, carry, diagnostics, joint


__all__ = [
    "HasegawaCarryV2",
    "HasegawaDiagnosticsV2",
    "hasegawa_step_with_external_v2",
    "initialize_hasegawa_carry_v2",
    "select_hasegawa_branch_v2",
]
