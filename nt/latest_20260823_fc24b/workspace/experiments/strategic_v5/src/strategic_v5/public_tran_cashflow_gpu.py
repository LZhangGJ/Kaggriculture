"""Exact GPU controller for public G16 Tran Cashflow."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_PRODUCTS,
    PRODUCTS,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.types import Action, State


_WHEAT = PRODUCTS.index("WHEAT")
_STRAWBERRY = PRODUCTS.index("STRAWBERRY")
_MELON = PRODUCTS.index("MELON")
_MILK = PRODUCTS.index("MILK")
_FERTILIZER = PRODUCTS.index("FERTILIZER")

# Python compares the product name as the final tuple field.
_LEXICAL_ITEM_RANK = jnp.asarray(
    [
        sorted(PRODUCTS).index(product)
        for product in PRODUCTS
    ],
    dtype=jnp.int16,
)
_TERMINAL_UNIT_RANK = jnp.asarray(
    [1, 0, 0, 2, 0, 0, 4, 0, 3], dtype=jnp.int16
)
_TERMINAL_MARKET_RANK = jnp.asarray(
    [1, 0, 0, 3, 0, 0, 4, 0, 2], dtype=jnp.int16
)
_CASHFLOW_SELL_RANK = jnp.asarray(
    [6, 4, 16, 3, 1, 7, 2, 0, 5], dtype=jnp.int16
)


class TranCashflowCarryV1(NamedTuple):
    pending_pasture_actor: jax.Array
    pending_pasture_pos: jax.Array
    pending_pasture_step: jax.Array
    farmer_shift_end: jax.Array
    pending_plant_actor: jax.Array
    pending_plant_pos: jax.Array
    pending_plant_step: jax.Array
    pending_water_planter: jax.Array
    pending_water_pos: jax.Array
    pending_water_step: jax.Array
    water_shift_actor: jax.Array
    water_shift_end: jax.Array
    soil_repair_activated: jax.Array
    cashflow_active: jax.Array


def initialize_tran_cashflow_carry_v1(batch_size: int) -> TranCashflowCarryV1:
    pos = (batch_size, 2)
    return TranCashflowCarryV1(
        pending_pasture_actor=jnp.full((batch_size,), -1, dtype=jnp.int8),
        pending_pasture_pos=jnp.full(pos, -1, dtype=jnp.int8),
        pending_pasture_step=jnp.full((batch_size,), -1, dtype=jnp.int16),
        farmer_shift_end=jnp.full((batch_size,), -1, dtype=jnp.int16),
        pending_plant_actor=jnp.full((batch_size,), -1, dtype=jnp.int8),
        pending_plant_pos=jnp.full(pos, -1, dtype=jnp.int8),
        pending_plant_step=jnp.full((batch_size,), -1, dtype=jnp.int16),
        pending_water_planter=jnp.full((batch_size,), -1, dtype=jnp.int8),
        pending_water_pos=jnp.full(pos, -1, dtype=jnp.int8),
        pending_water_step=jnp.full((batch_size,), -1, dtype=jnp.int16),
        water_shift_actor=jnp.full((batch_size,), -1, dtype=jnp.int8),
        water_shift_end=jnp.full((batch_size,), -1, dtype=jnp.int16),
        soil_repair_activated=jnp.zeros((batch_size,), dtype=jnp.bool_),
        cashflow_active=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


def _trace_action(states: State, bank, skeleton_id: jax.Array) -> Action:
    """Return the submitted list literally, including surplus hand actions."""

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


def _replace_unit(
    action: Action,
    actor: jax.Array,
    enabled: jax.Array,
    op: jax.Array | int,
    item: jax.Array | int = -1,
    amount: jax.Array | int = 1,
) -> Action:
    batch = jnp.arange(action.unit_count.shape[0])
    safe = jnp.clip(actor.astype(jnp.int32), 0, MAX_UNITS - 1)
    op = jnp.asarray(op, dtype=action.unit_op.dtype)
    item = jnp.asarray(item, dtype=action.unit_item.dtype)
    amount = jnp.asarray(amount, dtype=action.unit_amount.dtype)
    old_op = action.unit_op[batch, safe]
    old_item = action.unit_item[batch, safe]
    old_amount = action.unit_amount[batch, safe]
    return action._replace(
        unit_op=action.unit_op.at[batch, safe].set(jnp.where(enabled, op, old_op)),
        unit_item=action.unit_item.at[batch, safe].set(
            jnp.where(enabled, item, old_item)
        ),
        unit_amount=action.unit_amount.at[batch, safe].set(
            jnp.where(enabled, amount, old_amount)
        ),
    )


def _tile_kind_at(states: State, player: int, position: jax.Array) -> jax.Array:
    batch = jnp.arange(states.step.shape[0])
    x = jnp.clip(position[:, 0].astype(jnp.int32), 0, 9)
    y = jnp.clip(position[:, 1].astype(jnp.int32), 0, 9)
    return states.tile_kind[batch, player, y, x]


def _terminal_choice(
    quantity: jax.Array,
    prices: jax.Array,
    activated: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Vectorized Python ``max((rank, value, price, qty, item))``."""

    valid = quantity > 0
    rank = jnp.where(
        activated[:, None, None], _TERMINAL_UNIT_RANK[None, None, :], 0
    )
    rank = jnp.broadcast_to(rank, quantity.shape)
    price = jnp.broadcast_to(prices[:, None, :], quantity.shape)
    value = price.astype(jnp.int32) * quantity.astype(jnp.int32)
    lexical = jnp.broadcast_to(_LEXICAL_ITEM_RANK[None, None, :], quantity.shape)

    best_rank = jnp.max(jnp.where(valid, rank, -1), axis=2, keepdims=True)
    candidate = valid & (rank == best_rank)
    best_value = jnp.max(jnp.where(candidate, value, -1), axis=2, keepdims=True)
    candidate &= value == best_value
    best_price = jnp.max(jnp.where(candidate, price, -1), axis=2, keepdims=True)
    candidate &= price == best_price
    best_quantity = jnp.max(
        jnp.where(candidate, quantity, -1), axis=2, keepdims=True
    )
    candidate &= quantity == best_quantity
    best_lexical = jnp.max(
        jnp.where(candidate, lexical, -1), axis=2, keepdims=True
    )
    candidate &= lexical == best_lexical
    item = jnp.argmax(candidate.astype(jnp.int8), axis=2).astype(jnp.int8)
    any_item = jnp.any(valid, axis=2)
    selected_quantity = jnp.take_along_axis(
        quantity, item.astype(jnp.int32)[..., None], axis=2
    )[..., 0]
    return jnp.where(any_item, item, -1).astype(jnp.int8), jnp.where(
        any_item, selected_quantity, 0
    ).astype(jnp.int32)


def _terminal_action(
    states: State,
    action: Action,
    activated: jax.Array,
    player: int,
) -> Action:
    use = states.step >= 716
    actual_count = jnp.sum(
        states.unit_active[:, player].astype(jnp.int32), axis=1
    ).astype(jnp.int8)
    inventory = states.unit_inventory[:, player, :, :NUM_PRODUCTS].astype(jnp.int32)
    prices = states.market_price.astype(jnp.int32)
    item, quantity = _terminal_choice(inventory, prices, activated)
    active_unit = jnp.arange(MAX_UNITS)[None, :] < actual_count[:, None]
    place = active_unit & (item >= 0)
    unit_op = jnp.where(place, UnitOp.PLACE, UnitOp.PASS).astype(jnp.int8)
    unit_item = jnp.where(place, item, -1).astype(jnp.int8)
    unit_amount = jnp.where(place, quantity, 1).astype(jnp.int32)

    placed = jnp.sum(
        jax.nn.one_hot(
            jnp.clip(item.astype(jnp.int32), 0, NUM_PRODUCTS - 1),
            NUM_PRODUCTS,
            dtype=jnp.int32,
        )
        * jnp.where(place, quantity, 0)[..., None],
        axis=1,
    )
    totals = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int32) + placed
    available = totals > 0
    market_op = jnp.full_like(action.market_op, MarketOp.NONE)
    market_item = jnp.full_like(action.market_item, -1)
    market_amount = jnp.zeros_like(action.market_amount)
    market_count = jnp.sum(available, axis=1).astype(jnp.int8)
    market_rank = jnp.where(
        activated[:, None], _TERMINAL_MARKET_RANK[None, :], 0
    )
    market_rank = jnp.broadcast_to(market_rank, totals.shape)
    notional = totals * prices
    lexical = jnp.broadcast_to(_LEXICAL_ITEM_RANK[None, :], totals.shape)
    batch = jnp.arange(states.step.shape[0])
    for slot in range(NUM_PRODUCTS):
        best_rank = jnp.max(jnp.where(available, market_rank, -1), axis=1)
        candidate = available & (market_rank == best_rank[:, None])
        best_value = jnp.max(jnp.where(candidate, notional, -1), axis=1)
        candidate &= notional == best_value[:, None]
        best_price = jnp.max(jnp.where(candidate, prices, -1), axis=1)
        candidate &= prices == best_price[:, None]
        best_quantity = jnp.max(jnp.where(candidate, totals, -1), axis=1)
        candidate &= totals == best_quantity[:, None]
        best_lexical = jnp.max(jnp.where(candidate, lexical, -1), axis=1)
        candidate &= lexical == best_lexical[:, None]
        selected = jnp.argmax(candidate.astype(jnp.int8), axis=1).astype(jnp.int32)
        enabled = slot < market_count.astype(jnp.int32)
        market_op = market_op.at[:, slot].set(
            jnp.where(enabled, MarketOp.SELL, MarketOp.NONE)
        )
        market_item = market_item.at[:, slot].set(
            jnp.where(enabled, selected, -1)
        )
        market_amount = market_amount.at[:, slot].set(
            jnp.where(enabled, totals[batch, selected], 0)
        )
        available = available.at[batch, selected].set(
            jnp.where(enabled, False, available[batch, selected])
        )

    terminal = action._replace(
        unit_op=unit_op,
        unit_item=unit_item,
        unit_amount=unit_amount,
        market_op=market_op,
        market_item=market_item,
        market_amount=market_amount,
        market_count=market_count,
    )
    use_units = use[:, None]
    return Action(
        jnp.where(use_units, terminal.unit_op, action.unit_op),
        jnp.where(use_units, terminal.unit_item, action.unit_item),
        jnp.where(use_units, terminal.unit_amount, action.unit_amount),
        jnp.where(use, actual_count, action.unit_count),
        jnp.where(use[:, None], terminal.market_op, action.market_op),
        jnp.where(use[:, None], terminal.market_item, action.market_item),
        jnp.where(use[:, None], terminal.market_amount, action.market_amount),
        jnp.where(use, terminal.market_count, action.market_count),
    )


def _previous_unit(bank, skeleton_id, step, actor):
    batch = jnp.arange(step.shape[0])
    previous = jnp.clip(step - 1, 0, 718)
    safe_actor = jnp.clip(actor.astype(jnp.int32), 0, MAX_UNITS - 1)
    return (
        bank.unit_op[skeleton_id, previous, safe_actor],
        bank.unit_item[skeleton_id, previous, safe_actor],
        bank.unit_amount[skeleton_id, previous, safe_actor],
        safe_actor < bank.unit_count[skeleton_id, previous].astype(jnp.int32),
    )


def _pasture_repair(
    states: State,
    bank,
    skeleton_id,
    action: Action,
    carry: TranCashflowCarryV1,
    player: int,
) -> tuple[Action, TranCashflowCarryV1]:
    step = states.step.astype(jnp.int32)
    actual_count = jnp.sum(
        states.unit_active[:, player].astype(jnp.int32), axis=1
    )
    submitted_count = action.unit_count.astype(jnp.int32)
    active_shift = (carry.farmer_shift_end >= step) & (step < 716)
    previous_op, previous_item, previous_amount, previous_present = _previous_unit(
        bank, skeleton_id, step, jnp.zeros_like(carry.pending_pasture_actor)
    )
    action = _replace_unit(
        action,
        jnp.zeros_like(carry.pending_pasture_actor),
        active_shift & previous_present,
        previous_op,
        previous_item,
        previous_amount,
    )
    farmer_shift_end = jnp.where(
        (carry.farmer_shift_end >= 0) & (~active_shift), -1, carry.farmer_shift_end
    ).astype(jnp.int16)

    pending = (carry.pending_pasture_actor >= 0) & (step < 716)
    expected = pending & (carry.pending_pasture_step.astype(jnp.int32) == step)
    pending_actor = carry.pending_pasture_actor.astype(jnp.int32)
    current_pos = states.unit_pos[
        jnp.arange(step.shape[0]),
        player,
        jnp.clip(pending_actor, 0, MAX_UNITS - 1),
    ]
    same_position = jnp.all(current_pos == carry.pending_pasture_pos, axis=1)
    empty = _tile_kind_at(states, player, current_pos) == TileKind.EMPTY
    present = (pending_actor < actual_count) & (pending_actor < submitted_count)
    action = _replace_unit(
        action,
        pending_actor,
        expected & same_position & empty & present,
        UnitOp.BUILD_PASTURE,
    )

    # Python clears an old pending request on the next call before discovering
    # a new weed blockage.
    pending_actor_out = jnp.full_like(carry.pending_pasture_actor, -1)
    pending_pos_out = jnp.full_like(carry.pending_pasture_pos, -1)
    pending_step_out = jnp.full_like(carry.pending_pasture_step, -1)

    farmer_pos = states.unit_pos[:, player, 0]
    farmer_weed = _tile_kind_at(states, player, farmer_pos) == TileKind.WEED
    farmer_trigger = (
        (step < 716)
        & (action.unit_op[:, 0] == UnitOp.BUILD_PASTURE)
        & farmer_weed
    )
    action = _replace_unit(
        action,
        jnp.zeros_like(pending_actor_out),
        farmer_trigger,
        UnitOp.DIG,
    )
    pending_actor_out = jnp.where(farmer_trigger, 0, pending_actor_out).astype(jnp.int8)
    pending_pos_out = jnp.where(
        farmer_trigger[:, None], farmer_pos, pending_pos_out
    ).astype(jnp.int8)
    pending_step_out = jnp.where(
        farmer_trigger, step + 1, pending_step_out
    ).astype(jnp.int16)
    late = farmer_trigger & ((step % 24) >= 20)
    farmer_shift_end = jnp.where(
        late, (step // 24 + 1) * 24 - 1, farmer_shift_end
    ).astype(jnp.int16)

    selected = farmer_trigger
    for actor in range(1, MAX_UNITS):
        actor_present = (actor < actual_count) & (actor < submitted_count)
        pos = states.unit_pos[:, player, actor]
        trigger = (
            (step < 716)
            & (~selected)
            & actor_present
            & (action.unit_op[:, actor] == UnitOp.BUILD_PASTURE)
            & (_tile_kind_at(states, player, pos) == TileKind.WEED)
        )
        action = _replace_unit(
            action,
            jnp.full(step.shape, actor, dtype=jnp.int32),
            trigger,
            UnitOp.DIG,
        )
        pending_actor_out = jnp.where(
            trigger, actor, pending_actor_out
        ).astype(jnp.int8)
        pending_pos_out = jnp.where(
            trigger[:, None], pos, pending_pos_out
        ).astype(jnp.int8)
        pending_step_out = jnp.where(
            trigger, step + 1, pending_step_out
        ).astype(jnp.int16)
        selected |= trigger

    return action, carry._replace(
        pending_pasture_actor=pending_actor_out,
        pending_pasture_pos=pending_pos_out,
        pending_pasture_step=pending_step_out,
        farmer_shift_end=farmer_shift_end,
    )


def _soil_repair(
    states: State,
    bank,
    skeleton_id,
    action: Action,
    carry: TranCashflowCarryV1,
    player: int,
) -> tuple[Action, TranCashflowCarryV1]:
    step = states.step.astype(jnp.int32)
    actual_count = jnp.sum(
        states.unit_active[:, player].astype(jnp.int32), axis=1
    )
    submitted_count = action.unit_count.astype(jnp.int32)

    shift_actor = carry.water_shift_actor.astype(jnp.int32)
    active_shift = (
        (shift_actor >= 1)
        & (shift_actor < actual_count)
        & (shift_actor < submitted_count)
        & (carry.water_shift_end.astype(jnp.int32) >= step)
    )
    previous_op, previous_item, previous_amount, previous_present = _previous_unit(
        bank, skeleton_id, step, shift_actor
    )
    action = _replace_unit(
        action,
        shift_actor,
        active_shift & previous_present,
        previous_op,
        previous_item,
        previous_amount,
    )
    water_shift_actor = jnp.where(
        (carry.water_shift_actor >= 0) & (~active_shift), -1, carry.water_shift_actor
    ).astype(jnp.int8)
    water_shift_end = jnp.where(
        (carry.water_shift_actor >= 0) & (~active_shift), -1, carry.water_shift_end
    ).astype(jnp.int16)

    pending_water = carry.pending_water_planter >= 1
    expected_water = pending_water & (
        carry.pending_water_step.astype(jnp.int32) == step
    )
    water_tile = _tile_kind_at(states, player, carry.pending_water_pos)
    expected_water &= water_tile != TileKind.EMPTY
    planter = carry.pending_water_planter.astype(jnp.int32)
    chosen = jnp.full(step.shape, -1, dtype=jnp.int32)
    for actor in range(1, MAX_UNITS):
        co_located = jnp.all(
            states.unit_pos[:, player, actor] == carry.pending_water_pos, axis=1
        )
        choose = (
            expected_water
            & (chosen < 0)
            & (actor != planter)
            & (actor < actual_count)
            & (actor < submitted_count)
            & co_located
        )
        chosen = jnp.where(choose, actor, chosen)
    chosen = jnp.where(
        expected_water
        & (chosen < 0)
        & (planter < actual_count)
        & (planter < submitted_count),
        planter,
        chosen,
    )
    apply_water = expected_water & (chosen >= 1)
    action = _replace_unit(action, chosen, apply_water, UnitOp.WATER)
    water_shift_actor = jnp.where(
        apply_water, chosen, water_shift_actor
    ).astype(jnp.int8)
    water_shift_end = jnp.where(
        apply_water, (step // 24 + 1) * 24 - 1, water_shift_end
    ).astype(jnp.int16)

    pending_water_planter = jnp.full_like(carry.pending_water_planter, -1)
    pending_water_pos = jnp.full_like(carry.pending_water_pos, -1)
    pending_water_step = jnp.full_like(carry.pending_water_step, -1)

    pending_plant = carry.pending_plant_actor >= 1
    expected_plant = pending_plant & (
        carry.pending_plant_step.astype(jnp.int32) == step
    )
    plant_actor = carry.pending_plant_actor.astype(jnp.int32)
    safe_plant = jnp.clip(plant_actor, 0, MAX_UNITS - 1)
    plant_pos = states.unit_pos[
        jnp.arange(step.shape[0]), player, safe_plant
    ]
    plant_valid = (
        expected_plant
        & (plant_actor < actual_count)
        & (plant_actor < submitted_count)
        & jnp.all(plant_pos == carry.pending_plant_pos, axis=1)
        & (_tile_kind_at(states, player, plant_pos) == TileKind.EMPTY)
        & (states.seeds[:, player, _WHEAT] > 0)
    )
    action = _replace_unit(
        action, plant_actor, plant_valid, UnitOp.PLANT, _WHEAT
    )
    pending_water_planter = jnp.where(
        plant_valid, plant_actor, pending_water_planter
    ).astype(jnp.int8)
    pending_water_pos = jnp.where(
        plant_valid[:, None], carry.pending_plant_pos, pending_water_pos
    ).astype(jnp.int8)
    pending_water_step = jnp.where(
        plant_valid, step + 1, pending_water_step
    ).astype(jnp.int16)

    pending_plant_actor = jnp.full_like(carry.pending_plant_actor, -1)
    pending_plant_pos = jnp.full_like(carry.pending_plant_pos, -1)
    pending_plant_step = jnp.full_like(carry.pending_plant_step, -1)
    selected = jnp.zeros_like(carry.soil_repair_activated)
    for actor in range(1, MAX_UNITS):
        present = (actor < actual_count) & (actor < submitted_count)
        pos = states.unit_pos[:, player, actor]
        trigger = (
            (step == 636)
            & (~selected)
            & present
            & (action.unit_op[:, actor] == UnitOp.PLANT)
            & (action.unit_item[:, actor] == _WHEAT)
            & (_tile_kind_at(states, player, pos) == TileKind.WEED)
        )
        action = _replace_unit(
            action,
            jnp.full(step.shape, actor, dtype=jnp.int32),
            trigger,
            UnitOp.DIG,
        )
        pending_plant_actor = jnp.where(
            trigger, actor, pending_plant_actor
        ).astype(jnp.int8)
        pending_plant_pos = jnp.where(
            trigger[:, None], pos, pending_plant_pos
        ).astype(jnp.int8)
        pending_plant_step = jnp.where(
            trigger, step + 1, pending_plant_step
        ).astype(jnp.int16)
        selected |= trigger

    return action, carry._replace(
        pending_plant_actor=pending_plant_actor,
        pending_plant_pos=pending_plant_pos,
        pending_plant_step=pending_plant_step,
        pending_water_planter=pending_water_planter,
        pending_water_pos=pending_water_pos,
        pending_water_step=pending_water_step,
        water_shift_actor=water_shift_actor,
        water_shift_end=water_shift_end,
        soil_repair_activated=carry.soil_repair_activated | selected,
    )


def _cashflow_sort(states: State, action: Action, enabled: jax.Array) -> Action:
    op = action.market_op
    item = action.market_item.astype(jnp.int32)
    amount = action.market_amount
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    safe_product = jnp.clip(item, 0, NUM_PRODUCTS - 1)
    sell = active & (op == MarketOp.SELL)
    base = jnp.full(op.shape, 16, dtype=jnp.int32)
    base = jnp.where(sell, _CASHFLOW_SELL_RANK[safe_product], base)
    base = jnp.where(active & (op == MarketOp.HIRE), 8, base)
    base = jnp.where(
        active & (op == MarketOp.BUY_ANIMAL) & (item == 1), 9, base
    )
    base = jnp.where(
        active & (op == MarketOp.BUY_ANIMAL) & (item == 2), 10, base
    )
    base = jnp.where(active & (op == MarketOp.BUY_LAND), 11, base)
    base = jnp.where(
        active & (op == MarketOp.BUY_SEED) & (item == _MELON), 12, base
    )
    base = jnp.where(
        active & (op == MarketOp.BUY_SEED) & (item == _STRAWBERRY), 13, base
    )
    base = jnp.where(
        active & (op == MarketOp.BUY_SEED) & (item == _WHEAT), 14, base
    )
    base = jnp.where(
        active & (op == MarketOp.BUY_PRODUCT) & (item == _WHEAT), 15, base
    )
    key0 = jnp.where(active, jnp.where(sell, 0, base), 1000)
    score = states.market_price[jnp.arange(states.step.shape[0])[:, None], safe_product]
    score = score.astype(jnp.int32) * jnp.maximum(amount, 0).astype(jnp.int32)
    key1 = jnp.where(sell, -score, 0)
    key2 = base

    def swap_adjacent(values, swap, slot):
        left = values[:, slot]
        right = values[:, slot + 1]
        values = values.at[:, slot].set(jnp.where(swap, right, left))
        return values.at[:, slot + 1].set(jnp.where(swap, left, right))

    # Stable bubble network reproduces Python's stable tuple-key sort.
    for _ in range(MAX_MARKET_ORDERS - 1):
        for slot in range(MAX_MARKET_ORDERS - 1):
            greater = (key0[:, slot] > key0[:, slot + 1]) | (
                (key0[:, slot] == key0[:, slot + 1])
                & (
                    (key1[:, slot] > key1[:, slot + 1])
                    | (
                        (key1[:, slot] == key1[:, slot + 1])
                        & (key2[:, slot] > key2[:, slot + 1])
                    )
                )
            )
            swap = enabled & greater
            op = swap_adjacent(op, swap, slot)
            item = swap_adjacent(item, swap, slot)
            amount = swap_adjacent(amount, swap, slot)
            key0 = swap_adjacent(key0, swap, slot)
            key1 = swap_adjacent(key1, swap, slot)
            key2 = swap_adjacent(key2, swap, slot)
    return action._replace(
        market_op=op.astype(jnp.int8),
        market_item=item.astype(jnp.int8),
        market_amount=amount.astype(jnp.int32),
    )


def public_tran_cashflow_player_action_v1(
    states: State,
    bank,
    skeleton_id: jax.Array,
    carry: TranCashflowCarryV1,
    player: int,
) -> tuple[Action, TranCashflowCarryV1]:
    step = states.step.astype(jnp.int32)
    reset = step == 0
    fresh = initialize_tran_cashflow_carry_v1(states.step.shape[0])
    carry = TranCashflowCarryV1(
        *(
            jnp.where(
                reset.reshape((reset.shape[0],) + (1,) * (new.ndim - 1)),
                new,
                old,
            )
            for new, old in zip(fresh, carry, strict=True)
        )
    )

    action = _trace_action(states, bank, skeleton_id)
    action = _terminal_action(
        states, action, carry.soil_repair_activated, player
    )
    action, carry = _pasture_repair(
        states, bank, skeleton_id, action, carry, player
    )
    action, carry = _soil_repair(
        states, bank, skeleton_id, action, carry, player
    )

    opponent = 1 - player
    opponent_crops = states.tile_crop[:, opponent]
    opponent_animals = states.tile_animal[:, opponent]
    detected = (
        (jnp.sum(opponent_crops == _WHEAT, axis=(1, 2)) >= 5)
        & (jnp.sum(opponent_crops == _STRAWBERRY, axis=(1, 2)) >= 26)
        & (jnp.sum(opponent_crops == _MELON, axis=(1, 2)) == 6)
        & (jnp.sum(opponent_animals == 1, axis=(1, 2)) >= 8)
        & (jnp.sum(opponent_animals == 2, axis=(1, 2)) >= 6)
    )
    cashflow_active = jnp.where(
        step == 300, detected, carry.cashflow_active
    )
    action = _cashflow_sort(
        states,
        action,
        cashflow_active & (step >= 300) & (step < 715),
    )
    return action, carry._replace(cashflow_active=cashflow_active)
