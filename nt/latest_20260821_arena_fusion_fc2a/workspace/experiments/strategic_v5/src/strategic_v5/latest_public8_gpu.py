"""JAX runtime controllers for the 2026-08-20 latest-public-eight freeze.

This module only contains source-runtime semantics that are not already owned
by the accepted historical controllers.  Raw 719-step production tapes live
in ``latest_public8_route_bank_v1.npz``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_COST,
    CROP_SEED_COST,
    FLAG_FED,
    HIRE_COST,
    LAND_PRICES,
    MARKET_BASE_PRICES,
    MARKET_MIN_INVENTORY,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    PRODUCTS,
    SHOP_NAMES,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.types import Action, State
from strategic_v5.high_potential_v20_gpu import (
    MODE_BOATLEE,
    HighPotentialV20CarryV1,
    HighPotentialRuntimeTablesV1,
    _COUNTER_ITEMS,
    _LIQUIDATION,
    _PREMIUM,
    _append_or_merge_sale,
    _market_counters,
    _opponent_counts,
    _pickup_reserve,
    _planned_sales,
    _projected_shed_without_pickups,
    _rank_sell_slots_exact,
    _raw_with_weed,
    _repay as _hp_repay,
    _room_evac,
    _room_guard,
    _schedule_targets,
    _terminal_liquidation,
    _town_demand_at,
    _trim_ray_seeds,
    initialize_high_potential_v20_carry_v1,
)
from strategic_v5.public_g02_gpu import (
    PublicG02CarryV1,
    _compact_market,
    _original_front_run,
    _project_shed,
    _raw_action,
    _repay,
    _town_demand,
    _weed_repair,
    initialize_public_g02_carry_v1,
)


_WHEAT = PRODUCTS.index("WHEAT")
_OLD_PRICE_FLOOR_SENTINEL = 1
_HIRE_COST = jnp.asarray(HIRE_COST, dtype=jnp.int32)
_LAND_PRICE = jnp.asarray(LAND_PRICES, dtype=jnp.int32)
_SEED_COST = jnp.asarray(CROP_SEED_COST, dtype=jnp.int32)
_ANIMAL_COST = jnp.asarray(ANIMAL_COST, dtype=jnp.int32)
_BASE_PRICE = jnp.asarray(MARKET_BASE_PRICES, dtype=jnp.int32)
_ICE_CREAM = SHOP_NAMES.index("ICE_CREAM_SHOP")
_YARN_STORE = SHOP_NAMES.index("YARN_STORE")


def deniz_v111_player_action_v1(
    states: State,
    bank,
    route_id: jax.Array,
    carry: PublicG02CarryV1,
    player: int,
) -> tuple[Action, PublicG02CarryV1]:
    """Exact RC5 weed repair plus the source's one-step premium front-run."""

    action = _raw_action(states, bank, route_id, player)
    action, carry = _weed_repair(states, bank, route_id, action, carry, player)
    step = states.step
    action, due_step, due = _repay(action, step, carry.due_step, carry.due)
    action, due_step, due = _original_front_run(
        states, bank, route_id, action, player, due_step, due
    )
    return action, carry._replace(due_step=due_step, due=due)


class KaitoV36CarryV1(NamedTuple):
    """All mutable state owned by the two sparse V36 feedback experts."""

    base: PublicG02CarryV1
    preempt_near_streak: jax.Array
    preempt_latched: jax.Array
    mm_open_units: jax.Array
    mm_entry_step: jax.Array
    mm_baseline_wheat: jax.Array
    mm_near_streak: jax.Array
    mm_mirror_mode: jax.Array


def initialize_kaito_v36_carry_v1(batch_size: int) -> KaitoV36CarryV1:
    return KaitoV36CarryV1(
        base=initialize_public_g02_carry_v1(batch_size),
        preempt_near_streak=jnp.zeros((batch_size,), dtype=jnp.int16),
        preempt_latched=jnp.zeros((batch_size,), dtype=jnp.bool_),
        mm_open_units=jnp.zeros((batch_size,), dtype=jnp.int16),
        mm_entry_step=jnp.full((batch_size,), -1, dtype=jnp.int16),
        mm_baseline_wheat=jnp.zeros((batch_size,), dtype=jnp.int16),
        mm_near_streak=jnp.zeros((batch_size,), dtype=jnp.int16),
        # -1 is Python None, 0 is False, 1 is True.
        mm_mirror_mode=jnp.full((batch_size,), -1, dtype=jnp.int8),
    )


def _kaito_clone_distance(states: State) -> jax.Array:
    """Exact ``scripts.v19_terminal.clone_distance`` public signature."""

    kinds, crops, animals = states.tile_kind, states.tile_crop, states.tile_animal
    empty = (crops < 0) & (animals < 0)
    columns = (
        jnp.sum(crops == PRODUCTS.index("CARROT"), axis=(2, 3)),
        jnp.sum(empty & (kinds == TileKind.COOP), axis=(2, 3)),
        jnp.sum(animals == 1, axis=(2, 3)),
        jnp.sum(animals == 0, axis=(2, 3)),
        jnp.sum(crops == PRODUCTS.index("MELON"), axis=(2, 3)),
        jnp.sum(empty & (kinds == TileKind.PASTURE), axis=(2, 3)),
        jnp.sum(animals == 2, axis=(2, 3)),
        jnp.sum(crops == PRODUCTS.index("STRAWBERRY"), axis=(2, 3)),
        jnp.sum(crops == PRODUCTS.index("TOMATO"), axis=(2, 3)),
        jnp.sum(empty & (kinds == TileKind.WEED), axis=(2, 3)),
        jnp.sum(crops == PRODUCTS.index("WHEAT"), axis=(2, 3)),
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


def _append_market(
    action: Action,
    enabled: jax.Array,
    op: int,
    item: int,
    amount: jax.Array,
) -> Action:
    batch = jnp.arange(action.market_count.shape[0])
    slot = jnp.clip(action.market_count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1)
    enabled = enabled & (action.market_count < MAX_MARKET_ORDERS)
    return action._replace(
        market_op=action.market_op.at[batch, slot].set(
            jnp.where(enabled, op, action.market_op[batch, slot])
        ),
        market_item=action.market_item.at[batch, slot].set(
            jnp.where(enabled, item, action.market_item[batch, slot])
        ),
        market_amount=action.market_amount.at[batch, slot].set(
            jnp.where(enabled, amount, action.market_amount[batch, slot])
        ),
        market_count=(action.market_count + enabled.astype(jnp.int8)).astype(jnp.int8),
    )


def _kaito_repay(
    action: Action, step: jax.Array, due_step: jax.Array, due: jax.Array
) -> tuple[Action, jax.Array, jax.Array]:
    """Repay every shifted item and carry unpaid debt to the next step."""

    due_now = due_step.astype(jnp.int32) == step.astype(jnp.int32)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    keep = active.copy()
    amount = action.market_amount.astype(jnp.int32)
    remaining = due.astype(jnp.int32)
    batch = jnp.arange(step.shape[0])
    for slot in range(MAX_MARKET_ORDERS):
        is_sell = active[:, slot] & (action.market_op[:, slot] == MarketOp.SELL)
        item = jnp.clip(action.market_item[:, slot].astype(jnp.int32), 0, NUM_PRODUCTS - 1)
        owed = remaining[batch, item]
        reduction = jnp.where(
            due_now & is_sell,
            jnp.minimum(jnp.maximum(amount[:, slot], 0), owed),
            0,
        )
        amount = amount.at[:, slot].add(-reduction)
        remaining = remaining.at[batch, item].add(-reduction)
        keep = keep.at[:, slot].set(~(due_now & is_sell & (amount[:, slot] <= 0)))
    action = _compact_market(action._replace(market_amount=amount), keep)
    leftovers = due_now & (jnp.sum(remaining, axis=1) > 0)
    return (
        action,
        jnp.where(leftovers, step + 1, -1).astype(jnp.int16),
        jnp.where(leftovers[:, None], remaining, 0).astype(jnp.int16),
    )


def _kaito_preempt(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    bank,
    route_id: jax.Array,
    action: Action,
    carry: KaitoV36CarryV1,
    player: int,
) -> tuple[Action, KaitoV36CarryV1]:
    step = states.step.astype(jnp.int32)
    distance = _kaito_clone_distance(states)
    near_streak = jnp.where(
        distance <= 2, carry.preempt_near_streak.astype(jnp.int32) + 1, 0
    ).astype(jnp.int16)
    latched = carry.preempt_latched | (
        (step >= 48) & (near_streak.astype(jnp.int32) >= 24)
    )
    base = carry.base
    repaid, due_step, due = _kaito_repay(action, step, base.due_step, base.due)
    action = Action(
        *(jnp.where(latched.reshape((-1,) + (1,) * (field.ndim - 1)), field, old)
          for field, old in zip(repaid, action, strict=True))
    )
    due_step = jnp.where(latched, due_step, base.due_step).astype(jnp.int16)
    due = jnp.where(latched[:, None], due, base.due).astype(jnp.int16)

    future_step = jnp.clip(step + 1, 0, 718)
    future_active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < bank.market_count[
        route_id, future_step
    ][:, None]
    projected = _projected_shed_without_pickups(states, action, player)
    available = jnp.maximum(projected[:, :NUM_PRODUCTS] - _planned_sales(action), 0)
    shifted = jnp.zeros_like(due, dtype=jnp.int32)
    total = jnp.zeros(step.shape, dtype=jnp.int32)
    eligible = latched & (step >= 48) & (step < 679) & (
        action.market_count < MAX_MARKET_ORDERS
    )
    for product in _LIQUIDATION.tolist():
        requested = jnp.sum(
            jnp.where(
                future_active
                & (bank.market_op[route_id, future_step] == MarketOp.SELL)
                & (bank.market_item[route_id, future_step] == product),
                jnp.maximum(bank.market_amount[route_id, future_step], 0),
                0,
            ),
            axis=1,
        )
        quantity = jnp.minimum(
            jnp.minimum(requested, available[:, product]),
            jnp.maximum(100 - total, 0),
        )
        append = eligible & (quantity > 0) & (action.market_count < MAX_MARKET_ORDERS)
        action = _append_market(action, append, MarketOp.SELL, product, quantity)
        used = jnp.where(append, quantity, 0)
        available = available.at[:, product].add(-used)
        shifted = shifted.at[:, product].add(used)
        total = total + used
    changed = jnp.sum(shifted, axis=1) > 0
    due_step = jnp.where(changed, step + 1, due_step).astype(jnp.int16)
    due = jnp.where(changed[:, None], due.astype(jnp.int32) + shifted, due).astype(
        jnp.int16
    )
    ranked = _rank_sell_slots_exact(states, runtime, action)
    action = Action(
        *(jnp.where(changed.reshape((-1,) + (1,) * (field.ndim - 1)), field, old)
          for field, old in zip(ranked, action, strict=True))
    )
    return action, carry._replace(
        base=base._replace(due_step=due_step, due=due),
        preempt_near_streak=near_streak,
        preempt_latched=latched,
    )


def _old_price(runtime: HighPotentialRuntimeTablesV1, item: jax.Array, inventory: jax.Array):
    index = jnp.clip(
        inventory.astype(jnp.int32) - MARKET_MIN_INVENTORY,
        0,
        runtime.agent_market_price_lut.shape[1] - 1,
    )
    return runtime.agent_market_price_lut[item, index].astype(jnp.int32)


def _old_inventory_after_market(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    action: Action,
    product: int,
) -> jax.Array:
    inventory = states.market_inventory[:, product].astype(jnp.int32)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    floor_mask = runtime.agent_market_price_lut[product] <= 1
    floor = jnp.where(
        jnp.any(floor_mask),
        jnp.argmax(floor_mask),
        runtime.agent_market_price_lut.shape[1],
    ).astype(jnp.int32)
    for slot in range(MAX_MARKET_ORDERS):
        matching = active[:, slot] & (action.market_item[:, slot] == product)
        quantity = jnp.maximum(action.market_amount[:, slot], 0).astype(jnp.int32)
        buy = matching & (action.market_op[:, slot] == MarketOp.BUY_PRODUCT)
        sell = matching & (action.market_op[:, slot] == MarketOp.SELL)
        before_floor = jnp.maximum(
            floor - (inventory - MARKET_MIN_INVENTORY), 0
        )
        inventory = jnp.where(buy, inventory - quantity, inventory)
        inventory = jnp.where(sell, inventory + jnp.minimum(quantity, before_floor), inventory)
    return inventory


def _shed_after_market(projected: jax.Array, action: Action) -> jax.Array:
    shed = projected.astype(jnp.int32)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    batch = jnp.arange(action.market_count.shape[0])
    for slot in range(MAX_MARKET_ORDERS):
        item = jnp.clip(action.market_item[:, slot].astype(jnp.int32), 0, NUM_SHED_ITEMS - 1)
        quantity = jnp.maximum(action.market_amount[:, slot], 0).astype(jnp.int32)
        matching = active[:, slot] & (action.market_item[:, slot] >= 0) & (
            action.market_item[:, slot] < NUM_SHED_ITEMS
        )
        sell = matching & (action.market_op[:, slot] == MarketOp.SELL)
        shed = shed.at[batch, item].add(
            -jnp.where(sell, jnp.minimum(quantity, shed[batch, item]), 0)
        )
        buying = matching & (
            (action.market_op[:, slot] == MarketOp.BUY_PRODUCT)
            | (action.market_op[:, slot] == MarketOp.BUY_ANIMAL)
        )
        room = jnp.maximum(100 - jnp.sum(shed, axis=1), 0)
        shed = shed.at[batch, item].add(jnp.where(buying, jnp.minimum(quantity, room), 0))
    return shed


def _wheat_after_market(quantity: jax.Array, action: Action) -> jax.Array:
    quantity = quantity.astype(jnp.int32)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    for slot in range(MAX_MARKET_ORDERS):
        matching = active[:, slot] & (action.market_item[:, slot] == _WHEAT)
        units = jnp.maximum(action.market_amount[:, slot], 0).astype(jnp.int32)
        quantity = jnp.where(
            matching & (action.market_op[:, slot] == MarketOp.SELL),
            jnp.maximum(quantity - units, 0),
            quantity,
        )
        quantity = jnp.where(
            matching & (action.market_op[:, slot] == MarketOp.BUY_PRODUCT),
            quantity + units,
            quantity,
        )
    return quantity


def _kaito_investment_reserve(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    bank,
    route_id: jax.Array,
    player: int,
) -> jax.Array:
    """Literal two-step raw-route reserve used by ``planned_investment_cost``."""

    step = states.step.astype(jnp.int32)
    hires = states.hires_today[:, player].astype(jnp.int32)
    unlocked = states.unlocked_count[:, player].astype(jnp.int32)
    total = jnp.zeros(step.shape, dtype=jnp.float32)
    wheat_quote = _old_price(
        runtime,
        jnp.full(step.shape, _WHEAT, dtype=jnp.int32),
        states.market_inventory[:, _WHEAT],
    ).astype(jnp.float32)
    previous_day = step // 24
    for ahead in range(2):
        requested = step + ahead
        day = requested // 24
        hires = jnp.where(day != previous_day, 0, hires)
        previous_day = day
        route_step = jnp.clip(requested, 0, 718)
        count = bank.market_count[route_id, route_step].astype(jnp.int32)
        for slot in range(MAX_MARKET_ORDERS):
            active = (slot < count) & (requested < 719)
            op = bank.market_op[route_id, route_step, slot]
            item = bank.market_item[route_id, route_step, slot].astype(jnp.int32)
            amount = jnp.maximum(
                bank.market_amount[route_id, route_step, slot], 0
            ).astype(jnp.int32)
            hire_index = jnp.clip(hires, 0, len(HIRE_COST) - 1)
            hire = active & (op == MarketOp.HIRE)
            cost = jnp.where(hire, _HIRE_COST[hire_index], 0).astype(jnp.float32)
            land_index = jnp.clip(unlocked - 1, 0, len(LAND_PRICES) - 1)
            land_valid = active & (op == MarketOp.BUY_LAND) & (
                unlocked - 1 < len(LAND_PRICES)
            )
            cost = cost + jnp.where(land_valid, _LAND_PRICE[land_index], 0)
            crop = jnp.clip(item, 0, len(CROP_SEED_COST) - 1)
            cost = cost + jnp.where(
                active & (op == MarketOp.BUY_SEED) & (item < len(CROP_SEED_COST)),
                _SEED_COST[crop] * amount,
                0,
            )
            animal = item - NUM_PRODUCTS
            safe_animal = jnp.clip(animal, 0, len(ANIMAL_COST) - 1)
            cost = cost + jnp.where(
                active
                & (op == MarketOp.BUY_ANIMAL)
                & (animal >= 0)
                & (animal < len(ANIMAL_COST)),
                _ANIMAL_COST[safe_animal] * amount,
                0,
            )
            cost = cost + jnp.where(
                active & (op == MarketOp.BUY_PRODUCT), wheat_quote * amount, 0
            )
            total = total + cost.astype(jnp.float32)
            hires = hires + hire.astype(jnp.int32)
            unlocked = unlocked + land_valid.astype(jnp.int32)
    return total * 1.05


def _kaito_feed_reserve(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    action: Action,
    player: int,
) -> jax.Array:
    total_wheat = states.shed[:, player, _WHEAT].astype(jnp.int32) + jnp.sum(
        states.unit_inventory[:, player, :, _WHEAT].astype(jnp.int32), axis=1
    )
    batch = jnp.arange(states.step.shape[0])
    present = jnp.arange(MAX_UNITS)[None, :] < action.unit_count[:, None]
    positions = states.unit_pos[:, player].astype(jnp.int32)
    x = jnp.clip(positions[..., 0], 0, 9)
    y = jnp.clip(positions[..., 1], 0, 9)
    tile_animal = states.tile_animal[batch[:, None], player, y, x]
    flags = states.tile_flags[batch[:, None], player, y, x]
    feed = (
        present
        & (action.unit_op == UnitOp.FEED)
        & (tile_animal >= 0)
        & ((flags & FLAG_FED) == 0)
        & (states.unit_inventory[:, player, :, _WHEAT] > 0)
    )
    projected = total_wheat - jnp.sum(feed, axis=1).astype(jnp.int32)
    # Existing sells reduce protected stock; existing buys are deliberately not credited.
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    sold = jnp.sum(
        jnp.where(
            active
            & (action.market_op == MarketOp.SELL)
            & (action.market_item == _WHEAT),
            jnp.maximum(action.market_amount, 0),
            0,
        ),
        axis=1,
    )
    projected = jnp.maximum(projected - sold, 0)
    animal_count = jnp.sum(states.tile_animal[:, player] >= 0, axis=(1, 2)).astype(
        jnp.int32
    )
    deficit = jnp.maximum(2 * animal_count - projected, 0)
    inventory = states.market_inventory[:, _WHEAT].astype(jnp.int32)
    cost = jnp.zeros(inventory.shape, dtype=jnp.int32)
    for unit in range(64):
        use = unit < deficit
        quote = _old_price(
            runtime,
            jnp.full(inventory.shape, _WHEAT, dtype=jnp.int32),
            inventory - unit - 1,
        )
        cost = cost + jnp.where(use, quote, 0)
    return cost.astype(jnp.float32) * 1.05


def _kaito_market_maker(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    bank,
    route_id: jax.Array,
    action: Action,
    carry: KaitoV36CarryV1,
    player: int,
) -> tuple[Action, KaitoV36CarryV1]:
    step = states.step.astype(jnp.int32)
    distance = _kaito_clone_distance(states)
    near_streak = jnp.where(
        distance <= 2, carry.mm_near_streak.astype(jnp.int32) + 1, 0
    ).astype(jnp.int16)
    decide_mirror = (carry.mm_mirror_mode < 0) & (step >= 72)
    mirror_mode = jnp.where(
        decide_mirror,
        (near_streak.astype(jnp.int32) >= 48).astype(jnp.int8),
        carry.mm_mirror_mode,
    ).astype(jnp.int8)

    projected = _project_shed(states, action, player).astype(jnp.int32)
    observed_wheat = states.shed[:, player, _WHEAT].astype(jnp.int32)
    actor_delta = projected[:, _WHEAT] - observed_wheat
    operational = _wheat_after_market(
        jnp.maximum(carry.mm_baseline_wheat.astype(jnp.int32) + actor_delta, 0),
        action,
    )
    total_after = _wheat_after_market(projected[:, _WHEAT], action)
    available_position = jnp.maximum(total_after - operational, 0)
    has_position = carry.mm_open_units > 0
    exit_slot = action.market_count < MAX_MARKET_ORDERS
    exit_quantity = jnp.minimum(
        carry.mm_open_units.astype(jnp.int32), available_position
    )
    exit_now = has_position & exit_slot & (exit_quantity > 0)
    consumed = has_position & exit_slot & (exit_quantity <= 0)
    action = _append_market(action, exit_now, MarketOp.SELL, _WHEAT, exit_quantity)
    closed = exit_now | consumed
    open_units = jnp.where(closed, 0, carry.mm_open_units).astype(jnp.int16)
    baseline = jnp.where(closed, operational, carry.mm_baseline_wheat).astype(jnp.int16)

    # The source never re-enters on an exit/consumption turn and waits when no slot exists.
    may_enter = ~has_position
    projected_after_market = _shed_after_market(projected, action)
    capacity = jnp.maximum(100 - 10 - jnp.sum(projected_after_market, axis=1), 0)
    feed = _kaito_feed_reserve(states, runtime, action, player)
    investment = _kaito_investment_reserve(states, runtime, bank, route_id, player)
    available_cash = jnp.maximum(
        states.money[:, player].astype(jnp.float32) - 500.0 - feed - investment, 0.0
    )
    demand = _town_demand(states, _WHEAT).astype(jnp.int32)
    future = jnp.clip(step + 1, 0, 718)
    future_slot = (step + 1 < 719) & (bank.market_count[route_id, future] < 10)
    entry_gate = (
        may_enter
        & (step >= 72)
        & (step <= 716)
        & (demand > 0)
        & (action.market_count < MAX_MARKET_ORDERS)
        & future_slot
        & (capacity > 0)
        & (available_cash > 0)
    )
    inventory = _old_inventory_after_market(states, runtime, action, _WHEAT)
    best_q = jnp.zeros(step.shape, dtype=jnp.int32)
    best_profit = jnp.full(step.shape, -jnp.inf, dtype=jnp.float32)
    cumulative_cost = jnp.zeros(step.shape, dtype=jnp.int32)
    for quantity in range(1, 11):
        buy_quote = _old_price(
            runtime,
            jnp.full(step.shape, _WHEAT, dtype=jnp.int32),
            inventory - quantity,
        )
        cumulative_cost = cumulative_cost + buy_quote
        before_sell = inventory - quantity - demand
        revenue = jnp.zeros(step.shape, dtype=jnp.int32)
        sell_inventory = before_sell
        for unit in range(quantity):
            quote = _old_price(
                runtime,
                jnp.full(step.shape, _WHEAT, dtype=jnp.int32),
                sell_inventory,
            )
            revenue = revenue + quote
            sell_inventory = sell_inventory + (quote > _OLD_PRICE_FLOOR_SENTINEL)
        profit = (revenue - cumulative_cost).astype(jnp.float32)
        valid = (
            entry_gate
            & (quantity <= jnp.minimum(capacity, 10))
            & (cumulative_cost.astype(jnp.float32) * 1.05 <= available_cash)
            & (profit >= 1.0)
        )
        improve = valid & (profit > best_profit)
        best_profit = jnp.where(improve, profit, best_profit)
        best_q = jnp.where(improve, quantity, best_q)
    enter = best_q > 0
    action = _append_market(action, enter, MarketOp.BUY_PRODUCT, _WHEAT, best_q)
    # Baseline excludes the incremental MM buy but includes the base action's market.
    entry_baseline = _wheat_after_market(projected[:, _WHEAT], action._replace(
        market_count=(action.market_count - enter.astype(jnp.int8)).astype(jnp.int8)
    ))
    open_units = jnp.where(enter, best_q, open_units).astype(jnp.int16)
    entry_step = jnp.where(enter, step, carry.mm_entry_step).astype(jnp.int16)
    baseline = jnp.where(enter, entry_baseline, baseline).astype(jnp.int16)
    return action, carry._replace(
        mm_open_units=open_units,
        mm_entry_step=entry_step,
        mm_baseline_wheat=baseline,
        mm_near_streak=near_streak,
        mm_mirror_mode=mirror_mode,
    )


def kaito_v36_player_action_v1(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    bank,
    route_id: jax.Array,
    carry: KaitoV36CarryV1,
    player: int,
) -> tuple[Action, KaitoV36CarryV1]:
    """Exact JAX transcription of the latest fixed V36 hybrid runtime."""

    action = _raw_action(states, bank, route_id, player)
    action, base = _weed_repair(states, bank, route_id, action, carry.base, player)
    action = _rank_sell_slots_exact(states, runtime, action)
    carry = carry._replace(base=base)
    action, carry = _kaito_preempt(
        states, runtime, bank, route_id, action, carry, player
    )
    return _kaito_market_maker(
        states, runtime, bank, route_id, action, carry, player
    )


class X562CarryV1(NamedTuple):
    """Dynamic X540/X562 route choice plus the inherited feedback state."""

    base: HighPotentialV20CarryV1
    expert_decided: jax.Array
    expert_high: jax.Array


def initialize_x562_carry_v1(batch_size: int) -> X562CarryV1:
    return X562CarryV1(
        base=initialize_high_potential_v20_carry_v1(batch_size),
        expert_decided=jnp.zeros((batch_size,), dtype=jnp.bool_),
        expert_high=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


def _x562_route(states: State, carry: X562CarryV1) -> tuple[jax.Array, X562CarryV1]:
    """Freeze the public shop selector at step 168; route IDs 12/13 are low/high."""

    step = states.step.astype(jnp.int32)
    active = jnp.arange(states.town_shops.shape[1])[None, :] < states.town_count[:, None]
    has_yarn = jnp.any(active & (states.town_shops == _YARN_STORE), axis=1)
    dominated = (
        (states.town_count >= 2)
        & (states.town_shops[:, 0] == _ICE_CREAM)
        & (states.town_shops[:, 1] == _YARN_STORE)
    )
    decide = (~carry.expert_decided) & (step >= 168)
    high = jnp.where(decide, has_yarn & (~dominated), carry.expert_high)
    decided = carry.expert_decided | decide
    route = jnp.where(high, 13, 12).astype(jnp.int32)
    return route, carry._replace(expert_decided=decided, expert_high=high)


def _x562_preempt(
    states: State,
    bank,
    route: jax.Array,
    action: Action,
    carry: HighPotentialV20CarryV1,
    player: int,
) -> tuple[Action, HighPotentialV20CarryV1]:
    """X492 quote-gated one-step premium preemption, including cap 30."""

    step = states.step.astype(jnp.int32)
    eligible = (
        (step >= 120)
        & (step < 680)
        & (jnp.sum(carry.due.astype(jnp.int32), axis=1) == 0)
        & (_kaito_clone_distance(states) <= 6)
        & (action.market_count < MAX_MARKET_ORDERS)
    )
    future_step = jnp.clip(step + 1, 0, 718)
    future_active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < bank.market_count[
        route, future_step
    ][:, None]
    remaining = jnp.maximum(
        _projected_shed_without_pickups(states, action, player)[:, :NUM_PRODUCTS]
        - _planned_sales(action),
        0,
    )
    shifted = jnp.zeros_like(carry.due, dtype=jnp.int32)
    for product in _PREMIUM.tolist():
        future_quantity = jnp.sum(
            jnp.where(
                future_active
                & (bank.market_op[route, future_step] == MarketOp.SELL)
                & (bank.market_item[route, future_step] == product),
                jnp.maximum(bank.market_amount[route, future_step], 0),
                0,
            ),
            axis=1,
        )
        quantity = jnp.minimum(
            jnp.minimum(remaining[:, product], future_quantity), 30
        )
        append = (
            eligible
            & (future_quantity >= 4)
            & (states.market_price[:, product] >= _BASE_PRICE[product])
            & (quantity > 0)
            & (action.market_count < MAX_MARKET_ORDERS)
        )
        action = _append_market(action, append, MarketOp.SELL, product, quantity)
        used = jnp.where(append, quantity, 0)
        remaining = remaining.at[:, product].add(-used)
        shifted = shifted.at[:, product].add(used)
    changed = jnp.sum(shifted, axis=1) > 0
    return action, carry._replace(
        due_step=jnp.where(changed, step + 1, carry.due_step).astype(jnp.int16),
        due=jnp.where(changed[:, None], shifted, carry.due).astype(jnp.int16),
    )


def _x562_market_counters(
    states: State,
    action: Action,
    carry: HighPotentialV20CarryV1,
    runtime: HighPotentialRuntimeTablesV1,
    player: int,
) -> tuple[Action, HighPotentialV20CarryV1]:
    """Literal R5 step+2/fraction1 and MD step+1/fraction2 counters."""

    step = states.step.astype(jnp.int32)
    _, _, cows, sheep, _ = _opponent_counts(states, player)
    r5 = carry.r5_target | ((step >= 24) & (sheep >= 4) & (cows <= 3))
    md = carry.md_target | (
        (step >= 160)
        & (
            (
                (states.unlocked_count[:, 1 - player] >= 2)
                & (cows >= 4)
                & (sheep <= 2)
            )
            | (cows >= 9)
        )
    )
    carry = carry._replace(r5_target=r5, md_target=md)
    planned = _planned_sales(action)
    reserve = _pickup_reserve(action)
    shed = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int32)

    future_r5 = jnp.clip(step + 2, 0, 718)
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
        available = jnp.maximum(
            shed[:, product] - planned[:, product] - reserve[:, product], 0
        )
        quantity = jnp.minimum(available, jnp.maximum(1, target))
        enabled = r5 & (step + 2 < 719) & (target > 0) & clean & (quantity > 0)
        action, changed = _append_or_merge_sale(
            action, enabled, product, quantity
        )
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
        available = jnp.maximum(
            shed[:, product] - planned[:, product] - reserve[:, product], 0
        )
        quantity = jnp.minimum(available, jnp.maximum(1, target * 2))
        enabled = md & (step + 1 < 719) & (target > 0) & (quantity > 0)
        action, changed = _append_or_merge_sale(
            action, enabled, product, quantity
        )
        planned = planned.at[:, product].add(jnp.where(changed, quantity, 0))
    return action, carry


def _x562_seed_budget_guard(states: State, action: Action, player: int) -> Action:
    """Prevent official atomic PLANT cancellation without fixing a route quantity."""

    present = jnp.arange(MAX_UNITS)[None, :] < action.unit_count[:, None]
    batch = jnp.arange(states.step.shape[0])
    positions = states.unit_pos[:, player].astype(jnp.int32)
    x = jnp.clip(positions[..., 0], 0, 9)
    y = jnp.clip(positions[..., 1], 0, 9)
    empty = states.tile_kind[batch[:, None], player, y, x] == TileKind.EMPTY
    unit_op, unit_item, unit_amount = action.unit_op, action.unit_item, action.unit_amount
    actors = jnp.arange(MAX_UNITS)[None, :]
    for crop in range(5):
        request = present & (unit_op == UnitOp.PLANT) & (unit_item == crop)
        key = jnp.where(request, (~empty).astype(jnp.int32) * MAX_UNITS + actors, 1000 + actors)
        order = jnp.argsort(key, axis=1, stable=True)
        inverse = jnp.argsort(order, axis=1, stable=True)
        budget = states.seeds[:, player, crop].astype(jnp.int32)
        keep = request & (inverse < budget[:, None])
        cancel = request & (~keep)
        unit_op = jnp.where(cancel, UnitOp.PASS, unit_op).astype(jnp.int8)
        unit_item = jnp.where(cancel, -1, unit_item).astype(jnp.int8)
        unit_amount = jnp.where(cancel, 1, unit_amount).astype(jnp.int32)
    return action._replace(
        unit_op=unit_op, unit_item=unit_item, unit_amount=unit_amount
    )


def _x562_idle_fertilizer_sale(
    states: State, action: Action, player: int
) -> Action:
    fertilizer = PRODUCTS.index("FERTILIZER")
    present = jnp.arange(MAX_UNITS)[None, :] < action.unit_count[:, None]
    pickups = jnp.sum(
        jnp.where(
            present
            & (action.unit_op == UnitOp.PICKUP)
            & (action.unit_item == fertilizer),
            jnp.maximum(action.unit_amount, 0),
            0,
        ),
        axis=1,
    )
    existing = _planned_sales(action)[:, fertilizer]
    available = jnp.maximum(
        states.shed[:, player, fertilizer].astype(jnp.int32) - existing - pickups, 0
    )
    quantity = jnp.minimum(12, available)
    enabled = (
        (states.step.astype(jnp.int32) % 24 == 23)
        & (states.market_price[:, fertilizer] >= 25)
        & (quantity > 0)
        & (action.market_count < MAX_MARKET_ORDERS)
    )
    return _append_market(
        action, enabled, MarketOp.SELL, fertilizer, quantity
    )


def x562_player_action_v1(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    bank,
    carry: X562CarryV1,
    player: int,
) -> tuple[Action, X562CarryV1]:
    """Exact latest X562 runtime, retaining every observation-conditioned layer."""

    route, carry = _x562_route(states, carry)
    action, base = _raw_with_weed(states, bank, route, carry.base, player)
    # The source feed guard exists but is frozen off.
    action, base = _room_evac(states, action, base, player)
    action, base = _hp_repay(
        action, base, states.step.astype(jnp.int32), MODE_BOATLEE
    )
    action = _rank_sell_slots_exact(states, runtime, action)
    action, base = _x562_preempt(states, bank, route, action, base, player)
    action, base = _x562_market_counters(states, action, base, runtime, player)
    action = _room_guard(states, action, player)
    action = _x562_seed_budget_guard(states, action, player)
    action = _terminal_liquidation(states, action, player)
    action = _x562_idle_fertilizer_sale(states, action, player)
    action = _trim_ray_seeds(states, runtime, route, action, player)
    return action, carry._replace(base=base)
