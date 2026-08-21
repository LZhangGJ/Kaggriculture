"""Fixed-shape, state-driven route genomes for R6/R7 offline search.

The genome is data, never Python control flow.  Candidate values may change
between calls without changing tensor shapes or forcing a new JIT executable.
The action spine comes from a frozen high-quality or native choreography, while
visible-state feedback controls investments, animal scale, sales, liquidation,
and bounded retries of failed worker tasks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import hashlib
import json
from typing import Iterable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import (
    ANIMAL_COST,
    ANIMAL_FIRST_YIELD_DAY,
    ANIMAL_PRODUCT,
    HIRE_COST,
    LAND_PRICES,
    MARKET_BASE_PRICES,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    PRODUCTS,
    SHED_CAPACITY,
    SHOP_NAMES,
    TURNS_PER_DAY,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_WATERED,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.types import Action, State, StaticTables
from strategic_v5.boatlee_v16_gpu import BoatleePlayerCarryV1, _rank_sell_slots

from .trace_core import (
    CalibrationTraceV1,
    initialize_trace_player_carry_v1,
    skeleton_player_action_v1,
)
from .schema import FertilizerPolicyV1, RouteScheduleV1


class RouteGenomeDirectionV1(IntEnum):
    GOLD_CONVERSION = 0
    NATIVE_EXPANSION = 1


class RouteGenomeFamilyV1(IntEnum):
    MILK = 5
    WOOL = 6


class ShopGateV1(IntEnum):
    NONE = 0
    TARGET_DEMAND_SHOP = 1


class RouteGenomeV1(NamedTuple):
    """One fixed set of leading-dimension genome tensors."""

    candidate_id: jax.Array
    family_id: jax.Array
    direction: jax.Array
    base_skeleton_id: jax.Array
    target_animal_id: jax.Array
    opening_day: jax.Array
    expansion_day: jax.Array
    final_expansion_day: jax.Array
    opening_animals: jax.Array
    expansion_animals: jax.Array
    final_animals: jax.Array
    land_start_day: jax.Array
    final_land_count: jax.Array
    hire_start_day: jax.Array
    daily_hand_target: jax.Array
    buy_batch_max: jax.Array
    feed_reserve_days: jax.Array
    cash_reserve: jax.Array
    wheat_seed_scale_pct: jax.Array
    target_sell_threshold_bp: jax.Array
    target_product_reserve: jax.Array
    extra_sell_start_day: jax.Array
    force_sell_day: jax.Array
    investment_stop_day: jax.Array
    liquidation_start_step: jax.Array
    shop_gate: jax.Array
    retry_limit: jax.Array


@dataclass(frozen=True)
class RouteGenomeSpecV1:
    candidate_id: int
    family: str
    direction: str
    base_skeleton_id: int
    target_animal_id: int
    opening_day: int
    expansion_day: int
    final_expansion_day: int
    opening_animals: int
    expansion_animals: int
    final_animals: int
    land_start_day: int
    final_land_count: int
    hire_start_day: int
    daily_hand_target: int
    buy_batch_max: int
    feed_reserve_days: int
    cash_reserve: int
    wheat_seed_scale_pct: int
    target_sell_threshold_bp: int
    target_product_reserve: int
    extra_sell_start_day: int
    force_sell_day: int
    investment_stop_day: int
    liquidation_start_step: int
    shop_gate: int
    retry_limit: int

    def receipt(self) -> dict[str, object]:
        row = asdict(self)
        row["spec_sha256"] = hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return row


class RouteGenomeCarryV1(NamedTuple):
    trace: BoatleePlayerCarryV1
    pending_valid: jax.Array
    pending_op: jax.Array
    pending_item: jax.Array
    pending_amount: jax.Array
    pending_pos: jax.Array
    pending_inventory: jax.Array
    pending_tile_kind: jax.Array
    pending_tile_crop: jax.Array
    pending_tile_animal: jax.Array
    pending_tile_yield: jax.Array
    pending_tile_flags: jax.Array
    pending_tile_fertilized: jax.Array
    retry_count: jax.Array
    confirmed_tasks: jax.Array
    retry_tasks: jax.Array
    hard_failures: jax.Array


class RouteGenomeStepDiagnosticsV1(NamedTuple):
    attempted_tasks: jax.Array
    confirmed_tasks: jax.Array
    retried_tasks: jax.Array
    hard_failures: jax.Array
    deferred_cash: jax.Array
    deferred_shop: jax.Array


_ANIMAL_COST = jnp.asarray(ANIMAL_COST, dtype=jnp.int32)
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int8)
_ANIMAL_FIRST = jnp.asarray(ANIMAL_FIRST_YIELD_DAY, dtype=jnp.int16)
_HIRE_COST = jnp.asarray(HIRE_COST, dtype=jnp.int32)
_LAND_PRICES = jnp.asarray(LAND_PRICES, dtype=jnp.int32)
_MOVE_DX = jnp.asarray((0, 0, 0, 1, -1), dtype=jnp.int8)
_MOVE_DY = jnp.asarray((0, -1, 1, 0, 0), dtype=jnp.int8)
_TARGET_SHOPS = jnp.asarray(
    (
        SHOP_NAMES.index("ICE_CREAM_SHOP"),
        SHOP_NAMES.index("PIZZA_SHOP"),
        SHOP_NAMES.index("SMOOTHIE_SHOP"),
        SHOP_NAMES.index("YARN_STORE"),
    ),
    dtype=jnp.int8,
)


def initialize_route_genome_carry_v1(batch_size: int) -> RouteGenomeCarryV1:
    units = (int(batch_size), MAX_UNITS)
    return RouteGenomeCarryV1(
        trace=initialize_trace_player_carry_v1(batch_size),
        pending_valid=jnp.zeros(units, dtype=jnp.bool_),
        pending_op=jnp.full(units, UnitOp.PASS, dtype=jnp.int8),
        pending_item=jnp.full(units, -1, dtype=jnp.int8),
        pending_amount=jnp.ones(units, dtype=jnp.int32),
        pending_pos=jnp.zeros((*units, 2), dtype=jnp.int8),
        pending_inventory=jnp.zeros(
            (*units, NUM_SHED_ITEMS), dtype=jnp.int16
        ),
        pending_tile_kind=jnp.zeros(units, dtype=jnp.int8),
        pending_tile_crop=jnp.full(units, -1, dtype=jnp.int8),
        pending_tile_animal=jnp.full(units, -1, dtype=jnp.int8),
        pending_tile_yield=jnp.zeros(units, dtype=jnp.int16),
        pending_tile_flags=jnp.zeros(units, dtype=jnp.uint8),
        pending_tile_fertilized=jnp.full(units, -1, dtype=jnp.int8),
        retry_count=jnp.zeros(units, dtype=jnp.int8),
        confirmed_tasks=jnp.zeros((batch_size,), dtype=jnp.int32),
        retry_tasks=jnp.zeros((batch_size,), dtype=jnp.int32),
        hard_failures=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


def _tile_at_positions(states: State, player: int, positions: jax.Array):
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)[:, None]
    x = jnp.clip(positions[..., 0].astype(jnp.int32), 0, 9)
    y = jnp.clip(positions[..., 1].astype(jnp.int32), 0, 9)
    return (
        states.tile_kind[batch, player, y, x],
        states.tile_crop[batch, player, y, x],
        states.tile_animal[batch, player, y, x],
        states.tile_yield[batch, player, y, x],
        states.tile_flags[batch, player, y, x],
        states.tile_fertilized_until[batch, player, y, x],
    )


def _pending_task_success_v1(
    states: State, carry: RouteGenomeCarryV1, player: int
) -> jax.Array:
    op = carry.pending_op.astype(jnp.int32)
    item = carry.pending_item.astype(jnp.int32)
    current_pos = states.unit_pos[:, player]
    current_inventory = states.unit_inventory[:, player]
    kind, crop, animal, yield_units, flags, fertilized = _tile_at_positions(
        states, player, carry.pending_pos
    )
    safe_item = jnp.clip(item, 0, NUM_SHED_ITEMS - 1)
    current_item = jnp.take_along_axis(
        current_inventory, safe_item[..., None], axis=-1
    )[..., 0]
    previous_item = jnp.take_along_axis(
        carry.pending_inventory, safe_item[..., None], axis=-1
    )[..., 0]
    previous_total = jnp.sum(carry.pending_inventory.astype(jnp.int32), axis=-1)
    current_total = jnp.sum(current_inventory.astype(jnp.int32), axis=-1)
    move = (op >= UnitOp.NORTH) & (op <= UnitOp.WEST)
    safe_move = jnp.clip(op, 0, UnitOp.WEST)
    expected = carry.pending_pos + jnp.stack(
        (_MOVE_DX[safe_move], _MOVE_DY[safe_move]), axis=-1
    )
    move_ok = jnp.all(current_pos == expected, axis=-1)
    day_boundary = (states.step[:, None] % TURNS_PER_DAY) == 0
    old_animal = jnp.clip(
        carry.pending_tile_animal.astype(jnp.int32), 0, NUM_ANIMALS - 1
    )
    harvest_product = jnp.where(
        carry.pending_tile_animal >= 0,
        _ANIMAL_PRODUCT[old_animal],
        carry.pending_tile_crop,
    ).astype(jnp.int32)
    safe_harvest = jnp.clip(harvest_product, 0, NUM_PRODUCTS - 1)
    current_harvest = jnp.take_along_axis(
        current_inventory, safe_harvest[..., None], axis=-1
    )[..., 0]
    previous_harvest = jnp.take_along_axis(
        carry.pending_inventory, safe_harvest[..., None], axis=-1
    )[..., 0]
    success = jnp.ones_like(carry.pending_valid)
    success = jnp.where(move, move_ok, success)
    success = jnp.where(
        op == UnitOp.DROP,
        (previous_total == 0) | (current_total < previous_total),
        success,
    )
    success = jnp.where(
        op == UnitOp.PICKUP, current_item > previous_item, success
    )
    success = jnp.where(
        op == UnitOp.PLACE, current_item < previous_item, success
    )
    success = jnp.where(
        op == UnitOp.PLANT,
        (kind == TileKind.PLANT) & (crop == item),
        success,
    )
    success = jnp.where(
        op == UnitOp.WATER,
        day_boundary | ((flags & jnp.uint8(FLAG_WATERED)) != 0),
        success,
    )
    success = jnp.where(
        op == UnitOp.HARVEST,
        (yield_units < carry.pending_tile_yield)
        | (current_harvest > previous_harvest),
        success,
    )
    success = jnp.where(
        op == UnitOp.FERTILIZE,
        fertilized > carry.pending_tile_fertilized,
        success,
    )
    success = jnp.where(op == UnitOp.DIG, kind == TileKind.EMPTY, success)
    success = jnp.where(
        op == UnitOp.BUILD_COOP, kind == TileKind.COOP, success
    )
    success = jnp.where(
        op == UnitOp.BUILD_PASTURE, kind == TileKind.PASTURE, success
    )
    success = jnp.where(
        op == UnitOp.FEED,
        day_boundary | ((flags & jnp.uint8(FLAG_FED)) != 0),
        success,
    )
    success = jnp.where(
        op == UnitOp.COLLECT_FERTILIZER,
        (current_inventory[..., 8] > carry.pending_inventory[..., 8])
        | ((flags & jnp.uint8(FLAG_FERTILIZER_AVAILABLE)) == 0),
        success,
    )
    success = jnp.where(
        op == UnitOp.CARE,
        day_boundary | ((flags & jnp.uint8(FLAG_CARED)) != 0),
        success,
    )
    inactive = ~states.unit_active[:, player]
    return success | inactive | (~carry.pending_valid)


def _compact_market(
    op: jax.Array,
    item: jax.Array,
    amount: jax.Array,
    count: jax.Array,
    keep: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    keep = keep & (slot < count[:, None])
    order = jnp.argsort(~keep, axis=1, stable=True)
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


def _append_market(
    op: jax.Array,
    item: jax.Array,
    amount: jax.Array,
    count: jax.Array,
    enabled: jax.Array,
    new_op: int,
    new_item: jax.Array,
    new_amount: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    batch = jnp.arange(op.shape[0], dtype=jnp.int32)
    enabled = enabled & (count < MAX_MARKET_ORDERS)
    slot = jnp.clip(count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1)
    op = op.at[batch, slot].set(jnp.where(enabled, new_op, op[batch, slot]))
    item = item.at[batch, slot].set(
        jnp.where(enabled, new_item.astype(jnp.int8), item[batch, slot])
    )
    amount = amount.at[batch, slot].set(
        jnp.where(enabled, new_amount.astype(jnp.int32), amount[batch, slot])
    )
    return op, item, amount, (count + enabled.astype(jnp.int8)).astype(jnp.int8)


def _target_shop_present(states: State, genome: RouteGenomeV1) -> jax.Array:
    active = (
        jnp.arange(states.town_shops.shape[1])[None, :]
        < states.town_count[:, None]
    )
    shops = states.town_shops
    milk = jnp.any(
        active
        & (
            (shops == _TARGET_SHOPS[0])
            | (shops == _TARGET_SHOPS[1])
            | (shops == _TARGET_SHOPS[2])
        ),
        axis=1,
    )
    wool = jnp.any(active & (shops == _TARGET_SHOPS[3]), axis=1)
    return jnp.where(genome.target_animal_id == 1, milk, wool)


def _target_animal_count(states: State, genome: RouteGenomeV1, player: int):
    target = genome.target_animal_id.astype(jnp.int8)
    placed = jnp.sum(
        states.tile_animal[:, player] == target[:, None, None], axis=(1, 2)
    ).astype(jnp.int32)
    item = NUM_PRODUCTS + target.astype(jnp.int32)
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)
    shed = states.shed[batch, player, item].astype(jnp.int32)
    carried = jnp.sum(
        states.unit_inventory[:, player, :, :]
        * jax.nn.one_hot(item, NUM_SHED_ITEMS, dtype=jnp.int16)[:, None, :],
        axis=(1, 2),
        dtype=jnp.int32,
    )
    return placed, shed, carried


def route_genome_targets_v1(
    states: State, genome: RouteGenomeV1
) -> tuple[jax.Array, jax.Array, jax.Array]:
    day = states.step.astype(jnp.int32) // TURNS_PER_DAY
    animal = jnp.where(
        day < genome.opening_day,
        0,
        jnp.where(
            day < genome.expansion_day,
            genome.opening_animals,
            jnp.where(
                day < genome.final_expansion_day,
                genome.expansion_animals,
                genome.final_animals,
            ),
        ),
    ).astype(jnp.int32)
    land = jnp.where(
        day < genome.land_start_day, 1, genome.final_land_count
    ).astype(jnp.int32)
    hands = jnp.where(
        day < genome.hire_start_day, 0, genome.daily_hand_target
    ).astype(jnp.int32)
    return animal, land, hands


def _state_driven_market_v1(
    states: State,
    tables: StaticTables,
    action: Action,
    genome: RouteGenomeV1,
    player: int,
) -> tuple[Action, jax.Array, jax.Array]:
    op, item, amount, count = (
        action.market_op,
        action.market_item,
        action.market_amount,
        action.market_count,
    )
    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    active = slot < count[:, None]
    target_animal_item = (NUM_PRODUCTS + genome.target_animal_id).astype(jnp.int8)
    target_product = _ANIMAL_PRODUCT[genome.target_animal_id.astype(jnp.int32)]
    animal_item = (item >= NUM_PRODUCTS) & (item < NUM_PRODUCTS + NUM_ANIMALS)
    animal_unit = (
        (action.unit_item >= NUM_PRODUCTS)
        & (action.unit_item < NUM_PRODUCTS + NUM_ANIMALS)
        & (
            (action.unit_op == UnitOp.PICKUP)
            | (action.unit_op == UnitOp.PLACE)
        )
    )
    unit_item = jnp.where(
        animal_unit,
        target_animal_item[:, None],
        action.unit_item,
    ).astype(jnp.int8)
    unit_op = jnp.where(
        (genome.target_animal_id[:, None] > 0)
        & (action.unit_op == UnitOp.BUILD_COOP),
        UnitOp.BUILD_PASTURE,
        action.unit_op,
    ).astype(jnp.int8)
    item = jnp.where(
        active & (op == MarketOp.BUY_ANIMAL) & animal_item,
        target_animal_item[:, None],
        item,
    ).astype(jnp.int8)
    item = jnp.where(
        active
        & (op == MarketOp.SELL)
        & (item >= PRODUCTS.index("EGG"))
        & (item <= PRODUCTS.index("WOOL")),
        target_product[:, None],
        item,
    ).astype(jnp.int8)
    wheat_seed = active & (op == MarketOp.BUY_SEED) & (item == 0)
    amount = jnp.where(
        wheat_seed,
        jnp.maximum(
            1,
            (
                amount.astype(jnp.int32)
                * genome.wheat_seed_scale_pct[:, None].astype(jnp.int32)
                + 50
            )
            // 100,
        ),
        amount,
    ).astype(jnp.int32)

    target_animals, target_land, target_hands = route_genome_targets_v1(
        states, genome
    )
    placed, shed_animals, carried_animals = _target_animal_count(
        states, genome, player
    )
    total_animals = placed + shed_animals + carried_animals
    deficit_animals = jnp.maximum(target_animals - total_animals, 0)
    current_hands = jnp.maximum(
        jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int32) - 1,
        0,
    )
    current_land = states.unlocked_count[:, player].astype(jnp.int32)
    day = states.step.astype(jnp.int32) // TURNS_PER_DAY
    investment_open = day < genome.investment_stop_day.astype(jnp.int32)
    cash_free = states.money[:, player].astype(jnp.int32) - genome.cash_reserve
    animal_cost = _ANIMAL_COST[genome.target_animal_id.astype(jnp.int32)]

    # Replace candidate-sensitive investments with state-derived orders.  Keep
    # the rest of the mature financing choreography in its original order.
    controlled = active & (
        (op == MarketOp.BUY_ANIMAL)
        | (op == MarketOp.HIRE)
        | (op == MarketOp.BUY_LAND)
    )
    op, item, amount, count = _compact_market(
        op, item, amount, count, ~controlled
    )
    deferred_cash = (
        investment_open
        & (deficit_animals > 0)
        & (cash_free < animal_cost)
    )
    animal_quantity = jnp.minimum(
        deficit_animals, genome.buy_batch_max.astype(jnp.int32)
    )
    buy_animal = (
        investment_open
        & (animal_quantity > 0)
        & (cash_free >= animal_cost * animal_quantity)
    )
    op, item, amount, count = _append_market(
        op,
        item,
        amount,
        count,
        buy_animal,
        MarketOp.BUY_ANIMAL,
        target_animal_item,
        animal_quantity,
    )

    land_deficit = jnp.maximum(target_land - current_land, 0)
    land_index = jnp.clip(current_land - 1, 0, len(LAND_PRICES) - 1)
    land_cost = _LAND_PRICES[land_index]
    buy_land = (
        investment_open
        & (land_deficit > 0)
        & (cash_free >= land_cost)
    )
    op, item, amount, count = _append_market(
        op, item, amount, count, buy_land, MarketOp.BUY_LAND,
        jnp.full_like(target_animal_item, -1), jnp.ones_like(animal_quantity)
    )

    hand_deficit = jnp.maximum(target_hands - current_hands, 0)
    safe_hire = jnp.clip(states.hires_today[:, player].astype(jnp.int32), 0, len(HIRE_COST) - 1)
    hire_cost = _HIRE_COST[safe_hire]
    hire = investment_open & (hand_deficit > 0) & (cash_free >= hire_cost)
    op, item, amount, count = _append_market(
        op, item, amount, count, hire, MarketOp.HIRE,
        jnp.full_like(target_animal_item, -1), jnp.ones_like(animal_quantity)
    )

    wheat = states.shed[:, player, 0].astype(jnp.int32) + jnp.sum(
        states.unit_inventory[:, player, :, 0].astype(jnp.int32), axis=1
    )
    feed_target = placed * genome.feed_reserve_days.astype(jnp.int32)
    feed_deficit = jnp.maximum(feed_target - wheat, 0)
    wheat_price = states.market_price[:, 0].astype(jnp.int32)
    feed_amount = jnp.minimum(feed_deficit, 16)
    buy_feed = (
        investment_open
        & (feed_amount > 0)
        & (cash_free >= wheat_price * feed_amount)
        & (jnp.sum(states.shed[:, player].astype(jnp.int32), axis=1) < SHED_CAPACITY)
    )
    op, item, amount, count = _append_market(
        op, item, amount, count, buy_feed, MarketOp.BUY_PRODUCT,
        jnp.zeros_like(target_animal_item), feed_amount
    )

    active = slot < count[:, None]
    safe_item = jnp.clip(item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    planned = jnp.zeros((states.step.shape[0], NUM_PRODUCTS), dtype=jnp.int32)
    planned = planned.at[jnp.arange(states.step.shape[0])[:, None], safe_item].add(
        jnp.where(
            active & (op == MarketOp.SELL), jnp.maximum(amount, 0), 0
        )
    )
    shed_products = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int32)
    available = jnp.maximum(shed_products - planned, 0)
    target_available = jnp.take_along_axis(
        available, target_product[:, None].astype(jnp.int32), axis=1
    )[:, 0]
    target_price = jnp.take_along_axis(
        states.market_price, target_product[:, None].astype(jnp.int32), axis=1
    )[:, 0].astype(jnp.int32)
    base_price = jnp.asarray(
        (25, 35, 60, 120, 250, 50, 160, 200, 100), dtype=jnp.int32
    )[target_product]
    quote_bp = target_price * 10_000 // base_price
    force = day >= genome.force_sell_day.astype(jnp.int32)
    pressure = jnp.sum(shed_products, axis=1) >= 80
    shop_present = _target_shop_present(states, genome)
    shop_ok = (
        (genome.shop_gate == ShopGateV1.NONE) | shop_present | force | pressure
    )
    deferred_shop = (
        (genome.shop_gate == ShopGateV1.TARGET_DEMAND_SHOP)
        & (~shop_present)
        & (target_available > 0)
        & (~force)
        & (~pressure)
    )
    reserve = jnp.where(force, 0, genome.target_product_reserve).astype(jnp.int32)
    extra_quantity = jnp.maximum(target_available - reserve, 0)
    extra_sell = (
        (day >= genome.extra_sell_start_day.astype(jnp.int32))
        & (extra_quantity > 0)
        & shop_ok
        & (
            force
            | pressure
            | (quote_bp >= genome.target_sell_threshold_bp.astype(jnp.int32))
        )
    )
    op, item, amount, count = _append_market(
        op, item, amount, count, extra_sell, MarketOp.SELL,
        target_product, extra_quantity
    )

    liquidation = states.step >= genome.liquidation_start_step.astype(jnp.int32)
    for product_id in range(NUM_PRODUCTS):
        active = slot < count[:, None]
        planned_product = jnp.sum(
            jnp.where(
                active & (op == MarketOp.SELL) & (item == product_id),
                jnp.maximum(amount, 0),
                0,
            ),
            axis=1,
        )
        quantity = jnp.maximum(
            shed_products[:, product_id] - planned_product, 0
        )
        op, item, amount, count = _append_market(
            op,
            item,
            amount,
            count,
            liquidation & (quantity > 0),
            MarketOp.SELL,
            jnp.full_like(target_product, product_id),
            quantity,
        )
    op, item, amount = _rank_sell_slots(
        states, tables, op, item, amount, count
    )
    return (
        action._replace(
            unit_op=unit_op,
            unit_item=unit_item,
            market_op=op,
            market_item=item,
            market_amount=amount,
            market_count=count,
        ),
        deferred_cash.astype(jnp.int32),
        deferred_shop.astype(jnp.int32),
    )


def route_genome_player_action_v1(
    states: State,
    tables: StaticTables,
    bank: CalibrationTraceV1,
    genome: RouteGenomeV1,
    carry: RouteGenomeCarryV1,
    player: int,
) -> tuple[Action, RouteGenomeCarryV1, RouteGenomeStepDiagnosticsV1]:
    """Compile one state-conditioned action for every genome in the batch."""

    success = _pending_task_success_v1(states, carry, player)
    failed = carry.pending_valid & (~success)
    exhausted = failed & (
        carry.retry_count >= genome.retry_limit[:, None].astype(jnp.int8)
    )
    retry = failed & (~exhausted) & states.unit_active[:, player]
    base_action, trace_carry = skeleton_player_action_v1(
        states,
        tables,
        bank,
        genome.base_skeleton_id.astype(jnp.int32),
        carry.trace,
        player,
    )
    action, deferred_cash, deferred_shop = _state_driven_market_v1(
        states, tables, base_action, genome, player
    )
    unit_op = jnp.where(retry, carry.pending_op, action.unit_op).astype(jnp.int8)
    unit_item = jnp.where(retry, carry.pending_item, action.unit_item).astype(jnp.int8)
    unit_amount = jnp.where(
        retry, carry.pending_amount, action.unit_amount
    ).astype(jnp.int32)
    action = action._replace(
        unit_op=unit_op, unit_item=unit_item, unit_amount=unit_amount
    )
    positions = states.unit_pos[:, player]
    kind, crop, animal, yield_units, flags, fertilized = _tile_at_positions(
        states, player, positions
    )
    present = (
        jnp.arange(MAX_UNITS)[None, :] < action.unit_count[:, None]
    ) & states.unit_active[:, player]
    pending = present & (unit_op != UnitOp.PASS)
    confirmed = jnp.sum(
        carry.pending_valid & success, axis=1, dtype=jnp.int32
    )
    retried = jnp.sum(retry, axis=1, dtype=jnp.int32)
    hard = jnp.sum(exhausted, axis=1, dtype=jnp.int32)
    following = RouteGenomeCarryV1(
        trace=trace_carry,
        pending_valid=pending,
        pending_op=unit_op,
        pending_item=unit_item,
        pending_amount=unit_amount,
        pending_pos=positions,
        pending_inventory=states.unit_inventory[:, player],
        pending_tile_kind=kind,
        pending_tile_crop=crop,
        pending_tile_animal=animal,
        pending_tile_yield=yield_units,
        pending_tile_flags=flags,
        pending_tile_fertilized=fertilized,
        retry_count=jnp.where(retry, carry.retry_count + 1, 0).astype(jnp.int8),
        confirmed_tasks=carry.confirmed_tasks + confirmed,
        retry_tasks=carry.retry_tasks + retried,
        hard_failures=carry.hard_failures + hard,
    )
    diagnostics = RouteGenomeStepDiagnosticsV1(
        attempted_tasks=jnp.sum(pending, axis=1, dtype=jnp.int32),
        confirmed_tasks=confirmed,
        retried_tasks=retried,
        hard_failures=hard,
        deferred_cash=deferred_cash,
        deferred_shop=deferred_shop,
    )
    return action, following, diagnostics


_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


def _vdc(index: int, base: int) -> float:
    result = 0.0
    denominator = 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        result += remainder / denominator
    return result


def _point(index: int, offset: int) -> list[float]:
    return [_vdc(index + 1 + offset, prime) for prime in _PRIMES]


def _pick(unit: float, values: Iterable[int]) -> int:
    values = tuple(values)
    return int(values[min(int(unit * len(values)), len(values) - 1)])


def generate_route_genome_specs_v1(
    manifest: dict[str, object],
    family: str,
    count: int,
    seed: int = 20260815,
) -> list[RouteGenomeSpecV1]:
    """Generate equal gold-conversion/native-expansion low-discrepancy pools."""

    if family not in ("R6_MILK", "R7_WOOL"):
        raise ValueError("route genome pilot supports R6_MILK and R7_WOOL")
    if count < 2:
        raise ValueError("count must be at least two")
    rows = list(manifest["skeletons"])
    gold = [
        int(row["combined_skeleton_id"])
        for row in rows
        if row["origin"] == "PUBLIC_GOLD_CHOREOGRAPHY"
        and str(row["artifact"]).startswith(("03_", "13_"))
    ]
    native = [
        int(row["combined_skeleton_id"])
        for row in rows
        if row["origin"] == "FAMILY_NATIVE_SEMANTIC_CHOREOGRAPHY"
        and row.get("family") == family
    ]
    if not gold or not native:
        raise RuntimeError(f"missing genome parents for {family}")
    animal = 1 if family == "R6_MILK" else 2
    family_id = RouteGenomeFamilyV1.MILK if animal == 1 else RouteGenomeFamilyV1.WOOL
    result = []
    gold_count = count // 2
    for index in range(count):
        direction = (
            RouteGenomeDirectionV1.GOLD_CONVERSION
            if index < gold_count
            else RouteGenomeDirectionV1.NATIVE_EXPANSION
        )
        local = index if index < gold_count else index - gold_count
        offset = seed + int(family_id) * 100_003 + int(direction) * 1_000_003
        u = _point(local, offset)
        pool = gold if direction == RouteGenomeDirectionV1.GOLD_CONVERSION else native
        base = pool[local % len(pool)]
        if direction == RouteGenomeDirectionV1.GOLD_CONVERSION:
            opening = _pick(u[0], range(2, 6))
            expansion = max(opening, _pick(u[1], range(4, 10)))
            final = max(expansion, _pick(u[2], range(6, 13)))
            hand_values = range(5, 13)
            cash_values = range(200, 2601, 200)
        else:
            opening = _pick(u[0], range(1, 5))
            expansion = max(opening, _pick(u[1], range(3, 9)))
            final = max(expansion, _pick(u[2], range(4, 13)))
            hand_values = range(3, 11)
            cash_values = range(100, 2001, 100)
        opening_day = _pick(u[3], range(0, 5))
        expansion_day = max(opening_day + 1, _pick(u[4], range(4, 13)))
        final_day = max(expansion_day + 1, _pick(u[5], range(9, 20)))
        force_day = _pick(u[13], range(24, 30))
        liquidation_step = _pick(u[5], range(600, 657, 8))
        invest_stop = min(
            force_day,
            liquidation_step // TURNS_PER_DAY,
            _pick(u[14], range(21, 28)),
        )
        result.append(
            RouteGenomeSpecV1(
                candidate_id=index,
                family=family,
                direction=direction.name,
                base_skeleton_id=base,
                target_animal_id=animal,
                opening_day=opening_day,
                expansion_day=expansion_day,
                final_expansion_day=final_day,
                opening_animals=opening,
                expansion_animals=expansion,
                final_animals=final,
                land_start_day=_pick(u[6], range(0, 10)),
                final_land_count=_pick(u[7], (2, 3, 4)),
                hire_start_day=_pick(u[8], range(0, 8)),
                daily_hand_target=_pick(u[9], hand_values),
                buy_batch_max=_pick(u[10], range(1, 5)),
                feed_reserve_days=_pick(u[11], range(1, 6)),
                cash_reserve=_pick(u[12], cash_values),
                wheat_seed_scale_pct=_pick(u[13], range(75, 151, 5)),
                target_sell_threshold_bp=_pick(u[14], range(750, 1401, 50)),
                target_product_reserve=_pick(u[15], range(0, 17, 2)),
                extra_sell_start_day=_pick(u[3], range(8, 23)),
                force_sell_day=force_day,
                investment_stop_day=invest_stop,
                liquidation_start_step=liquidation_step,
                shop_gate=_pick(u[6], (0, 1)),
                retry_limit=_pick(u[7], (1, 2, 3)),
            )
        )
    return result


def stack_route_genomes_v1(specs: Iterable[RouteGenomeSpecV1]) -> RouteGenomeV1:
    specs = tuple(specs)
    if not specs:
        raise ValueError("cannot stack an empty genome list")
    family = np.asarray(
        [RouteGenomeFamilyV1.MILK if row.family == "R6_MILK" else RouteGenomeFamilyV1.WOOL for row in specs],
        dtype=np.int8,
    )
    direction = np.asarray(
        [RouteGenomeDirectionV1[row.direction] for row in specs], dtype=np.int8
    )
    def values(name: str, dtype):
        return jnp.asarray([getattr(row, name) for row in specs], dtype=dtype)
    return RouteGenomeV1(
        candidate_id=values("candidate_id", jnp.int32),
        family_id=jnp.asarray(family),
        direction=jnp.asarray(direction),
        base_skeleton_id=values("base_skeleton_id", jnp.int16),
        target_animal_id=values("target_animal_id", jnp.int8),
        opening_day=values("opening_day", jnp.int8),
        expansion_day=values("expansion_day", jnp.int8),
        final_expansion_day=values("final_expansion_day", jnp.int8),
        opening_animals=values("opening_animals", jnp.int8),
        expansion_animals=values("expansion_animals", jnp.int8),
        final_animals=values("final_animals", jnp.int8),
        land_start_day=values("land_start_day", jnp.int8),
        final_land_count=values("final_land_count", jnp.int8),
        hire_start_day=values("hire_start_day", jnp.int8),
        daily_hand_target=values("daily_hand_target", jnp.int8),
        buy_batch_max=values("buy_batch_max", jnp.int8),
        feed_reserve_days=values("feed_reserve_days", jnp.int8),
        cash_reserve=values("cash_reserve", jnp.int32),
        wheat_seed_scale_pct=values("wheat_seed_scale_pct", jnp.int16),
        target_sell_threshold_bp=values("target_sell_threshold_bp", jnp.int16),
        target_product_reserve=values("target_product_reserve", jnp.int16),
        extra_sell_start_day=values("extra_sell_start_day", jnp.int8),
        force_sell_day=values("force_sell_day", jnp.int8),
        investment_stop_day=values("investment_stop_day", jnp.int8),
        liquidation_start_step=values("liquidation_start_step", jnp.int16),
        shop_gate=values("shop_gate", jnp.int8),
        retry_limit=values("retry_limit", jnp.int8),
    )


def route_genome_schedule_v1(genome: RouteGenomeV1) -> RouteScheduleV1:
    """Compile macro genome values into the state-driven Full-core schedule.

    ``base_skeleton_id`` remains provenance for the gold/native parent.  It is
    deliberately *not* replayed as a 719-step script: changing an investment
    date invalidates fixed pickup/build/place steps.  Full-core instead plans
    each worker task from the current visible state while these dense tensors
    retain one fixed JAX shape for every candidate value.
    """

    batch = genome.candidate_id.shape[0]
    day = jnp.arange(30, dtype=jnp.int16)[None, :]
    animal_target = jnp.where(
        day < genome.opening_day[:, None],
        0,
        jnp.where(
            day < genome.expansion_day[:, None],
            genome.opening_animals[:, None],
            jnp.where(
                day < genome.final_expansion_day[:, None],
                genome.expansion_animals[:, None],
                genome.final_animals[:, None],
            ),
        ),
    ).astype(jnp.int8)
    animal_by_day = jnp.zeros((batch, 30, NUM_ANIMALS), dtype=jnp.int8)
    animal_by_day = animal_by_day.at[
        jnp.arange(batch, dtype=jnp.int32)[:, None],
        jnp.arange(30, dtype=jnp.int32)[None, :],
        genome.target_animal_id.astype(jnp.int32)[:, None],
    ].set(animal_target)

    # Wheat is the feed-production backbone.  Scaling it from the current
    # animal target keeps the route self-consistent when the candidate changes
    # herd size, unlike a frozen trace whose crop workload stays unchanged.
    wheat_target = jnp.where(
        animal_target > 0,
        jnp.clip(
            (
                animal_target.astype(jnp.int32)
                * genome.wheat_seed_scale_pct[:, None].astype(jnp.int32)
                + 99
            )
            // 100,
            2,
            24,
        ),
        0,
    ).astype(jnp.int8)
    crop_by_day = jnp.zeros((batch, 30, NUM_CROPS), dtype=jnp.int8)
    crop_by_day = crop_by_day.at[:, :, 0].set(wheat_target)

    land_by_day = jnp.where(
        day < genome.land_start_day[:, None],
        1,
        genome.final_land_count[:, None],
    ).astype(jnp.int8)
    hire_by_day = jnp.where(
        day < genome.hire_start_day[:, None],
        0,
        genome.daily_hand_target[:, None],
    ).astype(jnp.int8)
    seed_batch = jnp.ones((batch, NUM_CROPS), dtype=jnp.int8)
    seed_batch = seed_batch.at[:, 0].set(
        jnp.clip(
            (genome.wheat_seed_scale_pct.astype(jnp.int32) + 24) // 25,
            2,
            8,
        ).astype(jnp.int8)
    )
    sell_interval_value = jnp.where(
        genome.retry_limit <= 1,
        4,
        jnp.where(genome.retry_limit == 2, 12, 24),
    ).astype(jnp.int16)
    sell_interval = jnp.broadcast_to(
        sell_interval_value[:, None], (batch, NUM_PRODUCTS)
    )
    sell_phase = jnp.broadcast_to(
        (genome.base_skeleton_id.astype(jnp.int16) % 24)[:, None],
        (batch, NUM_PRODUCTS),
    )
    base_price = jnp.asarray(MARKET_BASE_PRICES, dtype=jnp.int32)
    sell_price_floor = jnp.ones((batch, NUM_PRODUCTS), dtype=jnp.int16)
    target_product = _ANIMAL_PRODUCT[genome.target_animal_id.astype(jnp.int32)]
    target_floor = jnp.maximum(
        1,
        base_price[target_product]
        * genome.target_sell_threshold_bp.astype(jnp.int32)
        // 10_000,
    ).astype(jnp.int16)
    sell_price_floor = sell_price_floor.at[
        jnp.arange(batch, dtype=jnp.int32), target_product.astype(jnp.int32)
    ].set(target_floor)

    return RouteScheduleV1(
        route_id=genome.candidate_id,
        family_id=genome.family_id,
        enabled=jnp.ones((batch,), dtype=jnp.bool_),
        crop_target_by_day=crop_by_day,
        animal_target_by_day=animal_by_day,
        land_target_by_day=land_by_day,
        hire_target_by_day=hire_by_day,
        hire_batch_max=genome.buy_batch_max.astype(jnp.int8),
        seed_batch=seed_batch,
        feed_reserve_days=genome.feed_reserve_days.astype(jnp.int8),
        feed_sell_reserve_days=jnp.minimum(
            genome.target_product_reserve.astype(jnp.int32) // 4, 4
        ).astype(jnp.int8),
        fertilizer_policy=jnp.full(
            (batch,), FertilizerPolicyV1.SELL, dtype=jnp.int8
        ),
        parallel_plant_lanes=jnp.clip(
            genome.buy_batch_max, 1, 4
        ).astype(jnp.int8),
        parallel_plant_min_maintenance_code=jnp.full(
            (batch,), 3, dtype=jnp.int8
        ),
        harvest_dispatch_lanes=jnp.clip(
            genome.daily_hand_target, 2, 8
        ).astype(jnp.int8),
        harvest_dispatch_start_step=(
            genome.expansion_day.astype(jnp.int16) * TURNS_PER_DAY
        ),
        deposit_batch_units=jnp.clip(
            genome.buy_batch_max, 1, 4
        ).astype(jnp.int16),
        crop_harvest_batch_units=jnp.full(
            (batch,), 4, dtype=jnp.int16
        ),
        chain_care_after_collection=jnp.ones((batch,), dtype=jnp.bool_),
        chain_animal_service_after_action=jnp.ones(
            (batch,), dtype=jnp.bool_
        ),
        chain_crop_harvest_after_action=jnp.ones(
            (batch,), dtype=jnp.bool_
        ),
        financing_cash_floor=genome.cash_reserve.astype(jnp.int32),
        sell_interval=sell_interval,
        sell_phase=sell_phase,
        sell_price_floor=sell_price_floor,
        investment_stop_step=(
            genome.investment_stop_day.astype(jnp.int16) * TURNS_PER_DAY
        ),
        liquidation_start_step=genome.liquidation_start_step.astype(jnp.int16),
    )
