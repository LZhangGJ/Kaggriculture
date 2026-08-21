"""Official-order, fixed-shape projected turn ledger for dynamic Full-core V2.

The persistent ``LedgerV1`` reserves resources across multi-turn tasks.  This
module has a different job: it projects the actions being assembled for the
*current* environment turn.  Unit actions are frozen first, then up to ten
market orders are appended and applied in order.  Candidate generation may
therefore use the latest projected cash/shed instead of the observation-start
snapshot.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from kaggriculture_jax import batched_project_unit_phase
from kaggriculture_jax.constants import (
    ANIMAL_COST,
    CROP_SEED_COST,
    HIRE_COST,
    LAND_PRICES,
    MARKET_LUT_SIZE,
    MARKET_MIN_INVENTORY,
    MAX_HANDS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    SHED_CAPACITY,
    MarketOp,
    UnitOp,
)
from kaggriculture_jax.types import Action, State, StaticTables

from .constants import TaskTypeV1
from .schema import CandidateV1, FeasibilityV1


_CROP_SEED_COST = jnp.asarray(CROP_SEED_COST, dtype=jnp.int32)
_ANIMAL_COST = jnp.asarray(ANIMAL_COST, dtype=jnp.int32)
_HIRE_COST = jnp.asarray(HIRE_COST, dtype=jnp.int32)
_LAND_PRICES = jnp.asarray(LAND_PRICES, dtype=jnp.int32)
_MAX_MARKET_ITERATIONS = 99_999


class TurnPhaseV2(IntEnum):
    UNIT = 0
    MARKET = 1
    CLOSED = 2


class TurnLedgerV2(NamedTuple):
    """A batched, actor-relative projection of one official action."""

    phase: jax.Array
    player: jax.Array
    money_nominal: jax.Array
    money_floor: jax.Array
    shed: jax.Array
    seeds: jax.Array
    unit_inventory: jax.Array
    unit_active: jax.Array
    unit_pos: jax.Array
    hires_today: jax.Array
    unlocked_count: jax.Array
    market_inventory: jax.Array
    market_price: jax.Array
    unit_op: jax.Array
    unit_item: jax.Array
    unit_amount: jax.Array
    unit_count: jax.Array
    unit_assigned: jax.Array
    market_op: jax.Array
    market_item: jax.Array
    market_amount: jax.Array
    market_count: jax.Array
    market_filled_amount: jax.Array
    market_order_guaranteed: jax.Array
    market_cash_delta: jax.Array
    invalid_order_count: jax.Array
    overflow_order_count: jax.Array
    price_lut_oob: jax.Array


class DynamicMarketMatrixV2(NamedTuple):
    """Compact 21-slot market view used inside the autoregressive hot path."""

    quantity: jax.Array
    present: jax.Array
    cash_required: jax.Array


def _actor(value: jax.Array, player: int) -> jax.Array:
    return value[:, player]


def initialize_turn_ledger_v2(states: State, player: int) -> TurnLedgerV2:
    """Initialize the current-turn projection before unit actions are frozen."""

    batch_size = states.step.shape[0]
    return TurnLedgerV2(
        phase=jnp.full((batch_size,), TurnPhaseV2.UNIT, dtype=jnp.int8),
        player=jnp.full((batch_size,), player, dtype=jnp.int8),
        money_nominal=_actor(states.money, player).astype(jnp.int32),
        money_floor=_actor(states.money, player).astype(jnp.int32),
        shed=_actor(states.shed, player).astype(jnp.int16),
        seeds=_actor(states.seeds, player).astype(jnp.int16),
        unit_inventory=_actor(states.unit_inventory, player).astype(jnp.int16),
        unit_active=_actor(states.unit_active, player),
        unit_pos=_actor(states.unit_pos, player).astype(jnp.int8),
        hires_today=_actor(states.hires_today, player).astype(jnp.int8),
        unlocked_count=_actor(states.unlocked_count, player).astype(jnp.int8),
        market_inventory=states.market_inventory.astype(jnp.int32),
        market_price=states.market_price.astype(jnp.int32),
        unit_op=jnp.full((batch_size, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8),
        unit_item=jnp.full((batch_size, MAX_UNITS), -1, dtype=jnp.int8),
        unit_amount=jnp.ones((batch_size, MAX_UNITS), dtype=jnp.int32),
        unit_count=jnp.ones((batch_size,), dtype=jnp.int8),
        unit_assigned=jnp.zeros((batch_size, MAX_UNITS), dtype=jnp.bool_),
        market_op=jnp.full(
            (batch_size, MAX_MARKET_ORDERS), MarketOp.NONE, dtype=jnp.int8
        ),
        market_item=jnp.full(
            (batch_size, MAX_MARKET_ORDERS), -1, dtype=jnp.int8
        ),
        market_amount=jnp.zeros(
            (batch_size, MAX_MARKET_ORDERS), dtype=jnp.int32
        ),
        market_count=jnp.zeros((batch_size,), dtype=jnp.int8),
        market_filled_amount=jnp.zeros(
            (batch_size, MAX_MARKET_ORDERS), dtype=jnp.int32
        ),
        market_order_guaranteed=jnp.zeros(
            (batch_size, MAX_MARKET_ORDERS), dtype=jnp.bool_
        ),
        market_cash_delta=jnp.zeros(
            (batch_size, MAX_MARKET_ORDERS), dtype=jnp.int32
        ),
        invalid_order_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        overflow_order_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        price_lut_oob=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


def close_unit_phase_v2(
    states: State, actions: Action, player: int
) -> tuple[TurnLedgerV2, State]:
    """Apply the exact official unit phase and open ordered market planning."""

    projected = batched_project_unit_phase(states, actions)
    return initialize_projected_unit_ledger_v2(projected, actions, player), projected


def initialize_projected_unit_ledger_v2(
    projected: State, actions: Action, player: int
) -> TurnLedgerV2:
    """Open market planning from an already-computed *joint* unit projection.

    ``close_unit_phase_v2`` remains the convenient standalone boundary.  A
    two-player rollout must instead combine both players' unit actions, project
    them once through the official simulator, then initialize both ledgers from
    that same state.  This avoids duplicate unit simulation and, importantly,
    lets both actors observe all public deposits/drops before market planning.
    """

    ledger = initialize_turn_ledger_v2(projected, player)
    count = _actor(actions.unit_count, player).astype(jnp.int8)
    slot = jnp.arange(MAX_UNITS, dtype=jnp.int16)[None, :]
    op = _actor(actions.unit_op, player).astype(jnp.int8)
    assigned = (slot < count[:, None]) & (op != UnitOp.PASS)
    ledger = ledger._replace(
        phase=jnp.full_like(ledger.phase, TurnPhaseV2.MARKET),
        unit_op=op,
        unit_item=_actor(actions.unit_item, player).astype(jnp.int8),
        unit_amount=_actor(actions.unit_amount, player).astype(jnp.int32),
        unit_count=count,
        unit_assigned=assigned,
    )
    return ledger


def _lookup_price_one(
    tables: StaticTables, item: jax.Array, inventory: jax.Array
) -> tuple[jax.Array, jax.Array]:
    safe_item = jnp.clip(item, 0, NUM_PRODUCTS - 1)
    index = inventory.astype(jnp.int32) - MARKET_MIN_INVENTORY
    oob = (index < 0) | (index >= MARKET_LUT_SIZE)
    safe_index = jnp.clip(index, 0, MARKET_LUT_SIZE - 1)
    return tables.market_price[safe_item, safe_index].astype(jnp.int32), oob


def _refresh_prices_one(
    tables: StaticTables, inventory: jax.Array
) -> tuple[jax.Array, jax.Array]:
    index = inventory.astype(jnp.int32) - MARKET_MIN_INVENTORY
    oob = (index < 0) | (index >= MARKET_LUT_SIZE)
    safe_index = jnp.clip(index, 0, MARKET_LUT_SIZE - 1)
    price = tables.market_price[jnp.arange(NUM_PRODUCTS), safe_index]
    return price.astype(jnp.int32), jnp.sum(oob.astype(jnp.int32), axis=-1)


def _valid_market_request_one(
    op: jax.Array, item: jax.Array, amount: jax.Array
) -> jax.Array:
    atomic = (op == MarketOp.HIRE) | (op == MarketOp.BUY_LAND)
    sell = (op == MarketOp.SELL) & (item >= 0) & (item < NUM_PRODUCTS)
    product = (op == MarketOp.BUY_PRODUCT) & ((item == 0) | (item == 8))
    seed = (op == MarketOp.BUY_SEED) & (item >= 0) & (item < NUM_CROPS)
    animal_id = item - NUM_PRODUCTS
    animal = (
        (op == MarketOp.BUY_ANIMAL)
        & (animal_id >= 0)
        & (animal_id < NUM_ANIMALS)
    )
    return (amount > 0) & (atomic | sell | product | seed | animal)


def _forward_market_price_sum_one(
    tables: StaticTables,
    item: jax.Array,
    inventory: jax.Array,
    quantity: jax.Array,
) -> jax.Array:
    """Sum clipped official quotes at inventory .. inventory+quantity-1."""

    safe_item = jnp.clip(item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    quantity = jnp.maximum(quantity.astype(jnp.int32), 0)
    raw_start = inventory.astype(jnp.int32) - MARKET_MIN_INVENTORY
    low = jnp.minimum(quantity, jnp.maximum(-raw_start, 0))
    middle_start = jnp.clip(raw_start, 0, MARKET_LUT_SIZE)
    middle = jnp.minimum(
        quantity - low, jnp.maximum(MARKET_LUT_SIZE - middle_start, 0)
    )
    middle_end = middle_start + middle
    tail = quantity - low - middle
    return (
        low * tables.market_price[safe_item, 0].astype(jnp.int32)
        + tables.market_price_prefix[safe_item, middle_end]
        - tables.market_price_prefix[safe_item, middle_start]
        + tail * tables.market_price[safe_item, -1].astype(jnp.int32)
    ).astype(jnp.int32)


def _backward_market_price_sum_one(
    tables: StaticTables,
    item: jax.Array,
    inventory: jax.Array,
    quantity: jax.Array,
) -> jax.Array:
    """Sum BUY_PRODUCT post-buy quotes inventory-1 .. inventory-quantity."""

    safe_item = jnp.clip(item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    quantity = jnp.maximum(quantity.astype(jnp.int32), 0)
    raw_end = inventory.astype(jnp.int32) - MARKET_MIN_INVENTORY
    high = jnp.minimum(quantity, jnp.maximum(raw_end - MARKET_LUT_SIZE, 0))
    middle_end = jnp.clip(raw_end, 0, MARKET_LUT_SIZE)
    middle = jnp.minimum(quantity - high, middle_end)
    middle_start = middle_end - middle
    low = quantity - high - middle
    return (
        high * tables.market_price[safe_item, -1].astype(jnp.int32)
        + tables.market_price_prefix[safe_item, middle_end]
        - tables.market_price_prefix[safe_item, middle_start]
        + low * tables.market_price[safe_item, 0].astype(jnp.int32)
    ).astype(jnp.int32)


def _apply_nonatomic_order_one(
    ledger: TurnLedgerV2,
    op: jax.Array,
    item: jax.Array,
    amount: jax.Array,
    tables: StaticTables,
) -> tuple[TurnLedgerV2, jax.Array, jax.Array, jax.Array]:
    """O(1) exact nominal settlement for one SELL/BUY order.

    Shed-limited orders can fill at most 100 units.  BUY_PRODUCT affordability
    is found by eight monotone prefix-sum bisections instead of a per-unit
    loop; all other order types have a closed-form fill count.  This preserves
    the official partial-fill and price-floor semantics.
    """

    batch = jnp.arange(ledger.phase.shape[0], dtype=jnp.int32)
    initial_money = ledger.money_nominal.astype(jnp.int32)
    request = jnp.minimum(
        jnp.maximum(amount.astype(jnp.int32), 0), _MAX_MARKET_ITERATIONS
    )
    safe_product = jnp.clip(item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    safe_crop = jnp.clip(item.astype(jnp.int32), 0, NUM_CROPS - 1)
    safe_animal = jnp.clip(
        item.astype(jnp.int32) - NUM_PRODUCTS, 0, NUM_ANIMALS - 1
    )
    safe_shed_item = jnp.clip(
        item.astype(jnp.int32), 0, NUM_SHED_ITEMS - 1
    )
    shed_total = jnp.sum(ledger.shed.astype(jnp.int32), axis=-1)
    room = jnp.maximum(SHED_CAPACITY - shed_total, 0)
    inventory = ledger.market_inventory[batch, safe_product].astype(jnp.int32)

    is_sell = op == MarketOp.SELL
    is_product = op == MarketOp.BUY_PRODUCT
    is_seed = op == MarketOp.BUY_SEED
    is_animal = op == MarketOp.BUY_ANIMAL

    sell_filled = jnp.minimum(
        request, ledger.shed[batch, safe_product].astype(jnp.int32)
    )
    floor_index = tables.market_first_floor_index[safe_product].astype(jnp.int32)
    floor_inventory = MARKET_MIN_INVENTORY + floor_index
    current_price, initial_oob = _lookup_price_one(tables, safe_product, inventory)
    sell_supply = jnp.where(
        current_price > 1,
        jnp.minimum(
            sell_filled,
            jnp.where(
                floor_index < MARKET_LUT_SIZE,
                jnp.maximum(floor_inventory - inventory, 0),
                sell_filled,
            ),
        ),
        0,
    ).astype(jnp.int32)
    sell_revenue = _forward_market_price_sum_one(
        tables, safe_product, inventory, sell_supply
    ) + (sell_filled - sell_supply)

    product_limit = jnp.minimum(request, room)

    def affordable_body(_, bounds):
        low, high = bounds
        middle = (low + high + 1) // 2
        cost = _backward_market_price_sum_one(
            tables, safe_product, inventory, middle
        )
        affordable = cost <= initial_money
        return (
            jnp.where(affordable, middle, low),
            jnp.where(affordable, high, middle - 1),
        )

    product_filled, _ = lax.fori_loop(
        0,
        8,
        affordable_body,
        (jnp.zeros_like(product_limit), product_limit.astype(jnp.int32)),
    )
    product_cost = _backward_market_price_sum_one(
        tables, safe_product, inventory, product_filled
    )

    seed_price = _CROP_SEED_COST[safe_crop].astype(jnp.int32)
    seed_filled = jnp.minimum(request, initial_money // seed_price)
    seed_cost = seed_filled * seed_price
    animal_price = _ANIMAL_COST[safe_animal].astype(jnp.int32)
    animal_filled = jnp.minimum(
        jnp.minimum(request, room), initial_money // animal_price
    )
    animal_cost = animal_filled * animal_price

    filled = jnp.where(
        is_sell,
        sell_filled,
        jnp.where(
            is_product,
            product_filled,
            jnp.where(is_seed, seed_filled, animal_filled),
        ),
    ).astype(jnp.int32)
    cash_delta = jnp.where(
        is_sell,
        sell_revenue,
        -jnp.where(
            is_product,
            product_cost,
            jnp.where(is_seed, seed_cost, animal_cost),
        ),
    ).astype(jnp.int32)
    shed_delta = jnp.where(
        is_sell,
        -sell_filled,
        jnp.where(is_product, product_filled, jnp.where(is_animal, animal_filled, 0)),
    ).astype(jnp.int16)
    market_delta = jnp.where(
        is_sell, sell_supply, jnp.where(is_product, -product_filled, 0)
    ).astype(jnp.int32)
    seed_delta = jnp.where(is_seed, seed_filled, 0).astype(jnp.int16)
    next_inventory = ledger.market_inventory.at[batch, safe_product].add(
        market_delta
    )
    price, refresh_oob = _refresh_prices_one(tables, next_inventory)
    quote_oob = (initial_oob & (filled > 0)).astype(jnp.int32)
    value = ledger._replace(
        money_nominal=initial_money + cash_delta,
        shed=ledger.shed.at[batch, safe_shed_item].add(shed_delta),
        seeds=ledger.seeds.at[batch, safe_crop].add(seed_delta),
        market_inventory=next_inventory,
        market_price=price,
        price_lut_oob=ledger.price_lut_oob + quote_oob + refresh_oob,
    )
    is_sell = op == MarketOp.SELL
    floor_after = jnp.where(
        is_sell,
        ledger.money_floor + filled,
        jnp.maximum(ledger.money_floor + jnp.minimum(cash_delta, 0), 0),
    ).astype(jnp.int32)
    value = value._replace(money_floor=floor_after)
    guaranteed = is_sell | (
        (op == MarketOp.BUY_SEED) & (ledger.money_floor >= -cash_delta)
    ) | (
        (op == MarketOp.BUY_ANIMAL) & (ledger.money_floor >= -cash_delta)
    )
    hit_cap = (
        (amount.astype(jnp.int32) > _MAX_MARKET_ITERATIONS)
        & (filled >= _MAX_MARKET_ITERATIONS)
    )
    return value, filled, guaranteed, hit_cap.astype(jnp.int32)


def _apply_atomic_order_one(
    ledger: TurnLedgerV2, op: jax.Array
) -> tuple[TurnLedgerV2, jax.Array, jax.Array]:
    """Apply one official atomic HIRE or BUY_LAND order."""

    is_hire = op == MarketOp.HIRE
    hire_index = jnp.clip(ledger.hires_today.astype(jnp.int32), 0, MAX_HANDS - 1)
    hire_cost = _HIRE_COST[hire_index]
    hire_ok = (
        is_hire
        & (ledger.hires_today.astype(jnp.int32) < MAX_HANDS)
        & (ledger.money_nominal >= hire_cost)
    )
    land_index = ledger.unlocked_count.astype(jnp.int32) - 1
    safe_land = jnp.clip(land_index, 0, len(LAND_PRICES) - 1)
    land_cost = _LAND_PRICES[safe_land]
    land_ok = (
        (op == MarketOp.BUY_LAND)
        & (land_index >= 0)
        & (land_index < len(LAND_PRICES))
        & (ledger.money_nominal >= land_cost)
    )
    ok = hire_ok | land_ok
    cost = jnp.where(hire_ok, hire_cost, jnp.where(land_ok, land_cost, 0))
    value = ledger._replace(
        money_nominal=ledger.money_nominal - cost,
        money_floor=jnp.maximum(ledger.money_floor - cost, 0),
        hires_today=ledger.hires_today + hire_ok.astype(jnp.int8),
        unlocked_count=ledger.unlocked_count + land_ok.astype(jnp.int8),
    )
    guaranteed = ok & (ledger.money_floor >= cost)
    return value, ok.astype(jnp.int32), guaranteed


def _batch_where_ledger_v2(
    condition: jax.Array, yes: TurnLedgerV2, no: TurnLedgerV2
) -> TurnLedgerV2:
    """Select ledger rows without batching immutable StaticTables."""

    def select(yes_value, no_value):
        shape = (condition.shape[0],) + (1,) * (yes_value.ndim - 1)
        return jnp.where(condition.reshape(shape), yes_value, no_value)

    return jax.tree.map(select, yes, no)


def _apply_market_order_one_v2(
    ledger: TurnLedgerV2,
    op: jax.Array,
    item: jax.Array,
    amount: jax.Array,
    tables: StaticTables,
) -> TurnLedgerV2:
    market_open = ledger.phase == TurnPhaseV2.MARKET
    has_slot = ledger.market_count.astype(jnp.int32) < MAX_MARKET_ORDERS
    request = market_open & has_slot
    valid = _valid_market_request_one(op, item, amount)
    apply = request & valid
    slot = jnp.clip(ledger.market_count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1)
    initial_money = ledger.money_nominal

    atomic = (op == MarketOp.HIRE) | (op == MarketOp.BUY_LAND)
    atomic_value, atomic_filled, atomic_guaranteed = _apply_atomic_order_one(
        ledger, op
    )
    nonatomic_value, nonatomic_filled, nonatomic_guaranteed, nonatomic_cap = (
        _apply_nonatomic_order_one(ledger, op, item, amount, tables)
    )
    committed = _batch_where_ledger_v2(atomic, atomic_value, nonatomic_value)
    value = _batch_where_ledger_v2(apply, committed, ledger)
    filled = jnp.where(
        apply, jnp.where(atomic, atomic_filled, nonatomic_filled), 0
    ).astype(jnp.int32)
    guaranteed = apply & jnp.where(
        atomic, atomic_guaranteed, nonatomic_guaranteed
    )
    cap_hit = jnp.where(apply & (~atomic), nonatomic_cap, 0).astype(jnp.int32)
    cash_delta = value.money_nominal - initial_money
    batch = jnp.arange(ledger.phase.shape[0], dtype=jnp.int32)
    value = value._replace(
        market_op=value.market_op.at[batch, slot].set(
            jnp.where(apply, op, value.market_op[batch, slot]).astype(jnp.int8)
        ),
        market_item=value.market_item.at[batch, slot].set(
            jnp.where(apply, item, value.market_item[batch, slot]).astype(jnp.int8)
        ),
        market_amount=value.market_amount.at[batch, slot].set(
            jnp.where(apply, amount, value.market_amount[batch, slot]).astype(jnp.int32)
        ),
        market_filled_amount=value.market_filled_amount.at[batch, slot].set(
            jnp.where(
                apply, filled, value.market_filled_amount[batch, slot]
            ).astype(jnp.int32)
        ),
        market_order_guaranteed=value.market_order_guaranteed.at[batch, slot].set(
            jnp.where(
                apply, guaranteed, value.market_order_guaranteed[batch, slot]
            )
        ),
        market_cash_delta=value.market_cash_delta.at[batch, slot].set(
            jnp.where(
                apply, cash_delta, value.market_cash_delta[batch, slot]
            ).astype(jnp.int32)
        ),
        market_count=(value.market_count + apply.astype(jnp.int8)).astype(jnp.int8),
        invalid_order_count=(
            value.invalid_order_count + (request & (~valid)).astype(jnp.int32) + cap_hit
        ),
        overflow_order_count=(
            value.overflow_order_count + (market_open & (~has_slot)).astype(jnp.int32)
        ),
    )
    return value


def apply_turn_market_order_v2(
    ledger: TurnLedgerV2,
    op: jax.Array | int,
    item: jax.Array | int,
    amount: jax.Array | int,
    tables: StaticTables,
) -> TurnLedgerV2:
    """Append one parameterized market order and apply its nominal effects."""

    batch_size = ledger.phase.shape[0]

    def vector(value, dtype):
        return jnp.broadcast_to(jnp.asarray(value, dtype=dtype), (batch_size,))

    return _apply_market_order_one_v2(
        ledger,
        vector(op, jnp.int8),
        vector(item, jnp.int8),
        vector(amount, jnp.int32),
        tables,
    )


def close_market_phase_v2(ledger: TurnLedgerV2) -> TurnLedgerV2:
    return ledger._replace(
        phase=jnp.where(
            ledger.phase == TurnPhaseV2.MARKET,
            TurnPhaseV2.CLOSED,
            ledger.phase,
        ).astype(jnp.int8)
    )


def projected_actor_state_v2(
    states: State, ledger: TurnLedgerV2, player: int
) -> State:
    """Overlay the current nominal ledger onto a unit-projected State.

    Board/unit-phase effects remain exactly as projected by the shared
    simulator.  Only fields affected by the ordered actor market prefix are
    replaced.  The opponent state is untouched.
    """

    return states._replace(
        money=states.money.at[:, player].set(ledger.money_nominal),
        shed=states.shed.at[:, player].set(ledger.shed),
        seeds=states.seeds.at[:, player].set(ledger.seeds),
        unit_inventory=states.unit_inventory.at[:, player].set(
            ledger.unit_inventory
        ),
        unit_active=states.unit_active.at[:, player].set(ledger.unit_active),
        unit_pos=states.unit_pos.at[:, player].set(ledger.unit_pos),
        hires_today=states.hires_today.at[:, player].set(ledger.hires_today),
        unlocked_count=states.unlocked_count.at[:, player].set(
            ledger.unlocked_count
        ),
        market_inventory=ledger.market_inventory,
        market_price=ledger.market_price,
    )


def _dynamic_buy_price_v2(
    ledger: TurnLedgerV2, tables: StaticTables, item: int
) -> jax.Array:
    inventory = ledger.market_inventory[:, item].astype(jnp.int32) - 1
    index = jnp.clip(
        inventory - MARKET_MIN_INVENTORY, 0, MARKET_LUT_SIZE - 1
    )
    return tables.market_price[item, index].astype(jnp.int32)


def _dynamic_buy_cost_v2(
    ledger: TurnLedgerV2,
    tables: StaticTables,
    item: int,
    quantity: jax.Array,
) -> jax.Array:
    """Exact sequential BUY_PRODUCT cost from the shared official price LUT."""

    quantity = jnp.maximum(quantity.astype(jnp.int32), 0)
    end = jnp.clip(
        ledger.market_inventory[:, item].astype(jnp.int32)
        - MARKET_MIN_INVENTORY,
        0,
        MARKET_LUT_SIZE,
    )
    available = jnp.minimum(quantity, end)
    start = end - available
    prefix = tables.market_price_prefix[item]
    cost = prefix[end] - prefix[start]
    tail = quantity - available
    return cost + tail * tables.market_price[item, 0].astype(jnp.int32)


_DYNAMIC_MARKET_SLOT_OP_V2 = jnp.asarray(
    (
        MarketOp.BUY_LAND,
        MarketOp.BUY_PRODUCT,
        MarketOp.BUY_PRODUCT,
        MarketOp.BUY_ANIMAL,
        MarketOp.BUY_ANIMAL,
        MarketOp.BUY_ANIMAL,
        MarketOp.BUY_SEED,
        MarketOp.BUY_SEED,
        MarketOp.BUY_SEED,
        MarketOp.BUY_SEED,
        MarketOp.BUY_SEED,
        MarketOp.HIRE,
        MarketOp.SELL,
        MarketOp.SELL,
        MarketOp.SELL,
        MarketOp.SELL,
        MarketOp.SELL,
        MarketOp.SELL,
        MarketOp.SELL,
        MarketOp.SELL,
        MarketOp.SELL,
    ),
    dtype=jnp.int8,
)
_DYNAMIC_MARKET_SLOT_ITEM_V2 = jnp.asarray(
    (-1, 8, 0, 9, 10, 11, 0, 1, 2, 3, 4, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8),
    dtype=jnp.int8,
)


def refresh_dynamic_market_matrix_v2(
    base_quantity: jax.Array,
    ledger: TurnLedgerV2,
    tables: StaticTables,
) -> DynamicMarketMatrixV2:
    """Refresh all 21 market slots as three dense matrices.

    This is equivalent to the legacy CandidateV1/FeasibilityV1 prefix refresh,
    but does not copy the unused 75 unit-route columns or unrelated feasibility
    fields on every market ordinal.
    """

    cash = ledger.money_nominal.astype(jnp.int32)
    room = jnp.maximum(
        SHED_CAPACITY - jnp.sum(ledger.shed.astype(jnp.int32), axis=-1), 0
    )
    has_slot = (
        (ledger.phase == TurnPhaseV2.MARKET)
        & (ledger.market_count.astype(jnp.int32) < MAX_MARKET_ORDERS)
    )
    requested = jnp.maximum(base_quantity[:, :21].astype(jnp.int32), 1)
    quantity = requested
    present = jnp.zeros_like(requested, dtype=jnp.bool_)

    land_index = ledger.unlocked_count.astype(jnp.int32) - 1
    safe_land = jnp.clip(land_index, 0, len(LAND_PRICES) - 1)
    land_cost = _LAND_PRICES[safe_land]
    present = present.at[:, 0].set(
        has_slot
        & (land_index >= 0)
        & (land_index < len(LAND_PRICES))
        & (cash >= land_cost)
    )
    quantity = quantity.at[:, 0].set(1)

    product_items = jnp.asarray((8, 0), dtype=jnp.int32)
    product_inventory = ledger.market_inventory[:, product_items].astype(jnp.int32) - 1
    product_index = jnp.clip(
        product_inventory - MARKET_MIN_INVENTORY, 0, MARKET_LUT_SIZE - 1
    )
    product_price = tables.market_price[
        product_items[None, :], product_index
    ].astype(jnp.int32)
    present = present.at[:, 1:3].set(
        has_slot[:, None] & (room[:, None] > 0) & (cash[:, None] >= product_price)
    )
    quantity = quantity.at[:, 1:3].set(
        jnp.minimum(requested[:, 1:3], room[:, None])
    )

    animal_cost = _ANIMAL_COST[None, :]
    present = present.at[:, 3:6].set(
        has_slot[:, None] & (room[:, None] > 0) & (cash[:, None] >= animal_cost)
    )
    quantity = quantity.at[:, 3:6].set(
        jnp.minimum(requested[:, 3:6], room[:, None])
    )
    seed_cost = _CROP_SEED_COST[None, :]
    present = present.at[:, 6:11].set(
        has_slot[:, None] & (cash[:, None] >= seed_cost)
    )

    hire_index = jnp.clip(
        ledger.hires_today.astype(jnp.int32), 0, MAX_HANDS - 1
    )
    hire_cost = _HIRE_COST[hire_index]
    present = present.at[:, 11].set(
        has_slot
        & (ledger.hires_today.astype(jnp.int32) < MAX_HANDS)
        & (cash >= hire_cost)
    )
    quantity = quantity.at[:, 11].set(1)
    sell_quantity = ledger.shed[:, :NUM_PRODUCTS].astype(jnp.int32)
    present = present.at[:, 12:21].set(
        has_slot[:, None] & (sell_quantity > 0)
    )
    quantity = quantity.at[:, 12:21].set(sell_quantity)
    quantity = jnp.where(present, quantity, 0).astype(jnp.int16)

    cash_required = jnp.zeros_like(requested, dtype=jnp.int32)
    cash_required = cash_required.at[:, 0].set(land_cost)
    cash_required = cash_required.at[:, 1].set(
        _dynamic_buy_cost_v2(ledger, tables, 8, quantity[:, 1])
    )
    cash_required = cash_required.at[:, 2].set(
        _dynamic_buy_cost_v2(ledger, tables, 0, quantity[:, 2])
    )
    cash_required = cash_required.at[:, 3:6].set(
        quantity[:, 3:6].astype(jnp.int32) * animal_cost
    )
    cash_required = cash_required.at[:, 6:11].set(
        quantity[:, 6:11].astype(jnp.int32) * seed_cost
    )
    cash_required = cash_required.at[:, 11].set(hire_cost)
    cash_required = jnp.where(present, cash_required, 0).astype(jnp.int32)
    return DynamicMarketMatrixV2(quantity, present, cash_required)


def apply_dynamic_market_matrix_slot_v2(
    ledger: TurnLedgerV2,
    market: DynamicMarketMatrixV2,
    selected: jax.Array,
    tables: StaticTables,
) -> TurnLedgerV2:
    """Apply one selected compact slot while treating STOP as a true no-op."""

    batch = jnp.arange(ledger.phase.shape[0], dtype=jnp.int32)
    safe = jnp.clip(selected.astype(jnp.int32), 0, 20)
    valid = (selected >= 0) & market.present[batch, safe]
    op = _DYNAMIC_MARKET_SLOT_OP_V2[safe]
    item = _DYNAMIC_MARKET_SLOT_ITEM_V2[safe]
    amount = market.quantity[batch, safe].astype(jnp.int32)
    updated = apply_turn_market_order_v2(ledger, op, item, amount, tables)

    def choose(new, old):
        shape = (valid.shape[0],) + (1,) * (new.ndim - 1)
        return jnp.where(valid.reshape(shape), new, old)

    return jax.tree.map(choose, updated, ledger)


def refresh_dynamic_market_candidates_v2(
    candidates: CandidateV1, ledger: TurnLedgerV2, tables: StaticTables
) -> CandidateV1:
    """Refresh fixed slots 0..20 from the latest projected market state.

    Task identity and the 96-slot schema stay unchanged.  Only quantities and
    official-now masks are replaced, so a DROP or SELL earlier in the same
    action can expose a candidate that was absent at observation start.
    """

    cash = ledger.money_nominal.astype(jnp.int32)
    room = SHED_CAPACITY - jnp.sum(ledger.shed.astype(jnp.int32), axis=-1)
    has_slot = (
        (ledger.phase == TurnPhaseV2.MARKET)
        & (ledger.market_count.astype(jnp.int32) < MAX_MARKET_ORDERS)
    )
    requested = jnp.maximum(candidates.quantity[:, :21].astype(jnp.int32), 1)
    quantity = requested
    present = jnp.zeros_like(candidates.present[:, :21])

    land_index = ledger.unlocked_count.astype(jnp.int32) - 1
    safe_land = jnp.clip(land_index, 0, len(LAND_PRICES) - 1)
    land_cost = _LAND_PRICES[safe_land]
    present = present.at[:, 0].set(
        has_slot
        & (land_index >= 0)
        & (land_index < len(LAND_PRICES))
        & (cash >= land_cost)
    )
    quantity = quantity.at[:, 0].set(1)

    for slot, item in ((1, 8), (2, 0)):
        price = _dynamic_buy_price_v2(ledger, tables, item)
        present = present.at[:, slot].set(has_slot & (room > 0) & (cash >= price))
        quantity = quantity.at[:, slot].set(jnp.minimum(requested[:, slot], jnp.maximum(room, 0)))

    for animal in range(NUM_ANIMALS):
        slot = 3 + animal
        present = present.at[:, slot].set(
            has_slot & (room > 0) & (cash >= _ANIMAL_COST[animal])
        )
        quantity = quantity.at[:, slot].set(
            jnp.minimum(requested[:, slot], jnp.maximum(room, 0))
        )

    for crop in range(NUM_CROPS):
        slot = 6 + crop
        present = present.at[:, slot].set(
            has_slot & (cash >= _CROP_SEED_COST[crop])
        )

    hire_index = jnp.clip(
        ledger.hires_today.astype(jnp.int32), 0, MAX_HANDS - 1
    )
    present = present.at[:, 11].set(
        has_slot
        & (ledger.hires_today.astype(jnp.int32) < MAX_HANDS)
        & (cash >= _HIRE_COST[hire_index])
    )
    quantity = quantity.at[:, 11].set(1)

    sell_quantity = ledger.shed[:, :NUM_PRODUCTS].astype(jnp.int32)
    present = present.at[:, 12:21].set(has_slot[:, None] & (sell_quantity > 0))
    quantity = quantity.at[:, 12:21].set(sell_quantity)
    quantity = jnp.where(present, quantity, 0).astype(jnp.int16)
    return candidates._replace(
        quantity=candidates.quantity.at[:, :21].set(quantity),
        present=candidates.present.at[:, :21].set(present),
        hard_mask=candidates.hard_mask.at[:, :21].set(present),
    )


def refresh_dynamic_market_feasibility_v2(
    feasibility: FeasibilityV1,
    candidates: CandidateV1,
    ledger: TurnLedgerV2,
    tables: StaticTables,
) -> FeasibilityV1:
    """Replace static observation-start feasibility for market slots 0..20."""

    present = candidates.present[:, :21]
    quantity = candidates.quantity[:, :21].astype(jnp.int32)
    cash_required = jnp.zeros_like(quantity)
    land_index = jnp.clip(
        ledger.unlocked_count.astype(jnp.int32) - 1, 0, len(LAND_PRICES) - 1
    )
    cash_required = cash_required.at[:, 0].set(_LAND_PRICES[land_index])
    cash_required = cash_required.at[:, 1].set(
        _dynamic_buy_cost_v2(ledger, tables, 8, quantity[:, 1])
    )
    cash_required = cash_required.at[:, 2].set(
        _dynamic_buy_cost_v2(ledger, tables, 0, quantity[:, 2])
    )
    for animal in range(NUM_ANIMALS):
        cash_required = cash_required.at[:, 3 + animal].set(
            _ANIMAL_COST[animal] * quantity[:, 3 + animal]
        )
    for crop in range(NUM_CROPS):
        cash_required = cash_required.at[:, 6 + crop].set(
            _CROP_SEED_COST[crop] * quantity[:, 6 + crop]
        )
    hire_index = jnp.clip(
        ledger.hires_today.astype(jnp.int32), 0, MAX_HANDS - 1
    )
    cash_required = cash_required.at[:, 11].set(_HIRE_COST[hire_index])
    cash_required = jnp.where(present, cash_required, 0).astype(jnp.int32)
    sell_item = jnp.broadcast_to(
        jnp.arange(NUM_PRODUCTS, dtype=jnp.int8)[None, :],
        (present.shape[0], NUM_PRODUCTS),
    )
    sell_required = candidates.quantity[:, 12:21]
    shed_in = jnp.zeros_like(quantity, dtype=jnp.int16)
    shed_in = shed_in.at[:, 1:6].set(candidates.quantity[:, 1:6])
    return feasibility._replace(
        legal_now=feasibility.legal_now.at[:, :21].set(present),
        unit_required=feasibility.unit_required.at[:, :21].set(False),
        plot_required=feasibility.plot_required.at[:, :21].set(False),
        operation_steps=feasibility.operation_steps.at[:, :21].set(
            present.astype(jnp.int16)
        ),
        bankable_before_terminal=feasibility.bankable_before_terminal.at[:, :21].set(
            present
        ),
        cash_required=feasibility.cash_required.at[:, :21].set(cash_required),
        shed_item=feasibility.shed_item.at[:, 12:21].set(sell_item),
        shed_item_required=feasibility.shed_item_required.at[:, 12:21].set(
            sell_required
        ),
        shed_reserved_in=feasibility.shed_reserved_in.at[:, :21].set(shed_in),
        market_slots_required=feasibility.market_slots_required.at[:, :21].set(
            present.astype(jnp.int8)
        ),
        land_purchases_required=feasibility.land_purchases_required.at[:, :21].set(
            jnp.zeros_like(present, dtype=jnp.int8).at[:, 0].set(
                present[:, 0].astype(jnp.int8)
            )
        ),
    )


def candidate_market_order_v2(
    candidates: CandidateV1, selected: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Map one selected fixed candidate to an official market order."""

    batch_size = candidates.present.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    valid = (selected >= 0) & (selected < 21)
    safe = jnp.clip(selected.astype(jnp.int32), 0, 20)
    task = candidates.task_type[batch, safe]
    item = candidates.item_id[batch, safe]
    quantity = candidates.quantity[batch, safe].astype(jnp.int32)
    op = jnp.where(
        task == TaskTypeV1.BUY_LAND,
        MarketOp.BUY_LAND,
        jnp.where(
            task == TaskTypeV1.BUY_PRODUCT,
            MarketOp.BUY_PRODUCT,
            jnp.where(
                task == TaskTypeV1.ANIMAL_PURCHASE,
                MarketOp.BUY_ANIMAL,
                jnp.where(
                    (task == TaskTypeV1.CROP_PRODUCTION)
                    & (candidates.owner_unit[batch, safe] < 0),
                    MarketOp.BUY_SEED,
                    jnp.where(
                        task == TaskTypeV1.HIRE_WORKER,
                        MarketOp.HIRE,
                        jnp.where(
                            (task == TaskTypeV1.SELL_INVENTORY)
                            | (task == TaskTypeV1.TERMINAL_LIQUIDATION),
                            MarketOp.SELL,
                            MarketOp.NONE,
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int8)
    valid = valid & (op != MarketOp.NONE) & candidates.present[batch, safe]
    return valid, op, item, quantity


def ordered_market_label_to_order_v2(
    selected: jax.Array, amount: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Map an ordered Replay label to an official market order.

    This mapping deliberately does not inspect candidate presence.  It is used
    by teacher-forced BC/audits *after* the current label has been evaluated,
    so only the already-consumed prefix can affect the next substep.  The
    stable 96-slot schema keeps all market identities in slots 0..20.
    """

    selected = jnp.asarray(selected, dtype=jnp.int32)
    amount = jnp.asarray(amount, dtype=jnp.int32)
    valid_slot = (selected >= 0) & (selected < 21)
    safe = jnp.clip(selected, 0, 20)
    is_atomic = (safe == 0) | (safe == 11)
    quantity = jnp.where(is_atomic, 1, amount).astype(jnp.int32)
    valid = valid_slot & (quantity > 0)

    op = jnp.where(
        safe == 0,
        MarketOp.BUY_LAND,
        jnp.where(
            (safe == 1) | (safe == 2),
            MarketOp.BUY_PRODUCT,
            jnp.where(
                (safe >= 3) & (safe <= 5),
                MarketOp.BUY_ANIMAL,
                jnp.where(
                    (safe >= 6) & (safe <= 10),
                    MarketOp.BUY_SEED,
                    jnp.where(
                        safe == 11,
                        MarketOp.HIRE,
                        MarketOp.SELL,
                    ),
                ),
            ),
        ),
    ).astype(jnp.int8)
    item = jnp.where(
        safe == 1,
        8,
        jnp.where(
            safe == 2,
            0,
            jnp.where(
                (safe >= 3) & (safe <= 5),
                NUM_PRODUCTS + safe - 3,
                jnp.where(
                    (safe >= 6) & (safe <= 10),
                    safe - 6,
                    jnp.where(safe >= 12, safe - 12, -1),
                ),
            ),
        ),
    ).astype(jnp.int8)
    return valid, op, item, quantity


def apply_dynamic_market_candidate_v2(
    ledger: TurnLedgerV2,
    candidates: CandidateV1,
    selected: jax.Array,
    tables: StaticTables,
) -> TurnLedgerV2:
    """Apply a selected dynamic market candidate without treating STOP as invalid."""

    valid, op, item, quantity = candidate_market_order_v2(candidates, selected)
    updated = apply_turn_market_order_v2(ledger, op, item, quantity, tables)

    def choose(new, old):
        shape = (valid.shape[0],) + (1,) * (new.ndim - 1)
        return jnp.where(valid.reshape(shape), new, old)

    return jax.tree.map(choose, updated, ledger)


def ledger_action_v2(ledger: TurnLedgerV2, player: int) -> Action:
    """Materialize a two-player Action with the ledger actor on one seat."""

    batch_size = ledger.phase.shape[0]
    unit_op = jnp.full(
        (batch_size, 2, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8
    ).at[:, player].set(ledger.unit_op)
    unit_item = jnp.full(
        (batch_size, 2, MAX_UNITS), -1, dtype=jnp.int8
    ).at[:, player].set(ledger.unit_item)
    unit_amount = jnp.ones(
        (batch_size, 2, MAX_UNITS), dtype=jnp.int32
    ).at[:, player].set(ledger.unit_amount)
    unit_count = jnp.ones((batch_size, 2), dtype=jnp.int8).at[:, player].set(
        ledger.unit_count
    )
    market_op = jnp.full(
        (batch_size, 2, MAX_MARKET_ORDERS), MarketOp.NONE, dtype=jnp.int8
    ).at[:, player].set(ledger.market_op)
    market_item = jnp.full(
        (batch_size, 2, MAX_MARKET_ORDERS), -1, dtype=jnp.int8
    ).at[:, player].set(ledger.market_item)
    market_amount = jnp.zeros(
        (batch_size, 2, MAX_MARKET_ORDERS), dtype=jnp.int32
    ).at[:, player].set(ledger.market_amount)
    market_count = jnp.zeros((batch_size, 2), dtype=jnp.int8).at[:, player].set(
        ledger.market_count
    )
    return Action(
        unit_op,
        unit_item,
        unit_amount,
        unit_count,
        market_op,
        market_item,
        market_amount,
        market_count,
    )
