"""GPU-native calibration trace and bounded choreography mutations."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import (
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    MARKET_BASE_PRICES,
    NUM_PRODUCTS,
    PRODUCTS,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.types import Action, State, StaticTables
from strategic_v5.boatlee_v16_gpu import (
    BoatleePlayerCarryV1,
    _rank_sell_slots,
    _tile_under_units,
    _weed_repair,
)


TRACE_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "traces"
    / "rank03_main_trace_v1.npz"
)
SKELETON_BANK_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "traces"
    / "gold_skeleton_bank_v1.npz"
)


class CalibrationTraceV1(NamedTuple):
    unit_op: jax.Array
    unit_item: jax.Array
    unit_amount: jax.Array
    unit_count: jax.Array
    market_op: jax.Array
    market_item: jax.Array
    market_amount: jax.Array
    market_count: jax.Array


class DynamicSellParamsV1(NamedTuple):
    threshold_bp: jax.Array
    reserve: jax.Array
    capacity_trigger: jax.Array
    force_sale_day: jax.Array


_LIQUIDATION_ORDER = jnp.asarray(
    [
        PRODUCTS.index("CARROT"),
        PRODUCTS.index("EGG"),
        PRODUCTS.index("FERTILIZER"),
        PRODUCTS.index("MELON"),
        PRODUCTS.index("MILK"),
        PRODUCTS.index("STRAWBERRY"),
        PRODUCTS.index("TOMATO"),
        PRODUCTS.index("WHEAT"),
        PRODUCTS.index("WOOL"),
    ],
    dtype=jnp.int8,
)


def load_calibration_trace_v1(path: Path = TRACE_PATH) -> CalibrationTraceV1:
    with np.load(path, allow_pickle=False) as data:
        return CalibrationTraceV1(
            *(jnp.asarray(data[name]) for name in CalibrationTraceV1._fields)
        )


def load_skeleton_bank_v1(
    path: Path = SKELETON_BANK_PATH,
) -> CalibrationTraceV1:
    """Load candidate-major teacher traces with shape [C, 719, ...]."""

    with np.load(path, allow_pickle=False) as data:
        return CalibrationTraceV1(
            *(jnp.asarray(data[name]) for name in CalibrationTraceV1._fields)
        )


def initialize_trace_player_carry_v1(batch_size: int) -> BoatleePlayerCarryV1:
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


def _weed_repair_selected_v1(
    states: State,
    player: int,
    carry: BoatleePlayerCarryV1,
    unit_op: jax.Array,
    unit_item: jax.Array,
    unit_amount: jax.Array,
    unit_count: jax.Array,
    replay_op: jax.Array,
    replay_item: jax.Array,
    replay_amount: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, BoatleePlayerCarryV1]:
    """Gold public weed repair for a different trace in every batch row."""

    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    present = jnp.arange(MAX_UNITS)[None, :] < unit_count[:, None]
    age = step[:, None] - carry.weed_start.astype(jnp.int32)
    existing = carry.weed_active & present & (age <= 9)
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
        weed_start=jnp.where(trigger, step[:, None], carry.weed_start).astype(jnp.int16),
        weed_intended_op=jnp.where(trigger, intended_op, carry.weed_intended_op).astype(jnp.int8),
        weed_intended_item=jnp.where(trigger, intended_item, carry.weed_intended_item).astype(jnp.int8),
        weed_intended_amount=jnp.where(trigger, intended_amount, carry.weed_intended_amount).astype(jnp.int32),
    )
    return unit_op, unit_item, unit_amount, following


def skeleton_player_action_v1(
    states: State,
    tables: StaticTables,
    bank: CalibrationTraceV1,
    skeleton_id: jax.Array,
    carry: BoatleePlayerCarryV1,
    player: int,
) -> tuple[Action, BoatleePlayerCarryV1]:
    """Emit one teacher skeleton per batch row with public runtime repairs."""

    action, carry = skeleton_player_raw_action_v1(
        states, bank, skeleton_id, carry, player
    )
    market_op, market_item, market_amount = _rank_sell_slots(
        states,
        tables,
        action.market_op,
        action.market_item,
        action.market_amount,
        action.market_count,
    )
    return _append_terminal_liquidation_v1(
        states,
        action._replace(
            market_op=market_op,
            market_item=market_item,
            market_amount=market_amount,
        ),
        player,
    ), carry


def skeleton_player_raw_action_v1(
    states: State,
    bank: CalibrationTraceV1,
    skeleton_id: jax.Array,
    carry: BoatleePlayerCarryV1,
    player: int,
) -> tuple[Action, BoatleePlayerCarryV1]:
    """Emit the selected trace after WEED repair but before market controllers.

    Exact ports of Python agents need to preserve their controller order.  In
    particular PRT V6 repays borrowed sales before ranking SELL slots, then
    preempts future sales, and only then appends terminal liquidation.  The
    regular skeleton path above remains behaviourally unchanged.
    """

    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    replay_step = jnp.clip(step - 1, 0, 718)
    unit_count = jnp.sum(states.unit_active[:, player], axis=-1).astype(jnp.int8)
    present = jnp.arange(MAX_UNITS)[None, :] < unit_count[:, None]
    unit_op = jnp.where(present, bank.unit_op[skeleton_id, step], UnitOp.PASS).astype(jnp.int8)
    unit_item = jnp.where(present, bank.unit_item[skeleton_id, step], -1).astype(jnp.int8)
    unit_amount = jnp.where(present, bank.unit_amount[skeleton_id, step], 1).astype(jnp.int32)
    unit_op, unit_item, unit_amount, carry = _weed_repair_selected_v1(
        states,
        player,
        carry,
        unit_op,
        unit_item,
        unit_amount,
        unit_count,
        bank.unit_op[skeleton_id, replay_step],
        bank.unit_item[skeleton_id, replay_step],
        bank.unit_amount[skeleton_id, replay_step],
    )
    market_op = bank.market_op[skeleton_id, step]
    market_item = bank.market_item[skeleton_id, step]
    market_amount = bank.market_amount[skeleton_id, step]
    market_count = bank.market_count[skeleton_id, step]
    return Action(
        unit_op=unit_op,
        unit_item=unit_item,
        unit_amount=unit_amount,
        unit_count=unit_count,
        market_op=market_op,
        market_item=market_item,
        market_amount=market_amount,
        market_count=market_count,
    ), carry


def dynamic_seller_player_action_v1(
    states: State,
    tables: StaticTables,
    bank: CalibrationTraceV1,
    skeleton_id: jax.Array,
    carry: BoatleePlayerCarryV1,
    player: int,
    params: DynamicSellParamsV1,
) -> tuple[Action, BoatleePlayerCarryV1]:
    """Replace replay SELL timing with an actor-visible inventory/price policy."""

    action, carry = skeleton_player_action_v1(
        states, tables, bank, skeleton_id, carry, player
    )
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    keep = active & (action.market_op != MarketOp.SELL)
    order = jnp.argsort(~keep, axis=1, stable=True)
    market_op = jnp.take_along_axis(action.market_op, order, axis=1)
    market_item = jnp.take_along_axis(action.market_item, order, axis=1)
    market_amount = jnp.take_along_axis(action.market_amount, order, axis=1)
    market_count = jnp.sum(keep, axis=1).astype(jnp.int8)
    kept_slot = jnp.arange(MAX_MARKET_ORDERS)[None, :] < market_count[:, None]
    market_op = jnp.where(kept_slot, market_op, MarketOp.NONE).astype(jnp.int8)
    market_item = jnp.where(kept_slot, market_item, -1).astype(jnp.int8)
    market_amount = jnp.where(kept_slot, market_amount, 0).astype(jnp.int32)

    available = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int32)
    day = states.step.astype(jnp.int32) // 24
    terminal = states.step >= 716
    pressure = jnp.sum(available, axis=1) >= params.capacity_trigger.astype(jnp.int32)
    forced = terminal | (day >= params.force_sale_day.astype(jnp.int32))
    reserve = jnp.where(forced[:, None], 0, params.reserve.astype(jnp.int32))
    quantity = jnp.maximum(available - reserve, 0)
    base = jnp.asarray(MARKET_BASE_PRICES, dtype=jnp.int32)[None, :]
    quote_bp = (states.market_price.astype(jnp.int32) * 10_000) // base
    eligible = (
        (quantity > 0)
        & (
            (quote_bp >= params.threshold_bp.astype(jnp.int32))
            | pressure[:, None]
            | forced[:, None]
        )
    )
    score = quote_bp.astype(jnp.float32) + 2.0 * quantity.astype(jnp.float32)
    score = jnp.where(eligible, score, -jnp.inf)
    ranked = jnp.argsort(-score, axis=1, stable=True)
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)
    for ordinal in range(NUM_PRODUCTS):
        product_id = ranked[:, ordinal].astype(jnp.int32)
        can_sell = (
            jnp.take_along_axis(eligible, product_id[:, None], axis=1)[:, 0]
            & (market_count < MAX_MARKET_ORDERS)
        )
        amount = jnp.take_along_axis(quantity, product_id[:, None], axis=1)[:, 0]
        slot = jnp.clip(market_count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1)
        market_op = market_op.at[batch, slot].set(
            jnp.where(can_sell, MarketOp.SELL, market_op[batch, slot])
        )
        market_item = market_item.at[batch, slot].set(
            jnp.where(
                can_sell,
                product_id.astype(jnp.int8),
                market_item[batch, slot],
            )
        )
        market_amount = market_amount.at[batch, slot].set(
            jnp.where(can_sell, amount, market_amount[batch, slot])
        )
        market_count = (market_count + can_sell.astype(jnp.int8)).astype(jnp.int8)
    market_op, market_item, market_amount = _rank_sell_slots(
        states, tables, market_op, market_item, market_amount, market_count
    )
    return action._replace(
        market_op=market_op,
        market_item=market_item,
        market_amount=market_amount,
        market_count=market_count,
    ), carry


def anchored_extra_seller_player_action_v1(
    states: State,
    tables: StaticTables,
    bank: CalibrationTraceV1,
    skeleton_id: jax.Array,
    carry: BoatleePlayerCarryV1,
    player: int,
    params: DynamicSellParamsV1,
    min_extra_day: jax.Array,
) -> tuple[Action, BoatleePlayerCarryV1]:
    """Preserve financing SELL slots and append only optional extra sales.

    Existing SELL item/amount/position values are immutable.  This is the
    causally safe search boundary established by sales_collapse_diagnosis_v1.
    """

    action, carry = skeleton_player_action_v1(
        states, tables, bank, skeleton_id, carry, player
    )
    market_op = action.market_op
    market_item = action.market_item
    market_amount = action.market_amount
    market_count = action.market_count
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < market_count[:, None]
    safe_item = jnp.clip(market_item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    planned = jnp.zeros((states.step.shape[0], NUM_PRODUCTS), dtype=jnp.int32)
    planned = planned.at[jnp.arange(states.step.shape[0])[:, None], safe_item].add(
        jnp.where(
            active & (market_op == MarketOp.SELL),
            jnp.maximum(market_amount, 0),
            0,
        )
    )

    shed = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int32)
    available = jnp.maximum(shed - planned, 0)
    day = states.step.astype(jnp.int32) // 24
    terminal = states.step >= 716
    enabled = day >= min_extra_day.astype(jnp.int32)
    pressure = jnp.sum(shed, axis=1) >= params.capacity_trigger.astype(jnp.int32)
    forced = terminal | (day >= params.force_sale_day.astype(jnp.int32))
    reserve = jnp.where(forced[:, None], 0, params.reserve.astype(jnp.int32))
    quantity = jnp.maximum(available - reserve, 0)
    base = jnp.asarray(MARKET_BASE_PRICES, dtype=jnp.int32)[None, :]
    quote_bp = (states.market_price.astype(jnp.int32) * 10_000) // base
    eligible = (
        enabled[:, None]
        & (quantity > 0)
        & (
            (quote_bp >= params.threshold_bp.astype(jnp.int32))
            | pressure[:, None]
            | forced[:, None]
        )
    )
    score = quote_bp.astype(jnp.float32) + 2.0 * quantity.astype(jnp.float32)
    score = jnp.where(eligible, score, -jnp.inf)
    ranked = jnp.argsort(-score, axis=1, stable=True)
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)
    for ordinal in range(NUM_PRODUCTS):
        product_id = ranked[:, ordinal].astype(jnp.int32)
        can_sell = (
            jnp.take_along_axis(eligible, product_id[:, None], axis=1)[:, 0]
            & (market_count < MAX_MARKET_ORDERS)
        )
        amount = jnp.take_along_axis(quantity, product_id[:, None], axis=1)[:, 0]
        slot = jnp.clip(market_count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1)
        market_op = market_op.at[batch, slot].set(
            jnp.where(can_sell, MarketOp.SELL, market_op[batch, slot])
        )
        market_item = market_item.at[batch, slot].set(
            jnp.where(
                can_sell,
                product_id.astype(jnp.int8),
                market_item[batch, slot],
            )
        )
        market_amount = market_amount.at[batch, slot].set(
            jnp.where(can_sell, amount, market_amount[batch, slot])
        )
        market_count = (market_count + can_sell.astype(jnp.int8)).astype(jnp.int8)
    return action._replace(
        market_op=market_op,
        market_item=market_item,
        market_amount=market_amount,
        market_count=market_count,
    ), carry


def _append_terminal_liquidation_v1(
    states: State, action: Action, player: int = 0
) -> Action:
    """Append safe public-shed liquidation on the final three interpreter steps."""

    market_op = action.market_op
    market_item = action.market_item
    market_amount = action.market_amount
    market_count = action.market_count
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < market_count[:, None]
    planned = jnp.zeros((states.step.shape[0], NUM_PRODUCTS), dtype=jnp.int32)
    safe_market_item = jnp.clip(market_item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    planned = planned.at[jnp.arange(states.step.shape[0])[:, None], safe_market_item].add(
        jnp.where(active & (market_op == MarketOp.SELL), jnp.maximum(market_amount, 0), 0)
    )
    terminal = states.step >= 716
    final_step = states.step >= 718
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)
    for product_id in _LIQUIDATION_ORDER.tolist():
        available = states.shed[:, player, product_id].astype(jnp.int32)
        extra = jnp.where(final_step, available, jnp.maximum(available - planned[:, product_id], 0))
        append = terminal & (extra > 0) & (market_count < MAX_MARKET_ORDERS)
        slot = jnp.clip(market_count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1)
        market_op = market_op.at[batch, slot].set(jnp.where(append, MarketOp.SELL, market_op[batch, slot]))
        market_item = market_item.at[batch, slot].set(jnp.where(append, product_id, market_item[batch, slot]))
        market_amount = market_amount.at[batch, slot].set(jnp.where(append, extra, market_amount[batch, slot]))
        market_count = (market_count + append.astype(jnp.int8)).astype(jnp.int8)
    return action._replace(
        market_op=market_op,
        market_item=market_item,
        market_amount=market_amount,
        market_count=market_count,
    )


def calibration_player_action_v1(
    states: State,
    tables: StaticTables,
    trace: CalibrationTraceV1,
    carry: BoatleePlayerCarryV1,
    player: int,
) -> tuple[Action, BoatleePlayerCarryV1]:
    """Emit one rank03 calibration action with only public runtime repairs."""

    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    unit_count = jnp.sum(states.unit_active[:, player], axis=-1).astype(jnp.int8)
    present = jnp.arange(MAX_UNITS)[None, :] < unit_count[:, None]
    unit_op = jnp.where(present, trace.unit_op[step], UnitOp.PASS).astype(jnp.int8)
    unit_item = jnp.where(present, trace.unit_item[step], -1).astype(jnp.int8)
    unit_amount = jnp.where(present, trace.unit_amount[step], 1).astype(jnp.int32)
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
    market_op = trace.market_op[step]
    market_item = trace.market_item[step]
    market_amount = trace.market_amount[step]
    market_count = trace.market_count[step]
    market_op, market_item, market_amount = _rank_sell_slots(
        states,
        tables,
        market_op,
        market_item,
        market_amount,
        market_count,
    )
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < market_count[:, None]
    planned = jnp.zeros((states.step.shape[0], NUM_PRODUCTS), dtype=jnp.int32)
    safe_market_item = jnp.clip(
        market_item.astype(jnp.int32), 0, NUM_PRODUCTS - 1
    )
    planned = planned.at[
        jnp.arange(states.step.shape[0])[:, None], safe_market_item
    ].add(
        jnp.where(
            active & (market_op == MarketOp.SELL),
            jnp.maximum(market_amount, 0),
            0,
        )
    )
    terminal = states.step >= 716
    final_step = states.step >= 718
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)
    for product_id in _LIQUIDATION_ORDER.tolist():
        available = states.shed[:, player, product_id].astype(jnp.int32)
        extra = jnp.where(
            final_step,
            available,
            jnp.maximum(available - planned[:, product_id], 0),
        )
        append = terminal & (extra > 0) & (market_count < MAX_MARKET_ORDERS)
        slot = jnp.clip(
            market_count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1
        )
        market_op = market_op.at[batch, slot].set(
            jnp.where(append, MarketOp.SELL, market_op[batch, slot])
        )
        market_item = market_item.at[batch, slot].set(
            jnp.where(append, product_id, market_item[batch, slot])
        )
        market_amount = market_amount.at[batch, slot].set(
            jnp.where(append, extra, market_amount[batch, slot])
        )
        market_count = (market_count + append.astype(jnp.int8)).astype(jnp.int8)
    return (
        Action(
            unit_op=unit_op,
            unit_item=unit_item,
            unit_amount=unit_amount,
            unit_count=unit_count,
            market_op=market_op,
            market_item=market_item,
            market_amount=market_amount,
            market_count=market_count,
        ),
        carry,
    )


def pair_trace_with_null_v1(player_action: Action, player: int = 0) -> Action:
    """Place one player action into a two-seat tensor; the other seat passes."""

    batch_size = player_action.unit_op.shape[0]
    unit_op = jnp.full(
        (batch_size, 2, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8
    ).at[:, player].set(player_action.unit_op)
    unit_item = jnp.full(
        (batch_size, 2, MAX_UNITS), -1, dtype=jnp.int8
    ).at[:, player].set(player_action.unit_item)
    unit_amount = jnp.ones(
        (batch_size, 2, MAX_UNITS), dtype=jnp.int32
    ).at[:, player].set(player_action.unit_amount)
    unit_count = jnp.ones((batch_size, 2), dtype=jnp.int8).at[:, player].set(
        player_action.unit_count
    )
    market_op = jnp.full(
        (batch_size, 2, MAX_MARKET_ORDERS), MarketOp.NONE, dtype=jnp.int8
    ).at[:, player].set(player_action.market_op)
    market_item = jnp.full(
        (batch_size, 2, MAX_MARKET_ORDERS), -1, dtype=jnp.int8
    ).at[:, player].set(player_action.market_item)
    market_amount = jnp.zeros(
        (batch_size, 2, MAX_MARKET_ORDERS), dtype=jnp.int32
    ).at[:, player].set(player_action.market_amount)
    market_count = jnp.zeros((batch_size, 2), dtype=jnp.int8).at[:, player].set(
        player_action.market_count
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
