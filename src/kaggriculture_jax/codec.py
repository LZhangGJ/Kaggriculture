"""Host-side conversion between Kaggle action dictionaries and tensor actions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from .constants import (
    ANIMALS,
    CROPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_PRODUCTS,
    PRODUCTS,
    SHED_ITEMS,
    MarketOp,
    UnitOp,
)
from .types import Action


UNIT_OPS = {name: int(value) for name, value in UnitOp.__members__.items()}
MARKET_OPS = {name: int(value) for name, value in MarketOp.__members__.items()}
SHED_ITEM_IDS = {name: i for i, name in enumerate(SHED_ITEMS)}
PRODUCT_IDS = {name: i for i, name in enumerate(PRODUCTS)}
CROP_IDS = {name: i for i, name in enumerate(CROPS)}
ANIMAL_IDS = {name: NUM_PRODUCTS + i for i, name in enumerate(ANIMALS)}


def _is_list(value: Any) -> bool:
    # The official parser requires a list, not any generic Sequence.
    return isinstance(value, list)


def _unit(action: Any) -> tuple[int, int, int]:
    if not _is_list(action) or not action:
        return int(UnitOp.PASS), -1, 1
    op_name = action[0]
    op = UNIT_OPS.get(op_name, -1)
    item = -1
    amount = 1
    if op == UnitOp.PLANT and len(action) >= 2:
        item = CROP_IDS.get(action[1], -1)
    elif op in (UnitOp.PICKUP, UnitOp.PLACE) and len(action) >= 2:
        item = SHED_ITEM_IDS.get(action[1], -1)
        if len(action) >= 3:
            try:
                amount = int(action[2])
            except (TypeError, ValueError, OverflowError):
                op = -1
                amount = 0
    return int(op), int(item), int(amount)


def _market(order: Any) -> tuple[int, int, int]:
    if not _is_list(order) or not order:
        return int(MarketOp.NONE), -1, 0
    op_name = order[0]
    op = MARKET_OPS.get(op_name, int(MarketOp.NONE))
    if op in (MarketOp.HIRE, MarketOp.BUY_LAND):
        return int(op), -1, 0
    if op not in (
        MarketOp.BUY_SEED,
        MarketOp.BUY_PRODUCT,
        MarketOp.BUY_ANIMAL,
        MarketOp.SELL,
    ):
        return int(MarketOp.NONE), -1, 0
    if len(order) < 3:
        return int(MarketOp.NONE), -1, 0
    if op == MarketOp.BUY_SEED:
        item = CROP_IDS.get(order[1], -1)
    elif op == MarketOp.BUY_ANIMAL:
        item = ANIMAL_IDS.get(order[1], -1)
    else:
        item = PRODUCT_IDS.get(order[1], -1)
    try:
        amount = int(order[2])
    except (TypeError, ValueError, OverflowError):
        return int(MarketOp.NONE), -1, 0
    return int(op), int(item), int(amount)


def encode_actions(actions: Sequence[Any]) -> Action:
    """Encode exactly two official action dictionaries into fixed tensors."""
    unit_op = np.full((2, MAX_UNITS), UnitOp.PASS, dtype=np.int8)
    unit_item = np.full((2, MAX_UNITS), -1, dtype=np.int8)
    unit_amount = np.ones((2, MAX_UNITS), dtype=np.int32)
    unit_count = np.ones((2,), dtype=np.int8)
    market_op = np.zeros((2, MAX_MARKET_ORDERS), dtype=np.int8)
    market_item = np.full((2, MAX_MARKET_ORDERS), -1, dtype=np.int8)
    market_amount = np.zeros((2, MAX_MARKET_ORDERS), dtype=np.int32)
    market_count = np.zeros((2,), dtype=np.int8)

    for player in range(2):
        raw = actions[player] if player < len(actions) else {}
        raw = raw if isinstance(raw, dict) else {}
        farmer = raw.get("farmer", ["PASS"])
        hands = raw.get("hands", [])
        hands = hands if _is_list(hands) else []
        units = [farmer, *hands]
        unit_count[player] = min(len(units), MAX_UNITS)
        for unit, unit_action in enumerate(units[:MAX_UNITS]):
            op, item, amount = _unit(unit_action)
            unit_op[player, unit] = op
            unit_item[player, unit] = item
            unit_amount[player, unit] = amount

        orders = raw.get("market", [])
        orders = orders if _is_list(orders) else []
        market_count[player] = min(len(orders), MAX_MARKET_ORDERS)
        for order_index, order in enumerate(orders[:MAX_MARKET_ORDERS]):
            op, item, amount = _market(order)
            market_op[player, order_index] = op
            market_item[player, order_index] = item
            market_amount[player, order_index] = amount

    return Action(
        unit_op=jnp.asarray(unit_op),
        unit_item=jnp.asarray(unit_item),
        unit_amount=jnp.asarray(unit_amount),
        unit_count=jnp.asarray(unit_count),
        market_op=jnp.asarray(market_op),
        market_item=jnp.asarray(market_item),
        market_amount=jnp.asarray(market_amount),
        market_count=jnp.asarray(market_count),
    )


def stack_actions(actions: Sequence[Action]) -> Action:
    return Action(*(jnp.stack(values, axis=0) for values in zip(*actions, strict=True)))
