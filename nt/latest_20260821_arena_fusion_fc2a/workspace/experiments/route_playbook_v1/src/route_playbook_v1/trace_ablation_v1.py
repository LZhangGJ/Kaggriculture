"""Paired complete-income-engine ablations for frozen action traces."""

from __future__ import annotations

import numpy as np

from kaggriculture_jax.constants import MarketOp, UnitOp


GROUP_NAMES = (
    "FAST_CROP_WHEAT_CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG_GOOSE_CHAIN",
    "MILK_COW_CHAIN",
    "WOOL_SHEEP_CHAIN",
)
_CROP_ITEMS = ((0, 1), (2,), (3,), (4,))


def _repack_market(
    op: np.ndarray, item: np.ndarray, amount: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active = op != int(MarketOp.NONE)
    order = np.argsort(~active, axis=1, kind="stable")
    return (
        np.take_along_axis(op, order, axis=1),
        np.take_along_axis(item, order, axis=1),
        np.take_along_axis(amount, order, axis=1),
        np.sum(active, axis=1, dtype=np.int16).astype(np.int8),
    )


def ablate_income_engine_v1(
    trace: dict[str, np.ndarray], group_index: int
) -> dict[str, np.ndarray]:
    """Return a fixed-choreography intervention removing one income engine."""

    group_index = int(group_index)
    if not 0 <= group_index < len(GROUP_NAMES):
        raise ValueError(f"group_index must be in [0, {len(GROUP_NAMES) - 1}]")
    out = {name: np.array(value, copy=True) for name, value in trace.items()}
    unit_op = out["unit_op"]
    unit_item = out["unit_item"]
    market_op = out["market_op"]
    market_item = out["market_item"]
    market_amount = out["market_amount"]

    if group_index < 4:
        products = _CROP_ITEMS[group_index]
        unit_mask = (unit_op == int(UnitOp.PLANT)) & np.isin(
            unit_item, products
        )
        market_mask = (
            np.isin(market_item, products)
            & np.isin(
                market_op,
                (int(MarketOp.BUY_SEED), int(MarketOp.SELL)),
            )
        )
    else:
        animal_id = group_index - 4
        product = 5 + animal_id
        animal_item = 9 + animal_id
        structure_op = (
            int(UnitOp.BUILD_COOP)
            if animal_id == 0
            else int(UnitOp.BUILD_PASTURE)
        )
        unit_mask = (
            (unit_op == structure_op)
            | (
                np.isin(unit_op, (int(UnitOp.PICKUP), int(UnitOp.PLACE)))
                & (unit_item == animal_item)
            )
        )
        market_mask = (
            ((market_op == int(MarketOp.BUY_ANIMAL)) & (market_item == animal_item))
            | ((market_op == int(MarketOp.SELL)) & (market_item == product))
        )

    unit_op[unit_mask] = int(UnitOp.PASS)
    unit_item[unit_mask] = -1
    out["unit_amount"][unit_mask] = 1
    market_op[market_mask] = int(MarketOp.NONE)
    market_item[market_mask] = -1
    market_amount[market_mask] = 0
    market_op, market_item, market_amount, market_count = _repack_market(
        market_op, market_item, market_amount
    )
    out["unit_op"] = unit_op
    out["unit_item"] = unit_item
    out["market_op"] = market_op
    out["market_item"] = market_item
    out["market_amount"] = market_amount
    out["market_count"] = market_count
    return out
