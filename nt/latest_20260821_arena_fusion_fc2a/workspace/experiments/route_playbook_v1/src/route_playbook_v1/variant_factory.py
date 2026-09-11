"""Deterministic teacher-trace mutations for route-family potential search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

import numpy as np

from kaggriculture_jax.constants import MarketOp, UnitOp


FAMILIES = (
    "R1_FAST_CROP",
    "R2_TOMATO",
    "R3_STRAWBERRY",
    "R4_MELON",
    "R5_EGG",
    "R6_MILK",
    "R7_WOOL",
    "R8_HYBRID",
)


@dataclass(frozen=True)
class TraceVariantSpecV1:
    candidate_id: int
    family: str
    stratum: str
    base_skeleton_id: int
    crop_map: tuple[int, int, int, int, int]
    animal_mode: str
    sell_shift: int
    sell_scale_pct: int
    seed_scale_pct: int
    animal_scale_pct: int
    hire_stop_day: int
    investment_stop_day: int

    def receipt(self) -> dict[str, object]:
        row = asdict(self)
        row["crop_map"] = list(self.crop_map)
        row["spec_sha256"] = hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return row


def _crop_map(family: str, rng: np.random.Generator, boundary: bool) -> tuple[int, ...]:
    mapping = [0, 1, 2, 3, 4]
    if family == "R1_FAST_CROP":
        target = int(rng.choice([0, 1]))
        mapping[2:] = [target, target, target]
    elif family == "R2_TOMATO":
        mapping[2:] = [2, 2, 2]
    elif family == "R3_STRAWBERRY":
        mapping[2:] = [3, 3, 3]
    elif family == "R4_MELON":
        mapping[2:] = [4, 4, 4]
    elif family == "R8_HYBRID" and boundary:
        mapping[int(rng.integers(1, 5))] = int(rng.integers(0, 5))
    return tuple(mapping)


def generate_variant_specs_v1(
    manifest: dict[str, object],
    per_family: int,
    seed: int = 20260815,
) -> list[TraceVariantSpecV1]:
    """Generate the frozen 40/40/20 sampling mix for every family."""

    rng = np.random.default_rng(seed)
    skeletons = list(manifest["skeletons"])
    anchors = [
        int(row["skeleton_id"])
        for row in skeletons
        if str(row["artifact"]).startswith(("03_", "13_"))
    ]
    independent = [
        int(row["skeleton_id"])
        for row in skeletons
        if int(row["skeleton_id"]) not in anchors
    ]
    all_ids = [int(row["skeleton_id"]) for row in skeletons]
    result: list[TraceVariantSpecV1] = []
    candidate_id = 0
    for family in FAMILIES:
        anchor_count = int(round(per_family * 0.40))
        independent_count = int(round(per_family * 0.40))
        strata = (
            ["anchor_perturbation"] * anchor_count
            + ["independent_stratified"] * independent_count
            + ["boundary_extreme"] * (per_family - anchor_count - independent_count)
        )
        rng.shuffle(strata)
        for stratum in strata:
            boundary = stratum == "boundary_extreme"
            pool = anchors if stratum == "anchor_perturbation" else independent
            if boundary:
                pool = all_ids
            if family == "R5_EGG":
                animal_mode = "goose"
            elif family == "R6_MILK":
                animal_mode = "cow"
            elif family == "R7_WOOL":
                animal_mode = "sheep"
            elif family in ("R1_FAST_CROP", "R2_TOMATO", "R3_STRAWBERRY", "R4_MELON"):
                animal_mode = str(rng.choice(["identity", "none"], p=[0.7, 0.3]))
            else:
                animal_mode = str(rng.choice(["identity", "cow", "sheep"], p=[0.7, 0.15, 0.15]))
            if boundary:
                sell_shift = int(rng.choice([-48, -24, 24, 48]))
                sell_scale = int(rng.choice([40, 60, 140, 180]))
                seed_scale = int(rng.choice([40, 60, 140, 180]))
                animal_scale = int(rng.choice([50, 150]))
                hire_stop = int(rng.choice([10, 15, 20, 29]))
                investment_stop = int(rng.choice([15, 20, 25, 29]))
            else:
                sell_shift = int(rng.integers(-24, 25))
                sell_scale = int(rng.integers(70, 131))
                seed_scale = int(rng.integers(75, 126))
                animal_scale = int(rng.integers(75, 126))
                hire_stop = int(rng.integers(18, 30))
                investment_stop = int(rng.integers(22, 30))
            result.append(
                TraceVariantSpecV1(
                    candidate_id=candidate_id,
                    family=family,
                    stratum=stratum,
                    base_skeleton_id=int(rng.choice(pool)),
                    crop_map=_crop_map(family, rng, boundary),
                    animal_mode=animal_mode,
                    sell_shift=sell_shift,
                    sell_scale_pct=sell_scale,
                    seed_scale_pct=seed_scale,
                    animal_scale_pct=animal_scale,
                    hire_stop_day=hire_stop,
                    investment_stop_day=investment_stop,
                )
            )
            candidate_id += 1
    return result


def _disable_market(mask: np.ndarray, op: np.ndarray, item: np.ndarray, amount: np.ndarray) -> None:
    op[mask] = int(MarketOp.NONE)
    item[mask] = -1
    amount[mask] = 0


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


def materialize_variant_v1(
    source: dict[str, np.ndarray], spec: TraceVariantSpecV1
) -> dict[str, np.ndarray]:
    """Materialize one bounded choreography mutation from a teacher skeleton."""

    out = {name: np.array(value[spec.base_skeleton_id], copy=True) for name, value in source.items()}
    unit_op, unit_item = out["unit_op"], out["unit_item"]
    market_op, market_item, market_amount = out["market_op"], out["market_item"], out["market_amount"]

    for old, new in enumerate(spec.crop_map):
        plant = (unit_op == int(UnitOp.PLANT)) & (unit_item == old)
        unit_item[plant] = new
        seed = (market_op == int(MarketOp.BUY_SEED)) & (market_item == old)
        market_item[seed] = new
        crop_sell = (market_op == int(MarketOp.SELL)) & (market_item == old)
        market_item[crop_sell] = new

    target = {"cow": 10, "sheep": 11, "goose": 9}.get(spec.animal_mode)
    if spec.animal_mode == "none":
        _disable_market(market_op == int(MarketOp.BUY_ANIMAL), market_op, market_item, market_amount)
        animal_unit = np.isin(unit_item, [9, 10, 11]) & np.isin(unit_op, [int(UnitOp.PICKUP), int(UnitOp.PLACE)])
        unit_op[animal_unit] = int(UnitOp.PASS)
        unit_item[animal_unit] = -1
        unit_op[np.isin(unit_op, [int(UnitOp.BUILD_COOP), int(UnitOp.BUILD_PASTURE)])] = int(UnitOp.PASS)
    elif target is not None:
        animal_buy = market_op == int(MarketOp.BUY_ANIMAL)
        market_item[animal_buy] = target
        animal_unit = np.isin(unit_item, [9, 10, 11]) & np.isin(unit_op, [int(UnitOp.PICKUP), int(UnitOp.PLACE)])
        unit_item[animal_unit] = target
        if spec.animal_mode == "goose":
            unit_op[unit_op == int(UnitOp.BUILD_PASTURE)] = int(UnitOp.BUILD_COOP)
            animal_product_target = 5
        elif spec.animal_mode == "cow":
            unit_op[unit_op == int(UnitOp.BUILD_COOP)] = int(UnitOp.BUILD_PASTURE)
            animal_product_target = 6
        else:
            unit_op[unit_op == int(UnitOp.BUILD_COOP)] = int(UnitOp.BUILD_PASTURE)
            animal_product_target = 7
        animal_sell = (market_op == int(MarketOp.SELL)) & np.isin(market_item, [5, 6, 7])
        market_item[animal_sell] = animal_product_target

    seed_buy = market_op == int(MarketOp.BUY_SEED)
    animal_buy = market_op == int(MarketOp.BUY_ANIMAL)
    sell = market_op == int(MarketOp.SELL)
    market_amount[seed_buy] = np.maximum(
        1, np.rint(market_amount[seed_buy] * spec.seed_scale_pct / 100.0)
    ).astype(np.int32)
    market_amount[animal_buy] = np.maximum(
        1, np.rint(market_amount[animal_buy] * spec.animal_scale_pct / 100.0)
    ).astype(np.int32)
    market_amount[sell] = np.maximum(
        1, np.rint(market_amount[sell] * spec.sell_scale_pct / 100.0)
    ).astype(np.int32)

    day = np.arange(market_op.shape[0])[:, None] // 24
    _disable_market(
        (market_op == int(MarketOp.HIRE)) & (day >= spec.hire_stop_day),
        market_op,
        market_item,
        market_amount,
    )
    investment = np.isin(
        market_op,
        [int(MarketOp.BUY_LAND), int(MarketOp.BUY_SEED), int(MarketOp.BUY_ANIMAL)],
    ) & (day >= spec.investment_stop_day)
    _disable_market(investment, market_op, market_item, market_amount)

    if spec.sell_shift:
        shifted_op = np.array(market_op, copy=True)
        shifted_item = np.array(market_item, copy=True)
        shifted_amount = np.array(market_amount, copy=True)
        _disable_market(sell, shifted_op, shifted_item, shifted_amount)
        for step, slot in zip(*np.nonzero(sell), strict=True):
            target_step = int(np.clip(step + spec.sell_shift, 0, market_op.shape[0] - 1))
            empty = np.flatnonzero(shifted_op[target_step] == int(MarketOp.NONE))
            if empty.size:
                destination = int(empty[0])
                shifted_op[target_step, destination] = market_op[step, slot]
                shifted_item[target_step, destination] = market_item[step, slot]
                shifted_amount[target_step, destination] = market_amount[step, slot]
        market_op, market_item, market_amount = shifted_op, shifted_item, shifted_amount

    market_op, market_item, market_amount, market_count = _repack_market(
        market_op, market_item, market_amount
    )
    out["market_op"] = market_op
    out["market_item"] = market_item
    out["market_amount"] = market_amount
    out["market_count"] = market_count
    return out


def stack_variants_v1(
    source: dict[str, np.ndarray], specs: Iterable[TraceVariantSpecV1]
) -> dict[str, np.ndarray]:
    variants = [materialize_variant_v1(source, spec) for spec in specs]
    return {name: np.stack([row[name] for row in variants]) for name in variants[0]}
