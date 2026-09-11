"""Deterministic, family-native RouteSchedule proposal generation.

These schedules do not mutate a gold action trace.  They describe daily
production targets and executor controls directly, so tomato, strawberry,
egg, and other families receive timing compatible with their own lifecycle.
The low-discrepancy generator preserves the frozen 40/40/20 strata without
pretending that a finite sample is a mathematical global optimum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import NUM_ANIMALS, NUM_CROPS, NUM_DAYS, NUM_PRODUCTS

from .schema import FertilizerPolicyV1, RouteFamilyV1, RouteScheduleV1


FAMILY_NAMES = tuple(
    (
        "R1_FAST_CROP",
        "R2_TOMATO",
        "R3_STRAWBERRY",
        "R4_MELON",
        "R5_EGG",
        "R6_MILK",
        "R7_WOOL",
        "R8_HYBRID",
    )
)

_CROP_FAMILY_ITEM = {
    "R1_FAST_CROP": 0,
    "R2_TOMATO": 2,
    "R3_STRAWBERRY": 3,
    "R4_MELON": 4,
}
_ANIMAL_FAMILY_ITEM = {
    "R5_EGG": 0,
    "R6_MILK": 1,
    "R7_WOOL": 2,
}


@dataclass(frozen=True)
class SemanticRouteSpecV1:
    candidate_id: int
    family: str
    stratum: str
    stage_day_1: int
    stage_day_2: int
    winddown_day: int
    main_target_0: int
    main_target_1: int
    main_target_2: int
    support_wheat_0: int
    support_wheat_1: int
    support_wheat_2: int
    fast_carrot_pct: int
    land_2_day: int
    land_3_day: int
    land_4_day: int
    hire_target_0: int
    hire_target_1: int
    hire_target_2: int
    hire_batch_max: int
    seed_batch_main: int
    seed_batch_wheat: int
    feed_reserve_days: int
    feed_sell_reserve_days: int
    fertilizer_policy: int
    parallel_plant_lanes: int
    harvest_dispatch_lanes: int
    harvest_dispatch_start_day: int
    deposit_batch_units: int
    crop_harvest_batch_units: int
    chain_care_after_collection: bool
    chain_animal_service_after_action: bool
    chain_crop_harvest_after_action: bool
    financing_cash_floor: int
    sell_interval: int
    sell_phase: int
    investment_stop_day: int
    liquidation_start_day: int

    def receipt(self) -> dict[str, object]:
        row = asdict(self)
        row["spec_sha256"] = hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return row


def _van_der_corput(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


_PRIMES = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
    61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
    137, 139, 149, 151, 157, 163, 167, 173,
)


def _halton(index: int, dimensions: int, offset: int) -> list[float]:
    sample_index = index + 1 + offset
    return [
        _van_der_corput(sample_index, _PRIMES[dimension])
        for dimension in range(dimensions)
    ]


def _pick(unit: float, values: Iterable[int]) -> int:
    domain = tuple(int(value) for value in values)
    return domain[min(int(unit * len(domain)), len(domain) - 1)]


def _domains(family: str, stratum: str) -> dict[str, tuple[int, ...]]:
    crop = family in _CROP_FAMILY_ITEM
    animal = family in _ANIMAL_FAMILY_ITEM
    if not (crop or animal):
        raise ValueError(f"family-native generator does not cover {family}")

    if crop:
        anchors = {
            "R1_FAST_CROP": ((8, 12, 16), (16, 24, 32), (24, 36, 48)),
            "R2_TOMATO": ((4, 8, 12), (12, 18, 24), (18, 28, 40)),
            "R3_STRAWBERRY": ((4, 8, 12), (10, 16, 22), (16, 24, 36)),
            "R4_MELON": ((3, 6, 9), (8, 12, 16), (12, 18, 28)),
        }[family]
        broad = {
            "main0": (2, 4, 6, 8, 12, 16, 20),
            "main1": (6, 10, 14, 18, 24, 32, 40),
            "main2": (10, 16, 24, 32, 40, 52, 64),
        }
        boundary = {
            "main0": (1, 2, 20, 24),
            "main1": (4, 6, 40, 48),
            "main2": (8, 12, 64, 72),
        }
    else:
        anchors = ((1, 2, 3), (3, 4, 6), (6, 8, 10, 12))
        broad = {
            "main0": (1, 2, 3, 4),
            "main1": (2, 4, 6, 8, 10),
            "main2": (4, 6, 8, 10, 12),
        }
        boundary = {
            "main0": (1, 4),
            "main1": (2, 10, 12),
            "main2": (4, 12),
        }

    if stratum == "anchor_perturbation":
        main = {"main0": anchors[0], "main1": anchors[1], "main2": anchors[2]}
        stage1 = (3, 5, 7)
        stage2 = (8, 10, 12, 14)
        support = (0, 4, 6, 8) if crop else (0, 4, 8, 12)
        land2, land3, land4 = (4, 6, 8), (9, 11, 13, 31), (14, 17, 20, 31)
        hires = (5, 7, 9, 11, 13)
    elif stratum == "independent_stratified":
        main = broad
        stage1 = (1, 3, 5, 7, 9)
        stage2 = (7, 9, 11, 13, 15, 17)
        support = (0, 2, 4, 6, 8, 12, 16)
        land2, land3, land4 = (0, 3, 6, 9, 12, 31), (5, 8, 11, 14, 17, 31), (10, 14, 18, 22, 31)
        hires = (3, 5, 7, 9, 11, 14, 18)
    elif stratum == "boundary_extreme":
        main = boundary
        stage1 = (0, 10)
        stage2 = (6, 18)
        support = (0, 16, 24)
        land2, land3, land4 = (0, 15, 31), (5, 20, 31), (10, 24, 31)
        hires = (1, 3, 16, 24)
    else:
        raise ValueError(f"unknown stratum {stratum}")
    return {
        **main,
        "stage1": stage1,
        "stage2": stage2,
        "support": support,
        "land2": land2,
        "land3": land3,
        "land4": land4,
        "hires": hires,
    }


def generate_semantic_route_specs_v1(
    family: str,
    count: int,
    *,
    seed: int = 2026081502,
) -> list[SemanticRouteSpecV1]:
    """Generate exactly ``count`` proposals with the frozen 40/40/20 mix."""

    if family not in _CROP_FAMILY_ITEM and family not in _ANIMAL_FAMILY_ITEM:
        raise ValueError(f"unsupported family: {family}")
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    counts = (
        ("anchor_perturbation", int(count * 0.40)),
        ("independent_stratified", int(count * 0.40)),
    )
    counts += (("boundary_extreme", count - sum(value for _, value in counts)),)
    result: list[SemanticRouteSpecV1] = []
    family_offset = FAMILY_NAMES.index(family) * 100_003 + seed
    for stratum_index, (stratum, stratum_count) in enumerate(counts):
        domains = _domains(family, stratum)
        for local_index in range(stratum_count):
            u = _halton(
                local_index,
                38,
                family_offset + stratum_index * 1_000_003,
            )
            cursor = iter(u)
            take = lambda values: _pick(next(cursor), values)
            stage1 = take(domains["stage1"])
            stage2 = max(stage1 + 2, take(domains["stage2"]))
            winddown = take((24, 25, 26, 27, 28))
            main0 = take(domains["main0"])
            main1 = max(main0, take(domains["main1"]))
            main2 = max(main1, take(domains["main2"]))
            support0 = take(domains["support"])
            support1 = take(domains["support"])
            support2 = take(domains["support"])
            if family == "R1_FAST_CROP":
                support0 = support1 = support2 = 0
            land2 = take(domains["land2"])
            land3 = max(land2, take(domains["land3"])) if land2 < 31 else 31
            land4 = max(land3, take(domains["land4"])) if land3 < 31 else 31
            hire0, hire1, hire2 = (
                take(domains["hires"]),
                take(domains["hires"]),
                take(domains["hires"]),
            )
            invest_stop = take((18, 20, 22, 24, 26, 28))
            liquidation = take((26, 27, 28, 29))
            result.append(
                SemanticRouteSpecV1(
                    candidate_id=len(result),
                    family=family,
                    stratum=stratum,
                    stage_day_1=stage1,
                    stage_day_2=stage2,
                    winddown_day=max(stage2 + 2, winddown),
                    main_target_0=main0,
                    main_target_1=main1,
                    main_target_2=main2,
                    support_wheat_0=support0,
                    support_wheat_1=support1,
                    support_wheat_2=support2,
                    fast_carrot_pct=take((0, 25, 50, 75, 100)),
                    land_2_day=land2,
                    land_3_day=land3,
                    land_4_day=land4,
                    hire_target_0=hire0,
                    hire_target_1=hire1,
                    hire_target_2=hire2,
                    hire_batch_max=take((1, 2, 4, 6, 8, 10)),
                    seed_batch_main=take((1, 2, 4, 6, 8, 12, 16)),
                    seed_batch_wheat=take((1, 2, 4, 6, 8, 12)),
                    feed_reserve_days=take((1, 2, 3, 4)),
                    feed_sell_reserve_days=take((0, 1, 2, 3, 4)),
                    fertilizer_policy=take((0, 1, 2)),
                    parallel_plant_lanes=take((0, 1, 2, 3, 4)),
                    harvest_dispatch_lanes=take((0, 2, 4, 6, 9)),
                    harvest_dispatch_start_day=take((0, 4, 6, 8, 10, 12)),
                    deposit_batch_units=take((1, 2, 4, 8, 16)),
                    crop_harvest_batch_units=take((1, 2, 4, 8, 16, 32)),
                    chain_care_after_collection=bool(take((0, 1))),
                    chain_animal_service_after_action=bool(take((0, 1))),
                    chain_crop_harvest_after_action=bool(take((0, 1))),
                    financing_cash_floor=take((0, 250, 500, 1000, 2000)),
                    sell_interval=take((4, 8, 12, 24, 48)),
                    sell_phase=take((0, 3, 5, 11, 17, 23)),
                    investment_stop_day=invest_stop,
                    liquidation_start_day=max(invest_stop, liquidation),
                )
            )
    return result


def _piecewise(
    stage1: int, stage2: int, values: tuple[int, int, int]
) -> np.ndarray:
    result = np.full((NUM_DAYS,), values[0], dtype=np.int16)
    result[stage1:] = values[1]
    result[stage2:] = values[2]
    return result


def stack_semantic_route_specs_v1(
    specs: Iterable[SemanticRouteSpecV1],
) -> RouteScheduleV1:
    """Materialize candidate-major dense RouteSchedule tensors."""

    specs = list(specs)
    if not specs:
        raise ValueError("specs must be non-empty")
    batch = len(specs)
    crops = np.zeros((batch, NUM_DAYS, NUM_CROPS), dtype=np.int8)
    animals = np.zeros((batch, NUM_DAYS, NUM_ANIMALS), dtype=np.int8)
    land = np.ones((batch, NUM_DAYS), dtype=np.int8)
    hires = np.zeros((batch, NUM_DAYS), dtype=np.int8)
    seed_batch = np.ones((batch, NUM_CROPS), dtype=np.int8)
    sell_interval = np.zeros((batch, NUM_PRODUCTS), dtype=np.int16)
    sell_phase = np.zeros((batch, NUM_PRODUCTS), dtype=np.int16)
    for index, spec in enumerate(specs):
        main = _piecewise(
            spec.stage_day_1,
            spec.stage_day_2,
            (spec.main_target_0, spec.main_target_1, spec.main_target_2),
        )
        support = _piecewise(
            spec.stage_day_1,
            spec.stage_day_2,
            (
                spec.support_wheat_0,
                spec.support_wheat_1,
                spec.support_wheat_2,
            ),
        )
        if spec.family == "R1_FAST_CROP":
            carrot = np.rint(main * spec.fast_carrot_pct / 100.0).astype(np.int16)
            crops[index, :, 0] = np.clip(main - carrot, 0, 127).astype(np.int8)
            crops[index, :, 1] = np.clip(carrot, 0, 127).astype(np.int8)
        elif spec.family in _CROP_FAMILY_ITEM:
            crops[index, :, 0] = np.clip(support, 0, 127).astype(np.int8)
            crops[index, :, _CROP_FAMILY_ITEM[spec.family]] = np.clip(
                main, 0, 127
            ).astype(np.int8)
        else:
            crops[index, :, 0] = np.clip(support, 0, 127).astype(np.int8)
            animals[index, :, _ANIMAL_FAMILY_ITEM[spec.family]] = np.clip(
                main, 0, 127
            ).astype(np.int8)
        if spec.family in _CROP_FAMILY_ITEM:
            crops[index, spec.winddown_day :, :] = 0
        for day, target in (
            (spec.land_2_day, 2),
            (spec.land_3_day, 3),
            (spec.land_4_day, 4),
        ):
            if day < NUM_DAYS:
                land[index, day:] = target
        hires[index] = np.clip(
            _piecewise(
                spec.stage_day_1,
                spec.stage_day_2,
                (
                    spec.hire_target_0,
                    spec.hire_target_1,
                    spec.hire_target_2,
                ),
            ),
            0,
            32,
        ).astype(np.int8)
        main_crop = _CROP_FAMILY_ITEM.get(spec.family, 0)
        seed_batch[index, 0] = spec.seed_batch_wheat
        seed_batch[index, main_crop] = spec.seed_batch_main
        sell_interval[index] = spec.sell_interval
        sell_phase[index] = spec.sell_phase

    family_ids = np.asarray(
        [FAMILY_NAMES.index(spec.family) for spec in specs], dtype=np.int8
    )
    return RouteScheduleV1(
        route_id=jnp.asarray([spec.candidate_id for spec in specs], dtype=jnp.int32),
        family_id=jnp.asarray(family_ids),
        enabled=jnp.ones((batch,), dtype=jnp.bool_),
        crop_target_by_day=jnp.asarray(crops),
        animal_target_by_day=jnp.asarray(animals),
        land_target_by_day=jnp.asarray(land),
        hire_target_by_day=jnp.asarray(hires),
        hire_batch_max=jnp.asarray(
            [spec.hire_batch_max for spec in specs], dtype=jnp.int8
        ),
        seed_batch=jnp.asarray(seed_batch),
        feed_reserve_days=jnp.asarray(
            [spec.feed_reserve_days for spec in specs], dtype=jnp.int8
        ),
        feed_sell_reserve_days=jnp.asarray(
            [spec.feed_sell_reserve_days for spec in specs], dtype=jnp.int8
        ),
        fertilizer_policy=jnp.asarray(
            [spec.fertilizer_policy for spec in specs], dtype=jnp.int8
        ),
        parallel_plant_lanes=jnp.asarray(
            [spec.parallel_plant_lanes for spec in specs], dtype=jnp.int8
        ),
        parallel_plant_min_maintenance_code=jnp.full(
            (batch,), 3, dtype=jnp.int8
        ),
        harvest_dispatch_lanes=jnp.asarray(
            [spec.harvest_dispatch_lanes for spec in specs], dtype=jnp.int8
        ),
        harvest_dispatch_start_step=jnp.asarray(
            [spec.harvest_dispatch_start_day * 24 for spec in specs],
            dtype=jnp.int16,
        ),
        deposit_batch_units=jnp.asarray(
            [spec.deposit_batch_units for spec in specs], dtype=jnp.int16
        ),
        crop_harvest_batch_units=jnp.asarray(
            [spec.crop_harvest_batch_units for spec in specs], dtype=jnp.int16
        ),
        chain_care_after_collection=jnp.asarray(
            [spec.chain_care_after_collection for spec in specs], dtype=jnp.bool_
        ),
        chain_animal_service_after_action=jnp.asarray(
            [spec.chain_animal_service_after_action for spec in specs],
            dtype=jnp.bool_,
        ),
        chain_crop_harvest_after_action=jnp.asarray(
            [spec.chain_crop_harvest_after_action for spec in specs],
            dtype=jnp.bool_,
        ),
        financing_cash_floor=jnp.asarray(
            [spec.financing_cash_floor for spec in specs], dtype=jnp.int32
        ),
        sell_interval=jnp.asarray(sell_interval),
        sell_phase=jnp.asarray(sell_phase),
        sell_price_floor=jnp.ones((batch, NUM_PRODUCTS), dtype=jnp.int16),
        investment_stop_step=jnp.asarray(
            [spec.investment_stop_day * 24 for spec in specs], dtype=jnp.int16
        ),
        liquidation_start_step=jnp.asarray(
            [spec.liquidation_start_day * 24 for spec in specs], dtype=jnp.int16
        ),
    )
