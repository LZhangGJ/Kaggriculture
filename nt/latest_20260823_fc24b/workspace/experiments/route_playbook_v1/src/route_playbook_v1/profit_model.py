"""Day-level mixed-integer profit model used only to propose route skeletons.

The exact game remains the acceptance oracle.  This model deliberately relaxes
pathing, within-day order, random shops, weeds, storage, and endogenous prices.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from kaggriculture_jax.constants import (
    ANIMAL_COST,
    ANIMAL_FIRST_YIELD_DAY,
    ANIMAL_INTERVAL,
    ANIMAL_MAX_HELD,
    ANIMAL_PRODUCT,
    CROP_FIRST_YIELD_DAY,
    CROP_INTERVAL,
    CROP_MAX_YIELD,
    CROP_MAX_YIELD_DAY,
    CROP_ONGOING,
    CROP_SEED_COST,
    HIRE_COST,
    LAND_PRICES,
    MARKET_BASE_PRICES,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_DAYS,
)


FAMILY_DOMAINS = {
    "R1_FAST_CROP": {"crops": (0, 1), "animals": (), "minimum": ("crop", 0, 10)},
    "R2_TOMATO": {"crops": (0, 2), "animals": (), "minimum": ("crop", 2, 8)},
    "R3_STRAWBERRY": {"crops": (0, 3), "animals": (), "minimum": ("crop", 3, 8)},
    "R4_MELON": {"crops": (0, 4), "animals": (), "minimum": ("crop", 4, 8)},
    "R5_EGG": {"crops": (0, 4), "animals": (0,), "minimum": ("animal", 0, 2)},
    "R6_MILK": {"crops": (0, 4), "animals": (1,), "minimum": ("animal", 1, 2)},
    "R7_WOOL": {"crops": (0, 4), "animals": (2,), "minimum": ("animal", 2, 2)},
    "R8_HYBRID": {"crops": (0, 3, 4), "animals": (1, 2), "minimum": ("animal", 1, 4)},
}


@dataclass(frozen=True)
class MacroProfitConfigV1:
    price_factor: float = 0.85
    fertilizer_price_factor: float = 0.75
    action_multiplier: float = 2.20
    maximum_daily_hires: int = 14
    initial_cash: int = 3000
    crop_batch_upper_bound: int = 25
    animal_batch_upper_bound: int = 16
    time_limit_seconds: float = 30.0


@dataclass(frozen=True)
class _Variable:
    name: str
    kind: str
    item: int
    day: int
    tier: int
    upper: float
    integer: int
    cash_delta: np.ndarray
    actions: np.ndarray
    occupancy: np.ndarray
    crop_output: np.ndarray
    product_output: np.ndarray


def _crop_due_days(crop: int, start: int) -> tuple[int, ...]:
    first = start + int(CROP_FIRST_YIELD_DAY[crop])
    if not CROP_ONGOING[crop]:
        return (first,)
    interval = int(CROP_INTERVAL[crop])
    return tuple(first + index * interval for index in range(int(CROP_MAX_YIELD[crop])))


def _crop_first_harvest_units(crop: int) -> int:
    if CROP_ONGOING[crop]:
        return 1
    first = int(CROP_FIRST_YIELD_DAY[crop])
    max_day = int(CROP_FIRST_YIELD_DAY[crop])
    # Non-ongoing plants start with one unit.  Daily watering begins adding
    # yield in the second half of the official lifespan.  At first maturity
    # this is a conservative executable yield for the current executor, which
    # harvests as soon as a plant is mature.
    window_start = (int(CROP_MAX_YIELD_DAY[crop]) + 1) // 2
    bonus_days = max(0, max_day - window_start + 1)
    return min(int(CROP_MAX_YIELD[crop]), 1 + bonus_days)


def _animal_due_days(animal: int, start: int) -> tuple[int, ...]:
    first = start + int(ANIMAL_FIRST_YIELD_DAY[animal])
    interval = int(ANIMAL_INTERVAL[animal])
    if first >= NUM_DAYS:
        return ()
    return tuple(range(first, NUM_DAYS, interval))


def _animal_units(animal: int, due_index: int) -> int:
    if due_index == 0:
        cared_days = int(ANIMAL_FIRST_YIELD_DAY[animal])
    else:
        cared_days = int(ANIMAL_INTERVAL[animal])
    return min(int(ANIMAL_MAX_HELD[animal]), 1 + cared_days)


def _new_vector() -> np.ndarray:
    return np.zeros((NUM_DAYS,), dtype=np.float64)


def _build_variables(
    family: str, config: MacroProfitConfigV1
) -> list[_Variable]:
    domain = FAMILY_DOMAINS[family]
    variables: list[_Variable] = []
    prices = np.asarray(MARKET_BASE_PRICES, dtype=np.float64) * config.price_factor
    prices[8] = MARKET_BASE_PRICES[8] * config.fertilizer_price_factor

    for crop in domain["crops"]:
        for start in range(NUM_DAYS):
            due = _crop_due_days(int(crop), start)
            if not due or due[-1] >= NUM_DAYS:
                continue
            end = due[-1]
            cash = _new_vector()
            actions = _new_vector()
            occupancy = _new_vector()
            crop_output = np.zeros((NUM_DAYS, NUM_CROPS), dtype=np.float64)
            product_output = np.zeros((NUM_DAYS, len(MARKET_BASE_PRICES)), dtype=np.float64)
            cash[start] += CROP_SEED_COST[int(crop)]
            actions[start] += 1.0
            actions[start : end + 1] += 1.0
            occupancy[start : end + 1] = 1.0
            for due_index, day in enumerate(due):
                units = 1 if CROP_ONGOING[int(crop)] else _crop_first_harvest_units(int(crop))
                actions[day] += 1.0
                cash[day] -= units * prices[int(crop)]
                crop_output[day, int(crop)] += units
                product_output[day, int(crop)] += units
            variables.append(
                _Variable(
                    name=f"crop_{crop}_start_{start}",
                    kind="crop",
                    item=int(crop),
                    day=start,
                    tier=-1,
                    upper=float(config.crop_batch_upper_bound),
                    integer=1,
                    cash_delta=cash,
                    actions=actions * config.action_multiplier,
                    occupancy=occupancy,
                    crop_output=crop_output,
                    product_output=product_output,
                )
            )

    wheat_price = float(prices[0])
    for animal in domain["animals"]:
        for start in range(NUM_DAYS):
            due = _animal_due_days(int(animal), start)
            if not due:
                continue
            cash = _new_vector()
            actions = _new_vector()
            occupancy = _new_vector()
            crop_output = np.zeros((NUM_DAYS, NUM_CROPS), dtype=np.float64)
            product_output = np.zeros((NUM_DAYS, len(MARKET_BASE_PRICES)), dtype=np.float64)
            cash[start] += ANIMAL_COST[int(animal)]
            cash[start:] += wheat_price
            occupancy[start:] = 1.0
            actions[start] += 3.0  # build, carry/pickup, place
            actions[start:] += 2.0  # feed and care
            if start + 1 < NUM_DAYS:
                actions[start + 1 :] += 1.0
                cash[start + 1 :] -= prices[8]
                product_output[start + 1 :, 8] += 1.0
            for due_index, day in enumerate(due):
                units = _animal_units(int(animal), due_index)
                product = int(ANIMAL_PRODUCT[int(animal)])
                actions[day] += 1.0
                cash[day] -= units * prices[product]
                product_output[day, product] += units
            variables.append(
                _Variable(
                    name=f"animal_{animal}_start_{start}",
                    kind="animal",
                    item=int(animal),
                    day=start,
                    tier=-1,
                    upper=float(config.animal_batch_upper_bound),
                    integer=1,
                    cash_delta=cash,
                    actions=actions * config.action_multiplier,
                    occupancy=occupancy,
                    crop_output=crop_output,
                    product_output=product_output,
                )
            )

    zero_outputs = np.zeros((NUM_DAYS, len(MARKET_BASE_PRICES)), dtype=np.float64)
    zero_crops = np.zeros((NUM_DAYS, NUM_CROPS), dtype=np.float64)
    for day in range(NUM_DAYS):
        for tier in range(config.maximum_daily_hires):
            cash = _new_vector()
            cash[day] = HIRE_COST[tier]
            variables.append(
                _Variable(
                    name=f"hire_day_{day}_tier_{tier}",
                    kind="hire",
                    item=-1,
                    day=day,
                    tier=tier,
                    upper=1.0,
                    integer=1,
                    cash_delta=cash,
                    actions=_new_vector(),
                    occupancy=_new_vector(),
                    crop_output=zero_crops,
                    product_output=zero_outputs,
                )
            )
    for day in range(NUM_DAYS):
        for tier in range(3):
            cash = _new_vector()
            cash[day] = LAND_PRICES[tier]
            variables.append(
                _Variable(
                    name=f"land_day_{day}_tier_{tier}",
                    kind="land",
                    item=-1,
                    day=day,
                    tier=tier,
                    upper=1.0,
                    integer=1,
                    cash_delta=cash,
                    actions=_new_vector(),
                    occupancy=_new_vector(),
                    crop_output=zero_crops,
                    product_output=zero_outputs,
                )
            )
    return variables


def _row(
    entries: Iterable[tuple[int, float]],
    row_index: int,
    row_ids: list[int],
    col_ids: list[int],
    values: list[float],
) -> None:
    for column, value in entries:
        if value:
            row_ids.append(row_index)
            col_ids.append(column)
            values.append(float(value))


def solve_macro_profit_v1(
    family: str, config: MacroProfitConfigV1 = MacroProfitConfigV1()
) -> dict[str, object]:
    """Solve one relaxed family model and return a JSON-safe route proposal."""

    if family not in FAMILY_DOMAINS:
        raise ValueError(f"unknown family: {family}")
    variables = _build_variables(family, config)
    count = len(variables)
    c = np.asarray([np.sum(variable.cash_delta) for variable in variables])
    lower = np.zeros((count,), dtype=np.float64)
    upper = np.asarray([variable.upper for variable in variables])
    integrality = np.asarray([variable.integer for variable in variables], dtype=np.uint8)
    row_ids: list[int] = []
    col_ids: list[int] = []
    values: list[float] = []
    constraint_lb: list[float] = []
    constraint_ub: list[float] = []

    def add(entries: Iterable[tuple[int, float]], lb: float, ub: float) -> None:
        index = len(constraint_lb)
        _row(entries, index, row_ids, col_ids, values)
        constraint_lb.append(lb)
        constraint_ub.append(ub)

    # Cash may be recycled after same-day production, but may never become
    # negative at a day boundary.
    for day in range(NUM_DAYS):
        add(
            (
                (index, float(np.sum(variable.cash_delta[: day + 1])))
                for index, variable in enumerate(variables)
            ),
            -np.inf,
            float(config.initial_cash),
        )

    # Daily unit-action capacity.  One base farmer supplies 24 slots; each
    # selected daily hire tier contributes another 24.
    for day in range(NUM_DAYS):
        add(
            (
                (
                    index,
                    float(variable.actions[day])
                    - (
                        24.0
                        if variable.kind == "hire" and variable.day == day
                        else 0.0
                    ),
                )
                for index, variable in enumerate(variables)
            ),
            -np.inf,
            24.0,
        )

    # One quadrant (25 tiles) is initially available.  Purchased quadrants
    # increase all following days' relaxed production capacity.
    for day in range(NUM_DAYS):
        add(
            (
                (
                    index,
                    float(variable.occupancy[day])
                    - (
                        25.0
                        if variable.kind == "land" and variable.day <= day
                        else 0.0
                    ),
                )
                for index, variable in enumerate(variables)
            ),
            -np.inf,
            25.0,
        )

    # Hires follow the official Fibonacci prefix within each day.
    for day in range(NUM_DAYS):
        day_hires = {
            variable.tier: index
            for index, variable in enumerate(variables)
            if variable.kind == "hire" and variable.day == day
        }
        for tier in range(1, config.maximum_daily_hires):
            add(((day_hires[tier], 1.0), (day_hires[tier - 1], -1.0)), -np.inf, 0.0)

    # Each land price tier can be paid once, and later quadrants require the
    # earlier tier to have been purchased by that day.
    land_index = {
        (variable.day, variable.tier): index
        for index, variable in enumerate(variables)
        if variable.kind == "land"
    }
    for tier in range(3):
        add(((land_index[(day, tier)], 1.0) for day in range(NUM_DAYS)), 0.0, 1.0)
    for day in range(NUM_DAYS):
        for tier in range(1, 3):
            add(
                tuple((land_index[(prior, tier)], 1.0) for prior in range(day + 1))
                + tuple((land_index[(prior, tier - 1)], -1.0) for prior in range(day + 1)),
                -np.inf,
                0.0,
            )

    minimum_kind, minimum_item, minimum_units = FAMILY_DOMAINS[family]["minimum"]
    add(
        (
            (index, 1.0)
            for index, variable in enumerate(variables)
            if variable.kind == minimum_kind and variable.item == minimum_item
        ),
        float(minimum_units),
        np.inf,
    )
    if family == "R8_HYBRID":
        add(
            (
                (index, 1.0)
                for index, variable in enumerate(variables)
                if variable.kind == "animal" and variable.item == 2
            ),
            2.0,
            np.inf,
        )

    matrix = coo_matrix(
        (values, (row_ids, col_ids)),
        shape=(len(constraint_lb), count),
        dtype=np.float64,
    ).tocsr()
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            matrix,
            np.asarray(constraint_lb),
            np.asarray(constraint_ub),
        ),
        options={"time_limit": config.time_limit_seconds, "mip_rel_gap": 0.002},
    )
    if result.x is None:
        raise RuntimeError(f"{family}: MILP failed: {result.message}")

    chosen = np.rint(result.x).astype(np.int32)
    daily_cash_delta = np.zeros((NUM_DAYS,), dtype=np.float64)
    daily_actions = np.zeros((NUM_DAYS,), dtype=np.float64)
    daily_occupancy = np.zeros((NUM_DAYS,), dtype=np.float64)
    daily_crop_output = np.zeros((NUM_DAYS, NUM_CROPS), dtype=np.float64)
    daily_product_output = np.zeros((NUM_DAYS, len(MARKET_BASE_PRICES)), dtype=np.float64)
    crop_target = np.zeros((NUM_DAYS, NUM_CROPS), dtype=np.int32)
    animal_target = np.zeros((NUM_DAYS, NUM_ANIMALS), dtype=np.int32)
    hire_target = np.zeros((NUM_DAYS,), dtype=np.int32)
    land_target = np.ones((NUM_DAYS,), dtype=np.int32)
    decisions: list[dict[str, int | str]] = []
    for amount, variable in zip(chosen, variables, strict=True):
        if amount <= 0:
            continue
        daily_cash_delta += amount * variable.cash_delta
        daily_actions += amount * variable.actions
        daily_occupancy += amount * variable.occupancy
        daily_crop_output += amount * variable.crop_output
        daily_product_output += amount * variable.product_output
        if variable.kind == "crop":
            active = variable.occupancy > 0
            crop_target[active, variable.item] += amount
        elif variable.kind == "animal":
            animal_target[variable.day :, variable.item] += amount
        elif variable.kind == "hire":
            hire_target[variable.day] += amount
        elif variable.kind == "land":
            land_target[variable.day :] += amount
        decisions.append(
            {
                "kind": variable.kind,
                "item": variable.item,
                "day": variable.day,
                "tier": variable.tier,
                "amount": int(amount),
            }
        )

    daily_capacity = 24.0 * (1.0 + hire_target)
    daily_land_capacity = 25.0 * land_target
    cash_after_day = config.initial_cash - np.cumsum(daily_cash_delta)
    proposal = {
        "family": family,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_objective_cost_minus_revenue": float(result.fun),
        "predicted_terminal_cash": float(config.initial_cash - result.fun),
        "config": config.__dict__,
        "relaxations": [
            "fixed discounted prices instead of endogenous market curve",
            "day-level cash ordering",
            "weighted action budget instead of exact paths",
            "no weeds shops storage or placement geometry",
            "all produced inventory is sold by day 29",
        ],
        "decisions": decisions,
        "route_schedule_proposal": {
            "crop_target_by_day": crop_target.tolist(),
            "animal_target_by_day": animal_target.tolist(),
            "land_target_by_day": land_target.tolist(),
            "hire_target_by_day": hire_target.tolist(),
        },
        "daily": {
            "cash_after_day": np.rint(cash_after_day).astype(int).tolist(),
            "weighted_action_demand": daily_actions.tolist(),
            "action_capacity": daily_capacity.tolist(),
            "tile_occupancy": daily_occupancy.tolist(),
            "tile_capacity": daily_land_capacity.tolist(),
            "crop_output": np.rint(daily_crop_output).astype(int).tolist(),
            "product_output": np.rint(daily_product_output).astype(int).tolist(),
        },
    }
    proposal["proposal_sha256"] = hashlib.sha256(
        json.dumps(proposal, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return proposal
