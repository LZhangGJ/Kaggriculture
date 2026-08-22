"""Strategy-derived decision candidates and deterministic task execution.

The taxonomy and priority skeleton mirror the accepted local 28-agent JAX
arena plus the K320/FC2 fusion work.  The implementation remains native to the
PyTorch branch and its official 1.32.6 action interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from math import log1p
from typing import Any, Mapping, Sequence

import numpy as np

from .board_policy import BOARD_SIZE, _canonical_farms, _positions
from .gpu_policy import (
    ANIMALS,
    CROPS,
    ITEMS,
    MARKET_ACTIONS,
    MARKET_INDEX,
    MAX_UNITS,
    PRODUCTS,
    _get,
    _mapping,
    _market_mask,
    _tile_kind,
)


MAX_DECISION_CANDIDATES = 96
CANDIDATE_FEATURES = 32
DECISION_GROUPS = MAX_UNITS + 1  # market plus one group per own unit

REFERENCE_STRATEGY_AGENTS = (
    "rayk_k320_adaptive_rank1",
    "gold_proxy_rank14_recursion",
    "tetsutani_adaptive_premium_queue",
    "gold_proxy_rank19_manu_nicholas_jacob",
    "tetsutani_adaptive_latest",
    "boatlee_v20_multi_route",
    "gold_proxy_rank07_junichiro_morita",
    "x562_latest",
    "flexonafft_v59_multi_route",
    "kaito_v36_latest",
    "public_g04_soil_rain",
    "local_prt_v6",
    "gold_proxy_rank12_ai_b2b67_saas",
    "public_g02_rc5_c166",
    "deniz_v111_8c4s_latest",
    "public_g07_c95",
    "public_g01_boatlee_v16",
    "kaito_v27_midgame_reset",
    "public_g10_four_hire",
    "public_g06_v25",
    "public_g09_c68_thunder",
    "public_g08_v14",
    "public_g12_v13_r3",
    "public_g11_v21",
    "public_g14_v19_control",
    "public_g13_bruce_route1",
    "public_g15_v18_closed_loop",
    "public_g16_tran_cashflow",
    "fc2b_k320_clone_aware_fusion",
    "fc2a_shadow_prt_fix",
)


class TaskType(IntEnum):
    NONE = 0
    IDLE_OR_PASS = 1
    SAFE_RECOVERY = 2
    CROP_PRODUCTION = 3
    WATER_CROP = 4
    CLEAR_OR_REMOVE_TILE = 5
    BUILD_ANIMAL_STRUCTURE = 6
    ANIMAL_PURCHASE = 7
    ANIMAL_PLACE = 8
    ANIMAL_FEED = 9
    ANIMAL_CARE = 10
    ANIMAL_COLLECT_PRODUCT = 11
    ANIMAL_COLLECT_FERTILIZER = 12
    BUY_LAND = 13
    BUY_PRODUCT = 14
    APPLY_FERTILIZER = 15
    HIRE_WORKER = 16
    SHED_PICKUP = 17
    SHED_DEPOSIT = 18
    SELL_INVENTORY = 19
    TERMINAL_LIQUIDATION = 20


class CandidateSource(IntEnum):
    NONE = 0
    LAND = 1
    FERTILIZER_BUY = 2
    FERTILIZER_APPLY = 3
    CROP = 4
    ANIMAL = 5
    MAINTENANCE = 6
    INVENTORY = 7
    TERMINAL = 8


TASK_PRIOR = {
    TaskType.IDLE_OR_PASS: 1_500.0,
    TaskType.SAFE_RECOVERY: 10_000.0,
    TaskType.CROP_PRODUCTION: 3_500.0,
    TaskType.WATER_CROP: 6_500.0,
    TaskType.CLEAR_OR_REMOVE_TILE: 5_500.0,
    TaskType.BUILD_ANIMAL_STRUCTURE: 2_000.0,
    TaskType.ANIMAL_PURCHASE: 2_500.0,
    TaskType.ANIMAL_PLACE: 7_500.0,
    TaskType.ANIMAL_FEED: 9_500.0,
    TaskType.ANIMAL_CARE: 6_000.0,
    TaskType.ANIMAL_COLLECT_PRODUCT: 8_000.0,
    TaskType.ANIMAL_COLLECT_FERTILIZER: 7_000.0,
    TaskType.BUY_LAND: 1_500.0,
    TaskType.BUY_PRODUCT: 4_000.0,
    TaskType.APPLY_FERTILIZER: 5_000.0,
    TaskType.HIRE_WORKER: 3_000.0,
    TaskType.SHED_PICKUP: 4_000.0,
    TaskType.SHED_DEPOSIT: 10_000.0,
    TaskType.SELL_INVENTORY: 4_500.0,
    TaskType.TERMINAL_LIQUIDATION: 11_000.0,
}

ITEM_INDEX = {item: index for index, item in enumerate(ITEMS)}
_BASE_PRICES = {
    "WHEAT": 25.0,
    "CARROT": 35.0,
    "TOMATO": 60.0,
    "STRAWBERRY": 120.0,
    "MELON": 250.0,
    "EGG": 50.0,
    "MILK": 160.0,
    "WOOL": 200.0,
    "FERTILIZER": 100.0,
}
_SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
_ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
_ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
_SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))
_MARKET_QUANTITIES = (1, 4, 8, 16)


@dataclass(frozen=True)
class DecisionCandidate:
    task_type: TaskType
    owner_unit: int = -1
    target_x: int = -1
    target_y: int = -1
    item_id: int = -1
    quantity: int = 0
    source: CandidateSource = CandidateSource.NONE
    mandatory: bool = False
    continuation: bool = False
    prior: float = 0.0
    path_steps: int = 0
    deadline_step: int = 719
    cash_required: float = 0.0
    expected_value: float = 0.0
    market_index: int = -1

    @property
    def group_id(self) -> int:
        return 0 if self.owner_unit < 0 else self.owner_unit + 1

    @property
    def item(self) -> str | None:
        return ITEMS[self.item_id] if 0 <= self.item_id < len(ITEMS) else None


@dataclass
class DecisionMemory:
    """Persistent high-level plans for one actor stream."""

    unit_plans: list[DecisionCandidate | None] = field(
        default_factory=lambda: [None] * MAX_UNITS
    )

    def reset(self) -> None:
        self.unit_plans[:] = [None] * MAX_UNITS


@dataclass(frozen=True)
class EncodedCandidateSet:
    candidates: tuple[DecisionCandidate, ...]
    features: np.ndarray
    task_ids: np.ndarray
    owner_ids: np.ndarray
    item_ids: np.ndarray
    source_ids: np.ndarray
    target_xy: np.ndarray
    group_ids: np.ndarray
    mask: np.ndarray


def _tile_at(farm: Any, x: int, y: int) -> Any:
    tiles = list(_get(farm, "tiles", []) or [])
    if 0 <= y < len(tiles):
        row = list(tiles[y])
        if 0 <= x < len(row):
            return row[x]
    return "LOCKED"


def _nearest_shed(position: Sequence[int]) -> tuple[int, int]:
    x, y = int(position[0]), int(position[1])
    return min(_SHED_ACCESS, key=lambda value: (abs(x - value[0]) + abs(y - value[1]), value))


def _distance(left: Sequence[int], right: Sequence[int]) -> int:
    return abs(int(left[0]) - int(right[0])) + abs(int(left[1]) - int(right[1]))


def _inventory(private: Any, unit: int) -> Mapping[str, Any]:
    inventories = list(_get(private, "inventories", []) or [])
    return _mapping(inventories[unit]) if unit < len(inventories) else {}


def _public_clone_similarity(own: Any, opponent: Any) -> float:
    mismatch = 0.0
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            left = _tile_at(own, x, y)
            right = _tile_at(opponent, x, y)
            mismatch += float(_tile_kind(left) != _tile_kind(right))
    money_gap = abs(float(_get(own, "money", 0)) - float(_get(opponent, "money", 0)))
    mismatch += min(money_gap / 1_000.0, 10.0)
    return max(0.0, 1.0 - mismatch / 110.0)


def _opponent_pressure(observation: Any, opponent: Any) -> float:
    step = int(_get(observation, "step", 0) or 0)
    animals = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            tile = _tile_at(opponent, x, y)
            if isinstance(tile, Mapping) and tile.get("animal"):
                animals.append(tile.get("animal"))
    return float(
        24 <= step <= 95
        and animals.count("SHEEP") >= 4
        and animals.count("COW") <= 2
        and float(_get(opponent, "money", 0)) <= 1_000
    )


def _best_legal_quantity(
    operation: str,
    item: str,
    desired: int,
    market_mask: np.ndarray,
) -> int | None:
    """Choose the closest affordable/legal quantity instead of dropping a task."""

    legal = [
        quantity
        for quantity in _MARKET_QUANTITIES
        if bool(market_mask[MARKET_INDEX[(operation, item, quantity)]])
    ]
    if not legal:
        return None
    return min(legal, key=lambda value: (abs(value - max(desired, 1)), value))


def _plan_relevant(
    observation: Any, candidate: DecisionCandidate, own: Any
) -> bool:
    if candidate.owner_unit < 0:
        return False
    positions = _positions(own)
    if candidate.owner_unit >= len(positions):
        return False
    private = _get(observation, "private", {}) or {}
    inventory = _inventory(private, candidate.owner_unit)
    tile = _tile_at(own, candidate.target_x, candidate.target_y)
    mapping = _mapping(tile)
    task = candidate.task_type
    if task in (TaskType.SAFE_RECOVERY, TaskType.SHED_DEPOSIT):
        return sum(int(value) for value in inventory.values()) > 0
    if task == TaskType.CROP_PRODUCTION:
        if tile is None:
            seeds = _mapping(_get(private, "seeds", {}))
            return candidate.item is not None and int(seeds.get(candidate.item, 0)) > 0
        return (
            mapping.get("kind") == "PLANT"
            and mapping.get("crop") == candidate.item
            and (
                int(mapping.get("yield_units", 0)) > 0
                or not bool(mapping.get("watered_today", False))
            )
        )
    if task == TaskType.WATER_CROP:
        return mapping.get("kind") == "PLANT" and not bool(mapping.get("watered_today", False))
    if task == TaskType.CLEAR_OR_REMOVE_TILE:
        return mapping.get("kind") == "WEED"
    if task == TaskType.BUILD_ANIMAL_STRUCTURE:
        return tile is None
    if task == TaskType.ANIMAL_PLACE:
        return mapping.get("kind") in ("COOP", "PASTURE") and not mapping.get("animal")
    if task == TaskType.ANIMAL_FEED:
        return bool(mapping.get("animal")) and not bool(mapping.get("fed_today", False))
    if task == TaskType.ANIMAL_CARE:
        return bool(mapping.get("animal")) and not bool(mapping.get("cared_today", False))
    if task == TaskType.ANIMAL_COLLECT_PRODUCT:
        return bool(mapping.get("animal")) and int(mapping.get("yield_units", 0)) > 0
    if task == TaskType.ANIMAL_COLLECT_FERTILIZER:
        return bool(mapping.get("fertilizer_available", False))
    if task == TaskType.APPLY_FERTILIZER:
        return mapping.get("kind") == "PLANT" and (
            int(inventory.get("FERTILIZER", 0)) > 0
            or int(_mapping(_get(private, "shed", {})).get("FERTILIZER", 0)) > 0
        )
    return task == TaskType.IDLE_OR_PASS


def _market_candidates(observation: Any, own: Any, opponent: Any) -> list[DecisionCandidate]:
    mask = _market_mask(observation)
    step = int(_get(observation, "step", 0) or 0)
    day = int(_get(observation, "day", step // 24) or 0)
    private = _get(observation, "private", {}) or {}
    shed = _mapping(_get(private, "shed", {}))
    seeds = _mapping(_get(private, "seeds", {}))
    market = _get(observation, "market", {}) or {}
    prices = _mapping(_get(market, "prices", {}))
    similarity = _public_clone_similarity(own, opponent)
    candidates = [
        DecisionCandidate(
            TaskType.IDLE_OR_PASS,
            prior=TASK_PRIOR[TaskType.IDLE_OR_PASS],
            market_index=MARKET_INDEX[("NONE", None, 0)],
        )
    ]

    def add(
        action: tuple[str, str | None, int],
        task: TaskType,
        source: CandidateSource,
        *,
        mandatory: bool = False,
        cash: float = 0.0,
        value: float = 0.0,
        prior: float | None = None,
    ) -> None:
        index = MARKET_INDEX.get(action)
        if index is None or not bool(mask[index]):
            return
        item = action[1]
        candidates.append(
            DecisionCandidate(
                task,
                item_id=ITEM_INDEX.get(item, -1),
                quantity=action[2],
                source=source,
                mandatory=mandatory,
                prior=TASK_PRIOR[task] if prior is None else prior,
                deadline_step=719,
                cash_required=cash,
                expected_value=value,
                market_index=index,
            )
        )

    add(("BUY_LAND", None, 1), TaskType.BUY_LAND, CandidateSource.LAND)

    plant_needs_fertilizer = any(
        isinstance(_tile_at(own, x, y), Mapping)
        and _mapping(_tile_at(own, x, y)).get("kind") == "PLANT"
        and int(_mapping(_tile_at(own, x, y)).get("fertilized_until_day", -1)) < day + 2
        for y in range(BOARD_SIZE)
        for x in range(BOARD_SIZE)
    )
    fertilizer_price = float(prices.get("FERTILIZER", _BASE_PRICES["FERTILIZER"]))
    add(
        ("BUY_PRODUCT", "FERTILIZER", 1),
        TaskType.BUY_PRODUCT,
        CandidateSource.FERTILIZER_BUY,
        mandatory=plant_needs_fertilizer and int(shed.get("FERTILIZER", 0)) <= 0,
        cash=fertilizer_price,
    )

    unfed = 0
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            tile = _mapping(_tile_at(own, x, y))
            unfed += int(bool(tile.get("animal")) and not bool(tile.get("fed_today", False)))
    wheat_deficit = max(unfed - int(shed.get("WHEAT", 0)), 0)
    wheat_quantity = _best_legal_quantity(
        "BUY_PRODUCT", "WHEAT", wheat_deficit, mask
    )
    wheat_price = float(prices.get("WHEAT", _BASE_PRICES["WHEAT"]))
    if wheat_quantity is not None:
        add(
            ("BUY_PRODUCT", "WHEAT", wheat_quantity),
            TaskType.BUY_PRODUCT,
            CandidateSource.ANIMAL,
            mandatory=wheat_deficit > 0,
            cash=wheat_price * wheat_quantity,
        )

    for animal in ANIMALS:
        add(
            ("BUY_ANIMAL", animal, 1),
            TaskType.ANIMAL_PURCHASE,
            CandidateSource.ANIMAL,
            cash=float(_ANIMAL_COST[animal]),
        )
    for crop in CROPS:
        deficit = max(4 - int(seeds.get(crop, 0)), 1)
        quantity = _best_legal_quantity("BUY_SEED", crop, deficit, mask)
        if quantity is None:
            continue
        add(
            ("BUY_SEED", crop, quantity),
            TaskType.CROP_PRODUCTION,
            CandidateSource.CROP,
            cash=float(_SEED_COST[crop] * quantity),
        )
    add(("HIRE", None, 1), TaskType.HIRE_WORKER, CandidateSource.INVENTORY)

    terminal = step >= 718
    for product in PRODUCTS:
        quantity = int(shed.get(product, 0))
        if quantity <= 0:
            continue
        task = TaskType.TERMINAL_LIQUIDATION if terminal else TaskType.SELL_INVENTORY
        prior = TASK_PRIOR[task]
        if step >= 120 and similarity >= 0.94 and product in ("STRAWBERRY", "MILK", "WOOL"):
            prior += 750.0
        add(
            ("SELL", product, 100),
            task,
            CandidateSource.TERMINAL if terminal else CandidateSource.INVENTORY,
            mandatory=terminal,
            value=float(prices.get(product, _BASE_PRICES[product])) * quantity,
            prior=prior,
        )
    return candidates


def _unit_options(
    observation: Any,
    own: Any,
    unit: int,
) -> list[DecisionCandidate]:
    positions = _positions(own)
    if unit >= len(positions):
        return []
    position = positions[unit]
    step = int(_get(observation, "step", 0) or 0)
    day = int(_get(observation, "day", step // 24) or 0)
    private = _get(observation, "private", {}) or {}
    shed = _mapping(_get(private, "shed", {}))
    seeds = _mapping(_get(private, "seeds", {}))
    inventory = _inventory(private, unit)
    prices = _mapping(_get(_get(observation, "market", {}) or {}, "prices", {}))
    options: list[DecisionCandidate] = []

    carried = sum(int(value) for value in inventory.values())
    if carried > 0:
        depot = _nearest_shed(position)
        options.append(
            DecisionCandidate(
                TaskType.SAFE_RECOVERY,
                owner_unit=unit,
                target_x=depot[0],
                target_y=depot[1],
                quantity=carried,
                source=CandidateSource.INVENTORY,
                mandatory=True,
                prior=TASK_PRIOR[TaskType.SAFE_RECOVERY],
                path_steps=_distance(position, depot),
                deadline_step=min(((step // 24) + 1) * 24, 719),
            )
        )

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            tile = _tile_at(own, x, y)
            mapping = _mapping(tile)
            distance = _distance(position, (x, y))
            deadline = min(((step // 24) + 1) * 24, 719)
            common = dict(owner_unit=unit, target_x=x, target_y=y, path_steps=distance, deadline_step=deadline)
            if mapping.get("kind") == "WEED":
                options.append(DecisionCandidate(TaskType.CLEAR_OR_REMOVE_TILE, source=CandidateSource.CROP, prior=5_500.0, **common))
            if mapping.get("kind") == "PLANT":
                crop = mapping.get("crop")
                item_id = ITEM_INDEX.get(crop, -1)
                if int(mapping.get("yield_units", 0)) > 0:
                    value = float(prices.get(crop, _BASE_PRICES.get(crop, 0))) * int(mapping.get("yield_units", 0))
                    options.append(DecisionCandidate(TaskType.CROP_PRODUCTION, item_id=item_id, source=CandidateSource.CROP, prior=8_500.0, expected_value=value, **common))
                if not bool(mapping.get("watered_today", False)):
                    urgent = int(mapping.get("consecutive_unwatered", 0)) >= 1
                    options.append(DecisionCandidate(TaskType.WATER_CROP, item_id=item_id, source=CandidateSource.MAINTENANCE, mandatory=urgent, prior=12_000.0 if urgent else 6_500.0, **common))
                fertilizer_available = int(inventory.get("FERTILIZER", 0)) > 0 or int(shed.get("FERTILIZER", 0)) > 0
                if fertilizer_available and int(mapping.get("fertilized_until_day", -1)) < day + 2:
                    options.append(DecisionCandidate(TaskType.APPLY_FERTILIZER, item_id=ITEM_INDEX["FERTILIZER"], source=CandidateSource.FERTILIZER_APPLY, prior=5_000.0, **common))
            animal = mapping.get("animal")
            if animal:
                wheat_available = int(inventory.get("WHEAT", 0)) > 0 or int(shed.get("WHEAT", 0)) > 0
                if wheat_available and not bool(mapping.get("fed_today", False)):
                    urgent = int(mapping.get("consecutive_unfed", 0)) >= 1
                    options.append(DecisionCandidate(TaskType.ANIMAL_FEED, item_id=ITEM_INDEX["WHEAT"], source=CandidateSource.MAINTENANCE, mandatory=urgent, prior=12_000.0 if urgent else 9_500.0, **common))
                if int(mapping.get("yield_units", 0)) > 0:
                    product = _ANIMAL_PRODUCT.get(str(animal))
                    value = float(prices.get(product, _BASE_PRICES.get(product, 0))) * int(mapping.get("yield_units", 0))
                    options.append(DecisionCandidate(TaskType.ANIMAL_COLLECT_PRODUCT, item_id=ITEM_INDEX.get(product, -1), source=CandidateSource.ANIMAL, prior=8_000.0, expected_value=value, **common))
                if bool(mapping.get("fertilizer_available", False)):
                    options.append(DecisionCandidate(TaskType.ANIMAL_COLLECT_FERTILIZER, item_id=ITEM_INDEX["FERTILIZER"], source=CandidateSource.ANIMAL, prior=7_000.0, **common))
                if not bool(mapping.get("cared_today", False)):
                    options.append(DecisionCandidate(TaskType.ANIMAL_CARE, item_id=ITEM_INDEX.get(str(animal), -1), source=CandidateSource.MAINTENANCE, prior=6_000.0, **common))
            if mapping.get("kind") in ("COOP", "PASTURE") and not animal:
                compatible = ("GOOSE",) if mapping.get("kind") == "COOP" else ("COW", "SHEEP")
                for wanted in compatible:
                    if int(inventory.get(wanted, 0)) > 0 or int(shed.get(wanted, 0)) > 0:
                        options.append(DecisionCandidate(TaskType.ANIMAL_PLACE, item_id=ITEM_INDEX[wanted], source=CandidateSource.ANIMAL, mandatory=True, prior=7_500.0, **common))
            if tile is None:
                for crop in CROPS:
                    if int(seeds.get(crop, 0)) <= 0:
                        continue
                    remaining_days = 29 - day
                    if remaining_days <= 0:
                        continue
                    crop_bias = min(float(prices.get(crop, _BASE_PRICES[crop])) / _BASE_PRICES[crop], 2.0) * 250.0
                    options.append(DecisionCandidate(TaskType.CROP_PRODUCTION, item_id=ITEM_INDEX[crop], source=CandidateSource.CROP, prior=3_500.0 + crop_bias, expected_value=float(prices.get(crop, _BASE_PRICES[crop])), **common))
                if unit == 0:
                    options.append(DecisionCandidate(TaskType.BUILD_ANIMAL_STRUCTURE, item_id=ITEM_INDEX["GOOSE"], source=CandidateSource.ANIMAL, prior=2_000.0, **common))
                    options.append(DecisionCandidate(TaskType.BUILD_ANIMAL_STRUCTURE, item_id=ITEM_INDEX["COW"], source=CandidateSource.ANIMAL, prior=1_999.0, **common))

    unique: dict[tuple[int, int, int, int], DecisionCandidate] = {}
    for candidate in options:
        key = (int(candidate.task_type), candidate.target_x, candidate.target_y, candidate.item_id)
        current = unique.get(key)
        if current is None or candidate.prior > current.prior:
            unique[key] = candidate
    return sorted(
        unique.values(),
        key=lambda candidate: (-candidate.prior, candidate.path_steps, candidate.target_y, candidate.target_x, candidate.item_id),
    )


def _candidate_features(
    observation: Any,
    candidate: DecisionCandidate,
    own: Any,
    opponent: Any,
    workload: int,
) -> np.ndarray:
    step = int(_get(observation, "step", 0) or 0)
    private = _get(observation, "private", {}) or {}
    shed = _mapping(_get(private, "shed", {}))
    inventory = _inventory(private, max(candidate.owner_unit, 0)) if candidate.owner_unit >= 0 else {}
    tile = _tile_at(own, candidate.target_x, candidate.target_y) if candidate.target_x >= 0 else None
    mapping = _mapping(tile)
    market = _get(observation, "market", {}) or {}
    prices = _mapping(_get(market, "prices", {}))
    market_inventory = _mapping(_get(market, "inventory", {}))
    item = candidate.item
    price = float(prices.get(item, _BASE_PRICES.get(item or "", 0))) if item else 0.0
    base_price = _BASE_PRICES.get(item or "", 1.0)
    roi = (candidate.expected_value - candidate.cash_required) / max(candidate.cash_required, 1.0)
    features = np.asarray(
        (
            1.0,
            float(candidate.mandatory),
            float(candidate.continuation),
            (candidate.target_x + 1.0) / (BOARD_SIZE + 1.0),
            (candidate.target_y + 1.0) / (BOARD_SIZE + 1.0),
            min(candidate.path_steps / 18.0, 2.0),
            min((candidate.path_steps + 8) / 36.0, 2.0),
            min(max(candidate.quantity, 0) / 100.0, 2.0),
            min(candidate.prior / 12_000.0, 1.25),
            log1p(max(candidate.cash_required, 0.0)) / log1p(2_000_000.0),
            log1p(max(candidate.expected_value, 0.0)) / log1p(2_000_000.0),
            min(max(roi, -2.0), 2.0) / 2.0,
            max(candidate.deadline_step - step, 0) / 719.0,
            float(step + candidate.path_steps + 1 <= candidate.deadline_step),
            float(candidate.task_type in (TaskType.ANIMAL_FEED, TaskType.ANIMAL_PLACE, TaskType.APPLY_FERTILIZER) and int(inventory.get(item or "", 0)) <= 0),
            min(sum(int(value) for value in inventory.values()) / 100.0, 2.0),
            min(sum(int(value) for value in shed.values()) / 100.0, 2.0),
            min(len(_positions(own)) / float(MAX_UNITS), 1.0),
            min(workload / float(MAX_UNITS), 2.0),
            min(float(mapping.get("yield_units", 0)) / 6.0, 2.0),
            min(float(max(mapping.get("consecutive_unwatered", 0), mapping.get("consecutive_unfed", 0))) / 2.0, 1.0),
            float(bool(mapping.get("watered_today", False))),
            float(bool(mapping.get("fed_today", False))),
            float(bool(mapping.get("cared_today", False))),
            float(bool(mapping.get("fertilizer_available", False))),
            float(tile is None),
            float(mapping.get("kind") == "PLANT"),
            float(bool(mapping.get("animal"))),
            min(price / max(base_price, 1.0), 4.0),
            (float(market_inventory.get(item, 10_000)) - 10_000.0) / 10_000.0 if item in PRODUCTS else 0.0,
            _opponent_pressure(observation, opponent),
            _public_clone_similarity(own, opponent),
        ),
        dtype=np.float32,
    )
    if features.shape != (CANDIDATE_FEATURES,):
        raise AssertionError("decision candidate feature width drifted")
    return features


def generate_decision_candidates(
    observation: Any,
    memory: DecisionMemory | None = None,
) -> EncodedCandidateSet:
    """Create one fixed-width, strategy-derived candidate set."""

    own, opponent = _canonical_farms(observation)
    positions = _positions(own)
    memory = DecisionMemory() if memory is None else memory
    candidates = _market_candidates(observation, own, opponent)
    optional: list[DecisionCandidate] = []
    workload = 0
    for unit in range(min(len(positions), MAX_UNITS)):
        candidates.append(
            DecisionCandidate(
                TaskType.IDLE_OR_PASS,
                owner_unit=unit,
                prior=TASK_PRIOR[TaskType.IDLE_OR_PASS],
            )
        )
        plan = memory.unit_plans[unit]
        if plan is not None and _plan_relevant(observation, plan, own):
            candidates.append(
                replace(plan, continuation=True, prior=min(plan.prior + 250.0, 12_000.0))
            )
        else:
            memory.unit_plans[unit] = None
        choices = _unit_options(observation, own, unit)
        workload += sum(candidate.prior >= 6_000.0 for candidate in choices)
        optional.extend(choices[: (7 if unit == 0 else 2)])

    existing = {
        (candidate.group_id, int(candidate.task_type), candidate.target_x, candidate.target_y, candidate.item_id)
        for candidate in candidates
    }
    for candidate in optional:
        key = (candidate.group_id, int(candidate.task_type), candidate.target_x, candidate.target_y, candidate.item_id)
        if key not in existing:
            candidates.append(candidate)
            existing.add(key)
    if len(candidates) > MAX_DECISION_CANDIDATES:
        fixed = [candidate for candidate in candidates if candidate.task_type == TaskType.IDLE_OR_PASS or candidate.mandatory or candidate.continuation]
        rest = [candidate for candidate in candidates if candidate not in fixed]
        rest.sort(key=lambda candidate: (-candidate.prior, candidate.path_steps))
        candidates = (fixed + rest)[:MAX_DECISION_CANDIDATES]

    count = len(candidates)
    features = np.zeros((MAX_DECISION_CANDIDATES, CANDIDATE_FEATURES), dtype=np.float32)
    task_ids = np.zeros(MAX_DECISION_CANDIDATES, dtype=np.int64)
    owner_ids = np.zeros(MAX_DECISION_CANDIDATES, dtype=np.int64)
    item_ids = np.zeros(MAX_DECISION_CANDIDATES, dtype=np.int64)
    source_ids = np.zeros(MAX_DECISION_CANDIDATES, dtype=np.int64)
    target_xy = np.full((MAX_DECISION_CANDIDATES, 2), -1, dtype=np.int64)
    group_ids = np.zeros(MAX_DECISION_CANDIDATES, dtype=np.int64)
    mask = np.zeros(MAX_DECISION_CANDIDATES, dtype=np.bool_)
    for index, candidate in enumerate(candidates):
        features[index] = _candidate_features(observation, candidate, own, opponent, workload)
        task_ids[index] = int(candidate.task_type)
        owner_ids[index] = candidate.owner_unit + 1
        item_ids[index] = candidate.item_id + 1
        source_ids[index] = int(candidate.source)
        target_xy[index] = (candidate.target_x, candidate.target_y)
        group_ids[index] = candidate.group_id
        mask[index] = True
    return EncodedCandidateSet(
        candidates=tuple(candidates),
        features=features,
        task_ids=task_ids,
        owner_ids=owner_ids,
        item_ids=item_ids,
        source_ids=source_ids,
        target_xy=target_xy,
        group_ids=group_ids,
        mask=mask,
    )


def _movement_toward(
    position: Sequence[int],
    target: Sequence[int],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    unit: int,
) -> list[str]:
    x, y = int(position[0]), int(position[1])
    target_x, target_y = int(target[0]), int(target[1])
    moves = (("NORTH", 0, -1), ("SOUTH", 0, 1), ("EAST", 1, 0), ("WEST", -1, 0))
    ranked = sorted(
        enumerate(moves),
        key=lambda entry: (
            abs(x + entry[1][1] - target_x) + abs(y + entry[1][2] - target_y),
            (entry[0] - unit) % len(moves),
        ),
    )
    for _, (name, dx, dy) in ranked:
        destination = (x + dx, y + dy)
        if not (0 <= destination[0] < BOARD_SIZE and 0 <= destination[1] < BOARD_SIZE):
            continue
        if destination in reserved or destination in (occupied - {(x, y)}):
            continue
        reserved.add(destination)
        return [name]
    reserved.add((x, y))
    return ["PASS"]


def _compile_unit_candidate(
    observation: Any,
    own: Any,
    candidate: DecisionCandidate,
    occupied: set[tuple[int, int]],
    reserved_moves: set[tuple[int, int]],
    reserved_targets: set[tuple[int, int]],
) -> list[Any]:
    unit = candidate.owner_unit
    positions = _positions(own)
    if unit < 0 or unit >= len(positions) or candidate.task_type == TaskType.IDLE_OR_PASS:
        return ["PASS"]
    position = positions[unit]
    private = _get(observation, "private", {}) or {}
    inventory = _inventory(private, unit)
    shed = _mapping(_get(private, "shed", {}))
    item = candidate.item
    needs_item = candidate.task_type in (
        TaskType.ANIMAL_FEED,
        TaskType.ANIMAL_PLACE,
        TaskType.APPLY_FERTILIZER,
        TaskType.SHED_PICKUP,
    )
    if needs_item and item and int(inventory.get(item, 0)) <= 0:
        depot = _nearest_shed(position)
        if tuple(position) != depot:
            return _movement_toward(position, depot, occupied, reserved_moves, unit)
        if int(shed.get(item, 0)) > 0:
            return ["PICKUP", item]
        return ["PASS"]
    if candidate.task_type in (TaskType.SAFE_RECOVERY, TaskType.SHED_DEPOSIT):
        depot = _nearest_shed(position)
        if tuple(position) != depot:
            return _movement_toward(position, depot, occupied, reserved_moves, unit)
        return ["DROP"] if sum(int(value) for value in inventory.values()) > 0 else ["PASS"]

    target = (candidate.target_x, candidate.target_y)
    if tuple(position) != target:
        return _movement_toward(position, target, occupied, reserved_moves, unit)
    if target in reserved_targets:
        return ["PASS"]
    reserved_targets.add(target)
    tile = _tile_at(own, *target)
    mapping = _mapping(tile)
    task = candidate.task_type
    if task == TaskType.CROP_PRODUCTION:
        if tile is None and item in CROPS:
            return ["PLANT", item]
        if mapping.get("kind") == "PLANT" and int(mapping.get("yield_units", 0)) > 0:
            return ["HARVEST"]
        if mapping.get("kind") == "PLANT" and not bool(mapping.get("watered_today", False)):
            return ["WATER"]
    if task == TaskType.WATER_CROP:
        return ["WATER"]
    if task == TaskType.CLEAR_OR_REMOVE_TILE:
        return ["DIG"]
    if task == TaskType.BUILD_ANIMAL_STRUCTURE:
        return ["BUILD_COOP" if item == "GOOSE" else "BUILD_PASTURE"]
    if task == TaskType.ANIMAL_PLACE and item:
        return ["PLACE", item]
    if task == TaskType.ANIMAL_FEED:
        return ["FEED"]
    if task == TaskType.ANIMAL_CARE:
        return ["CARE"]
    if task == TaskType.ANIMAL_COLLECT_PRODUCT:
        return ["HARVEST"]
    if task == TaskType.ANIMAL_COLLECT_FERTILIZER:
        return ["COLLECT_FERTILIZER"]
    if task == TaskType.APPLY_FERTILIZER:
        return ["FERTILIZE"]
    if task == TaskType.SHED_PICKUP and item:
        return ["PICKUP", item]
    return ["PASS"]


def compile_decision_action(
    observation: Any,
    encoded: EncodedCandidateSet,
    selected_indices: Sequence[int],
    memory: DecisionMemory | None = None,
) -> dict[str, Any]:
    """Compile selected high-level decisions into one official action."""

    own, _ = _canonical_farms(observation)
    positions = _positions(own)
    memory = DecisionMemory() if memory is None else memory
    selected: dict[int, DecisionCandidate] = {}
    for group, index in enumerate(selected_indices):
        if 0 <= int(index) < len(encoded.candidates):
            candidate = encoded.candidates[int(index)]
            if candidate.group_id == group:
                selected[group] = candidate

    market: list[list[Any]] = []
    market_candidate = selected.get(0)
    if market_candidate is not None and market_candidate.market_index >= 0:
        op, item, quantity = MARKET_ACTIONS[market_candidate.market_index]
        if op != "NONE":
            market = [[op]] if item is None else [[op, item, quantity]]

    occupied = {tuple(int(value) for value in position) for position in positions}
    reserved_moves: set[tuple[int, int]] = set()
    reserved_targets: set[tuple[int, int]] = set()
    unit_actions: list[list[Any]] = []
    for unit in range(min(len(positions), MAX_UNITS)):
        candidate = selected.get(unit + 1)
        if candidate is None:
            action = ["PASS"]
            memory.unit_plans[unit] = None
        else:
            action = _compile_unit_candidate(
                observation,
                own,
                candidate,
                occupied,
                reserved_moves,
                reserved_targets,
            )
            memory.unit_plans[unit] = (
                None
                if candidate.task_type == TaskType.IDLE_OR_PASS
                else replace(candidate, continuation=False)
            )
        unit_actions.append(action)
    return {
        "farmer": unit_actions[0] if unit_actions else ["PASS"],
        "hands": unit_actions[1:],
        "market": market,
    }
