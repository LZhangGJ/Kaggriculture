"""High-level intent labels and closure diagnostics shared by V4/V5 BC."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence

import numpy as np

from .board_policy import _canonical_farms, _positions
from .decision_schema import TaskType, _tile_at
from .gpu_policy import (
    ANIMALS,
    ITEMS,
    MARKET_ACTIONS,
    MARKET_INDEX,
    MAX_UNITS,
    PRODUCTS,
    _get,
    _mapping,
)
from .hierarchical_schema import (
    BUDGET_CATEGORIES,
    MAX_MARKET_ORDERS,
    BudgetCategory,
    EncodedTaskSet,
    MarketBudget,
    market_budget_category,
)


_MOVEMENT = {"NORTH", "SOUTH", "EAST", "WEST", "PASS"}
_QUANTITIES = (1, 4, 8, 16)
_ITEM_INDEX = {item: index for index, item in enumerate(ITEMS)}
_ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}


@dataclass(frozen=True)
class IntentLabel:
    task_type: TaskType
    target_x: int = -1
    target_y: int = -1
    item_id: int = -1


class ClosureReason(IntEnum):
    """Why an expert high-level intent did or did not enter the candidate set."""

    EXACT = 0
    TASK_TYPE_MISSING = 1
    TARGET_MISSING = 2
    ITEM_MISMATCH = 3
    PAIR_MASKED = 4
    UNRESOLVED = 5


@dataclass(frozen=True)
class BehaviorTargets:
    assignment_targets: np.ndarray
    assignment_active: np.ndarray
    assignment_closed: np.ndarray
    assignment_task_ids: np.ndarray
    assignment_closure_reasons: np.ndarray
    market_targets: np.ndarray
    market_active: np.ndarray


@dataclass(frozen=True)
class DailyBudgetSoftTarget:
    """Inverse-planned daily budget supervision derived from visible spending.

    Actual category spend is a lower bound rather than an assertion that the
    expert's unobserved planned envelope exactly equalled realized purchases.
    """

    day: int
    reserve_fraction: float
    category_fractions: np.ndarray
    category_floor_fractions: np.ndarray
    category_spend: np.ndarray
    total_spend: float
    end_cash: float
    cash_basis: float
    spend_active: bool
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "reserve_fraction": self.reserve_fraction,
            "category_fractions": {
                name: float(self.category_fractions[index])
                for index, name in enumerate(BUDGET_CATEGORIES)
            },
            "category_floor_fractions": {
                name: float(self.category_floor_fractions[index])
                for index, name in enumerate(BUDGET_CATEGORIES)
            },
            "category_spend": {
                name: float(self.category_spend[index])
                for index, name in enumerate(BUDGET_CATEGORIES)
            },
            "total_spend": self.total_spend,
            "end_cash": self.end_cash,
            "cash_basis": self.cash_basis,
            "spend_active": self.spend_active,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DailyBudgetSoftTarget":
        def vector(name: str) -> np.ndarray:
            values = raw.get(name, {}) or {}
            if isinstance(values, Mapping):
                return np.asarray(
                    [float(values.get(category, 0.0)) for category in BUDGET_CATEGORIES],
                    dtype=np.float32,
                )
            return np.asarray(list(values), dtype=np.float32)

        return cls(
            day=int(raw.get("day", 0)),
            reserve_fraction=float(raw.get("reserve_fraction", 0.0)),
            category_fractions=vector("category_fractions"),
            category_floor_fractions=vector("category_floor_fractions"),
            category_spend=vector("category_spend"),
            total_spend=float(raw.get("total_spend", 0.0)),
            end_cash=float(raw.get("end_cash", 0.0)),
            cash_basis=max(float(raw.get("cash_basis", 1.0)), 1.0),
            spend_active=bool(raw.get("spend_active", False)),
            confidence=float(raw.get("confidence", 1.0)),
        )


def _unit_action(action: Mapping[str, Any], unit: int) -> Sequence[Any]:
    if unit == 0:
        value = action.get("farmer", ["PASS"])
    else:
        hands = list(action.get("hands", []) or [])
        value = hands[unit - 1] if unit - 1 < len(hands) else ["PASS"]
    return value if isinstance(value, Sequence) and value else ["PASS"]


def _intent_from_primitive(
    observation: Any,
    unit: int,
    raw: Sequence[Any],
) -> IntentLabel | None:
    operation = str(raw[0]) if raw else "PASS"
    if operation in _MOVEMENT:
        return None
    own, _ = _canonical_farms(observation)
    positions = _positions(own)
    if unit >= len(positions):
        return None
    x, y = (int(value) for value in positions[unit])
    item = str(raw[1]) if len(raw) > 1 else None
    item_id = _ITEM_INDEX.get(item, -1)
    tile = _mapping(_tile_at(own, x, y))
    mapping = {
        "DROP": TaskType.SAFE_RECOVERY,
        "PICKUP": TaskType.SHED_PICKUP,
        "PLANT": TaskType.CROP_PRODUCTION,
        "WATER": TaskType.WATER_CROP,
        "FERTILIZE": TaskType.APPLY_FERTILIZER,
        "DIG": TaskType.CLEAR_OR_REMOVE_TILE,
        "BUILD_COOP": TaskType.BUILD_ANIMAL_STRUCTURE,
        "BUILD_PASTURE": TaskType.BUILD_ANIMAL_STRUCTURE,
        "PLACE": TaskType.ANIMAL_PLACE,
        "FEED": TaskType.ANIMAL_FEED,
        "CARE": TaskType.ANIMAL_CARE,
        "COLLECT_FERTILIZER": TaskType.ANIMAL_COLLECT_FERTILIZER,
    }
    if operation == "HARVEST":
        task = (
            TaskType.ANIMAL_COLLECT_PRODUCT
            if tile.get("animal")
            else TaskType.CROP_PRODUCTION
        )
        if item_id < 0:
            harvested_item = tile.get("crop")
            if tile.get("animal"):
                harvested_item = _ANIMAL_PRODUCT.get(str(tile.get("animal")))
            item_id = _ITEM_INDEX.get(str(harvested_item or ""), -1)
    else:
        task = mapping.get(operation)
    if task is None:
        return None
    if operation == "BUILD_COOP":
        item_id = _ITEM_INDEX["GOOSE"]
    elif operation == "BUILD_PASTURE":
        item_id = _ITEM_INDEX["COW"]
    elif operation == "FEED":
        item_id = _ITEM_INDEX["WHEAT"]
    elif operation in ("FERTILIZE", "COLLECT_FERTILIZER"):
        item_id = _ITEM_INDEX["FERTILIZER"]
    elif operation == "CARE" and item_id < 0:
        item_id = _ITEM_INDEX.get(str(tile.get("animal") or ""), -1)
    return IntentLabel(task, x, y, item_id)


def infer_episode_intents(
    observations: Sequence[Any],
    actions: Sequence[Mapping[str, Any]],
    *,
    horizon: int = 24,
) -> list[list[IntentLabel]]:
    """Collapse movement/pickup chains into the next same-day semantic task."""

    if len(observations) != len(actions):
        raise ValueError("observations and actions must have equal length")
    labels: list[list[IntentLabel]] = []
    for step, observation in enumerate(observations):
        own, _ = _canonical_farms(observation)
        unit_count = min(len(_positions(own)), MAX_UNITS)
        current_day = int(_get(observation, "day", 0) or 0)
        row: list[IntentLabel] = []
        for unit in range(unit_count):
            chosen: IntentLabel | None = None
            pickup_fallback: IntentLabel | None = None
            for future in range(step, min(len(actions), step + horizon + 1)):
                if int(_get(observations[future], "day", 0) or 0) != current_day:
                    break
                future_own, _ = _canonical_farms(observations[future])
                if unit >= len(_positions(future_own)):
                    break
                raw = _unit_action(actions[future], unit)
                candidate = _intent_from_primitive(observations[future], unit, raw)
                if candidate is None:
                    continue
                if candidate.task_type == TaskType.SHED_PICKUP:
                    pickup_fallback = candidate
                    continue
                chosen = candidate
                break
            row.append(
                chosen
                or pickup_fallback
                or IntentLabel(TaskType.IDLE_OR_PASS)
            )
        labels.append(row)
    return labels


def match_intents_with_diagnostics(
    encoded: EncodedTaskSet,
    intents: Sequence[IntentLabel],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map intents and retain a precise closure reason for every active unit."""

    targets = np.full(MAX_UNITS, -1, dtype=np.int64)
    active = np.zeros(MAX_UNITS, dtype=np.bool_)
    closed = np.zeros(MAX_UNITS, dtype=np.bool_)
    task_ids = np.full(MAX_UNITS, int(TaskType.NONE), dtype=np.int64)
    reasons = np.full(MAX_UNITS, -1, dtype=np.int64)
    idle = next(
        index
        for index, card in enumerate(encoded.tasks)
        if card.task_type == TaskType.IDLE_OR_PASS
    )
    for unit, intent in enumerate(intents[:MAX_UNITS]):
        active[unit] = True
        task_ids[unit] = int(intent.task_type)
        ranked: list[tuple[int, int]] = []
        for index, card in enumerate(encoded.tasks):
            if not encoded.pair_mask[unit, index]:
                continue
            candidate = card.candidate
            if candidate.task_type != intent.task_type:
                continue
            score = 1
            if (
                candidate.target_x == intent.target_x
                and candidate.target_y == intent.target_y
            ):
                score += 4
            if intent.item_id < 0 or candidate.item_id == intent.item_id:
                score += 2
            ranked.append((score, index))
        if ranked:
            score, index = max(ranked)
            targets[unit] = index
            closed[unit] = score >= 7 or intent.task_type == TaskType.IDLE_OR_PASS
        else:
            targets[unit] = idle
        if closed[unit]:
            reasons[unit] = int(ClosureReason.EXACT)
            continue

        typed = [
            (index, card)
            for index, card in enumerate(encoded.tasks)
            if card.task_type == intent.task_type
        ]
        targeted = [
            (index, card)
            for index, card in typed
            if card.candidate.target_x == intent.target_x
            and card.candidate.target_y == intent.target_y
        ]
        item_matched = [
            (index, card)
            for index, card in targeted
            if intent.item_id < 0 or card.candidate.item_id == intent.item_id
        ]
        pair_usable = [
            (index, card)
            for index, card in item_matched
            if bool(encoded.pair_mask[unit, index])
        ]
        if not typed:
            reason = ClosureReason.TASK_TYPE_MISSING
        elif not targeted:
            reason = ClosureReason.TARGET_MISSING
        elif not item_matched:
            reason = ClosureReason.ITEM_MISMATCH
        elif not pair_usable:
            reason = ClosureReason.PAIR_MASKED
        else:
            reason = ClosureReason.UNRESOLVED
        reasons[unit] = int(reason)
    return targets, active, closed, task_ids, reasons


def match_intents(
    encoded: EncodedTaskSet,
    intents: Sequence[IntentLabel],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible intent matching without the diagnostic arrays."""

    targets, active, closed, _, _ = match_intents_with_diagnostics(
        encoded, intents
    )
    return targets, active, closed


def closure_diagnostics(targets: BehaviorTargets) -> dict[str, Any]:
    """Aggregate exact closure and failure causes separately for each task."""

    active = targets.assignment_active.astype(np.bool_)
    task_ids = targets.assignment_task_ids
    reasons = targets.assignment_closure_reasons
    tasks: dict[str, dict[str, Any]] = {}
    for task in TaskType:
        if task == TaskType.NONE:
            continue
        mask = active & (task_ids == int(task))
        count = int(mask.sum())
        if not count:
            continue
        reason_counts = {
            reason.name: int((mask & (reasons == int(reason))).sum())
            for reason in ClosureReason
        }
        exact = reason_counts[ClosureReason.EXACT.name]
        tasks[task.name] = {
            "intents": count,
            "closed": exact,
            "closure_rate": exact / count,
            "reasons": reason_counts,
        }
    total = int(active.sum())
    exact = int((active & (reasons == int(ClosureReason.EXACT))).sum())
    return {
        "intents": total,
        "closed": exact,
        "closure_rate": exact / max(total, 1),
        "tasks": tasks,
    }


def _market_action_index(raw: Sequence[Any]) -> int:
    operation = str(raw[0]) if raw else "NONE"
    item = str(raw[1]) if len(raw) > 1 else None
    quantity = int(raw[2]) if len(raw) > 2 else 1
    if operation == "SELL" and item in PRODUCTS:
        key = (operation, item, 100)
    elif operation in ("HIRE", "BUY_LAND"):
        key = (operation, None, 1)
    else:
        legal_quantities = [
            value
            for value in _QUANTITIES
            if (operation, item, value) in MARKET_INDEX
        ]
        nearest = (
            min(legal_quantities, key=lambda value: abs(value - quantity))
            if legal_quantities
            else quantity
        )
        key = (operation, item, nearest)
    return MARKET_INDEX.get(key, 0)


def market_sequence_targets(
    action: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode ten orders, using an explicit STOP only when a slot remains."""

    targets = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)
    active = np.zeros(MAX_MARKET_ORDERS, dtype=np.bool_)
    orders = list(action.get("market", []) or [])[:MAX_MARKET_ORDERS]
    for slot, raw in enumerate(orders):
        if not isinstance(raw, Sequence) or not raw:
            break
        targets[slot] = _market_action_index(raw)
        active[slot] = True
    if len(orders) < MAX_MARKET_ORDERS:
        active[len(orders)] = True
        targets[len(orders)] = MARKET_INDEX[("NONE", None, 0)]
    return targets, active


def daily_budget_soft_targets(
    observations: Sequence[Any],
    actions: Sequence[Mapping[str, Any]],
    *,
    final_observation: Any | None = None,
) -> list[DailyBudgetSoftTarget]:
    """Infer one soft budget plan per day and repeat it for that day's steps.

    Purchases are replayed against each step's actual market state to recover
    realized cash spend.  Sales are counted only as available proceeds.  The
    target therefore teaches a reserve preference and category lower bounds,
    while leaving discretionary headroom for BC/RL to discover.
    """

    if len(observations) != len(actions):
        raise ValueError("observations and actions must have equal length")
    if not observations:
        return []

    def day_of(observation: Any) -> int:
        step = int(_get(observation, "step", 0) or 0)
        return int(_get(observation, "day", step // 24) or 0)

    def money_of(observation: Any) -> float:
        own, _ = _canonical_farms(observation)
        return max(float(_get(own, "money", 0.0) or 0.0), 0.0)

    day_indices: dict[int, list[int]] = {}
    for index, observation in enumerate(observations):
        day_indices.setdefault(day_of(observation), []).append(index)
    ordered_days = list(day_indices)
    targets: list[DailyBudgetSoftTarget | None] = [None] * len(observations)
    uniform = np.full(
        len(BUDGET_CATEGORIES), 1.0 / len(BUDGET_CATEGORIES), dtype=np.float32
    )
    for day_offset, day in enumerate(ordered_days):
        indices = day_indices[day]
        category_spend = np.zeros(len(BUDGET_CATEGORIES), dtype=np.float64)
        sale_proceeds = 0.0
        for index in indices:
            ledger = MarketBudget.from_observation(observations[index])
            for raw in list(actions[index].get("market", []) or [])[:MAX_MARKET_ORDERS]:
                if not isinstance(raw, Sequence) or not raw:
                    continue
                action_index = _market_action_index(raw)
                if action_index == 0:
                    continue
                before = ledger.money
                ledger.apply(action_index)
                delta = before - ledger.money
                category = market_budget_category(action_index)
                if category is not None and delta > 0:
                    category_spend[int(category)] += delta
                elif delta < 0:
                    sale_proceeds += -delta

        start_cash = money_of(observations[indices[0]])
        if day_offset + 1 < len(ordered_days):
            end_cash = money_of(observations[day_indices[ordered_days[day_offset + 1]][0]])
            confidence = 1.0
        elif final_observation is not None:
            end_cash = money_of(final_observation)
            confidence = 1.0
        else:
            end_cash = money_of(observations[indices[-1]])
            confidence = 0.5
        total_spend = float(category_spend.sum())
        cash_basis = max(
            start_cash + sale_proceeds,
            total_spend + end_cash,
            1.0,
        )
        spend_active = total_spend > 1e-6
        shares = (
            (category_spend / total_spend).astype(np.float32)
            if spend_active
            else uniform.copy()
        )
        target = DailyBudgetSoftTarget(
            day=day,
            reserve_fraction=float(np.clip(end_cash / cash_basis, 0.0, 1.0)),
            category_fractions=shares,
            category_floor_fractions=(category_spend / cash_basis).astype(np.float32),
            category_spend=category_spend.astype(np.float32),
            total_spend=total_spend,
            end_cash=end_cash,
            cash_basis=cash_basis,
            spend_active=spend_active,
            confidence=confidence,
        )
        for index in indices:
            targets[index] = target
    if any(target is None for target in targets):
        raise RuntimeError("failed to create an aligned daily budget target")
    return [target for target in targets if target is not None]


def behavior_targets(
    encoded: Sequence[EncodedTaskSet],
    intents: Sequence[Sequence[IntentLabel]],
    actions: Sequence[Mapping[str, Any]],
) -> BehaviorTargets:
    assignment = [
        match_intents_with_diagnostics(task_set, row)
        for task_set, row in zip(encoded, intents, strict=True)
    ]
    market = [market_sequence_targets(action) for action in actions]
    return BehaviorTargets(
        assignment_targets=np.stack([value[0] for value in assignment]),
        assignment_active=np.stack([value[1] for value in assignment]),
        assignment_closed=np.stack([value[2] for value in assignment]),
        assignment_task_ids=np.stack([value[3] for value in assignment]),
        assignment_closure_reasons=np.stack([value[4] for value in assignment]),
        market_targets=np.stack([value[0] for value in market]),
        market_active=np.stack([value[1] for value in market]),
    )


def market_teacher_masks(
    observations: Sequence[Any],
    targets: np.ndarray,
) -> np.ndarray:
    """Recreate the conservative autoregressive legality mask for BC."""

    masks = np.zeros(
        (len(observations), MAX_MARKET_ORDERS, len(MARKET_ACTIONS)),
        dtype=np.bool_,
    )
    budgets = [MarketBudget.from_observation(value) for value in observations]
    stopped = np.zeros(len(observations), dtype=np.bool_)
    for slot in range(MAX_MARKET_ORDERS):
        for row, budget in enumerate(budgets):
            if stopped[row]:
                masks[row, slot, 0] = True
                continue
            masks[row, slot] = budget.legal_mask()
            choice = int(targets[row, slot])
            # Expert traces are the authority for BC.  The local budget model is
            # deliberately conservative and may miss partial fills or simulator
            # details, so keep the demonstrated target trainable.
            masks[row, slot, choice] = True
            if choice == 0:
                stopped[row] = True
            else:
                budget.apply(choice)
    return masks
