"""Experimental causal opponent-aware ranking for existing SELL slots.

This is the portable inference half of the replay-counterfactual linear ranker
from ``../meta_agent/runs/trade-order-supervised-v1/model-v1``.  The safety
wrapper is intentionally stricter than the original experiment: it may only
permute orders inside an already-contiguous SELL block.  It cannot change an
order's operation, quantity, slot phase, or cross a BUY/production barrier.

The opponent estimate consumes only the acting player's legal observation.
The old estimator remains in the project-level research directory while this
candidate is being evaluated; it must be copied into a promoted submission.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TRADE_ITEMS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
BASE_PRICES = {
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
ORDER_OPS = (
    "SELL", "BUY_PRODUCT", "HIRE", "BUY_LAND",
    "BUY_SEED", "BUY_ANIMAL", "OTHER",
)
ORDER_OP_INDEX = {name: index for index, name in enumerate(ORDER_OPS)}
ANIMAL_TO_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
SUPPORTED_OPS = frozenset(ORDER_OPS[:-1])
FEATURE_DIM = 35

# Exact float32 weights selected by the old held-out experiment.  Keeping the
# tiny artifact as source text makes the A/B candidate self-identifying and
# avoids silently loading a later checkpoint from another run directory.
LINEAR_WEIGHTS = np.asarray([
    0.19462795555591583, 0.013868379406630993, -0.033596210181713104,
    0.09924611449241638, -0.25102686882019043, -0.023119358345866203,
    0.0, -0.10217299312353134, -0.0652201771736145, 0.0,
    0.22302454710006714, -0.6229332685470581, 0.0,
    0.13881917297840118, 0.02448180690407753, 0.3383510112762451,
    -0.11665122956037521, 0.12261322140693665, 0.11375389248132706,
    -0.12275125086307526, 0.0, 0.0, -0.2084963321685791,
    0.43734031915664673, 1.0045280456542969, 0.4867108166217804,
    0.0, 0.028871387243270874, 0.028871387243270874,
    -0.16099287569522858, -0.527565062046051, -0.32683807611465454,
    -0.06564990431070328, 0.0, 0.0,
], dtype=np.float32)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _step(observation: Any) -> int:
    return int(_number(_get(observation, "step", 0)))


def _is_sell(order: Sequence[Any]) -> bool:
    return bool(order) and str(order[0]) == "SELL" and len(order) >= 3


def _opponent_farm(observation: Any) -> Mapping[str, Any]:
    seat = int(_number(_get(observation, "player", 0)))
    farms = list(_get(observation, "farms", []) or [])
    return farms[1 - seat] if len(farms) == 2 else {}


class CausalOpponentSellRanker:
    """Rank existing SELL runs from legal own/opponent-market evidence."""

    def __init__(self, shed_capacity: int = 100) -> None:
        self.shed_capacity = max(1, int(shed_capacity))
        self.estimator: Any | None = None
        self.estimate: Any | None = None
        self.player = -1
        self.last_step = -1
        self.eligible_turns = 0
        self.changed_turns = 0
        self.eligible_blocks = 0
        self.changed_blocks = 0

    @staticmethod
    def _estimator_class() -> Any:
        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
        from research.opponent_inventory.estimator import OpponentInventoryEstimator

        return OpponentInventoryEstimator

    def reset(self, observation: Any) -> None:
        estimator_class = self._estimator_class()
        self.player = int(_number(_get(observation, "player", 0)))
        self.estimator = estimator_class(
            observer_player=self.player,
            shed_capacity=self.shed_capacity,
        )
        self.estimate = self.estimator.reset(observation)
        self.last_step = _step(observation)
        self.eligible_turns = 0
        self.changed_turns = 0
        self.eligible_blocks = 0
        self.changed_blocks = 0

    def observe(self, observation: Any) -> None:
        step = _step(observation)
        player = int(_number(_get(observation, "player", 0)))
        if self.estimator is None or player != self.player or step == 0:
            self.reset(observation)
        elif step != self.last_step:
            self.estimate = self.estimator.update(observation)
            self.last_step = step

    def rerank_action(self, observation: Any, action: Mapping[str, Any]) -> dict[str, Any]:
        self.observe(observation)
        result = {
            "farmer": list(action.get("farmer") or ["PASS"]),
            "hands": [list(row or ["PASS"]) for row in action.get("hands", [])],
            "market": [list(row or ["PASS"]) for row in action.get("market", [])],
        }
        market = result["market"]
        scores = self._market_scores(observation, market)
        changed_turn = False
        eligible_turn = False
        start = 0
        while start < len(market):
            if not _is_sell(market[start]):
                start += 1
                continue
            stop = start + 1
            while stop < len(market) and _is_sell(market[stop]):
                stop += 1
            if stop - start > 1:
                eligible_turn = True
                self.eligible_blocks += 1
                before = [list(order) for order in market[start:stop]]
                market[start:stop] = sorted(
                    market[start:stop],
                    key=lambda order: -scores[id(order)],
                )
                if market[start:stop] != before:
                    self.changed_blocks += 1
                    changed_turn = True
            start = stop
        if eligible_turn:
            self.eligible_turns += 1
        if changed_turn:
            self.changed_turns += 1
        return result

    def _market_scores(
        self, observation: Any, market: Sequence[list[Any]],
    ) -> dict[int, float]:
        candidates = [
            (slot, order)
            for slot, order in enumerate(market[:10])
            if order and str(order[0]) in SUPPORTED_OPS
        ]
        count = len(candidates)
        if not count:
            return {}
        own = _get(observation, "farms", [])[
            int(_number(_get(observation, "player", 0)))
        ]
        money = max(0.0, _number(_get(own, "money", 0)))
        private = _get(observation, "private", {}) or {}
        shed = _get(private, "shed", {}) or {}
        shed_fill = sum(max(0.0, _number(value)) for value in shed.values()) \
            / float(self.shed_capacity)
        market_state = _get(observation, "market", {}) or {}
        prices = _get(market_state, "prices", {}) or {}
        inventory = _get(market_state, "inventory", {}) or {}
        opponent_money = max(0.0, _number(_get(_opponent_farm(observation), "money", 0)))
        step = _step(observation)
        estimate = self.estimate
        product_index = {item: index for index, item in enumerate(TRADE_ITEMS)}
        features = np.zeros((count, FEATURE_DIM), dtype=np.float32)
        for index, (slot, order) in enumerate(candidates):
            op = str(order[0])
            item = str(order[1]) if len(order) >= 2 else ""
            product = ANIMAL_TO_PRODUCT.get(item, item)
            quantity = max(1, int(_number(order[2], 1))) if len(order) >= 3 else 1
            features[index, ORDER_OP_INDEX.get(op, ORDER_OP_INDEX["OTHER"])] = 1.0
            if product in product_index:
                features[index, len(ORDER_OPS) + product_index[product]] = 1.0
            price = max(1.0, _number(prices.get(product, BASE_PRICES.get(product, 1))))
            if op == "BUY_SEED":
                price = float({
                    "WHEAT": 10, "CARROT": 20, "TOMATO": 50,
                    "STRAWBERRY": 100, "MELON": 80,
                }.get(item, 1))
            elif op == "BUY_ANIMAL":
                price = float({"GOOSE": 300, "COW": 400, "SHEEP": 500}.get(item, 1))
            signed_cash = quantity * price * (1.0 if op == "SELL" else -1.0)
            scalar = len(ORDER_OPS) + len(TRADE_ITEMS)
            features[index, scalar:scalar + 9] = np.asarray([
                min(2.0, quantity / 100.0),
                min(2.0, math.log1p(quantity) / math.log1p(100.0)),
                min(4.0, price / max(1.0, BASE_PRICES.get(product, price))),
                float(np.clip(signed_cash / 4000.0, -2.0, 2.0)),
                math.log1p(money) / math.log1p(200_000.0),
                min(2.0, shed_fill),
                float(op in {"HIRE", "BUY_LAND", "BUY_SEED", "BUY_ANIMAL"}),
                1.0 - index / max(1, count - 1),
                1.0 - slot / 9.0,
            ], dtype=np.float32)
            if product in product_index and estimate is not None:
                opponent_shed = float(estimate.shed.get(product, 0))
                lower, upper = estimate.shed_interval.get(product, (0, 0))
                total = float(estimate.total.get(product, 0))
                market_level = (
                    float(inventory.get(product, 10_000)) - 10_000.0
                ) / 2_000.0
                features[index, 25:] = np.asarray([
                    min(3.0, opponent_shed / self.shed_capacity),
                    min(3.0, float(lower) / self.shed_capacity),
                    min(3.0, float(upper) / self.shed_capacity),
                    min(3.0, max(0.0, float(upper) - float(lower)) / self.shed_capacity),
                    min(3.0, total / self.shed_capacity),
                    float(np.clip(market_level, -3.0, 3.0)),
                    float(bool(estimate.saturated.get(product, False))),
                    float(bool(estimate.ambiguous.get(product, False))),
                    math.log1p(opponent_money) / math.log1p(200_000.0),
                    min(1.0, step / 719.0),
                ], dtype=np.float32)
        values = features @ LINEAR_WEIGHTS
        return {
            id(order): float(values[index])
            for index, (_, order) in enumerate(candidates)
        }

    def stats(self) -> dict[str, int]:
        return {
            "eligible_turns": self.eligible_turns,
            "changed_turns": self.changed_turns,
            "eligible_blocks": self.eligible_blocks,
            "changed_blocks": self.changed_blocks,
        }
