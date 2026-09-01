"""Causal observation fingerprints for trajectory alignment.

Only information available to the acting player is used.  In particular, the
opponent's private shed and unit inventories never enter the fingerprint.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping


ITEMS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP",
)
BASE_PRICES = {
    "WHEAT": 30.0,
    "CARROT": 40.0,
    "TOMATO": 60.0,
    "STRAWBERRY": 120.0,
    "MELON": 250.0,
    "EGG": 50.0,
    "MILK": 160.0,
    "WOOL": 200.0,
    "FERTILIZER": 100.0,
}
QUADRANTS = ("NW", "NE", "SW", "SE")


def observation_day(observation: Any) -> int:
    """Read the public game day without relying on a synthetic ``step`` field."""
    try:
        return int(_get(observation, "day", 0) or 0)
    except (TypeError, ValueError):
        return 0


class LandStateTracker:
    """Track the first observed unlock day for each of the four own lands."""

    def __init__(self) -> None:
        self.unlock_days: dict[str, int] = {}

    def reset(self) -> None:
        self.unlock_days.clear()

    def update(self, observation: Any) -> dict[str, int]:
        obs = observation if isinstance(observation, dict) else dict(observation)
        player = int(obs.get("player", 0) or 0)
        farms = list(obs.get("farms", []) or [])
        own = farms[player] if player < len(farms) else {}
        day = observation_day(obs)
        for name in _get(own, "unlocked_quadrants", []) or []:
            text = str(name)
            if text in QUADRANTS:
                self.unlock_days.setdefault(text, day)
        return dict(self.unlock_days)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _tile_features(farm: dict[str, Any], prefix: str) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for row in _get(farm, "tiles", []) or []:
        for tile in row or []:
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind") or "EMPTY")
            counts[f"kind:{kind}"] += 1
            crop = tile.get("crop")
            animal = tile.get("animal")
            if crop:
                counts[f"crop:{crop}"] += 1
                counts["crop:yield"] += max(0, int(tile.get("yield_units", 0) or 0))
                counts["crop:dry"] += int(not tile.get("watered_today", False))
                counts["crop:weed_risk"] += int(tile.get("consecutive_unwatered", 0) or 0)
            if animal:
                counts[f"animal:{animal}"] += 1
                counts["animal:yield"] += max(0, int(tile.get("yield_units", 0) or 0))
                counts["animal:unfed"] += int(not tile.get("fed_today", False))
                counts["animal:uncared"] += int(not tile.get("cared_today", False))
                counts["animal:escape_risk"] += int(tile.get("consecutive_unfed", 0) or 0)
    return {f"{prefix}{key}": value / 100.0 for key, value in counts.items()}


def _quadrant_name(x: int, y: int) -> str:
    return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")


def _land_features(
    farm: dict[str, Any],
    *,
    day: int,
    unlock_days: Mapping[str, int] | None,
) -> dict[str, dict[str, float]]:
    """Return observable, per-quadrant phase/layout/workload features.

    Counts are normalized to one 5x5 land.  Prefixes deliberately separate
    phase, persistent layout, and today's workload so distance components stay
    interpretable.
    """
    unlocked = {str(value) for value in (_get(farm, "unlocked_quadrants", []) or [])}
    counts: dict[str, Counter[str]] = {name: Counter() for name in QUADRANTS}
    tiles = list(_get(farm, "tiles", []) or [])
    for y, row in enumerate(tiles):
        for x, raw_tile in enumerate(row or []):
            if x >= 10 or y >= 10:
                continue
            name = _quadrant_name(x, y)
            if not isinstance(raw_tile, dict):
                counts[name]["layout:empty"] += int(raw_tile is None)
                counts[name]["layout:locked"] += int(str(raw_tile) == "LOCKED")
                continue
            tile = raw_tile
            kind = str(tile.get("kind") or "EMPTY")
            counts[name][f"layout:kind:{kind}"] += 1
            crop = tile.get("crop")
            animal = tile.get("animal")
            if crop:
                counts[name][f"layout:crop:{crop}"] += 1
                counts[name]["work:crop_yield"] += max(0, int(tile.get("yield_units", 0) or 0))
                counts[name]["work:crop_dry"] += int(not tile.get("watered_today", False))
                counts[name]["work:crop_risk"] += max(
                    0, int(tile.get("consecutive_unwatered", 0) or 0)
                )
            if animal:
                counts[name][f"layout:animal:{animal}"] += 1
                counts[name]["work:animal_yield"] += max(
                    0, int(tile.get("yield_units", 0) or 0)
                )
                counts[name]["work:animal_unfed"] += int(not tile.get("fed_today", False))
                counts[name]["work:animal_uncared"] += int(not tile.get("cared_today", False))
                counts[name]["work:animal_risk"] += max(
                    0, int(tile.get("consecutive_unfed", 0) or 0)
                )

    unit_counts: Counter[str] = Counter()
    positions = [_get(farm, "farmer", None), *list(_get(farm, "hands", []) or [])]
    for position in positions:
        try:
            unit_counts[_quadrant_name(int(position[0]), int(position[1]))] += 1
        except (IndexError, TypeError, ValueError):
            continue

    result: dict[str, dict[str, float]] = {}
    for name in QUADRANTS:
        is_unlocked = name in unlocked
        first_day = None if unlock_days is None else unlock_days.get(name)
        values = {
            "phase:unlocked": float(is_unlocked),
            # Unknown ages remain zero and are ignored whenever only one side
            # has land-aware metadata (backward compatibility with v1 files).
            "phase:age": (
                max(0.0, min(1.0, (day - int(first_day)) / 30.0))
                if is_unlocked and first_day is not None
                else 0.0
            ),
            "work:units": unit_counts[name] / 12.0,
        }
        for key, value in counts[name].items():
            scale = 100.0 if key.endswith("yield") or key.endswith("risk") else 25.0
            values[key] = float(value) / scale
        result[name] = values
    return result


def _farm_features(farm: dict[str, Any], prefix: str) -> dict[str, float]:
    result = {
        f"{prefix}money": math.log1p(max(0.0, float(_get(farm, "money", 0.0) or 0.0))) / math.log1p(200_000.0),
        f"{prefix}hands": len(_get(farm, "hands", []) or []) / 12.0,
        f"{prefix}land": len(_get(farm, "unlocked_quadrants", []) or []) / 4.0,
        f"{prefix}hires_today": float(_get(farm, "hires_today", 0) or 0) / 12.0,
    }
    positions = [_get(farm, "farmer", [4, 4]), *list(_get(farm, "hands", []) or [])]
    for position in positions:
        try:
            x, y = int(position[0]), int(position[1])
        except (IndexError, TypeError, ValueError):
            continue
        key = f"{prefix}position:{x},{y}"
        result[key] = result.get(key, 0.0) + 1.0 / 12.0
    result.update(_tile_features(farm, prefix))
    return result


def observation_fingerprint(
    observation: Any,
    *,
    unlock_days: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return a compact JSON-serializable, causally valid state fingerprint."""
    obs = observation if isinstance(observation, dict) else dict(observation)
    player = int(obs.get("player", 0) or 0)
    farms = list(obs.get("farms", []) or [])
    own = farms[player] if player < len(farms) else {}
    opponent = farms[1 - player] if len(farms) >= 2 else {}
    private = dict(obs.get("private", {}) or {})
    numeric: dict[str, float] = {}
    numeric.update(_farm_features(own, "own:"))
    numeric.update(_farm_features(opponent, "opp:"))

    scale = math.log1p(100.0)
    seeds = dict(private.get("seeds", {}) or {})
    shed = dict(private.get("shed", {}) or {})
    carried: Counter[str] = Counter()
    for inventory in private.get("inventories", []) or []:
        for item, amount in dict(inventory or {}).items():
            carried[str(item)] += max(0, int(amount or 0))
    for item in ITEMS:
        numeric[f"own:seed:{item}"] = math.log1p(max(0, float(seeds.get(item, 0) or 0))) / scale
        numeric[f"own:shed:{item}"] = math.log1p(max(0, float(shed.get(item, 0) or 0))) / scale
        numeric[f"own:carried:{item}"] = math.log1p(max(0, float(carried.get(item, 0) or 0))) / scale

    market = dict(obs.get("market", {}) or {})
    inventories = dict(market.get("inventory", {}) or {})
    prices = dict(market.get("prices", {}) or {})
    for item, base_price in BASE_PRICES.items():
        numeric[f"market:inventory:{item}"] = (
            float(inventories.get(item, 10_000)) - 10_000.0
        ) / 1_000.0
        numeric[f"market:price:{item}"] = float(prices.get(item, base_price) or base_price) / base_price

    shops = tuple(str(value) for value in (obs.get("town", {}) or {}).get("unlocked_shops", []) or [])
    return {
        "step": int(obs.get("step", int(obs.get("day", 0) or 0) * 24 + int(obs.get("hour", 0) or 0)) or 0),
        "shops": shops,
        "numeric": numeric,
        "lands": _land_features(
            own,
            day=observation_day(obs),
            unlock_days=unlock_days,
        ),
    }


def _mean_absolute(left: dict[str, float], right: dict[str, float], prefix: str) -> float:
    keys = {key for key in left if key.startswith(prefix)} | {
        key for key in right if key.startswith(prefix)
    }
    if not keys:
        return 0.0
    return sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys) / len(keys)


def _land_distances(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_lands = dict(left.get("lands", {}) or {})
    right_lands = dict(right.get("lands", {}) or {})
    if not left_lands or not right_lands:
        return {"land": 0.0, "land_phase": 0.0, "land_layout": 0.0, "land_work": 0.0}

    def component(prefix: str) -> float:
        values = []
        for name in QUADRANTS:
            left_values = dict(left_lands.get(name, {}) or {})
            right_values = dict(right_lands.get(name, {}) or {})
            keys = {key for key in left_values if key.startswith(prefix)} | {
                key for key in right_values if key.startswith(prefix)
            }
            values.extend(
                abs(float(left_values.get(key, 0.0)) - float(right_values.get(key, 0.0)))
                for key in keys
            )
        return sum(values) / len(values) if values else 0.0

    phase = component("phase:")
    layout = component("layout:")
    work = component("work:")
    return {
        "land": 0.40 * phase + 0.40 * layout + 0.20 * work,
        "land_phase": phase,
        "land_layout": layout,
        "land_work": work,
    }


def fingerprint_distance(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    land_weight: float = 4.0,
) -> dict[str, float]:
    """Return interpretable distance components and their weighted total."""
    left_numeric = dict(left.get("numeric", {}) or {})
    right_numeric = dict(right.get("numeric", {}) or {})
    left_shops = tuple(left.get("shops", ()) or ())
    right_shops = tuple(right.get("shops", ()) or ())
    if left_shops == right_shops:
        town = 0.0
    else:
        union = set(left_shops) | set(right_shops)
        set_distance = len(set(left_shops) ^ set(right_shops)) / max(1, len(union))
        order_distance = float(left_shops[: len(right_shops)] != right_shops[: len(left_shops)])
        town = 0.75 * set_distance + 0.25 * order_distance
    own = _mean_absolute(left_numeric, right_numeric, "own:")
    opponent = _mean_absolute(left_numeric, right_numeric, "opp:")
    market = _mean_absolute(left_numeric, right_numeric, "market:")
    land_parts = _land_distances(left, right)
    total = (
        5.0 * own
        + 2.0 * town
        + 1.25 * opponent
        + 1.0 * market
        + float(land_weight) * land_parts["land"]
    )
    return {
        "total": total,
        "own": own,
        "town": town,
        "opponent": opponent,
        "market": market,
        **land_parts,
    }
