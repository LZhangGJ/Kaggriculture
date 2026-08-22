"""Compact observable opponent histories for belief-conditioned policies."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .board_policy import _canonical_farms
from .gpu_policy import PRODUCTS, _get, _mapping


OPPONENT_HISTORY_LAGS = (120, 72, 24, 0)
OPPONENT_FEATURE_NAMES = (
    "step",
    "day",
    "hour",
    "own_money",
    "opponent_money",
    "money_advantage",
    "own_workers",
    "opponent_workers",
    "own_unlocked",
    "opponent_unlocked",
    "opponent_crop_tiles",
    "opponent_structures",
    "opponent_animals",
    "opponent_watered",
    "opponent_fertilized",
    "opponent_fed",
    "opponent_yield",
    "opponent_empty_tiles",
    "own_crop_tiles",
    "own_animal_tiles",
    "market_price_mean",
    "market_price_std",
    "market_inventory_pressure",
    "bias",
)
OPPONENT_FEATURES = len(OPPONENT_FEATURE_NAMES)


def _farm_summary(farm: Any) -> dict[str, float]:
    crops = structures = animals = watered = fertilized = fed = empty = 0
    yield_units = 0
    for row in list(_get(farm, "tiles", []) or []):
        for raw_tile in list(row or []):
            tile = _mapping(raw_tile)
            if not tile or tile.get("kind") is None:
                empty += 1
                continue
            kind = str(tile.get("kind") or "")
            if tile.get("crop") or kind == "CROP":
                crops += 1
            if kind in ("COOP", "PASTURE"):
                structures += 1
            if tile.get("animal"):
                animals += 1
            watered += int(bool(tile.get("watered", tile.get("watered_today", False))))
            fertilized += int(bool(tile.get("fertilized", False)))
            fed += int(bool(tile.get("fed_today", False)))
            yield_units += max(int(tile.get("yield_units", 0) or 0), 0)
    return {
        "crops": crops / 100.0,
        "structures": structures / 100.0,
        "animals": animals / 100.0,
        "watered": watered / 100.0,
        "fertilized": fertilized / 100.0,
        "fed": fed / 100.0,
        "yield": min(yield_units / 500.0, 2.0),
        "empty": empty / 100.0,
    }


def opponent_public_features(observation: Any) -> np.ndarray:
    """Encode only information available against an anonymous opponent."""

    own, opponent = _canonical_farms(observation)
    own_summary = _farm_summary(own)
    opponent_summary = _farm_summary(opponent)
    own_money = max(float(_get(own, "money", 0.0) or 0.0), 0.0)
    opponent_money = max(float(_get(opponent, "money", 0.0) or 0.0), 0.0)
    money_scale = np.log1p(2_000_000.0)
    own_log = np.log1p(own_money) / money_scale
    opponent_log = np.log1p(opponent_money) / money_scale
    step = int(_get(observation, "step", 0) or 0)
    day = int(_get(observation, "day", step // 24) or 0)
    hour = int(_get(observation, "hour", step % 24) or 0)
    market = _get(observation, "market", {}) or {}
    prices = _mapping(_get(market, "prices", {}))
    inventory = _mapping(_get(market, "inventory", {}))
    price_values = np.asarray(
        [max(float(prices.get(item, 0.0) or 0.0), 0.0) for item in PRODUCTS],
        dtype=np.float64,
    )
    inventory_values = np.asarray(
        [float(inventory.get(item, 10_000) or 10_000) for item in PRODUCTS],
        dtype=np.float64,
    )
    values = np.asarray(
        (
            step / 719.0,
            day / 30.0,
            hour / 23.0,
            own_log,
            opponent_log,
            own_log - opponent_log,
            len(_get(own, "hands", []) or []) / 8.0,
            len(_get(opponent, "hands", []) or []) / 8.0,
            len(_get(own, "unlocked_quadrants", []) or []) / 4.0,
            len(_get(opponent, "unlocked_quadrants", []) or []) / 4.0,
            opponent_summary["crops"],
            opponent_summary["structures"],
            opponent_summary["animals"],
            opponent_summary["watered"],
            opponent_summary["fertilized"],
            opponent_summary["fed"],
            opponent_summary["yield"],
            opponent_summary["empty"],
            own_summary["crops"],
            own_summary["animals"],
            float(np.log1p(price_values).mean() / np.log1p(1_000.0)),
            float(np.log1p(price_values).std() / np.log1p(1_000.0)),
            float(np.abs(inventory_values - 10_000.0).mean() / 10_000.0),
            1.0,
        ),
        dtype=np.float32,
    )
    if values.shape != (OPPONENT_FEATURES,) or not np.isfinite(values).all():
        raise ValueError("invalid observable opponent feature vector")
    return values


def build_opponent_history_sequences(
    observations: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Build [step, lag, feature] histories at 0/24/72/120-step lookbacks."""

    features = np.stack([opponent_public_features(row) for row in observations])
    histories = np.zeros(
        (len(observations), len(OPPONENT_HISTORY_LAGS), OPPONENT_FEATURES),
        dtype=np.float32,
    )
    mask = np.zeros(
        (len(observations), len(OPPONENT_HISTORY_LAGS)), dtype=np.bool_
    )
    for step in range(len(observations)):
        for slot, lag in enumerate(OPPONENT_HISTORY_LAGS):
            source = step - lag
            if source >= 0:
                histories[step, slot] = features[source]
                mask[step, slot] = True
    return histories, mask


def runtime_opponent_history(memory: Any, observation: Any) -> tuple[np.ndarray, np.ndarray]:
    """Update a policy memory and select the same lagged history used offline."""

    step = int(_get(observation, "step", 0) or 0)
    feature = opponent_public_features(observation)
    rows = memory.opponent_observation_history
    if not rows or int(rows[-1][0]) != step:
        rows.append((step, tuple(float(value) for value in feature)))
        if len(rows) > max(OPPONENT_HISTORY_LAGS) + 1:
            del rows[: -(max(OPPONENT_HISTORY_LAGS) + 1)]
    lookup = {int(row_step): values for row_step, values in rows}
    history = np.zeros(
        (len(OPPONENT_HISTORY_LAGS), OPPONENT_FEATURES), dtype=np.float32
    )
    mask = np.zeros(len(OPPONENT_HISTORY_LAGS), dtype=np.bool_)
    for slot, lag in enumerate(OPPONENT_HISTORY_LAGS):
        values = lookup.get(step - lag)
        if values is not None:
            history[slot] = np.asarray(values, dtype=np.float32)
            mask[slot] = True
    return history, mask
