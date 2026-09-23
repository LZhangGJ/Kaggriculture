#!/usr/bin/env python3
"""Training-only, cross-fitted day-state control variate for v3 PPO.

This module deliberately has no dependency on the live actor or production
agent.  It consumes only the pre-action day states already stored by native
PPO rollout and returns out-of-fold, day-aligned advantages.
"""

from __future__ import annotations

import hashlib
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


ITEMS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP",
)
CROPS = ITEMS[:5]
ANIMALS = ITEMS[9:]
SHOPS = (
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
    "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE",
)
STUDENT_STEPS = tuple(range(288, 673, 24))
OBSERVATION_WIDTH = 3074
REWARD_MIN, REWARD_MAX = -1.1, 1.1
DISCRETE_TOLERANCE = 2e-3


def _feature_names() -> tuple[str, ...]:
    names = ["day", "remaining_day", "cash_difference", "cash_total"]
    for owner in ("own", "rival"):
        names.extend((
            f"{owner}.money", f"{owner}.unlock_mask",
            f"{owner}.hires_today", f"{owner}.farmer_x",
            f"{owner}.farmer_y",
        ))
        names.extend(f"{owner}.kind_{kind}.count" for kind in range(7))
        for crop in CROPS:
            names.extend((
                f"{owner}.crop.{crop}.count",
                f"{owner}.crop.{crop}.yield_sum",
                f"{owner}.crop.{crop}.age_sum",
                f"{owner}.crop.{crop}.unwatered_sum",
            ))
        for animal in ANIMALS:
            names.extend((
                f"{owner}.animal.{animal}.count",
                f"{owner}.animal.{animal}.yield_sum",
                f"{owner}.animal.{animal}.age_sum",
                f"{owner}.animal.{animal}.unfed_sum",
                f"{owner}.animal.{animal}.care_sum",
            ))
        names.extend((
            f"{owner}.status.fertilized_count",
            f"{owner}.status.watered_count",
            f"{owner}.status.fed_count",
            f"{owner}.status.cared_count",
            f"{owner}.status.fertilizer_available_count",
        ))
    names.extend(f"private.shed.{item}" for item in ITEMS)
    names.extend(f"private.seed.{crop}" for crop in CROPS)
    names.extend(f"private.hand.{item}" for item in ITEMS)
    names.extend(f"market.inventory.{item}" for item in ITEMS[:9])
    names.extend(f"market.price.{item}" for item in ITEMS[:9])
    names.extend(f"town.shop.{shop}" for shop in SHOPS)
    return tuple(names)


def _names_sha256(names: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


DAY_STATE_FEATURE_NAMES = _feature_names()
CONTROL_VARIATE_FEATURE_NAMES = ("peer_return_mean", *DAY_STATE_FEATURE_NAMES)
# Literals are filled from the reviewed schema and asserted at import.  A
# feature rename/reorder must therefore be an explicit contract change.
DAY_STATE_FEATURE_SCHEMA_SHA256 = (
    "05518c043d23a2be45e060bc2755bf51c3bf8ba9921e548afaa4022c23b962dd")
CONTROL_VARIATE_FEATURE_SCHEMA_SHA256 = (
    "8d69d599415553cffb5c37a1f5d46a9ae5a6396d432a584fbb929569830d7a99")
if len(DAY_STATE_FEATURE_NAMES) != 163:
    raise RuntimeError("day-state feature contract is not 163-dimensional")
if (_names_sha256(DAY_STATE_FEATURE_NAMES) !=
        DAY_STATE_FEATURE_SCHEMA_SHA256 or
        _names_sha256(CONTROL_VARIATE_FEATURE_NAMES) !=
        CONTROL_VARIATE_FEATURE_SCHEMA_SHA256):
    raise RuntimeError("day-state feature schema hash drift")
if any(name.lower().split(".", 1)[0] in (
        "seed", "environment_seed", "opponent", "seat", "route",
        "policy_seed") for name in CONTROL_VARIATE_FEATURE_NAMES):
    raise RuntimeError("identity field entered the control-variate forward schema")


def _as_numpy(value, dtype=np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _integers(values: np.ndarray, name: str) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"non-finite packed field: {name}")
    rounded = np.rint(values)
    error = float(np.max(np.abs(values - rounded), initial=0.0))
    if error > DISCRETE_TOLERANCE:
        raise RuntimeError(f"non-integral packed field {name}: max error {error}")
    return rounded.astype(np.int32)


def _farm_features(raw: np.ndarray, base: int, days: np.ndarray) -> np.ndarray:
    count = len(raw)
    result = np.empty((count, 52), dtype=np.float32)
    header = raw[:, base:base + 6]
    board = raw[:, base + 6:base + 1506].reshape(count, 100, 15)
    board_i = _integers(board, f"farm@{base}.tiles")
    farm_i = _integers(header[:, 1:], f"farm@{base}.header")

    kinds, crop_ids, animal_ids = (
        board_i[:, :, 0], board_i[:, :, 1], board_i[:, :, 2])
    if (np.any((kinds < 0) | (kinds > 6)) or
            np.any((crop_ids < -1) | (crop_ids > 4)) or
            np.any(~np.isin(animal_ids, (-1, 9, 10, 11))) or
            np.any((board_i[:, :, 11:] < 0) | (board_i[:, :, 11:] > 1))):
        raise RuntimeError(f"invalid packed tile enum/status at farm base {base}")

    result[:, 0] = raw[:, base]
    result[:, 1] = farm_i[:, 3]  # unlock mask
    result[:, 2] = farm_i[:, 4]  # hires today
    result[:, 3] = farm_i[:, 0]  # farmer x
    result[:, 4] = farm_i[:, 1]  # farmer y
    column = 5
    for kind in range(7):
        result[:, column] = np.sum(kinds == kind, axis=1)
        column += 1
    day_matrix = days[:, None]
    for crop in range(5):
        mask = (kinds == 3) & (crop_ids == crop)
        result[:, column] = np.sum(mask, axis=1)
        result[:, column + 1] = np.sum(board_i[:, :, 5] * mask, axis=1)
        result[:, column + 2] = np.sum(
            (day_matrix - board_i[:, :, 3]) * mask, axis=1)
        result[:, column + 3] = np.sum(board_i[:, :, 6] * mask, axis=1)
        column += 4
    for animal in (9, 10, 11):
        mask = (kinds == 6) & (animal_ids == animal)
        result[:, column] = np.sum(mask, axis=1)
        result[:, column + 1] = np.sum(board_i[:, :, 5] * mask, axis=1)
        result[:, column + 2] = np.sum(
            (day_matrix - board_i[:, :, 4]) * mask, axis=1)
        result[:, column + 3] = np.sum(board_i[:, :, 7] * mask, axis=1)
        result[:, column + 4] = np.sum(board_i[:, :, 9] * mask, axis=1)
        column += 5
    result[:, column] = np.sum(board_i[:, :, 8] >= day_matrix, axis=1)
    result[:, column + 1] = np.sum(board_i[:, :, 11], axis=1)
    result[:, column + 2] = np.sum(board_i[:, :, 12], axis=1)
    result[:, column + 3] = np.sum(board_i[:, :, 13], axis=1)
    result[:, column + 4] = np.sum(board_i[:, :, 14], axis=1)
    if column + 5 != result.shape[1]:
        raise AssertionError("farm feature width drift")
    return result


def extract_day_state_features(
        normalized_observation: np.ndarray,
        normalized_observation_length: np.ndarray,
        day_steps: np.ndarray,
        normalization: dict,
        *, chunk_size: int = 1024) -> np.ndarray:
    """Extract the frozen 163D pre-action state from native PPO arrays."""
    observation = np.asarray(normalized_observation, dtype=np.float32)
    lengths = np.asarray(normalized_observation_length, dtype=np.float64).reshape(-1)
    steps = np.asarray(day_steps, dtype=np.int64).reshape(-1)
    if (observation.ndim != 2 or observation.shape[1] != OBSERVATION_WIDTH or
            len(observation) != len(lengths) or len(observation) != len(steps) or
            not len(observation) or chunk_size <= 0):
        raise ValueError("invalid native day-state array shapes")
    if not np.all(np.isin(steps, STUDENT_STEPS)):
        raise RuntimeError("day-state steps are outside the post-288 actor scope")

    mean = _as_numpy(normalization.get("observation_mean"))
    std = _as_numpy(normalization.get("observation_std"))
    length_mean = float(_as_numpy(
        normalization.get("observation_length_mean")).reshape(-1)[0])
    length_std = float(_as_numpy(
        normalization.get("observation_length_std")).reshape(-1)[0])
    if (mean.shape != (OBSERVATION_WIDTH,) or
            std.shape != (OBSERVATION_WIDTH,) or
            not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or
            np.any(std <= 0) or not math.isfinite(length_mean) or
            not math.isfinite(length_std) or length_std <= 0):
        raise RuntimeError("invalid behavior-checkpoint observation normalization")

    raw_lengths = _integers(
        lengths * length_std + length_mean, "observation_length").reshape(-1)
    output = np.empty((len(observation), len(DAY_STATE_FEATURE_NAMES)),
                      dtype=np.float32)
    for offset in range(0, len(observation), chunk_size):
        stop = min(len(observation), offset + chunk_size)
        raw = (observation[offset:stop].astype(np.float64) * std + mean)
        if not np.all(np.isfinite(raw)):
            raise RuntimeError("non-finite de-normalized packed observation")
        step = steps[offset:stop]
        day = step // 24
        header = _integers(raw[:, :4], "clock/player")
        if (not np.array_equal(header[:, 0], step) or
                not np.array_equal(header[:, 1], day) or
                np.any(header[:, 2] != 0) or np.any(header[:, 3] != 0)):
            raise RuntimeError("canonical day-start clock/player contract drift")
        fixed = _integers(
            raw[:, [7, 9, 1513, 1515, 3033, 3046]], "day-start fixed fields")
        if (np.any(fixed[:, :4] != 0) or np.any(fixed[:, 4] != 1) or
                np.any(fixed[:, 5] != 0)):
            raise RuntimeError("day-start hands/hires/inventory contract drift")

        shops = _integers(raw[:, 3065:3074], "shops")
        shop_counts = shops[:, 0]
        if np.any((shop_counts < 0) | (shop_counts > 8)):
            raise RuntimeError("invalid unlocked shop count")
        expected_lengths = 3066 + shop_counts
        if not np.array_equal(raw_lengths[offset:stop], expected_lengths):
            raise RuntimeError("packed observation length/shop count mismatch")
        positions = np.arange(8)[None, :] < shop_counts[:, None]
        shop_ids = shops[:, 1:]
        if np.any(positions & ((shop_ids < 0) | (shop_ids > 7))):
            raise RuntimeError("invalid unlocked shop id")

        own = _farm_features(raw, 4, day)
        rival = _farm_features(raw, 1510, day)
        values = output[offset:stop]
        values[:, 0] = day
        values[:, 1] = 29 - day
        values[:, 2] = own[:, 0] - rival[:, 0]
        values[:, 3] = own[:, 0] + rival[:, 0]
        values[:, 4:56] = own
        values[:, 56:108] = rival
        private_market = _integers(
            raw[:, 3016:3065], "private/market")
        values[:, 108:137] = np.concatenate((
            private_market[:, 0:12], private_market[:, 12:17],
            private_market[:, 18:30]), axis=1)
        values[:, 137:155] = private_market[:, 31:49]
        shop_features = values[:, 155:163]
        for shop in range(8):
            shop_features[:, shop] = np.any(
                positions & (shop_ids == shop), axis=1)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("non-finite day-state feature")
    return output


def same_seed_peer_baseline(seeds: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    seeds = np.asarray(seeds).reshape(-1)
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if len(seeds) != len(rewards) or not len(seeds) or not np.all(np.isfinite(rewards)):
        raise ValueError("invalid peer-baseline inputs")
    unique, inverse, counts = np.unique(seeds, return_inverse=True,
                                        return_counts=True)
    if len(unique) < 2 or np.any(counts < 2) or np.any(counts != counts[0]):
        raise RuntimeError("peer baseline requires equal complete multi-game seed blocks")
    sums = np.bincount(inverse, weights=rewards, minlength=len(unique))
    return (sums[inverse] - rewards) / (counts[inverse] - 1)


def whole_seed_folds(seeds: np.ndarray, n_splits: int = 6) -> np.ndarray:
    seeds = np.asarray(seeds).reshape(-1)
    unique, inverse = np.unique(seeds, return_inverse=True)
    if n_splits < 2 or len(unique) < n_splits:
        raise RuntimeError("not enough whole environment-seed groups for cross-fit")
    # Sorted round-robin is deterministic and exactly balanced for contiguous
    # formal seed blocks.  Numeric seed never enters a model input.
    unique_folds = np.arange(len(unique), dtype=np.int32) % n_splits
    return unique_folds[inverse]


def actionable_day_counts(
        day_count: int, event_day_index: np.ndarray,
        event_legal_mask: np.ndarray) -> np.ndarray:
    event_days = np.asarray(event_day_index, dtype=np.int64).reshape(-1)
    masks = np.asarray(event_legal_mask, dtype=np.uint32).reshape(-1)
    if (day_count <= 0 or len(event_days) != len(masks) or
            np.any((event_days < 0) | (event_days >= day_count))):
        raise ValueError("invalid event/day arrays")
    lookup = np.asarray([value.bit_count() for value in range(1 << 11)],
                        dtype=np.uint8)
    if np.any(masks >= len(lookup)):
        raise RuntimeError("event legal mask exceeds the 11-class contract")
    counts = np.zeros(day_count, dtype=np.int32)
    active_days = event_days[lookup[masks] > 1]
    np.add.at(counts, active_days, 1)
    return counts


def _variance(values: np.ndarray) -> float:
    value = float(np.var(np.asarray(values, dtype=np.float64)))
    if not math.isfinite(value):
        raise RuntimeError("non-finite variance")
    return value


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (values.shape != weights.shape or np.any(weights < 0) or
            not float(weights.sum())):
        raise ValueError("invalid variance weights")
    mean = float(np.average(values, weights=weights))
    return float(np.average((values - mean) ** 2, weights=weights))


@dataclass(frozen=True)
class CrossFitControlVariate:
    predictions: np.ndarray
    advantages: np.ndarray
    peer_baseline: np.ndarray
    game_folds: np.ndarray
    metrics: dict


def crossfit_global_hgb(
        features: np.ndarray,
        day_session_index: np.ndarray,
        day_steps: np.ndarray,
        seeds: np.ndarray,
        rewards: np.ndarray,
        *, actionable_counts: np.ndarray | None = None,
        n_splits: int = 6, max_workers: int = 6) -> CrossFitControlVariate:
    """Fit six seed-held-out global HGBs and return frozen OOF advantages."""
    features = np.asarray(features, dtype=np.float32)
    sessions = np.asarray(day_session_index, dtype=np.int64).reshape(-1)
    steps = np.asarray(day_steps, dtype=np.int64).reshape(-1)
    seeds = np.asarray(seeds).reshape(-1)
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    game_count = len(rewards)
    if (features.shape != (len(sessions), len(DAY_STATE_FEATURE_NAMES)) or
            len(steps) != len(sessions) or not len(sessions) or
            len(seeds) != game_count or
            np.any((sessions < 0) | (sessions >= game_count)) or
            not np.all(np.isfinite(features)) or
            not np.all(np.isfinite(rewards)) or
            np.any(rewards < REWARD_MIN - 1e-6) or
            np.any(rewards > REWARD_MAX + 1e-6) or max_workers <= 0):
        raise ValueError("invalid cross-fit arrays")
    if not np.all(np.isin(steps, STUDENT_STEPS)):
        raise RuntimeError("cross-fit contains a non-student day")
    days_per_game = np.bincount(sessions, minlength=game_count)
    if np.any(days_per_game != len(STUDENT_STEPS)):
        raise RuntimeError("cross-fit requires all 17 day states per game")
    for session in range(game_count):
        if not np.array_equal(np.sort(steps[sessions == session]), STUDENT_STEPS):
            raise RuntimeError("game day-step coverage drift")

    peer = same_seed_peer_baseline(seeds, rewards)
    game_folds = whole_seed_folds(seeds, n_splits)
    day_folds = game_folds[sessions]
    targets = rewards[sessions]
    inputs = np.empty((len(features), len(CONTROL_VARIATE_FEATURE_NAMES)),
                      dtype=np.float32)
    inputs[:, 0] = peer[sessions]
    inputs[:, 1:] = features
    predictions = np.full(len(inputs), np.nan, dtype=np.float64)

    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    def fit_one(fold: int):
        train = day_folds != fold
        held = ~train
        if not np.any(train) or not np.any(held):
            raise RuntimeError(f"empty cross-fit partition {fold}")
        model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=100,
            max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=5.0,
            early_stopping=False, random_state=8000 + fold)
        model.fit(inputs[train], targets[train])
        return fold, np.flatnonzero(held), model.predict(inputs[held])

    fit_started = time.perf_counter()
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=min(n_splits, max_workers)) as pool:
            for fold, held, values in pool.map(fit_one, range(n_splits)):
                if np.any(day_folds[held] != fold):
                    raise AssertionError("cross-fit held rows changed fold")
                predictions[held] = values
    fit_seconds = time.perf_counter() - fit_started
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("cross-fit did not predict every held-out day")
    unclipped = predictions.copy()
    predictions = np.clip(predictions, REWARD_MIN, REWARD_MAX)
    advantages = targets - predictions
    current = targets - peer[sessions]
    if actionable_counts is None:
        valid = np.ones(len(features), dtype=bool)
        counts = np.ones(len(features), dtype=np.float64)
    else:
        counts = np.asarray(actionable_counts, dtype=np.float64).reshape(-1)
        if len(counts) != len(features) or np.any(counts < 0):
            raise ValueError("invalid actionable day counts")
        valid = counts > 0
        if not np.any(valid):
            raise RuntimeError("cross-fit rollout has no actionable day")
    denominator = _variance(current[valid])
    if denominator <= 0:
        raise RuntimeError("current advantage has zero variance")

    fold_ratios, step_ratios = [], {}
    for fold in range(n_splits):
        selected = valid & (day_folds == fold)
        fold_denominator = _variance(current[selected])
        fold_ratios.append(
            _variance(advantages[selected]) / fold_denominator
            if fold_denominator > 0 else None)
    for step in STUDENT_STEPS:
        selected = valid & (steps == step)
        step_denominator = _variance(current[selected])
        step_ratios[str(step)] = (
            _variance(advantages[selected]) / step_denominator
            if step_denominator > 0 else None)
    weighted_denominator = _weighted_variance(current[valid], counts[valid])
    metrics = {
        "method": "whole_seed_6fold_global_hgb_pre_day_state",
        "day_state_feature_count": len(DAY_STATE_FEATURE_NAMES),
        "day_state_feature_schema_sha256": DAY_STATE_FEATURE_SCHEMA_SHA256,
        "model_feature_count": len(CONTROL_VARIATE_FEATURE_NAMES),
        "model_feature_schema_sha256": CONTROL_VARIATE_FEATURE_SCHEMA_SHA256,
        "folds": n_splits,
        "workers": min(n_splits, max_workers),
        "fit_seconds": fit_seconds,
        "games": game_count,
        "environment_seeds": int(len(np.unique(seeds))),
        "days": len(features),
        "actionable_days": int(np.sum(valid)),
        "current_advantage_variance": denominator,
        "crossfit_advantage_variance": _variance(advantages[valid]),
        "variance_ratio": _variance(advantages[valid]) / denominator,
        "slot_weighted_variance_ratio": (
            _weighted_variance(advantages[valid], counts[valid]) /
            weighted_denominator if weighted_denominator > 0 else None),
        "fold_variance_ratios": fold_ratios,
        "step_variance_ratios": step_ratios,
        "prediction_min_unclipped": float(unclipped.min()),
        "prediction_max_unclipped": float(unclipped.max()),
        "prediction_abs_p99_unclipped": float(
            np.quantile(np.abs(unclipped), 0.99)),
        "prediction_clip_fraction": float(np.mean(predictions != unclipped)),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
    }
    return CrossFitControlVariate(
        predictions=predictions, advantages=advantages,
        peer_baseline=peer, game_folds=game_folds, metrics=metrics)
