#!/usr/bin/env python3
"""Select robust shallow route-switch trees from paired counterfactual searches."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from meta_agent.src.fingerprints import ITEMS
from meta_agent.src.recurrent_meta import (
    ANIMALS,
    BASE_PRICES,
    CROPS,
    PLAN_HORIZONS,
    QUADRANTS,
    STRUCTURES,
    TRADE_ITEMS,
)


def _public_names(prefix: str) -> list[str]:
    names = [
        f"{prefix}_money_log", f"{prefix}_hands", f"{prefix}_land",
        f"{prefix}_hires_today", f"{prefix}_farmer_x", f"{prefix}_farmer_y",
    ]
    crop_fields = ("count", "yield", "needs_water", "unwatered", "harvestable", "fertilized")
    animal_fields = ("count", "yield", "needs_feed", "needs_care", "unfed", "harvestable")
    for item in CROPS:
        names.extend(f"{prefix}_{item.lower()}_{field}" for field in crop_fields)
    for item in ANIMALS:
        names.extend(f"{prefix}_{item.lower()}_{field}" for field in animal_fields)
    names.extend(f"{prefix}_{item.lower()}_count" for item in STRUCTURES)
    for quadrant in QUADRANTS:
        names.extend(
            f"{prefix}_{quadrant.lower()}_{field}"
            for field in ("unlocked", "empty", "crops", "animals", "workers")
        )
    return names


def _private_names() -> list[str]:
    names = [f"self_shed_{item.lower()}_log" for item in ITEMS]
    names.extend(f"self_carried_{item.lower()}_log" for item in ITEMS)
    names.extend(f"self_seed_{item.lower()}_log" for item in CROPS)
    names.extend(("self_shed_total", "self_carried_total", "self_shed_near_full"))
    for horizon in PLAN_HORIZONS:
        names.extend(
            [
                f"plan_{horizon}_hire", f"plan_{horizon}_buy_land",
                f"plan_{horizon}_buy_seed", f"plan_{horizon}_buy_product",
            ]
        )
        names.extend(f"plan_{horizon}_sell_{item.lower()}" for item in TRADE_ITEMS)
    return names


def _market_names() -> list[str]:
    names = []
    for item in TRADE_ITEMS:
        names.extend((f"market_{item.lower()}_inventory", f"market_{item.lower()}_price"))
    names.extend(f"shop_hash_{index}" for index in range(16))
    names.extend(("step", "day", "hour"))
    return names


def feature_names() -> list[str]:
    from meta_agent.src.route_switch_features import route_switch_feature_names

    return route_switch_feature_names()


def _load_groups(paths: list[Path]) -> tuple[dict[tuple[str, int], list[tuple]], list[str], list[str]]:
    grouped: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    all_openings: list[str] = []
    all_targets: list[str] = []
    for path in paths:
        with np.load(path) as saved:
            games = saved["games"]
            states = saved["states"]
            openings = saved["openings"].astype(str).tolist()
            targets = saved["targets"].astype(str).tolist()
            opponents = saved["opponents"].astype(str).tolist()
        for value in openings:
            if value not in all_openings:
                all_openings.append(value)
        for value in targets:
            if value not in all_targets:
                all_targets.append(value)
        valid = np.flatnonzero(games[:, -1] == 0)
        for index in valid:
            row = games[index]
            opening = openings[int(row[1])]
            target = targets[int(row[2])]
            opponent = opponents[int(row[3])]
            checkpoint = int(row[4])
            grouped[(opening, checkpoint)].append(
                (
                    opponent, int(row[5]), int(row[6]), target,
                    float(row[10]), float(row[9]), states[index].astype(np.float32),
                )
            )
    return grouped, all_openings, all_targets


def _load_compact_groups(paths: list[Path]) -> tuple[dict[tuple[str, int], tuple], list[str], list[str]]:
    """Load native compact tensors without duplicating states for every target."""
    grouped: dict[tuple[str, int], tuple] = {}
    all_openings: list[str] = []
    all_targets: list[str] = []
    for path in paths:
        with np.load(path) as saved:
            outcome = saved["outcome"].astype(np.float32) * 0.5
            margins = saved["margin"].astype(np.float32)
            states = saved["states"].astype(np.float32)
            openings = saved["openings"].astype(str).tolist()
            targets = saved["targets"].astype(str).tolist()
            opponents = saved["opponents"].astype(str)
            checkpoints = saved["checkpoints"].astype(int).tolist()
            seed_values = saved["seeds"].astype(np.int64)
        for value in openings:
            if value not in all_openings:
                all_openings.append(value)
        for value in targets:
            if value not in all_targets:
                all_targets.append(value)
        opponent_grid = np.broadcast_to(
            opponents[:, None, None], (len(opponents), len(seed_values), 2)
        ).reshape(-1)
        seed_grid = np.broadcast_to(
            seed_values[None, :, None], (len(opponents), len(seed_values), 2)
        ).reshape(-1)
        for oi, opening in enumerate(openings):
            for ci, checkpoint in enumerate(checkpoints):
                matrix = states[oi, ci].reshape(-1, states.shape[-1])
                node_scores = outcome[oi, ci].reshape(len(targets), -1).T
                node_margins = margins[oi, ci].reshape(len(targets), -1).T
                quality = node_scores.astype(np.float64) * 1e12 + node_margins
                labels = np.asarray(targets)[np.argmax(quality, axis=1)]
                node_outcomes = [
                    {
                        target: (float(node_scores[sample, target_index]),
                                 float(node_margins[sample, target_index]))
                        for target_index, target in enumerate(targets)
                    }
                    for sample in range(len(matrix))
                ]
                grouped[(opening, checkpoint)] = (
                    matrix, labels, seed_grid.copy(), opponent_grid.copy(), node_outcomes
                )
    return grouped, all_openings, all_targets


def _samples(rows: list[tuple]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    list[dict[str, tuple[float, float]]],
]:
    pairs: dict[tuple[str, int, int], list[tuple]] = defaultdict(list)
    for row in rows:
        pairs[(row[0], row[1], row[2])].append(row)
    matrix, labels, seeds, opponents, outcomes = [], [], [], [], []
    for (opponent, seed, _), variants in sorted(pairs.items()):
        by_target = {row[3]: (row[4], row[5]) for row in variants}
        reference = variants[0][6]
        if any(np.max(np.abs(row[6] - reference)) > 1e-3 for row in variants):
            raise RuntimeError("counterfactual states are not fully paired")
        best = max(by_target, key=lambda target: by_target[target])
        matrix.append(reference)
        labels.append(best)
        seeds.append(seed)
        opponents.append(opponent)
        outcomes.append(by_target)
    return (
        np.asarray(matrix, dtype=np.float32), np.asarray(labels),
        np.asarray(seeds, dtype=np.int64), np.asarray(opponents), outcomes,
    )


def _predict_shifted(
    model: DecisionTreeClassifier,
    matrix: np.ndarray,
    train_matrix: np.ndarray,
    mode: int,
    random_signs: np.ndarray | None = None,
) -> np.ndarray:
    tree = model.tree_
    spans = np.zeros(tree.node_count, dtype=np.float64)
    for node in np.flatnonzero(tree.children_left >= 0):
        values = train_matrix[:, tree.feature[node]]
        span = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
        spans[node] = max(1e-6, 0.05 * span)
    result = []
    for vector in matrix:
        node = 0
        while tree.children_left[node] >= 0:
            sign = mode if random_signs is None else int(random_signs[node])
            threshold = float(tree.threshold[node]) + sign * spans[node]
            node = tree.children_left[node] if vector[tree.feature[node]] <= threshold else tree.children_right[node]
        result.append(model.classes_[int(np.argmax(tree.value[node, 0]))])
    return np.asarray(result)


def _scores(predictions: np.ndarray, outcomes: list[dict[str, tuple[float, float]]]) -> np.ndarray:
    return np.asarray([outcome[str(prediction)][0] for prediction, outcome in zip(predictions, outcomes)])


def _payoffs(
    outcomes: list[dict[str, tuple[float, float]]], targets: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([[row[target][0] for target in targets] for row in outcomes]),
        np.asarray([[row[target][1] for target in targets] for row in outcomes]),
    )


def _direct_node_classes(
    model: DecisionTreeRegressor, matrix: np.ndarray,
    wins: np.ndarray, margins: np.ndarray,
    opponents: np.ndarray | None = None,
) -> np.ndarray:
    """Best route at every node; optionally maximize worst-opponent win rate."""
    path = model.decision_path(matrix).tocsc()
    result = np.zeros(model.tree_.node_count, dtype=np.int32)
    for node in range(model.tree_.node_count):
        rows = path.indices[path.indptr[node]:path.indptr[node + 1]]
        if len(rows):
            mean_wins = np.mean(wins[rows], axis=0)
            mean_margins = np.mean(margins[rows], axis=0)
            worst_wins = (
                np.min([
                    np.mean(wins[rows][opponents[rows] == opponent], axis=0)
                    for opponent in np.unique(opponents[rows])
                ], axis=0)
                if opponents is not None else mean_wins
            )
            result[node] = max(
                range(wins.shape[1]),
                key=lambda route: (
                    worst_wins[route], mean_wins[route], mean_margins[route], -route
                ),
            )
    return result


def _predict_direct(
    model: DecisionTreeRegressor, matrix: np.ndarray,
    node_classes: np.ndarray, targets: list[str],
) -> np.ndarray:
    return np.asarray(targets)[node_classes[model.apply(matrix)]]


def _confidence(diff: np.ndarray, seeds: np.ndarray) -> dict[str, Any]:
    per_seed = np.asarray([np.mean(diff[seeds == seed]) for seed in np.unique(seeds)])
    mean = float(np.mean(per_seed))
    if len(per_seed) >= 2:
        se = float(np.std(per_seed, ddof=1) / math.sqrt(len(per_seed)))
        lower = mean - float(student_t.ppf(0.95, len(per_seed) - 1)) * se
    else:
        se, lower = 0.0, mean
    return {
        "mean": mean, "one_sided_95pct_lower": lower, "seed_standard_error": se,
        "per_seed": {str(seed): float(value) for seed, value in zip(np.unique(seeds), per_seed)},
    }


def _cross_validate(
    matrix: np.ndarray,
    labels: np.ndarray,
    seeds: np.ndarray,
    cv_groups: np.ndarray,
    outcomes: list[dict[str, tuple[float, float]]],
    stay: str,
    depth: int,
    leaf: int,
) -> dict[str, Any]:
    folds = min(5, len(np.unique(cv_groups)))
    splitter = GroupKFold(n_splits=folds)
    nominal = np.empty(len(labels), dtype=labels.dtype)
    variants = [np.empty(len(labels), dtype=labels.dtype) for _ in range(10)]
    from fast_kaggriculture import native_threshold_variants

    for fold, (train, valid) in enumerate(splitter.split(matrix, labels, cv_groups)):
        model = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=leaf, random_state=20260824 + fold
        ).fit(matrix[train], labels[train])
        nominal[valid] = model.predict(matrix[valid])
        leaf_class = np.argmax(model.tree_.value[:, 0, :], axis=1).astype(np.int32)
        shifted = np.asarray(native_threshold_variants(
            model.tree_.children_left.astype(np.int32),
            model.tree_.children_right.astype(np.int32),
            model.tree_.feature.astype(np.int32),
            model.tree_.threshold.astype(np.float64),
            leaf_class,
            np.ascontiguousarray(matrix[valid], dtype=np.float32),
            np.ascontiguousarray(matrix[train], dtype=np.float32),
            20260824 + fold * 100,
        ))
        for index in range(len(variants)):
            variants[index][valid] = model.classes_[shifted[index]]
    nominal_scores = _scores(nominal, outcomes)
    stay_scores = _scores(np.full(len(labels), stay), outcomes)
    variant_scores = [_scores(value, outcomes) for value in variants]
    variant_means = [float(np.mean(value)) for value in variant_scores]
    threshold_worst = min(variant_means)
    flip_rates = [float(np.mean(value != nominal)) for value in variants]
    confidence = _confidence(nominal_scores - stay_scores, seeds)
    robust_improvement = min(
        threshold_worst - float(np.mean(stay_scores)),
        confidence["one_sided_95pct_lower"],
    )
    return {
        "depth": depth,
        "min_leaf": leaf,
        "score": float(np.mean(nominal_scores)),
        "stay_score": float(np.mean(stay_scores)),
        "improvement": float(np.mean(nominal_scores - stay_scores)),
        "threshold_worst_score": threshold_worst,
        "threshold_worst_improvement": threshold_worst - float(np.mean(stay_scores)),
        "threshold_max_flip_rate": max(flip_rates),
        "threshold_variant_scores": variant_means,
        "confidence": confidence,
        "robust_improvement": robust_improvement,
    }


def _cross_validate_direct(
    matrix: np.ndarray,
    seeds: np.ndarray,
    cv_groups: np.ndarray,
    outcomes: list[dict[str, tuple[float, float]]],
    targets: list[str],
    stay: str,
    depth: int,
    leaf: int,
    leave_one_group_out: bool = False,
    opponents: np.ndarray | None = None,
    maximin: bool = False,
) -> dict[str, Any]:
    wins, margins = _payoffs(outcomes, targets)
    splitter = (
        LeaveOneGroupOut()
        if leave_one_group_out
        else GroupKFold(n_splits=min(5, len(np.unique(cv_groups))))
    )
    nominal = np.empty(len(matrix), dtype=np.asarray(targets).dtype)
    variants = [np.empty_like(nominal) for _ in range(10)]
    from fast_kaggriculture import native_threshold_variants

    for fold, (train, valid) in enumerate(splitter.split(matrix, groups=cv_groups)):
        model = DecisionTreeRegressor(
            max_depth=depth, min_samples_leaf=leaf, random_state=20260824 + fold
        ).fit(matrix[train], wins[train])
        node_classes = _direct_node_classes(
            model, matrix[train], wins[train], margins[train],
            opponents[train] if maximin else None,
        )
        nominal[valid] = _predict_direct(model, matrix[valid], node_classes, targets)
        shifted = np.asarray(native_threshold_variants(
            model.tree_.children_left.astype(np.int32),
            model.tree_.children_right.astype(np.int32),
            model.tree_.feature.astype(np.int32),
            model.tree_.threshold.astype(np.float64),
            node_classes,
            np.ascontiguousarray(matrix[valid], dtype=np.float32),
            np.ascontiguousarray(matrix[train], dtype=np.float32),
            20260824 + fold * 100,
        ))
        for index in range(len(variants)):
            variants[index][valid] = np.asarray(targets)[shifted[index]]

    route_index = {target: index for index, target in enumerate(targets)}
    chosen = np.asarray([route_index[str(value)] for value in nominal])
    stay_index = targets.index(stay)
    row = np.arange(len(matrix))
    nominal_scores, nominal_margins = wins[row, chosen], margins[row, chosen]
    stay_scores, stay_margins = wins[:, stay_index], margins[:, stay_index]
    variant_scores = [
        wins[row, np.asarray([route_index[str(value)] for value in prediction])]
        for prediction in variants
    ]
    variant_means = [float(np.mean(value)) for value in variant_scores]
    threshold_worst = min(variant_means)
    confidence = _confidence(nominal_scores - stay_scores, seeds)
    robust_improvement = min(
        threshold_worst - float(np.mean(stay_scores)),
        confidence["one_sided_95pct_lower"],
    )
    opponent_min_improvement = (
        min(
            float(np.mean(nominal_scores[opponents == opponent]
                          - stay_scores[opponents == opponent]))
            for opponent in np.unique(opponents)
        )
        if opponents is not None else float(np.mean(nominal_scores - stay_scores))
    )
    if maximin:
        robust_improvement = min(robust_improvement, opponent_min_improvement)
    return {
        "depth": depth,
        "min_leaf": leaf,
        "score": float(np.mean(nominal_scores)),
        "stay_score": float(np.mean(stay_scores)),
        "improvement": float(np.mean(nominal_scores - stay_scores)),
        "margin": float(np.mean(nominal_margins)),
        "stay_margin": float(np.mean(stay_margins)),
        "margin_improvement": float(np.mean(nominal_margins - stay_margins)),
        "threshold_worst_score": threshold_worst,
        "threshold_worst_improvement": threshold_worst - float(np.mean(stay_scores)),
        "threshold_max_flip_rate": max(float(np.mean(value != nominal)) for value in variants),
        "threshold_variant_scores": variant_means,
        "confidence": confidence,
        "robust_improvement": robust_improvement,
        "opponent_min_improvement": opponent_min_improvement,
    }


def _combined_validation(seed_cv: dict[str, Any], opponent_cv: dict[str, Any]) -> dict[str, Any]:
    result = {
        "depth": seed_cv["depth"],
        "min_leaf": seed_cv["min_leaf"],
        "score": min(seed_cv["score"], opponent_cv["score"]),
        "stay_score": seed_cv["stay_score"],
        "improvement": min(seed_cv["improvement"], opponent_cv["improvement"]),
        "threshold_worst_score": min(
            seed_cv["threshold_worst_score"], opponent_cv["threshold_worst_score"]
        ),
        "threshold_worst_improvement": min(
            seed_cv["threshold_worst_improvement"],
            opponent_cv["threshold_worst_improvement"],
        ),
        "threshold_max_flip_rate": max(
            seed_cv["threshold_max_flip_rate"], opponent_cv["threshold_max_flip_rate"]
        ),
        "robust_improvement": min(
            seed_cv["robust_improvement"], opponent_cv["robust_improvement"]
        ),
        "seed_group_cv": seed_cv,
        "opponent_family_group_cv": opponent_cv,
    }
    for key in ("margin", "stay_margin", "margin_improvement"):
        if key in seed_cv:
            result[key] = min(seed_cv[key], opponent_cv[key])
    return result


def _tree_payload(model: DecisionTreeClassifier) -> dict[str, Any]:
    tree = model.tree_
    return {
        "class_kind": "family",
        "classes": [str(value) for value in model.classes_],
        "left": tree.children_left.astype(int).tolist(),
        "right": tree.children_right.astype(int).tolist(),
        "feature": tree.feature.astype(int).tolist(),
        "threshold": tree.threshold.astype(float).tolist(),
        "value": tree.value[:, 0, :].astype(float).tolist(),
    }


def _direct_tree_payload(
    model: DecisionTreeRegressor, node_classes: np.ndarray, targets: list[str]
) -> dict[str, Any]:
    tree = model.tree_
    values = np.zeros((tree.node_count, len(targets)), dtype=np.float64)
    values[np.arange(tree.node_count), node_classes] = 1.0
    return {
        "class_kind": "family",
        "classes": targets,
        "left": tree.children_left.astype(int).tolist(),
        "right": tree.children_right.astype(int).tolist(),
        "feature": tree.feature.astype(int).tolist(),
        "threshold": tree.threshold.astype(float).tolist(),
        "value": values.tolist(),
    }


def _threshold_audit(model: DecisionTreeClassifier, matrix: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    tree = model.tree_
    nominal = model.predict(matrix)
    rows = []
    for node in np.flatnonzero(tree.children_left >= 0):
        feature = int(tree.feature[node])
        values = matrix[:, feature]
        span = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
        delta = max(1e-6, 0.05 * span)
        threshold = float(tree.threshold[node])
        changed = []
        for sign in (-1, 1):
            original = tree.threshold[node]
            tree.threshold[node] = original + sign * delta
            changed.append(float(np.mean(model.predict(matrix) != nominal)))
            tree.threshold[node] = original
        rows.append({
            "node": int(node), "feature_index": feature, "feature": names[feature],
            "threshold": threshold, "perturbation": delta,
            "threshold_interval": [threshold - delta, threshold + delta],
            "samples": int(tree.n_node_samples[node]),
            "fraction_near_threshold": float(np.mean(np.abs(values - threshold) <= delta)),
            "prediction_flip_rate_minus": changed[0],
            "prediction_flip_rate_plus": changed[1],
        })
    return rows


def _threshold_audit_direct(
    model: DecisionTreeRegressor, matrix: np.ndarray, names: list[str],
    node_classes: np.ndarray, targets: list[str],
) -> list[dict[str, Any]]:
    tree = model.tree_
    nominal = _predict_direct(model, matrix, node_classes, targets)
    rows = []
    for node in np.flatnonzero(tree.children_left >= 0):
        feature = int(tree.feature[node])
        values = matrix[:, feature]
        span = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
        delta = max(1e-6, 0.05 * span)
        threshold = float(tree.threshold[node])
        changed = []
        for sign in (-1, 1):
            tree.threshold[node] = threshold + sign * delta
            changed.append(float(np.mean(
                _predict_direct(model, matrix, node_classes, targets) != nominal
            )))
        tree.threshold[node] = threshold
        rows.append({
            "node": int(node), "feature_index": feature, "feature": names[feature],
            "threshold": threshold, "perturbation": delta,
            "threshold_interval": [threshold - delta, threshold + delta],
            "samples": int(tree.n_node_samples[node]),
            "fraction_near_threshold": float(np.mean(np.abs(values - threshold) <= delta)),
            "prediction_flip_rate_minus": changed[0],
            "prediction_flip_rate_plus": changed[1],
        })
    return rows


def _bootstrap_stability(
    matrix: np.ndarray,
    labels: np.ndarray,
    seeds: np.ndarray,
    depth: int,
    leaf: int,
    names: list[str],
    repetitions: int = 64,
    workers: int = 1,
    direct_wins: np.ndarray | None = None,
) -> dict[str, Any]:
    unique_seeds = np.unique(seeds)
    rng = np.random.default_rng(20260824)
    sampled_seeds = [rng.choice(unique_seeds, size=len(unique_seeds), replace=True)
                     for _ in range(repetitions)]

    def fit(iteration: int) -> tuple[set[int], dict[int, list[float]], int | None]:
        sampled = sampled_seeds[iteration]
        indices = np.concatenate([np.flatnonzero(seeds == seed) for seed in sampled])
        model_class = DecisionTreeRegressor if direct_wins is not None else DecisionTreeClassifier
        fit_targets = direct_wins[indices] if direct_wins is not None else labels[indices]
        model = model_class(
            max_depth=depth, min_samples_leaf=leaf,
            random_state=20260824 + iteration,
        ).fit(matrix[indices], fit_targets)
        internal = np.flatnonzero(model.tree_.children_left >= 0)
        used = set(int(model.tree_.feature[node]) for node in internal)
        thresholds = defaultdict(list)
        for node in internal:
            thresholds[int(model.tree_.feature[node])].append(float(model.tree_.threshold[node]))
        return used, thresholds, int(model.tree_.feature[0]) if len(internal) else None

    selected = np.zeros(matrix.shape[1], dtype=np.int32)
    thresholds: dict[int, list[float]] = defaultdict(list)
    root_features: list[int] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fitted = list(pool.map(fit, range(repetitions)))
    for used, fitted_thresholds, root in fitted:
        for feature in used:
            selected[feature] += 1
        for feature, values in fitted_thresholds.items():
            thresholds[feature].extend(values)
        if root is not None:
            root_features.append(root)
    rows = []
    for feature in np.flatnonzero(selected):
        values = np.asarray(thresholds[int(feature)], dtype=np.float64)
        span = float(np.quantile(matrix[:, feature], 0.95) - np.quantile(matrix[:, feature], 0.05))
        rows.append({
            "feature_index": int(feature), "feature": names[int(feature)],
            "selection_frequency": float(selected[feature] / repetitions),
            "threshold_median": float(np.median(values)),
            "threshold_p10": float(np.quantile(values, 0.10)),
            "threshold_p90": float(np.quantile(values, 0.90)),
            "threshold_spread_over_feature_span": float(
                (np.quantile(values, 0.90) - np.quantile(values, 0.10)) / max(span, 1e-9)
            ),
        })
    rows.sort(key=lambda row: (-row["selection_frequency"], row["feature"]))
    root_counts = {
        names[feature]: root_features.count(feature) / max(1, len(root_features))
        for feature in sorted(set(root_features))
    }
    def feature_group(name: str) -> str:
        if name.startswith(("market_", "shop_")):
            return "market_and_shops"
        if name.startswith("opponent_"):
            return "opponent_public_history"
        if name.startswith("self_"):
            return "self_state_history"
        if name.startswith("plan_"):
            return "route_feasibility"
        return "other"
    root_group_counts: dict[str, float] = defaultdict(float)
    for name, frequency in root_counts.items():
        root_group_counts[feature_group(name)] += frequency
    return {
        "repetitions": repetitions,
        "features": rows,
        "root_feature_frequencies": dict(
            sorted(root_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "root_feature_group_frequencies": dict(
            sorted(root_group_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def _train_node(task: tuple) -> dict[str, Any]:
    (
        opening, checkpoint, compact, raw_group, names, depths, leaves,
        min_robust_improvement, simplicity_tolerance, trial_workers, direct_payoff,
        direct_payoff_maximin,
    ) = task
    if compact:
        matrix, labels, seeds, opponents, outcomes = raw_group
    else:
        matrix, labels, seeds, opponents, outcomes = _samples(raw_group)
    targets = list(dict.fromkeys(target for row in outcomes for target in row))
    direct_wins, direct_margins = _payoffs(outcomes, targets)
    def evaluate(parameters: tuple[int, int]) -> dict[str, Any]:
        depth, leaf = parameters
        if direct_payoff:
            return _combined_validation(
                _cross_validate_direct(
                    matrix, seeds, seeds, outcomes, targets, opening, depth, leaf,
                    opponents=opponents, maximin=direct_payoff_maximin,
                ),
                _cross_validate_direct(
                    matrix, seeds, opponents, outcomes, targets, opening, depth, leaf,
                    leave_one_group_out=True,
                    opponents=opponents, maximin=direct_payoff_maximin,
                ),
            )
        return _combined_validation(
            _cross_validate(
                matrix, labels, seeds, seeds, outcomes, opening, depth, leaf
            ),
            _cross_validate(
                matrix, labels, seeds, opponents, outcomes, opening, depth, leaf
            ),
        )
    parameters = [(depth, leaf) for depth in depths for leaf in leaves if leaf * 2 <= len(labels)]
    with ThreadPoolExecutor(max_workers=trial_workers) as pool:
        trials = list(pool.map(evaluate, parameters))
    trials.sort(
        key=lambda row: (
            -row["robust_improvement"], row["depth"], -row["min_leaf"],
            -row["improvement"],
        )
    )
    best_robust = trials[0]["robust_improvement"]
    selection_floor = (
        max(min_robust_improvement, best_robust - simplicity_tolerance)
        if best_robust >= min_robust_improvement
        else best_robust - simplicity_tolerance
    )
    near_best = [
        row for row in trials if row["robust_improvement"] >= selection_floor
    ]
    selected = min(
        near_best,
        key=lambda row: (row["depth"], -row["min_leaf"], -row["robust_improvement"]),
    )
    model_class = DecisionTreeRegressor if direct_payoff else DecisionTreeClassifier
    fit_targets = direct_wins if direct_payoff else labels
    model = model_class(
        max_depth=selected["depth"], min_samples_leaf=selected["min_leaf"],
        random_state=20260824,
    ).fit(matrix, fit_targets)
    stability = _bootstrap_stability(
        matrix, labels, seeds, selected["depth"], selected["min_leaf"], names,
        workers=trial_workers, direct_wins=direct_wins if direct_payoff else None,
    )
    root_frequency = max(
        stability["root_feature_group_frequencies"].values(), default=0.0
    )
    enabled = (
        selected["robust_improvement"] >= min_robust_improvement
        and root_frequency >= 0.35
    )
    node_classes = (
        _direct_node_classes(
            model, matrix, direct_wins, direct_margins,
            opponents if direct_payoff_maximin else None,
        )
        if direct_payoff else None
    )
    return {
        "selected": {
            "opening": opening, "checkpoint": checkpoint,
            "samples": len(labels),
            "seed_count": len(np.unique(seeds)),
            "enabled": enabled,
            "metrics": selected,
            "tree": (
                _direct_tree_payload(model, node_classes, targets)
                if direct_payoff else _tree_payload(model)
            ),
            "rules": export_text(model, feature_names=names, decimals=3),
            "threshold_audit": (
                _threshold_audit_direct(model, matrix, names, node_classes, targets)
                if direct_payoff else _threshold_audit(model, matrix, names)
            ),
            "bootstrap_stability": stability,
        },
        "hyperparameter_trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depths", default="2,3,4,5,6")
    parser.add_argument("--min-leaves", default="16,32,64")
    parser.add_argument("--min-robust-improvement", type=float, default=0.01)
    parser.add_argument(
        "--simplicity-tolerance", type=float, default=0.01,
        help="Prefer the shallowest tree within this robust-score distance of the best.",
    )
    parser.add_argument("--node-workers", type=int, default=1)
    parser.add_argument(
        "--direct-payoff", action="store_true",
        help=(
            "Opt in to regression-tree splits over every route's win outcome; "
            "each leaf selects the route with best empirical mean (win, margin)."
        ),
    )
    parser.add_argument(
        "--direct-payoff-maximin", action="store_true",
        help="Choose each direct-payoff leaf by worst-opponent mean win rate.",
    )
    args = parser.parse_args()
    if args.direct_payoff_maximin:
        args.direct_payoff = True
    if args.node_workers < 1:
        parser.error("--node-workers must be positive")
    depths = [int(value) for value in args.depths.split(",")]
    leaves = [int(value) for value in args.min_leaves.split(",")]
    with np.load(args.search[0]) as probe:
        compact = "outcome" in probe
    grouped, openings, targets = (
        _load_compact_groups(args.search) if compact else _load_groups(args.search)
    )
    names = feature_names()
    node_tasks = []
    for opening in openings:
        for checkpoint in sorted(cp for family, cp in grouped if family == opening):
            node_tasks.append((
                opening, checkpoint, compact, grouped[(opening, checkpoint)], names,
                depths, leaves, args.min_robust_improvement, args.simplicity_tolerance,
            ))
    outer_workers = min(args.node_workers, max(1, len(node_tasks)))
    trial_workers = max(1, args.node_workers // outer_workers)
    node_tasks = [
        (*task, trial_workers, args.direct_payoff, args.direct_payoff_maximin)
        for task in node_tasks
    ]
    with ThreadPoolExecutor(max_workers=outer_workers) as pool:
        nodes = list(pool.map(_train_node, node_tasks))
    payload = {
        "schema_version": 3,
        "kind": "robust_counterfactual_search_route_trees",
        "feature_schema": "semantic_route_switch_v1",
        "compact_native_search": compact,
        "sources": [str(path) for path in args.search],
        "feature_names": names,
        "openings": openings,
        "targets": targets,
        "nodes": nodes,
        "robustness": {
            "seed_grouped_cross_validation": True,
            "opponent_family_grouped_cross_validation": True,
            "threshold_perturbation": "all thresholds shifted by ±5% of train p05-p95 span plus 8 deterministic sign patterns",
            "confidence": "one-sided 95% Student-t lower bound over paired per-seed improvements",
            "min_robust_improvement": args.min_robust_improvement,
            "simplicity_tolerance": args.simplicity_tolerance,
            "parallel_node_workers": args.node_workers,
            "parallel_outer_nodes": outer_workers,
            "parallel_per_node_trials": trial_workers,
        },
    }
    if args.direct_payoff:
        payload["training_objective"] = "direct_leaf_mean_win_then_margin"
    if args.direct_payoff_maximin:
        payload["training_objective"] = "direct_leaf_worst_opponent_win_then_mean"
        payload["robustness"]["leave_one_opponent_out"] = True
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "selected": [
            {
                "opening": row["selected"]["opening"],
                "checkpoint": row["selected"]["checkpoint"],
                "enabled": row["selected"]["enabled"],
                **{key: row["selected"]["metrics"][key] for key in (
                    "depth", "min_leaf", "score", "stay_score", "improvement",
                    "threshold_worst_improvement", "threshold_max_flip_rate",
                    "robust_improvement",
                )},
                **({
                    key: row["selected"]["metrics"][key]
                    for key in ("margin", "stay_margin", "margin_improvement")
                } if args.direct_payoff else {}),
            }
            for row in nodes
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
