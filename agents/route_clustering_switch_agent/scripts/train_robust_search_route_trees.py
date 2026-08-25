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
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

from meta_agent.src.route_switch_features import route_switch_feature_names


def feature_names() -> list[str]:
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


def _combined_validation(seed_cv: dict[str, Any], opponent_cv: dict[str, Any]) -> dict[str, Any]:
    return {
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


def _bootstrap_stability(
    matrix: np.ndarray,
    labels: np.ndarray,
    seeds: np.ndarray,
    depth: int,
    leaf: int,
    names: list[str],
    repetitions: int = 64,
) -> dict[str, Any]:
    unique_seeds = np.unique(seeds)
    rng = np.random.default_rng(20260824)
    selected = np.zeros(matrix.shape[1], dtype=np.int32)
    thresholds: dict[int, list[float]] = defaultdict(list)
    root_features: list[int] = []
    for iteration in range(repetitions):
        sampled = rng.choice(unique_seeds, size=len(unique_seeds), replace=True)
        indices = np.concatenate([np.flatnonzero(seeds == seed) for seed in sampled])
        model = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=leaf,
            random_state=20260824 + iteration,
        ).fit(matrix[indices], labels[indices])
        internal = np.flatnonzero(model.tree_.children_left >= 0)
        used = set(int(model.tree_.feature[node]) for node in internal)
        for feature in used:
            selected[feature] += 1
        for node in internal:
            feature = int(model.tree_.feature[node])
            thresholds[feature].append(float(model.tree_.threshold[node]))
        if len(internal):
            root_features.append(int(model.tree_.feature[0]))
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
        min_robust_improvement, simplicity_tolerance,
    ) = task
    if compact:
        matrix, labels, seeds, opponents, outcomes = raw_group
    else:
        matrix, labels, seeds, opponents, outcomes = _samples(raw_group)
    trials = [
        _combined_validation(
            _cross_validate(
                matrix, labels, seeds, seeds, outcomes, opening, depth, leaf
            ),
            _cross_validate(
                matrix, labels, seeds, opponents, outcomes, opening, depth, leaf
            ),
        )
        for depth in depths for leaf in leaves if leaf * 2 <= len(labels)
    ]
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
    model = DecisionTreeClassifier(
        max_depth=selected["depth"], min_samples_leaf=selected["min_leaf"],
        random_state=20260824,
    ).fit(matrix, labels)
    stability = _bootstrap_stability(
        matrix, labels, seeds, selected["depth"], selected["min_leaf"], names
    )
    root_frequency = max(
        stability["root_feature_group_frequencies"].values(), default=0.0
    )
    enabled = (
        selected["robust_improvement"] >= min_robust_improvement
        and root_frequency >= 0.35
    )
    return {
        "selected": {
            "opening": opening, "checkpoint": checkpoint,
            "samples": len(labels),
            "seed_count": len(np.unique(seeds)),
            "enabled": enabled,
            "metrics": selected,
            "tree": _tree_payload(model),
            "rules": export_text(model, feature_names=names, decimals=3),
            "threshold_audit": _threshold_audit(model, matrix, names),
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
    args = parser.parse_args()
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
    with ThreadPoolExecutor(max_workers=args.node_workers) as pool:
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
        },
    }
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
            }
            for row in nodes
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
