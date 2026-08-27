#!/usr/bin/env python3
"""Train a backward, at-most-one-switch selector from counterfactual matrices."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text


CHECKPOINTS = (144, 168, 216)


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


def _load_matrices(paths: list[Path]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    targets: np.ndarray | None = None
    seeds: np.ndarray | None = None
    margins = []
    for path in paths:
        saved = np.load(path, allow_pickle=False)
        current_targets = saved["targets"].astype(str)
        current_seeds = saved["seeds"].astype(np.int64)
        if targets is None:
            targets = current_targets
            seeds = current_seeds
        elif not np.array_equal(targets, current_targets):
            raise ValueError(f"target mismatch: {path}")
        elif not np.array_equal(seeds, current_seeds):
            raise ValueError(f"seed mismatch: {path}")
        # The experiments use exactly one opening.
        margins.append(saved["margins"][0].astype(np.float64))
    assert targets is not None and seeds is not None
    return targets, margins, seeds


def _load_features(path: Path, seeds: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    saved = np.load(path, allow_pickle=False)
    if not np.array_equal(saved["seeds"], seeds):
        raise ValueError(f"feature seed mismatch: {path}")
    names = saved["feature_names"].astype(str).tolist()
    matrices = [
        saved[f"features_{checkpoint}"].reshape(len(seeds) * 2, -1).astype(np.float32)
        for checkpoint in CHECKPOINTS
    ]
    return matrices, names


def _selected_values(values: np.ndarray, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    lookup = {str(target): index for index, target in enumerate(targets)}
    indices = np.asarray([lookup[str(value)] for value in predictions], dtype=np.int64)
    flat = values.reshape(len(targets), -1)
    return flat[indices, np.arange(flat.shape[1])]


def _weights(utilities: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    ordered = np.sort(utilities, axis=0)
    gap = ordered[-1] - ordered[-2]
    return 1.0 + np.clip(gap / 5000.0, 0.0, 10.0)


def _fit_backward(
    features: list[np.ndarray], margins: list[np.ndarray], targets: np.ndarray,
    opening: str, depth: int, leaf: int, weight_mode: str,
) -> list[DecisionTreeClassifier]:
    opening_index = int(np.flatnonzero(targets == opening)[0])
    models: list[DecisionTreeClassifier | None] = [None, None, None]
    continuation: np.ndarray | None = None
    for node_index in (2, 1, 0):
        utilities = margins[node_index].reshape(len(targets), -1).copy()
        if continuation is not None:
            utilities[opening_index] = continuation
        labels = targets[np.argmax(utilities, axis=0)]
        model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=20260827 + node_index,
        ).fit(
            features[node_index], labels,
            sample_weight=_weights(utilities, weight_mode),
        )
        predictions = model.predict(features[node_index]).astype(str)
        selected = _selected_values(margins[node_index], predictions, targets)
        if continuation is not None:
            selected = np.where(predictions == opening, continuation, selected)
        continuation = selected
        models[node_index] = model
    return [model for model in models if model is not None]


def _evaluate(
    models: list[DecisionTreeClassifier], features: list[np.ndarray],
    margins: list[np.ndarray], targets: np.ndarray, opening: str,
) -> tuple[dict[str, float], np.ndarray, dict[int, Counter[str]]]:
    scenarios = features[0].shape[0]
    active = np.ones(scenarios, dtype=bool)
    result = np.zeros(scenarios, dtype=np.float64)
    prediction_counts: dict[int, Counter[str]] = {}
    for node_index, checkpoint in enumerate(CHECKPOINTS):
        predictions = models[node_index].predict(features[node_index]).astype(str)
        prediction_counts[checkpoint] = Counter(predictions.tolist())
        values = _selected_values(margins[node_index], predictions, targets)
        switch = active & (predictions != opening)
        result[switch] = values[switch]
        active[switch] = False
        if node_index == len(CHECKPOINTS) - 1:
            result[active] = values[active]
    paired = result.reshape(-1, 2)
    metrics = {
        "raw_win_rate": float(np.mean(result > 0)),
        "score_rate": float(np.mean(result > 0) + 0.5 * np.mean(result == 0)),
        "mean_margin": float(np.mean(result)),
        "minimum_margin": float(np.min(result)),
        "both_seats_win_rate": float(np.mean(np.all(paired > 0, axis=1))),
        "p10_paired_margin": float(np.quantile(np.mean(paired, axis=1), 0.10)),
    }
    return metrics, result, prediction_counts


def _oracle(margins: list[np.ndarray]) -> dict[str, float]:
    combined = np.concatenate(margins, axis=0)
    best = np.max(combined, axis=0)
    return {
        "raw_win_rate": float(np.mean(best > 0)),
        "mean_margin": float(np.mean(best)),
        "minimum_margin": float(np.min(best)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-matrices", nargs=3, type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--valid-matrices", nargs=3, type=Path, required=True)
    parser.add_argument("--valid-features", type=Path, required=True)
    parser.add_argument("--opening", default="G001")
    parser.add_argument("--depths", default="3,4,5,6,8,10,12")
    parser.add_argument("--min-leaves", default="4,8,16,32,64")
    parser.add_argument("--output-policy", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    targets, train_margins, train_seeds = _load_matrices(args.train_matrices)
    valid_targets, valid_margins, valid_seeds = _load_matrices(args.valid_matrices)
    if not np.array_equal(targets, valid_targets):
        raise ValueError("training and validation targets differ")
    train_features, feature_names = _load_features(args.train_features, train_seeds)
    valid_features, valid_names = _load_features(args.valid_features, valid_seeds)
    if feature_names != valid_names:
        raise ValueError("training and validation feature schemas differ")

    trials = []
    fitted: dict[tuple[int, int, str], list[DecisionTreeClassifier]] = {}
    for depth in (int(value) for value in args.depths.split(",")):
        for leaf in (int(value) for value in args.min_leaves.split(",")):
            if leaf * 2 > len(train_seeds) * 2:
                continue
            for weight_mode in ("none", "regret"):
                models = _fit_backward(
                    train_features, train_margins, targets,
                    args.opening, depth, leaf, weight_mode,
                )
                train_metrics, _, _ = _evaluate(
                    models, train_features, train_margins, targets, args.opening
                )
                valid_metrics, _, _ = _evaluate(
                    models, valid_features, valid_margins, targets, args.opening
                )
                key = (depth, leaf, weight_mode)
                fitted[key] = models
                trials.append({
                    "depth": depth,
                    "min_leaf": leaf,
                    "weight_mode": weight_mode,
                    "node_count": int(sum(model.tree_.node_count for model in models)),
                    "train": train_metrics,
                    "valid": valid_metrics,
                })
    trials.sort(key=lambda row: (
        -row["valid"]["raw_win_rate"],
        -row["valid"]["both_seats_win_rate"],
        -row["valid"]["mean_margin"],
        row["node_count"], row["depth"], -row["min_leaf"], row["weight_mode"],
    ))
    selected = trials[0]
    models = fitted[(selected["depth"], selected["min_leaf"], selected["weight_mode"])]
    train_metrics, _, train_counts = _evaluate(
        models, train_features, train_margins, targets, args.opening
    )
    valid_metrics, _, valid_counts = _evaluate(
        models, valid_features, valid_margins, targets, args.opening
    )
    nodes = []
    for checkpoint, model in zip(CHECKPOINTS, models):
        nodes.append({
            "selected": {
                "opening": args.opening,
                "checkpoint": checkpoint,
                "enabled": True,
                "samples": len(train_seeds) * 2,
                "tree": _tree_payload(model),
                "rules": export_text(model, feature_names=feature_names, decimals=3),
            }
        })
    policy = {
        "schema_version": 4,
        "kind": "backward_counterfactual_route_selector",
        "feature_schema": "semantic_route_switch_v1",
        "feature_names": feature_names,
        "openings": [args.opening],
        "targets": targets.tolist(),
        "nodes": nodes,
        "training": {
            "method": "backward fitted route choice with at-most-one-switch semantics",
            "selected_hyperparameters": {
                key: selected[key] for key in (
                    "depth", "min_leaf", "weight_mode", "node_count"
                )
            },
            "train_seeds": train_seeds.tolist(),
            "validation_seeds": valid_seeds.tolist(),
            "train_metrics": train_metrics,
            "validation_metrics": valid_metrics,
        },
    }
    report = {
        "schema": "backward-counter-selector-training-v1",
        "targets": targets.tolist(),
        "train_seed_count": len(train_seeds),
        "valid_seed_count": len(valid_seeds),
        "train_oracle": _oracle(train_margins),
        "valid_oracle": _oracle(valid_margins),
        "selected": selected,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "train_prediction_counts": {
            str(cp): dict(counts) for cp, counts in train_counts.items()
        },
        "valid_prediction_counts": {
            str(cp): dict(counts) for cp, counts in valid_counts.items()
        },
        "trials": trials,
    }
    args.output_policy.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_policy.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "policy": str(args.output_policy.resolve()),
        "report": str(args.output_report.resolve()),
        "train_oracle": report["train_oracle"],
        "valid_oracle": report["valid_oracle"],
        "selected": selected,
        "valid_prediction_counts": report["valid_prediction_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
