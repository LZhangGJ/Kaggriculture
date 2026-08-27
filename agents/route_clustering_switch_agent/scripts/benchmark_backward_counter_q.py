#!/usr/bin/env python3
"""Benchmark fitted route-Q ensembles before committing them to the runtime."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from train_backward_counter_selector import (
    CHECKPOINTS,
    _load_features,
    _load_matrices,
    _selected_values,
)


def _utility(margins: np.ndarray, margin_weight: float) -> np.ndarray:
    return (margins > 0).astype(np.float64) + margin_weight * np.tanh(
        margins / 10000.0
    )


def _model(
    kind: str, estimators: int, leaf: int, max_features: float, random_state: int,
):
    cls = ExtraTreesRegressor if kind == "extra" else RandomForestRegressor
    return cls(
        n_estimators=estimators,
        min_samples_leaf=leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=random_state,
    )


def _fit_backward(
    kind: str, estimators: int, leaf: int, max_features: float,
    margin_weight: float, features: list[np.ndarray], margins: list[np.ndarray],
    targets: np.ndarray, opening: str,
) -> list:
    opening_index = int(np.flatnonzero(targets == opening)[0])
    models: list[object | None] = [None, None, None]
    continuation_margin: np.ndarray | None = None
    for node_index in (2, 1, 0):
        utility = _utility(
            margins[node_index].reshape(len(targets), -1), margin_weight
        )
        if continuation_margin is not None:
            utility[opening_index] = _utility(
                continuation_margin, margin_weight
            )
        model = _model(
            kind, estimators, leaf, max_features, 20260827 + node_index
        )
        model.fit(features[node_index], utility.T)
        predictions = targets[np.argmax(model.predict(features[node_index]), axis=1)]
        selected = _selected_values(margins[node_index], predictions, targets)
        if continuation_margin is not None:
            selected = np.where(
                predictions == opening, continuation_margin, selected
            )
        continuation_margin = selected
        models[node_index] = model
    return [model for model in models if model is not None]


def _evaluate(
    models: list, features: list[np.ndarray], margins: list[np.ndarray],
    targets: np.ndarray, opening: str,
) -> tuple[dict[str, float], dict[str, dict[str, int]]]:
    scenarios = features[0].shape[0]
    active = np.ones(scenarios, dtype=bool)
    result = np.zeros(scenarios, dtype=np.float64)
    counts = {}
    for node_index, checkpoint in enumerate(CHECKPOINTS):
        predictions = targets[np.argmax(
            models[node_index].predict(features[node_index]), axis=1
        )]
        counts[str(checkpoint)] = dict(Counter(predictions.tolist()))
        selected = _selected_values(margins[node_index], predictions, targets)
        switch = active & (predictions != opening)
        result[switch] = selected[switch]
        active[switch] = False
        if node_index == 2:
            result[active] = selected[active]
    paired = result.reshape(-1, 2)
    return {
        "raw_win_rate": float(np.mean(result > 0)),
        "mean_margin": float(np.mean(result)),
        "minimum_margin": float(np.min(result)),
        "both_seats_win_rate": float(np.mean(np.all(paired > 0, axis=1))),
        "p10_paired_margin": float(np.quantile(np.mean(paired, axis=1), 0.10)),
    }, counts


def _csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",")]


def _csv_floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-matrices", nargs=3, type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--valid-matrices", nargs=3, type=Path, required=True)
    parser.add_argument("--valid-features", type=Path, required=True)
    parser.add_argument("--opening", default="G001")
    parser.add_argument("--kinds", default="extra")
    parser.add_argument("--estimators", type=int, default=128)
    parser.add_argument("--min-leaves", default="1,2,4,8,16")
    parser.add_argument("--max-features", default="0.5,1.0")
    parser.add_argument("--margin-weights", default="0,0.05")
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    args = parser.parse_args()

    targets, train_margins, train_seeds = _load_matrices(args.train_matrices)
    valid_targets, valid_margins, valid_seeds = _load_matrices(args.valid_matrices)
    if not np.array_equal(targets, valid_targets):
        raise ValueError("training and validation targets differ")
    train_features, _ = _load_features(args.train_features, train_seeds)
    valid_features, _ = _load_features(args.valid_features, valid_seeds)

    fitted = {}
    trials = []
    for kind in args.kinds.split(","):
        for leaf in _csv_ints(args.min_leaves):
            for max_features in _csv_floats(args.max_features):
                for margin_weight in _csv_floats(args.margin_weights):
                    key = (kind, leaf, max_features, margin_weight)
                    models = _fit_backward(
                        kind, args.estimators, leaf, max_features, margin_weight,
                        train_features, train_margins, targets, args.opening,
                    )
                    train_metrics, _ = _evaluate(
                        models, train_features, train_margins, targets, args.opening
                    )
                    valid_metrics, _ = _evaluate(
                        models, valid_features, valid_margins, targets, args.opening
                    )
                    fitted[key] = models
                    row = {
                        "kind": kind,
                        "estimators": args.estimators,
                        "min_leaf": leaf,
                        "max_features": max_features,
                        "margin_weight": margin_weight,
                        "train": train_metrics,
                        "valid": valid_metrics,
                    }
                    trials.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
    trials.sort(key=lambda row: (
        -row["valid"]["raw_win_rate"],
        -row["valid"]["both_seats_win_rate"],
        -row["valid"]["mean_margin"],
        row["min_leaf"], -row["max_features"], row["margin_weight"],
    ))
    selected = trials[0]
    key = (
        selected["kind"], selected["min_leaf"],
        selected["max_features"], selected["margin_weight"],
    )
    models = fitted[key]
    train_metrics, train_counts = _evaluate(
        models, train_features, train_margins, targets, args.opening
    )
    valid_metrics, valid_counts = _evaluate(
        models, valid_features, valid_margins, targets, args.opening
    )
    bundle = {
        "schema": "backward-counter-q-model-v1",
        "opening": args.opening,
        "targets": targets.tolist(),
        "checkpoints": CHECKPOINTS,
        "selected": selected,
        "models": models,
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_model, compress=3)
    report = {
        "schema": "backward-counter-q-benchmark-v1",
        "targets": targets.tolist(),
        "train_seed_count": len(train_seeds),
        "valid_seed_count": len(valid_seeds),
        "selected": selected,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "train_prediction_counts": train_counts,
        "valid_prediction_counts": valid_counts,
        "trials": trials,
        "model_path": str(args.output_model.resolve()),
        "model_bytes": args.output_model.stat().st_size,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "report": str(args.output_report.resolve()),
        "model": str(args.output_model.resolve()),
        "model_bytes": report["model_bytes"],
        "selected": selected,
        "valid_prediction_counts": valid_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
