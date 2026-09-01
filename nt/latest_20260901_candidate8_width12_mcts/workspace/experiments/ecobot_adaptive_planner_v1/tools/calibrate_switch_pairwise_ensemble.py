#!/usr/bin/env python3
"""Calibrate a convex ensemble of frozen pairwise project-value models.

Only the calibration corpus is used to choose weights.  The optional test
corpus is evaluated once with the frozen weights and is never part of the
search objective.
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from train_switch_pairwise_policy import (
    model_features,
    pair_accuracy,
    policy_metrics,
    predict_pairs,
    sha256,
)
from train_switch_value_regressor import load


def frozen_predictions(
    data: dict[str, np.ndarray],
    model_paths: list[Path],
    feature_modes: list[str],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], list[np.ndarray]]:
    structure = None
    probabilities: list[np.ndarray] = []
    for model_path, feature_mode in zip(model_paths, feature_modes, strict=True):
        features, _, _ = model_features(
            data["features"], data["feature_names"], feature_mode
        )
        model = lgb.Booster(model_file=str(model_path))
        left, right, pair_key, difference, probability = predict_pairs(
            model, data, features
        )
        current = (left, right, pair_key, difference)
        if structure is None:
            structure = current
        else:
            for expected, observed in zip(structure, current, strict=True):
                if not np.array_equal(expected, observed):
                    raise ValueError("model pair structures differ")
        probabilities.append(probability)
    if structure is None:
        raise ValueError("at least one model is required")
    return structure, probabilities


def blend_prediction(
    structure: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    probabilities: list[np.ndarray],
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probability = np.zeros_like(probabilities[0], dtype=np.float64)
    for weight, component in zip(weights, probabilities, strict=True):
        probability += float(weight) * component
    return (*structure, probability)


def metrics(
    data: dict[str, np.ndarray],
    prediction: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    difference = prediction[3]
    probability = prediction[4]
    return {
        "pairwise": [
            pair_accuracy(difference, probability, gap)
            for gap in (0.0, 500.0, 1000.0, 2000.0)
        ],
        "raw_screening_policy": policy_metrics(data, prediction, 0.0, 0.0),
    }


def selection_key(result: dict[str, object]) -> tuple[float, ...]:
    pairwise = result["pairwise"]
    policy = result["raw_screening_policy"]
    top4 = policy["topk_oracle_recall_with_keep"]["4"]
    # W3 asks for both pairwise accuracy and oracle screening recall.  First
    # prefer candidates satisfying the calibration Top-4 recall gate; within
    # that feasible set maximize literal all-pair accuracy.  Remaining terms
    # are deterministic tie-breakers only.
    return (
        float(top4 >= 0.90),
        float(pairwise[0]["accuracy"]),
        float(pairwise[2]["accuracy"]),
        float(top4),
        float(policy["mean_realized_delta"]),
    )


def simplex_weights(count: int, denominator: int):
    for cuts in itertools.combinations_with_replacement(range(denominator + 1), count - 1):
        points = (0, *cuts, denominator)
        amounts = np.diff(points)
        yield amounts.astype(np.float64) / float(denominator)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument(
        "--feature-mode",
        choices=("legacy", "shop_bits", "shop_bits_scale"),
        action="append",
        required=True,
    )
    parser.add_argument("--grid-denominator", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.model) != len(args.feature_mode):
        raise ValueError("--model and --feature-mode counts must match")
    if args.grid_denominator <= 0:
        raise ValueError("--grid-denominator must be positive")

    calibration = load(args.calibration)
    structure, component_predictions = frozen_predictions(
        calibration, args.model, args.feature_mode
    )
    best = None
    searched = 0
    for weights in simplex_weights(len(args.model), args.grid_denominator):
        prediction = blend_prediction(structure, component_predictions, weights)
        result = metrics(calibration, prediction)
        searched += 1
        candidate = (selection_key(result), weights.copy(), result)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    _, best_weights, calibration_metrics = best

    payload: dict[str, object] = {
        "schema": "kaggriculture.switch-pairwise-convex-ensemble.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "invariants": {
            "component_models_frozen": True,
            "weights_selected_on_calibration_only": True,
            "test_not_used_for_weight_selection": True,
            "keep_is_ordinary_candidate": True,
        },
        "calibration": {
            "path": str(args.calibration),
            "sha256": sha256(args.calibration),
            "rows": int(len(calibration["expected_delta"])),
        },
        "components": [
            {
                "path": str(path),
                "sha256": sha256(path),
                "feature_mode": mode,
                "weight": float(weight),
            }
            for path, mode, weight in zip(
                args.model, args.feature_mode, best_weights, strict=True
            )
        ],
        "grid": {
            "denominator": args.grid_denominator,
            "combinations_searched": searched,
        },
        "calibration_metrics": calibration_metrics,
    }

    if args.test is not None:
        test = load(args.test)
        test_structure, test_components = frozen_predictions(
            test, args.model, args.feature_mode
        )
        test_prediction = blend_prediction(
            test_structure, test_components, best_weights
        )
        payload["test"] = {
            "path": str(args.test),
            "sha256": sha256(args.test),
            "rows": int(len(test["expected_delta"])),
            "metrics": metrics(test, test_prediction),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
