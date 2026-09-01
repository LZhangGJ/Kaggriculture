#!/usr/bin/env python3
"""Audit whether public-state features can predict profitable plan switches."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-games", type=int, required=True)
    args = parser.parse_args()

    data = np.load(args.dataset)
    features = np.asarray(data["features"], dtype=np.float64)
    actual_delta = np.asarray(data["actual_delta"], dtype=np.float64)
    seed = np.asarray(data["seed"], dtype=np.int64)
    names = [str(value) for value in data["feature_names"]]
    positive = (actual_delta > 0).astype(np.int8)

    classifier_oof = np.zeros(len(positive), dtype=np.float64)
    regressor_oof = np.zeros(len(positive), dtype=np.float64)
    classifier_importance = np.zeros(features.shape[1], dtype=np.float64)
    regressor_importance = np.zeros(features.shape[1], dtype=np.float64)
    for fold in range(5):
        validation = seed % 5 == fold
        training = ~validation
        classifier = lgb.LGBMClassifier(
            n_estimators=180,
            num_leaves=12,
            max_depth=5,
            min_child_samples=25,
            learning_rate=0.03,
            reg_lambda=20.0,
            reg_alpha=5.0,
            verbosity=-1,
            n_jobs=16,
            random_state=100 + fold,
        )
        classifier.fit(features[training], positive[training])
        classifier_oof[validation] = classifier.predict_proba(
            features[validation]
        )[:, 1]
        classifier_importance += classifier.feature_importances_

        regressor = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=220,
            num_leaves=12,
            max_depth=5,
            min_child_samples=25,
            learning_rate=0.025,
            reg_lambda=30.0,
            reg_alpha=10.0,
            verbosity=-1,
            n_jobs=16,
            random_state=200 + fold,
        )
        regressor.fit(features[training], actual_delta[training])
        regressor_oof[validation] = regressor.predict(features[validation])
        regressor_importance += regressor.feature_importances_

    def threshold_rows(scores: np.ndarray, thresholds: list[float]):
        rows = []
        for threshold in thresholds:
            selected = scores >= threshold
            selected_delta = actual_delta[selected]
            rows.append({
                "threshold": threshold,
                "selected": int(selected.sum()),
                "coverage_of_switch_candidates": float(selected.mean()),
                "precision": (
                    float(np.mean(selected_delta > 0))
                    if selected_delta.size else 0.0
                ),
                "mean_delta_selected": (
                    float(selected_delta.mean()) if selected_delta.size else 0.0
                ),
                "gain_per_all_game": float(selected_delta.sum() / args.total_games),
                "p10_delta_selected": (
                    float(np.quantile(selected_delta, 0.10))
                    if selected_delta.size else 0.0
                ),
            })
        return rows

    def importance_rows(values: np.ndarray):
        order = np.argsort(-values)[:24]
        return [
            {"feature": names[index], "importance": float(values[index])}
            for index in order
        ]

    payload = {
        "schema": "kaggriculture.switch-counterfactual-model-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "examples": int(len(positive)),
        "total_games": args.total_games,
        "raw_switch_precision": float(positive.mean()),
        "raw_switch_gain_per_all_game": float(actual_delta.sum() / args.total_games),
        "classifier": {
            "oof_auc": float(roc_auc_score(positive, classifier_oof)),
            "thresholds": threshold_rows(
                classifier_oof,
                [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
            ),
            "feature_importance": importance_rows(classifier_importance),
        },
        "regressor": {
            "oof_sign_accuracy": float(np.mean((regressor_oof > 0) == positive)),
            "thresholds": threshold_rows(
                regressor_oof,
                [-2000.0, -1000.0, 0.0, 1000.0, 2000.0, 4000.0, 6000.0],
            ),
            "feature_importance": importance_rows(regressor_importance),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "examples": payload["examples"],
        "raw_switch_precision": payload["raw_switch_precision"],
        "raw_switch_gain_per_all_game": payload["raw_switch_gain_per_all_game"],
        "classifier_oof_auc": payload["classifier"]["oof_auc"],
        "regressor_oof_sign_accuracy": payload["regressor"]["oof_sign_accuracy"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
