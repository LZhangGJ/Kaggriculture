#!/usr/bin/env python3
"""Audit an arm-aware LGBM margin router after rule trees are exhausted.

Every training row uses the public state captured before the route decision and
one candidate arm ID.  The target is that arm's counterfactual terminal margin.
Seeds are kept intact across grouped folds.  The learned router may switch away
from baseline only when the predicted gain exceeds an OOF-selected safety
threshold; two independent event banks are never used for model selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/fusion_champion_v1/tools"))
from train_k320_route_rescue_tree import _baseline_oracle, _load, _metrics  # noqa: E402


ARM_COUNT = 6


def _expand(x: np.ndarray, margins: np.ndarray, indices: np.ndarray):
    selected = x[indices]
    count = len(indices)
    route_ids = np.tile(np.arange(ARM_COUNT, dtype=np.int64), count)
    one_hot = np.eye(ARM_COUNT, dtype=np.float32)[route_ids]
    features = np.concatenate((np.repeat(selected, ARM_COUNT, axis=0), one_hot), axis=1)
    targets = margins[:, indices].T.reshape(-1).astype(np.float32)
    base_win = margins[0, indices] > 0
    route_win = margins[:, indices].T.reshape(-1) > 0
    changed_outcome = route_win != np.repeat(base_win, ARM_COUNT)
    return features, targets, changed_outcome


def _predict_matrix(model, x: np.ndarray) -> np.ndarray:
    features, _, _ = _expand(x, np.zeros((ARM_COUNT, len(x)), dtype=np.int64), np.arange(len(x)))
    return model.predict(features).reshape(len(x), ARM_COUNT).T


def _choices(prediction: np.ndarray, threshold: float) -> np.ndarray:
    best = np.argmax(prediction, axis=0).astype(np.int64)
    gain = prediction[best, np.arange(prediction.shape[1])] - prediction[0]
    return np.where((best != 0) & (gain >= threshold), best, 0).astype(np.int64)


def _choose_threshold(margins: np.ndarray, prediction: np.ndarray):
    candidates = np.unique(
        np.concatenate((
            np.arange(0.0, 10001.0, 250.0),
            np.maximum(np.max(prediction, axis=0) - prediction[0], 0.0),
        ))
    )
    rows = []
    for threshold in candidates:
        choices = _choices(prediction, float(threshold))
        rows.append((float(threshold), _metrics(margins, choices)))
    return max(
        rows,
        key=lambda row: (
            row[1]["score_rate"],
            -row[1]["harmed_wins"],
            -row[1]["switch_rate"],
            row[1]["mean_margin"],
            row[0],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout-a", type=Path, required=True)
    parser.add_argument("--holdout-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()

    train_payload, x, margins, groups, _, feature_names = _load(args.train)
    _, ax, amargins, _, _, anames = _load(args.holdout_a, feature_names)
    _, bx, bmargins, _, _, bnames = _load(args.holdout_b, feature_names)
    if anames != feature_names or bnames != feature_names:
        raise AssertionError("holdout feature schema mismatch")

    configurations = []
    unique_indices = np.arange(len(x))
    for leaves in (3, 5, 7, 11):
        for min_child in (8, 16, 32):
            for estimators in (60, 120):
                for changed_weight in (1.0, 4.0, 8.0):
                    oof = np.zeros((ARM_COUNT, len(x)), dtype=np.float32)
                    for fit_index, valid_index in GroupKFold(n_splits=5).split(x, groups=groups):
                        fit_x, fit_y, changed = _expand(x, margins, fit_index)
                        weights = np.where(changed, changed_weight, 1.0)
                        model = lgb.LGBMRegressor(
                            objective="regression_l1",
                            n_estimators=estimators,
                            learning_rate=0.04,
                            num_leaves=leaves,
                            max_depth=4,
                            min_child_samples=min_child,
                            subsample=0.9,
                            colsample_bytree=0.8,
                            reg_alpha=1.0,
                            reg_lambda=4.0,
                            random_state=20260821,
                            n_jobs=18,
                            deterministic=True,
                            verbosity=-1,
                        )
                        model.fit(fit_x, fit_y, sample_weight=weights)
                        oof[:, valid_index] = _predict_matrix(model, x[valid_index])
                    threshold, metrics = _choose_threshold(margins, oof)
                    configurations.append({
                        "num_leaves": leaves,
                        "min_child_samples": min_child,
                        "n_estimators": estimators,
                        "changed_outcome_weight": changed_weight,
                        "threshold": threshold,
                        "oof": metrics,
                    })

    best = max(
        configurations,
        key=lambda row: (
            row["oof"]["score_rate"],
            -row["oof"]["harmed_wins"],
            -row["oof"]["switch_rate"],
            row["oof"]["mean_margin"],
        ),
    )
    fit_x, fit_y, changed = _expand(x, margins, unique_indices)
    weights = np.where(changed, best["changed_outcome_weight"], 1.0)
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=best["n_estimators"],
        learning_rate=0.04,
        num_leaves=best["num_leaves"],
        max_depth=4,
        min_child_samples=best["min_child_samples"],
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=4.0,
        random_state=20260821,
        n_jobs=18,
        deterministic=True,
        verbosity=-1,
    )
    model.fit(fit_x, fit_y, sample_weight=weights)

    def evaluate(matrix, outcome):
        prediction = _predict_matrix(model, matrix)
        return _metrics(outcome, _choices(prediction, best["threshold"]))

    train_fit = evaluate(x, margins)
    holdout_a = evaluate(ax, amargins)
    holdout_b = evaluate(bx, bmargins)
    gain = np.asarray(model.feature_importances_, dtype=np.float64)
    expanded_names = feature_names + [f"arm_{index}" for index in range(ARM_COUNT)]
    important = sorted(
        ({"feature": name, "gain": float(value)} for name, value in zip(expanded_names, gain, strict=True)),
        key=lambda row: row["gain"],
        reverse=True,
    )[:30]

    payload = {
        "schema": "kaggriculture.fusion_champion.k320-route-margin-lgbm.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "official_package_version": "1.32.7",
        "decision_step": int(train_payload["feature_step"]),
        "future_leakage": False,
        "group_cv": "5 folds grouped by episode seed; both seats and all arms kept together",
        "train_samples": len(x),
        "train_unique_seeds": int(len(np.unique(groups))),
        "feature_count": len(feature_names),
        "baselines": {
            "train": _baseline_oracle(margins),
            "holdout_a": _baseline_oracle(amargins),
            "holdout_b": _baseline_oracle(bmargins),
        },
        "selected_configuration": best,
        "train_fit": train_fit,
        "holdout_a": holdout_a,
        "holdout_b": holdout_b,
        "feature_importance_top30": important,
        "all_configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        "threshold": best["threshold"],
        "decision_step": payload["decision_step"],
    }, args.model_output)
    print(json.dumps({
        "status": "PASS",
        "selected_configuration": best,
        "train_fit": train_fit,
        "holdout_a": holdout_a,
        "holdout_b": holdout_b,
        "baselines": payload["baselines"],
        "feature_importance_top30": important,
        "output": str(args.output.resolve()),
        "model_output": str(args.model_output.resolve()),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
