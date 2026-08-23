#!/usr/bin/env python3
"""Fit a grouped-CV LGBM router against direct win-score deltas."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold


warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/fusion_champion_v1/tools"))
from train_k320_route_rescue_tree import _baseline_oracle, _load, _metrics  # noqa: E402


ARM_COUNT = 6


def _expand(x: np.ndarray, margins: np.ndarray, indices: np.ndarray):
    """Rows for the five override arms; target is {-1,0,+1} score delta."""

    count = len(indices)
    route_ids = np.tile(np.arange(1, ARM_COUNT, dtype=np.int64), count)
    one_hot = np.eye(ARM_COUNT - 1, dtype=np.float32)[route_ids - 1]
    features = np.concatenate(
        (np.repeat(x[indices], ARM_COUNT - 1, axis=0), one_hot), axis=1
    )
    base_win = margins[0, indices] > 0
    route_win = margins[1:, indices].T.reshape(-1) > 0
    targets = (
        route_win.astype(np.float32)
        - np.repeat(base_win, ARM_COUNT - 1).astype(np.float32)
    )
    changed = targets != 0
    return features, targets, changed


def _predict_matrix(model, x: np.ndarray) -> np.ndarray:
    dummy = np.zeros((ARM_COUNT, len(x)), dtype=np.int64)
    features, _, _ = _expand(x, dummy, np.arange(len(x)))
    prediction = np.zeros((ARM_COUNT, len(x)), dtype=np.float32)
    prediction[1:] = model.predict(features).reshape(len(x), ARM_COUNT - 1).T
    return prediction


def _choices(prediction: np.ndarray, threshold: float) -> np.ndarray:
    best = 1 + np.argmax(prediction[1:], axis=0).astype(np.int64)
    score = prediction[best, np.arange(prediction.shape[1])]
    return np.where(score >= threshold, best, 0).astype(np.int64)


def _choose_threshold(margins: np.ndarray, prediction: np.ndarray):
    candidates = np.unique(
        np.concatenate(
            (
                np.linspace(-0.25, 0.75, 201),
                np.max(prediction[1:], axis=0),
            )
        )
    )
    rows = []
    for threshold in candidates:
        choice = _choices(prediction, float(threshold))
        rows.append((float(threshold), _metrics(margins, choice)))
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
    for leaves in (3, 5, 7, 11):
        for min_child in (8, 16, 32):
            for estimators in (60, 120):
                for changed_weight in (4.0, 8.0, 16.0):
                    oof = np.zeros((ARM_COUNT, len(x)), dtype=np.float32)
                    for fit_index, valid_index in GroupKFold(n_splits=5).split(
                        x, groups=groups
                    ):
                        fit_x, fit_y, changed = _expand(x, margins, fit_index)
                        model = lgb.LGBMRegressor(
                            objective="regression_l2",
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
                        model.fit(
                            fit_x,
                            fit_y,
                            sample_weight=np.where(changed, changed_weight, 1.0),
                        )
                        oof[:, valid_index] = _predict_matrix(model, x[valid_index])
                    threshold, metrics = _choose_threshold(margins, oof)
                    configurations.append(
                        {
                            "num_leaves": leaves,
                            "min_child_samples": min_child,
                            "n_estimators": estimators,
                            "changed_outcome_weight": changed_weight,
                            "threshold": threshold,
                            "oof": metrics,
                        }
                    )

    best = max(
        configurations,
        key=lambda row: (
            row["oof"]["score_rate"],
            -row["oof"]["harmed_wins"],
            -row["oof"]["switch_rate"],
            row["oof"]["mean_margin"],
        ),
    )
    fit_x, fit_y, changed = _expand(x, margins, np.arange(len(x)))
    model = lgb.LGBMRegressor(
        objective="regression_l2",
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
    model.fit(
        fit_x,
        fit_y,
        sample_weight=np.where(changed, best["changed_outcome_weight"], 1.0),
    )

    def evaluate(features, outcomes):
        prediction = _predict_matrix(model, features)
        return _metrics(outcomes, _choices(prediction, best["threshold"]))

    expanded_names = feature_names + [f"override_arm_{arm}" for arm in range(1, 6)]
    importance = sorted(
        (
            {"feature": name, "gain": float(value)}
            for name, value in zip(
                expanded_names, model.feature_importances_, strict=True
            )
        ),
        key=lambda row: row["gain"],
        reverse=True,
    )[:30]
    payload = {
        "schema": "kaggriculture.fusion_champion.k320-route-score-delta-lgbm.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "official_package_version": "1.32.7",
        "decision_step": int(train_payload["feature_step"]),
        "future_leakage": False,
        "target": "route_win_indicator - baseline_win_indicator",
        "group_cv": "5 folds grouped by episode seed",
        "train_samples": len(x),
        "train_unique_seeds": int(len(np.unique(groups))),
        "feature_count": len(feature_names),
        "baselines": {
            "train": _baseline_oracle(margins),
            "holdout_a": _baseline_oracle(amargins),
            "holdout_b": _baseline_oracle(bmargins),
        },
        "selected_configuration": best,
        "train_fit": evaluate(x, margins),
        "holdout_a": evaluate(ax, amargins),
        "holdout_b": evaluate(bx, bmargins),
        "feature_importance_top30": importance,
        "all_configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "threshold": best["threshold"],
            "decision_step": payload["decision_step"],
            "target": payload["target"],
        },
        args.model_output,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_configuration": best,
                "train_fit": payload["train_fit"],
                "holdout_a": payload["holdout_a"],
                "holdout_b": payload["holdout_b"],
                "baselines": payload["baselines"],
                "feature_importance_top30": importance,
                "output": str(args.output.resolve()),
                "model_output": str(args.model_output.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
