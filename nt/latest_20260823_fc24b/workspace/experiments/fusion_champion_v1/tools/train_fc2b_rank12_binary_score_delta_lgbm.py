#!/usr/bin/env python3
"""Grouped-CV, no-future-leakage binary route rescue for current FC2B."""

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
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/fusion_champion_v1/tools"))
from screen_k320_rank12_binary_public_threshold_rule import load_binary  # noqa: E402
from train_k320_route_rescue_tree import _baseline_oracle, _metrics  # noqa: E402


def target(margins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = (margins[1] > 0).astype(np.float32) - (margins[0] > 0).astype(np.float32)
    return values, values != 0


def choices(prediction: np.ndarray, threshold: float) -> np.ndarray:
    return (prediction >= threshold).astype(np.int64)


def choose_threshold(margins: np.ndarray, prediction: np.ndarray):
    candidates = np.unique(
        np.concatenate((np.linspace(-0.25, 0.75, 201), prediction))
    )
    rows = []
    for threshold in candidates:
        metric = _metrics(margins, choices(prediction, float(threshold)))
        rows.append((float(threshold), metric))
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


def make_model(leaves: int, min_child: int, estimators: int):
    return lgb.LGBMRegressor(
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout-a", type=Path, required=True)
    parser.add_argument("--holdout-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()

    train_payload, x, margins, groups, names = load_binary(args.train)
    _, ax, amargins, _, anames = load_binary(args.holdout_a, names)
    _, bx, bmargins, _, bnames = load_binary(args.holdout_b, names)
    if anames != names or bnames != names:
        raise AssertionError("holdout feature schema mismatch")
    y, changed = target(margins)

    configurations = []
    for leaves in (3, 5, 7, 11):
        for min_child in (8, 16, 32):
            for estimators in (60, 120):
                for changed_weight in (4.0, 8.0, 16.0):
                    oof = np.zeros(len(x), dtype=np.float32)
                    for fit_index, valid_index in GroupKFold(n_splits=5).split(
                        x, groups=groups
                    ):
                        model = make_model(leaves, min_child, estimators)
                        model.fit(
                            x[fit_index],
                            y[fit_index],
                            sample_weight=np.where(
                                changed[fit_index], changed_weight, 1.0
                            ),
                        )
                        oof[valid_index] = model.predict(x[valid_index])
                    threshold, metric = choose_threshold(margins, oof)
                    configurations.append(
                        {
                            "num_leaves": leaves,
                            "min_child_samples": min_child,
                            "n_estimators": estimators,
                            "changed_outcome_weight": changed_weight,
                            "threshold": threshold,
                            "oof": metric,
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
    model = make_model(
        best["num_leaves"], best["min_child_samples"], best["n_estimators"]
    )
    model.fit(
        x,
        y,
        sample_weight=np.where(changed, best["changed_outcome_weight"], 1.0),
    )

    def evaluate(matrix: np.ndarray, outcomes: np.ndarray) -> dict:
        return _metrics(
            outcomes, choices(model.predict(matrix), best["threshold"])
        )

    importance = sorted(
        (
            {"feature": name, "gain": float(value)}
            for name, value in zip(names, model.feature_importances_, strict=True)
        ),
        key=lambda row: row["gain"],
        reverse=True,
    )[:30]
    payload = {
        "schema": "kaggriculture.fusion_champion.fc2b-rank12-binary-score-delta-lgbm.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "official_package_version": "1.32.7",
        "decision_step": int(train_payload["feature_step"]),
        "candidate_stack": train_payload.get("candidate_stack"),
        "future_leakage": False,
        "target": "route4_win_indicator - fc2b_win_indicator",
        "group_cv": "5 folds grouped by episode seed; both seats stay together",
        "train_samples": int(len(x)),
        "train_unique_seeds": int(len(np.unique(groups))),
        "feature_count": int(len(names)),
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
            "feature_names": names,
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
