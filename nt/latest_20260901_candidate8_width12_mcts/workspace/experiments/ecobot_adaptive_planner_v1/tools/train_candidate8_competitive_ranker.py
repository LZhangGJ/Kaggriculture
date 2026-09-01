#!/usr/bin/env python3
"""Train a leave-opponent-out Candidate8 competitive ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


SCORE_WEIGHT = 10_000_000.0
MARGIN_WEIGHT = 1.0
CASH_WEIGHT = 0.001


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def make_features(raw: np.ndarray, family: np.ndarray, names: list[str]) -> np.ndarray:
    values = raw.astype(np.float64)
    for index, name in enumerate(names):
        if "x100" in name:
            values[:, index] /= 100.0
    one_hot = np.eye(9, dtype=np.float64)[family.astype(np.int64)]
    return np.concatenate([values, one_hot], axis=1)


def utility(score: np.ndarray, margin: np.ndarray, cash: np.ndarray) -> np.ndarray:
    return SCORE_WEIGHT * score + MARGIN_WEIGHT * margin + CASH_WEIGHT * cash


def split_groups(
    opponents: np.ndarray, random_state: int, test_fraction: float,
    calibration_fraction: float,
) -> tuple[set[str], set[str], set[str]]:
    groups = np.unique(opponents)
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(groups)
    test_count = max(1, int(round(len(groups) * test_fraction)))
    test = {str(value) for value in shuffled[:test_count]}
    remaining = shuffled[test_count:]
    calibration_count = max(1, int(round(len(remaining) * calibration_fraction)))
    calibration = {str(value) for value in remaining[:calibration_count]}
    fit = {str(value) for value in remaining[calibration_count:]}
    return fit, calibration, test


def state_rows(
    state_id: np.ndarray,
    opponents: np.ndarray,
    signature: np.ndarray,
    actual_score: np.ndarray,
    actual_margin: np.ndarray,
    actual_cash: np.ndarray,
    predicted_score: np.ndarray,
    predicted_margin: np.ndarray,
    predicted_cash: np.ndarray,
    groups: set[str],
    safe_threshold: float | None,
) -> list[dict]:
    actual_utility = utility(actual_score, actual_margin, actual_cash)
    predicted_utility = utility(predicted_score, predicted_margin, predicted_cash)
    rows = []
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        opponent = str(opponents[indices[0]])
        if opponent not in groups:
            continue
        keep = 0
        oracle = int(np.argmax(actual_utility[indices]))
        predicted_order = np.lexsort((signature[indices], -predicted_utility[indices]))
        raw = int(predicted_order[0])
        uplift = float(
            predicted_utility[indices[raw]] - predicted_utility[indices[keep]]
        )
        selected = raw
        if safe_threshold is not None and uplift <= safe_threshold:
            selected = keep
        oracle_index = int(indices[oracle])
        raw_index = int(indices[raw])
        selected_index = int(indices[selected])
        keep_index = int(indices[keep])
        top8 = {int(indices[value]) for value in predicted_order[:8]}
        rows.append({
            "state_id": int(state),
            "opponent": opponent,
            "arms": int(len(indices)),
            "oracle_signature": int(signature[oracle_index]),
            "selected_signature": int(signature[selected_index]),
            "oracle_exact": selected_index == oracle_index,
            "oracle_in_predicted_top8": oracle_index in top8,
            "activated": selected != keep,
            "predicted_uplift": uplift,
            "score_gain": float(actual_score[selected_index] - actual_score[keep_index]),
            "margin_gain": float(actual_margin[selected_index] - actual_margin[keep_index]),
            "cash_gain": float(actual_cash[selected_index] - actual_cash[keep_index]),
            "utility_gain": float(actual_utility[selected_index] - actual_utility[keep_index]),
            "raw_utility_gain": float(actual_utility[raw_index] - actual_utility[keep_index]),
            "selected_score": float(actual_score[selected_index]),
            "keep_score": float(actual_score[keep_index]),
            "selected_margin": float(actual_margin[selected_index]),
            "keep_margin": float(actual_margin[keep_index]),
        })
    return rows


def summarize(rows: list[dict], threshold: float | None) -> dict:
    return {
        "states": len(rows),
        "opponents": len({row["opponent"] for row in rows}),
        "safe_threshold": threshold,
        "activation_rate": float(np.mean([row["activated"] for row in rows])),
        "exact_oracle_recall": float(np.mean([row["oracle_exact"] for row in rows])),
        "oracle_recall_at_8": float(np.mean([
            row["oracle_in_predicted_top8"] for row in rows
        ])),
        "mean_score_gain_vs_keep": float(np.mean([row["score_gain"] for row in rows])),
        "mean_margin_gain_vs_keep": float(np.mean([row["margin_gain"] for row in rows])),
        "mean_cash_gain_vs_keep": float(np.mean([row["cash_gain"] for row in rows])),
        "mean_utility_gain_vs_keep": float(np.mean([
            row["utility_gain"] for row in rows
        ])),
        "negative_utility_rate": float(np.mean([
            row["utility_gain"] < 0 for row in rows
        ])),
        "score_regression_rate": float(np.mean([
            row["score_gain"] < 0 for row in rows
        ])),
        "positive_score_gain_rate": float(np.mean([
            row["score_gain"] > 0 for row in rows
        ])),
        "rows": rows,
    }


def calibrate_threshold(rows: list[dict]) -> tuple[float, dict]:
    predicted = np.asarray([row["predicted_uplift"] for row in rows])
    realized = np.asarray([row["raw_utility_gain"] for row in rows])
    thresholds = np.unique(np.concatenate([
        np.asarray([0.0, np.inf]),
        np.quantile(predicted, np.linspace(0.0, 0.98, 80)),
    ]))
    candidates = []
    for threshold in thresholds:
        active = predicted > threshold
        gains = np.where(active, realized, 0.0)
        candidates.append({
            "threshold": float(threshold),
            "activation_rate": float(active.mean()),
            "mean_utility_gain": float(gains.mean()),
            "negative_utility_rate": float(np.mean(gains < 0)),
            "positive_utility_rate": float(np.mean(gains > 0)),
        })
    eligible = [
        row for row in candidates
        if row["negative_utility_rate"] <= 0.05
        and row["activation_rate"] >= 0.05
    ]
    best = max(
        eligible or candidates,
        key=lambda row: (
            row["mean_utility_gain"],
            -row["negative_utility_rate"],
            row["activation_rate"],
        ),
    )
    return float(best["threshold"]), best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--test-opponent-fraction", type=float, default=0.20)
    parser.add_argument("--calibration-opponent-fraction", type=float, default=0.20)
    parser.add_argument("--estimators", type=int, default=384)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=20260830)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"])
        opponents = np.asarray(data["opponent"])
        signature = np.asarray(data["signature"])
        family = np.asarray(data["family"])
        raw_features = np.asarray(data["features"])
        score = np.asarray(data["expected_score_rate"])
        margin = np.asarray(data["expected_margin"])
        cash = np.asarray(data["expected_own_cash"])
        feature_names = [str(value) for value in data["feature_names"]]

    fit_groups, calibration_groups, test_groups = split_groups(
        opponents,
        args.random_state,
        args.test_opponent_fraction,
        args.calibration_opponent_fraction,
    )
    fit_mask = np.isin(opponents, list(fit_groups))
    x = make_features(raw_features, family, feature_names)
    y = np.column_stack([score, margin / 100_000.0, cash / 100_000.0])
    model = ExtraTreesRegressor(
        n_estimators=args.estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=0.9,
        n_jobs=16,
        random_state=args.random_state,
    )
    model.fit(x[fit_mask], y[fit_mask])
    prediction = model.predict(x)
    predicted_score = prediction[:, 0]
    predicted_margin = prediction[:, 1] * 100_000.0
    predicted_cash = prediction[:, 2] * 100_000.0

    calibration_raw = state_rows(
        state_id, opponents, signature, score, margin, cash,
        predicted_score, predicted_margin, predicted_cash,
        calibration_groups, None,
    )
    threshold, threshold_summary = calibrate_threshold(calibration_raw)
    calibration = summarize(state_rows(
        state_id, opponents, signature, score, margin, cash,
        predicted_score, predicted_margin, predicted_cash,
        calibration_groups, threshold,
    ), threshold)
    test = summarize(state_rows(
        state_id, opponents, signature, score, margin, cash,
        predicted_score, predicted_margin, predicted_cash,
        test_groups, threshold,
    ), threshold)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": [
            *feature_names,
            *(f"family_{index}" for index in range(9)),
        ],
        "safe_threshold": threshold,
        "objective": {
            "score_weight": SCORE_WEIGHT,
            "margin_weight": MARGIN_WEIGHT,
            "cash_weight": CASH_WEIGHT,
        },
        "fit_opponents": sorted(fit_groups),
        "calibration_opponents": sorted(calibration_groups),
        "test_opponents": sorted(test_groups),
    }, args.model_output, compress=3)

    payload = {
        "schema": "kaggriculture.candidate8_competitive_ranker.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(args.dataset),
            "sha256": sha256(args.dataset),
            "rows": int(len(state_id)),
            "states": int(len(np.unique(state_id))),
            "opponents": int(len(np.unique(opponents))),
        },
        "split": {
            "fit_opponents": sorted(fit_groups),
            "calibration_opponents": sorted(calibration_groups),
            "test_opponents": sorted(test_groups),
        },
        "model": {
            "type": "ExtraTreesRegressor_multioutput",
            "parameters": model.get_params(),
            "artifact": str(args.model_output),
            "sha256": sha256(args.model_output),
        },
        "objective": {
            "score_weight": SCORE_WEIGHT,
            "margin_weight": MARGIN_WEIGHT,
            "cash_weight": CASH_WEIGHT,
        },
        "safe_threshold_calibration": threshold_summary,
        "calibration": calibration,
        "test": test,
        "gate": {
            "heldout_score_gain_positive": test["mean_score_gain_vs_keep"] > 0,
            "heldout_margin_gain_positive": test["mean_margin_gain_vs_keep"] > 0,
            "heldout_negative_utility_rate_le_5pct": (
                test["negative_utility_rate"] <= 0.05
            ),
            "heldout_oracle_recall_at_8_ge_80pct": (
                test["oracle_recall_at_8"] >= 0.80
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "model": str(args.model_output),
        "fit_opponents": len(fit_groups),
        "calibration_opponents": len(calibration_groups),
        "test_opponents": len(test_groups),
        "safe_threshold": threshold,
        "test": {key: value for key, value in test.items() if key != "rows"},
        "gate": payload["gate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
