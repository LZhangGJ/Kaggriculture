#!/usr/bin/env python3
"""Train Candidate8 on paired causal effects against KEEP."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from train_candidate8_competitive_ranker import make_features, split_groups


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def game_score(own: np.ndarray, opponent: np.ndarray) -> np.ndarray:
    margin = own - opponent
    return np.where(margin > 0, 1.0, np.where(margin == 0, 0.5, 0.0))


def paired_labels(
    state_id: np.ndarray, future_own: np.ndarray, future_opponent: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = len(state_id)
    paired_net = np.zeros(rows, dtype=np.float64)
    score_gain = np.zeros(rows, dtype=np.float64)
    margin_gain = np.zeros(rows, dtype=np.float64)
    cash_gain = np.zeros(rows, dtype=np.float64)
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        keep = int(indices[0])
        keep_own = future_own[keep]
        keep_opponent = future_opponent[keep]
        keep_score = game_score(keep_own, keep_opponent)
        keep_margin = keep_own - keep_opponent
        for index in indices:
            own = future_own[index]
            opponent = future_opponent[index]
            score = game_score(own, opponent)
            margin = own - opponent
            score_delta = score - keep_score
            margin_delta = margin - keep_margin
            cash_delta = own - keep_own
            sign = np.zeros(len(score), dtype=np.float64)
            better = (score_delta > 0) | (
                (score_delta == 0) & (
                    (margin_delta > 0) | (
                        (margin_delta == 0) & (cash_delta > 0)
                    )
                )
            )
            worse = (score_delta < 0) | (
                (score_delta == 0) & (
                    (margin_delta < 0) | (
                        (margin_delta == 0) & (cash_delta < 0)
                    )
                )
            )
            sign[better] = 1.0
            sign[worse] = -1.0
            paired_net[index] = sign.mean()
            score_gain[index] = score_delta.mean()
            margin_gain[index] = margin_delta.mean()
            cash_gain[index] = cash_delta.mean()
    return {
        "paired_net": paired_net,
        "score_gain": score_gain,
        "margin_gain": margin_gain,
        "cash_gain": cash_gain,
    }


def predicted_priority(
    paired_net: np.ndarray, score_gain: np.ndarray,
    margin_gain: np.ndarray, cash_gain: np.ndarray,
) -> np.ndarray:
    # Expected match score remains first.  Paired consistency breaks close
    # calls before margin/cash and is not allowed to override a clear score
    # difference.
    return (
        10_000_000.0 * score_gain
        + 500_000.0 * paired_net
        + margin_gain
        + 0.001 * cash_gain
    )


def evaluate(
    state_id: np.ndarray, opponents: np.ndarray, signatures: np.ndarray,
    labels: dict[str, np.ndarray], predictions: np.ndarray,
    groups: set[str], threshold: float,
) -> list[dict]:
    pred_priority = predicted_priority(
        predictions[:, 0], predictions[:, 1],
        predictions[:, 2] * 100_000.0, predictions[:, 3] * 100_000.0,
    )
    actual_priority = predicted_priority(
        labels["paired_net"], labels["score_gain"],
        labels["margin_gain"], labels["cash_gain"],
    )
    rows = []
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        opponent = str(opponents[indices[0]])
        if opponent not in groups:
            continue
        keep = 0
        order = np.lexsort((signatures[indices], -pred_priority[indices]))
        raw = int(order[0])
        predicted_uplift = float(
            pred_priority[indices[raw]] - pred_priority[indices[keep]]
        )
        selected = raw if predicted_uplift > threshold else keep
        selected_index = int(indices[selected])
        rows.append({
            "state_id": int(state),
            "opponent": opponent,
            "activated": selected != keep,
            "predicted_uplift": predicted_uplift,
            "paired_net": float(labels["paired_net"][selected_index]),
            "score_gain": float(labels["score_gain"][selected_index]),
            "margin_gain": float(labels["margin_gain"][selected_index]),
            "cash_gain": float(labels["cash_gain"][selected_index]),
            "priority_gain": float(actual_priority[selected_index]),
            "oracle_in_top8": int(indices[np.argmax(actual_priority[indices])])
            in {int(indices[value]) for value in order[:8]},
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    return {
        "states": len(rows),
        "opponents": len({row["opponent"] for row in rows}),
        "activation_rate": float(np.mean([row["activated"] for row in rows])),
        "mean_paired_net": float(np.mean([row["paired_net"] for row in rows])),
        "mean_score_gain": float(np.mean([row["score_gain"] for row in rows])),
        "mean_margin_gain": float(np.mean([row["margin_gain"] for row in rows])),
        "mean_cash_gain": float(np.mean([row["cash_gain"] for row in rows])),
        "negative_priority_rate": float(np.mean([
            row["priority_gain"] < 0 for row in rows
        ])),
        "positive_score_gain_rate": float(np.mean([
            row["score_gain"] > 0 for row in rows
        ])),
        "score_regression_rate": float(np.mean([
            row["score_gain"] < 0 for row in rows
        ])),
        "oracle_recall_at_8": float(np.mean([
            row["oracle_in_top8"] for row in rows
        ])),
        "rows": rows,
    }


def calibrate(raw_rows: list[dict]) -> tuple[float, dict]:
    predicted = np.asarray([row["predicted_uplift"] for row in raw_rows])
    realized = np.asarray([row["priority_gain"] for row in raw_rows])
    thresholds = np.unique(np.concatenate([
        np.asarray([0.0, np.inf]),
        np.quantile(predicted, np.linspace(0.0, 0.99, 100)),
    ]))
    candidates = []
    for threshold in thresholds:
        active = predicted > threshold
        gains = np.where(active, realized, 0.0)
        candidates.append({
            "threshold": float(threshold),
            "activation_rate": float(active.mean()),
            "mean_gain": float(gains.mean()),
            "negative_rate": float(np.mean(gains < 0)),
        })
    eligible = [
        row for row in candidates
        if row["negative_rate"] <= 0.05 and row["activation_rate"] >= 0.02
    ]
    best = max(
        eligible or candidates,
        key=lambda row: (row["mean_gain"], -row["negative_rate"]),
    )
    return float(best["threshold"]), best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--days", default="9")
    parser.add_argument("--estimators", type=int, default=768)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=20260830)
    args = parser.parse_args()
    wanted_days = {int(value) for value in args.days.split(",")}

    with np.load(args.dataset, allow_pickle=False) as raw:
        data = {name: np.asarray(raw[name]) for name in raw.files}
    mask = np.isin(data["decision_day"], list(wanted_days))
    for name, values in list(data.items()):
        if name != "feature_names":
            data[name] = values[mask]
    # State ids need not be consecutive; grouping only needs equality.
    labels = paired_labels(
        data["state_id"], data["future_own_cash"], data["future_opponent_cash"]
    )
    feature_names = [str(value) for value in data["feature_names"]]
    x = make_features(data["features"], data["family"], feature_names)
    fit, calibration, test = split_groups(
        data["opponent"], args.random_state, 0.20, 0.20
    )
    fit_mask = np.isin(data["opponent"], list(fit))
    y = np.column_stack([
        labels["paired_net"], labels["score_gain"],
        labels["margin_gain"] / 100_000.0,
        labels["cash_gain"] / 100_000.0,
    ])
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
    calibration_raw = evaluate(
        data["state_id"], data["opponent"], data["signature"],
        labels, prediction, calibration, -np.inf,
    )
    threshold, threshold_summary = calibrate(calibration_raw)
    calibration_rows = evaluate(
        data["state_id"], data["opponent"], data["signature"],
        labels, prediction, calibration, threshold,
    )
    test_rows = evaluate(
        data["state_id"], data["opponent"], data["signature"],
        labels, prediction, test, threshold,
    )
    test_summary = summarize(test_rows)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": [
            *feature_names, *(f"family_{index}" for index in range(9))
        ],
        "safe_threshold": threshold,
        "days": sorted(wanted_days),
        "fit_opponents": sorted(fit),
        "calibration_opponents": sorted(calibration),
        "test_opponents": sorted(test),
    }, args.model_output, compress=3)

    payload = {
        "schema": "kaggriculture.candidate8_paired_effect_ranker.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "days": sorted(wanted_days),
        "dataset": {
            "path": str(args.dataset), "sha256": sha256(args.dataset),
            "rows": int(len(data["state_id"])),
            "states": int(len(np.unique(data["state_id"]))),
        },
        "split": {
            "fit_opponents": sorted(fit),
            "calibration_opponents": sorted(calibration),
            "test_opponents": sorted(test),
        },
        "safe_threshold_calibration": threshold_summary,
        "calibration": summarize(calibration_rows),
        "test": test_summary,
        "model": {
            "path": str(args.model_output), "sha256": sha256(args.model_output),
            "parameters": model.get_params(),
        },
        "gate": {
            "heldout_score_gain_positive": test_summary["mean_score_gain"] > 0,
            "heldout_paired_net_positive": test_summary["mean_paired_net"] > 0,
            "negative_priority_rate_le_5pct": (
                test_summary["negative_priority_rate"] <= 0.05
            ),
            "oracle_recall_at_8_ge_80pct": (
                test_summary["oracle_recall_at_8"] >= 0.80
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
        "safe_threshold": threshold,
        "test": {key: value for key, value in test_summary.items() if key != "rows"},
        "gate": payload["gate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
