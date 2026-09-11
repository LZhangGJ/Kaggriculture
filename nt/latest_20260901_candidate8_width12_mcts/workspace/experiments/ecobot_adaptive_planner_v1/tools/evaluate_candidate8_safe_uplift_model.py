#!/usr/bin/env python3
"""Evaluate a frozen Candidate8 safe-uplift model on an untouched dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from train_candidate8_day6_safe_uplift import (
    make_features,
    paired_mean_margin_target,
    paired_targets,
    select_rows,
    summarize,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bank", choices=("even", "odd", "all"), default="odd")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"])
        opponents = np.asarray(data["opponent"])
        seat = np.asarray(data["seat"])
        family = np.asarray(data["family"])
        raw_features = np.asarray(data["features"])
        feature_names = [str(value) for value in data["feature_names"]]
        own = np.asarray(data["future_own_cash"], dtype=np.float64)
        rival = np.asarray(data["future_opponent_cash"], dtype=np.float64)

    model = joblib.load(args.model)
    x, model_feature_names = make_features(
        raw_features, family, seat, feature_names
    )
    if model_feature_names != list(model["feature_names"]):
        raise ValueError("model and dataset feature schemas differ")
    if args.bank == "even":
        columns = np.arange(0, own.shape[1], 2, dtype=np.int64)
    elif args.bank == "odd":
        columns = np.arange(1, own.shape[1], 2, dtype=np.int64)
    else:
        columns = np.arange(own.shape[1], dtype=np.int64)
    score_target, margin_target, cash_target = paired_targets(
        state_id, own[:, columns], rival[:, columns], float(model["z"])
    )
    margin_mean_target = paired_mean_margin_target(
        state_id, own[:, columns], rival[:, columns]
    )
    predicted_score = model["score_model"].predict(x)
    predicted_margin = model["margin_model"].predict(x)
    predicted_margin_mean = (
        model["margin_mean_model"].predict(x)
        if "margin_mean_model" in model
        else predicted_margin
    )
    all_states = {int(value) for value in np.unique(state_id)}
    rows = select_rows(
        all_states,
        state_id,
        predicted_score,
        predicted_margin,
        score_target,
        margin_target,
        cash_target,
        float(model["score_threshold"]),
        float(model["margin_threshold"]),
        predicted_margin_mean,
        margin_mean_target,
    )
    state_opponent = {}
    state_seat = {}
    for state in all_states:
        index = int(np.flatnonzero(state_id == state)[0])
        state_opponent[state] = str(opponents[index])
        state_seat[state] = int(seat[index])

    def subset(predicate) -> dict:
        selected = [row for row in rows if predicate(row["state_id"])]
        return summarize(selected)

    payload = {
        "schema": "kaggriculture.candidate8-safe-uplift-frozen-eval.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "The model and thresholds are frozen before this dataset is read. "
            "Only the selected future columns are used for evaluation labels."
        ),
        "inputs": {
            "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
            "model": {"path": str(args.model), "sha256": sha256(args.model)},
        },
        "bank": {"name": args.bank, "columns": columns.tolist()},
        "thresholds": {
            "score": float(model["score_threshold"]),
            "margin": float(model["margin_threshold"]),
        },
        "overall": summarize(rows),
        "prediction_targets": {
            "score": "paired win-rate delta versus KEEP",
            "margin": "paired mean final-cash margin delta versus KEEP",
            "margin_safety": "paired margin lower confidence bound versus KEEP",
        },
        "mean_margin_target_summary": {
            "mean": float(np.mean(margin_mean_target)),
            "standard_deviation": float(np.std(margin_mean_target)),
        },
        "by_opponent": {
            opponent: subset(lambda state, value=opponent: state_opponent[state] == value)
            for opponent in sorted(set(state_opponent.values()))
        },
        "by_seat": {
            str(value): subset(lambda state, expected=value: state_seat[state] == expected)
            for value in (0, 1)
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"overall": payload["overall"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
