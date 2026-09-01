#!/usr/bin/env python3
"""Evaluate an information-consistent public-context Candidate8 lookup.

Every exact public context receives one candidate preference aggregated across
fit opponents.  Held-out opponent routes cannot influence the lookup.  This is
particularly useful for early states where many future route scripts are still
observationally indistinguishable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from train_candidate8_competitive_ranker import (
    CASH_WEIGHT,
    MARGIN_WEIGHT,
    SCORE_WEIGHT,
    split_groups,
    utility,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def state_records(data: dict[str, np.ndarray]) -> list[dict]:
    records = []
    util = utility(
        data["expected_score_rate"],
        data["expected_margin"],
        data["expected_own_cash"],
    )
    for state in np.unique(data["state_id"]):
        indices = np.flatnonzero(data["state_id"] == state)
        # Candidate fields occupy columns 0..19; context is identical for all
        # arms in a frozen state.
        context = data["features"][indices[0], 20:].tobytes()
        records.append({
            "state_id": int(state),
            "opponent": str(data["opponent"][indices[0]]),
            "day": int(data["decision_day"][indices[0]]),
            "context": context,
            "indices": indices,
            "signature_to_index": {
                int(data["signature"][index]): int(index) for index in indices
            },
            "utility": util,
        })
    return records


def build_lookup(
    records: list[dict], data: dict[str, np.ndarray], fit_groups: set[str],
    minimum_context_support: int,
) -> dict[bytes, dict]:
    accum: dict[bytes, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    context_opponents: dict[bytes, set[str]] = defaultdict(set)
    for record in records:
        if record["opponent"] not in fit_groups:
            continue
        context_opponents[record["context"]].add(record["opponent"])
        for signature, index in record["signature_to_index"].items():
            accum[record["context"]][signature].append(index)

    lookup = {}
    for context, signatures in accum.items():
        support = len(context_opponents[context])
        if support < minimum_context_support:
            continue
        candidates = []
        for signature, indices in signatures.items():
            index_array = np.asarray(indices, dtype=np.int64)
            candidates.append({
                "signature": signature,
                "support": len(indices),
                "mean_score": float(data["expected_score_rate"][index_array].mean()),
                "mean_margin": float(data["expected_margin"][index_array].mean()),
                "mean_cash": float(data["expected_own_cash"][index_array].mean()),
            })
        candidates.sort(key=lambda row: (
            row["mean_score"], row["mean_margin"], row["mean_cash"],
            -row["signature"],
        ), reverse=True)
        # KEEP is guaranteed to be arm zero by the Candidate8 interface.  Infer
        # its signature from the data instead of freezing a numeric constant.
        first_signatures = []
        for record in records:
            if record["context"] == context and record["opponent"] in fit_groups:
                first_signatures.append(int(data["signature"][record["indices"][0]]))
        keep_signature = max(set(first_signatures), key=first_signatures.count)
        keep = next(row for row in candidates if row["signature"] == keep_signature)
        best = candidates[0]
        predicted_uplift = (
            SCORE_WEIGHT * (best["mean_score"] - keep["mean_score"])
            + MARGIN_WEIGHT * (best["mean_margin"] - keep["mean_margin"])
            + CASH_WEIGHT * (best["mean_cash"] - keep["mean_cash"])
        )
        lookup[context] = {
            "support_opponents": support,
            "best_signature": best["signature"],
            "keep_signature": keep["signature"],
            "predicted_uplift": float(predicted_uplift),
        }
    return lookup


def evaluate(
    records: list[dict], data: dict[str, np.ndarray], groups: set[str],
    lookup: dict[bytes, dict], threshold: float,
) -> list[dict]:
    rows = []
    for record in records:
        if record["opponent"] not in groups:
            continue
        indices = record["indices"]
        keep_index = int(indices[0])
        entry = lookup.get(record["context"])
        selected_index = keep_index
        covered = entry is not None
        available = False
        predicted_uplift = 0.0
        if entry is not None:
            predicted_uplift = entry["predicted_uplift"]
            selected_index = record["signature_to_index"].get(
                entry["best_signature"], keep_index
            )
            available = selected_index != keep_index or (
                entry["best_signature"]
                == int(data["signature"][keep_index])
            )
            if predicted_uplift <= threshold:
                selected_index = keep_index
        score_gain = float(
            data["expected_score_rate"][selected_index]
            - data["expected_score_rate"][keep_index]
        )
        margin_gain = float(
            data["expected_margin"][selected_index]
            - data["expected_margin"][keep_index]
        )
        cash_gain = float(
            data["expected_own_cash"][selected_index]
            - data["expected_own_cash"][keep_index]
        )
        rows.append({
            "state_id": record["state_id"],
            "opponent": record["opponent"],
            "day": record["day"],
            "covered": covered,
            "selected_available": available,
            "activated": selected_index != keep_index,
            "predicted_uplift": predicted_uplift,
            "score_gain": score_gain,
            "margin_gain": margin_gain,
            "cash_gain": cash_gain,
            "utility_gain": float(
                SCORE_WEIGHT * score_gain
                + MARGIN_WEIGHT * margin_gain
                + CASH_WEIGHT * cash_gain
            ),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    return {
        "states": len(rows),
        "opponents": len({row["opponent"] for row in rows}),
        "coverage_rate": float(np.mean([row["covered"] for row in rows])),
        "activation_rate": float(np.mean([row["activated"] for row in rows])),
        "mean_score_gain": float(np.mean([row["score_gain"] for row in rows])),
        "mean_margin_gain": float(np.mean([row["margin_gain"] for row in rows])),
        "mean_cash_gain": float(np.mean([row["cash_gain"] for row in rows])),
        "negative_utility_rate": float(np.mean([
            row["utility_gain"] < 0 for row in rows
        ])),
        "positive_score_gain_rate": float(np.mean([
            row["score_gain"] > 0 for row in rows
        ])),
        "score_regression_rate": float(np.mean([
            row["score_gain"] < 0 for row in rows
        ])),
    }


def calibrate(rows: list[dict]) -> tuple[float, dict]:
    predicted = np.asarray([row["predicted_uplift"] for row in rows])
    realized = np.asarray([row["utility_gain"] for row in rows])
    thresholds = np.unique(np.concatenate([
        np.asarray([0.0, np.inf]),
        np.quantile(predicted, np.linspace(0.0, 0.98, 60)),
    ]))
    candidates = []
    for threshold in thresholds:
        active = predicted > threshold
        gain = np.where(active, realized, 0.0)
        candidates.append({
            "threshold": float(threshold),
            "activation_rate": float(active.mean()),
            "mean_gain": float(gain.mean()),
            "negative_rate": float(np.mean(gain < 0)),
        })
    eligible = [row for row in candidates if row["negative_rate"] <= 0.05]
    best = max(
        eligible or candidates,
        key=lambda row: (row["mean_gain"], -row["negative_rate"]),
    )
    return float(best["threshold"]), best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-context-support", type=int, default=3)
    parser.add_argument("--test-opponent-fraction", type=float, default=0.20)
    parser.add_argument("--calibration-opponent-fraction", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=20260830)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as raw:
        data = {name: np.asarray(raw[name]) for name in raw.files}
    records = state_records(data)
    fit, calibration, test = split_groups(
        data["opponent"], args.random_state,
        args.test_opponent_fraction, args.calibration_opponent_fraction,
    )
    lookup = build_lookup(records, data, fit, args.minimum_context_support)
    calibration_raw = evaluate(records, data, calibration, lookup, -np.inf)
    test_raw = evaluate(records, data, test, lookup, -np.inf)
    threshold, threshold_summary = calibrate(calibration_raw)
    calibration_rows = evaluate(records, data, calibration, lookup, threshold)
    test_rows = evaluate(records, data, test, lookup, threshold)
    by_day = {}
    for day in sorted({row["day"] for row in test_rows}):
        by_day[str(day)] = summarize([
            row for row in test_rows if row["day"] == day
        ])

    payload = {
        "schema": "kaggriculture.candidate8_public_context_lookup.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "minimum_context_support": args.minimum_context_support,
            "random_state": args.random_state,
        },
        "split": {
            "fit_opponents": sorted(fit),
            "calibration_opponents": sorted(calibration),
            "test_opponents": sorted(test),
        },
        "lookup_contexts": len(lookup),
        "safe_threshold_calibration": threshold_summary,
        "raw_test": summarize(test_raw),
        "calibration": summarize(calibration_rows),
        "test": summarize(test_rows),
        "test_by_day": by_day,
        "test_rows": test_rows,
        "input": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "lookup_contexts": len(lookup),
        "safe_threshold": threshold,
        "raw_test": payload["raw_test"],
        "test": payload["test"],
        "test_by_day": by_day,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
