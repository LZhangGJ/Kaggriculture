#!/usr/bin/env python3
"""Benchmark the deployable public-only O1.5 selector on one candidate state."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from train_candidate8_day6_safe_uplift import make_features


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names.*",
        category=UserWarning,
    )

    bundle = joblib.load(args.model)
    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"])
        day = np.asarray(data["decision_day"])
        family = np.asarray(data["family"])
        seat = np.asarray(data["seat"])
        raw = np.asarray(data["features"], dtype=np.float32)
        source_names = [str(value) for value in data["feature_names"]]

    model_x, model_names = make_features(raw, family, seat, source_names)
    model_x = np.asarray(model_x, dtype=np.float32)
    if model_names != list(bundle["feature_names"]):
        raise RuntimeError("runtime public feature contract differs from model")
    if bundle["variant"] != "RESPONSE_DISTILLED_LISTWISE":
        raise RuntimeError("benchmark target is not the O1.5 distilled model")

    stage_for_day = {6: "EARLY", 12: "MID", 18: "LATE"}
    stage_results = {}
    for decision_day, stage_name in stage_for_day.items():
        candidate_states = np.unique(state_id[day == decision_day])
        selected_state = max(
            candidate_states,
            key=lambda value: int(np.sum(state_id == value)),
        )
        indices = np.flatnonzero(state_id == selected_state)
        x = model_x[indices]
        stage = bundle["stage_models"][stage_name]

        def predict_once() -> None:
            bundle["state_effect_model"].predict_proba(x)
            stage["ranker"].predict(x)
            stage["positive_lcb_model"].predict_proba(x)
            stage["negative_lcb_model"].predict_proba(x)
            stage["score_improve_model"].predict_proba(x)
            stage["score_regress_model"].predict_proba(x)

        for _ in range(10):
            predict_once()
        durations_ms = np.empty(args.repetitions, dtype=np.float64)
        for repetition in range(args.repetitions):
            started = time.perf_counter()
            predict_once()
            durations_ms[repetition] = (time.perf_counter() - started) * 1000.0
        stage_results[stage_name] = {
            "decision_day": decision_day,
            "candidate_rows": int(len(indices)),
            "median_ms": float(np.median(durations_ms)),
            "p95_ms": float(np.quantile(durations_ms, 0.95)),
            "maximum_ms": float(np.max(durations_ms)),
            "p95_microseconds_per_candidate": float(
                np.quantile(durations_ms, 0.95) * 1000.0 / len(indices)
            ),
        }

    maximum_p95_ms = max(value["p95_ms"] for value in stage_results.values())
    payload = {
        "schema": "kaggriculture.candidate8-o15-runtime-benchmark.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if maximum_p95_ms < 1000.0 else "FAIL",
        "boundary": (
            "Measures only the deployable public-feature LightGBM selector on "
            "one complete candidate state. It does not claim that the full "
            "candidate generator or an integrated Agent passes the one-second budget."
        ),
        "runtime_response_rollouts_required": False,
        "public_feature_width": int(model_x.shape[1]),
        "repetitions": args.repetitions,
        "stages": stage_results,
        "maximum_p95_ms": maximum_p95_ms,
        "one_second_selector_gate_passed": maximum_p95_ms < 1000.0,
        "inputs": {
            "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
            "model": {"path": str(args.model), "sha256": sha256(args.model)},
        },
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
