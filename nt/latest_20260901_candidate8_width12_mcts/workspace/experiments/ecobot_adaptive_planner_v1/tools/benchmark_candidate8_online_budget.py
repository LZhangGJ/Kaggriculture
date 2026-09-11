#!/usr/bin/env python3
"""Benchmark broad candidate generation and warm Python ranker inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

import fast_kaggriculture as fk


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def broad_context() -> dict:
    return {
        "day": 8,
        "targets": [8, 4, 2, 10, 5, 2, 4, 4],
        "irreversible_floor": [4, 0, 0, 4, 2, 1, 2, 2],
        "caps": [40, 24, 24, 40, 24, 12, 20, 20],
        "marginal_value": [600, 900, 1500, 2600, 2200, 6000, 4600, 3200],
        "purchase_cost": [48, 42, 96, 228, 169, 1800, 1200, 600],
        "daily_action_load": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.5],
        "first_cash_lag_days": [3, 4, 5, 6, 7, 3, 4, 2],
        "liquid_cash": 100_000,
        "protected_cash": 1_000,
        "financeable_inventory_value": 5_000,
        "unlocked_quadrants": 2,
        "maximum_quadrants": 4,
        "productive_tiles": 39,
        "hands": 6,
        "maximum_hands": 12,
        "next_hand_cost": 100,
        "next_quadrant_cost": 4_000,
        "tiles_per_quadrant": 25,
        "current_daily_action_load": 80,
        "hard_deadline_load": 10,
        "estimated_travel_load": 20,
        "delayed_loss": 300,
        "market_slots_available": 10,
        "sellable_inventory": [10, 8, 4, 2, 6, 3, 5, 2, 1],
        "market_prices": [48, 42, 96, 228, 169, 60, 80, 120, 90],
        "demand_within_day": [2, 1, 0, 4, 2, 0, 3, 1, 0],
        "recovery_issues": 15,
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Measure only steady-state decision time. Module/model startup occurs once
    # when the submission agent is imported, not once per game step.
    generation_ms = []
    sample = None
    for _ in range(args.repeats + 5):
        started = time.perf_counter_ns()
        sample = fk.generate_adaptive_candidates(broad_context())
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if len(generation_ms) >= 5:
            generation_ms.append(elapsed)
        else:
            generation_ms.append(elapsed)
    generation_ms = generation_ms[5:]

    load_started = time.perf_counter()
    artifact = joblib.load(args.model)
    model = artifact["model"]
    model_load_seconds = time.perf_counter() - load_started
    with np.load(args.dataset, allow_pickle=False) as data:
        state = int(np.asarray(data["state_id"])[0])
        rows = np.flatnonzero(np.asarray(data["state_id"]) == state)[:64]
        raw = np.asarray(data["features"][rows], dtype=np.float64)
        family = np.asarray(data["family"][rows], dtype=np.int64)
        feature_names = [str(value) for value in data["feature_names"].tolist()]
    for index, name in enumerate(feature_names):
        if "x100" in name:
            raw[:, index] /= 100.0
    features = np.concatenate([raw, np.eye(9)[family]], axis=1)
    model.predict(features)  # warm thread pool and tree traversal
    inference_ms = []
    for _ in range(args.repeats):
        started = time.perf_counter_ns()
        model.predict(features)
        inference_ms.append((time.perf_counter_ns() - started) / 1e6)

    combined_p95 = percentile(generation_ms, 0.95) + percentile(inference_ms, 0.95)
    payload = {
        "schema": "kaggriculture.candidate8_online_budget.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if combined_p95 < 1_000.0 else "FAIL",
        "boundary": (
            "Technical latency only. The benchmarked ranker failed its value "
            "quality gate and is not approved for deployment."
        ),
        "repeats": args.repeats,
        "candidate_counts": {
            "raw": int(sample["raw_count"]),
            "feasible": int(sample["feasible_count"]),
            "shortlist": len(sample["shortlist"]),
        },
        "candidate_generation_ms": {
            "mean": statistics.fmean(generation_ms),
            "p50": percentile(generation_ms, 0.50),
            "p95": percentile(generation_ms, 0.95),
            "max": max(generation_ms),
        },
        "warm_ranker_64_ms": {
            "mean": statistics.fmean(inference_ms),
            "p50": percentile(inference_ms, 0.50),
            "p95": percentile(inference_ms, 0.95),
            "max": max(inference_ms),
        },
        "combined_p95_ms": combined_p95,
        "one_time_model_load_seconds": model_load_seconds,
        "inputs": {
            "model": {"path": str(args.model), "sha256": sha256(args.model)},
            "dataset": {
                "path": str(args.dataset), "sha256": sha256(args.dataset)
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
