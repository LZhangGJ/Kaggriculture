#!/usr/bin/env python3
"""Calibrate Candidate8 soft quotas from multi-future counterfactual receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FAMILIES = [
    "KEEP",
    "SCHEDULE_LAYOUT",
    "CONTINUOUS_SCALE",
    "UNILATERAL",
    "MULTI_PROJECT",
    "TIMING",
    "MARKET_TRANSACTION",
    "PHASE_SUFFIX",
    "LOCAL_RECOVERY",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_states(paths: list[Path]) -> list[dict[str, np.ndarray]]:
    states = []
    for source_index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            for state_id in np.unique(data["state_id"]):
                rows = np.flatnonzero(data["state_id"] == state_id)
                states.append(
                    {
                        "source": np.asarray([source_index]),
                        "family": np.asarray(data["family"][rows], dtype=np.int8),
                        "estimated": np.asarray(
                            data["features"][rows, 17], dtype=np.float64
                        ),
                        "signature": np.asarray(
                            data["signature"][rows], dtype=np.uint64
                        ),
                        "reward": np.asarray(
                            data["expected_reward"][rows], dtype=np.float64
                        ),
                    }
                )
    return states


def selected_indices(state: dict[str, np.ndarray], quotas: list[int]) -> np.ndarray:
    family = state["family"]
    estimated = state["estimated"]
    signature = state["signature"]
    selected: set[int] = set()
    for family_id, quota in enumerate(quotas):
        members = np.flatnonzero(family == family_id)
        order = np.lexsort((signature[members], -estimated[members]))
        selected.update(int(index) for index in members[order[:quota]])
    counts = np.bincount(
        family[np.fromiter(selected, dtype=np.int64)], minlength=len(FAMILIES)
    )
    remaining = np.asarray(
        [index for index in range(len(family)) if index not in selected],
        dtype=np.int64,
    )
    if len(remaining):
        diversity_score = estimated[remaining] / (
            1.0 + 0.20 * counts[family[remaining]]
        )
        order = np.lexsort((signature[remaining], -diversity_score))
        for index in remaining[order]:
            if len(selected) >= 64:
                break
            selected.add(int(index))
    return np.asarray(sorted(selected), dtype=np.int64)


def metrics(states: list[dict[str, np.ndarray]], quotas: list[int]) -> dict:
    regrets = []
    missed_family = np.zeros(len(FAMILIES), dtype=np.int64)
    for state in states:
        reward = state["reward"]
        oracle = int(np.argmax(reward))
        selected = selected_indices(state, quotas)
        regret = float(reward[oracle] - np.max(reward[selected]))
        regrets.append(regret)
        if regret > 1e-9:
            missed_family[int(state["family"][oracle])] += 1
    values = np.asarray(regrets, dtype=np.float64)
    return {
        "states": len(states),
        "mean_regret": float(values.mean()),
        "p90_regret": float(np.quantile(values, 0.90)),
        "max_regret": float(values.max()),
        "exact_oracle_recall": float(np.mean(values <= 1e-9)),
        "near_oracle_within_500": float(np.mean(values <= 500.0)),
        "near_oracle_within_1000": float(np.mean(values <= 1_000.0)),
        "missed_oracle_by_family": {
            family: int(count)
            for family, count in zip(FAMILIES, missed_family, strict=True)
        },
    }


def objective(result: dict) -> float:
    # Stable value loss is primary. P90 and near-oracle coverage prevent the
    # average from hiding a small number of catastrophic omissions.
    return (
        result["mean_regret"]
        + 0.35 * result["p90_regret"]
        + 500.0 * (1.0 - result["near_oracle_within_1000"])
        + 100.0 * (1.0 - result["exact_oracle_recall"])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, action="append", required=True)
    parser.add_argument("--holdout", type=Path, action="append", default=[])
    parser.add_argument(
        "--initial", default="1,8,14,6,18,5,6,3,3",
        help="Nine comma-separated quotas summing to 64",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    initial = [int(value) for value in args.initial.split(",")]
    if len(initial) != len(FAMILIES) or sum(initial) != 64 or initial[0] != 1:
        raise ValueError("initial quotas must have nine values, sum to 64, KEEP=1")
    calibration = load_states(args.calibration)
    holdout = load_states(args.holdout) if args.holdout else []

    quotas = initial.copy()
    path = [{"quotas": quotas.copy(), "metrics": metrics(calibration, quotas)}]
    while True:
        current_score = objective(path[-1]["metrics"])
        best_score = current_score
        best_quotas = None
        best_metrics = None
        for donor in range(1, len(FAMILIES)):
            if quotas[donor] <= 2:
                continue
            for receiver in range(1, len(FAMILIES)):
                if receiver == donor:
                    continue
                proposal = quotas.copy()
                proposal[donor] -= 1
                proposal[receiver] += 1
                result = metrics(calibration, proposal)
                score = objective(result)
                if score < best_score - 1e-9:
                    best_score = score
                    best_quotas = proposal
                    best_metrics = result
        if best_quotas is None:
            break
        quotas = best_quotas
        path.append({"quotas": quotas.copy(), "metrics": best_metrics})

    payload = {
        "schema": "kaggriculture.candidate8_soft_quota_calibration.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "method": (
            "Deterministic one-slot coordinate descent over stable multi-future "
            "counterfactual value; no Replay frequency is used as a quota."
        ),
        "initial_quotas": dict(zip(FAMILIES, initial, strict=True)),
        "recommended_quotas": dict(zip(FAMILIES, quotas, strict=True)),
        "calibration_before": metrics(calibration, initial),
        "calibration_after": metrics(calibration, quotas),
        "holdout_before": metrics(holdout, initial) if holdout else None,
        "holdout_after": metrics(holdout, quotas) if holdout else None,
        "search_path": path,
        "inputs": {
            "calibration": [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.calibration
            ],
            "holdout": [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.holdout
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
