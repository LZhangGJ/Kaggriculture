#!/usr/bin/env python3
"""Audit a frozen paired lower-confidence-bound Candidate8 target."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def mean_lcb(values: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=1)
    if values.shape[1] <= 1:
        return mean, mean
    standard_error = values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])
    return mean, mean - z * standard_error


def select(
    own: np.ndarray, rival: np.ndarray, z: float,
) -> tuple[int, dict[str, float]]:
    keep_own = own[0:1]
    keep_rival = rival[0:1]
    candidate_score = np.where(own > rival, 1.0, np.where(own == rival, 0.5, 0.0))
    keep_score = np.where(
        keep_own > keep_rival, 1.0, np.where(keep_own == keep_rival, 0.5, 0.0)
    )
    score_delta = candidate_score - keep_score
    margin_delta = (own - rival) - (keep_own - keep_rival)
    cash_delta = own - keep_own
    score_mean, score_lcb = mean_lcb(score_delta, z)
    margin_mean, margin_lcb = mean_lcb(margin_delta, z)
    cash_mean, cash_lcb = mean_lcb(cash_delta, z)

    eligible = np.flatnonzero((score_lcb >= 0.0) & (margin_lcb > 0.0))
    if not len(eligible):
        return 0, {
            "score_mean": 0.0,
            "score_lcb": 0.0,
            "margin_mean": 0.0,
            "margin_lcb": 0.0,
            "cash_mean": 0.0,
            "cash_lcb": 0.0,
        }
    chosen = int(max(
        eligible,
        key=lambda index: (
            float(score_lcb[index]),
            float(margin_lcb[index]),
            float(cash_lcb[index]),
            -int(index),
        ),
    ))
    return chosen, {
        "score_mean": float(score_mean[chosen]),
        "score_lcb": float(score_lcb[chosen]),
        "margin_mean": float(margin_mean[chosen]),
        "margin_lcb": float(margin_lcb[chosen]),
        "cash_mean": float(cash_mean[chosen]),
        "cash_lcb": float(cash_lcb[chosen]),
    }


def evaluate_selected(
    own: np.ndarray, rival: np.ndarray, selected: int,
) -> dict[str, float]:
    candidate_score = np.where(
        own[selected] > rival[selected],
        1.0,
        np.where(own[selected] == rival[selected], 0.5, 0.0),
    )
    keep_score = np.where(
        own[0] > rival[0], 1.0, np.where(own[0] == rival[0], 0.5, 0.0)
    )
    score_delta = candidate_score - keep_score
    margin_delta = (own[selected] - rival[selected]) - (own[0] - rival[0])
    cash_delta = own[selected] - own[0]
    return {
        "score_delta": float(score_delta.mean()),
        "margin_delta": float(margin_delta.mean()),
        "cash_delta": float(cash_delta.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--z", type=float, default=1.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"])
        own = np.asarray(data["future_own_cash"], dtype=np.float64)
        rival = np.asarray(data["future_opponent_cash"], dtype=np.float64)
    if own.shape[1] < 4 or own.shape[1] % 2:
        raise ValueError("future bank must contain an even count >= 4")

    half = own.shape[1] // 2
    rows: list[dict] = []
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        state_own = own[indices]
        state_rival = rival[indices]
        for direction, train_slice, test_slice in (
            ("A_to_B", slice(0, half), slice(half, own.shape[1])),
            ("B_to_A", slice(half, own.shape[1]), slice(0, half)),
        ):
            chosen, training = select(
                state_own[:, train_slice], state_rival[:, train_slice], args.z
            )
            test = evaluate_selected(
                state_own[:, test_slice], state_rival[:, test_slice], chosen
            )
            rows.append({
                "state_id": int(state),
                "direction": direction,
                "selected_local_index": chosen,
                "activated": chosen != 0,
                "training": training,
                "test": test,
            })

    active = [row for row in rows if row["activated"]]
    def mean(rows_: list[dict], key: str) -> float:
        return float(np.mean([row["test"][key] for row in rows_])) if rows_ else 0.0

    metrics = {
        "states": int(len(np.unique(state_id))),
        "directions": len(rows),
        "future_count": int(own.shape[1]),
        "half_count": int(half),
        "z": args.z,
        "activation_rate": len(active) / len(rows),
        "mean_score_delta_all": mean(rows, "score_delta"),
        "mean_margin_delta_all": mean(rows, "margin_delta"),
        "mean_cash_delta_all": mean(rows, "cash_delta"),
        "mean_score_delta_active": mean(active, "score_delta"),
        "mean_margin_delta_active": mean(active, "margin_delta"),
        "mean_cash_delta_active": mean(active, "cash_delta"),
        "active_score_regression_rate": float(np.mean([
            row["test"]["score_delta"] < 0 for row in active
        ])) if active else 0.0,
        "active_margin_negative_rate": float(np.mean([
            row["test"]["margin_delta"] < 0 for row in active
        ])) if active else 0.0,
    }
    metrics["gate"] = {
        "activation_rate_ge_0_05": metrics["activation_rate"] >= 0.05,
        "score_regression_rate_le_0_10": metrics["active_score_regression_rate"] <= 0.10,
        "margin_negative_rate_le_0_20": metrics["active_margin_negative_rate"] <= 0.20,
        "mean_score_delta_nonnegative": metrics["mean_score_delta_active"] >= 0.0,
        "mean_margin_delta_positive": metrics["mean_margin_delta_active"] > 0.0,
    }
    metrics["passed"] = all(metrics["gate"].values())

    payload = {
        "schema": "kaggriculture.candidate8-paired-lcb-stability.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "Frozen paired target: candidate score-delta LCB must be nonnegative "
            "and margin-delta LCB positive; rank by score, margin, then cash LCB."
        ),
        "dataset": {
            "path": str(args.dataset),
            "sha256": sha256(args.dataset),
        },
        "metrics": metrics,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
