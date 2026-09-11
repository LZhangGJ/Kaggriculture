#!/usr/bin/env python3
"""Audit whether Candidate8 terminal labels are stable across future banks."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCORE_WEIGHT = 10_000_000.0
MARGIN_WEIGHT = 1.0
CASH_WEIGHT = 0.001


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def utility(own: np.ndarray, rival: np.ndarray) -> np.ndarray:
    score = np.where(own > rival, 1.0, np.where(own == rival, 0.5, 0.0)).mean(axis=1)
    own_mean = own.mean(axis=1)
    margin = own_mean - rival.mean(axis=1)
    return SCORE_WEIGHT * score + MARGIN_WEIGHT * margin + CASH_WEIGHT * own_mean


def objective_values(own: np.ndarray, rival: np.ndarray) -> dict[str, np.ndarray]:
    score = np.where(own > rival, 1.0, np.where(own == rival, 0.5, 0.0)).mean(axis=1)
    own_mean = own.mean(axis=1)
    margin = own_mean - rival.mean(axis=1)
    return {
        "score_rate": score,
        "margin": margin,
        "own_cash": own_mean,
        "lexicographic_utility": (
            SCORE_WEIGHT * score + MARGIN_WEIGHT * margin + CASH_WEIGHT * own_mean
        ),
    }


def audit_count(
    state_id: np.ndarray, own: np.ndarray, rival: np.ndarray, count: int,
) -> dict:
    if count % 2 or count > own.shape[1]:
        raise ValueError(f"future count {count} must be even and <= {own.shape[1]}")
    half = count // 2
    correlations: list[float] = []
    exact: list[bool] = []
    recall8: list[bool] = []
    top8_jaccard: list[float] = []
    oracle_gain_sign_agreement: list[bool] = []
    cross_negative: list[bool] = []
    cross_improves_keep: list[bool] = []
    robust_candidate_exists: list[bool] = []
    objective_rows: dict[str, dict[str, list[float | bool]]] = {
        name: {
            "correlation": [],
            "recall8": [],
            "cross_negative": [],
            "cross_improves": [],
        }
        for name in ("score_rate", "margin", "own_cash", "lexicographic_utility")
    }

    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        first = utility(own[indices, :half], rival[indices, :half])
        second = utility(own[indices, half:count], rival[indices, half:count])
        if np.std(first) == 0 or np.std(second) == 0:
            correlation = 1.0 if np.array_equal(first, second) else 0.0
        else:
            correlation = float(np.corrcoef(first, second)[0, 1])
        correlations.append(correlation)

        first_order = np.argsort(-first, kind="stable")
        second_order = np.argsort(-second, kind="stable")
        first_best = int(first_order[0])
        second_best = int(second_order[0])
        first_top8 = set(int(value) for value in first_order[:8])
        second_top8 = set(int(value) for value in second_order[:8])
        exact.append(first_best == second_best)
        recall8.append(first_best in second_top8)
        top8_jaccard.append(
            len(first_top8 & second_top8) / max(1, len(first_top8 | second_top8))
        )

        # Row zero is KEEP by the generator contract.
        first_gain = float(first[first_best] - first[0])
        second_gain = float(second[second_best] - second[0])
        oracle_gain_sign_agreement.append((first_gain > 0) == (second_gain > 0))

        cross_gain = float(second[first_best] - second[0])
        cross_negative.append(cross_gain < 0)
        cross_improves_keep.append(cross_gain > 0)
        robust_candidate_exists.append(bool(np.any(
            (first > first[0]) & (second > second[0])
        )))

        first_objectives = objective_values(
            own[indices, :half], rival[indices, :half]
        )
        second_objectives = objective_values(
            own[indices, half:count], rival[indices, half:count]
        )
        for name in objective_rows:
            first_values = first_objectives[name]
            second_values = second_objectives[name]
            if np.std(first_values) == 0 or np.std(second_values) == 0:
                objective_correlation = (
                    1.0 if np.array_equal(first_values, second_values) else 0.0
                )
            else:
                objective_correlation = float(
                    np.corrcoef(first_values, second_values)[0, 1]
                )
            objective_first_order = np.argsort(-first_values, kind="stable")
            objective_second_order = np.argsort(-second_values, kind="stable")
            objective_best = int(objective_first_order[0])
            objective_cross_gain = float(
                second_values[objective_best] - second_values[0]
            )
            objective_rows[name]["correlation"].append(objective_correlation)
            objective_rows[name]["recall8"].append(
                objective_best in set(int(value) for value in objective_second_order[:8])
            )
            objective_rows[name]["cross_negative"].append(objective_cross_gain < 0)
            objective_rows[name]["cross_improves"].append(objective_cross_gain > 0)

    metrics = {
        "future_count": count,
        "half_count": half,
        "states": len(correlations),
        "mean_candidate_utility_correlation": float(np.mean(correlations)),
        "half_bank_exact_oracle_agreement": float(np.mean(exact)),
        "half_bank_oracle_recall_at_8": float(np.mean(recall8)),
        "mean_top8_jaccard": float(np.mean(top8_jaccard)),
        "oracle_gain_sign_agreement": float(np.mean(oracle_gain_sign_agreement)),
        "cross_selected_negative_rate": float(np.mean(cross_negative)),
        "cross_selected_improves_keep_rate": float(np.mean(cross_improves_keep)),
        "robust_candidate_exists_rate": float(np.mean(robust_candidate_exists)),
        "by_objective": {
            name: {
                "mean_candidate_correlation": float(np.mean(rows["correlation"])),
                "half_bank_oracle_recall_at_8": float(np.mean(rows["recall8"])),
                "cross_selected_negative_rate": float(np.mean(rows["cross_negative"])),
                "cross_selected_improves_keep_rate": float(np.mean(rows["cross_improves"])),
            }
            for name, rows in objective_rows.items()
        },
    }
    if count == own.shape[1]:
        metrics["gate"] = {
            "utility_correlation_ge_0_50": metrics["mean_candidate_utility_correlation"] >= 0.50,
            "oracle_recall_at_8_ge_0_50": metrics["half_bank_oracle_recall_at_8"] >= 0.50,
            "gain_sign_agreement_ge_0_80": metrics["oracle_gain_sign_agreement"] >= 0.80,
            "cross_negative_rate_le_0_20": metrics["cross_selected_negative_rate"] <= 0.20,
        }
        metrics["passed"] = all(metrics["gate"].values())
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--counts", default="8,16,32")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    counts = [int(value.strip()) for value in args.counts.split(",") if value.strip()]
    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"])
        own = np.asarray(data["future_own_cash"], dtype=np.float64)
        rival = np.asarray(data["future_opponent_cash"], dtype=np.float64)

    results = [audit_count(state_id, own, rival, count) for count in counts]
    payload = {
        "schema": "kaggriculture.candidate8-future-stability.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "The gate was frozen before reading this probe. It tests whether "
            "two disjoint future halves support the same public-state decision."
        ),
        "dataset": {
            "path": str(args.dataset),
            "sha256": sha256(args.dataset),
            "rows": int(len(state_id)),
            "states": int(len(np.unique(state_id))),
            "available_future_count": int(own.shape[1]),
        },
        "results": results,
        "final_gate_passed": bool(results[-1].get("passed", False)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
