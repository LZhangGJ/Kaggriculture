#!/usr/bin/env python3
"""Select a seat-aware Day0 opening by expectation over training opponents."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_HOLDOUT_OPPONENTS = "G001,G003,G049,G245"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def parse_strings(raw: str) -> set[str]:
    return {value.strip() for value in raw.split(",") if value.strip()}


def lexicographic_best(
    score: np.ndarray, margin: np.ndarray, cash: np.ndarray,
    signature: np.ndarray,
) -> int:
    return max(
        range(len(score)),
        key=lambda index: (
            float(score[index]), float(margin[index]), float(cash[index]),
            -int(signature[index]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--holdout-opponents", type=parse_strings,
        default=parse_strings(DEFAULT_HOLDOUT_OPPONENTS),
    )
    parser.add_argument("--holdout-seed-count", type=int, default=4)
    parser.add_argument("--q25-margin-floor", type=float, default=0.0)
    parser.add_argument("--minimum-opponent-margin-floor", type=float, default=-1000.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"])
        opponent = np.asarray(data["opponent"])
        prefix_seed = np.asarray(data["prefix_seed"])
        seat = np.asarray(data["seat"])
        day = np.asarray(data["decision_day"])
        signature = np.asarray(data["signature"])
        score = np.asarray(data["expected_score_rate"])
        margin = np.asarray(data["expected_margin"])
        cash = np.asarray(data["expected_own_cash"])
    if np.any(day != 0):
        raise ValueError("Day0 selector requires a Day0-only dataset")

    unique_seeds = np.unique(prefix_seed)
    if args.holdout_seed_count <= 0 or args.holdout_seed_count >= len(unique_seeds):
        raise ValueError("holdout-seed-count must leave training seeds")
    test_seeds = {int(value) for value in unique_seeds[-args.holdout_seed_count:]}
    train_seeds = {int(value) for value in unique_seeds[:-args.holdout_seed_count]}
    all_opponents = {str(value) for value in np.unique(opponent)}
    test_opponents = all_opponents & args.holdout_opponents
    train_opponents = all_opponents - test_opponents
    if not test_opponents or not train_opponents:
        raise ValueError("opponent split must leave non-empty train and test sets")

    selected_by_seat: dict[str, dict] = {}
    for candidate_seat in (0, 1):
        fit = (
            (seat == candidate_seat)
            & np.isin(prefix_seed, list(train_seeds))
            & np.isin(opponent, list(train_opponents))
        )
        signatures = np.unique(signature[fit])
        aggregate = []
        for value in signatures:
            rows = fit & (signature == value)
            row_indices = np.flatnonzero(rows)
            keep_indices = np.asarray([
                np.flatnonzero(state_id == state_id[index])[0]
                for index in row_indices
            ], dtype=np.int64)
            score_gain = score[row_indices] - score[keep_indices]
            margin_gain = margin[row_indices] - margin[keep_indices]
            cash_gain = cash[row_indices] - cash[keep_indices]
            by_opponent = []
            for route in sorted(train_opponents):
                local = opponent[row_indices] == route
                by_opponent.append((
                    float(score_gain[local].mean()),
                    float(margin_gain[local].mean()),
                    float(cash_gain[local].mean()),
                ))
            by_opponent = np.asarray(by_opponent, dtype=np.float64)
            aggregate.append({
                "signature": int(value),
                "score": float(score[rows].mean()),
                "margin": float(margin[rows].mean()),
                "cash": float(cash[rows].mean()),
                "mean_score_gain_vs_keep": float(score_gain.mean()),
                "mean_margin_gain_vs_keep": float(margin_gain.mean()),
                "mean_cash_gain_vs_keep": float(cash_gain.mean()),
                "q25_opponent_score_gain": float(np.quantile(by_opponent[:, 0], 0.25)),
                "q25_opponent_margin_gain": float(np.quantile(by_opponent[:, 1], 0.25)),
                "minimum_opponent_margin_gain": float(by_opponent[:, 1].min()),
            })
        eligible = [
            row for row in aggregate
            if row["q25_opponent_margin_gain"] >= args.q25_margin_floor
            and row["minimum_opponent_margin_gain"]
            >= args.minimum_opponent_margin_floor
        ]
        chosen = max(
            eligible or aggregate,
            key=lambda row: (
                row["mean_score_gain_vs_keep"],
                row["q25_opponent_score_gain"],
                row["mean_margin_gain_vs_keep"],
                row["mean_cash_gain_vs_keep"],
                -row["signature"],
            ),
        )

        # Candidate order is deterministic at Day0.  Verify that the chosen
        # signature maps to one stable shortlist rank in every public state.
        ranks = []
        for state in np.unique(state_id[seat == candidate_seat]):
            indices = np.flatnonzero(state_id == state)
            matches = np.flatnonzero(signature[indices] == chosen["signature"])
            if len(matches) != 1:
                raise RuntimeError(
                    f"signature {chosen['signature']} missing/duplicated in state {state}"
                )
            ranks.append(int(matches[0]))
        if len(set(ranks)) != 1:
            raise RuntimeError(f"Day0 signature rank is not stable: {sorted(set(ranks))}")
        chosen["rank"] = ranks[0]
        selected_by_seat[str(candidate_seat)] = chosen

    def evaluate(opponents: set[str], seeds: set[int]) -> dict:
        rows = []
        for state in np.unique(state_id):
            indices = np.flatnonzero(state_id == state)
            route = str(opponent[indices[0]])
            seed_value = int(prefix_seed[indices[0]])
            if route not in opponents or seed_value not in seeds:
                continue
            candidate_seat = int(seat[indices[0]])
            chosen_signature = selected_by_seat[str(candidate_seat)]["signature"]
            selected = int(np.flatnonzero(
                signature[indices] == chosen_signature
            )[0])
            keep = 0
            oracle = lexicographic_best(
                score[indices], margin[indices], cash[indices], signature[indices]
            )
            rows.append({
                "state_id": int(state),
                "opponent": route,
                "prefix_seed": seed_value,
                "seat": candidate_seat,
                "selected_score": float(score[indices[selected]]),
                "keep_score": float(score[indices[keep]]),
                "oracle_score": float(score[indices[oracle]]),
                "selected_margin": float(margin[indices[selected]]),
                "keep_margin": float(margin[indices[keep]]),
                "oracle_margin": float(margin[indices[oracle]]),
                "selected_cash": float(cash[indices[selected]]),
                "keep_cash": float(cash[indices[keep]]),
                "oracle_cash": float(cash[indices[oracle]]),
                "oracle_exact": selected == oracle,
            })
        return {
            "states": len(rows),
            "mean_score_gain_vs_keep": float(np.mean([
                row["selected_score"] - row["keep_score"] for row in rows
            ])),
            "mean_margin_gain_vs_keep": float(np.mean([
                row["selected_margin"] - row["keep_margin"] for row in rows
            ])),
            "mean_cash_gain_vs_keep": float(np.mean([
                row["selected_cash"] - row["keep_cash"] for row in rows
            ])),
            "mean_oracle_margin_headroom": float(np.mean([
                row["oracle_margin"] - row["selected_margin"] for row in rows
            ])),
            "exact_opponent_specific_oracle_recall": float(np.mean([
                row["oracle_exact"] for row in rows
            ])),
            "rows": rows,
        }

    partitions = {
        "fit": evaluate(train_opponents, train_seeds),
        "test_unseen_seeds": evaluate(train_opponents, test_seeds),
        "test_unseen_opponents": evaluate(test_opponents, train_seeds),
        "test_joint_unseen": evaluate(test_opponents, test_seeds),
    }
    test = partitions["test_joint_unseen"]
    payload = {
        "schema": "kaggriculture.candidate8-day0-robust-opening.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "The Day0 action is selected by expectation over training opponents; "
            "opponent identity and seed are not model inputs."
        ),
        "selection_objective": {
            "description": (
                "Distributionally robust opening: require nonnegative lower-"
                "quartile opponent margin gain and cap the worst opponent "
                "margin loss, then maximize mean score gain."
            ),
            "q25_margin_floor": args.q25_margin_floor,
            "minimum_opponent_margin_floor": args.minimum_opponent_margin_floor,
        },
        "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
        "split": {
            "train_opponents": sorted(train_opponents),
            "test_opponents": sorted(test_opponents),
            "train_seeds": sorted(train_seeds),
            "test_seeds": sorted(test_seeds),
        },
        "selected_by_seat": selected_by_seat,
        "metrics": partitions,
        "test": test,
        "gate": {
            "score_gain_nonnegative": test["mean_score_gain_vs_keep"] >= 0,
            "margin_gain_positive": test["mean_margin_gain_vs_keep"] > 0,
            "cash_gain_nonnegative": test["mean_cash_gain_vs_keep"] >= 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_by_seat": selected_by_seat,
        "metrics": {
            name: {key: value for key, value in metric.items() if key != "rows"}
            for name, metric in partitions.items()
        },
        "gate": payload["gate"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
