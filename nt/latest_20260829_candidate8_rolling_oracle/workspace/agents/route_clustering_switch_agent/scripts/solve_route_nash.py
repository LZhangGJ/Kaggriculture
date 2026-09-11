#!/usr/bin/env python3
"""Solve the searched route payoff matrix and export opening/response supports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payoffs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-weight", type=float, default=1e-5)
    args = parser.parse_args()

    if args.payoffs.suffix == ".npz":
        with np.load(args.payoffs) as saved:
            matrix = np.asarray(saved["score"], dtype=np.float64)
            families = saved["families"].astype(str).tolist()
        opponent_families = families
    else:
        source = json.loads(args.payoffs.read_text(encoding="utf-8"))
        matrix = np.asarray(source["score_matrix"], dtype=np.float64)
        families = [str(value) for value in source["families"]]
        opponent_families = [
            str(value) for value in source.get("opponent_families", families)
        ]
    if matrix.shape != (len(families), len(opponent_families)):
        raise ValueError("payoff matrix labels do not match its rectangular shape")
    # Native round-robin intentionally skips self-play, leaving the diagonal at
    # zero.  In a symmetric zero-sum game self-play is a draw, so restore 0.5
    # before solving the maximin LP.
    if families == opponent_families and matrix.shape[0] == matrix.shape[1]:
        np.fill_diagonal(matrix, 0.5)
    row_count = len(families)
    column_count = len(opponent_families)
    objective = np.r_[np.zeros(row_count), -1.0]
    result = linprog(
        objective,
        A_ub=np.c_[-matrix.T, np.ones(column_count)],
        b_ub=np.zeros(column_count),
        A_eq=np.r_[np.ones(row_count), 0.0][None, :],
        b_eq=[1.0],
        bounds=[(0.0, None)] * row_count + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    probabilities = result.x[:row_count]
    best = np.argmax(matrix, axis=0)
    response_counts = Counter(int(value) for value in best.tolist())
    payload = {
        "schema_version": 1,
        "payoffs": str(args.payoffs),
        "value": float(result.x[-1]),
        "opening_support": [
            {
                "family": families[index],
                "weight": float(probabilities[index]),
                "mean_score": float(np.mean(matrix[index])),
                "worst_score": float(np.min(matrix[index])),
            }
            for index in np.flatnonzero(probabilities > args.min_weight)
        ],
        "best_response_support": [
            {
                "family": families[index],
                "opponent_families": count,
            }
            for index, count in response_counts.most_common()
        ],
        "conditional_oracle": {
            "mean_score": float(np.mean(matrix[best, np.arange(column_count)])),
            "worst_score": float(np.min(matrix[best, np.arange(column_count)])),
        },
        "opponent_families": opponent_families,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
