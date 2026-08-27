#!/usr/bin/env python3
"""Combine one-switch counter matrices and measure schedule-oracle coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrices", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-portfolio", type=int, default=32)
    args = parser.parse_args()
    if len(args.matrices) != len(args.checkpoints):
        raise ValueError("--matrices and --checkpoints must have equal length")

    schedule_names: list[str] = []
    margin_blocks = []
    score_blocks = []
    expected_seeds: np.ndarray | None = None
    for path, checkpoint in zip(args.matrices, args.checkpoints):
        saved = np.load(path, allow_pickle=False)
        openings = saved["openings"].astype(str)
        targets = saved["targets"].astype(str)
        seeds = saved["seeds"].astype(np.int64)
        if expected_seeds is None:
            expected_seeds = seeds
        elif not np.array_equal(expected_seeds, seeds):
            raise ValueError(f"seed mismatch: {path}")
        margins = saved["margins"].astype(np.float64)
        scores = saved["scores"].astype(np.float64)
        for opening_index, opening in enumerate(openings):
            for target_index, target in enumerate(targets):
                schedule_names.append(f"{opening}@{checkpoint}->{target}")
                margin_blocks.append(margins[opening_index, target_index])
                score_blocks.append(scores[opening_index, target_index])

    assert expected_seeds is not None
    margin_matrix = np.asarray(margin_blocks)
    score_matrix = np.asarray(score_blocks)
    win_matrix = margin_matrix > 0
    scenario_count = margin_matrix.shape[1] * margin_matrix.shape[2]

    order = sorted(range(len(schedule_names)), key=lambda index: (
        -float(np.mean(win_matrix[index])),
        -float(np.mean(score_matrix[index])),
        -float(np.mean(margin_matrix[index])),
        schedule_names[index],
    ))
    fixed = [
        {
            "rank": rank,
            "schedule": schedule_names[index],
            "raw_win_rate": float(np.mean(win_matrix[index])),
            "score_rate": float(np.mean(score_matrix[index])),
            "mean_margin": float(np.mean(margin_matrix[index])),
        }
        for rank, index in enumerate(order[:32], 1)
    ]

    best_margin = np.max(margin_matrix, axis=0)
    oracle_win = best_margin > 0
    best_index = np.argmax(margin_matrix, axis=0)
    counts = {
        schedule_names[index]: int(np.count_nonzero(best_index == index))
        for index in np.unique(best_index)
    }

    selected: list[int] = []
    covered = np.zeros_like(oracle_win)
    current_best = np.full_like(best_margin, -np.inf)
    remaining = set(range(len(schedule_names)))
    trace = []
    for portfolio_rank in range(1, args.max_portfolio + 1):
        ranked = []
        for index in remaining:
            candidate_covered = covered | win_matrix[index]
            candidate_margin = np.maximum(current_best, margin_matrix[index])
            ranked.append((
                int(np.count_nonzero(candidate_covered)),
                float(np.mean(candidate_margin)),
                index,
            ))
        coverage_count, mean_margin, chosen = max(ranked)
        previous = int(np.count_nonzero(covered))
        if coverage_count == previous:
            break
        selected.append(chosen)
        remaining.remove(chosen)
        covered |= win_matrix[chosen]
        current_best = np.maximum(current_best, margin_matrix[chosen])
        trace.append({
            "portfolio_rank": portfolio_rank,
            "schedule": schedule_names[chosen],
            "new_wins": coverage_count - previous,
            "covered_scenarios": coverage_count,
            "raw_win_oracle_coverage": coverage_count / scenario_count,
            "portfolio_oracle_mean_margin": mean_margin,
        })

    payload = {
        "schema": "schedule-counter-oracle-analysis-v1",
        "matrices": [str(path.resolve()) for path in args.matrices],
        "schedule_count": len(schedule_names),
        "seed_count": len(expected_seeds),
        "scenario_count": scenario_count,
        "best_fixed_schedules": fixed,
        "oracle": {
            "raw_win_rate": float(np.mean(oracle_win)),
            "score_rate": float(np.mean(np.max(score_matrix, axis=0))),
            "mean_best_margin": float(np.mean(best_margin)),
            "minimum_best_margin": float(np.min(best_margin)),
            "both_seats_win_rate": float(np.mean(np.all(oracle_win, axis=1))),
            "no_winning_schedule_scenarios": int(np.count_nonzero(~oracle_win)),
            "best_schedule_counts": dict(sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )),
        },
        "greedy_win_coverage_portfolio": trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "schedule_count": payload["schedule_count"],
        "best_fixed_schedules": fixed[:10],
        "oracle": payload["oracle"],
        "portfolio": trace,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
