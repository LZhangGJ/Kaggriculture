#!/usr/bin/env python3
"""Measure fixed-route and route-oracle coverage against a switch policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-portfolio", type=int, default=32)
    args = parser.parse_args()

    saved = np.load(args.matrix, allow_pickle=False)
    families = saved["families"].astype(str)
    seeds = saved["seeds"].astype(np.int64)
    margins = saved["margins"].astype(np.float64)
    scores = saved["scores"].astype(np.float64)
    wins = margins > 0
    scenarios = margins.shape[1] * margins.shape[2]

    fixed_rows = []
    for index, family in enumerate(families):
        paired_win = wins[index].mean(axis=1)
        paired_score = scores[index].mean(axis=1)
        paired_margin = margins[index].mean(axis=1)
        fixed_rows.append({
            "family": str(family),
            "raw_win_rate": float(np.mean(wins[index])),
            "score_rate": float(np.mean(scores[index])),
            "mean_margin": float(np.mean(margins[index])),
            "both_seats_win_rate": float(np.mean(paired_win == 1.0)),
            "paired_seed_score": _summary(paired_score),
            "paired_seed_margin": _summary(paired_margin),
        })
    fixed_rows.sort(key=lambda row: (
        -row["raw_win_rate"], -row["score_rate"], -row["mean_margin"], row["family"]
    ))
    for rank, row in enumerate(fixed_rows, 1):
        row["rank"] = rank

    best_indices = np.argmax(margins, axis=0)
    best_margins = np.max(margins, axis=0)
    oracle_wins = best_margins > 0
    oracle_scores = np.max(scores, axis=0)
    oracle_family_counts = {
        str(families[index]): int(np.count_nonzero(best_indices == index))
        for index in np.unique(best_indices)
    }

    selected: list[int] = []
    covered = np.zeros((len(seeds), 2), dtype=bool)
    trace = []
    remaining = set(range(len(families)))
    for portfolio_rank in range(1, args.max_portfolio + 1):
        ranked = []
        for index in remaining:
            candidate_covered = covered | wins[index]
            ranked.append((
                int(np.count_nonzero(candidate_covered)),
                float(np.mean(np.maximum(
                    np.max(margins[selected], axis=0) if selected else -np.inf,
                    margins[index],
                ))),
                index,
            ))
        coverage_count, portfolio_margin, chosen = max(ranked)
        previous_count = int(np.count_nonzero(covered))
        if coverage_count == previous_count:
            break
        selected.append(chosen)
        remaining.remove(chosen)
        covered |= wins[chosen]
        trace.append({
            "portfolio_rank": portfolio_rank,
            "family": str(families[chosen]),
            "new_wins": coverage_count - previous_count,
            "covered_scenarios": coverage_count,
            "raw_win_oracle_coverage": coverage_count / scenarios,
            "portfolio_oracle_mean_margin": portfolio_margin,
        })

    no_winner = ~np.any(wins, axis=0)
    payload = {
        "schema": "policy-counter-oracle-analysis-v1",
        "matrix": str(args.matrix.resolve()),
        "routes": int(len(families)),
        "seeds": int(len(seeds)),
        "scenarios": int(scenarios),
        "best_fixed": fixed_rows[:32],
        "oracle": {
            "raw_win_rate": float(np.mean(oracle_wins)),
            "score_rate": float(np.mean(oracle_scores)),
            "mean_best_margin": float(np.mean(best_margins)),
            "both_seats_win_rate": float(np.mean(np.all(oracle_wins, axis=1))),
            "no_winning_route_scenarios": int(np.count_nonzero(no_winner)),
            "no_winning_route_seed_count": int(np.count_nonzero(np.any(no_winner, axis=1))),
            "best_margin": _summary(best_margins),
            "best_family_counts": dict(sorted(
                oracle_family_counts.items(), key=lambda item: (-item[1], item[0])
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
        "routes": payload["routes"],
        "seeds": payload["seeds"],
        "scenarios": payload["scenarios"],
        "best_fixed": payload["best_fixed"][:10],
        "oracle": payload["oracle"],
        "portfolio": trace,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
