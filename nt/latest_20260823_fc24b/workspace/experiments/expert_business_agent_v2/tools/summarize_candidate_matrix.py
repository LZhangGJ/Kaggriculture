#!/usr/bin/env python3
"""Summarize an official candidate-matrix receipt without rerunning games."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(rows)
    wins = sum(bool(row.get("win")) for row in rows)
    ties = sum(bool(row.get("tie")) for row in rows)
    margins = [float(row.get("margin", 0.0)) for row in rows]
    return {
        "games": games,
        "wins": wins,
        "ties": ties,
        "win_rate": wins / games if games else 0.0,
        "mean_margin": sum(margins) / games if games else 0.0,
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.matrix.read_text(encoding="utf-8"))
    rows = list(payload.get("rows", []))

    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_opponent_candidate: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_case: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        candidate = str(row["candidate"])
        opponent = str(row["opponent"])
        by_candidate[candidate].append(row)
        by_opponent_candidate[(opponent, candidate)].append(row)
        by_case[(opponent, int(row["seed"]), int(row["candidate_seat"]))].append(row)

    candidate_summary = []
    for candidate, candidate_rows in by_candidate.items():
        summary = score_rows(candidate_rows)
        per_opponent = {
            opponent: score_rows(by_opponent_candidate[(opponent, candidate)])
            for opponent in sorted({str(row["opponent"]) for row in candidate_rows})
        }
        summary.update({
            "candidate": candidate,
            "minimum_opponent_win_rate": min(
                (value["win_rate"] for value in per_opponent.values()), default=0.0
            ),
            "per_opponent": per_opponent,
        })
        candidate_summary.append(summary)
    candidate_summary.sort(
        key=lambda item: (
            item["minimum_opponent_win_rate"],
            item["win_rate"],
            item["mean_margin"],
        ),
        reverse=True,
    )

    opponents = sorted({str(row["opponent"]) for row in rows})
    opponent_summary: dict[str, Any] = {}
    for opponent in opponents:
        fixed = []
        for candidate in by_candidate:
            candidate_rows = by_opponent_candidate.get((opponent, candidate), [])
            if candidate_rows:
                item = score_rows(candidate_rows)
                item["candidate"] = candidate
                fixed.append(item)
        fixed.sort(key=lambda item: (item["win_rate"], item["mean_margin"]), reverse=True)

        case_rows = [case for key, case in by_case.items() if key[0] == opponent]
        oracle_games = len(case_rows)
        oracle_wins = sum(any(bool(row.get("win")) for row in case) for case in case_rows)
        oracle_margins = [max(float(row.get("margin", 0.0)) for row in case) for case in case_rows]
        opponent_summary[opponent] = {
            "best_fixed": fixed[:5],
            "per_case_oracle": {
                "games": oracle_games,
                "wins": oracle_wins,
                "win_rate": oracle_wins / oracle_games if oracle_games else 0.0,
                "mean_best_margin": sum(oracle_margins) / oracle_games if oracle_games else 0.0,
            },
        }

    all_cases = list(by_case.values())
    oracle_games = len(all_cases)
    oracle_wins = sum(any(bool(row.get("win")) for row in case) for case in all_cases)
    oracle_margins = [max(float(row.get("margin", 0.0)) for row in case) for case in all_cases]

    # Greedy coverage is descriptive only.  It shows how many fixed route
    # streams are needed even with impossible hindsight access to the outcome.
    uncovered = set(by_case)
    greedy_cover = []
    while uncovered:
        best_candidate = None
        best_covered: set[tuple[str, int, int]] = set()
        for candidate in by_candidate:
            covered = {
                key
                for key in uncovered
                if any(
                    row["candidate"] == candidate and bool(row.get("win"))
                    for row in by_case[key]
                )
            }
            if len(covered) > len(best_covered):
                best_candidate = candidate
                best_covered = covered
        if not best_candidate or not best_covered:
            break
        greedy_cover.append({
            "candidate": best_candidate,
            "newly_covered": len(best_covered),
            "remaining": len(uncovered - best_covered),
        })
        uncovered -= best_covered

    output = {
        "source": str(args.matrix),
        "all_done": bool(payload.get("all_done")),
        "game_count": len(rows),
        "case_count": len(by_case),
        "candidate_count": len(by_candidate),
        "opponent_count": len(opponents),
        "top_candidates_minimax_then_overall": candidate_summary[: args.top],
        "per_opponent": opponent_summary,
        "per_case_oracle": {
            "games": oracle_games,
            "wins": oracle_wins,
            "win_rate": oracle_wins / oracle_games if oracle_games else 0.0,
            "mean_best_margin": sum(oracle_margins) / oracle_games if oracle_games else 0.0,
            "unwinnable_cases": [
                {"opponent": key[0], "seed": key[1], "candidate_seat": key[2]}
                for key, case in by_case.items()
                if not any(bool(row.get("win")) for row in case)
            ],
        },
        "greedy_hindsight_cover": greedy_cover,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
