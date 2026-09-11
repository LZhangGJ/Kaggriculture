#!/usr/bin/env python3
"""Compare deployable shop-route suffixes by the step-168 public shop prefix."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


ROUTES = {
    "current": ("current_x562", False),
    "route4": ("any_yarn_route4", False),
    "route7": ("any_yarn_route7", False),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    margins = {}
    for short, (rule, counters) in ROUTES.items():
        row = next(
            row
            for row in screen["rows"]
            if row["rule"] == rule and row["enable_counters"] == counters
        )
        margins[short] = {
            (int(game["seed"]), int(game["candidate_seat"])): int(game["margin"])
            for game in row["per_game"]
        }
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for game in trajectory["pairs"][0]["per_game"]:
        key = (int(game["seed"]), int(game["candidate_seat"]))
        snapshot = min(game["daily"], key=lambda row: abs(int(row["state_step"]) - 168))
        prefix = tuple(int(value) for value in snapshot["town_shops"][:2])
        groups[",".join(map(str, prefix))].append(key)

    rows = []
    for prefix, keys in sorted(groups.items()):
        route_stats = {}
        for route, lookup in margins.items():
            values = np.asarray([lookup[key] for key in keys], dtype=np.int64)
            route_stats[route] = {
                "wins": int(np.sum(values > 0)),
                "ties": int(np.sum(values == 0)),
                "losses": int(np.sum(values < 0)),
                "score_rate": float(np.mean(values > 0) + 0.5 * np.mean(values == 0)),
                "mean_margin": float(np.mean(values)),
            }
        best_mean = max(route_stats, key=lambda route: route_stats[route]["mean_margin"])
        best_score = max(route_stats, key=lambda route: (route_stats[route]["score_rate"], route_stats[route]["mean_margin"]))
        rows.append({
            "shop_prefix": prefix,
            "games": len(keys),
            "routes": route_stats,
            "best_mean_margin_route": best_mean,
            "best_score_route": best_score,
        })

    matrix = np.stack(
        [np.asarray([lookup[key] for key in sorted(margins["current"])]) for lookup in margins.values()],
        axis=0,
    )
    oracle = np.max(matrix, axis=0)
    payload = {
        "schema": "kaggriculture.fusion_champion.shop_route_choice_analysis.v1",
        "status": "PASS",
        "screen": str(args.screen.resolve()),
        "trajectory": str(args.trajectory.resolve()),
        "games": int(matrix.shape[1]),
        "oracle": {
            "wins": int(np.sum(oracle > 0)),
            "ties": int(np.sum(oracle == 0)),
            "losses": int(np.sum(oracle < 0)),
            "score_rate": float(np.mean(oracle > 0) + 0.5 * np.mean(oracle == 0)),
            "mean_margin": float(np.mean(oracle)),
        },
        "prefix_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
