#!/usr/bin/env python3
"""Aggregate public money, production, and market trajectories from an arena."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any


def avg(values: list[float]) -> float:
    return mean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.matrix.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row["candidate"] == args.candidate]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["opponent"])].append(row)

    result = {}
    for opponent, opponent_rows in sorted(grouped.items()):
        steps = sorted({int(step) for row in opponent_rows for step in row["public_state_steps"]})
        trajectory = []
        for step in steps:
            snapshots = [row["public_state_steps"][str(step)] for row in opponent_rows]
            trajectory.append({
                "step": step,
                "own_money": avg([float(value["own_stats"]["money"]) for value in snapshots]),
                "opponent_money": avg([float(value["opponent_stats"]["money"]) for value in snapshots]),
                "cash_gap": avg([
                    float(value["own_stats"]["money"]) - float(value["opponent_stats"]["money"])
                    for value in snapshots
                ]),
                "own_cows": avg([float(value["own_stats"]["animal_count"]["COW"]) for value in snapshots]),
                "own_sheep": avg([float(value["own_stats"]["animal_count"]["SHEEP"]) for value in snapshots]),
                "opponent_cows": avg([float(value["opponent_stats"]["animal_count"]["COW"]) for value in snapshots]),
                "opponent_sheep": avg([float(value["opponent_stats"]["animal_count"]["SHEEP"]) for value in snapshots]),
                "own_plant_tiles": avg([float(value["own_stats"]["plant_tiles"]) for value in snapshots]),
                "opponent_plant_tiles": avg([float(value["opponent_stats"]["plant_tiles"]) for value in snapshots]),
                "milk_price": avg([float(value["prices"]["MILK"]) for value in snapshots]),
                "wool_price": avg([float(value["prices"]["WOOL"]) for value in snapshots]),
                "milk_inventory": avg([float(value["inventory"]["MILK"]) for value in snapshots]),
                "wool_inventory": avg([float(value["inventory"]["WOOL"]) for value in snapshots]),
            })
        result[opponent] = {
            "games": len(opponent_rows),
            "wins": sum(bool(row["win"]) for row in opponent_rows),
            "win_rate": sum(bool(row["win"]) for row in opponent_rows) / len(opponent_rows),
            "mean_final_margin": avg([float(row["margin"]) for row in opponent_rows]),
            "trajectory": trajectory,
        }
    output = {
        "schema": "arena-public-snapshot-trajectory-v1",
        "source_matrix": str(args.matrix),
        "candidate": args.candidate,
        "per_opponent": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
