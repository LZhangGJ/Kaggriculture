#!/usr/bin/env python3
"""Join the stable-rule outcomes to daily public trajectories for attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-receipt", type=Path, required=True)
    parser.add_argument("--trajectory-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stable = json.loads(args.stable_receipt.read_text(encoding="utf-8"))
    trajectory = json.loads(args.trajectory_receipt.read_text(encoding="utf-8"))
    stable_row = next(
        row
        for row in stable["rows"]
        if row["rule"] == "any_yarn_route7" and not row["enable_counters"]
    )
    outcomes = {
        (int(row["seed"]), int(row["candidate_seat"])): row
        for row in stable_row["per_game"]
    }
    games = trajectory["pairs"][0]["per_game"]
    joined = []
    for game in games:
        key = (int(game["seed"]), int(game["candidate_seat"]))
        if key not in outcomes:
            continue
        joined.append((outcomes[key], game))
    if len(joined) != len(outcomes):
        raise AssertionError(f"join mismatch: {len(joined)} != {len(outcomes)}")

    steps = (24, 72, 120, 168, 192, 216, 240, 288, 360, 432, 504, 576, 648, 696, 719)
    groups = {
        "stable_wins": [pair for pair in joined if int(pair[0]["margin"]) > 0],
        "stable_losses": [pair for pair in joined if int(pair[0]["margin"]) < 0],
    }
    summaries = {}
    for group_name, pairs in groups.items():
        rows = []
        for requested_step in steps:
            selected = []
            for _, game in pairs:
                day = min(game["daily"], key=lambda row: abs(int(row["state_step"]) - requested_step))
                selected.append(day)
            rows.append({
                "state_step": int(selected[0]["state_step"]) if selected else requested_step,
                "money_own": mean([row["money_own"] for row in selected]),
                "money_rival": mean([row["money_rival"] for row in selected]),
                "cash_gap": mean([row["money_own"] - row["money_rival"] for row in selected]),
                "own_cows": mean([row["animals_own"][1] for row in selected]),
                "own_sheep": mean([row["animals_own"][2] for row in selected]),
                "rival_cows": mean([row["animals_rival"][1] for row in selected]),
                "rival_sheep": mean([row["animals_rival"][2] for row in selected]),
                "own_milk_ready": mean([row["animal_yield_own"][1] for row in selected]),
                "own_wool_ready": mean([row["animal_yield_own"][2] for row in selected]),
                "rival_milk_ready": mean([row["animal_yield_rival"][1] for row in selected]),
                "rival_wool_ready": mean([row["animal_yield_rival"][2] for row in selected]),
                "own_strawberry_tiles": mean([row["crops_own"][3] for row in selected]),
                "rival_strawberry_tiles": mean([row["crops_rival"][3] for row in selected]),
                "milk_price": mean([row["market_price"][6] for row in selected]),
                "wool_price": mean([row["market_price"][7] for row in selected]),
                "strawberry_price": mean([row["market_price"][3] for row in selected]),
            })
        summaries[group_name] = {
            "games": len(pairs),
            "mean_stable_margin": mean([pair[0]["margin"] for pair in pairs]),
            "mean_exact_x562_margin": mean([pair[1]["margin"] for pair in pairs]),
            "timeline": rows,
        }

    loss_rows = []
    for stable_outcome, game in groups["stable_losses"]:
        final = game["daily"][-1]
        loss_rows.append({
            "seed": int(game["seed"]),
            "candidate_seat": int(game["candidate_seat"]),
            "stable_margin": int(stable_outcome["margin"]),
            "exact_x562_margin": int(game["margin"]),
            "rival_route_diagnostic": int(final["route_rival_diagnostic"]),
            "town_shops": final["town_shops"],
            "terminal_public_animals_own": final["animals_own"],
            "terminal_public_animals_rival": final["animals_rival"],
        })
    payload = {
        "schema": "kaggriculture.fusion_champion.rank14_residual_attribution.v1",
        "status": "PASS",
        "stable_receipt": str(args.stable_receipt.resolve()),
        "trajectory_receipt": str(args.trajectory_receipt.resolve()),
        "joined_games": len(joined),
        "summaries": summaries,
        "stable_losses": loss_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
