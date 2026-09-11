#!/usr/bin/env python3
"""Print compact public-state and route-outcome rows for selected arena cases."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--opponents", default="")
    parser.add_argument("--candidates", default="")
    parser.add_argument("--step", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.matrix.read_text(encoding="utf-8"))
    opponents = {value.strip() for value in args.opponents.split(",") if value.strip()}
    candidates = {value.strip() for value in args.candidates.split(",") if value.strip()}
    grouped = defaultdict(list)
    for row in payload["rows"]:
        if opponents and row["opponent"] not in opponents:
            continue
        if candidates and row["candidate"] not in candidates:
            continue
        grouped[(row["opponent"], int(row["seed"]), int(row["candidate_seat"]))].append(row)

    cases = []
    for (opponent, seed, seat), rows in sorted(grouped.items()):
        state = rows[0]["public_state_steps"][str(args.step)]
        opp = state["opponent_stats"]
        own = state["own_stats"]
        route_rows = sorted(rows, key=lambda row: (bool(row["win"]), float(row["margin"])), reverse=True)
        cases.append({
            "opponent": opponent,
            "seed": seed,
            "candidate_seat": seat,
            "public": {
                "shops": state["shops"],
                "prices": state["prices"],
                "inventory": state["inventory"],
                "own_money": own["money"],
                "opponent_money": opp["money"],
                "opponent_hands": opp["hands"],
                "opponent_plant_tiles": opp["plant_tiles"],
                "opponent_pasture_tiles": opp["pasture_tiles"],
                "opponent_crops": opp["crop_count"],
                "opponent_animals": opp["animal_count"],
            },
            "routes": [
                {
                    "candidate": row["candidate"],
                    "win": bool(row["win"]),
                    "margin": float(row["margin"]),
                    "candidate_reward": float(row["candidate_reward"]),
                    "opponent_reward": float(row["opponent_reward"]),
                }
                for row in route_rows
            ],
        })
    rendered = json.dumps({"step": args.step, "cases": cases}, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
