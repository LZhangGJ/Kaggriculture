#!/usr/bin/env python3
"""Screen frozen high-support routes for one reproducible G001 win."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--top-routes", type=int, default=30)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    entries = list(metadata["opponent_routes"])
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    g001 = bundle.index("G001")
    rows = []
    for entry in entries[: args.top_routes]:
        opponent_name = str(entry["family"])
        if opponent_name == "G001":
            continue
        opponent = bundle.index(opponent_name)
        games = []
        for seed in range(args.seed_start, args.seed_start + args.seed_count):
            for candidate_seat in (0, 1):
                if candidate_seat == 0:
                    result = bundle.executor.play(
                        g001, opponent, seed, -1, -1, -1, -1, False
                    )
                else:
                    result = bundle.executor.play(
                        opponent, g001, seed, -1, -1, -1, -1, False
                    )
                rewards = [float(value) for value in result["rewards"]]
                own = rewards[candidate_seat]
                rival = rewards[1 - candidate_seat]
                games.append(
                    {
                        "seed": seed,
                        "seat": candidate_seat,
                        "own_cash": own,
                        "opponent_cash": rival,
                        "margin": own - rival,
                    }
                )
        margins = np.asarray([game["margin"] for game in games], dtype=np.float64)
        own = np.asarray([game["own_cash"] for game in games], dtype=np.float64)
        rival = np.asarray([game["opponent_cash"] for game in games], dtype=np.float64)
        wins = [game for game in games if game["margin"] > 0]
        best_win = max(wins, key=lambda game: game["margin"]) if wins else None
        closest_wins = sorted(wins, key=lambda game: game["margin"])[:8]
        rows.append(
            {
                "opponent": opponent_name,
                "team": str(entry.get("team", "")),
                "route_id": str(entry.get("route_id", "")),
                "support": int(entry.get("support", 0)),
                "source_execution_hard_failures": int(
                    entry.get("source_execution_hard_failures", 0)
                ),
                "games": len(games),
                "wins": int(np.count_nonzero(margins > 0)),
                "win_rate": float(np.mean(margins > 0)),
                "mean_own_cash": float(own.mean()),
                "mean_opponent_cash": float(rival.mean()),
                "mean_margin": float(margins.mean()),
                "best_winning_game": best_win,
                "closest_winning_games": closest_wins,
            }
        )
    close_wins = []
    for row in rows:
        if row["source_execution_hard_failures"] != 0:
            continue
        for game in row["closest_winning_games"]:
            close_wins.append(
                {
                    "opponent": row["opponent"],
                    "team": row["team"],
                    "route_id": row["route_id"],
                    "support": row["support"],
                    **game,
                }
            )
    close_wins.sort(
        key=lambda game: (
            game["margin"],
            -game["opponent_cash"],
            -game["support"],
        )
    )
    payload = {
        "schema": "kaggriculture.g001-strong-route-screen.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "config": {
            "top_routes": args.top_routes,
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "dual_seat": True,
        },
        "rows": rows,
        "closest_zero_hard_failure_wins": close_wins,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            sorted(rows, key=lambda row: (-row["win_rate"], -row["support"]))[:20],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
