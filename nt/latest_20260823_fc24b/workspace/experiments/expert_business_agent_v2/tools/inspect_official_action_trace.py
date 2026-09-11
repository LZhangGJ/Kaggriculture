#!/usr/bin/env python3
"""Inspect market actions and public state from one official 1.32.7 game."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: str) -> str:
    candidate = Path(path)
    return str(candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve())


def animal_counts(farm: dict) -> dict[str, int]:
    counts = {"COW": 0, "SHEEP": 0, "GOOSE": 0}
    for row in farm.get("tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if isinstance(tile, dict) and tile.get("animal") in counts:
                counts[str(tile["animal"])] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from kaggle_environments import make

    agents = [resolve(args.candidate), resolve(args.opponent)]
    if args.seat == 1:
        agents.reverse()
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": args.seed}, debug=False)
    env.run(agents)
    rows = []
    for step in range(max(0, args.start_step), len(env.steps) - 1):
        actor = env.steps[step][args.seat]
        action = actor.action or {}
        market = list(action.get("market", []) or []) if isinstance(action, dict) else []
        farmer = list(action.get("farmer", []) or []) if isinstance(action, dict) else []
        hands = [list(value or []) for value in list(action.get("hands", []) or [])] if isinstance(action, dict) else []
        unit_actions = [farmer, *hands]
        tracked_unit = any(
            unit and str(unit[0]) in {"PICKUP", "PLACE", "HARVEST", "DROP"}
            for unit in unit_actions
        )
        if not market and not tracked_unit:
            continue
        obs = actor.observation
        farms = list(obs.get("farms", []) or [])
        farm = farms[args.seat]
        opponent = farms[1 - args.seat]
        positions = [list(farm.get("farmer", []) or []), *[list(value or []) for value in list(farm.get("hands", []) or [])]]
        tiles = list(farm.get("tiles", []) or [])

        def tile_at(position):
            if len(position) < 2:
                return None
            row, col = int(position[0]), int(position[1])
            if 0 <= row < len(tiles) and isinstance(tiles[row], list) and 0 <= col < len(tiles[row]):
                return tiles[row][col]
            return None

        rows.append({
            "step": step,
            "day": int(obs.get("day", 0) or 0),
            "hour": int(obs.get("hour", 0) or 0),
            "shops": list((obs.get("town", {}) or {}).get("unlocked_shops", []) or []),
            "market": market,
            "farmer_action": farmer,
            "hand_actions": hands,
            "own_money": int(farm.get("money", 0) or 0),
            "opponent_money": int(opponent.get("money", 0) or 0),
            "own_animals": animal_counts(farm),
            "opponent_animals": animal_counts(opponent),
            "private_shed": dict((obs.get("private", {}) or {}).get("shed", {}) or {}),
            "private_inventories": [dict(value or {}) for value in list((obs.get("private", {}) or {}).get("inventories", []) or [])],
            "unit_positions": positions,
            "unit_tiles": [tile_at(position) for position in positions],
            "prices": dict((obs.get("market", {}) or {}).get("prices", {}) or {}),
            "inventory": dict((obs.get("market", {}) or {}).get("inventory", {}) or {}),
        })
    payload = {
        "schema": "kaggriculture-official-action-trace-v1",
        "seed": args.seed,
        "candidate_seat": args.seat,
        "frames": len(env.steps),
        "final_rewards": [float(agent.reward) for agent in env.steps[-1]],
        "final_status": [str(agent.status) for agent in env.steps[-1]],
        "market_action_rows": rows,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
