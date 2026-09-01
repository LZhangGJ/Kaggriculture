#!/usr/bin/env python3
"""Emit compact action and milestone diagnostics from an official Replay JSON."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path


def operation(action: object) -> str:
    if isinstance(action, (list, tuple)) and action:
        return str(action[0])
    return "PASS"


def tile_label(tile: object) -> str:
    if tile is None:
        return "EMPTY"
    if isinstance(tile, str):
        return tile
    if isinstance(tile, dict):
        for key in ("crop", "animal", "building", "type", "content"):
            if key in tile:
                value = tile[key]
                if isinstance(value, dict):
                    return f"{key}:{value.get('type', value.get('name', 'DICT'))}"
                return f"{key}:{value}"
        return "DICT:" + ",".join(sorted(str(key) for key in tile))
    if isinstance(tile, (list, tuple)) and tile:
        return str(tile[0])
    return type(tile).__name__


def snapshot(observation: dict, player: int) -> dict[str, object]:
    farm = observation["farms"][player]
    private = observation["private"]
    tiles: collections.Counter[str] = collections.Counter()
    tile_positions: dict[str, list[list[int]]] = collections.defaultdict(list)
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            label = tile_label(tile)
            if label not in ("EMPTY", "LOCKED"):
                tiles[label] += 1
                tile_positions[label].append([x, y])
    carried: collections.Counter[str] = collections.Counter()
    for inventory in private["inventories"]:
        for item, quantity in inventory.items():
            carried[str(item)] += int(quantity)
    return {
        "step": int(observation["step"]),
        "day": int(observation["day"]),
        "hour": int(observation["hour"]),
        "money": float(farm["money"]),
        "hands": len(farm["hands"]),
        "unlocked_quadrants": list(farm["unlocked_quadrants"]),
        "shed": {key: int(value) for key, value in private["shed"].items() if value},
        "seeds": {key: int(value) for key, value in private["seeds"].items() if value},
        "carried": dict(carried),
        "tiles": dict(tiles),
        "tile_positions": dict(tile_positions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--days", default="1,2,4,6,9,11,13,16,21,26,30")
    args = parser.parse_args()
    opener = gzip.open if args.replay.suffix == ".gz" else open
    with opener(args.replay, "rt", encoding="utf-8") as handle:
        replay = json.load(handle)
    unit_attempts: collections.Counter[str] = collections.Counter()
    market_attempts: collections.Counter[str] = collections.Counter()
    market_quantities: collections.Counter[str] = collections.Counter()
    samples: dict[str, object] = {}
    snapshots: list[dict[str, object]] = []
    chosen_days = {int(value) for value in args.days.split(",") if value.strip()}
    for step in replay["steps"][1:]:
        row = step[args.seat]
        action = row.get("action") or {"farmer": ["PASS"], "hands": [], "market": []}
        for value in [action.get("farmer", ["PASS"]), *action.get("hands", [])]:
            op = operation(value)
            unit_attempts[op] += 1
            samples.setdefault(op, value)
        for value in action.get("market", []):
            op = operation(value)
            market_attempts[op] += 1
            quantity = value[-1] if len(value) >= 3 else 1
            if isinstance(quantity, (int, float)):
                market_quantities[op] += int(quantity)
            samples.setdefault(op, value)
        observation = row["observation"]
        if int(observation["hour"]) == 23 and int(observation["day"]) + 1 in chosen_days:
            snapshots.append(snapshot(observation, args.seat))
    final_observation = replay["steps"][-1][args.seat]["observation"]
    payload = {
        "replay": str(args.replay),
        "seat": args.seat,
        "final_rewards": replay.get("rewards"),
        "unit_attempts": dict(unit_attempts),
        "market_attempts": dict(market_attempts),
        "market_quantities": dict(market_quantities),
        "action_samples": samples,
        "snapshots": snapshots,
        "terminal_observation": snapshot(final_observation, args.seat),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
