"""Compare a gold Replay with one M3.7 controller trace day by day.

This is a diagnostic only.  Replay actions are read to count expert workload;
they are never compiled into or executed by the controller.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = (
    ROOT
    / "replay"
    / "gold_top20_latest_2026-08-18_082723"
    / "latest_per_gold"
    / "episode-94051618-replay.json"
)
DEFAULT_TRACE = PROJECT / "artifacts" / "traces" / "m37c_conflict_target_preemption_trace_v12.npz"
DEFAULT_OUTPUT = PROJECT / "receipts" / "m37c_gold_v12_daily_comparison_v1.json"

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))
MOVE_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}
UNIT_OP_NAMES = {
    0: "PASS",
    1: "NORTH",
    2: "SOUTH",
    3: "EAST",
    4: "WEST",
    5: "DROP",
    6: "PICKUP",
    7: "PLACE",
    8: "PLANT",
    9: "WATER",
    10: "HARVEST",
    11: "FERTILIZE",
    12: "DIG",
    13: "BUILD_COOP",
    14: "BUILD_PASTURE",
    15: "FEED",
    16: "COLLECT_FERTILIZER",
    17: "CARE",
}
MARKET_OP_NAMES = {
    0: "NONE",
    1: "HIRE",
    2: "BUY_LAND",
    3: "BUY_SEED",
    4: "BUY_PRODUCT",
    5: "BUY_ANIMAL",
    6: "SELL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--player", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def nearest_shed_distance(y: int, x: int) -> int:
    return min(abs(y - sy) + abs(x - sx) for sy, sx in SHED_ACCESS)


def layout_summary_from_official(observation: dict, player: int) -> dict:
    farm = observation["farms"][player]
    crop_tiles: list[list[list[int]]] = [[] for _ in CROPS]
    animal_tiles: list[list[list[int]]] = [[] for _ in ANIMALS]
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT" and tile.get("crop") in CROPS:
                crop_tiles[CROPS.index(tile["crop"])].append([y, x])
            animal = tile.get("animal")
            if animal in ANIMALS:
                animal_tiles[ANIMALS.index(animal)].append([y, x])
    all_service = [pos for group in crop_tiles + animal_tiles for pos in group]
    return {
        "money": int(farm["money"]),
        "units": 1 + len(farm.get("hands", [])),
        "land": len(farm.get("unlocked_quadrants", [])),
        "crop_tiles": [len(group) for group in crop_tiles],
        "animal_tiles": [len(group) for group in animal_tiles],
        "crop_positions": crop_tiles,
        "animal_positions": animal_tiles,
        "service_distance_sum": int(sum(nearest_shed_distance(*pos) for pos in all_service)),
        "service_distance_mean": (
            float(np.mean([nearest_shed_distance(*pos) for pos in all_service]))
            if all_service
            else 0.0
        ),
        "shed": [int(observation["private"]["shed"].get(name, 0)) for name in (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER", *ANIMALS)],
        "seeds": [int(observation["private"]["seeds"].get(name, 0)) for name in CROPS],
    }


def layout_summary_from_trace(trace: np.lib.npyio.NpzFile, state_index: int, player: int) -> dict:
    kind = np.asarray(trace["state_tile_kind"][state_index, 0, player])
    crop = np.asarray(trace["state_tile_crop"][state_index, 0, player])
    animal = np.asarray(trace["state_tile_animal"][state_index, 0, player])
    crop_tiles: list[list[list[int]]] = []
    animal_tiles: list[list[list[int]]] = []
    for item in range(len(CROPS)):
        positions = np.argwhere((kind == 3) & (crop == item))
        crop_tiles.append(positions.astype(int).tolist())
    for item in range(len(ANIMALS)):
        positions = np.argwhere(animal == item)
        animal_tiles.append(positions.astype(int).tolist())
    all_service = [pos for group in crop_tiles + animal_tiles for pos in group]
    return {
        "money": int(trace["state_money"][state_index, 0, player]),
        "units": int(np.asarray(trace["state_unit_active"][state_index, 0, player]).sum()),
        "land": int(trace["state_unlocked_count"][state_index, 0, player]),
        "crop_tiles": [len(group) for group in crop_tiles],
        "animal_tiles": [len(group) for group in animal_tiles],
        "crop_positions": crop_tiles,
        "animal_positions": animal_tiles,
        "service_distance_sum": int(sum(nearest_shed_distance(*pos) for pos in all_service)),
        "service_distance_mean": (
            float(np.mean([nearest_shed_distance(*pos) for pos in all_service]))
            if all_service
            else 0.0
        ),
        "shed": np.asarray(trace["state_shed"][state_index, 0, player]).astype(int).tolist(),
        "seeds": np.asarray(trace["state_seeds"][state_index, 0, player]).astype(int).tolist(),
    }


def gold_action_counts(replay: dict, player: int, start: int, stop: int) -> dict:
    unit = Counter()
    market = Counter()
    unit_offsets: dict[str, list[int]] = {}
    market_offsets: dict[str, list[int]] = {}
    market_orders: list[list[object]] = []
    slots = 0
    for state_step in range(start + 1, stop + 1):
        offset = state_step - (start + 1)
        action = replay["steps"][state_step][player].get("action") or {}
        farmer = action.get("farmer") or ["PASS"]
        hands = action.get("hands") or []
        unit_actions = [farmer, *hands]
        slots += len(unit_actions)
        for value in unit_actions:
            op = str(value[0]) if value else "PASS"
            unit[op] += 1
            unit_offsets.setdefault(op, []).append(offset)
        for value in action.get("market") or []:
            op = str(value[0]) if value else "NONE"
            market[op] += 1
            market_offsets.setdefault(op, []).append(offset)
            market_orders.append([offset, *value])
    non_pass = sum(count for op, count in unit.items() if op != "PASS")
    productive = sum(count for op, count in unit.items() if op not in MOVE_OPS | {"PASS"})
    return {
        "available_unit_action_slots": slots,
        "non_pass_unit_actions": non_pass,
        "productive_unit_actions": productive,
        "movement_actions": sum(unit[op] for op in MOVE_OPS),
        "unit_ops": dict(sorted(unit.items())),
        "market_ops": dict(sorted(market.items())),
        "unit_op_offsets": dict(sorted(unit_offsets.items())),
        "market_op_offsets": dict(sorted(market_offsets.items())),
        "market_orders": market_orders,
    }


def trace_action_counts(trace: np.lib.npyio.NpzFile, start: int, stop: int) -> dict:
    unit = Counter()
    market = Counter()
    unit_offsets: dict[str, list[int]] = {}
    market_offsets: dict[str, list[int]] = {}
    market_orders: list[list[object]] = []
    slots = 0
    for action_step in range(start, stop):
        offset = action_step - start
        count = int(trace["action_unit_count"][action_step, 0])
        slots += count
        for value in np.asarray(trace["action_unit_op"][action_step, 0, :count]).tolist():
            op = UNIT_OP_NAMES[int(value)]
            unit[op] += 1
            unit_offsets.setdefault(op, []).append(offset)
        market_count = int(trace["action_market_count"][action_step, 0])
        for order_index, value in enumerate(
            np.asarray(trace["action_market_op"][action_step, 0, :market_count]).tolist()
        ):
            op = MARKET_OP_NAMES[int(value)]
            market[op] += 1
            market_offsets.setdefault(op, []).append(offset)
            market_orders.append(
                [
                    offset,
                    op,
                    int(trace["action_market_item"][action_step, 0, order_index]),
                    int(trace["action_market_amount"][action_step, 0, order_index]),
                ]
            )
    non_pass = sum(count for op, count in unit.items() if op != "PASS")
    productive = sum(count for op, count in unit.items() if op not in MOVE_OPS | {"PASS"})
    return {
        "available_unit_action_slots": slots,
        "non_pass_unit_actions": non_pass,
        "productive_unit_actions": productive,
        "movement_actions": sum(unit[op] for op in MOVE_OPS),
        "unit_ops": dict(sorted(unit.items())),
        "market_ops": dict(sorted(market.items())),
        "unit_op_offsets": dict(sorted(unit_offsets.items())),
        "market_op_offsets": dict(sorted(market_offsets.items())),
        "market_orders": market_orders,
    }


def vector_delta(left: list[int], right: list[int]) -> list[int]:
    return (np.asarray(left, dtype=np.int64) - np.asarray(right, dtype=np.int64)).astype(int).tolist()


def main() -> None:
    args = parse_args()
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    if replay.get("module_version") != "1.32.7":
        raise ValueError(f"expected 1.32.7 Replay, got {replay.get('module_version')}")
    rows = []
    with np.load(args.trace, allow_pickle=False) as trace:
        if int(np.asarray(trace["player"])[0]) != args.player:
            raise ValueError("trace player does not match --player")
        for day_index in range(30):
            start = day_index * 24
            stop = min((day_index + 1) * 24, 719)
            if start >= stop:
                break
            gold_end = layout_summary_from_official(replay["steps"][stop][args.player]["observation"], args.player)
            candidate_end = layout_summary_from_trace(trace, stop - 1, args.player)
            gold_actions = gold_action_counts(replay, args.player, start, stop)
            candidate_actions = trace_action_counts(trace, start, stop)
            rows.append(
                {
                    "day": day_index + 1,
                    "action_steps": [start, stop - 1],
                    "gold": {"end": gold_end, "actions": gold_actions},
                    "candidate": {"end": candidate_end, "actions": candidate_actions},
                    "candidate_minus_gold": {
                        "money": candidate_end["money"] - gold_end["money"],
                        "units": candidate_end["units"] - gold_end["units"],
                        "land": candidate_end["land"] - gold_end["land"],
                        "crop_tiles": vector_delta(candidate_end["crop_tiles"], gold_end["crop_tiles"]),
                        "animal_tiles": vector_delta(candidate_end["animal_tiles"], gold_end["animal_tiles"]),
                        "productive_actions": candidate_actions["productive_unit_actions"] - gold_actions["productive_unit_actions"],
                        "movement_actions": candidate_actions["movement_actions"] - gold_actions["movement_actions"],
                        "service_distance_sum": candidate_end["service_distance_sum"] - gold_end["service_distance_sum"],
                    },
                }
            )

    focus = rows[5:12]
    output = {
        "schema": "kaggriculture.m37c_gold_v12_daily_comparison.v1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "scope": "diagnostic comparison only; Replay raw actions are counted but never executed by the controller",
        "official_version": replay["module_version"],
        "episode_id": int(replay["info"]["EpisodeId"]),
        "seed": int(replay["info"]["seed"]),
        "player": args.player,
        "replay_sha256": sha256(args.replay),
        "trace_sha256": sha256(args.trace),
        "daily": rows,
        "focus_days_6_to_12": focus,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            [
                {
                    "day": row["day"],
                    "gold_crop": row["gold"]["end"]["crop_tiles"],
                    "candidate_crop": row["candidate"]["end"]["crop_tiles"],
                    "gold_animals": row["gold"]["end"]["animal_tiles"],
                    "candidate_animals": row["candidate"]["end"]["animal_tiles"],
                    "gold_actions": row["gold"]["actions"]["unit_ops"],
                    "candidate_actions": row["candidate"]["actions"]["unit_ops"],
                    "delta": row["candidate_minus_gold"],
                }
                for row in focus
            ],
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
