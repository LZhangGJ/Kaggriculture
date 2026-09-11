"""Compare M3.6 controller action use with the source gold Replay.

This is a diagnostic only.  Replay actions are read for attribution and are
never compiled or played by the controller.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
DEFAULT_REPLAY = (
    ROOT
    / "replay"
    / "gold_top20_latest_2026-08-18_082723"
    / "latest_per_gold"
    / "episode-94051618-replay.json"
)
DEFAULT_TRACE = PROJECT / "artifacts" / "traces" / "m36d_controller_trace_v7.npz"

UNIT_OP = {
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
MARKET_OP = {
    0: "NONE",
    1: "HIRE",
    2: "BUY_LAND",
    3: "BUY_SEED",
    4: "BUY_PRODUCT",
    5: "BUY_ANIMAL",
    6: "SELL",
}
SHED_ITEM = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
    "GOOSE",
    "COW",
    "SHEEP",
)
MOVE_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}
OPERATE_OPS = {
    "DROP",
    "PICKUP",
    "PLACE",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--player", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "receipts" / "m36d_action_efficiency_audit_v1.json",
    )
    return parser.parse_args()


def raw_op(action) -> str:
    if not isinstance(action, list) or not action:
        return "PASS"
    return str(action[0])


def raw_amount(action) -> int:
    if not isinstance(action, list) or len(action) < 3:
        return 1
    return int(action[2])


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def summarize(counter: Counter[str]) -> dict[str, object]:
    movement = sum(counter[name] for name in MOVE_OPS)
    operations = sum(counter[name] for name in OPERATE_OPS)
    non_pass = sum(value for key, value in counter.items() if key != "PASS")
    return {
        "by_operation": counter_dict(counter),
        "movement": int(movement),
        "operations": int(operations),
        "non_pass": int(non_pass),
        "movement_per_operation": (
            float(movement / operations) if operations else None
        ),
    }


def main() -> None:
    args = parse_args()
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    trace = np.load(args.trace, allow_pickle=False)
    steps = int(trace["action_unit_op"].shape[0])
    if len(replay["steps"]) != steps + 1:
        raise ValueError("Replay states and controller transitions do not align")

    candidate_unit: Counter[str] = Counter()
    candidate_market: Counter[str] = Counter()
    candidate_pickup_amount: Counter[str] = Counter()
    candidate_pickup_calls_by_item: Counter[str] = Counter()
    candidate_pickup_units_by_item: Counter[str] = Counter()
    gold_unit: Counter[str] = Counter()
    gold_market: Counter[str] = Counter()
    gold_pickup_amount: Counter[str] = Counter()
    gold_pickup_calls_by_item: Counter[str] = Counter()
    gold_pickup_units_by_item: Counter[str] = Counter()
    candidate_day = [Counter() for _ in range(30)]
    gold_day = [Counter() for _ in range(30)]

    for transition in range(steps):
        day = transition // 24
        unit_count = int(trace["action_unit_count"][transition, 0])
        for lane in range(unit_count):
            op = UNIT_OP[int(trace["action_unit_op"][transition, 0, lane])]
            candidate_unit[op] += 1
            candidate_day[day][op] += 1
            if op == "PICKUP":
                amount = int(trace["action_unit_amount"][transition, 0, lane])
                candidate_pickup_amount[str(amount)] += 1
                item_id = int(trace["action_unit_item"][transition, 0, lane])
                item = SHED_ITEM[item_id] if 0 <= item_id < len(SHED_ITEM) else str(item_id)
                candidate_pickup_calls_by_item[item] += 1
                candidate_pickup_units_by_item[item] += amount
        market_count = int(trace["action_market_count"][transition, 0])
        for lane in range(market_count):
            op = MARKET_OP[int(trace["action_market_op"][transition, 0, lane])]
            candidate_market[op] += 1

        raw = replay["steps"][transition + 1][args.player].get("action") or {}
        unit_actions = [raw.get("farmer", [])] + list(raw.get("hands", []))
        for action in unit_actions:
            op = raw_op(action)
            gold_unit[op] += 1
            gold_day[day][op] += 1
            if op == "PICKUP":
                amount = raw_amount(action)
                item = str(action[1]) if len(action) >= 2 else "UNKNOWN"
                gold_pickup_amount[str(amount)] += 1
                gold_pickup_calls_by_item[item] += 1
                gold_pickup_units_by_item[item] += amount
        for action in raw.get("market", []):
            gold_market[raw_op(action)] += 1

    day_rows = []
    for day in range(30):
        candidate = summarize(candidate_day[day])
        gold = summarize(gold_day[day])
        day_rows.append(
            {
                "day": day + 1,
                "candidate": candidate,
                "gold": gold,
                "candidate_minus_gold": {
                    "movement": candidate["movement"] - gold["movement"],
                    "operations": candidate["operations"] - gold["operations"],
                    "non_pass": candidate["non_pass"] - gold["non_pass"],
                },
            }
        )

    candidate_summary = summarize(candidate_unit)
    gold_summary = summarize(gold_unit)
    receipt = {
        "schema": "kaggriculture.m36d_action_efficiency_audit.v1",
        "boundary": "DIAGNOSTIC_ONLY_REPLAY_ACTIONS_NOT_USED_BY_CONTROLLER",
        "episode_id": int(replay["info"]["EpisodeId"]),
        "player": args.player,
        "controller_trace": str(args.trace.resolve()),
        "gold_replay": str(args.replay.resolve()),
        "transition_count": steps,
        "candidate": {
            "unit": candidate_summary,
            "market": counter_dict(candidate_market),
            "pickup_amount_histogram": counter_dict(candidate_pickup_amount),
            "pickup_calls_by_item": counter_dict(candidate_pickup_calls_by_item),
            "pickup_units_by_item": counter_dict(candidate_pickup_units_by_item),
        },
        "gold": {
            "unit": gold_summary,
            "market": counter_dict(gold_market),
            "pickup_amount_histogram": counter_dict(gold_pickup_amount),
            "pickup_calls_by_item": counter_dict(gold_pickup_calls_by_item),
            "pickup_units_by_item": counter_dict(gold_pickup_units_by_item),
        },
        "candidate_minus_gold": {
            "movement": candidate_summary["movement"] - gold_summary["movement"],
            "operations": candidate_summary["operations"] - gold_summary["operations"],
            "non_pass": candidate_summary["non_pass"] - gold_summary["non_pass"],
        },
        "by_day": day_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
