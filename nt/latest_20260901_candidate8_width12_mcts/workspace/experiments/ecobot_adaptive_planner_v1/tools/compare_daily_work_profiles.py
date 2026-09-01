#!/usr/bin/env python3
"""Compare daily unit operation throughput in an official Replay and native trace."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


MOVE_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}


def op_name(action: object) -> str:
    if isinstance(action, (list, tuple)) and action:
        return str(action[0])
    return "PASS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-replay", type=Path, required=True)
    parser.add_argument("--official-seat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--native-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    replay = json.loads(args.official_replay.read_text(encoding="utf-8"))
    trace = json.loads(args.native_trace.read_text(encoding="utf-8"))
    official = [collections.Counter() for _ in range(30)]
    native = [collections.Counter() for _ in range(30)]
    for step in replay["steps"][1:]:
        row = step[args.official_seat]
        day = int(row["observation"]["day"])
        action = row.get("action") or {"farmer": ["PASS"], "hands": []}
        for unit_action in [action.get("farmer", ["PASS"]), *action.get("hands", [])]:
            official[day][op_name(unit_action)] += 1
    for entry in trace["timeline"]:
        day = int(entry["day"])
        for unit in entry["units"]:
            native[day][str(unit["op"])] += 1

    operations = sorted(set().union(*(set(row) for row in official + native)))
    days = []
    for day in range(30):
        off_move = sum(official[day][op] for op in MOVE_OPS)
        nat_move = sum(native[day][op] for op in MOVE_OPS)
        off_pass = official[day]["PASS"]
        nat_pass = native[day]["PASS"]
        off_work = sum(value for op, value in official[day].items()
                       if op not in MOVE_OPS and op != "PASS")
        nat_work = sum(value for op, value in native[day].items()
                       if op not in MOVE_OPS and op != "PASS")
        days.append({
            "day": day + 1,
            "official": dict(official[day]),
            "native": dict(native[day]),
            "delta_native_minus_official": {
                "move": nat_move - off_move,
                "work": nat_work - off_work,
                "pass": nat_pass - off_pass,
                **{op: native[day][op] - official[day][op] for op in operations},
            },
        })

    off_total = sum((row for row in official), collections.Counter())
    nat_total = sum((row for row in native), collections.Counter())
    payload = {
        "schema": "kaggriculture.daily-work-profile-comparison.v1",
        "official_replay": str(args.official_replay),
        "official_seat": args.official_seat,
        "native_trace": str(args.native_trace),
        "totals": {
            "official": dict(off_total),
            "native": dict(nat_total),
            "delta_native_minus_official": {
                op: nat_total[op] - off_total[op] for op in operations
            },
        },
        "days": days,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    compact = {
        "schema": payload["schema"],
        "totals": payload["totals"],
        "largest_work_deficits": sorted(
            ({"day": row["day"], **row["delta_native_minus_official"]}
             for row in days), key=lambda row: row["work"]
        )[:10],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
