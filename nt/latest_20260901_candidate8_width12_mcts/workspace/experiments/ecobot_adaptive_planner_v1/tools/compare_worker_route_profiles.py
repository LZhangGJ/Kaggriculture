#!/usr/bin/env python3
"""Compare per-worker route locality in one official Replay and native trace."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Iterable


MOVE_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}


def op_name(action: object) -> str:
    if isinstance(action, (list, tuple)) and action:
        return str(action[0])
    if isinstance(action, str):
        return action
    return "PASS"


def official_rows(path: Path, seat: int) -> Iterable[tuple[int, int, int, list[int], str]]:
    replay = json.loads(path.read_text(encoding="utf-8"))
    for step_index, step in enumerate(replay["steps"][1:]):
        row = step[seat]
        observation = row["observation"]
        day = int(observation["day"])
        farm = observation["farms"][seat]
        positions = [farm["farmer"], *farm["hands"]]
        action = row.get("action") or {"farmer": ["PASS"], "hands": []}
        actions = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
        for unit, unit_action in enumerate(actions):
            if unit < len(positions):
                yield day, step_index, unit, positions[unit], op_name(unit_action)


def native_rows(path: Path) -> Iterable[tuple[int, int, int, list[int], str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in payload["timeline"]:
        day = int(entry["day"])
        step = int(entry["step"])
        for unit in entry["units"]:
            yield day, step, int(unit["unit"]), unit["position"], str(unit["op"])


def summarize(rows: Iterable[tuple[int, int, int, list[int], str]]) -> dict[str, object]:
    lanes: dict[tuple[int, int], list[tuple[int, tuple[int, int], str]]] = collections.defaultdict(list)
    for day, step, unit, position, op in rows:
        lanes[(day, unit)].append((step, (int(position[0]), int(position[1])), op))

    rendered = []
    totals = collections.Counter()
    quadrant_transition_ops: collections.Counter[str] = collections.Counter()
    for (day, unit), events in sorted(lanes.items()):
        moves = sum(op in MOVE_OPS for _, _, op in events)
        passes = sum(op == "PASS" for _, _, op in events)
        work_events = [
            (position, op)
            for _, position, op in events
            if op not in MOVE_OPS and op != "PASS"
        ]
        work_positions = [position for position, _ in work_events]
        work = len(work_positions)
        work_jumps = [
            abs(a[0] - b[0]) + abs(a[1] - b[1])
            for a, b in zip(work_positions, work_positions[1:])
        ]
        quadrant_switches = sum(
            ((a[0] >= 5) + 2 * (a[1] >= 5)) != ((b[0] >= 5) + 2 * (b[1] >= 5))
            for a, b in zip(work_positions, work_positions[1:])
        )
        for (a_pos, a_op), (b_pos, b_op) in zip(work_events, work_events[1:]):
            a_quadrant = (a_pos[0] >= 5) + 2 * (a_pos[1] >= 5)
            b_quadrant = (b_pos[0] >= 5) + 2 * (b_pos[1] >= 5)
            if a_quadrant != b_quadrant:
                quadrant_transition_ops[f"{a_op}->{b_op}"] += 1
        if work_positions:
            xs = [p[0] for p in work_positions]
            ys = [p[1] for p in work_positions]
            span = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        else:
            span = 0
        row = {
            "day": day + 1,
            "unit": unit,
            "moves": moves,
            "work": work,
            "passes": passes,
            "moves_per_work": moves / max(1, work),
            "unique_work_cells": len(set(work_positions)),
            "work_bbox_area": span,
            "mean_work_cell_jump": statistics.fmean(work_jumps) if work_jumps else 0.0,
            "quadrant_switches": quadrant_switches,
        }
        rendered.append(row)
        totals.update(moves=moves, work=work, passes=passes, quadrant_switches=quadrant_switches)

    active = [row for row in rendered if row["work"] > 0]
    return {
        "totals": dict(totals),
        "moves_per_work": totals["moves"] / max(1, totals["work"]),
        "active_worker_days": len(active),
        "mean_worker_day_moves_per_work": statistics.fmean(row["moves_per_work"] for row in active),
        "median_worker_day_moves_per_work": statistics.median(row["moves_per_work"] for row in active),
        "mean_worker_day_unique_work_cells": statistics.fmean(row["unique_work_cells"] for row in active),
        "mean_worker_day_bbox_area": statistics.fmean(row["work_bbox_area"] for row in active),
        "mean_worker_day_work_cell_jump": statistics.fmean(row["mean_work_cell_jump"] for row in active),
        "top_quadrant_transition_ops": [
            {"transition": transition, "count": count}
            for transition, count in quadrant_transition_ops.most_common(24)
        ],
        "worker_days": rendered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-replay", type=Path, required=True)
    parser.add_argument("--official-seat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--native-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    official = summarize(official_rows(args.official_replay, args.official_seat))
    native = summarize(native_rows(args.native_trace))
    payload = {
        "schema": "kaggriculture.worker-route-profile-comparison.v1",
        "official_replay": str(args.official_replay),
        "official_seat": args.official_seat,
        "native_trace": str(args.native_trace),
        "official": official,
        "native": native,
        "delta_native_minus_official": {
            key: native[key] - official[key]
            for key in (
                "moves_per_work",
                "mean_worker_day_moves_per_work",
                "median_worker_day_moves_per_work",
                "mean_worker_day_unique_work_cells",
                "mean_worker_day_bbox_area",
                "mean_worker_day_work_cell_jump",
            )
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    compact = {key: value for key, value in payload.items() if key not in {"official", "native"}}
    compact["official"] = {key: value for key, value in official.items() if key != "worker_days"}
    compact["native"] = {key: value for key, value in native.items() if key != "worker_days"}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
