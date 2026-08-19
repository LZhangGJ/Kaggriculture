"""Audit idle unit slots while executable farm obligations remain."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = PROJECT / "artifacts" / "traces" / "m37c_calendar_replan_cash_trace_v14.npz"
DEFAULT_OUTPUT = PROJECT / "receipts" / "m37c_idle_capacity_diagnostic_v14.json"

UNIT_OP_NAMES = (
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP", "PICKUP",
    "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.trace, allow_pickle=False) as trace:
        player = int(np.asarray(trace["player"])[0])
        rows = []
        for action_step in range(1, 719):
            # state[action_step-1] is the state immediately before this action.
            pre_index = action_step - 1
            kind = np.asarray(trace["state_tile_kind"][pre_index, 0, player])
            crop = np.asarray(trace["state_tile_crop"][pre_index, 0, player])
            animal = np.asarray(trace["state_tile_animal"][pre_index, 0, player])
            flags = np.asarray(trace["state_tile_flags"][pre_index, 0, player])
            tile_yield = np.asarray(trace["state_tile_yield"][pre_index, 0, player])
            neglect = np.asarray(trace["state_tile_neglect"][pre_index, 0, player])
            active_units = int(np.asarray(trace["state_unit_active"][pre_index, 0, player]).sum())
            unit_positions = np.asarray(
                trace["state_unit_pos"][pre_index, 0, player, :active_units]
            ).astype(int)
            action_count = int(trace["action_unit_count"][action_step, 0])
            ops = np.asarray(trace["action_unit_op"][action_step, 0, :action_count]).astype(int)
            op_counts = Counter(UNIT_OP_NAMES[value] for value in ops.tolist())
            task_status = np.asarray(trace["internal_unit_task_status"][action_step, 0])
            task_type = np.asarray(trace["internal_unit_task_task_type"][action_step, 0])
            active_task_types = task_type[task_status == 1].astype(int)
            task_counts = Counter(str(value) for value in active_task_types.tolist())
            unwatered = int(((kind == 3) & ((flags & 1) == 0)).sum())
            harvestable_crop = int(((kind == 3) & (tile_yield > 0)).sum())
            survival_feed = int(((animal >= 0) & (neglect >= 1) & ((flags & 2) == 0)).sum())
            fertilizer_ready = int(((animal >= 0) & ((flags & 8) != 0)).sum())
            pass_count = int(op_counts["PASS"])
            unwatered_positions = np.argwhere(
                (kind == 3) & ((flags & 1) == 0)
            ).astype(int)
            pass_unit_ids = np.flatnonzero(ops == 0).astype(int)
            pass_min_unwatered_distance = []
            for unit_id in pass_unit_ids.tolist():
                if unwatered_positions.size == 0:
                    pass_min_unwatered_distance.append(0)
                    continue
                # State positions are stored as (x, y); np.argwhere returns
                # tile coordinates as (y, x).
                ux, uy = unit_positions[unit_id]
                distance = np.abs(unwatered_positions[:, 0] - uy) + np.abs(
                    unwatered_positions[:, 1] - ux
                )
                pass_min_unwatered_distance.append(int(distance.min()))
            rows.append(
                {
                    "action_step": action_step,
                    "day": action_step // 24 + 1,
                    "hour": action_step % 24,
                    "active_units": active_units,
                    "pass_count": pass_count,
                    "active_task_count": int(active_task_types.size),
                    "unwatered_crop_count": unwatered,
                    "harvestable_crop_count": harvestable_crop,
                    "survival_feed_count": survival_feed,
                    "fertilizer_ready_count": fertilizer_ready,
                    "idle_while_unwatered": pass_count > 0 and unwatered > 0,
                    "unit_positions_xy": unit_positions.tolist(),
                    "pass_unit_ids": pass_unit_ids.tolist(),
                    "pass_min_unwatered_distance": pass_min_unwatered_distance,
                    "unwatered_positions_yx": unwatered_positions.tolist(),
                    "task_target_xy": np.stack(
                        (
                            np.asarray(trace["internal_unit_task_target_x"][action_step, 0, :active_units]),
                            np.asarray(trace["internal_unit_task_target_y"][action_step, 0, :active_units]),
                        ),
                        axis=-1,
                    ).astype(int).tolist(),
                    "unit_ops": dict(sorted(op_counts.items())),
                    "active_task_types": dict(sorted(task_counts.items())),
                    "scheduler": {
                        name: int(trace[f"internal_scheduler_{name}"][action_step, 0])
                        for name in (
                            "sticky_task_count",
                            "crop_candidate_count",
                            "animal_candidate_count",
                            "hard_candidate_count",
                            "selected_crop_count",
                            "selected_animal_count",
                            "deadline_preemption_count",
                        )
                    },
                }
            )

    daily = []
    for day in range(1, 31):
        selected = [row for row in rows if row["day"] == day]
        if not selected:
            continue
        daily.append(
            {
                "day": day,
                "idle_unit_slots": sum(row["pass_count"] for row in selected),
                "idle_while_unwatered_steps": sum(row["idle_while_unwatered"] for row in selected),
                "idle_while_unwatered_unit_slots": sum(
                    row["pass_count"] for row in selected if row["idle_while_unwatered"]
                ),
                "max_unwatered": max(row["unwatered_crop_count"] for row in selected),
                "min_active_task_count": min(row["active_task_count"] for row in selected),
                "max_active_units": max(row["active_units"] for row in selected),
            }
        )
    output = {
        "schema": "kaggriculture.m37c_idle_capacity_diagnostic.v1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "trace": args.trace.as_posix(),
        "summary": {
            "idle_while_unwatered_steps": sum(row["idle_while_unwatered"] for row in rows),
            "idle_while_unwatered_unit_slots": sum(
                row["pass_count"] for row in rows if row["idle_while_unwatered"]
            ),
        },
        "daily": daily,
        "steps": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": output["summary"], "days_1_to_12": daily[:12]}, indent=2))


if __name__ == "__main__":
    main()
