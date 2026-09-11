from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kaggriculture_jax.constants import FLAG_FED, UnitOp
from strategic_v5.constants import TaskStatusV1, TaskTypeV1


ANIMAL_TASKS = {
    int(TaskTypeV1.BUILD_ANIMAL_STRUCTURE),
    int(TaskTypeV1.ANIMAL_PLACE),
    int(TaskTypeV1.ANIMAL_FEED),
    int(TaskTypeV1.ANIMAL_CARE),
    int(TaskTypeV1.ANIMAL_COLLECT_PRODUCT),
    int(TaskTypeV1.ANIMAL_COLLECT_FERTILIZER),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=719)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = np.load(args.trace, allow_pickle=False)
    player = int(trace["player"][0])
    rows = []
    for row in range(trace["state_step"].shape[0]):
        step = int(trace["state_step"][row, 0])
        if not args.start <= step <= args.end:
            continue
        animal = trace["state_tile_animal"][row, 0, player]
        neglect = trace["state_tile_neglect"][row, 0, player]
        flags = trace["state_tile_flags"][row, 0, player]
        survival = (
            (animal >= 0)
            & (neglect == 1)
            & ((flags & int(FLAG_FED)) == 0)
        )
        task_type = trace["internal_unit_task_task_type"][row, 0]
        task_status = trace["internal_unit_task_status"][row, 0]
        active_task = task_status == int(TaskStatusV1.ACTIVE)
        active_unit = trace["state_unit_active"][row, 0, player]
        animal_task = np.isin(task_type, tuple(ANIMAL_TASKS)) & active_task
        crop_task = active_task & ~animal_task
        feed_task = active_task & (task_type == int(TaskTypeV1.ANIMAL_FEED))
        op = trace["action_unit_op"][row, 0]
        survival_target_ids = [
            int(y * 10 + x) for y, x in np.argwhere(survival)
        ]
        feed_task_details = []
        for unit in np.flatnonzero(feed_task):
            feed_task_details.append(
                {
                    "unit": int(unit),
                    "target_id": int(
                        trace["internal_unit_task_target_id"][row, 0, unit]
                    ),
                    "phase": int(
                        trace["internal_unit_task_phase"][row, 0, unit]
                    ),
                    "start_step": int(
                        trace["internal_unit_task_start_step"][row, 0, unit]
                    ),
                    "unit_pos": [
                        int(value)
                        for value in trace["state_unit_pos"][
                            row, 0, player, unit
                        ]
                    ],
                    "action_op": int(op[unit]),
                }
            )
        active_task_details = [
            {
                "unit": int(unit),
                "task_type": int(task_type[unit]),
                "target_id": int(
                    trace["internal_unit_task_target_id"][row, 0, unit]
                ),
                "item_id": int(
                    trace["internal_unit_task_item_id"][row, 0, unit]
                ),
                "phase": int(
                    trace["internal_unit_task_phase"][row, 0, unit]
                ),
            }
            for unit in np.flatnonzero(active_task)
        ]
        rows.append(
            {
                "step": step,
                "day": step // 24 + 1,
                "hour": step % 24,
                "survival_due": int(np.sum(survival)),
                "survival_due_by_species": [
                    int(np.sum(survival & (animal == species)))
                    for species in range(3)
                ],
                "survival_target_ids": survival_target_ids,
                "active_units": int(np.sum(active_unit)),
                "active_tasks": int(np.sum(active_task)),
                "active_task_details": active_task_details,
                "free_units": int(np.sum(active_unit & ~active_task)),
                "crop_tasks": int(np.sum(crop_task)),
                "animal_tasks": int(np.sum(animal_task)),
                "feed_tasks": int(np.sum(feed_task)),
                "feed_task_details": feed_task_details,
                "feed_actions": int(np.sum(op == int(UnitOp.FEED))),
                "deadline_preemption_diagnostic": int(
                    trace["internal_scheduler_deadline_preemption_count"][row, 0]
                ),
                "wheat_held": int(trace["state_shed"][row, 0, player, 0])
                + int(
                    np.sum(
                        trace["state_unit_inventory"][row, 0, player, :, 0]
                    )
                ),
            }
        )
    summary = {
        "rows": len(rows),
        "steps_with_survival_due": int(sum(row["survival_due"] > 0 for row in rows)),
        "steps_with_survival_due_but_no_feed_task": int(
            sum(
                row["survival_due"] > 0 and row["feed_tasks"] == 0
                for row in rows
            )
        ),
        "max_simultaneous_survival_due": max(
            (row["survival_due"] for row in rows), default=0
        ),
    }
    result = {
        "schema": "m37c_preemption_window_v1",
        "trace": str(args.trace.resolve()),
        "player": player,
        "window": [args.start, args.end],
        "summary": summary,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in rows:
        if row["survival_due"] > 0 or row["feed_tasks"] > 0 or row["feed_actions"] > 0:
            print(
                row["step"],
                row["survival_due"],
                row["feed_tasks"],
                row["crop_tasks"],
                row["free_units"],
                row["feed_actions"],
                row["deadline_preemption_diagnostic"],
                row["wheat_held"],
            )


if __name__ == "__main__":
    main()
