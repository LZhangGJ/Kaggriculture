from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kaggriculture_jax.constants import NUM_CROPS, TileKind, UnitOp
from strategic_v5.constants import TaskStatusV1, TaskTypeV1


CROP_NAMES = ("wheat", "carrot", "tomato", "strawberry", "melon")
TURNS_PER_DAY = 24


def _crop_counts(trace: np.lib.npyio.NpzFile, row: int, player: int) -> list[int]:
    crop = trace["state_tile_crop"][row, 0, player]
    return [int(np.sum(crop == crop_id)) for crop_id in range(NUM_CROPS)]


def analyze(trace_path: Path) -> dict:
    trace = np.load(trace_path, allow_pickle=False)
    player = int(trace["player"][0])
    steps = int(trace["state_step"].shape[0])
    task_type = trace["internal_unit_task_task_type"][:, 0]
    task_status = trace["internal_unit_task_status"][:, 0]
    task_start = trace["internal_unit_task_start_step"][:, 0]
    task_target = trace["internal_unit_task_target_id"][:, 0]
    task_item = trace["internal_unit_task_item_id"][:, 0]
    task_phase = trace["internal_unit_task_phase"][:, 0]
    task_last_progress = trace["internal_unit_task_last_progress_step"][:, 0]
    action_op = trace["action_unit_op"][:, 0]
    action_item = trace["action_unit_item"][:, 0]
    preemption = trace["internal_scheduler_deadline_preemption_count"][:, 0]

    crop_task_mask = (
        (task_type == int(TaskTypeV1.CROP_PRODUCTION))
        & (task_status == int(TaskStatusV1.ACTIVE))
    )
    water_task_mask = (
        (task_type == int(TaskTypeV1.WATER_CROP))
        & (task_status == int(TaskStatusV1.ACTIVE))
    )
    instances: dict[tuple[int, int, int, int, int], dict] = {}
    for row in range(steps):
        state_step = int(trace["state_step"][row, 0])
        relevant = crop_task_mask[row] | water_task_mask[row]
        for unit in np.flatnonzero(relevant):
            key = (
                int(unit),
                int(task_start[row, unit]),
                int(task_target[row, unit]),
                int(task_item[row, unit]),
                int(task_type[row, unit]),
            )
            target_id = int(task_target[row, unit])
            target_y, target_x = divmod(max(target_id, 0), 10)
            target_y = int(np.clip(target_y, 0, 9))
            target_x = int(np.clip(target_x, 0, 9))
            initial_kind = int(
                trace["state_tile_kind"][row, 0, player, target_y, target_x]
            )
            initial_yield = int(
                trace["state_tile_yield"][row, 0, player, target_y, target_x]
            )
            initial_intent = (
                "PLANT"
                if initial_kind == int(TileKind.EMPTY)
                else "HARVEST"
                if initial_kind == int(TileKind.PLANT) and initial_yield > 0
                else "MAINTAIN"
            )
            record = instances.setdefault(
                key,
                {
                    "unit": int(unit),
                    "owner_kind": "farmer" if unit == 0 else "hand",
                    "task_type": (
                        "CROP_PRODUCTION"
                        if task_type[row, unit] == int(TaskTypeV1.CROP_PRODUCTION)
                        else "WATER_CROP"
                    ),
                    "crop_id": int(np.clip(task_item[row, unit], 0, NUM_CROPS - 1)),
                    "crop": CROP_NAMES[
                        int(np.clip(task_item[row, unit], 0, NUM_CROPS - 1))
                    ],
                    "target_id": int(task_target[row, unit]),
                    "initial_target_kind": initial_kind,
                    "initial_target_yield": initial_yield,
                    "initial_intent": initial_intent,
                    "start_step": int(task_start[row, unit]),
                    "first_observed_step": state_step,
                    "last_observed_step": state_step,
                    "last_progress_step": int(task_last_progress[row, unit]),
                    "observed_rows": 0,
                    "plant_actions": 0,
                    "water_actions": 0,
                    "harvest_actions": 0,
                    "drop_actions": 0,
                    "phases_observed": [],
                },
            )
            record["last_observed_step"] = state_step
            record["last_progress_step"] = max(
                record["last_progress_step"], int(task_last_progress[row, unit])
            )
            record["observed_rows"] += 1
            phase = int(task_phase[row, unit])
            if phase not in record["phases_observed"]:
                record["phases_observed"].append(phase)
            record["plant_actions"] += int(action_op[row, unit] == int(UnitOp.PLANT))
            record["water_actions"] += int(action_op[row, unit] == int(UnitOp.WATER))
            record["harvest_actions"] += int(action_op[row, unit] == int(UnitOp.HARVEST))
            record["drop_actions"] += int(action_op[row, unit] == int(UnitOp.DROP))

    instance_rows = sorted(
        instances.values(), key=lambda item: (item["first_observed_step"], item["unit"])
    )
    for record in instance_rows:
        record["no_effect_action"] = bool(
            record["plant_actions"]
            + record["water_actions"]
            + record["harvest_actions"]
            + record["drop_actions"]
            == 0
        )
        record["plant_without_water_in_same_task"] = bool(
            record["task_type"] == "CROP_PRODUCTION"
            and record["plant_actions"] > 0
            and record["water_actions"] == 0
        )
        record["ended_at_or_after_day_reset"] = bool(
            record["last_observed_step"] // TURNS_PER_DAY
            > record["start_step"] // TURNS_PER_DAY
        )

    daily = []
    for day in range(30):
        first = min(day * TURNS_PER_DAY, steps - 1)
        last = min((day + 1) * TURNS_PER_DAY - 1, steps - 1)
        sl = slice(first, last + 1)
        started = [
            row
            for row in instance_rows
            if day * TURNS_PER_DAY <= row["start_step"] < (day + 1) * TURNS_PER_DAY
        ]
        started_production = [
            row for row in started if row["task_type"] == "CROP_PRODUCTION"
        ]
        started_water = [
            row for row in started if row["task_type"] == "WATER_CROP"
        ]
        started_plant_routes = [
            row
            for row in started_production
            if row["initial_intent"] == "PLANT"
        ]
        daily.append(
            {
                "day": day + 1,
                "start_step": int(trace["state_step"][first, 0]),
                "end_step": int(trace["state_step"][last, 0]),
                "money_start": int(trace["state_money"][first, 0, player]),
                "money_end": int(trace["state_money"][last, 0, player]),
                "active_crop_start": _crop_counts(trace, first, player),
                "active_crop_end": _crop_counts(trace, last, player),
                "seed_start": [
                    int(x) for x in trace["state_seeds"][first, 0, player]
                ],
                "seed_end": [int(x) for x in trace["state_seeds"][last, 0, player]],
                "new_crop_task_instances": len(started),
                "new_production_task_instances": len(started_production),
                "new_water_task_instances": len(started_water),
                "new_plant_route_instances": len(started_plant_routes),
                "new_crop_tasks_without_effect_action": int(
                    sum(row["no_effect_action"] for row in started)
                ),
                "new_production_tasks_without_effect_action": int(
                    sum(row["no_effect_action"] for row in started_production)
                ),
                "new_plant_routes_without_plant_action": int(
                    sum(row["plant_actions"] == 0 for row in started_plant_routes)
                ),
                "plant_actions": int(np.sum(action_op[sl] == int(UnitOp.PLANT))),
                "water_actions": int(np.sum(action_op[sl] == int(UnitOp.WATER))),
                "harvest_actions": int(np.sum(action_op[sl] == int(UnitOp.HARVEST))),
                "deadline_preemptions": int(np.sum(preemption[sl])),
                "steps_with_deadline_preemption": int(np.sum(preemption[sl] > 0)),
            }
        )

    production = [
        row for row in instance_rows if row["task_type"] == "CROP_PRODUCTION"
    ]
    return {
        "schema": "m37c_crop_pipeline_diagnostic_v1",
        "trace": str(trace_path.resolve()),
        "player": player,
        "summary": {
            "crop_task_instance_count": len(instance_rows),
            "production_task_instance_count": len(production),
            "production_task_without_effect_action_count": int(
                sum(row["no_effect_action"] for row in production)
            ),
            "plant_without_water_in_same_task_count": int(
                sum(row["plant_without_water_in_same_task"] for row in production)
            ),
            "deadline_preemption_count": int(np.sum(preemption)),
            "steps_with_deadline_preemption": int(np.sum(preemption > 0)),
            "total_plant_actions": int(np.sum(action_op == int(UnitOp.PLANT))),
            "total_water_actions": int(np.sum(action_op == int(UnitOp.WATER))),
        },
        "daily": daily,
        "task_instances": instance_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("artifacts/traces/m37b_emergency_feed_trace_v7.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("receipts/m37c_crop_pipeline_diagnostic_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
