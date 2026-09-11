from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kaggriculture_jax.constants import NUM_PRODUCTS, TileKind, UnitOp
from strategic_v5.constants import TaskStatusV1, TaskTypeV1


ANIMAL_NAMES = ("chicken", "cow", "sheep")
STRUCTURE_KIND = (
    int(TileKind.COOP),
    int(TileKind.PASTURE),
    int(TileKind.PASTURE),
)
TURNS_PER_DAY = 24


def _scalar(array: np.ndarray, step: int) -> int:
    return int(array[step, 0])


def _state_counts(trace: np.lib.npyio.NpzFile, step: int, player: int) -> dict:
    animals = trace["state_tile_animal"][step, 0, player]
    kinds = trace["state_tile_kind"][step, 0, player]
    inventory = trace["state_unit_inventory"][step, 0, player]
    shed = trace["state_shed"][step, 0, player]
    active = trace["state_unit_active"][step, 0, player]
    return {
        "step": int(trace["state_step"][step, 0]),
        "money": int(trace["state_money"][step, 0, player]),
        "active_animals": [int(np.sum(animals == i)) for i in range(3)],
        "shed_animals": [int(shed[NUM_PRODUCTS + i]) for i in range(3)],
        "carried_animals": [
            int(np.sum(inventory[:, NUM_PRODUCTS + i])) for i in range(3)
        ],
        "empty_structures": [
            int(np.sum((kinds == STRUCTURE_KIND[i]) & (animals < 0)))
            for i in range(3)
        ],
        "active_units": int(np.sum(active)),
    }


def analyze(trace_path: Path) -> dict:
    trace = np.load(trace_path, allow_pickle=False)
    player = int(trace["player"][0])
    steps = int(trace["state_step"].shape[0])
    task_type = trace["internal_unit_task_task_type"][:, 0]
    task_status = trace["internal_unit_task_status"][:, 0]
    task_active = (task_type == int(TaskTypeV1.ANIMAL_PLACE)) & (
        task_status == int(TaskStatusV1.ACTIVE)
    )
    task_start = trace["internal_unit_task_start_step"][:, 0]
    task_target = trace["internal_unit_task_target_id"][:, 0]
    task_item = trace["internal_unit_task_item_id"][:, 0]
    task_expected = trace["internal_unit_task_expected_finish_step"][:, 0]
    task_last_progress = trace["internal_unit_task_last_progress_step"][:, 0]
    action_op = trace["action_unit_op"][:, 0]
    action_item = trace["action_unit_item"][:, 0]

    instances: dict[tuple[int, int, int, int], dict] = {}
    for row in range(steps):
        state_step = int(trace["state_step"][row, 0])
        day_end = ((state_step // TURNS_PER_DAY) + 1) * TURNS_PER_DAY - 1
        for unit in np.flatnonzero(task_active[row]):
            key = (
                int(unit),
                int(task_start[row, unit]),
                int(task_target[row, unit]),
                int(task_item[row, unit]),
            )
            record = instances.setdefault(
                key,
                {
                    "unit": int(unit),
                    "owner_kind": "farmer" if unit == 0 else "hand",
                    "start_step": int(task_start[row, unit]),
                    "first_observed_step": state_step,
                    "last_observed_step": state_step,
                    "target_id": int(task_target[row, unit]),
                    "item_id": int(task_item[row, unit]),
                    "species": ANIMAL_NAMES[
                        int(np.clip(task_item[row, unit] - NUM_PRODUCTS, 0, 2))
                    ],
                    "initial_expected_finish_step": int(task_expected[row, unit]),
                    "initial_day_end": day_end,
                    "initial_route_exceeds_day_end": bool(
                        # expected_finish is the post-action state index.  A
                        # terminal PLACE on action ``day_end`` appears at
                        # state ``day_end + 1`` and is still feasible.
                        task_expected[row, unit] > day_end + 1
                    ),
                    "observed_rows": 0,
                    "pickup_actions": 0,
                    "place_actions": 0,
                    "last_progress_step": int(task_last_progress[row, unit]),
                },
            )
            record["last_observed_step"] = state_step
            record["observed_rows"] += 1
            record["last_progress_step"] = max(
                record["last_progress_step"], int(task_last_progress[row, unit])
            )
            if action_op[row, unit] == int(UnitOp.PICKUP):
                record["pickup_actions"] += 1
            if action_op[row, unit] == int(UnitOp.PLACE):
                record["place_actions"] += 1

    instance_rows = sorted(
        instances.values(), key=lambda row: (row["first_observed_step"], row["unit"])
    )
    for record in instance_rows:
        last = record["last_observed_step"]
        record["cleared_at_day_boundary"] = bool(
            last % TURNS_PER_DAY == 0
            and record["place_actions"] == 0
        )

    daily = []
    for day in range(30):
        first_row = min(day * TURNS_PER_DAY, steps - 1)
        last_row = min((day + 1) * TURNS_PER_DAY - 1, steps - 1)
        start = _state_counts(trace, first_row, player)
        end = _state_counts(trace, last_row, player)
        day_slice = slice(first_row, last_row + 1)
        daily.append(
            {
                "day": day + 1,
                "start": start,
                "end": end,
                "place_task_rows": int(np.sum(task_active[day_slice])),
                "pickup_actions": int(
                    np.sum(action_op[day_slice] == int(UnitOp.PICKUP))
                ),
                "place_actions": int(
                    np.sum(action_op[day_slice] == int(UnitOp.PLACE))
                ),
            }
        )

    animal_events = []
    previous = _state_counts(trace, 0, player)
    for row in range(1, steps):
        current = _state_counts(trace, row, player)
        changed = any(
            current[field] != previous[field]
            for field in ("active_animals", "shed_animals", "carried_animals")
        )
        if changed:
            animal_events.append(
                {
                    "step": current["step"],
                    "active_animals": current["active_animals"],
                    "shed_animals": current["shed_animals"],
                    "carried_animals": current["carried_animals"],
                    "action_previous_step": {
                        "pickup_units": [
                            int(i)
                            for i in np.flatnonzero(
                                action_op[row - 1] == int(UnitOp.PICKUP)
                            )
                            if action_item[row - 1, i] >= NUM_PRODUCTS
                        ],
                        "place_units": [
                            int(i)
                            for i in np.flatnonzero(
                                action_op[row - 1] == int(UnitOp.PLACE)
                            )
                        ],
                    },
                }
            )
        previous = current

    hand_instances = [row for row in instance_rows if row["owner_kind"] == "hand"]
    infeasible = [row for row in hand_instances if row["initial_route_exceeds_day_end"]]
    boundary_clears = [row for row in hand_instances if row["cleared_at_day_boundary"]]
    return {
        "schema": "m37b_animal_pipeline_diagnostic_v1",
        "trace": str(trace_path.resolve()),
        "player": player,
        "state_rows": steps,
        "summary": {
            "place_task_instance_count": len(instance_rows),
            "place_task_with_terminal_place_action_count": int(
                sum(row["place_actions"] > 0 for row in instance_rows)
            ),
            "place_task_without_terminal_place_action_count": int(
                sum(row["place_actions"] == 0 for row in instance_rows)
            ),
            "hand_place_task_instance_count": len(hand_instances),
            "hand_route_exceeds_day_end_count": len(infeasible),
            "hand_cleared_at_day_boundary_without_place_count": len(boundary_clears),
            "max_observed_place_task_rows": max(
                (row["observed_rows"] for row in instance_rows), default=0
            ),
        },
        "daily": daily,
        "place_task_instances": instance_rows,
        "animal_state_change_events": animal_events,
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
        default=Path("receipts/m37b_animal_pipeline_diagnostic_v1.json"),
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
