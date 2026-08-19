"""Formal M3.6B acceptance for complete 30-day gold-route calendars."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import jax
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
for source in (
    EXPERIMENT_ROOT / "src",
    REPO_ROOT / "gpu_sim" / "src",
    REPO_ROOT / "experiments" / "strategic_v5" / "src",
):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from kaggriculture_jax.constants import ANIMALS, CROPS  # noqa: E402
from project_route_search_v2.m36_calendar import validate_route_calendar_v3  # noqa: E402
from project_route_search_v2.m36_replay_calendar import (  # noqa: E402
    compile_gold_replay_calendar_v3,
    stack_route_calendars_v3,
)


DEFAULT_REPLAY_ROOT = (
    REPO_ROOT
    / "replay"
    / "gold_top20_latest_2026-08-18_082723"
    / "latest_per_gold"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "receipts" / "m36b_gold_calendar_acceptance_v1.json"

ROUTES = (
    ("KAWASHIGI_6C12S_4LAND", 94051618, 1),
    ("TETSUYA_11C4S_3LAND", 94052517, 1),
    ("RECURSION_6C7S_3LAND", 94054256, 1),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _direct_source_audit(replay: Mapping[str, Any], player: int) -> dict[str, Any]:
    crop_id = {name: index for index, name in enumerate(CROPS)}
    animal_id = {name: index for index, name in enumerate(ANIMALS)}
    day6_crop_peak = np.zeros((len(CROPS),), dtype=np.int16)
    day6_service_end = np.zeros((len(ANIMALS),), dtype=np.int16)
    requested_animals = np.zeros((len(ANIMALS),), dtype=np.int32)
    last_day6_hour = -1
    first_land = None
    last_land = None
    observed_days: set[int] = set()

    for step in replay["steps"]:
        row = step[player]
        observation = row["observation"]
        day = int(observation["day"])
        hour = int(observation["hour"])
        observed_days.add(day)
        farm = observation["farms"][player]
        land = len(farm.get("unlocked_quadrants", ()))
        first_land = land if first_land is None else first_land
        last_land = land

        if day == 6:
            current_crops = np.zeros_like(day6_crop_peak)
            current_service = np.zeros_like(day6_service_end)
            for tile_row in farm.get("tiles", ()):
                for tile in tile_row:
                    if not isinstance(tile, dict):
                        continue
                    crop = crop_id.get(str(tile.get("crop")))
                    animal = animal_id.get(str(tile.get("animal")))
                    if tile.get("kind") == "PLANT" and crop is not None:
                        current_crops[crop] += 1
                    if animal is not None:
                        current_service[animal] += 1
            day6_crop_peak = np.maximum(day6_crop_peak, current_crops)
            if hour >= last_day6_hour:
                day6_service_end = current_service
                for inventory in observation.get("private", {}).get("inventories", ()):
                    for name, index in animal_id.items():
                        current_service[index] += int(inventory.get(name, 0) or 0)
                day6_service_end = current_service.copy()
                last_day6_hour = hour

        for order in (row.get("action", {}) or {}).get("market", ()):
            if isinstance(order, list) and len(order) >= 3 and order[0] == "BUY_ANIMAL":
                index = animal_id.get(str(order[1]))
                if index is not None:
                    requested_animals[index] += int(order[2])

    return {
        "observed_days": len(observed_days),
        "requested_animal_totals": requested_animals.tolist(),
        "land_addition_count": int((last_land or 0) - (first_land or 0)),
        "day6_crop_peak": day6_crop_peak.tolist(),
        "day6_service_end": day6_service_end.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    route_rows = []
    calendars = []
    for index, (route_name, episode, player) in enumerate(ROUTES):
        path = args.replay_root / f"episode-{episode}-replay.json"
        replay = json.loads(path.read_text(encoding="utf-8"))
        calendar, diagnostics = compile_gold_replay_calendar_v3(
            replay, player=player, candidate_id=3_600 + index
        )
        errors = validate_route_calendar_v3(calendar)
        direct = _direct_source_audit(replay, player)
        diagnostic = diagnostics.to_dict()
        roundtrip = {
            key: np.asarray(diagnostic[key]).tolist() == np.asarray(direct[key]).tolist()
            for key in (
                "observed_days",
                "requested_animal_totals",
                "land_addition_count",
                "day6_crop_peak",
                "day6_service_end",
            )
        }
        route_rows.append(
            {
                "route_name": route_name,
                "source": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "source_sha256": _sha256(path),
                "module_version": replay.get("module_version"),
                "diagnostics": diagnostic,
                "direct_source_audit": direct,
                "roundtrip_checks": roundtrip,
                "calendar_validation_errors": errors,
                "event_overflow_count": int(np.asarray(calendar.event_overflow_count)[0]),
                "calendar_field_names": list(calendar._fields),
            }
        )
        calendars.append(calendar)

    stacked = stack_route_calendars_v3(calendars)
    stacked_errors = validate_route_calendar_v3(stacked)
    signatures = [row["diagnostics"]["route_signature"] for row in route_rows]
    gates = {
        "three_structurally_distinct_gold_routes": len(set(signatures)) == 3,
        "all_30_days_observed": all(
            row["diagnostics"]["observed_days"] == 30 for row in route_rows
        ),
        "all_route_calendars_valid": all(
            not row["calendar_validation_errors"] for row in route_rows
        ),
        "stacked_calendar_valid": not stacked_errors,
        "event_overflow_count_zero": all(
            row["event_overflow_count"] == 0 for row in route_rows
        ),
        "day6_production_scale_roundtrip_exact": all(
            row["roundtrip_checks"]["day6_crop_peak"]
            and row["roundtrip_checks"]["day6_service_end"]
            for row in route_rows
        ),
        "all_compiled_fields_roundtrip_exact": all(
            all(row["roundtrip_checks"].values()) for row in route_rows
        ),
        "no_raw_action_payload_in_calendar": all(
            not row["diagnostics"]["raw_action_payload_stored"]
            and not any("action" in field for field in row["calendar_field_names"])
            for row in route_rows
        ),
    }
    receipt = {
        "schema": "kaggriculture.m36b_gold_calendar_acceptance.v1",
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "scope": "M3.6B calendar representation and gold Replay compilation",
        "boundary": {
            "proven": "Three structurally different official gold Replays compile into complete 30-day high-level calendars without truncation; compiled day-6 targets exactly match an independent source audit.",
            "not_proven": "The controller can execute those calendars to the same day-6 or terminal state. That behavioral gate belongs to M3.6C/D.",
            "raw_replay_action_playback": False,
        },
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "calendar_shapes": {
            name: list(np.asarray(getattr(stacked, name)).shape)
            for name in stacked._fields
        },
        "route_signatures": signatures,
        "routes": route_rows,
        "stacked_validation_errors": stacked_errors,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"decision": receipt["decision"], "gates": gates, "output": str(args.output)}, ensure_ascii=False))
    if receipt["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
