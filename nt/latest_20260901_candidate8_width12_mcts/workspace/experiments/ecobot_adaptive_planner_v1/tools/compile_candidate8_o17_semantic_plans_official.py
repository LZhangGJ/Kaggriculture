#!/usr/bin/env python3
"""Compile O1.7 reference traces into action-free semantic plan profiles.

This script must run in the Windows environment that contains the frozen
official ``kaggle-environments==1.32.7`` package.  It replays each temporary
trace, verifies native/official reward parity, and writes three plans:

* ``full``: all daily semantic targets, task flows and transaction windows;
* ``target``: 30-day state targets only;
* ``milestone``: day 0/6/12/18/29 target snapshots held piecewise.

None of the NPZ files contains raw actions, coordinates, opponent identity or
future random events.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[3]
REFERENCE_COMPILER = (
    ROOT
    / "experiments"
    / "general_project_planner_v1"
    / "tools"
    / "compile_reference_obligation_plan.py"
)
ECONOMIC_COMPILER = (
    ROOT
    / "experiments"
    / "ecobot_adaptive_planner_v1"
    / "tools"
    / "compile_g001_backbone_plan.py"
)
FLOW_FIELDS = (
    "crop_plant_by_day",
    "crop_water_by_day",
    "crop_harvest_by_day",
    "crop_fertilize_by_day",
    "crop_clear_by_day",
    "animal_feed_by_day",
    "animal_care_by_day",
    "animal_product_by_day",
    "animal_fertilizer_by_day",
)
TARGET_FIELDS = (
    "hand_target_by_day",
    "unlocked_target_by_day",
    "crop_target_by_day",
    "animal_owned_target_by_day",
    "animal_service_target_by_day",
    "wheat_buffer_by_day",
)
MILESTONE_DAYS = (0, 6, 12, 18, 29)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _clone(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value).copy() for name, value in arrays.items()}


def _enrich(arrays: dict[str, np.ndarray]) -> None:
    animal_owned = np.asarray(arrays["animal_owned_target_by_day"], dtype=np.int16)
    animal_additions = np.maximum(
        animal_owned
        - np.concatenate(
            (np.zeros_like(animal_owned[:, :1]), animal_owned[:, :-1]), axis=1
        ),
        0,
    ).astype(np.int16)
    unlocked = np.asarray(arrays["unlocked_target_by_day"], dtype=np.int16)
    land_additions = np.maximum(
        unlocked
        - np.concatenate((np.ones_like(unlocked[:, :1]), unlocked[:, :-1]), axis=1),
        0,
    ).astype(np.int8)
    arrays["animal_purchase_additions_by_day"] = animal_additions
    arrays["land_additions_by_day"] = land_additions
    arrays["plan_mode"] = np.asarray([0], dtype=np.int8)
    arrays["schema_version"] = np.asarray([1], dtype=np.int16)


def _target_only(full: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = _clone(full)
    for name in FLOW_FIELDS:
        out[name] = np.zeros_like(out[name])
    for name in (
        "sell_first_hour_by_day",
        "sell_last_hour_by_day",
        "buy_first_hour_by_day",
        "buy_last_hour_by_day",
    ):
        out[name] = np.full_like(out[name], -1)
    out["terminal_liquidation_start_step"] = np.asarray([648], dtype=np.int16)
    _enrich(out)
    return out


def _milestone_only(target: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = _clone(target)
    for name in TARGET_FIELDS:
        original = np.asarray(target[name])
        compressed = np.empty_like(original)
        for index, start in enumerate(MILESTONE_DAYS):
            stop = MILESTONE_DAYS[index + 1] if index + 1 < len(MILESTONE_DAYS) else 30
            compressed[:, start:stop] = original[:, start : start + 1]
        out[name] = compressed
    _enrich(out)
    return out


def _plan_summary(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "peak_hands": int(np.max(arrays["hand_target_by_day"])),
        "peak_quadrants": int(np.max(arrays["unlocked_target_by_day"])),
        "peak_crop_targets": np.max(arrays["crop_target_by_day"], axis=1)[0]
        .astype(int)
        .tolist(),
        "peak_animal_targets": np.max(
            arrays["animal_owned_target_by_day"], axis=1
        )[0]
        .astype(int)
        .tolist(),
        "crop_flow_actions": int(
            sum(np.sum(arrays[name]) for name in FLOW_FIELDS[:5])
        ),
        "animal_flow_actions": int(
            sum(np.sum(arrays[name]) for name in FLOW_FIELDS[5:])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    version = importlib.metadata.version("kaggle-environments")
    if version != "1.32.7":
        raise RuntimeError(f"official package 1.32.7 required, got {version}")
    reference_compiler = _load_module("o17_reference_compiler", REFERENCE_COMPILER)
    economic_compiler = _load_module("o17_economic_compiler", ECONOMIC_COMPILER)
    with gzip.open(args.trace_bundle, "rt", encoding="utf-8") as stream:
        bundle = json.load(stream)
    if bundle.get("cases") != 36:
        raise RuntimeError(f"expected 36 trace cases, got {bundle.get('cases')}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for progress, game in enumerate(bundle["games"], start=1):
        official = make(
            "kaggriculture",
            configuration={"episodeSteps": 720, "seed": int(game["seed"])},
            debug=False,
        )
        official.reset(2)
        for joint_action in game["trace"]:
            official.step(joint_action)
        replay = official.toJSON()
        official_rewards = [float(value) for value in replay["rewards"]]
        reward_exact = official_rewards == [
            float(value) for value in game["native_rewards"]
        ]
        statuses = [str(value) for value in replay["statuses"]]
        complete = len(replay["steps"]) == 720 and statuses == ["DONE", "DONE"]
        if not reward_exact or not complete:
            raise RuntimeError(
                f"official parity failed for case {game['case_id']}: "
                f"reward={reward_exact}, complete={complete}"
            )

        player = int(game["candidate_seat"])
        full, observed_days = reference_compiler.compile_plan(replay, player)
        full.update(economic_compiler._daily_economic_envelope(replay, player))
        _enrich(full)
        if observed_days != list(range(30)):
            raise RuntimeError(
                f"incomplete semantic days for case {game['case_id']}: {observed_days}"
            )
        profiles = {
            "full": full,
            "target": _target_only(full),
        }
        profiles["milestone"] = _milestone_only(profiles["target"])

        outputs: dict[str, dict[str, Any]] = {}
        for profile, arrays in profiles.items():
            output = args.output_dir / f"case_{int(game['case_id']):03d}_{profile}.npz"
            np.savez_compressed(output, **arrays)
            outputs[profile] = {
                "path": str(output.resolve()),
                "sha256": _sha256(output),
                "summary": _plan_summary(arrays),
            }
        rows.append(
            {
                "case_id": int(game["case_id"]),
                "opponent": str(game["opponent"]),
                "seed": int(game["seed"]),
                "seat": player,
                "reference_route": str(game["reference_route"]),
                "reference_rewards": official_rewards,
                "reference_margin": float(game["reference_margin"]),
                "reward_exact": reward_exact,
                "complete": complete,
                "observed_days": observed_days,
                "raw_action_payload_stored": False,
                "raw_coordinates_stored": False,
                "profiles": outputs,
            }
        )
        print(
            json.dumps(
                {
                    "progress": progress,
                    "total": len(bundle["games"]),
                    "case": game["case_id"],
                    "reference": game["reference_route"],
                }
            ),
            flush=True,
        )

    payload = {
        "schema": "kaggriculture.candidate8-o17-semantic-plans.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "official_package_version": version,
        "boundary": (
            "Plan NPZ files contain only daily semantic targets, obligations and "
            "economic windows. No raw action, coordinate, identity feature or future "
            "random event is stored."
        ),
        "profiles": {
            "full": "30-day targets plus daily task flows and transaction windows",
            "target": "30-day state/resource targets without demonstrated task flows",
            "milestone": "day 0/6/12/18/29 target snapshots held piecewise",
        },
        "milestone_days": list(MILESTONE_DAYS),
        "cases": len(rows),
        "reward_exact_cases": sum(int(row["reward_exact"]) for row in rows),
        "complete_cases": sum(int(row["complete"]) for row in rows),
        "raw_action_payload_stored_in_plans": False,
        "raw_coordinates_stored_in_plans": False,
        "simulation_seconds": time.perf_counter() - started,
        "input": {
            "path": str(args.trace_bundle.resolve()),
            "sha256": _sha256(args.trace_bundle),
        },
        "compilers": {
            "reference": {
                "path": str(REFERENCE_COMPILER.resolve()),
                "sha256": _sha256(REFERENCE_COMPILER),
            },
            "economic": {
                "path": str(ECONOMIC_COMPILER.resolve()),
                "sha256": _sha256(ECONOMIC_COMPILER),
            },
        },
        "rows": rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "cases": payload["cases"],
                "reward_exact_cases": payload["reward_exact_cases"],
                "manifest": str(args.manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
