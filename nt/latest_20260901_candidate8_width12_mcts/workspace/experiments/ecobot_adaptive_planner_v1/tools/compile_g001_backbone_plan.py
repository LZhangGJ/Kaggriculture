#!/usr/bin/env python3
"""Compile an official reference game into a searchable semantic default plan.

The artifact stores daily obligations and economic envelopes, never the raw
719-action tape or Replay coordinates.  Demonstrated values are the initial
genome point for later search, not hard-coded planner decisions.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REFERENCE_COMPILER = (
    ROOT
    / "experiments"
    / "general_project_planner_v1"
    / "tools"
    / "compile_reference_obligation_plan.py"
)


def _load_reference_compiler():
    spec = importlib.util.spec_from_file_location(
        "ecobot_reference_obligation_compiler", REFERENCE_COMPILER
    )
    if spec is None or spec.loader is None:
        raise ImportError(REFERENCE_COMPILER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _read_replay(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def _daily_economic_envelope(replay: Mapping[str, Any], player: int) -> dict[str, np.ndarray]:
    money_by_day: list[list[int]] = [[] for _ in range(30)]
    sell_steps_by_day: list[list[int]] = [[] for _ in range(30)]
    buy_steps_by_day: list[list[int]] = [[] for _ in range(30)]
    terminal_sell_steps: list[int] = []
    for frame_index, states in enumerate(replay.get("steps", ()) or ()):
        row = states[player]
        observation = row.get("observation", {}) or {}
        day = int(observation.get("day", -1) or 0)
        if 0 <= day < 30:
            farm = (observation.get("farms", ()) or ())[player]
            money_by_day[day].append(int(farm.get("money", 0) or 0))
        action = row.get("action", {}) or {}
        for order in action.get("market", ()) or ():
            if not isinstance(order, list) or not order:
                continue
            op = str(order[0])
            if op == "SELL" and 0 <= day < 30:
                sell_steps_by_day[day].append(frame_index)
                if day >= 27:
                    terminal_sell_steps.append(frame_index)
            elif op.startswith("BUY_") or op == "HIRE":
                if 0 <= day < 30:
                    buy_steps_by_day[day].append(frame_index)

    cash_start = np.zeros((30,), dtype=np.int32)
    cash_floor = np.zeros((30,), dtype=np.int32)
    cash_end = np.zeros((30,), dtype=np.int32)
    sell_first = np.full((30,), -1, dtype=np.int16)
    sell_last = np.full((30,), -1, dtype=np.int16)
    buy_first = np.full((30,), -1, dtype=np.int16)
    buy_last = np.full((30,), -1, dtype=np.int16)
    prior = 0
    for day in range(30):
        values = money_by_day[day]
        if values:
            cash_start[day] = values[0]
            cash_floor[day] = min(values)
            cash_end[day] = values[-1]
            prior = values[-1]
        else:
            cash_start[day] = prior
            cash_floor[day] = prior
            cash_end[day] = prior
        if sell_steps_by_day[day]:
            sell_first[day] = min(sell_steps_by_day[day]) % 24
            sell_last[day] = max(sell_steps_by_day[day]) % 24
        if buy_steps_by_day[day]:
            buy_first[day] = min(buy_steps_by_day[day]) % 24
            buy_last[day] = max(buy_steps_by_day[day]) % 24
    terminal_start = min(terminal_sell_steps) if terminal_sell_steps else -1
    return {
        "cash_start_by_day": cash_start[None],
        "cash_floor_by_day": cash_floor[None],
        "cash_end_by_day": cash_end[None],
        "sell_first_hour_by_day": sell_first[None],
        "sell_last_hour_by_day": sell_last[None],
        "buy_first_hour_by_day": buy_first[None],
        "buy_last_hour_by_day": buy_last[None],
        "terminal_liquidation_start_step": np.asarray([terminal_start], dtype=np.int16),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--player", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--reference-label",
        default="G001",
        help="Human-readable provenance only; it is never exposed to the planner.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    replay = _read_replay(args.replay)
    compiler = _load_reference_compiler()
    arrays, observed_days = compiler.compile_plan(replay, args.player)
    arrays.update(_daily_economic_envelope(replay, args.player))

    animal_owned = np.asarray(arrays["animal_owned_target_by_day"], dtype=np.int16)
    animal_additions = np.maximum(
        animal_owned - np.concatenate((np.zeros_like(animal_owned[:, :1]), animal_owned[:, :-1]), axis=1),
        0,
    ).astype(np.int16)
    unlocked = np.asarray(arrays["unlocked_target_by_day"], dtype=np.int16)
    land_additions = np.maximum(
        unlocked - np.concatenate((np.ones_like(unlocked[:, :1]), unlocked[:, :-1]), axis=1),
        0,
    ).astype(np.int8)
    arrays["animal_purchase_additions_by_day"] = animal_additions
    arrays["land_additions_by_day"] = land_additions
    arrays["plan_mode"] = np.asarray([0], dtype=np.int8)  # 0 == FOLLOW_REFERENCE
    arrays["schema_version"] = np.asarray([1], dtype=np.int16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    reference_rewards = replay.get("rewards", ()) or ()
    payload = {
        "schema": "kaggriculture.ecobot.semantic-plan.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if observed_days == list(range(30)) else "FAIL",
        "source_replay": str(args.replay.resolve()),
        "source_replay_sha256": _sha256(args.replay),
        "player": args.player,
        "reference_label": args.reference_label,
        "reference_final_bank": int(reference_rewards[args.player]),
        "observed_days": observed_days,
        "raw_action_payload_stored": False,
        "raw_coordinates_stored": False,
        "values_are": "searchable initial genome point, not runtime constants",
        "identity_features_allowed": False,
        "fields": {name: list(np.asarray(value).shape) for name, value in arrays.items()},
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "compiler": str(Path(__file__).resolve()),
        "reference_compiler": str(REFERENCE_COMPILER.resolve()),
        "reference_compiler_sha256": _sha256(REFERENCE_COMPILER),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
