#!/usr/bin/env python3
"""Compare native Salemali7 with the frozen Python source at every step."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from fast_kaggriculture import Config, FastEnv


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "opponents/salemali7_2900/main.py"
ASSETS = HERE / "salemali7_2900.assets.bin"
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def native_module():
    candidates = sorted((HERE / "build").glob("salemali7_2900_native*.so"))
    if len(candidates) != 1:
        raise RuntimeError("run build.sh first; expected exactly one native module")
    return load_path("salemali7_2900_native", candidates[0]), candidates[0]


def normalize(action: dict) -> dict:
    def unit(value):
        value = list(value or ["PASS"])
        return value + [1] if len(value) == 2 else value

    return {
        "farmer": unit(action.get("farmer")),
        "hands": [unit(value) for value in action.get("hands", [])],
        "market": [list(value) for value in action.get("market", [])
                   if value and value[0] != "PASS" and
                   (len(value) < 3 or int(value[2]) != 0)],
    }


def python_trace(seed: int, seat: int, steps: int):
    module = load_path(f"salemali7_ref_{seed}_{seat}", SOURCE)
    env = FastEnv(Config(), seed)
    observations = list(env.reset(seed))
    trace = []
    unit_overlay_frames = 0
    market_overlay_frames = 0
    for step in range(steps):
        observation = observations[seat]
        observation["step"] = step
        observation["player"] = seat
        raw = normalize(module._align_hands(
            module._copy_action(module._ACTIONS[step]), observation))
        action = module.agent(observation)
        trace.append(action)
        final = normalize(action)
        unit_overlay_frames += int(
            raw["farmer"] != final["farmer"] or raw["hands"] != final["hands"])
        market_overlay_frames += int(raw["market"] != final["market"])
        joint = [PASS, PASS]
        joint[seat] = action
        observations = list(env.step(joint))
    return trace, unit_overlay_frames, market_overlay_frames


def python_self_trace(seed: int, steps: int):
    module = load_path(f"salemali7_self_ref_{seed}", SOURCE)
    env = FastEnv(Config(), seed)
    observations = list(env.reset(seed))
    traces = [[], []]
    overlays = [[0, 0], [0, 0]]
    for step in range(steps):
        actions = []
        for seat in (0, 1):
            observation = observations[seat]
            observation["step"] = step
            observation["player"] = seat
            raw = normalize(module._align_hands(
                module._copy_action(module._ACTIONS[step]), observation))
            action = module.agent(observation)
            final = normalize(action)
            traces[seat].append(action)
            overlays[seat][0] += int(
                raw["farmer"] != final["farmer"] or
                raw["hands"] != final["hands"])
            overlays[seat][1] += int(raw["market"] != final["market"])
            actions.append(action)
        observations = list(env.step(actions))
    return traces, overlays


def fail_closed_checks(native) -> dict:
    corrupt = bytearray(ASSETS.read_bytes())
    corrupt[0] ^= 0xFF
    with tempfile.NamedTemporaryFile(suffix=".bin") as file:
        file.write(corrupt)
        file.flush()
        try:
            native.Opponent(file.name)
        except RuntimeError:
            corrupt_asset_rejected = True
        else:
            corrupt_asset_rejected = False

    env = FastEnv(Config(), 2631999999)
    env.step([PASS, PASS])
    opponent = native.Opponent(str(ASSETS))
    try:
        opponent.action(env, 0)
    except RuntimeError:
        skipped_step_rejected = True
    else:
        skipped_step_rejected = False
    assert corrupt_asset_rejected and skipped_step_rejected
    return {"corrupt_asset_rejected": corrupt_asset_rejected,
            "skipped_step_rejected": skipped_step_rejected}


def mismatch(expected: dict, actual: dict):
    left, right = normalize(expected), normalize(actual)
    for field in ("farmer", "hands", "market"):
        if left[field] != right[field]:
            return field, left[field], right[field]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2631000000)
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--steps", type=int, default=719)
    parser.add_argument("--seat", type=int, choices=(0, 1))
    parser.add_argument("--active-self", action="store_true",
                        help="run both source seats against each other")
    parser.add_argument("--output", type=Path, default=HERE / "parity_report.json")
    args = parser.parse_args()
    if args.seed_count < 1 or not 1 <= args.steps <= 719:
        parser.error("seed-count must be positive and steps in [1,719]")
    if args.active_self and args.seat is not None:
        parser.error("--seat is incompatible with --active-self")

    native, native_path = native_module()
    fail_closed = fail_closed_checks(native)
    seats = (args.seat,) if args.seat is not None else (0, 1)
    runs = []
    for seed in range(args.seed, args.seed + args.seed_count):
        if args.active_self:
            expected_traces, overlays = python_self_trace(seed, args.steps)
            actual_traces = native.play_self(str(ASSETS), seed, args.steps)["traces"]
            cases = ((seat, expected_traces[seat], actual_traces[seat], *overlays[seat])
                     for seat in (0, 1))
        else:
            cases = []
            for seat in seats:
                expected, unit_overlays, market_overlays = python_trace(
                    seed, seat, args.steps)
                actual = native.play(str(ASSETS), seed, seat, args.steps)["trace"]
                cases.append((seat, expected, actual, unit_overlays, market_overlays))
        for seat, expected, actual, unit_overlays, market_overlays in cases:
            first = None
            for step, (left, right) in enumerate(zip(expected, actual)):
                detail = mismatch(left, right)
                if detail:
                    field, python_value, native_value = detail
                    first = {"step": step, "field": field,
                             "python": python_value, "native": native_value}
                    break
            runs.append({"seed": seed, "seat": seat, "steps": args.steps,
                         "unit_overlay_frames": unit_overlays,
                         "market_overlay_frames": market_overlays,
                         "exact": first is None, "first_mismatch": first})
            print(json.dumps(runs[-1], separators=(",", ":")), flush=True)

    report = {
        "schema": "salemali7-2900-native-parity-v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "native_build": str(native_path.relative_to(ROOT)),
        "native_build_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "asset_sha256": hashlib.sha256(ASSETS.read_bytes()).hexdigest(),
        "seed_start": args.seed,
        "seed_count": args.seed_count,
        "steps": args.steps,
        "both_seats": args.seat is None,
        "active_self": args.active_self,
        "fail_closed": fail_closed,
        "unit_overlay_frames": sum(run["unit_overlay_frames"] for run in runs),
        "market_overlay_frames": sum(run["market_overlay_frames"] for run in runs),
        "exact_runs": sum(run["exact"] for run in runs),
        "total_runs": len(runs),
        "runs": runs,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("exact_runs", "total_runs")},
                     sort_keys=True))
    sys.exit(0 if report["exact_runs"] == report["total_runs"] else 1)


if __name__ == "__main__":
    main()
