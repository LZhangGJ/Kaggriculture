#!/usr/bin/env python3
"""Compare the native port against the authoritative Python agent stepwise."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from fast_kaggriculture import Config, FastEnv


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "work/new_public_opponents/the-2965-master-hybrid-engine/output/main.py"
ASSETS = HERE / "metav4_2965.assets.bin"
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def native_module(build_dir: Path):
    candidates = sorted(build_dir.glob("metav4_2965_native*.so"))
    if len(candidates) != 1:
        raise RuntimeError("run build.sh first; expected exactly one native module")
    return load_path("metav4_2965_native", candidates[0]), candidates[0]


def normalize(action: dict) -> dict:
    def unit(value):
        value = list(value or ["PASS"])
        return value + [1] if len(value) == 2 else value

    return {
        "farmer": unit(action.get("farmer")),
        "hands": [unit(value) for value in action.get("hands", [])],
        # Empty and zero-quantity orders are execution no-ops.  Keep real list
        # ordering exact because simultaneous market priority is semantic.
        "market": [list(value) for value in action.get("market", [])
                   if value and value[0] != "PASS" and
                   (len(value) < 3 or int(value[2]) != 0)],
    }


def python_trace(seed: int, seat: int, steps: int, rival_route: int | None,
                 source: Path):
    module = load_path(f"metav4_2965_ref_{seed}_{seat}", source)
    # The asset generator explicitly installs the lazy opening before writing
    # its route table. Freeze the Python rival tape at that same post-install
    # point; import-time routes are not the exported C++ fixture.
    if rival_route is not None:
        module._alt_install(module._ALT_MODE)
    rival_tape = (copy.deepcopy(module._IMPL.chassis.routes[rival_route])
                  if rival_route is not None else None)
    env = FastEnv(Config(), seed)
    observations = list(env.reset(seed))
    trace = []
    for step in range(steps):
        observation = observations[seat]
        observation["step"] = step
        observation["player"] = seat
        action = module.agent(observation, {})
        trace.append(action)
        joint = [PASS, PASS]
        joint[seat] = action
        if rival_tape is not None:
            joint[1 - seat] = rival_tape[step]
        observations = list(env.step(joint))
    native = module._IMPL.chassis.players.get(seat, {})
    return trace, int(native.get("route", 0))


def mismatch(expected: dict, actual: dict):
    left, right = normalize(expected), normalize(actual)
    for field in ("farmer", "hands", "market"):
        if left[field] != right[field]:
            return field, left[field], right[field]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2609500000)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--steps", type=int, default=719)
    parser.add_argument("--seat", type=int, choices=(0, 1))
    parser.add_argument("--rival-route", type=int,
                        help="drive the other seat with this raw replay route")
    parser.add_argument("--output", type=Path, default=HERE / "parity_report.json")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--native-build", type=Path, default=HERE / "build")
    parser.add_argument("--soil-variant", action="store_true")
    parser.add_argument("--v57-variant", action="store_true")
    parser.add_argument("--v15-variant", action="store_true")
    args = parser.parse_args()
    if args.seed_count < 1 or not 1 <= args.steps <= 719:
        parser.error("seed-count must be positive and steps in [1,719]")

    native, native_path = native_module(args.native_build)
    seats = (args.seat,) if args.seat is not None else (0, 1)
    runs = []
    for seed in range(args.seed, args.seed + args.seed_count):
        for seat in seats:
            expected, expected_route = python_trace(
                seed, seat, args.steps, args.rival_route, args.source)
            result = (native.play(str(args.assets), seed, seat, args.steps,
                                  args.soil_variant, args.v57_variant,
                                  args.v15_variant)
                      if args.rival_route is None else
                      native.play_vs_route(str(args.assets), seed, seat, args.steps,
                                           args.rival_route, args.soil_variant,
                                           args.v57_variant, args.v15_variant))
            first = None
            for step, (left, right) in enumerate(zip(expected, result["trace"])):
                detail = mismatch(left, right)
                if detail:
                    field, python_value, native_value = detail
                    first = {
                        "step": step,
                        "field": field,
                        "python": python_value,
                        "native": native_value,
                    }
                    break
            runs.append({
                "seed": seed,
                "seat": seat,
                "steps": args.steps,
                "expected_route": expected_route,
                "native_route": int(result["route"]),
                "exact": first is None and expected_route == int(result["route"]),
                "first_mismatch": first,
            })
            print(json.dumps(runs[-1], separators=(",", ":")), flush=True)

    report = {
        "schema": "metav4-2965-native-parity-v1",
        "source": str(args.source.resolve().relative_to(ROOT)),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "native_build": str(native_path.resolve().relative_to(ROOT)),
        "native_build_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "asset_sha256": hashlib.sha256(args.assets.read_bytes()).hexdigest(),
        "seed_start": args.seed,
        "seed_count": args.seed_count,
        "steps": args.steps,
        "both_seats": args.seat is None,
        "rival_route": args.rival_route,
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
