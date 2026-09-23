#!/usr/bin/env python3
"""Compare frozen Python Fieldcraft with native C++ on fresh passive games."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from fast_kaggriculture import Config, FastEnv


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_DIR = (ROOT / "work/new_public_opponents/"
              "kaggriculture-2887-score-fieldcraft-agent/output/extracted")
SOURCE = SOURCE_DIR / "main.py"
ASSETS = HERE / "fieldcraft_2887.assets.bin"
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def native_module():
    paths = sorted((HERE / "build").glob("fieldcraft_2887_native*.so"))
    if len(paths) != 1:
        raise RuntimeError("run build.sh first; expected one native module")
    return load_path("fieldcraft_2887_native", paths[0]), paths[0]


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


def difference(expected: dict, actual: dict):
    expected, actual = normalize(expected), normalize(actual)
    for field in ("farmer", "hands", "market"):
        if expected[field] != actual[field]:
            return {"field": field, "python": expected[field],
                    "native": actual[field]}
    return None


def python_trace(seed: int, seat: int, steps: int):
    if str(SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(SOURCE_DIR))
    source = load_path(f"fieldcraft_ref_{seed}_{seat}", SOURCE)
    env = FastEnv(Config(), seed)
    observations = list(env.reset(seed))
    trace = []
    for step in range(steps):
        observation = observations[seat]
        observation["step"] = step
        observation["player"] = seat
        action = source.agent(observation, {})
        trace.append(action)
        joint = [PASS, PASS]
        joint[seat] = action
        observations = list(env.step(joint))
    return trace, source.TAPE.route


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2632500000)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--steps", type=int, default=719)
    parser.add_argument("--output", type=Path,
                        default=HERE / "parity_live_report.json")
    args = parser.parse_args()
    if args.seed_count < 1 or not 1 <= args.steps <= 719:
        parser.error("seed-count must be positive and steps in [1,719]")
    native, native_path = native_module()
    runs = []
    for seed in range(args.seed, args.seed + args.seed_count):
        for seat in (0, 1):
            expected, python_route = python_trace(seed, seat, args.steps)
            native_result = native.play(str(ASSETS), seed, seat, args.steps)
            actual = native_result["trace"]
            first = None
            for step, (left, right) in enumerate(zip(expected, actual)):
                detail = difference(left, right)
                if detail is not None:
                    first = {"step": step, **detail}
                    break
            run = {"seed": seed, "seat": seat, "steps": args.steps,
                   "python_route": python_route,
                   "native_route": native_result["route"],
                   "exact_steps": args.steps if first is None else first["step"],
                   "first_mismatch": first}
            runs.append(run)
            print(json.dumps(run, separators=(",", ":")), flush=True)
    report = {
        "schema": "fieldcraft-2887-native-live-parity-v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "asset_sha256": hashlib.sha256(ASSETS.read_bytes()).hexdigest(),
        "native_build": str(native_path.relative_to(ROOT)),
        "native_build_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "seed_start": args.seed, "seed_count": args.seed_count,
        "steps": args.steps, "runs": runs,
        "exact_runs": sum(run["first_mismatch"] is None for run in runs),
        "total_runs": len(runs),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("exact_runs", "total_runs")},
                     sort_keys=True))
    sys.exit(0 if report["exact_runs"] == report["total_runs"] else 1)


if __name__ == "__main__":
    main()
