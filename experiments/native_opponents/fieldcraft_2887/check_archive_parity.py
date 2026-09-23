#!/usr/bin/env python3
"""Replay recorded games and report Fieldcraft native's first action mismatch."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from fast_kaggriculture import Config, FastEnv


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = HERE / "fieldcraft_2887.assets.bin"
DEFAULT_GLOB = (
    "data/bc/midgame-v1/**/candidate/fieldcraft_2887/*.jsonl.gz"
)


def load_native():
    paths = sorted((HERE / "build").glob("fieldcraft_2887_native*.so"))
    if len(paths) != 1:
        raise RuntimeError("run build.sh first; expected one native module")
    spec = importlib.util.spec_from_file_location("fieldcraft_2887_native", paths[0])
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalized(action: dict) -> dict:
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


def first_difference(expected: dict, actual: dict):
    expected, actual = normalized(expected), normalized(actual)
    for field in ("farmer", "hands", "market"):
        if expected[field] != actual[field]:
            return {"field": field, "recorded": expected[field],
                    "native": actual[field]}
    return None


def check_file(native, path: Path, max_steps: int) -> dict:
    path = path.resolve()
    with gzip.open(path, "rt") as stream:
        meta = json.loads(next(stream))
        frames = [json.loads(line) for line in stream]
    frames = [frame for frame in frames if frame.get("type") == "step"]
    seed, our_seat = int(meta["seed"]), int(meta["seat"])
    opponent_seat = 1 - our_seat
    env = FastEnv(Config(), seed)
    env.reset(seed)
    opponent = native.Opponent(str(ASSETS))
    for frame in frames[:max_steps]:
        step = int(frame["step"])
        if env.step_count != step:
            raise RuntimeError(f"non-contiguous archive at step {step}")
        actual = opponent.action(env, opponent_seat)
        detail = first_difference(frame["actions"][opponent_seat], actual)
        if detail is not None:
            return {"path": str(path.relative_to(ROOT)), "seed": seed,
                    "opponent_seat": opponent_seat, "exact_steps": step,
                    "route": opponent.route(opponent_seat),
                    "first_mismatch": {"step": step, **detail}}
        env.step(frame["actions"])
    return {"path": str(path.relative_to(ROOT)), "seed": seed,
            "opponent_seat": opponent_seat,
            "exact_steps": min(len(frames), max_steps),
            "route": opponent.route(opponent_seat), "first_mismatch": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="*")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--steps", type=int, default=719)
    parser.add_argument("--output", type=Path,
                        default=HERE / "parity_archive_report.json")
    args = parser.parse_args()
    if args.limit < 1 or not 1 <= args.steps <= 719:
        parser.error("limit must be positive and steps in [1,719]")
    paths = args.paths or sorted(ROOT.glob(DEFAULT_GLOB))
    if not paths:
        parser.error("no archived Fieldcraft trajectories found")
    native = load_native()
    reports = []
    for path in paths[:args.limit]:
        report = check_file(native, path, args.steps)
        reports.append(report)
        print(json.dumps(report, separators=(",", ":")), flush=True)
    summary = {
        "schema": "fieldcraft-2887-native-archive-parity-v1",
        "asset_sha256": hashlib.sha256(ASSETS.read_bytes()).hexdigest(),
        "requested_steps": args.steps,
        "total_games": len(reports),
        "exact_games": sum(report["first_mismatch"] is None
                           for report in reports),
        "exact_frames": sum(report["exact_steps"] for report in reports),
        "compared_frames": sum(
            args.steps if report["first_mismatch"] is None
            else report["exact_steps"] + 1 for report in reports),
        "routes": sorted({report["route"] for report in reports}),
        "opponent_seats": sorted({report["opponent_seat"]
                                  for report in reports}),
        "runs": reports,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    sys.exit(0 if all(report["first_mismatch"] is None
                      for report in reports) else 1)


if __name__ == "__main__":
    main()
