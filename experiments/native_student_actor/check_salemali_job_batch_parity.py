#!/usr/bin/env python3
"""Compare JobBatch Salemali actions with the frozen Python source."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

from fast_kaggriculture import Config, FastEnv

from experiments.native_opponents.salemali7_2900.check_parity import (
    load_path, normalize,
)
from experiments.native_student_actor.smoke_arbitrary_job_batch import (
    META, ROOT, ROLLOUT, THOMAS, settings,
)
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


SOURCE = ROOT / "opponents/salemali7_2900/main.py"
ASSET = ROOT / "experiments/native_opponents/salemali7_2900/salemali7_2900.assets.bin"
NATIVE_DIR = ROOT / "experiments/native_opponents/salemali7_2900/build"
OPS = (
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP", "PICKUP",
    "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER",
    "CARE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT",
    "BUY_ANIMAL", "SELL",
)
ITEMS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG",
    "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP",
)
OP_INDEX = {name: index for index, name in enumerate(OPS)}
ITEM_INDEX = {name: index for index, name in enumerate(ITEMS)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(action: dict) -> list[int]:
    action = normalize(action)
    units = [action["farmer"], *action["hands"]]
    market = action["market"]
    output = [len(units), len(market)]
    for atom in (*units, *market):
        op = OP_INDEX[str(atom[0])]
        item = -1 if len(atom) == 1 else ITEM_INDEX[str(atom[1])]
        quantity = 1 if len(atom) < 3 else int(atom[2])
        output.extend((op, item, quantity))
    return output


def decode(values: list[int]) -> dict:
    units, market = map(int, values[:2])
    if len(values) != 2 + 3 * (units + market) or units < 1:
        raise ValueError("malformed JobBatch action trace")
    atoms = []
    for offset in range(2, len(values), 3):
        op, item, quantity = map(int, values[offset:offset + 3])
        if not 0 <= op < len(OPS) or not -1 <= item < len(ITEMS):
            raise ValueError("invalid JobBatch action atom")
        if item < 0:
            if quantity != 1:
                raise ValueError("itemless action has non-unit quantity")
            atoms.append([OPS[op]])
        else:
            atoms.append([OPS[op], ITEMS[item], quantity])
    return {
        "farmer": atoms[0],
        "hands": atoms[1:units],
        "market": atoms[units:units + market],
    }


def replay_source(seed: int, seat: int, own_trace: list[list[int]],
                  rival_trace: list[list[int]]) -> tuple[dict | None, tuple[float, float]]:
    source = load_path(f"salemali_jobbatch_{seed}_{seat}", SOURCE)
    env = FastEnv(Config(), seed)
    observations = list(env.reset(seed))
    rival_seat = 1 - seat
    first_mismatch = None
    for step, (own_encoded, job_rival) in enumerate(zip(own_trace, rival_trace)):
        observation = observations[rival_seat]
        observation["step"] = step
        observation["player"] = rival_seat
        rival_action = source.agent(observation)
        source_encoded = encode(rival_action)
        if source_encoded != list(job_rival) and first_mismatch is None:
            first_mismatch = {
                "step": step,
                "python": source_encoded,
                "job_batch": list(job_rival),
            }
        actions = [{"farmer": ["PASS"], "hands": [], "market": []}
                   for _ in range(2)]
        actions[seat] = decode(list(own_encoded))
        actions[rival_seat] = rival_action
        observations = list(env.step(actions))
    if not env.done or env.step_count != 719:
        raise RuntimeError("Python-source replay did not terminate at step 719")
    return first_mismatch, tuple(map(float, env.rewards))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=2632700000)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--binary", type=Path, default=(
        ROOT / "work/agent-student-actor-owned-v3.so"))
    parser.add_argument("--weights", type=Path, default=(
        ROOT / "work/student-v1/native-v15-mix3-1536g.bin"))
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/native-student-rollout/"
        "salemali-jobbatch-parity-8seed-2632700000.json"))
    args = parser.parse_args()
    if args.seed_count < 8 or args.threads < 1 or args.output.exists():
        parser.error("seed-count must be >=8, threads positive, and output new")

    job_modules = sorted((ROLLOUT / "build").glob("_paused_plan*.so"))
    if len(job_modules) != 1:
        raise RuntimeError("expected exactly one JobBatch module")
    job_module = job_modules[0]
    native_modules = sorted(NATIVE_DIR.glob("salemali7_2900_native*.so"))
    if len(native_modules) != 1:
        raise RuntimeError("expected exactly one standalone Salemali module")
    artifacts = {
        "job_batch_module": sha256(job_module),
        "salemali_source": sha256(SOURCE),
        "salemali_asset": sha256(ASSET),
        "salemali_standalone_module": sha256(native_modules[0]),
        "student_binary": sha256(args.binary),
        "student_weights": sha256(args.weights),
        "route_actions": sha256(ROOT / "agent/route_actions.json.zlib"),
        "route_library": sha256(ROOT / "agent/route_library.json"),
        "deployment": sha256(ROOT / "agent/replay_deployment.json"),
        "r1_config": sha256(ROOT / "policy/r1/config.json"),
    }

    sys.path.insert(0, str(ROLLOUT / "build"))
    native = importlib.import_module("_paused_plan")
    bundle = NativeTeammateBundle(
        ROOT / "agent/teammate_base.py",
        ROOT / "agent/route_actions.json.zlib",
        ROOT / "agent/route_library.json")
    deployment_routes = [bundle.index(name) for name in
                         ("G275", "G195", "G024", "G316", "G267")]
    seeds = [seed for seed in range(args.seed_start,
                                    args.seed_start + args.seed_count)
             for _ in range(2)]
    seats = [seat for _ in range(args.seed_count) for seat in (0, 1)]
    count = len(seeds)
    batch = native.JobBatch(
        str(args.binary), bundle.executor, seeds, seats, [4] * count,
        [-1] * count, [202609232700 + i for i in range(count)],
        settings(args.binary), deployment_routes, str(THOMAS), str(META),
        str(ASSET))
    batch.run(args.threads, 2 << 20, True)
    suffix = batch.run_native_actor_suffix(
        str(args.weights), 0, args.threads, 2 << 20, True)
    traces = batch.action_traces()
    summaries = batch.summary()["cases"]

    cases = []
    for index, (seed, seat, summary) in enumerate(zip(seeds, seats, summaries)):
        own_trace = traces["own"][index]
        rival_trace = traces["rival"][index]
        if len(own_trace) != 719 or len(rival_trace) != 719:
            raise RuntimeError("JobBatch did not capture 719 complete frames")
        mismatch, rewards = replay_source(seed, seat, own_trace, rival_trace)
        terminal_exact = (
            rewards[seat] == float(summary["own_cash"]) and
            rewards[1 - seat] == float(summary["rival_cash"]) and
            bool(summary["done"]) and int(summary["step"]) == 719)
        cases.append({
            "seed": seed,
            "seat": seat,
            "frames": 719,
            "actions_exact": mismatch is None,
            "first_mismatch": mismatch,
            "terminal_exact": terminal_exact,
            "own_cash": float(summary["own_cash"]),
            "rival_cash": float(summary["rival_cash"]),
        })

    artifacts_after = {
        "job_batch_module": sha256(job_module),
        "salemali_source": sha256(SOURCE),
        "salemali_asset": sha256(ASSET),
    }
    stable = all(artifacts[name] == digest
                 for name, digest in artifacts_after.items())
    passed = stable and all(case["actions_exact"] and case["terminal_exact"]
                            for case in cases)
    report = {
        "schema": "salemali-2900-jobbatch-parity-v1",
        "status": "PASS" if passed else "FAIL",
        "reference": "opponents/salemali7_2900/main.py",
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "both_seats": True,
        "frames_per_case": 719,
        "exact_action_frames": sum(719 for case in cases
                                   if case["actions_exact"]),
        "total_action_frames": 719 * len(cases),
        "exact_terminals": sum(case["terminal_exact"] for case in cases),
        "total_cases": len(cases),
        "artifacts_stable_during_run": stable,
        "artifacts_sha256": artifacts,
        "native_suffix_metrics": dict(suffix),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "seed_start", "seed_count", "exact_action_frames",
        "total_action_frames", "exact_terminals", "total_cases")},
        indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
