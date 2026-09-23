#!/usr/bin/env python3
"""Gate Fieldcraft JobBatch code 5 against its standalone native port."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from fast_kaggriculture import Config, FastEnv
from experiments.native_student_actor.smoke_arbitrary_job_batch import (
    META, ROOT, ROLLOUT, THOMAS, settings,
)
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


HERE = ROOT / "experiments/native_opponents/fieldcraft_2887"
SOURCE_DIR = (ROOT / "work/new_public_opponents/"
              "kaggriculture-2887-score-fieldcraft-agent/output/extracted")
SOURCE = SOURCE_DIR / "main.py"
MIRROR = SOURCE_DIR / "mirror_plan.py"
ASSET = HERE / "fieldcraft_2887.assets.bin"
SALEMALI = (ROOT / "experiments/native_opponents/salemali7_2900/"
            "salemali7_2900.assets.bin")
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


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode(action: dict) -> list[int]:
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


def replay_standalone(native, seed: int, seat: int,
                      own_trace: list[list[int]],
                      rival_trace: list[list[int]]) -> dict:
    env = FastEnv(Config(), seed)
    env.reset(seed)
    opponent = native.Opponent(str(ASSET))
    rival_seat = 1 - seat
    first_mismatch = None
    for step, (own_encoded, job_rival) in enumerate(
            zip(own_trace, rival_trace)):
        rival_action = opponent.action(env, rival_seat)
        standalone_encoded = encode(rival_action)
        if standalone_encoded != list(job_rival) and first_mismatch is None:
            first_mismatch = {
                "step": step,
                "standalone": standalone_encoded,
                "job_batch": list(job_rival),
            }
        actions = [{"farmer": ["PASS"], "hands": [], "market": []}
                   for _ in range(2)]
        actions[seat] = decode(list(own_encoded))
        actions[rival_seat] = rival_action
        env.step(actions)
    if not env.done or env.step_count != 719:
        raise RuntimeError("standalone replay did not terminate at step 719")
    rewards = tuple(map(float, env.rewards))
    return {
        "actions_exact": first_mismatch is None,
        "first_mismatch": first_mismatch,
        "standalone_route": int(opponent.route(rival_seat)),
        "rewards": rewards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=2632800100)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--binary", type=Path, default=(
        ROOT / "work/agent-student-actor-owned-v3.so"))
    parser.add_argument("--weights", type=Path, default=(
        ROOT / "work/student-v1/native-v21-hard2-1536g.bin"))
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/native-student-rollout/"
        "fieldcraft-jobbatch-b8-parity-2632800100-v2.json"))
    args = parser.parse_args()
    if args.seed_count < 4 or args.threads < 1 or args.output.exists():
        parser.error("seed-count must be >=4, threads positive, and output new")

    job_modules = sorted((ROLLOUT / "build").glob("_paused_plan*.so"))
    standalone_modules = sorted((HERE / "build").glob(
        "fieldcraft_2887_native*.so"))
    if len(job_modules) != 1 or len(standalone_modules) != 1:
        raise RuntimeError("expected one JobBatch and one Fieldcraft module")
    job_module, standalone_module = job_modules[0], standalone_modules[0]
    artifacts_paths = {
        "job_batch_module": job_module,
        "fieldcraft_source": SOURCE,
        "fieldcraft_mirror": MIRROR,
        "fieldcraft_asset": ASSET,
        "fieldcraft_standalone_module": standalone_module,
        "student_binary": args.binary,
        "student_weights": args.weights,
        "route_actions": ROOT / "agent/route_actions.json.zlib",
        "route_library": ROOT / "agent/route_library.json",
        "deployment": ROOT / "agent/replay_deployment.json",
        "r1_config": ROOT / "policy/r1/config.json",
    }
    artifacts = {name: sha256(path) for name, path in artifacts_paths.items()}

    sys.path.insert(0, str(ROLLOUT / "build"))
    job_native = importlib.import_module("_paused_plan")
    standalone = load_path("fieldcraft_2887_native", standalone_module)
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
    policy_seeds = [202609238100 + index for index in range(len(seeds))]
    batch = job_native.JobBatch(
        str(args.binary), bundle.executor, seeds, seats, [5] * len(seeds),
        [-1] * len(seeds), policy_seeds, settings(args.binary),
        deployment_routes, str(THOMAS), str(META), str(SALEMALI), str(ASSET))
    batch.run(args.threads, 2 << 20, True)
    suffix = dict(batch.run_native_actor_suffix(
        str(args.weights), 0, args.threads, 2 << 20, True))
    traces = batch.action_traces()
    summaries = batch.summary()["cases"]
    arrays = {name: np.asarray(value)
              for name, value in batch.ppo_arrays().items()}

    event_indices = np.arange(arrays["event_action"].shape[0])
    selected_legal = arrays["event_legal"][
        event_indices, arrays["event_action"].astype(np.int64)]
    illegal = int(np.count_nonzero(~selected_legal))
    array_identity_exact = (
        np.array_equal(arrays["seed"], np.asarray(seeds)) and
        np.array_equal(arrays["seat"], np.asarray(seats)) and
        np.all(arrays["opponent"] == 5) and
        np.all(arrays["route"] == -1) and
        np.array_equal(arrays["policy_seed"], np.asarray(policy_seeds)))

    cases = []
    for index, (seed, seat, policy_seed, summary) in enumerate(
            zip(seeds, seats, policy_seeds, summaries)):
        own_trace = traces["own"][index]
        rival_trace = traces["rival"][index]
        if len(own_trace) != 719 or len(rival_trace) != 719:
            raise RuntimeError("JobBatch did not capture 719 complete frames")
        replay = replay_standalone(
            standalone, seed, seat, own_trace, rival_trace)
        rewards = replay.pop("rewards")
        terminal_exact = (
            rewards[seat] == float(summary["own_cash"]) and
            rewards[1 - seat] == float(summary["rival_cash"]) and
            bool(summary["done"]) and int(summary["step"]) == 719)
        identity_exact = (
            int(summary["seed"]) == seed and
            int(summary["seat"]) == seat and
            summary["opponent"] == "fieldcraft" and
            int(summary["route"]) == -1 and
            int(summary["policy_seed"]) == policy_seed and
            int(summary["actor_seed"]) == policy_seed and
            int(summary["validated_steps"]) == 288 and
            int(summary["suffix_steps"]) == 431 and
            int(summary["captured_actions"]) == 719 and
            int(summary["actor_days"]) == 17 and
            summary["error"] == "")
        cases.append({
            "seed": seed,
            "seat": seat,
            "opponent_code": 5,
            "requested_route": -1,
            "policy_seed": policy_seed,
            "frames": 719,
            **replay,
            "terminal_exact": terminal_exact,
            "identity_exact": identity_exact,
            "own_cash": float(summary["own_cash"]),
            "rival_cash": float(summary["rival_cash"]),
        })

    artifacts_after = {
        name: sha256(path) for name, path in artifacts_paths.items()}
    artifacts_stable = artifacts == artifacts_after
    passed = (
        artifacts_stable and array_identity_exact and illegal == 0 and
        all(case["actions_exact"] and case["terminal_exact"] and
            case["identity_exact"] for case in cases))
    report = {
        "schema": "fieldcraft-2887-jobbatch-parity-v1",
        "status": "PASS" if passed else "FAIL",
        "scope": "JobBatch code5 route=-1 versus standalone native Fieldcraft",
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "both_seats": True,
        "opponent_code": 5,
        "requested_route": -1,
        "exact_action_frames": sum(719 for case in cases
                                   if case["actions_exact"]),
        "total_action_frames": 719 * len(cases),
        "exact_terminals": sum(case["terminal_exact"] for case in cases),
        "identity_exact_cases": sum(case["identity_exact"] for case in cases),
        "total_cases": len(cases),
        "illegal": illegal,
        "fallbacks": 0,
        "fallback_semantics": "fail_closed; no native actor fallback path",
        "ppo_array_identity_exact": array_identity_exact,
        "artifacts_stable_during_run": artifacts_stable,
        "artifacts_sha256": artifacts,
        "native_suffix_metrics": suffix,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "exact_action_frames", "total_action_frames",
        "exact_terminals", "identity_exact_cases", "total_cases",
        "illegal", "fallbacks")}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
