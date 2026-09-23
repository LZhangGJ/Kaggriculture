#!/usr/bin/env python3
"""Run cache-free native jobs on fresh seeds and persist PPO arrays."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from experiments.native_student_actor.native_job_batch import ppo_games
from experiments.native_student_actor.smoke_arbitrary_job_batch import (
    META, ROOT, ROLLOUT, THOMAS, WEIGHTS, settings,
)
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=(
        ROOT / "work/agent-student-actor-owned-v3.so"))
    parser.add_argument("--weights", type=Path, default=(
        ROOT / "work/student-v1/native-v9-daybundle-stratified.bin"))
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=2631000000)
    parser.add_argument("--policy-seed", type=int, default=2026092400)
    parser.add_argument("--threads", type=int, default=min(128, os.cpu_count() or 1))
    parser.add_argument("--arrays", type=Path, default=(
        ROOT / "work/native-student-rollout/arbitrary-job-new-seed-b128-arrays.npz"))
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/native-student-rollout/arbitrary-job-new-seed-b128.json"))
    args = parser.parse_args()
    if (args.games < 2 or args.games % 2 or args.games > 2048 or
            args.threads < 1 or args.arrays.exists() or args.output.exists()):
        parser.error("games must be even in [2,2048], threads positive, outputs new")
    sys.path.insert(0, str(ROLLOUT / "build"))
    native = importlib.import_module("_paused_plan")
    setup_started = time.perf_counter()
    bundle = NativeTeammateBundle(
        ROOT / "agent/teammate_base.py",
        ROOT / "agent/route_actions.json.zlib",
        ROOT / "agent/route_library.json")
    deployment_routes = [bundle.index(name) for name in
                         ("G275", "G195", "G024", "G316", "G267")]
    seeds = [args.seed_start + index // 2 for index in range(args.games)]
    seats = [index % 2 for index in range(args.games)]
    opponents = [1 + ((index // 2) % 2) for index in range(args.games)]
    routes = [-1] * args.games
    policy_seeds = [args.policy_seed + index for index in range(args.games)]
    batch = native.JobBatch(
        str(args.binary), bundle.executor, seeds, seats, opponents, routes,
        policy_seeds, settings(args.binary), deployment_routes,
        str(THOMAS), str(META))
    setup_seconds = time.perf_counter() - setup_started
    prefix_started = time.perf_counter()
    batch.run(args.threads, 2 << 20, False)
    prefix_seconds = time.perf_counter() - prefix_started
    suffix = batch.run_native_actor_suffix(
        str(args.weights), 0, args.threads, 2 << 20, True)
    capture_started = time.perf_counter()
    arrays = {name: np.asarray(value) for name, value in batch.ppo_arrays().items()}
    games = ppo_games(arrays)
    capture_seconds = time.perf_counter() - capture_started
    if (len(games) != args.games or any(
            len(game["days"]) != 17 or game["frames"] != 719 or
            not game["days"] or
            [day["step"] for day in game["days"]] != list(range(288, 673, 24))
            for game in games)):
        raise RuntimeError("native arrays do not satisfy current PPO game schema")
    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    save_started = time.perf_counter()
    np.savez(args.arrays, **arrays)
    save_seconds = time.perf_counter() - save_started
    summary = batch.summary()["cases"]
    if not all(row["done"] and row["step"] == 719 and
               row["actor_days"] == 17 and row["suffix_steps"] == 431 and
               not row["error"] for row in summary):
        raise RuntimeError("non-terminal native job")
    result = {
        "status": "PASS",
        "scope": "cache-free-native-job-fresh-seed-ppo-array-smoke",
        "games": args.games,
        "unique_seeds": len(set(seeds)),
        "seed_start": args.seed_start,
        "seed_end": max(seeds),
        "opponents": {"thomas": opponents.count(1), "meta": opponents.count(2)},
        "threads": args.threads,
        "setup_seconds": setup_seconds,
        "prefix_seconds": prefix_seconds,
        "suffix_seconds": suffix["wall_seconds"],
        "actor_plan_seconds": suffix["plan_seconds"],
        "environment_seconds": suffix["environment_seconds"],
        "capture_seconds": capture_seconds,
        "array_save_seconds": save_seconds,
        "total_rollout_seconds": prefix_seconds + suffix["wall_seconds"],
        "games_per_second": args.games / (prefix_seconds + suffix["wall_seconds"]),
        "events": int(arrays["event_action"].shape[0]),
        "days": int(arrays["day_step"].shape[0]),
        "ppo_schema_games": len(games),
        "arrays": str(args.arrays.resolve()),
        "arrays_bytes": args.arrays.stat().st_size,
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "opening_switch_counts": {
            str(step): sum(row["opening_switch_step"] == step for row in summary)
            for step in (-1, 144, 168)
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
