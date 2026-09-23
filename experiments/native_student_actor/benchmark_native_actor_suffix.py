#!/usr/bin/env python3
"""Benchmark the all-C++ actor over complete native suffix games."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/native_student_rollout"
MANIFEST = ROOT / "work/native-student-rollout/prefix-cache-v1/manifest.json"
WEIGHTS = ROOT / "work/student-v1/native-r3-parent.bin"
THOMAS = ROOT / "experiments/native_opponents/thomas_2945_cpp/thomas_2945.assets.bin"
META = ROOT / "experiments/native_opponents/metav4_2965/metav4_2965.assets.bin"
DEPLOYMENT = ROOT / "agent/replay_deployment.json"


def run(native, manifest: dict, size: int, seed: int, threads: int) -> dict:
    cases = manifest["cases"]
    caches = [cases[index % len(cases)]["cache"] for index in range(size)]
    started = time.perf_counter()
    prefix = native.PrefixBatch(
        manifest["binary"], caches, str(THOMAS), str(META), str(DEPLOYMENT))
    prefix_started = time.perf_counter()
    prefix.run(threads)
    prefix_seconds = time.perf_counter() - prefix_started
    suffix_started = time.perf_counter()
    metrics = prefix.run_native_actor_suffix(
        str(WEIGHTS), seed, threads, 2 << 20, False)
    suffix_call_seconds = time.perf_counter() - suffix_started
    terminal = prefix.summary()["cases"]
    if (prefix.current_step != 719 or not all(
            row["done"] and not row["error"] and row["suffix_steps"] == 431
            and row["actor_days"] == 17 for row in terminal)):
        raise RuntimeError(prefix.summary())
    total_seconds = time.perf_counter() - started
    return {
        "sessions": size,
        "threads": threads,
        "events": metrics["events"],
        "prefix_seconds": prefix_seconds,
        "actor_plan_seconds": metrics["plan_seconds"],
        "environment_seconds": metrics["environment_seconds"],
        "suffix_seconds": metrics["wall_seconds"],
        "suffix_call_seconds_including_weight_load": suffix_call_seconds,
        "total_seconds_including_prefix": total_seconds,
        "suffix_games_per_second": size / metrics["wall_seconds"],
        "total_games_per_second": size / total_seconds,
        "actor_events_per_plan_second": (
            metrics["events"] / metrics["plan_seconds"]),
        "actor_events_per_suffix_second": (
            metrics["events"] / metrics["wall_seconds"]),
        "event_counts": [row["actor_events"] for row in terminal],
        "terminal_action_hashes": [row["action_hash"] for row in terminal],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--batches", default="16,64,128,192,256")
    parser.add_argument("--policy-seed", type=int, default=2026092301)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-threads", type=int, default=256)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/native-student-rollout/native-actor-full-scaling.json"))
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    if (not batches or min(batches) < 1 or max(batches) > 256 or
            args.repeats < 1 or args.max_threads < 1 or args.output.exists()):
        parser.error("invalid batch/repeat/thread setting or existing output")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sys.path.insert(0, str(HERE / "build"))
    native = importlib.import_module("_paused_plan")
    rows = []
    for size in batches:
        runs = []
        for repeat in range(args.repeats):
            row = run(native, manifest, size, args.policy_seed,
                      min(size, args.max_threads))
            row["repeat"] = repeat
            runs.append(row)
            print(json.dumps({"event": "native_actor_suffix", **row}),
                  flush=True)
        median = statistics.median(run["suffix_seconds"] for run in runs)
        rows.append({
            "sessions": size,
            "median_suffix_seconds": median,
            "median_suffix_games_per_second": size / median,
            "runs": runs,
        })
    result = {
        "status": "PASS",
        "scope": "all-cpp-r3-actor-full-719-suffix-scaling",
        "batches": batches,
        "repeats": args.repeats,
        "policy_seed": args.policy_seed,
        "host_logical_cpus": os.cpu_count(),
        "shared_read_only_weights": True,
        "per_session_state": "hidden+previous+splitmix64-counter",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "output": str(args.output)}))


if __name__ == "__main__":
    main()
