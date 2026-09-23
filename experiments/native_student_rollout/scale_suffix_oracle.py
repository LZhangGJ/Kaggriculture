#!/usr/bin/env python3
"""Scale the native prefix/planner/suffix seam with the suggested-action Oracle."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run_one(native, manifest: dict, size: int, threads: int) -> dict:
    base = manifest["cases"]
    caches = [base[index % len(base)]["cache"] for index in range(size)]
    total_started = time.perf_counter()
    cpu_started = time.process_time()
    construct_started = time.perf_counter()
    prefix = native.PrefixBatch(
        manifest["binary"], caches,
        str(ROOT / "experiments/native_opponents/thomas_2945_cpp/thomas_2945.assets.bin"),
        str(ROOT / "experiments/native_opponents/metav4_2965/metav4_2965.assets.bin"),
        str(ROOT / "agent/replay_deployment.json"))
    construct_seconds = time.perf_counter() - construct_started
    prefix_started = time.perf_counter()
    prefix.run(threads)
    prefix_seconds = time.perf_counter() - prefix_started
    plan_constructor = plan_callbacks = plan_join = interval_seconds = 0.0
    events = rounds = 0
    event_counts = np.zeros(size, dtype=np.int64)
    for step in range(288, 673, 24):
        started = time.perf_counter()
        plan = native.PlanBatch(
            manifest["binary"], prefix.handles, prefix.packed, False)
        plan_constructor += time.perf_counter() - started
        try:
            started = time.perf_counter()
            while True:
                ready = plan.collect_ready(60_000, True)
                indices = np.asarray(ready["session_indices"], dtype=np.int32)
                if indices.size == 0:
                    if bool(np.asarray(ready["terminal"]).all()):
                        break
                    raise TimeoutError(plan.summary())
                suggested = np.asarray(ready["suggested"], dtype=np.int32)
                event_counts[indices] += 1
                events += int(indices.size)
                rounds += 1
                plan.apply(indices, suggested)
            plan_callbacks += time.perf_counter() - started
            started = time.perf_counter()
            plan.join()
            plan_join += time.perf_counter() - started
            summary = plan.summary()
            if summary["states"] != ["done"] * size:
                raise RuntimeError(summary)
        finally:
            plan.close()
        started = time.perf_counter()
        prefix.advance_to(step + 24, threads)
        interval_seconds += time.perf_counter() - started
    tail_started = time.perf_counter()
    prefix.advance_to(719, threads)
    tail_seconds = time.perf_counter() - tail_started
    terminal = prefix.summary()["cases"]
    if not all(row["done"] and not row["error"] and
               row["suffix_steps"] == 431 for row in terminal):
        raise RuntimeError(terminal)
    wall = time.perf_counter() - total_started
    cpu = time.process_time() - cpu_started
    result = {
        "sessions": size,
        "source_cases": len(base),
        "cache_semantics": "four_real_cases_repeated_by_index",
        "unique_environment_seeds": len({row["seed"] for row in base}),
        "events": events,
        "event_count_min": int(event_counts.min()),
        "event_count_mean": float(event_counts.mean()),
        "event_count_max": int(event_counts.max()),
        "callback_rounds_total": rounds,
        "construct_seconds": construct_seconds,
        "prefix_0_288_seconds": prefix_seconds,
        "plan_constructor_seconds": plan_constructor,
        "plan_callback_install_seconds": plan_callbacks,
        "plan_join_seconds": plan_join,
        "planner_17_days_seconds": plan_constructor + plan_callbacks + plan_join,
        "env_opponent_288_696_seconds": interval_seconds,
        "tail_696_719_seconds": tail_seconds,
        "wall_seconds": wall,
        "process_cpu_seconds": cpu,
        "aggregate_cpu_cores": cpu / wall,
        "games_per_second": size / wall,
        "seconds_per_game": wall / size,
        "events_per_planner_second": events / (
            plan_constructor + plan_callbacks + plan_join),
        "terminal_wins": sum(row["own_cash"] > row["rival_cash"]
                             for row in terminal),
        "terminal_loss": sum(row["own_cash"] < row["rival_cash"]
                             for row in terminal),
    }
    del prefix
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=(
        ROOT / "work/native-student-rollout/prefix-cache-v1/manifest.json"))
    parser.add_argument("--batches", default="16,64,128,256")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--module-dir", type=Path, default=HERE / "build")
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/native-student-rollout/native-suffix-oracle-scaling.json"))
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    if (not batches or min(batches) < 1 or max(batches) > 256 or
            len(set(batches)) != len(batches)):
        parser.error("batches must be unique values in [1,256]")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sys.path.insert(0, str(args.module_dir.resolve()))
    native = importlib.import_module("_paused_plan")
    rows = []
    for size in batches:
        row = run_one(native, manifest, size, args.threads)
        rows.append(row)
        print(json.dumps({"event": "native_suffix_oracle_scale", **row}),
              flush=True)
    result = {
        "status": "PASS",
        "scope": "native-prefix-paused-oracle-native-suffix-scaling",
        "warning": ("Throughput-only replication: four real opponent/seed/seat "
                    "cases are repeated; this is not an evaluation sample."),
        "manifest": str(args.manifest.resolve()),
        "planner_binary_sha256": manifest["binary_sha256"],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"status": "PASS", "output": str(args.output)}))


if __name__ == "__main__":
    main()
