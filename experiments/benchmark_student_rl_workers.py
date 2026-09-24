#!/usr/bin/env python3
"""Measure rollout worker counts on exactly the same native PPO jobs."""

from __future__ import annotations

import argparse
import gc
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from experiments.train_midgame_student_v1 import _sha256
from experiments.train_student_action_event_rl_v3 import (
    ROOT,
    _collect_one,
    _jobs,
    _native_artifacts,
    _worker_init,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--games", type=int, default=384)
    parser.add_argument("--workers", default="192,128,192,128")
    parser.add_argument("--seed-start", type=int, default=2630000900)
    parser.add_argument("--policy-seed", type=int, default=2026092399)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worker_counts = [int(value) for value in args.workers.split(",")]
    if (args.games < 1 or not worker_counts or any(value < 1 for value in worker_counts)
            or args.output.exists()):
        parser.error("invalid games/workers or output already exists")
    for path in (args.checkpoint, args.manifest, args.binary):
        if not path.is_file():
            parser.error(f"missing {path}")

    opponents = ("thomas_2945_cpp", "metav4_2965")
    fingerprints = {
        "checkpoint_sha256": _sha256(args.checkpoint),
        "manifest_sha256": _sha256(args.manifest),
        "binary_sha256": _sha256(args.binary),
    }
    artifacts = _native_artifacts(opponents)
    jobs = _jobs(SimpleNamespace(
        games=args.games, opponents=opponents, seed_start=args.seed_start,
        policy_seed=args.policy_seed, checkpoint=args.checkpoint,
        manifest=args.manifest, binary=args.binary, temperature=1.0,
        margin_weight=0.1, margin_scale=10000.0,
        opponent_backend="native_cpp"), fingerprints, artifacts)

    runs = []
    for run_index, workers in enumerate(worker_counts):
        started = time.perf_counter()
        results = []
        with ProcessPoolExecutor(
                max_workers=min(workers, len(jobs)),
                initializer=_worker_init) as pool:
            futures = [pool.submit(_collect_one, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
        wall = time.perf_counter() - started
        failures = [row for row in results if row["status"] != "PASS"]
        if failures:
            raise RuntimeError(f"rollout benchmark failed: {failures[:3]}")
        row = {
            "run": run_index, "workers": workers, "games": len(results),
            "events": sum(len(day["events"]) for game in results
                          for day in game["days"]),
            "wall_seconds": wall, "games_per_second": len(results) / wall,
            "max_game_seconds": max(game["elapsed_seconds"] for game in results),
        }
        runs.append(row)
        print(json.dumps(row), flush=True)
        del results
        gc.collect()

    payload = {
        "schema": "student-rl-worker-ab-v1",
        "checkpoint_sha256": fingerprints["checkpoint_sha256"],
        "jobs": args.games,
        "seed_start": args.seed_start,
        "policy_seed": args.policy_seed,
        "opponents": list(opponents),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
