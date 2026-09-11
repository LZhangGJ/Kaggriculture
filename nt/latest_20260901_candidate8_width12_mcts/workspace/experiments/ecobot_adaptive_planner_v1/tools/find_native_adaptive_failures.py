#!/usr/bin/env python3
"""List per-game hard diagnostics for the native adaptive planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fast_kaggriculture import adaptive_default_genome
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    args = parser.parse_args()
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    genomes = np.asarray([adaptive_default_genome()], dtype=np.float64)
    tasks = np.asarray([
        [0, -1, seed, seat]
        for seed in range(args.seed_start, args.seed_start + args.seed_count)
        for seat in (0, 1)
    ], dtype=np.int64)
    rewards, diagnostics = bundle.adaptive_executor.play_batch(genomes, tasks)
    rows = []
    for task, reward, diag in zip(tasks, rewards, diagnostics):
        if int(diag[2]) or int(diag[3]) or int(diag[4]):
            seat = int(task[3])
            rows.append({
                "seed": int(task[2]),
                "seat": seat,
                "reward": float(reward[seat]),
                "crop_losses": int(diag[2]),
                "animal_losses": int(diag[3]),
                "end_overflow": int(diag[4]),
            })
    rows.sort(key=lambda row: (-row["animal_losses"], -row["end_overflow"], row["reward"]))
    print(json.dumps({"failures": rows, "count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
