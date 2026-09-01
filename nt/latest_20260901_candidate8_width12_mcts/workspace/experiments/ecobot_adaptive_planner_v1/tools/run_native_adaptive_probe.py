#!/usr/bin/env python3
"""Run the native general adaptive planner against passive or a route family."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fast_kaggriculture import adaptive_default_genome, adaptive_genome_names
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", type=Path)
    parser.add_argument(
        "--override-grid",
        help="Clone one base genome over NAME=V1,V2,... for a controlled A/B grid.",
    )
    parser.add_argument(
        "--base-override",
        action="append",
        default=[],
        help="Set NAME=VALUE on every genome before the controlled grid; repeatable.",
    )
    parser.add_argument("--base-genome-index", type=int, default=0)
    parser.add_argument("--opponent", default="PASSIVE")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--single-seat", type=int, choices=(0, 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    setup_started = time.perf_counter()
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    setup_seconds = time.perf_counter() - setup_started
    names = list(adaptive_genome_names())
    if args.genomes is None:
        genomes = np.asarray([adaptive_default_genome()], dtype=np.float64)
        genome_labels = ["default"]
    else:
        payload = json.loads(args.genomes.read_text(encoding="utf-8"))
        genome_labels = [str(row.get("label", f"g{index}")) for index, row in enumerate(payload["genomes"])]
        defaults = dict(zip(names, adaptive_default_genome(), strict=True))
        # A grid file may define one shared champion/base configuration and
        # keep each candidate row limited to its true controlled differences.
        # This avoids silently falling back to the C++ struct defaults when an
        # experiment intends to extend an already accepted semantic scheduler.
        defaults.update({
            str(name): float(value)
            for name, value in payload.get("base_values", {}).items()
        })
        genomes = np.asarray(
            [[float(row["values"].get(name, defaults[name])) for name in names]
             for row in payload["genomes"]],
            dtype=np.float64,
        )
    for raw_override in args.base_override:
        name, raw_value = raw_override.split("=", 1)
        if name not in names:
            raise ValueError(f"unknown genome field: {name}")
        genomes[:, names.index(name)] = float(raw_value)
    if args.override_grid:
        name, raw_values = args.override_grid.split("=", 1)
        if name not in names:
            raise ValueError(f"unknown genome field: {name}")
        if not (0 <= args.base_genome_index < len(genomes)):
            raise ValueError("base-genome-index out of range")
        values = [float(value) for value in raw_values.split(",")]
        base = genomes[args.base_genome_index].copy()
        column = names.index(name)
        genomes = np.tile(base, (len(values), 1))
        genomes[:, column] = values
        base_label = genome_labels[args.base_genome_index]
        genome_labels = [f"{base_label}__{name}_{value:g}" for value in values]
    if genomes.ndim != 2 or genomes.shape[1] != len(names):
        raise ValueError((genomes.shape, len(names)))
    opponent_index = -1 if args.opponent.upper() == "PASSIVE" else bundle.index(args.opponent)
    seats = (args.single_seat,) if args.single_seat is not None else (0, 1)
    tasks = np.asarray([
        [genome, opponent_index, seed, seat]
        for genome in range(len(genomes))
        for seed in range(args.seed_start, args.seed_start + args.seed_count)
        for seat in seats
    ], dtype=np.int64)
    started = time.perf_counter()
    rewards_raw, diagnostics_raw = bundle.adaptive_executor.play_batch(genomes, tasks)
    seconds = time.perf_counter() - started
    rewards = np.asarray(rewards_raw, dtype=np.float64)
    diagnostics = np.asarray(diagnostics_raw, dtype=np.int32)
    rows = []
    for genome, label in enumerate(genome_labels):
        indices = np.flatnonzero(tasks[:, 0] == genome)
        own = np.asarray([rewards[i, tasks[i, 3]] for i in indices], dtype=np.float64)
        rival = np.asarray([rewards[i, 1 - tasks[i, 3]] for i in indices], dtype=np.float64)
        margin = own - rival
        diag = diagnostics[indices]
        max_animal_targets = (
            diag[:, 6:9] if diag.shape[1] >= 9
            else np.zeros((len(diag), 3), dtype=np.int32)
        )
        max_crop_targets = (
            diag[:, 9:14] if diag.shape[1] >= 14
            else np.zeros((len(diag), 5), dtype=np.int32)
        )
        final_animals = (
            diag[:, 14:17] if diag.shape[1] >= 17
            else np.zeros((len(diag), 3), dtype=np.int32)
        )
        final_prior = (
            diag[:, 17] if diag.shape[1] >= 18
            else np.full(len(diag), -2, dtype=np.int32)
        )
        prior_switches = (
            diag[:, 18] if diag.shape[1] >= 19
            else np.zeros(len(diag), dtype=np.int32)
        )
        animal_profiles = Counter(
            tuple(int(value) for value in profile) for profile in max_animal_targets
        )
        failures = []
        for index, diagnostic in zip(indices, diag, strict=True):
            if (int(diagnostic[2]) == 0 and int(diagnostic[3]) == 0 and
                    int(diagnostic[4]) == 0):
                continue
            failures.append({
                "seed": int(tasks[index, 2]),
                "seat": int(tasks[index, 3]),
                "reward": float(rewards[index, tasks[index, 3]]),
                "avoidable_crop_losses": int(diagnostic[2]),
                "avoidable_animal_losses": int(diagnostic[3]),
                "end_overflow": int(diagnostic[4]),
            })
        ranked_indices = sorted(
            indices,
            key=lambda index: float(rewards[index, tasks[index, 3]]),
        )

        def reward_example(index: int) -> dict[str, object]:
            return {
                "seed": int(tasks[index, 2]),
                "seat": int(tasks[index, 3]),
                "reward": float(rewards[index, tasks[index, 3]]),
                "opponent_reward": float(rewards[index, 1 - tasks[index, 3]]),
            }

        rows.append({
            "genome_index": genome,
            "label": label,
            "games": len(indices),
            "mean_reward": float(own.mean()),
            "median_reward": float(np.median(own)),
            "p10_reward": float(np.quantile(own, 0.10)),
            "p90_reward": float(np.quantile(own, 0.90)),
            "min_reward": float(own.min()),
            "max_reward": float(own.max()),
            "mean_opponent_reward": float(rival.mean()),
            "wins": int(np.count_nonzero(margin > 0)),
            "ties": int(np.count_nonzero(margin == 0)),
            "losses": int(np.count_nonzero(margin < 0)),
            "score_rate": float(np.mean((margin > 0) + 0.5 * (margin == 0))),
            "mean_margin": float(margin.mean()),
            "mean_replans": float(diag[:, 0].mean()),
            "mean_non_pass_operations": float(diag[:, 1].mean()),
            "avoidable_crop_losses": int(diag[:, 2].sum()),
            "avoidable_animal_losses": int(diag[:, 3].sum()),
            "end_overflow": int(diag[:, 4].sum()),
            "mean_max_animal_targets": [float(value) for value in max_animal_targets.mean(axis=0)],
            "mean_max_crop_targets": [float(value) for value in max_crop_targets.mean(axis=0)],
            "mean_final_animals": [float(value) for value in final_animals.mean(axis=0)],
            "mean_operating_prior_switches": float(prior_switches.mean()),
            "final_operating_prior_distribution": {
                str(int(value)): int(np.count_nonzero(final_prior == value))
                for value in np.unique(final_prior)
            },
            "distinct_max_animal_profiles": len(animal_profiles),
            "top_max_animal_profiles": [
                {"targets": list(profile), "games": count}
                for profile, count in animal_profiles.most_common(12)
            ],
            "hard_failure_examples": failures[:32],
            "lowest_reward_examples": [
                reward_example(index) for index in ranked_indices[:5]
            ],
            "highest_reward_examples": [
                reward_example(index) for index in ranked_indices[-5:][::-1]
            ],
        })
    rows.sort(key=lambda row: (-float(row["mean_reward"]), -float(row["p10_reward"])))
    payload = {
        "schema": "kaggriculture.native-adaptive-probe.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(math.isfinite(value) for value in rewards.flat) else "FAIL",
        "engine": "C++ NativeAdaptiveExecutor.play_batch",
        "policy_semantics": "general public-state daily/event planning; no opponent identity input",
        "opponent": args.opponent,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "seats": list(seats),
        "games": len(tasks),
        "setup_seconds": setup_seconds,
        "simulation_seconds": seconds,
        "games_per_second": len(tasks) / seconds if seconds else 0.0,
        "genome_names": names,
        "genomes": genomes.tolist(),
        "ranking": rows,
        "inputs": {
            "source": {"path": str(args.source), "sha256": _sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": _sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": _sha256(args.metadata)},
            "backbone": None if args.backbone is None else {
                "path": str(args.backbone), "sha256": _sha256(args.backbone)
            },
            "genome_file": None if args.genomes is None else {
                "path": str(args.genomes), "sha256": _sha256(args.genomes)
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "games": len(tasks),
        "simulation_seconds": seconds, "games_per_second": payload["games_per_second"],
        "ranking": rows,
    }, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
