#!/usr/bin/env python3
"""Verify R8 match-state isolation and native latency on frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fast_kaggriculture import adaptive_default_genome, adaptive_genome_names
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_genome(path: Path, index: int) -> np.ndarray:
    names = list(adaptive_genome_names())
    values = dict(zip(names, adaptive_default_genome(), strict=True))
    payload = json.loads(path.read_text(encoding="utf-8"))
    values.update(payload.get("base_values", {}))
    values.update(payload["genomes"][index].get("values", {}))
    return np.asarray([float(values[name]) for name in names], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", type=Path, required=True)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--opponent", default="PASSIVE")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--latency-games", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)
    genomes = genome[None, :]
    opponent = -1 if args.opponent.upper() == "PASSIVE" else bundle.index(args.opponent)
    tasks = np.asarray([
        [0, opponent, args.seed_start + i // 2, i % 2]
        for i in range(args.games)
    ], dtype=np.int64)

    config = (24, 8, 8, 4, 0.01)
    started = time.perf_counter()
    rewards_a_raw, diag_a_raw = bundle.adaptive_executor.play_r8_batch(
        genomes, tasks, *config
    )
    seconds_a = time.perf_counter() - started
    reverse_order = np.arange(len(tasks) - 1, -1, -1)
    started = time.perf_counter()
    rewards_b_raw, diag_b_raw = bundle.adaptive_executor.play_r8_batch(
        genomes, tasks[reverse_order], *config
    )
    seconds_b = time.perf_counter() - started
    rewards_a = np.asarray(rewards_a_raw)
    diag_a = np.asarray(diag_a_raw)
    rewards_b = np.asarray(rewards_b_raw)[reverse_order]
    diag_b = np.asarray(diag_b_raw)[reverse_order]

    latency_ms: list[float] = []
    for row in tasks[: min(args.latency_games, len(tasks))]:
        started = time.perf_counter()
        bundle.adaptive_executor.play_r8(
            genome, int(row[1]), int(row[2]), int(row[3]), *config, False
        )
        latency_ms.append(1000.0 * (time.perf_counter() - started))

    reward_mismatches = int(np.count_nonzero(rewards_a != rewards_b))
    diagnostic_mismatches = int(np.count_nonzero(diag_a != diag_b))
    hard_failures = int(diag_a[:, 0:3].sum() + diag_a[:, 17:19].sum())
    payload = {
        "schema": "kaggriculture.r8-state-isolation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if reward_mismatches == 0 and
            diagnostic_mismatches == 0 and hard_failures == 0 else "FAIL",
        "opponent": args.opponent,
        "games": int(len(tasks)),
        "seed_start": args.seed_start,
        "config": {
            "day_horizon_steps": config[0],
            "joint_horizon_steps": config[1],
            "joint_candidate_limit": config[2],
            "feature_level": config[3],
            "lookahead_scale": config[4],
        },
        "isolation": {
            "reward_value_mismatches": reward_mismatches,
            "diagnostic_value_mismatches": diagnostic_mismatches,
            "hard_failure_count": hard_failures,
            "ascending_seconds": seconds_a,
            "reverse_seconds": seconds_b,
            "ascending_games_per_second": len(tasks) / seconds_a,
            "reverse_games_per_second": len(tasks) / seconds_b,
        },
        "single_game_native_latency": {
            "samples": len(latency_ms),
            "p50_ms_per_719_step_game": float(np.quantile(latency_ms, 0.50)),
            "p95_ms_per_719_step_game": float(np.quantile(latency_ms, 0.95)),
            "p50_us_per_environment_step": float(
                1000.0 * np.quantile(latency_ms, 0.50) / 719.0
            ),
            "p95_us_per_environment_step": float(
                1000.0 * np.quantile(latency_ms, 0.95) / 719.0
            ),
            "scope": "native full-game call; excludes online Python observation parsing",
        },
        "inputs": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "genomes": {"path": str(args.genomes), "sha256": sha256(args.genomes)},
            "backbone": None if args.backbone is None else {
                "path": str(args.backbone), "sha256": sha256(args.backbone)
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
