#!/usr/bin/env python3
"""Paired R6/R8 executor ablation on identical seeds, seats and macro genome."""

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


def summarize(
    rewards: np.ndarray,
    diagnostics: np.ndarray,
    tasks: np.ndarray,
    seconds: float,
    r8: bool,
) -> dict[str, object]:
    own = rewards[np.arange(len(tasks)), tasks[:, 3]]
    rival = rewards[np.arange(len(tasks)), 1 - tasks[:, 3]]
    margin = own - rival
    row: dict[str, object] = {
        "games": int(len(tasks)),
        "seconds": float(seconds),
        "games_per_second": float(len(tasks) / seconds),
        "mean_own_cash": float(own.mean()),
        "median_own_cash": float(np.median(own)),
        "p10_own_cash": float(np.quantile(own, 0.10)),
        "mean_opponent_cash": float(rival.mean()),
        "mean_margin": float(margin.mean()),
        "wins": int(np.count_nonzero(margin > 0)),
        "ties": int(np.count_nonzero(margin == 0)),
        "losses": int(np.count_nonzero(margin < 0)),
        "score_rate": float(np.mean((margin > 0) + 0.5 * (margin == 0))),
        "avoidable_crop_losses": int(diagnostics[:, 2 if not r8 else 0].sum()),
        "avoidable_animal_losses": int(diagnostics[:, 3 if not r8 else 1].sum()),
        "end_overflow": int(diagnostics[:, 4 if not r8 else 2].sum()),
    }
    if r8:
        row.update({
            "mean_day_plan_rebuilds": float(diagnostics[:, 5].mean()),
            "mean_rolling_updates": float(diagnostics[:, 6].mean()),
            "mean_task_nodes_created": float(diagnostics[:, 7].mean()),
            "mean_task_reassignments": float(diagnostics[:, 8].mean()),
            "mean_joint_matches": float(diagnostics[:, 9].mean()),
            "mean_lookahead_evaluations": float(diagnostics[:, 10].mean()),
            "mean_idle_unit_actions": float(diagnostics[:, 11].mean()),
            "mean_peak_active_tasks": float(diagnostics[:, 12].mean()),
            "mean_move_unit_actions": float(diagnostics[:, 13].mean()),
            "mean_resolved_task_nodes": float(diagnostics[:, 14].mean()),
            "mean_task_resolution_p50_steps": float(diagnostics[:, 15].mean()),
            "mean_task_resolution_p95_steps": float(diagnostics[:, 16].mean()),
            "unresolved_hard_day_tasks": int(diagnostics[:, 17].sum()),
            "duplicate_reservation_violations": int(diagnostics[:, 18].sum()),
        })
    return row


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
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--day-horizon", type=int, default=24)
    parser.add_argument("--joint-horizon", type=int, default=8)
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--feature-level", type=int, default=4)
    parser.add_argument("--lookahead-scale", type=float, default=0.01)
    parser.add_argument(
        "--sequence-solver", choices=("beam", "exact"), default="exact"
    )
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)
    genomes = genome[None, :]
    opponent = -1 if args.opponent.upper() == "PASSIVE" else bundle.index(args.opponent)
    tasks = np.asarray([
        [0, opponent, seed, seat]
        for seed in range(args.seed_start, args.seed_start + args.seed_count)
        for seat in (0, 1)
    ], dtype=np.int64)

    started = time.perf_counter()
    r6_rewards_raw, r6_diag_raw = bundle.adaptive_executor.play_batch(genomes, tasks)
    r6_seconds = time.perf_counter() - started
    started = time.perf_counter()
    r8_rewards_raw, r8_diag_raw = bundle.adaptive_executor.play_r8_batch(
        genomes, tasks, args.day_horizon, args.joint_horizon,
        args.candidate_limit, args.feature_level, args.lookahead_scale,
        0 if args.sequence_solver == "beam" else 1, args.beam_width,
    )
    r8_seconds = time.perf_counter() - started
    r6_rewards = np.asarray(r6_rewards_raw, dtype=np.float64)
    r8_rewards = np.asarray(r8_rewards_raw, dtype=np.float64)
    r6_diag = np.asarray(r6_diag_raw, dtype=np.int64)
    r8_diag = np.asarray(r8_diag_raw, dtype=np.int64)
    own_indices = tasks[:, 3]
    row_indices = np.arange(len(tasks))
    r6_own = r6_rewards[row_indices, own_indices]
    r8_own = r8_rewards[row_indices, own_indices]
    r6_margin = r6_own - r6_rewards[row_indices, 1 - own_indices]
    r8_margin = r8_own - r8_rewards[row_indices, 1 - own_indices]
    cash_delta = r8_own - r6_own
    margin_delta = r8_margin - r6_margin

    payload = {
        "schema": "kaggriculture.r8-executor-ablation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if np.isfinite(r8_rewards).all() else "FAIL",
        "opponent": args.opponent,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "games": len(tasks),
        "r8_config": {
            "day_horizon_steps": args.day_horizon,
            "joint_horizon_steps": args.joint_horizon,
            "joint_candidate_limit": args.candidate_limit,
            "feature_level": args.feature_level,
            "lookahead_scale": args.lookahead_scale,
            "sequence_solver": args.sequence_solver,
            "beam_width": args.beam_width,
        },
        "r6": summarize(r6_rewards, r6_diag, tasks, r6_seconds, False),
        "r8": summarize(r8_rewards, r8_diag, tasks, r8_seconds, True),
        "paired": {
            "mean_own_cash_delta": float(cash_delta.mean()),
            "median_own_cash_delta": float(np.median(cash_delta)),
            "mean_margin_delta": float(margin_delta.mean()),
            "improved_cash_states": int(np.count_nonzero(cash_delta > 0)),
            "equal_cash_states": int(np.count_nonzero(cash_delta == 0)),
            "regressed_cash_states": int(np.count_nonzero(cash_delta < 0)),
            "maximum_cash_gain": float(cash_delta.max()),
            "maximum_cash_regression": float(cash_delta.min()),
        },
        "diagnostic_states": {
            "r6_crop_loss": [
                {"seed": int(tasks[i, 2]), "seat": int(tasks[i, 3]),
                 "count": int(r6_diag[i, 2])}
                for i in np.flatnonzero(r6_diag[:, 2] > 0)
            ],
            "r8_crop_loss": [
                {"seed": int(tasks[i, 2]), "seat": int(tasks[i, 3]),
                 "count": int(r8_diag[i, 0])}
                for i in np.flatnonzero(r8_diag[:, 0] > 0)
            ],
            "r6_animal_loss": [
                {"seed": int(tasks[i, 2]), "seat": int(tasks[i, 3]),
                 "count": int(r6_diag[i, 3])}
                for i in np.flatnonzero(r6_diag[:, 3] > 0)
            ],
            "r8_animal_loss": [
                {"seed": int(tasks[i, 2]), "seat": int(tasks[i, 3]),
                 "count": int(r8_diag[i, 1])}
                for i in np.flatnonzero(r8_diag[:, 1] > 0)
            ],
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
    print(json.dumps({
        "status": payload["status"], "r6": payload["r6"],
        "r8": payload["r8"], "paired": payload["paired"],
    }, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
