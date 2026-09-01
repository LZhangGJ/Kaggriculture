#!/usr/bin/env python3
"""Counterfactually compare KEEP and multiple SWITCH candidates at one state."""

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

from calibrate_portfolio_switch_policy import SWITCH_FEATURE_NAMES


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_genome(path: Path, index: int, names: list[str]) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = dict(zip(names, adaptive_default_genome(), strict=True))
    values.update({
        str(name): float(value)
        for name, value in payload.get("base_values", {}).items()
    })
    values.update({
        str(name): float(value)
        for name, value in payload["genomes"][index]["values"].items()
    })
    return np.asarray([values[name] for name in names], dtype=np.float64)


def quantiles(values: np.ndarray) -> dict[str, float]:
    if not values.size:
        return {key: 0.0 for key in ("mean", "p10", "median", "p90", "min", "max")}
    return {
        "mean": float(values.mean()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", type=Path, required=True)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--switch-margin", type=float, default=0.05)
    parser.add_argument("--candidate-ranks", type=int, default=8)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, required=True)
    args = parser.parse_args()

    names = list(adaptive_genome_names())
    base = load_genome(args.genomes, args.genome_index, names)
    switch_column = names.index("portfolio_bundle_switch_margin")
    cooldown_column = names.index("portfolio_switch_cooldown_days")
    rank_column = names.index("portfolio_switch_candidate_rank")
    base[switch_column] = 0.0
    arms = [base.copy()]
    for rank in range(args.candidate_ranks):
        candidate = base.copy()
        candidate[switch_column] = args.switch_margin
        candidate[cooldown_column] = 30.0
        candidate[rank_column] = float(rank)
        arms.append(candidate)
    genomes = np.stack(arms)

    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    tasks = np.asarray([
        [arm, -1, seed, seat]
        for arm in range(len(arms))
        for seed in range(args.seed_start, args.seed_start + args.seed_count)
        for seat in (0, 1)
    ], dtype=np.int64)
    started = time.perf_counter()
    raw_rewards, raw_diagnostics = bundle.adaptive_executor.play_batch(genomes, tasks)
    elapsed = time.perf_counter() - started
    rewards = np.asarray(raw_rewards, dtype=np.float64)
    diagnostics = np.asarray(raw_diagnostics, dtype=np.int32)
    games_per_arm = args.seed_count * 2
    feature_dim = len(SWITCH_FEATURE_NAMES)
    if diagnostics.shape[1] < 22 + feature_dim:
        raise RuntimeError("C++ diagnostics do not expose the expected switch features")

    arm_rewards = []
    arm_diagnostics = []
    for arm in range(len(arms)):
        begin = arm * games_per_arm
        end = begin + games_per_arm
        arm_tasks = tasks[begin:end]
        arm_rewards.append(np.asarray([
            rewards[begin + index, task[3]]
            for index, task in enumerate(arm_tasks)
        ], dtype=np.float64))
        arm_diagnostics.append(diagnostics[begin:end])
    base_reward = arm_rewards[0]

    # Indices that describe the pre-edit state, excluding candidate identity,
    # candidate targets, analytic scores and candidate-derived workload.
    state_indices = np.asarray(
        list(range(0, 11)) + list(range(21, 37)) + list(range(53, 72)) +
        list(range(80, 98)) + list(range(106, 112)),
        dtype=np.int64,
    )
    rows_features: list[np.ndarray] = []
    rows_delta: list[float] = []
    rows_rank: list[int] = []
    rows_seed: list[int] = []
    rows_seat: list[int] = []
    groups = 0
    complete_groups = 0
    state_aligned_candidates = 0
    pairwise_correct = 0
    pairwise_total = 0
    top1_oracle = 0
    top3_oracle = 0
    keep_oracle = 0

    for index in range(games_per_arm):
        reference_features: np.ndarray | None = None
        candidates: list[tuple[int, float, float, np.ndarray]] = []
        for rank in range(args.candidate_ranks):
            diag = arm_diagnostics[rank + 1][index]
            if diag[19] <= 0:
                continue
            features = diag[22:22 + feature_dim].copy()
            if reference_features is None:
                reference_features = features
            elif not np.array_equal(
                features[state_indices], reference_features[state_indices]
            ):
                continue
            delta = float(arm_rewards[rank + 1][index] - base_reward[index])
            analytic_gain = float(features[16] - features[15])
            candidates.append((rank, delta, analytic_gain, features))
            rows_features.append(features)
            rows_delta.append(delta)
            rows_rank.append(rank)
            rows_seed.append(int(tasks[index, 2]))
            rows_seat.append(int(tasks[index, 3]))
            state_aligned_candidates += 1
        if not candidates:
            continue
        groups += 1
        if len(candidates) == args.candidate_ranks:
            complete_groups += 1
        actual_values = np.asarray([candidate[1] for candidate in candidates])
        analytic_values = np.asarray([candidate[2] for candidate in candidates])
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                actual_sign = np.sign(actual_values[left] - actual_values[right])
                analytic_sign = np.sign(analytic_values[left] - analytic_values[right])
                if actual_sign == 0:
                    continue
                pairwise_total += 1
                pairwise_correct += actual_sign == analytic_sign
        oracle_position = int(np.argmax(np.r_[0.0, actual_values]))
        if oracle_position == 0:
            keep_oracle += 1
            # KEEP is always explicitly retained by the analytic shortlist.
            top1_oracle += 1
            top3_oracle += 1
        else:
            best_candidate_index = oracle_position - 1
            analytic_order = np.argsort(-analytic_values)
            top1_oracle += best_candidate_index == int(analytic_order[0])
            top3_oracle += best_candidate_index in analytic_order[:3]

    feature_matrix = np.asarray(rows_features, dtype=np.int32)
    delta_vector = np.asarray(rows_delta, dtype=np.float64)
    rank_vector = np.asarray(rows_rank, dtype=np.int16)
    seed_vector = np.asarray(rows_seed, dtype=np.int64)
    seat_vector = np.asarray(rows_seat, dtype=np.int8)
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.dataset_output,
        features=feature_matrix,
        actual_delta=delta_vector,
        candidate_rank=rank_vector,
        seed=seed_vector,
        seat=seat_vector,
        feature_names=np.asarray(SWITCH_FEATURE_NAMES),
    )

    payload = {
        "schema": "kaggriculture.portfolio-candidate-ranking-calibration.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "seat_swap": True,
        "candidate_ranks": args.candidate_ranks,
        "switch_margin": args.switch_margin,
        "games": int(len(tasks)),
        "simulation_seconds": elapsed,
        "games_per_second": float(len(tasks) / elapsed),
        "state_aligned_groups": groups,
        "complete_groups": complete_groups,
        "state_aligned_candidate_rows": state_aligned_candidates,
        "analytic_pairwise_accuracy": (
            pairwise_correct / pairwise_total if pairwise_total else 0.0
        ),
        "pairwise_comparisons": pairwise_total,
        "oracle_keep_rate": keep_oracle / groups if groups else 0.0,
        "analytic_top1_oracle_recall": top1_oracle / groups if groups else 0.0,
        "analytic_top3_oracle_recall": top3_oracle / groups if groups else 0.0,
        "candidate_delta": quantiles(delta_vector),
        "hard_failures": {
            str(arm): int(arm_diagnostics[arm][:, 2:5].sum())
            for arm in range(len(arms))
        },
        "inputs": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "genomes": {"path": str(args.genomes), "sha256": sha256(args.genomes)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "games": payload["games"],
        "games_per_second": payload["games_per_second"],
        "state_aligned_groups": groups,
        "complete_groups": complete_groups,
        "candidate_rows": state_aligned_candidates,
        "pairwise_accuracy": payload["analytic_pairwise_accuracy"],
        "top1_oracle_recall": payload["analytic_top1_oracle_recall"],
        "top3_oracle_recall": payload["analytic_top3_oracle_recall"],
        "hard_failures": payload["hard_failures"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
