#!/usr/bin/env python3
"""Audit the 64-candidate shortlist against every feasible C++ continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fast_kaggriculture import adaptive_default_genome, adaptive_genome_names
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


FAMILY_NAMES = [
    "KEEP",
    "SCHEDULE_LAYOUT",
    "CONTINUOUS_SCALE",
    "UNILATERAL",
    "MULTI_PROJECT",
    "TIMING",
    "MARKET_TRANSACTION",
    "PHASE_SUFFIX",
    "LOCAL_RECOVERY",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_genome(path: Path, index: int) -> np.ndarray:
    names = list(adaptive_genome_names())
    defaults = dict(zip(names, adaptive_default_genome(), strict=True))
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = dict(payload.get("base_values", {}))
    values.update(payload["genomes"][index]["values"])
    return np.asarray(
        [[float(values.get(name, defaults[name])) for name in names]],
        dtype=np.float64,
    )


def run_batch(executor, genomes: np.ndarray, tasks: list[list[int]]):
    raw_tasks = np.asarray(tasks, dtype=np.int64)
    started = time.perf_counter()
    rewards, diagnostics = executor.play_candidate8_batch(genomes, raw_tasks)
    return (
        raw_tasks,
        np.asarray(rewards, dtype=np.float64),
        np.asarray(diagnostics, dtype=np.int64),
        time.perf_counter() - started,
    )


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
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--minimum-decision-day", type=int, default=9)
    parser.add_argument("--single-seat", type=int, choices=(0, 1))
    parser.add_argument("--dataset-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    setup_started = time.perf_counter()
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    setup_seconds = time.perf_counter() - setup_started
    genomes = load_genome(args.genomes, args.genome_index)
    opponent = -1 if args.opponent.upper() == "PASSIVE" else bundle.index(args.opponent)
    seats = (args.single_seat,) if args.single_seat is not None else (0, 1)
    states = [
        (seed, seat)
        for seed in range(args.seed_start, args.seed_start + args.seed_count)
        for seat in seats
    ]

    probe_tasks = [
        [0, opponent, seed, seat, 0, args.minimum_decision_day, 1]
        for seed, seat in states
    ]
    _, probe_rewards, probe_diag, probe_seconds = run_batch(
        bundle.adaptive_executor, genomes, probe_tasks
    )
    state_meta = []
    for state_index, (seed, seat) in enumerate(states):
        diagnostic = probe_diag[state_index]
        if diagnostic[6] != 1 or diagnostic[11] <= 0:
            raise RuntimeError(
                f"candidate decision unavailable for seed={seed} seat={seat}: "
                f"{diagnostic.tolist()}"
            )
        state_meta.append(
            {
                "seed": seed,
                "seat": seat,
                "raw_count": int(diagnostic[10]),
                "feasible_count": int(diagnostic[11]),
                "shortlist_count": int(diagnostic[12]),
                "decision_day": int(diagnostic[7]),
            }
        )

    feasible_tasks: list[list[int]] = []
    shortlist_tasks: list[list[int]] = []
    feasible_owner: list[int] = []
    shortlist_owner: list[int] = []
    for state_index, meta in enumerate(state_meta):
        seed, seat = meta["seed"], meta["seat"]
        for rank in range(meta["feasible_count"]):
            feasible_tasks.append(
                [0, opponent, seed, seat, rank, args.minimum_decision_day, 1]
            )
            feasible_owner.append(state_index)
        for rank in range(meta["shortlist_count"]):
            shortlist_tasks.append(
                [0, opponent, seed, seat, rank, args.minimum_decision_day, 0]
            )
            shortlist_owner.append(state_index)

    feasible_raw, feasible_rewards, feasible_diag, feasible_seconds = run_batch(
        bundle.adaptive_executor, genomes, feasible_tasks
    )
    shortlist_raw, shortlist_rewards, shortlist_diag, shortlist_seconds = run_batch(
        bundle.adaptive_executor, genomes, shortlist_tasks
    )

    state_rows = []
    dataset_state = []
    dataset_seed = []
    dataset_seat = []
    dataset_signature = []
    dataset_family = []
    dataset_features = []
    dataset_reward = []
    dataset_gain = []
    dataset_in_shortlist = []
    all_hard_errors = np.zeros(3, dtype=np.int64)
    family_gains: dict[int, list[float]] = defaultdict(list)
    shortlist_hits = 0
    top8_hits = 0
    shortlist_regrets = []
    best_gains = []
    for state_index, meta in enumerate(state_meta):
        feasible_indices = [
            index for index, owner in enumerate(feasible_owner) if owner == state_index
        ]
        shortlist_indices = [
            index for index, owner in enumerate(shortlist_owner) if owner == state_index
        ]
        seat = meta["seat"]
        feasible_own = np.asarray(
            [feasible_rewards[index, seat] for index in feasible_indices]
        )
        shortlist_own = np.asarray(
            [shortlist_rewards[index, seat] for index in shortlist_indices]
        )
        feasible_signatures = [
            int(feasible_diag[index, 9]) for index in feasible_indices
        ]
        shortlist_signatures = [
            int(shortlist_diag[index, 9]) for index in shortlist_indices
        ]
        feasible_families = [
            int(feasible_diag[index, 8]) for index in feasible_indices
        ]
        best_feasible_local = int(np.argmax(feasible_own))
        best_shortlist_local = int(np.argmax(shortlist_own))
        best_signature = feasible_signatures[best_feasible_local]
        shortlist_rank = (
            shortlist_signatures.index(best_signature)
            if best_signature in shortlist_signatures
            else -1
        )
        shortlist_hits += shortlist_rank >= 0
        top8_hits += 0 <= shortlist_rank < 8
        regret = float(
            feasible_own[best_feasible_local] - shortlist_own[best_shortlist_local]
        )
        shortlist_regrets.append(regret)
        keep_reward = float(feasible_own[0])
        best_gain = float(feasible_own[best_feasible_local] - keep_reward)
        best_gains.append(best_gain)
        for local_index, family in enumerate(feasible_families):
            family_gains[family].append(float(feasible_own[local_index] - keep_reward))
            global_index = feasible_indices[local_index]
            dataset_state.append(state_index)
            dataset_seed.append(meta["seed"])
            dataset_seat.append(meta["seat"])
            dataset_signature.append(feasible_signatures[local_index])
            dataset_family.append(family)
            dataset_features.append(feasible_diag[global_index, 14:].copy())
            dataset_reward.append(float(feasible_own[local_index]))
            dataset_gain.append(float(feasible_own[local_index] - keep_reward))
            dataset_in_shortlist.append(
                feasible_signatures[local_index] in shortlist_signatures
            )
        hard_errors = np.asarray(
            [feasible_diag[index, 2:5] for index in feasible_indices],
            dtype=np.int64,
        ).sum(axis=0)
        all_hard_errors += hard_errors
        family_counts = Counter(feasible_families)
        state_rows.append(
            {
                **meta,
                "keep_reward": keep_reward,
                "best_feasible_reward": float(feasible_own[best_feasible_local]),
                "best_feasible_rank": best_feasible_local,
                "best_feasible_family": FAMILY_NAMES[
                    feasible_families[best_feasible_local]
                ],
                "best_feasible_signature": best_signature,
                "best_shortlist_reward": float(shortlist_own[best_shortlist_local]),
                "best_shortlist_rank": best_shortlist_local,
                "best_shortlist_family": FAMILY_NAMES[
                    int(shortlist_diag[shortlist_indices[best_shortlist_local], 8])
                ],
                "oracle_in_shortlist": shortlist_rank >= 0,
                "oracle_shortlist_rank": shortlist_rank,
                "shortlist_regret": regret,
                "best_gain_over_keep": best_gain,
                "feasible_by_family": {
                    FAMILY_NAMES[family]: int(count)
                    for family, count in sorted(family_counts.items())
                },
                "hard_errors": {
                    "avoidable_crop_losses": int(hard_errors[0]),
                    "avoidable_animal_losses": int(hard_errors[1]),
                    "end_overflow": int(hard_errors[2]),
                },
            }
        )

    family_summary = {}
    for family, gains in sorted(family_gains.items()):
        family_summary[FAMILY_NAMES[family]] = {
            "arms": len(gains),
            "mean_gain_vs_keep": statistics.fmean(gains),
            "median_gain_vs_keep": statistics.median(gains),
            "positive_rate": sum(gain > 0 for gain in gains) / len(gains),
            "best_gain": max(gains),
            "worst_gain": min(gains),
        }

    total_games = len(probe_tasks) + len(feasible_tasks) + len(shortlist_tasks)
    simulation_seconds = probe_seconds + feasible_seconds + shortlist_seconds
    payload = {
        "schema": "kaggriculture.candidate8_oracle_recall.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "opponent": args.opponent,
            "opponent_route": opponent,
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "seats": list(seats),
            "minimum_decision_day": args.minimum_decision_day,
            "online_shortlist_limit": 64,
            "threads": 16,
        },
        "summary": {
            "states": len(states),
            "raw_count_range": [
                min(row["raw_count"] for row in state_meta),
                max(row["raw_count"] for row in state_meta),
            ],
            "feasible_count_range": [
                min(row["feasible_count"] for row in state_meta),
                max(row["feasible_count"] for row in state_meta),
            ],
            "shortlist_count_range": [
                min(row["shortlist_count"] for row in state_meta),
                max(row["shortlist_count"] for row in state_meta),
            ],
            "oracle_recall_at_64": shortlist_hits / len(states),
            "oracle_recall_at_8_within_shortlist": top8_hits / len(states),
            "mean_shortlist_regret": statistics.fmean(shortlist_regrets),
            "max_shortlist_regret": max(shortlist_regrets),
            "mean_best_gain_over_keep": statistics.fmean(best_gains),
            "positive_best_gain_rate": sum(gain > 0 for gain in best_gains)
            / len(best_gains),
            "hard_errors": {
                "avoidable_crop_losses": int(all_hard_errors[0]),
                "avoidable_animal_losses": int(all_hard_errors[1]),
                "end_overflow": int(all_hard_errors[2]),
            },
            "total_complete_games": total_games,
            "setup_seconds": setup_seconds,
            "simulation_seconds": simulation_seconds,
            "complete_games_per_second": total_games / simulation_seconds,
        },
        "family_summary": family_summary,
        "states": state_rows,
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in {
                "source": args.source,
                "actions": args.actions,
                "metadata": args.metadata,
                "genomes": args.genomes,
            }.items()
        },
    }
    if args.dataset_output is not None:
        args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
        feature_names = [
            *(f"target_delta_{index}" for index in range(8)),
            "hand_delta",
            "quadrant_delta",
            "effective_delay_days",
            "schedule_profile",
            "market_profile",
            "recovery_profile",
            "suffix_project",
            "market_item",
            "recovery_issue",
            "estimated_value_x100",
            "estimated_cash_cost_x100",
            "estimated_daily_action_load_x100",
            "context_day",
            *(f"context_target_{index}" for index in range(8)),
            *(f"context_irreversible_floor_{index}" for index in range(8)),
            *(f"context_cap_{index}" for index in range(8)),
            *(f"context_marginal_value_x100_{index}" for index in range(8)),
            *(f"context_purchase_cost_x100_{index}" for index in range(8)),
            *(f"context_daily_action_load_x100_{index}" for index in range(8)),
            *(f"context_first_cash_lag_days_{index}" for index in range(8)),
            "context_liquid_cash_x100",
            "context_protected_cash_x100",
            "context_financeable_inventory_value_x100",
            "context_unlocked_quadrants",
            "context_maximum_quadrants",
            "context_hands",
            "context_maximum_hands",
            "context_next_hand_cost_x100",
            "context_next_quadrant_cost_x100",
            "context_tiles_per_quadrant",
            "context_productive_tiles",
            "context_current_daily_action_load_x100",
            "context_hard_deadline_load_x100",
            "context_estimated_travel_load_x100",
            "context_delayed_loss_x100",
            "context_market_slots_available",
            "context_recovery_issues",
            *(f"context_sellable_inventory_{index}" for index in range(9)),
            *(f"context_market_price_{index}" for index in range(9)),
            *(f"context_demand_within_day_{index}" for index in range(9)),
        ]
        if len(feature_names) != len(dataset_features[0]):
            raise RuntimeError(
                f"feature schema mismatch: names={len(feature_names)} "
                f"values={len(dataset_features[0])}"
            )
        np.savez_compressed(
            args.dataset_output,
            state_id=np.asarray(dataset_state, dtype=np.int32),
            seed=np.asarray(dataset_seed, dtype=np.int64),
            seat=np.asarray(dataset_seat, dtype=np.int8),
            signature=np.asarray(dataset_signature, dtype=np.int64),
            family=np.asarray(dataset_family, dtype=np.int8),
            features=np.asarray(dataset_features, dtype=np.int64),
            reward=np.asarray(dataset_reward, dtype=np.float64),
            gain_vs_keep=np.asarray(dataset_gain, dtype=np.float64),
            in_shortlist=np.asarray(dataset_in_shortlist, dtype=np.bool_),
            feature_names=np.asarray(feature_names),
        )
        payload["dataset"] = {
            "path": str(args.dataset_output),
            "rows": len(dataset_state),
            "feature_count": len(feature_names),
            "sha256": sha256(args.dataset_output),
        }
    if args.backbone is not None:
        payload["inputs"]["backbone"] = {
            "path": str(args.backbone),
            "sha256": sha256(args.backbone),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
