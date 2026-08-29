#!/usr/bin/env python3
"""Audit Candidate8 with several future RNG suffixes per frozen state."""

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
        [float(values.get(name, defaults[name])) for name in names],
        dtype=np.float64,
    )


def candidate_feature_names() -> list[str]:
    return [
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
    ]


def context_feature_names() -> list[str]:
    return [
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", type=Path, required=True)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--opponent", default="PASSIVE")
    parser.add_argument("--prefix-seed-start", type=int, required=True)
    parser.add_argument("--prefix-seed-count", type=int, default=16)
    parser.add_argument("--future-seed-start", type=int, required=True)
    parser.add_argument("--future-count", type=int, default=8)
    parser.add_argument("--minimum-decision-day", type=int, default=9)
    parser.add_argument("--single-seat", type=int, choices=(0, 1))
    parser.add_argument("--maximum-arms", type=int, default=4096)
    parser.add_argument("--dataset-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    setup_started = time.perf_counter()
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    setup_seconds = time.perf_counter() - setup_started
    genome = load_genome(args.genomes, args.genome_index)
    opponent = -1 if args.opponent.upper() == "PASSIVE" else bundle.index(args.opponent)
    seats = (args.single_seat,) if args.single_seat is not None else (0, 1)

    state_rows: list[dict] = []
    dataset_state: list[int] = []
    dataset_prefix_seed: list[int] = []
    dataset_seat: list[int] = []
    dataset_signature: list[int] = []
    dataset_family: list[int] = []
    dataset_features: list[np.ndarray] = []
    dataset_expected_reward: list[float] = []
    dataset_reward_std: list[float] = []
    dataset_gain: list[float] = []
    dataset_samples: list[np.ndarray] = []
    dataset_in_shortlist: list[bool] = []
    family_gains: dict[int, list[float]] = defaultdict(list)
    total_games = 0
    simulation_seconds = 0.0
    total_overflow = 0

    state_id = 0
    for prefix_offset in range(args.prefix_seed_count):
        prefix_seed = args.prefix_seed_start + prefix_offset
        for seat in seats:
            # Different frozen prefixes receive independent future banks, but
            # every arm within one state shares exactly the same bank.
            future_base = args.future_seed_start + state_id * args.future_count
            future_seeds = [
                future_base + index for index in range(args.future_count)
            ]
            started = time.perf_counter()
            feasible = bundle.adaptive_executor.candidate8_counterfactual(
                genome,
                opponent,
                prefix_seed,
                future_seeds,
                seat,
                args.minimum_decision_day,
                True,
                args.maximum_arms,
            )
            shortlist = bundle.adaptive_executor.candidate8_counterfactual(
                genome,
                opponent,
                prefix_seed,
                future_seeds,
                seat,
                args.minimum_decision_day,
                False,
                64,
            )
            simulation_seconds += time.perf_counter() - started
            if not feasible["decision_found"] or not shortlist["decision_found"]:
                raise RuntimeError(
                    f"candidate decision unavailable for prefix={prefix_seed} "
                    f"seat={seat}"
                )

            rewards = np.asarray(feasible["rewards"], dtype=np.float64)
            shortlist_rewards = np.asarray(shortlist["rewards"], dtype=np.float64)
            signatures = np.asarray(feasible["arm_signature"], dtype=np.uint64)
            shortlist_signatures = set(
                np.asarray(shortlist["arm_signature"], dtype=np.uint64).tolist()
            )
            families = np.asarray(feasible["arm_family"], dtype=np.int8)
            arm_features = np.asarray(feasible["arm_features"], dtype=np.int32)
            context = np.asarray(feasible["context_features"], dtype=np.int32)
            expected = rewards.mean(axis=0)
            reward_std = rewards.std(axis=0)
            shortlist_expected = shortlist_rewards.mean(axis=0)
            oracle = int(np.argmax(expected))
            shortlist_best = int(np.argmax(shortlist_expected))
            keep_reward = float(expected[0])
            oracle_signature = int(signatures[oracle])
            oracle_in_shortlist = oracle_signature in shortlist_signatures
            regret = float(expected[oracle] - shortlist_expected[shortlist_best])
            best_gain = float(expected[oracle] - keep_reward)
            near_500 = regret <= 500.0
            near_1000 = regret <= 1000.0
            overflow = int(np.asarray(feasible["end_overflow"]).sum())
            total_overflow += overflow
            total_games += rewards.size + shortlist_rewards.size

            counts = Counter(int(value) for value in families)
            state_rows.append(
                {
                    "state_id": state_id,
                    "prefix_seed": prefix_seed,
                    "seat": seat,
                    "decision_step": int(feasible["decision_step"]),
                    "future_seeds": future_seeds,
                    "feasible_count": int(rewards.shape[1]),
                    "shortlist_count": int(shortlist_rewards.shape[1]),
                    "keep_expected_reward": keep_reward,
                    "oracle_expected_reward": float(expected[oracle]),
                    "oracle_reward_std": float(reward_std[oracle]),
                    "oracle_family": FAMILY_NAMES[int(families[oracle])],
                    "oracle_signature": oracle_signature,
                    "oracle_in_shortlist": oracle_in_shortlist,
                    "shortlist_best_expected_reward": float(
                        shortlist_expected[shortlist_best]
                    ),
                    "shortlist_regret": regret,
                    "shortlist_within_500": near_500,
                    "shortlist_within_1000": near_1000,
                    "best_gain_over_keep": best_gain,
                    "overflow_events": overflow,
                    "feasible_by_family": {
                        FAMILY_NAMES[family]: count
                        for family, count in sorted(counts.items())
                    },
                }
            )
            for arm in range(rewards.shape[1]):
                family = int(families[arm])
                gain = float(expected[arm] - keep_reward)
                family_gains[family].append(gain)
                dataset_state.append(state_id)
                dataset_prefix_seed.append(prefix_seed)
                dataset_seat.append(seat)
                dataset_signature.append(int(signatures[arm]))
                dataset_family.append(family)
                dataset_features.append(
                    np.concatenate([arm_features[arm], context]).astype(np.int32)
                )
                dataset_expected_reward.append(float(expected[arm]))
                dataset_reward_std.append(float(reward_std[arm]))
                dataset_gain.append(gain)
                dataset_samples.append(rewards[:, arm].copy())
                dataset_in_shortlist.append(int(signatures[arm]) in shortlist_signatures)
            state_id += 1

    family_summary = {}
    for family, gains in sorted(family_gains.items()):
        family_summary[FAMILY_NAMES[family]] = {
            "arms": len(gains),
            "mean_expected_gain_vs_keep": statistics.fmean(gains),
            "median_expected_gain_vs_keep": statistics.median(gains),
            "positive_rate": sum(gain > 0 for gain in gains) / len(gains),
            "best_gain": max(gains),
            "worst_gain": min(gains),
        }

    regrets = [row["shortlist_regret"] for row in state_rows]
    best_gains = [row["best_gain_over_keep"] for row in state_rows]
    payload = {
        "schema": "kaggriculture.candidate8_multifuture_oracle.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "opponent": args.opponent,
            "opponent_route": opponent,
            "prefix_seed_start": args.prefix_seed_start,
            "prefix_seed_count": args.prefix_seed_count,
            "future_seed_start": args.future_seed_start,
            "future_count": args.future_count,
            "seats": list(seats),
            "minimum_decision_day": args.minimum_decision_day,
            "maximum_arms": args.maximum_arms,
            "threads": 16,
        },
        "summary": {
            "states": len(state_rows),
            "feasible_count_range": [
                min(row["feasible_count"] for row in state_rows),
                max(row["feasible_count"] for row in state_rows),
            ],
            "oracle_recall_at_64": sum(
                row["oracle_in_shortlist"] for row in state_rows
            ) / len(state_rows),
            "near_oracle_recall_within_500": sum(
                row["shortlist_within_500"] for row in state_rows
            ) / len(state_rows),
            "near_oracle_recall_within_1000": sum(
                row["shortlist_within_1000"] for row in state_rows
            ) / len(state_rows),
            "mean_shortlist_regret": statistics.fmean(regrets),
            "max_shortlist_regret": max(regrets),
            "mean_best_gain_over_keep": statistics.fmean(best_gains),
            "positive_best_gain_rate": sum(gain > 0 for gain in best_gains)
            / len(best_gains),
            "end_overflow_events": total_overflow,
            "total_complete_continuations": total_games,
            "setup_seconds": setup_seconds,
            "simulation_seconds": simulation_seconds,
            "complete_continuations_per_second": total_games
            / simulation_seconds,
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
    if args.backbone is not None:
        payload["inputs"]["backbone"] = {
            "path": str(args.backbone),
            "sha256": sha256(args.backbone),
        }
    if args.dataset_output is not None:
        names = candidate_feature_names() + context_feature_names()
        if len(names) != len(dataset_features[0]):
            raise RuntimeError(
                f"feature schema mismatch: names={len(names)} "
                f"values={len(dataset_features[0])}"
            )
        args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dataset_output,
            state_id=np.asarray(dataset_state, dtype=np.int32),
            prefix_seed=np.asarray(dataset_prefix_seed, dtype=np.int64),
            seat=np.asarray(dataset_seat, dtype=np.int8),
            signature=np.asarray(dataset_signature, dtype=np.uint64),
            family=np.asarray(dataset_family, dtype=np.int8),
            features=np.asarray(dataset_features, dtype=np.int32),
            expected_reward=np.asarray(dataset_expected_reward, dtype=np.float64),
            reward_std=np.asarray(dataset_reward_std, dtype=np.float64),
            gain_vs_keep=np.asarray(dataset_gain, dtype=np.float64),
            future_rewards=np.asarray(dataset_samples, dtype=np.float64),
            in_shortlist=np.asarray(dataset_in_shortlist, dtype=np.bool_),
            feature_names=np.asarray(names),
        )
        payload["dataset"] = {
            "path": str(args.dataset_output),
            "rows": len(dataset_state),
            "feature_count": len(names),
            "sha256": sha256(args.dataset_output),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
