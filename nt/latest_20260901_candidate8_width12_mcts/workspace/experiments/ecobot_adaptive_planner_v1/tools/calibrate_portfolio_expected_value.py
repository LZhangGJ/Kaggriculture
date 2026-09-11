#!/usr/bin/env python3
"""Fork one public state and average KEEP/SWITCH value over future RNG draws."""

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


def keep_features(candidate: np.ndarray) -> np.ndarray:
    at = {name: index for index, name in enumerate(SWITCH_FEATURE_NAMES)}
    keep = candidate.copy()
    keep[at["source_project"]] = -1
    keep[at["destination_project"]] = -1
    keep[at["remove_count"]] = 0
    keep[at["add_count"]] = 0
    for suffix in (
        "analytic_score", "estimated_hands", "quadrants",
        "unmet_crops", "unmet_animals", "total_targets",
        "workload_x100",
    ):
        keep[at[f"candidate_{suffix}"]] = keep[at[f"baseline_{suffix}"]]
    for project in (
        "wheat", "carrot", "tomato", "strawberry", "melon",
        "geese", "cows", "sheep",
    ):
        keep[at[f"candidate_{project}"]] = keep[at[f"baseline_{project}"]]
        keep[at[f"delta_{project}"]] = 0
    for component in (
        "crop_gross", "animal_gross", "fertilizer_gross",
        "seed_cost", "feed_cost", "animal_purchase_cost",
        "action_cost", "move_cost", "hire_cost", "land_cost",
        "lockup_cost", "crop_units", "animal_product_units",
        "fertilizer_used", "fertilizer_sellable", "setup_turns",
        "first_cash_lag", "immediate_commitment_actions_x100",
        "today_action_capacity_x100", "today_deadline_slack_x100",
        "peak_daily_utilization_x100",
        "window_action_demand_lag2_x100",
        "window_action_demand_lag4_x100",
        "window_action_demand_lag8_x100",
        "window_action_capacity_lag2_x100",
        "window_action_capacity_lag4_x100",
        "window_action_capacity_lag8_x100",
        "window_net_cash_lag2", "window_net_cash_lag4",
        "window_net_cash_lag8", "minimum_window_slack_x100",
        "current_task_count", "current_hard_task_count",
        "assigned_task_count", "assigned_hard_task_count",
        "total_assignment_distance", "total_assignment_steps",
        "minimum_assignment_slack",
        "deadline_infeasible_task_count",
        "unassigned_hard_task_count",
        "assigned_expected_cash_gain", "unassigned_delayed_loss",
        "assigned_value_per_step_x100",
        "commissioning_job_count", "commissioning_total_steps",
        "commissioning_makespan_steps",
        "commissioning_first_cash_slack_steps",
        "commissioning_unreachable_count", "new_service_distance",
        "new_service_route_span",
        "commissioning_steps_per_added_unit_x100",
    ):
        keep[at[f"candidate_{component}"]] = keep[
            at[f"baseline_{component}"]
        ]
    return keep


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
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override one genome field as NAME=VALUE; may be repeated.",
    )
    parser.add_argument("--opponent", default="PASSIVE")
    parser.add_argument("--prefix-seed-start", type=int, required=True)
    parser.add_argument("--prefix-seed-count", type=int, required=True)
    parser.add_argument("--future-seed-start", type=int, required=True)
    parser.add_argument("--future-samples", type=int, default=32)
    parser.add_argument("--candidate-ranks", type=int, default=4)
    parser.add_argument("--switch-margin", type=float, default=1e-6)
    parser.add_argument("--minimum-decision-day", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, required=True)
    args = parser.parse_args()

    names = list(adaptive_genome_names())
    genome = load_genome(args.genomes, args.genome_index, names)
    for raw_override in args.override:
        name, raw_value = raw_override.split("=", 1)
        if name not in names:
            raise ValueError(f"unknown genome field: {name}")
        genome[names.index(name)] = float(raw_value)
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    opponent = -1 if args.opponent.upper() == "PASSIVE" else bundle.index(args.opponent)

    row_features = []
    row_delta = []
    row_std = []
    row_prefix_seed = []
    row_seat = []
    row_rank = []
    row_samples = []
    state_summaries = []
    pairwise_correct = 0
    pairwise_total = 0
    top1 = 0
    top3 = 0
    states = 0
    missing_decision = 0
    total_continuations = 0
    overflow_total = 0
    started = time.perf_counter()

    for prefix_offset in range(args.prefix_seed_count):
        prefix_seed = args.prefix_seed_start + prefix_offset
        for seat in (0, 1):
            future_base = args.future_seed_start + (
                prefix_offset * 2 + seat
            ) * args.future_samples
            future_seeds = list(range(future_base, future_base + args.future_samples))
            result = bundle.adaptive_executor.portfolio_counterfactual(
                genome,
                opponent,
                prefix_seed,
                future_seeds,
                seat,
                args.candidate_ranks,
                args.switch_margin,
                args.minimum_decision_day,
            )
            if not result["decision_found"]:
                missing_decision += 1
                continue
            rewards = np.asarray(result["rewards"], dtype=np.float64)
            overflow = np.asarray(result["end_overflow"], dtype=np.int32)
            available = np.asarray(result["arm_available"], dtype=bool)
            features = np.asarray(result["arm_features"], dtype=np.int32)
            valid_arms = np.flatnonzero(available)
            if valid_arms.size <= 1:
                missing_decision += 1
                continue
            total_continuations += int(rewards[:, valid_arms].size)
            overflow_total += int(overflow[:, valid_arms].sum())
            keep_mean = float(rewards[:, 0].mean())
            group_delta = []
            group_analytic = []
            group_rank = []
            for arm in valid_arms:
                if arm == 0:
                    feature = keep_features(features[valid_arms[1]])
                    rank = -1
                else:
                    feature = features[arm]
                    rank = int(arm) - 1
                samples = rewards[:, arm] - rewards[:, 0]
                mean_delta = float(samples.mean())
                group_delta.append(mean_delta)
                group_analytic.append(float(feature[16]))
                group_rank.append(rank)
                row_features.append(feature)
                row_delta.append(mean_delta)
                row_std.append(float(samples.std(ddof=1)))
                row_prefix_seed.append(prefix_seed)
                row_seat.append(seat)
                row_rank.append(rank)
                row_samples.append(samples)
            actual = np.asarray(group_delta)
            analytic = np.asarray(group_analytic)
            for left in range(len(actual)):
                for right in range(left + 1, len(actual)):
                    actual_sign = np.sign(actual[left] - actual[right])
                    if actual_sign == 0:
                        continue
                    pairwise_total += 1
                    pairwise_correct += actual_sign == np.sign(
                        analytic[left] - analytic[right]
                    )
            actual_best = int(np.argmax(actual))
            analytic_order = np.argsort(-analytic, kind="stable")
            top1 += actual_best == int(analytic_order[0])
            top3 += actual_best in analytic_order[:3]
            states += 1
            state_summaries.append({
                "prefix_seed": prefix_seed,
                "seat": seat,
                "decision_step": int(result["decision_step"]),
                "available_ranks": [int(value) for value in group_rank],
                "keep_mean_reward": keep_mean,
                "expected_delta": group_delta,
                "oracle_rank": int(group_rank[actual_best]),
                "analytic_rank": int(group_rank[int(analytic_order[0])]),
            })

    elapsed = time.perf_counter() - started
    feature_matrix = np.asarray(row_features, dtype=np.int32)
    delta_vector = np.asarray(row_delta, dtype=np.float64)
    std_vector = np.asarray(row_std, dtype=np.float64)
    prefix_vector = np.asarray(row_prefix_seed, dtype=np.int64)
    seat_vector = np.asarray(row_seat, dtype=np.int8)
    rank_vector = np.asarray(row_rank, dtype=np.int16)
    samples_matrix = np.asarray(row_samples, dtype=np.float64)
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.dataset_output,
        features=feature_matrix,
        expected_delta=delta_vector,
        future_std=std_vector,
        prefix_seed=prefix_vector,
        seat=seat_vector,
        candidate_rank=rank_vector,
        future_delta_samples=samples_matrix,
        feature_names=np.asarray(SWITCH_FEATURE_NAMES),
    )

    payload = {
        "schema": "kaggriculture.portfolio-expected-value-calibration.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "opponent": args.opponent,
        "prefix_seed_start": args.prefix_seed_start,
        "prefix_seed_count": args.prefix_seed_count,
        "future_seed_start": args.future_seed_start,
        "future_samples_per_state": args.future_samples,
        "candidate_ranks": args.candidate_ranks,
        "switch_margin": args.switch_margin,
        "minimum_decision_day": args.minimum_decision_day,
        "overrides": list(args.override),
        "states": states,
        "missing_decision_states": missing_decision,
        "continuations": total_continuations,
        "elapsed_seconds": elapsed,
        "continuations_per_second": total_continuations / elapsed,
        "analytic_pairwise_accuracy": (
            pairwise_correct / pairwise_total if pairwise_total else 0.0
        ),
        "pairwise_comparisons": pairwise_total,
        "analytic_top1_oracle_recall": top1 / states if states else 0.0,
        "analytic_top3_oracle_recall": top3 / states if states else 0.0,
        "expected_candidate_delta": quantiles(
            delta_vector[rank_vector >= 0]
        ),
        "future_std": quantiles(std_vector[rank_vector >= 0]),
        "end_overflow": overflow_total,
        "state_summaries": state_summaries,
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
        "states": states,
        "missing_decision_states": missing_decision,
        "continuations": total_continuations,
        "continuations_per_second": payload["continuations_per_second"],
        "pairwise_accuracy": payload["analytic_pairwise_accuracy"],
        "top1_oracle_recall": payload["analytic_top1_oracle_recall"],
        "top3_oracle_recall": payload["analytic_top3_oracle_recall"],
        "end_overflow": overflow_total,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
