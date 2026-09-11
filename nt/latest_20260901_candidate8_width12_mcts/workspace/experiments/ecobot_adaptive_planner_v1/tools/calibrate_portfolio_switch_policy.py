#!/usr/bin/env python3
"""Pair an analytic SWITCH policy with its zero-switch rollback on common seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fast_kaggriculture import adaptive_default_genome, adaptive_genome_names
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


SWITCH_FEATURE_NAMES = [
    "day", "hour", "cash", "hands", "quadrants", "shed_units",
    "carried_units", "weeds", "ready_harvest", "hard_water", "hard_feed",
    "source_project", "destination_project", "remove_count", "add_count",
    "baseline_analytic_score", "candidate_analytic_score",
    "baseline_estimated_hands", "candidate_estimated_hands",
    "baseline_quadrants", "candidate_quadrants",
    "owned_geese", "owned_cows", "owned_sheep",
    "field_geese", "field_cows", "field_sheep",
    "field_wheat", "field_carrot", "field_tomato", "field_strawberry",
    "field_melon", "committed_wheat", "committed_carrot",
    "committed_tomato", "committed_strawberry", "committed_melon",
    "baseline_wheat", "baseline_carrot", "baseline_tomato",
    "baseline_strawberry", "baseline_melon", "baseline_geese",
    "baseline_cows", "baseline_sheep", "candidate_wheat",
    "candidate_carrot", "candidate_tomato", "candidate_strawberry",
    "candidate_melon", "candidate_geese", "candidate_cows",
    "candidate_sheep",
    *[f"market_price_{item}" for item in range(9)],
    *[f"market_inventory_{item}" for item in range(9)],
    "shop_mask",
    "delta_wheat", "delta_carrot", "delta_tomato", "delta_strawberry",
    "delta_melon", "delta_geese", "delta_cows", "delta_sheep",
    "seed_wheat", "seed_carrot", "seed_tomato", "seed_strawberry",
    "seed_melon", "fertilizer_units", "wheat_units", "empty_tiles",
    "empty_pastures", "empty_coops", "worker_shed_distance",
    "crop_shed_distance", "animal_shed_distance", "crop_yield_held",
    "animal_yield_held", "fertilizer_ready", "unwatered_tiles",
    "unfed_animals", "baseline_unmet_crops", "candidate_unmet_crops",
    "baseline_unmet_animals", "candidate_unmet_animals",
    "baseline_total_targets", "candidate_total_targets",
    "baseline_workload_x100", "candidate_workload_x100", "liquid_capital",
    "cash_above_reserve", "shed_market_value", "carried_market_value",
    "previous_plan_day", "replan_count",
    "opponent_field_wheat", "opponent_field_carrot",
    "opponent_field_tomato", "opponent_field_strawberry",
    "opponent_field_melon", "opponent_field_geese",
    "opponent_field_cows", "opponent_field_sheep",
    "opponent_wheat_yield", "opponent_carrot_yield",
    "opponent_tomato_yield", "opponent_strawberry_yield",
    "opponent_melon_yield", "opponent_egg_yield",
    "opponent_milk_yield", "opponent_wool_yield",
    "opponent_hands", "opponent_quadrants", "opponent_weeds",
    "opponent_ready_harvest", "opponent_hard_water", "opponent_hard_feed",
    *[f"opponent_visible_supply_x100_{item}" for item in range(9)],
    "opponent_cash", "candidate_seat",
    *[f"self_shed_{item}" for item in range(9)],
    *[f"self_carried_{item}" for item in range(9)],
    *[f"self_project_yield_{item}" for item in range(8)],
    *[f"self_project_age_sum_{item}" for item in range(8)],
    *[f"self_project_stress_{item}" for item in range(8)],
    *[f"self_project_shed_distance_{item}" for item in range(8)],
    *[
        f"{owner}_committed_supply_lag{lag}_x100_{item}"
        for lag in (2, 4, 8)
        for owner in ("self", "opponent")
        for item in range(9)
    ],
    *[f"opponent_project_count_trend_x100_{item}" for item in range(8)],
    *[f"opponent_project_yield_trend_x100_{item}" for item in range(8)],
    "opponent_cash_trend", "opponent_hands_trend_x100",
    "opponent_quadrants_trend_x100", "opponent_history_samples",
    *[f"market_daily_drift_x100_{item}" for item in range(9)],
    *[
        f"{plan}_{component}"
        for plan in ("baseline", "candidate")
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
        )
    ],
    "continuation_max_quadrants",
    "continuation_stop_new_crops_day",
    "continuation_stop_new_animals_day",
    "continuation_local_edit_hold_days",
    "continuation_land_capacity_trigger_x100",
    "continuation_proactive_land_investment_x100",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_genome(path: Path, index: int, names: list[str]) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    defaults = dict(zip(names, adaptive_default_genome(), strict=True))
    defaults.update({
        str(name): float(value)
        for name, value in payload.get("base_values", {}).items()
    })
    row = payload["genomes"][index]
    values = dict(defaults)
    values.update({str(name): float(value) for name, value in row["values"].items()})
    return np.asarray([values[name] for name in names], dtype=np.float64)


def quantiles(values: np.ndarray) -> dict[str, float]:
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
    parser.add_argument("--switch-margin", type=float, required=True)
    parser.add_argument("--switch-cooldown-days", type=int, default=0)
    parser.add_argument("--opponent", default="PASSIVE")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path)
    args = parser.parse_args()

    names = list(adaptive_genome_names())
    base = load_genome(args.genomes, args.genome_index, names)
    challenger = base.copy()
    switch_column = names.index("portfolio_bundle_switch_margin")
    cooldown_column = names.index("portfolio_switch_cooldown_days")
    base[switch_column] = 0.0
    challenger[switch_column] = args.switch_margin
    challenger[cooldown_column] = args.switch_cooldown_days
    genomes = np.stack([base, challenger])

    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    opponent = -1 if args.opponent.upper() == "PASSIVE" else bundle.index(args.opponent)
    tasks = np.asarray([
        [genome, opponent, seed, seat]
        for genome in (0, 1)
        for seed in range(args.seed_start, args.seed_start + args.seed_count)
        for seat in (0, 1)
    ], dtype=np.int64)
    started = time.perf_counter()
    raw_rewards, raw_diagnostics = bundle.adaptive_executor.play_batch(genomes, tasks)
    elapsed = time.perf_counter() - started
    rewards = np.asarray(raw_rewards, dtype=np.float64)
    diagnostics = np.asarray(raw_diagnostics, dtype=np.int32)
    games_per_arm = args.seed_count * 2
    if diagnostics.shape[1] < 22:
        raise RuntimeError("C++ diagnostics do not expose portfolio switch telemetry")

    base_tasks = tasks[:games_per_arm]
    challenger_tasks = tasks[games_per_arm:]
    if not np.array_equal(base_tasks[:, 2:], challenger_tasks[:, 2:]):
        raise AssertionError("common-random-number pairing was not preserved")
    base_own = np.asarray([
        rewards[index, task[3]]
        for index, task in enumerate(base_tasks)
    ], dtype=np.float64)
    challenger_own = np.asarray([
        rewards[games_per_arm + index, task[3]]
        for index, task in enumerate(challenger_tasks)
    ], dtype=np.float64)
    delta = challenger_own - base_own
    base_diag = diagnostics[:games_per_arm]
    challenger_diag = diagnostics[games_per_arm:]
    switches = challenger_diag[:, 19]
    predicted_gain = challenger_diag[:, 20]
    first_day = challenger_diag[:, 21]
    switch_features = challenger_diag[:, 22:22 + len(SWITCH_FEATURE_NAMES)]
    if switch_features.shape[1] not in (0, len(SWITCH_FEATURE_NAMES)):
        raise RuntimeError("C++ switch feature telemetry has an unexpected width")
    switched = switches > 0
    oracle_positive = delta > 0
    oracle_negative = delta < 0
    oracle_tie = delta == 0
    changed_profile = np.any(
        base_diag[:, 6:14] != challenger_diag[:, 6:14], axis=1
    )

    switched_count = int(switched.sum())
    actual_positive_given_switch = int(np.count_nonzero(switched & oracle_positive))
    actual_negative_given_switch = int(np.count_nonzero(switched & oracle_negative))
    actual_tie_given_switch = int(np.count_nonzero(switched & oracle_tie))
    pairwise_accuracy = (
        actual_positive_given_switch / switched_count if switched_count else 0.0
    )
    oracle_positive_count = int(oracle_positive.sum())
    topk_recall_proxy = (
        int(np.count_nonzero(switched & oracle_positive)) / oracle_positive_count
        if oracle_positive_count else 0.0
    )

    def group_summary(values: np.ndarray) -> dict[str, float | int]:
        return {
            "games": int(values.size),
            "positive": int(np.count_nonzero(values > 0)),
            "negative": int(np.count_nonzero(values < 0)),
            "accuracy": float(np.mean(values > 0)) if values.size else 0.0,
            "mean_delta": float(values.mean()) if values.size else 0.0,
            "p10_delta": float(np.quantile(values, 0.10)) if values.size else 0.0,
        }

    by_first_day = {
        str(int(group)): group_summary(delta[switched & (first_day == group)])
        for group in np.unique(first_day[switched])
    } if switched_count else {}
    by_source_destination: dict[str, dict[str, float | int]] = {}
    if switched_count and switch_features.shape[1]:
        source = switch_features[:, 11]
        destination = switch_features[:, 12]
        for src, dst in sorted(set(zip(source[switched], destination[switched]))):
            mask = switched & (source == src) & (destination == dst)
            by_source_destination[f"{int(src)}->{int(dst)}"] = group_summary(delta[mask])

    profile_counter: Counter[tuple[int, ...]] = Counter()
    for index in np.flatnonzero(switched):
        animal_delta = challenger_diag[index, 6:9] - base_diag[index, 6:9]
        crop_delta = challenger_diag[index, 9:14] - base_diag[index, 9:14]
        profile_counter[tuple(int(value) for value in np.r_[animal_delta, crop_delta])] += 1

    bad_indices = np.flatnonzero(switched & oracle_negative)
    good_indices = np.flatnonzero(switched & oracle_positive)
    bad_indices = bad_indices[np.argsort(delta[bad_indices])][:64]
    good_indices = good_indices[np.argsort(-delta[good_indices])][:32]

    def example(index: int) -> dict[str, object]:
        return {
            "seed": int(base_tasks[index, 2]),
            "seat": int(base_tasks[index, 3]),
            "base_reward": float(base_own[index]),
            "challenger_reward": float(challenger_own[index]),
            "actual_delta": float(delta[index]),
            "switches": int(switches[index]),
            "predicted_gain": int(predicted_gain[index]),
            "first_switch_day": int(first_day[index]),
            "animal_target_delta": [
                int(value) for value in challenger_diag[index, 6:9] - base_diag[index, 6:9]
            ],
            "crop_target_delta": [
                int(value) for value in challenger_diag[index, 9:14] - base_diag[index, 9:14]
            ],
        }

    payload = {
        "schema": "kaggriculture.portfolio-switch-calibration.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "policy_semantics": "public-state SWITCH-only portfolio edits; no replay or opponent identity",
        "opponent": args.opponent,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "seat_swap": True,
        "games_per_arm": games_per_arm,
        "simulation_seconds": elapsed,
        "games_per_second": len(tasks) / elapsed,
        "switch_margin": args.switch_margin,
        "switch_cooldown_days": args.switch_cooldown_days,
        "baseline": quantiles(base_own),
        "challenger": quantiles(challenger_own),
        "paired_delta": quantiles(delta),
        "telemetry": {
            "switched_games": switched_count,
            "changed_profile_games": int(changed_profile.sum()),
            "positive_given_switch": actual_positive_given_switch,
            "negative_given_switch": actual_negative_given_switch,
            "tie_given_switch": actual_tie_given_switch,
            "pairwise_accuracy": pairwise_accuracy,
            "oracle_positive_games": oracle_positive_count,
            "topk_recall_proxy": topk_recall_proxy,
            "mean_switches_given_switch": (
                float(switches[switched].mean()) if switched_count else 0.0
            ),
            "mean_predicted_gain_given_switch": (
                float(predicted_gain[switched].mean()) if switched_count else 0.0
            ),
            "first_switch_day_distribution": {
                str(int(day)): int(np.count_nonzero(first_day[switched] == day))
                for day in np.unique(first_day[switched])
            } if switched_count else {},
            "by_first_switch_day": by_first_day,
            "by_source_destination": by_source_destination,
        },
        "switch_feature_names": SWITCH_FEATURE_NAMES,
        "top_target_delta_profiles": [
            {"delta": list(profile), "games": count}
            for profile, count in profile_counter.most_common(24)
        ],
        "worst_false_positive_examples": [example(int(index)) for index in bad_indices],
        "best_true_positive_examples": [example(int(index)) for index in good_indices],
        "hard_failures": {
            "baseline": int(base_diag[:, 2:5].sum()),
            "challenger": int(challenger_diag[:, 2:5].sum()),
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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.dataset_output is not None:
        args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dataset_output,
            features=switch_features[switched],
            actual_delta=delta[switched],
            predicted_gain=predicted_gain[switched],
            switches=switches[switched],
            seed=base_tasks[switched, 2],
            seat=base_tasks[switched, 3],
            feature_names=np.asarray(SWITCH_FEATURE_NAMES),
        )
    print(json.dumps({
        "games": len(tasks),
        "games_per_second": payload["games_per_second"],
        "baseline_mean": payload["baseline"]["mean"],
        "challenger_mean": payload["challenger"]["mean"],
        "paired_delta_mean": payload["paired_delta"]["mean"],
        "switched_games": switched_count,
        "pairwise_accuracy": pairwise_accuracy,
        "topk_recall_proxy": topk_recall_proxy,
        "hard_failures": payload["hard_failures"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
