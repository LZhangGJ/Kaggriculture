#!/usr/bin/env python3
"""Generate multi-opponent Candidate8 labels with public-state features.

The opponent route is used only to run the offline counterfactual.  It is kept
as a group key for leave-route-out validation and is never included in model
features.  Labels follow the competition objective: score rate, then margin,
then own cash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_candidate8_multifuture_oracle import (
    FAMILY_NAMES,
    candidate_feature_names,
    context_feature_names,
    load_genome,
)
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


PREVIEW_UNIT_OPS = (
    "pass", "north", "south", "east", "west", "drop", "pickup",
    "place", "plant", "water", "harvest", "fertilize", "dig",
    "build_coop", "build_pasture", "feed", "collect_fertilizer", "care",
)
PREVIEW_MARKET_OPS = (
    "hire", "buy_land", "buy_seed", "buy_product", "buy_animal", "sell",
)
RESPONSE_SCENARIOS = (
    "pass", "balanced_autonomous", "expand", "sell_now", "hold_demand",
)
RESPONSE_OUTCOME_NAMES = tuple(
    f"{horizon}_{metric}"
    for horizon in (24, 48, "terminal")
    for metric in ("own_money_x100", "opponent_money_x100", "margin_x100")
)


def consequence_feature_names() -> list[str]:
    state = [
        "money_x100", "minimum_money_x100", "hands", "quadrants",
        "shed_quantity", "shed_value_x100", "carry_quantity",
        "carry_value_x100", "seed_quantity", "crop_total", "animal_total",
        "weeds", "ready_yield", "hard_water", "hard_feed",
    ]
    state += [f"project_count_{index}" for index in range(8)]
    state += [f"project_yield_{index}" for index in range(8)]
    state += [f"market_inventory_{index}" for index in range(9)]
    execution = [f"unit_op_count_{name}" for name in PREVIEW_UNIT_OPS]
    execution += [f"market_op_count_{name}" for name in PREVIEW_MARKET_OPS]
    execution += [f"market_quantity_{name}" for name in PREVIEW_MARKET_OPS]
    execution += [
        "cash_shortfall_count", "cash_shortfall_total_x100", "overflow_total",
        "sell_before_buy_steps", "buy_before_sell_steps", "mixed_market_steps",
    ]
    names = [
        f"preview{horizon}_{name}"
        for horizon in (24, 48, "terminal")
        for name in state + execution
    ]
    if len(names) != 228:
        raise AssertionError(f"unexpected consequence schema width {len(names)}")
    return names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_days(raw: str) -> list[int]:
    days = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not days or days[0] < 0 or days[-1] > 29:
        raise argparse.ArgumentTypeError("days must be a non-empty subset of 0..29")
    return days


def parse_day0_ranks(raw: str) -> tuple[int, int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 2 or any(value < 0 or value >= 4096 for value in values):
        raise argparse.ArgumentTypeError(
            "committed Day0 ranks must be two values in [0, 4095]"
        )
    return values[0], values[1]


def select_opponents(merged: dict, group: str) -> list[str]:
    if group == "hard16":
        return list(merged["route_bands"]["oracle_below_50pct"])
    if group == "selector43":
        return [
            row["opponent"]
            for row in merged["opponents"]
            if row["oracle"]["win_rate"] >= 0.90
            and row["actual"]["win_rate"] < 0.90
        ]
    return [row["opponent"] for row in merged["opponents"]]


def best_index(score: np.ndarray, margin: np.ndarray, cash: np.ndarray) -> int:
    return max(
        range(len(score)),
        key=lambda index: (
            score[index], margin[index], cash[index], -index
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--merged-receipt", required=True, type=Path)
    parser.add_argument(
        "--group", choices=("selector43", "hard16", "all"), default="selector43"
    )
    parser.add_argument(
        "--opponents",
        type=str,
        help="Optional comma-separated route names; overrides --group.",
    )
    parser.add_argument("--days", type=parse_days, default=parse_days("3,9,15,21"))
    parser.add_argument("--prefix-seed-start", required=True, type=int)
    parser.add_argument("--prefix-seed-count", type=int, default=1)
    parser.add_argument("--future-seed-start", required=True, type=int)
    parser.add_argument("--future-count", type=int, default=4)
    parser.add_argument("--maximum-arms", type=int, default=256)
    parser.add_argument(
        "--candidate-pool",
        choices=("feasible", "shortlist"),
        default="feasible",
        help="Use the generation-order feasible pool or diverse shortlist.",
    )
    parser.add_argument(
        "--committed-day0-ranks",
        type=parse_day0_ranks,
        help=(
            "Optional shortlist ranks for seats 0 and 1.  When generating "
            "Day1+ states, the selected Day0 edit is committed before the "
            "new public decision state is captured."
        ),
    )
    parser.add_argument("--single-seat", type=int, choices=(0, 1))
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print one flushed progress record after this many decision states.",
    )
    parser.add_argument(
        "--response-scenarios",
        action="store_true",
        help=(
            "Also run the five fixed O1.5 public-belief rival response "
            "scenarios. This changes features only, never labels."
        ),
    )
    parser.add_argument("--dataset-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")

    merged = json.loads(args.merged_receipt.read_text(encoding="utf-8"))
    opponent_names = (
        [value.strip() for value in args.opponents.split(",") if value.strip()]
        if args.opponents
        else select_opponents(merged, args.group)
    )
    if not opponent_names:
        raise ValueError("at least one opponent is required")
    seats = (args.single_seat,) if args.single_seat is not None else (0, 1)
    use_feasible_pool = args.candidate_pool == "feasible"

    setup_started = time.perf_counter()
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)
    setup_seconds = time.perf_counter() - setup_started

    state_ids: list[int] = []
    opponent_rows: list[str] = []
    prefix_seeds: list[int] = []
    seats_rows: list[int] = []
    decision_days: list[int] = []
    signatures: list[int] = []
    families: list[int] = []
    features: list[np.ndarray] = []
    consequence_features: list[np.ndarray] = []
    response_scenario_features: list[np.ndarray] = []
    response_scenario_outcomes: list[np.ndarray] = []
    expected_scores: list[float] = []
    expected_margins: list[float] = []
    expected_own_cash: list[float] = []
    expected_opponent_cash: list[float] = []
    reward_stds: list[float] = []
    future_own_rows: list[np.ndarray] = []
    future_opponent_rows: list[np.ndarray] = []
    first_action_change_rows: list[int] = []
    action_change_24_rows: list[int] = []
    action_change_full_rows: list[int] = []
    first_state_change_rows: list[int] = []
    state_change_24_rows: list[int] = []
    state_change_full_rows: list[int] = []
    state_summaries: list[dict] = []
    per_opponent = defaultdict(lambda: {
        "states": 0,
        "arms": 0,
        "oracle_score_gain": [],
        "oracle_margin_gain": [],
        "oracle_cash_gain": [],
        "oracle_family": Counter(),
    })

    simulation_seconds = 0.0
    total_continuations = 0
    total_overflow = 0
    missing_states = 0
    state_id = 0
    processed_cases = 0
    requested_cases = (
        len(opponent_names) * args.prefix_seed_count * len(seats) * len(args.days)
    )
    run_started = time.perf_counter()
    for opponent_name in opponent_names:
        opponent = bundle.index(opponent_name)
        for prefix_offset in range(args.prefix_seed_count):
            prefix_seed = args.prefix_seed_start + prefix_offset
            for seat_index, seat in enumerate(seats):
                for day_index, day in enumerate(args.days):
                    # Every opponent sees the same future bank for the same
                    # public prefix/seat/day.  In particular, Day0 labels can
                    # then be averaged across plausible opponents without
                    # opponent-specific future RNG leaking into the target.
                    public_case = (
                        (prefix_offset * len(seats) + seat_index) * len(args.days)
                        + day_index
                    )
                    future_base = (
                        args.future_seed_start
                        + public_case * args.future_count
                    )
                    future_seeds = [
                        future_base + offset for offset in range(args.future_count)
                    ]
                    started = time.perf_counter()
                    result = bundle.adaptive_executor.candidate8_counterfactual(
                        genome,
                        opponent,
                        prefix_seed,
                        future_seeds,
                        seat,
                        day,
                        use_feasible_pool,
                        args.maximum_arms,
                        [0] if args.committed_day0_ranks and day > 0 else [],
                        [args.committed_day0_ranks[seat]]
                        if args.committed_day0_ranks and day > 0
                        else [],
                        args.response_scenarios,
                    )
                    simulation_seconds += time.perf_counter() - started
                    processed_cases += 1
                    if not result["decision_found"]:
                        missing_states += 1
                        if args.progress_every and (
                            processed_cases % args.progress_every == 0
                            or processed_cases == requested_cases
                        ):
                            print(json.dumps({
                                "progress": processed_cases,
                                "total": requested_cases,
                                "percent": 100.0 * processed_cases / requested_cases,
                                "states": state_id,
                                "rows": len(state_ids),
                                "missing_states": missing_states,
                                "elapsed_seconds": time.perf_counter() - run_started,
                            }), file=sys.stderr, flush=True)
                        continue
                    own = np.asarray(result["rewards"], dtype=np.float64)
                    opponent_cash = np.asarray(
                        result["opponent_rewards"], dtype=np.float64
                    )
                    margin_samples = own - opponent_cash
                    score_samples = np.where(
                        margin_samples > 0,
                        1.0,
                        np.where(margin_samples == 0, 0.5, 0.0),
                    )
                    expected_score = score_samples.mean(axis=0)
                    expected_margin = margin_samples.mean(axis=0)
                    expected_cash = own.mean(axis=0)
                    expected_opp = opponent_cash.mean(axis=0)
                    std = own.std(axis=0)
                    family = np.asarray(result["arm_family"], dtype=np.int8)
                    signature = np.asarray(result["arm_signature"], dtype=np.uint64)
                    arm_features = np.asarray(result["arm_features"], dtype=np.int32)
                    arm_consequences = np.asarray(
                        result["consequence_features"], dtype=np.int32
                    )
                    arm_response_features = np.asarray(
                        result["response_scenario_features"], dtype=np.int32
                    )
                    arm_response_outcomes = np.asarray(
                        result["response_scenario_outcomes"], dtype=np.int32
                    )
                    if args.response_scenarios:
                        expected_shape = (own.shape[1], len(RESPONSE_SCENARIOS), 228)
                        if arm_response_features.shape != expected_shape:
                            raise RuntimeError(
                                "unexpected response scenario feature shape: "
                                f"{arm_response_features.shape} != {expected_shape}"
                            )
                        if arm_response_outcomes.shape != (
                            own.shape[1], len(RESPONSE_SCENARIOS), 9
                        ):
                            raise RuntimeError(
                                "unexpected response scenario outcome shape: "
                                f"{arm_response_outcomes.shape}"
                            )
                    context = np.asarray(result["context_features"], dtype=np.int32)
                    first_action_change = np.asarray(
                        result["first_action_change_offset"], dtype=np.int16
                    )
                    action_change_24 = np.asarray(
                        result["action_change_count_24"], dtype=np.int16
                    )
                    action_change_full = np.asarray(
                        result["action_change_count_full"], dtype=np.int16
                    )
                    first_state_change = np.asarray(
                        result["first_state_change_offset"], dtype=np.int16
                    )
                    state_change_24 = np.asarray(
                        result["state_change_count_24"], dtype=np.int16
                    )
                    state_change_full = np.asarray(
                        result["state_change_count_full"], dtype=np.int16
                    )
                    if len(np.unique(signature)) != len(signature):
                        raise RuntimeError(
                            "Candidate8 pool contains duplicate signatures; "
                            "the requested decision state was not refreshed"
                        )
                    overflow = int(np.asarray(result["end_overflow"]).sum())
                    oracle = best_index(expected_score, expected_margin, expected_cash)
                    keep = 0

                    state_summaries.append({
                        "state_id": state_id,
                        "opponent": opponent_name,
                        "prefix_seed": prefix_seed,
                        "seat": seat,
                        "requested_day": day,
                        "decision_step": int(result["decision_step"]),
                        "arms": int(own.shape[1]),
                        "oracle_family": FAMILY_NAMES[int(family[oracle])],
                        "oracle_signature": int(signature[oracle]),
                        "oracle_score_rate": float(expected_score[oracle]),
                        "keep_score_rate": float(expected_score[keep]),
                        "oracle_margin": float(expected_margin[oracle]),
                        "keep_margin": float(expected_margin[keep]),
                        "oracle_own_cash": float(expected_cash[oracle]),
                        "keep_own_cash": float(expected_cash[keep]),
                        "overflow_events": overflow,
                    })
                    stats = per_opponent[opponent_name]
                    stats["states"] += 1
                    stats["arms"] += int(own.shape[1])
                    stats["oracle_score_gain"].append(
                        float(expected_score[oracle] - expected_score[keep])
                    )
                    stats["oracle_margin_gain"].append(
                        float(expected_margin[oracle] - expected_margin[keep])
                    )
                    stats["oracle_cash_gain"].append(
                        float(expected_cash[oracle] - expected_cash[keep])
                    )
                    stats["oracle_family"][FAMILY_NAMES[int(family[oracle])]] += 1

                    for arm in range(own.shape[1]):
                        state_ids.append(state_id)
                        opponent_rows.append(opponent_name)
                        prefix_seeds.append(prefix_seed)
                        seats_rows.append(seat)
                        decision_days.append(day)
                        signatures.append(int(signature[arm]))
                        families.append(int(family[arm]))
                        features.append(
                            np.concatenate([arm_features[arm], context]).astype(np.int32)
                        )
                        consequence_features.append(arm_consequences[arm].copy())
                        if args.response_scenarios:
                            response_scenario_features.append(
                                arm_response_features[arm].copy()
                            )
                            response_scenario_outcomes.append(
                                arm_response_outcomes[arm].copy()
                            )
                        expected_scores.append(float(expected_score[arm]))
                        expected_margins.append(float(expected_margin[arm]))
                        expected_own_cash.append(float(expected_cash[arm]))
                        expected_opponent_cash.append(float(expected_opp[arm]))
                        reward_stds.append(float(std[arm]))
                        future_own_rows.append(own[:, arm].copy())
                        future_opponent_rows.append(opponent_cash[:, arm].copy())
                        first_action_change_rows.append(
                            int(first_action_change[arm])
                        )
                        action_change_24_rows.append(int(action_change_24[arm]))
                        action_change_full_rows.append(int(action_change_full[arm]))
                        first_state_change_rows.append(int(first_state_change[arm]))
                        state_change_24_rows.append(int(state_change_24[arm]))
                        state_change_full_rows.append(int(state_change_full[arm]))
                    total_continuations += own.size
                    total_overflow += overflow
                    state_id += 1
                    if args.progress_every and (
                        processed_cases % args.progress_every == 0
                        or processed_cases == requested_cases
                    ):
                        elapsed = time.perf_counter() - run_started
                        print(json.dumps({
                            "progress": processed_cases,
                            "total": requested_cases,
                            "percent": 100.0 * processed_cases / requested_cases,
                            "states": state_id,
                            "rows": len(state_ids),
                            "missing_states": missing_states,
                            "elapsed_seconds": elapsed,
                            "estimated_remaining_seconds": (
                                elapsed * (requested_cases - processed_cases)
                                / processed_cases
                            ),
                        }), file=sys.stderr, flush=True)

    names = candidate_feature_names() + context_feature_names()
    if not features or len(names) != len(features[0]):
        raise RuntimeError(
            f"feature schema mismatch: names={len(names)} "
            f"values={len(features[0]) if features else 0}"
        )

    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        state_id=np.asarray(state_ids, dtype=np.int32),
        opponent=np.asarray(opponent_rows),
        prefix_seed=np.asarray(prefix_seeds, dtype=np.int64),
        seat=np.asarray(seats_rows, dtype=np.int8),
        decision_day=np.asarray(decision_days, dtype=np.int8),
        signature=np.asarray(signatures, dtype=np.uint64),
        family=np.asarray(families, dtype=np.int8),
        features=np.asarray(features, dtype=np.int32),
        consequence_features=np.asarray(consequence_features, dtype=np.int32),
        expected_score_rate=np.asarray(expected_scores, dtype=np.float64),
        expected_margin=np.asarray(expected_margins, dtype=np.float64),
        expected_own_cash=np.asarray(expected_own_cash, dtype=np.float64),
        expected_opponent_cash=np.asarray(expected_opponent_cash, dtype=np.float64),
        own_cash_std=np.asarray(reward_stds, dtype=np.float64),
        future_own_cash=np.asarray(future_own_rows, dtype=np.float64),
        future_opponent_cash=np.asarray(future_opponent_rows, dtype=np.float64),
        first_action_change_offset=np.asarray(
            first_action_change_rows, dtype=np.int16
        ),
        action_change_count_24=np.asarray(action_change_24_rows, dtype=np.int16),
        action_change_count_full=np.asarray(
            action_change_full_rows, dtype=np.int16
        ),
        first_state_change_offset=np.asarray(
            first_state_change_rows, dtype=np.int16
        ),
        state_change_count_24=np.asarray(state_change_24_rows, dtype=np.int16),
        state_change_count_full=np.asarray(state_change_full_rows, dtype=np.int16),
        feature_names=np.asarray(names),
        consequence_feature_names=np.asarray(consequence_feature_names()),
    )
    if args.response_scenarios:
        arrays.update(
            response_scenario_features=np.asarray(
                response_scenario_features, dtype=np.int32
            ),
            response_scenario_outcomes=np.asarray(
                response_scenario_outcomes, dtype=np.int32
            ),
            response_scenario_names=np.asarray(RESPONSE_SCENARIOS),
            response_scenario_outcome_names=np.asarray(RESPONSE_OUTCOME_NAMES),
        )
    np.savez_compressed(args.dataset_output, **arrays)

    opponent_summary = {}
    for name, stats in per_opponent.items():
        opponent_summary[name] = {
            "states": stats["states"],
            "arms": stats["arms"],
            "mean_oracle_score_gain": float(np.mean(stats["oracle_score_gain"])),
            "mean_oracle_margin_gain": float(np.mean(stats["oracle_margin_gain"])),
            "mean_oracle_cash_gain": float(np.mean(stats["oracle_cash_gain"])),
            "oracle_family": dict(stats["oracle_family"]),
        }

    payload = {
        "schema": "kaggriculture.candidate8_competitive_pool.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "group": args.group,
            "opponents": opponent_names,
            "days": args.days,
            "prefix_seed_start": args.prefix_seed_start,
            "prefix_seed_count": args.prefix_seed_count,
            "future_seed_start": args.future_seed_start,
            "future_count": args.future_count,
            "seats": list(seats),
            "maximum_arms": args.maximum_arms,
            "candidate_pool": args.candidate_pool,
            "committed_day0_ranks": (
                list(args.committed_day0_ranks)
                if args.committed_day0_ranks
                else None
            ),
            "response_scenarios": args.response_scenarios,
        },
        "summary": {
            "opponents": len(opponent_names),
            "states": state_id,
            "missing_states": missing_states,
            "rows": len(state_ids),
            "feature_count": len(names),
            "consequence_feature_count": len(consequence_feature_names()),
            "response_scenario_count": (
                len(RESPONSE_SCENARIOS) if args.response_scenarios else 0
            ),
            "complete_continuations": total_continuations,
            "end_overflow_events": total_overflow,
            "setup_seconds": setup_seconds,
            "simulation_seconds": simulation_seconds,
            "complete_continuations_per_second": (
                total_continuations / simulation_seconds
                if simulation_seconds else 0.0
            ),
            "oracle_score_improves_rate": float(np.mean([
                row["oracle_score_rate"] > row["keep_score_rate"]
                for row in state_summaries
            ])),
            "candidate_action_effect_rate": float(np.mean(
                np.asarray(action_change_full_rows, dtype=np.int16) > 0
            )),
            "candidate_state_effect_rate": float(np.mean(
                np.asarray(state_change_full_rows, dtype=np.int16) > 0
            )),
            "oracle_margin_improves_rate": float(np.mean([
                row["oracle_margin"] > row["keep_margin"]
                for row in state_summaries
            ])),
        },
        "per_opponent": opponent_summary,
        "states": state_summaries,
        "dataset": {
            "path": str(args.dataset_output),
            "sha256": sha256(args.dataset_output),
        },
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in {
                "source": args.source,
                "actions": args.actions,
                "metadata": args.metadata,
                "genomes": args.genomes,
                "merged_receipt": args.merged_receipt,
            }.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "dataset": str(args.dataset_output),
        **payload["summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
