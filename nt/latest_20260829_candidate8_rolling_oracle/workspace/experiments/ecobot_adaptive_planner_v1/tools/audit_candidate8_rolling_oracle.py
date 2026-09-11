#!/usr/bin/env python3
"""Measure Candidate8's multi-stage receding-horizon cash ceiling."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_candidate8_multifuture_oracle import FAMILY_NAMES, load_genome, sha256
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def distribution(values: list[float]) -> dict[str, float]:
    ordered = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(ordered.mean()),
        "median": float(np.median(ordered)),
        "p10": float(np.percentile(ordered, 10)),
        "p25": float(np.percentile(ordered, 25)),
        "p75": float(np.percentile(ordered, 75)),
        "p90": float(np.percentile(ordered, 90)),
        "minimum": float(ordered.min()),
        "maximum": float(ordered.max()),
        "rate_ge_140k": float((ordered >= 140_000).mean()),
        "rate_ge_150k": float((ordered >= 150_000).mean()),
        "rate_ge_160k": float((ordered >= 160_000).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", type=Path, required=True)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--opponent", default="PASSIVE")
    parser.add_argument("--actual-seed-start", type=int, required=True)
    parser.add_argument("--actual-seed-count", type=int, default=4)
    parser.add_argument("--future-seed-start", type=int, required=True)
    parser.add_argument("--future-count", type=int, default=8)
    parser.add_argument(
        "--decision-days", type=int, nargs="+",
        default=[3, 6, 9, 12, 15, 18, 21, 24, 27],
    )
    parser.add_argument("--maximum-arms", type=int, default=4096)
    parser.add_argument("--single-seat", type=int, choices=(0, 1))
    parser.add_argument("--skip-single-stage", action="store_true")
    parser.add_argument("--single-stage-day", type=int, default=9)
    parser.add_argument(
        "--clairvoyant-actual-future", action="store_true",
        help="Rank each stage on the exact copied RNG state of the actual game.",
    )
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

    rows: list[dict] = []
    baseline_cash: list[float] = []
    single_cash: list[float] = []
    rolling_cash: list[float] = []
    rolling_margin: list[float] = []
    family_counts: Counter[str] = Counter()
    family_stage_gain: dict[str, list[float]] = defaultdict(list)
    total_continuations = 0
    total_overflow = 0
    started = time.perf_counter()

    state_id = 0
    for offset in range(args.actual_seed_count):
        actual_seed = args.actual_seed_start + offset
        for seat in seats:
            baseline = bundle.adaptive_executor.play(
                genome, opponent, actual_seed, seat, False
            )
            base_own = float(baseline["rewards"][seat])
            baseline_cash.append(base_own)

            future_base = (
                args.future_seed_start + state_id * 20_000_003
            )
            single_row = None
            if not args.skip_single_stage:
                future_seeds = [
                    future_base + index for index in range(args.future_count)
                ]
                cf = bundle.adaptive_executor.candidate8_counterfactual(
                    genome,
                    opponent,
                    actual_seed,
                    future_seeds,
                    seat,
                    args.single_stage_day,
                    True,
                    args.maximum_arms,
                )
                expected = np.asarray(cf["rewards"], dtype=np.float64).mean(axis=0)
                rank = int(np.argmax(expected))
                played = bundle.adaptive_executor.play_candidate8(
                    genome,
                    opponent,
                    actual_seed,
                    seat,
                    rank,
                    args.single_stage_day,
                    True,
                    False,
                )
                single_own = float(played["rewards"][seat])
                single_cash.append(single_own)
                total_continuations += int(np.asarray(cf["rewards"]).size)
                single_row = {
                    "selected_rank": rank,
                    "selected_family": FAMILY_NAMES[
                        int(np.asarray(cf["arm_family"])[rank])
                    ],
                    "selected_expected_cash": float(expected[rank]),
                    "keep_expected_cash": float(expected[0]),
                    "realized_cash": single_own,
                    "realized_gain_vs_baseline": single_own - base_own,
                }

            rolling = bundle.adaptive_executor.candidate8_rolling_oracle(
                genome,
                opponent,
                actual_seed,
                args.decision_days,
                future_base + 10_000_019,
                args.future_count,
                seat,
                True,
                args.maximum_arms,
                args.clairvoyant_actual_future,
            )
            own = float(rolling["rewards"][seat])
            other = float(rolling["rewards"][1 - seat])
            rolling_cash.append(own)
            rolling_margin.append(own - other)
            total_continuations += int(rolling["complete_continuations"])
            total_overflow += int(rolling["end_overflow"])

            stages = []
            for index, family_value in enumerate(rolling["selected_family"]):
                family = FAMILY_NAMES[int(family_value)]
                gain = float(rolling["stage_expected_gain"][index])
                family_counts[family] += 1
                family_stage_gain[family].append(gain)
                stages.append(
                    {
                        "day": int(rolling["decision_day"][index]),
                        "step": int(rolling["decision_step"][index]),
                        "feasible_count": int(rolling["feasible_count"][index]),
                        "selected_rank": int(rolling["selected_rank"][index]),
                        "selected_family": family,
                        "selected_signature": int(
                            rolling["selected_signature"][index]
                        ),
                        "selected_expected_cash": float(
                            rolling["selected_expected_reward"][index]
                        ),
                        "keep_expected_cash": float(
                            rolling["keep_expected_reward"][index]
                        ),
                        "expected_gain_vs_keep": gain,
                    }
                )
            rows.append(
                {
                    "state_id": state_id,
                    "actual_seed": actual_seed,
                    "seat": seat,
                    "baseline_cash": base_own,
                    "single_stage": single_row,
                    "rolling_cash": own,
                    "opponent_cash": other,
                    "rolling_margin": own - other,
                    "rolling_gain_vs_baseline": own - base_own,
                    "end_overflow": int(rolling["end_overflow"]),
                    "avoidable_crop_losses": int(
                        rolling["avoidable_crop_losses"]
                    ),
                    "avoidable_animal_losses": int(
                        rolling["avoidable_animal_losses"]
                    ),
                    "complete_continuations": int(
                        rolling["complete_continuations"]
                    ),
                    "stages": stages,
                }
            )
            state_id += 1

    simulation_seconds = time.perf_counter() - started
    rolling_gains = [
        row["rolling_gain_vs_baseline"] for row in rows
    ]
    payload = {
        "schema": "kaggriculture.candidate8_rolling_oracle.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "opponent": args.opponent,
            "opponent_route": opponent,
            "actual_seed_start": args.actual_seed_start,
            "actual_seed_count": args.actual_seed_count,
            "future_seed_start": args.future_seed_start,
            "future_count": args.future_count,
            "decision_days": args.decision_days,
            "seats": list(seats),
            "maximum_arms": args.maximum_arms,
            "single_stage_day": None if args.skip_single_stage else args.single_stage_day,
            "threads": 16,
            "selection_objective": (
                "exact_actual_future_terminal_cash"
                if args.clairvoyant_actual_future
                else "mean_own_terminal_cash"
            ),
            "clairvoyant_actual_future": args.clairvoyant_actual_future,
            "actual_seed_is_disjoint_from_selection_future_bank": (
                not args.clairvoyant_actual_future
            ),
        },
        "summary": {
            "states": len(rows),
            "baseline_cash": distribution(baseline_cash),
            "single_stage_cash": (
                distribution(single_cash) if single_cash else None
            ),
            "rolling_oracle_cash": distribution(rolling_cash),
            "rolling_oracle_margin": distribution(rolling_margin),
            "rolling_gain_vs_baseline": distribution(rolling_gains),
            "rolling_positive_gain_rate": sum(value > 0 for value in rolling_gains)
            / len(rolling_gains),
            "end_overflow_events": total_overflow,
            "complete_continuations": total_continuations,
            "setup_seconds": setup_seconds,
            "simulation_seconds": simulation_seconds,
            "complete_continuations_per_second": total_continuations
            / simulation_seconds,
        },
        "selected_family": {
            family: {
                "count": count,
                "mean_stage_expected_gain": statistics.fmean(
                    family_stage_gain[family]
                ),
                "maximum_stage_expected_gain": max(family_stage_gain[family]),
            }
            for family, count in sorted(family_counts.items())
        },
        "states": rows,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
