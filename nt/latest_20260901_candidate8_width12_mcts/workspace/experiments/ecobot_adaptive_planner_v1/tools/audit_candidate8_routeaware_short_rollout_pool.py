#!/usr/bin/env python3
"""Audit limited Candidate8 rollouts over a frozen C++ opponent pool.

This is deliberately labelled *route-aware*: every hypothetical continuation
uses the exact frozen opponent route that produced the current state.  The RNG
future may be independently sampled, but knowing the opponent route is still
not information available to a submitted agent.  Therefore this tool measures
whether a short-rollout selector is worth pursuing; it does not measure a
deployable policy.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_candidate8_multifuture_oracle import FAMILY_NAMES, load_genome
from generate_candidate8_competitive_pool import select_opponents, sha256
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def parse_days(raw: str) -> list[int]:
    days = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not days or days[0] < 0 or days[-1] > 29:
        raise argparse.ArgumentTypeError("days must be a non-empty subset of 0..29")
    return days


def competitive(margins: list[float]) -> dict[str, float | int]:
    wins = sum(value > 0 for value in margins)
    ties = sum(value == 0 for value in margins)
    losses = sum(value < 0 for value in margins)
    games = len(margins)
    return {
        "games": games,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "score_rate": (wins + 0.5 * ties) / games if games else 0.0,
        "win_rate": wins / games if games else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
    }


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
    parser.add_argument("--actual-seed-start", required=True, type=int)
    parser.add_argument("--actual-seed-count", type=int, default=2)
    parser.add_argument("--future-seed-start", required=True, type=int)
    parser.add_argument("--future-count", type=int, default=1)
    parser.add_argument("--days", type=parse_days, default=parse_days("9,15"))
    parser.add_argument("--maximum-arms", type=int, default=64)
    parser.add_argument("--single-seat", type=int, choices=(0, 1))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    merged = json.loads(args.merged_receipt.read_text(encoding="utf-8"))
    opponent_names = select_opponents(merged, args.group)
    seats = (args.single_seat,) if args.single_seat is not None else (0, 1)

    setup_started = time.perf_counter()
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)
    setup_seconds = time.perf_counter() - setup_started

    baseline_margins: list[float] = []
    rolling_margins: list[float] = []
    baseline_cash: list[float] = []
    rolling_cash: list[float] = []
    rows: list[dict] = []
    family_counts: Counter[str] = Counter()
    total_continuations = 0
    total_overflow = 0
    started = time.perf_counter()

    for route_offset, opponent_name in enumerate(opponent_names):
        opponent = bundle.index(opponent_name)
        route_base: list[float] = []
        route_roll: list[float] = []
        route_rows: list[dict] = []
        for seed_offset in range(args.actual_seed_count):
            actual_seed = args.actual_seed_start + seed_offset
            for seat in seats:
                baseline = bundle.adaptive_executor.play(
                    genome, opponent, actual_seed, seat, False
                )
                base_own = float(baseline["rewards"][seat])
                base_opp = float(baseline["rewards"][1 - seat])
                base_margin = base_own - base_opp

                future_base = (
                    args.future_seed_start
                    + route_offset * 1_000_000_007
                    + seed_offset * 10_000_019
                    + seat * 1_000_003
                )
                rolling = bundle.adaptive_executor.candidate8_rolling_oracle(
                    genome,
                    opponent,
                    actual_seed,
                    args.days,
                    future_base,
                    args.future_count,
                    seat,
                    True,
                    args.maximum_arms,
                    False,
                    True,
                    False,
                    1,
                    8,
                    24,
                    8,
                    8,
                    4,
                    0.01,
                )
                own = float(rolling["rewards"][seat])
                opp = float(rolling["rewards"][1 - seat])
                margin = own - opp
                selected_families = [
                    FAMILY_NAMES[int(value)] for value in rolling["selected_family"]
                ]
                family_counts.update(selected_families)
                total_continuations += int(rolling["complete_continuations"])
                total_overflow += int(rolling["end_overflow"])

                baseline_margins.append(base_margin)
                rolling_margins.append(margin)
                baseline_cash.append(base_own)
                rolling_cash.append(own)
                route_base.append(base_margin)
                route_roll.append(margin)
                route_rows.append({
                    "seed": actual_seed,
                    "seat": seat,
                    "baseline_cash": base_own,
                    "baseline_opponent_cash": base_opp,
                    "baseline_margin": base_margin,
                    "rolling_cash": own,
                    "rolling_opponent_cash": opp,
                    "rolling_margin": margin,
                    "selected_rank": [int(v) for v in rolling["selected_rank"]],
                    "selected_family": selected_families,
                    "selected_expected_win_rate": [
                        float(v) for v in rolling["selected_expected_win_rate"]
                    ],
                })
        rows.append({
            "opponent": opponent_name,
            "baseline": competitive(route_base),
            "routeaware_short_rollout": competitive(route_roll),
            "rows": route_rows,
        })

    simulation_seconds = time.perf_counter() - started
    baseline_summary = competitive(baseline_margins)
    rolling_summary = competitive(rolling_margins)
    route_rates = [row["routeaware_short_rollout"]["score_rate"] for row in rows]
    result = {
        "schema": "kaggriculture.candidate8-routeaware-short-rollout-pool.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC_ONLY",
        "boundary": (
            "Not deployable: continuations know the exact frozen opponent route. "
            "Only future RNG is independently sampled."
        ),
        "config": {
            "group": args.group,
            "opponents": len(opponent_names),
            "actual_seed_start": args.actual_seed_start,
            "actual_seed_count": args.actual_seed_count,
            "seats": list(seats),
            "decision_days": args.days,
            "future_seed_start": args.future_seed_start,
            "future_count": args.future_count,
            "maximum_arms": args.maximum_arms,
            "objective": "expected score rate, then margin, then own cash",
            "clairvoyant_actual_future": False,
            "exact_opponent_route_known": True,
        },
        "summary": {
            "baseline": baseline_summary,
            "routeaware_short_rollout": rolling_summary,
            "score_rate_gain": (
                rolling_summary["score_rate"] - baseline_summary["score_rate"]
            ),
            "mean_cash_gain": float(np.mean(rolling_cash) - np.mean(baseline_cash)),
            "routes_at_least_90pct": sum(value >= 0.90 for value in route_rates),
            "routes_at_least_50pct": sum(value >= 0.50 for value in route_rates),
            "total_continuations": total_continuations,
            "end_overflow": total_overflow,
            "setup_seconds": setup_seconds,
            "simulation_seconds": simulation_seconds,
            "continuations_per_second": (
                total_continuations / simulation_seconds if simulation_seconds else 0.0
            ),
            "mean_decision_latency_ms": (
                simulation_seconds * 1000.0
                / (len(opponent_names) * args.actual_seed_count * len(seats) * len(args.days))
            ),
        },
        "selected_family_counts": dict(sorted(family_counts.items())),
        "inputs": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "genomes": {"path": str(args.genomes), "sha256": sha256(args.genomes)},
            "merged_receipt": {
                "path": str(args.merged_receipt),
                "sha256": sha256(args.merged_receipt),
            },
        },
        "opponents": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
