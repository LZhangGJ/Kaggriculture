#!/usr/bin/env python3
"""Compare Candidate8 beam depths over the audited opponent pool."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_candidate8_multifuture_oracle import FAMILY_NAMES, load_genome, sha256
from generate_candidate8_competitive_pool import select_opponents
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def parse_ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def summarize(rows: list[dict]) -> dict:
    margins = np.asarray([row["margin"] for row in rows], dtype=np.float64)
    return {
        "games": len(rows),
        "wins": int((margins > 0).sum()),
        "ties": int((margins == 0).sum()),
        "losses": int((margins < 0).sum()),
        "score_rate": float(np.mean(np.where(
            margins > 0, 1.0, np.where(margins == 0, 0.5, 0.0)
        ))),
        "mean_margin": float(margins.mean()),
        "expanded_nodes": int(sum(row["expanded_nodes"] for row in rows)),
        "complete_continuations": int(
            sum(row["complete_continuations"] for row in rows)
        ),
    }


def summarize_full_paths(rows: list[dict]) -> dict:
    margins = np.asarray([
        margin
        for row in rows
        for margin in row.get("final_path_margins", [])
    ], dtype=np.float64)
    if margins.size == 0:
        return {"paths": 0}
    wins = margins[margins > 0]
    ties = margins[margins == 0]
    losses = margins[margins < 0]

    def distribution(values: np.ndarray) -> dict:
        if values.size == 0:
            return {"count": 0}
        return {
            "count": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "minimum": float(values.min()),
            "p10": float(np.quantile(values, 0.10)),
            "p25": float(np.quantile(values, 0.25)),
            "p75": float(np.quantile(values, 0.75)),
            "p90": float(np.quantile(values, 0.90)),
            "maximum": float(values.max()),
        }

    return {
        "paths": int(margins.size),
        "wins": int(wins.size),
        "ties": int(ties.size),
        "losses": int(losses.size),
        "win_rate": float(wins.size / margins.size),
        "non_loss_rate": float((wins.size + ties.size) / margins.size),
        "all_margin": distribution(margins),
        "winning_margin": distribution(wins),
        "losing_margin": distribution(losses),
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
        "--group", choices=("hard16", "selector43", "all"), default="hard16"
    )
    parser.add_argument(
        "--opponents",
        type=str,
        help="Optional comma-separated route names; overrides --group.",
    )
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--single-seat", type=int, choices=(0, 1))
    parser.add_argument("--decision-days", type=parse_ints, default=parse_ints("3,6,9"))
    parser.add_argument("--depths", type=parse_ints, default=parse_ints("1,2,3"))
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--per-node-arms", type=int, default=32)
    parser.add_argument(
        "--candidate-pool",
        choices=("feasible", "shortlist"),
        default="feasible",
        help="Use the generation-order feasible pool or the diverse shortlist.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if any(depth <= 0 or depth > len(args.decision_days) for depth in args.depths):
        raise ValueError("depths must be within the decision-day list")

    merged = json.loads(args.merged_receipt.read_text(encoding="utf-8"))
    opponent_names = (
        [value.strip() for value in args.opponents.split(",") if value.strip()]
        if args.opponents
        else select_opponents(merged, args.group)
    )
    if not opponent_names:
        raise ValueError("at least one opponent is required")
    seats = (args.single_seat,) if args.single_seat is not None else (0, 1)
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = load_genome(args.genomes, args.genome_index)
    use_feasible_pool = args.candidate_pool == "feasible"

    all_rows: dict[int, list[dict]] = defaultdict(list)
    route_rows: dict[str, dict[int, list[dict]]] = {
        opponent: defaultdict(list) for opponent in opponent_names
    }
    family_counts: dict[int, Counter[str]] = {
        depth: Counter() for depth in args.depths
    }
    started = time.perf_counter()
    for opponent_name in opponent_names:
        opponent = bundle.index(opponent_name)
        for seed in range(args.seed_start, args.seed_start + args.seed_count):
            for seat in seats:
                for depth in args.depths:
                    days = args.decision_days[:depth]
                    result = bundle.adaptive_executor.candidate8_sequence_oracle(
                        genome,
                        opponent,
                        seed,
                        days,
                        seat,
                        args.beam_width,
                        args.per_node_arms,
                        use_feasible_pool,
                        True,
                    )
                    own = float(result["rewards"][seat])
                    rival = float(result["rewards"][1 - seat])
                    families = [
                        "NONE" if int(value) < 0 else FAMILY_NAMES[int(value)]
                        for value in result["selected_family"]
                    ]
                    family_counts[depth].update(families)
                    final_candidate_rewards = np.asarray(
                        result.get("final_path_candidate_rewards", []),
                        dtype=np.float64,
                    )
                    final_opponent_rewards = np.asarray(
                        result.get("final_path_opponent_rewards", []),
                        dtype=np.float64,
                    )
                    if final_candidate_rewards.shape != final_opponent_rewards.shape:
                        raise RuntimeError("final-path reward arrays have different shapes")
                    sequence_length = int(result.get("final_path_sequence_length", 0))
                    flat_final_ranks = np.asarray(
                        result.get("final_path_selected_ranks", []), dtype=np.int16
                    )
                    flat_final_families = np.asarray(
                        result.get("final_path_selected_families", []), dtype=np.int8
                    )
                    if sequence_length:
                        final_ranks = flat_final_ranks.reshape(-1, sequence_length)
                        final_families = flat_final_families.reshape(-1, sequence_length)
                    else:
                        final_ranks = np.empty((0, 0), dtype=np.int16)
                        final_families = np.empty((0, 0), dtype=np.int8)
                    if final_ranks.shape[0] != final_candidate_rewards.size:
                        raise RuntimeError("final-path sequence and reward counts differ")
                    row = {
                        "opponent": opponent_name,
                        "seed": seed,
                        "seat": seat,
                        "cash": own,
                        "opponent_cash": rival,
                        "margin": own - rival,
                        "selected_families": families,
                        "selected_ranks": [int(value) for value in result["selected_rank"]],
                        "expanded_nodes": int(result["expanded_nodes"]),
                        "complete_continuations": int(result["complete_continuations"]),
                        "final_path_margins": (
                            final_candidate_rewards - final_opponent_rewards
                        ).tolist(),
                        "final_path_selected_ranks": final_ranks.tolist(),
                        "final_path_selected_families": final_families.tolist(),
                    }
                    all_rows[depth].append(row)
                    route_rows[opponent_name][depth].append(row)
    simulation_seconds = time.perf_counter() - started

    per_opponent: list[dict] = []
    for opponent in opponent_names:
        entry = {"opponent": opponent, "depths": {}}
        baseline_depth = min(args.depths)
        depth1 = summarize(route_rows[opponent][baseline_depth])
        for depth in args.depths:
            summary = summarize(route_rows[opponent][depth])
            summary["score_gain_vs_depth1"] = (
                summary["score_rate"] - depth1["score_rate"]
            )
            summary["margin_gain_vs_depth1"] = (
                summary["mean_margin"] - depth1["mean_margin"]
            )
            entry["depths"][str(depth)] = summary
        per_opponent.append(entry)

    summaries: dict[str, dict] = {}
    for depth in args.depths:
        summary = summarize(all_rows[depth])
        route_rates = np.asarray([
            summarize(route_rows[opponent][depth])["score_rate"]
            for opponent in opponent_names
        ])
        summary.update({
            "decision_days": args.decision_days[:depth],
            "opponents_at_least_90pct": int((route_rates >= 0.90).sum()),
            "opponents_at_least_50pct": int((route_rates >= 0.50).sum()),
            "worst_opponent_score": float(route_rates.min()),
            "selected_family_counts": dict(sorted(family_counts[depth].items())),
            "full_path_outcomes": summarize_full_paths(all_rows[depth]),
        })
        summaries[str(depth)] = summary

    payload = {
        "schema": "kaggriculture.candidate8-sequence-pool.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "EXACT_FUTURE_DIAGNOSTIC",
        "boundary": (
            "Every sequence is evaluated against the exact frozen opponent "
            "route and actual future. This is an offline capacity probe, not a "
            "deployable selector."
        ),
        "config": {
            "group": args.group,
            "explicit_opponents": args.opponents,
            "opponents": len(opponent_names),
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "seats": list(seats),
            "decision_days": args.decision_days,
            "depths": args.depths,
            "beam_width": args.beam_width,
            "per_node_arms": args.per_node_arms,
            "candidate_pool": args.candidate_pool,
        },
        "summary": summaries,
        "simulation_seconds": simulation_seconds,
        "per_opponent": per_opponent,
        "rows": {str(depth): all_rows[depth] for depth in args.depths},
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
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "summary": summaries,
        "simulation_seconds": simulation_seconds,
        "largest_depth_gains": sorted(
            [
                {
                    "opponent": row["opponent"],
                    **row["depths"][str(max(args.depths))],
                }
                for row in per_opponent
            ],
            key=lambda row: -row["score_gain_vs_depth1"],
        )[:16],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
