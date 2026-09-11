#!/usr/bin/env python3
"""Compare one-, two-, and three-stage exact-future Candidate8 beam search."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_candidate8_multifuture_oracle import FAMILY_NAMES, load_genome, sha256
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def summarize(rows: list[dict]) -> dict:
    own = np.asarray([row["cash"] for row in rows], dtype=np.float64)
    rival = np.asarray([row["opponent_cash"] for row in rows], dtype=np.float64)
    margin = own - rival
    return {
        "states": len(rows),
        "wins": int(np.count_nonzero(margin > 0)),
        "ties": int(np.count_nonzero(margin == 0)),
        "losses": int(np.count_nonzero(margin < 0)),
        "score_rate": float(np.mean((margin > 0) + 0.5 * (margin == 0))),
        "mean_cash": float(own.mean()),
        "median_cash": float(np.median(own)),
        "mean_opponent_cash": float(rival.mean()),
        "mean_margin": float(margin.mean()),
        "median_margin": float(np.median(margin)),
        "minimum_margin": float(margin.min()),
        "maximum_margin": float(margin.max()),
        "expanded_nodes": int(sum(row["expanded_nodes"] for row in rows)),
        "complete_continuations": int(
            sum(row["complete_continuations"] for row in rows)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--genomes", type=Path, required=True)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--opponent", default="G001")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--single-seat", type=int, choices=(0, 1))
    parser.add_argument("--decision-days", type=int, nargs="+", default=[3, 6, 9])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--per-node-arms", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if any(depth <= 0 or depth > len(args.decision_days) for depth in args.depths):
        raise ValueError("every depth must be within the decision-day list")

    setup_started = time.perf_counter()
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    setup_seconds = time.perf_counter() - setup_started
    genome = load_genome(args.genomes, args.genome_index)
    opponent = -1 if args.opponent.upper() == "PASSIVE" else bundle.index(args.opponent)
    seats = (args.single_seat,) if args.single_seat is not None else (0, 1)

    all_rows: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    started = time.perf_counter()
    for depth in args.depths:
        rows: list[dict] = []
        days = args.decision_days[:depth]
        family_counts: Counter[str] = Counter()
        for seed in range(args.seed_start, args.seed_start + args.seed_count):
            for seat in seats:
                result = bundle.adaptive_executor.candidate8_sequence_oracle(
                    genome,
                    opponent,
                    seed,
                    days,
                    seat,
                    args.beam_width,
                    args.per_node_arms,
                    True,
                    True,
                )
                own = float(result["rewards"][seat])
                rival = float(result["rewards"][1 - seat])
                families = [
                    "NONE" if int(value) < 0 else FAMILY_NAMES[int(value)]
                    for value in result["selected_family"]
                ]
                family_counts.update(families)
                rows.append(
                    {
                        "actual_seed": seed,
                        "seat": seat,
                        "cash": own,
                        "opponent_cash": rival,
                        "margin": own - rival,
                        "decision_days": [int(value) for value in result["decision_day"]],
                        "selected_ranks": [int(value) for value in result["selected_rank"]],
                        "selected_families": families,
                        "selected_signatures": [
                            int(value) for value in result["selected_signature"]
                        ],
                        "feasible_counts": [
                            int(value) for value in result["feasible_count"]
                        ],
                        "expanded_nodes": int(result["expanded_nodes"]),
                        "complete_continuations": int(
                            result["complete_continuations"]
                        ),
                        "maximum_live_beam": int(result["maximum_live_beam"]),
                    }
                )
        key = f"depth_{depth}"
        all_rows[key] = rows
        summaries[key] = summarize(rows)
        summaries[key]["decision_days"] = days
        summaries[key]["selected_family_counts"] = dict(sorted(family_counts.items()))

    simulation_seconds = time.perf_counter() - started
    baseline_key = f"depth_{min(args.depths)}"
    baseline_by_state = {
        (row["actual_seed"], row["seat"]): row for row in all_rows[baseline_key]
    }
    for key, rows in all_rows.items():
        if key == baseline_key:
            continue
        margin_gain = np.asarray(
            [
                row["margin"]
                - baseline_by_state[(row["actual_seed"], row["seat"])]["margin"]
                for row in rows
            ],
            dtype=np.float64,
        )
        summaries[key]["margin_gain_vs_depth1_mean"] = float(margin_gain.mean())
        summaries[key]["states_improved_vs_depth1"] = int(
            np.count_nonzero(margin_gain > 0)
        )

    payload = {
        "schema": "kaggriculture.candidate8_sequence_oracle.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "opponent": args.opponent,
            "opponent_route": opponent,
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "seats": list(seats),
            "decision_days": args.decision_days,
            "depths": args.depths,
            "beam_width": args.beam_width,
            "per_node_arms": args.per_node_arms,
            "objective": "exact_future_win_then_margin_then_own_cash",
            "candidate_pool": "complete_feasible_pool_capped_per_node",
        },
        "summary": {
            **summaries,
            "setup_seconds": setup_seconds,
            "simulation_seconds": simulation_seconds,
        },
        "states": all_rows,
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
            "path": str(args.backbone), "sha256": sha256(args.backbone)
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
