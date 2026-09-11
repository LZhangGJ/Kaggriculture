#!/usr/bin/env python3
"""Measure one/two/three-stage Candidate8 sequence headroom on many rivals.

This is an exact-future representational audit, not a deployable selector.  It
answers only whether complementary Day6/12/18 edits exist inside the current
candidate language and executor.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_candidate8_multifuture_oracle import FAMILY_NAMES, load_genome, sha256
from generate_candidate8_competitive_pool import select_opponents
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def summarize(rows: list[dict]) -> dict:
    margin = np.asarray([row["margin"] for row in rows], dtype=np.float64)
    own = np.asarray([row["own_cash"] for row in rows], dtype=np.float64)
    return {
        "cases": len(rows),
        "score_rate": float(np.mean((margin > 0) + 0.5 * (margin == 0))),
        "mean_margin": float(np.mean(margin)),
        "minimum_margin": float(np.min(margin)),
        "mean_own_cash": float(np.mean(own)),
        "non_keep_later_stage_rate": float(np.mean([
            any(rank > 0 for rank in row["selected_ranks"][1:])
            for row in rows
        ])) if rows and len(rows[0]["selected_ranks"]) > 1 else 0.0,
        "complete_continuations": int(sum(row["complete_continuations"] for row in rows)),
        "expanded_nodes": int(sum(row["expanded_nodes"] for row in rows)),
        "maximum_live_beam": int(max(row["maximum_live_beam"] for row in rows)),
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
    parser.add_argument("--group", choices=("selector43", "hard16", "all"), default="all")
    parser.add_argument("--opponents")
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--decision-days", type=int, nargs="+", default=[6, 12, 18])
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--per-node-arms", type=int, default=32)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    merged = json.loads(args.merged_receipt.read_text(encoding="utf-8"))
    opponents = (
        [value.strip() for value in args.opponents.split(",") if value.strip()]
        if args.opponents else select_opponents(merged, args.group)
    )
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata, args.backbone)
    genome = load_genome(args.genomes, args.genome_index)
    rows_by_depth: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    started = time.perf_counter()
    for opponent_name in opponents:
        opponent = bundle.index(opponent_name)
        for seed in range(args.seed_start, args.seed_start + args.seed_count):
            for seat in (0, 1):
                for depth in (1, 2, 3):
                    result = bundle.adaptive_executor.candidate8_sequence_oracle(
                        genome, opponent, seed, args.decision_days[:depth], seat,
                        args.beam_width, args.per_node_arms, True, True,
                    )
                    if len(result["selected_rank"]) != depth:
                        raise RuntimeError(
                            "sequence oracle returned an incomplete decision path: "
                            f"depth={depth}, ranks={result['selected_rank']}"
                        )
                    own = float(result["rewards"][seat])
                    rival = float(result["rewards"][1 - seat])
                    rows_by_depth[depth].append({
                        "opponent": opponent_name, "seed": seed, "seat": seat,
                        "own_cash": own, "opponent_cash": rival,
                        "margin": own - rival,
                        "selected_ranks": [int(x) for x in result["selected_rank"]],
                        "selected_families": [
                            "NONE" if int(x) < 0 else FAMILY_NAMES[int(x)]
                            for x in result["selected_family"]
                        ],
                        "complete_continuations": int(result["complete_continuations"]),
                        "expanded_nodes": int(result["expanded_nodes"]),
                        "maximum_live_beam": int(result["maximum_live_beam"]),
                    })

    summary = {f"depth{depth}": summarize(rows) for depth, rows in rows_by_depth.items()}
    baseline = {
        (row["opponent"], row["seed"], row["seat"]): row
        for row in rows_by_depth[1]
    }
    for depth in (2, 3):
        gains = np.asarray([
            row["margin"] - baseline[(row["opponent"], row["seed"], row["seat"])]["margin"]
            for row in rows_by_depth[depth]
        ], dtype=np.float64)
        summary[f"depth{depth}"]["mean_margin_gain_vs_depth1"] = float(np.mean(gains))
        summary[f"depth{depth}"]["improved_case_rate_vs_depth1"] = float(np.mean(gains > 0))
    depth2_by_case = {
        (row["opponent"], row["seed"], row["seat"]): row
        for row in rows_by_depth[2]
    }
    gains_3_vs_2 = np.asarray([
        row["margin"] - depth2_by_case[
            (row["opponent"], row["seed"], row["seat"])
        ]["margin"]
        for row in rows_by_depth[3]
    ], dtype=np.float64)
    summary["depth3"]["mean_margin_gain_vs_depth2"] = float(np.mean(gains_3_vs_2))
    summary["depth3"]["improved_case_rate_vs_depth2"] = float(np.mean(gains_3_vs_2 > 0))

    payload = {
        "schema": "kaggriculture.candidate8-o15-cross-stage-headroom.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AUDIT_COMPLETE",
        "boundary": (
            "Exact actual future and frozen opponent route are used only to "
            "measure representational headroom. Results are not deployable labels."
        ),
        "parameters": {
            "opponents": opponents, "seed_start": args.seed_start,
            "seed_count": args.seed_count, "decision_days": args.decision_days,
            "beam_width": args.beam_width, "per_node_arms": args.per_node_arms,
        },
        "summary": summary,
        "simulation_seconds": time.perf_counter() - started,
        "rows": {f"depth{depth}": rows for depth, rows in rows_by_depth.items()},
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in {
                "source": args.source, "actions": args.actions,
                "metadata": args.metadata, "genomes": args.genomes,
                "merged_receipt": args.merged_receipt,
            }.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "simulation_seconds": payload["simulation_seconds"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
