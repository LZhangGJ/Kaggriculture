#!/usr/bin/env python3
"""Test a frozen semantic Day0/Day1 opening pool on unseen seeds."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_candidate8_multifuture_oracle import FAMILY_NAMES, load_genome, sha256
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


PROJECTS = [
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "GOOSE", "COW", "SHEEP",
]
SCHEDULES = [
    "CURRENT", "MIN_TOTAL_TRAVEL", "DEADLINE_FIRST", "VALUE_PER_STEP",
    "REGION_BALANCED", "GLOBAL_MATCHING", "CHAIN_CONTINUITY",
    "COMPACT_LAYOUT", "SERVICE_LANES",
]
MARKETS = ["CURRENT", "SELL_NOW", "HOLD_FOR_DEMAND", "PARTIAL_SELL", "SELL_TO_FINANCE"]


def semantic_key(row: dict) -> tuple[int, ...]:
    target = row.get("target_delta", {})
    suffix = row.get("suffix_project")
    return (
        *(int(target.get(project, 0)) for project in PROJECTS),
        int(row["hand_delta"]),
        int(row["quadrant_delta"]),
        int(row["delay_days"]),
        SCHEDULES.index(row["schedule"]),
        MARKETS.index(row["market"]),
        int(row["recovery_profile"]),
        -1 if suffix is None else PROJECTS.index(suffix),
        int(row["market_item"]),
        int(row["recovery_issue"]),
    )


def load_templates(path: Path) -> list[tuple[tuple[str, tuple[int, ...]], ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    templates: set[tuple[tuple[str, tuple[int, ...]], ...]] = set()
    for row in payload["rows"]:
        for survivor in row["surviving_openings"]:
            templates.add(tuple(
                (stage["family"], semantic_key(stage))
                for stage in survivor["opening"]
            ))
    return sorted(templates)


def result_row(mode: str, result: dict, seat: int) -> dict:
    own = float(result["rewards"][seat])
    rival = float(result["rewards"][1 - seat])
    return {
        "mode": mode,
        "cash": own,
        "opponent_cash": rival,
        "margin": own - rival,
        "win": own > rival,
        "selected_ranks": [int(value) for value in result["selected_rank"]],
        "selected_families": [
            FAMILY_NAMES[int(value)] if int(value) >= 0 else "NONE"
            for value in result["selected_family"]
        ],
        "expanded_nodes": int(result["expanded_nodes"]),
    }


def summarize(rows: list[dict], mode: str) -> dict:
    selected = [row for row in rows if row["mode"] == mode]
    margins = np.asarray([row["margin"] for row in selected], dtype=np.float64)
    return {
        "games": len(selected),
        "wins": int((margins > 0).sum()),
        "ties": int((margins == 0).sum()),
        "losses": int((margins < 0).sum()),
        "win_rate": float((margins > 0).mean()),
        "mean_margin": float(margins.mean()),
        "minimum_margin": float(margins.min()),
        "median_margin": float(np.median(margins)),
        "maximum_margin": float(margins.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opening-receipt", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--opponent", default="G001")
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-count", default=4, type=int)
    parser.add_argument("--beam-width", default=32, type=int)
    parser.add_argument("--per-node-arms", default=64, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    templates = load_templates(args.opening_receipt)
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata, None)
    opponent = bundle.index(args.opponent)
    genome = load_genome(args.genomes, 0)
    later_days = [3, 6, 9, 12, 18, 24]
    full_days = [0, 1, *later_days]
    rows: list[dict] = []
    started = time.perf_counter()

    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        for seat in (0, 1):
            day0 = bundle.adaptive_executor.candidate8_counterfactual(
                genome, opponent, seed, [seed], seat, 0, False, 64, [], []
            )
            day0_features = np.asarray(day0["arm_features"], dtype=np.int32)
            day0_families = np.asarray(day0["arm_family"], dtype=np.int8)
            day0_lookup: dict[tuple[str, tuple[int, ...]], list[int]] = {}
            for rank in range(day0_features.shape[0]):
                key = (FAMILY_NAMES[int(day0_families[rank])], tuple(
                    int(value) for value in day0_features[rank, :17]
                ))
                day0_lookup.setdefault(key, []).append(rank)

            rank_pairs: set[tuple[int, int]] = set()
            day1_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for template in templates:
                for day0_rank in day0_lookup.get(template[0], []):
                    if day0_rank not in day1_cache:
                        day1 = bundle.adaptive_executor.candidate8_counterfactual(
                            genome, opponent, seed, [seed], seat, 1, False, 64,
                            [0], [day0_rank],
                        )
                        day1_cache[day0_rank] = (
                            np.asarray(day1["arm_features"], dtype=np.int32),
                            np.asarray(day1["arm_family"], dtype=np.int8),
                        )
                    features, families = day1_cache[day0_rank]
                    for day1_rank in range(features.shape[0]):
                        key = (FAMILY_NAMES[int(families[day1_rank])], tuple(
                            int(value) for value in features[day1_rank, :17]
                        ))
                        if key == template[1]:
                            rank_pairs.add((day0_rank, day1_rank))

            base = {"seed": seed, "seat": seat, "matched_openings": len(rank_pairs)}
            if rank_pairs:
                restricted = bundle.adaptive_executor.candidate8_sequence_oracle(
                    genome, opponent, seed, later_days, seat,
                    args.beam_width, args.per_node_arms, False, True,
                    [0, 1], sorted(rank_pairs),
                )
                rows.append({**base, **result_row("frozen_opening_pool_plus_later_oracle", restricted, seat)})
            default = bundle.adaptive_executor.candidate8_sequence_oracle(
                genome, opponent, seed, later_days, seat,
                args.beam_width, args.per_node_arms, False, True,
            )
            rows.append({**base, **result_row("r6_opening_plus_later_oracle", default, seat)})
            full = bundle.adaptive_executor.candidate8_sequence_oracle(
                genome, opponent, seed, full_days, seat,
                args.beam_width, args.per_node_arms, False, True,
            )
            rows.append({**base, **result_row("full_eight_stage_oracle", full, seat)})

    modes = sorted({row["mode"] for row in rows})
    payload = {
        "schema": "candidate8-opening-pool-unseen.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "The Day0/Day1 pool is frozen from prior seeds. Later decisions and "
            "the full comparator use the exact actual future and are offline Oracles."
        ),
        "config": {
            "opponent": args.opponent,
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "seats": [0, 1],
            "opening_templates": len(templates),
            "later_days": later_days,
            "beam_width": args.beam_width,
            "per_node_arms": args.per_node_arms,
        },
        "summary": {mode: summarize(rows, mode) for mode in modes},
        "simulation_seconds": time.perf_counter() - started,
        "rows": rows,
        "inputs": {
            "opening_receipt": {"path": str(args.opening_receipt), "sha256": sha256(args.opening_receipt)},
            "genomes": {"path": str(args.genomes), "sha256": sha256(args.genomes)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "rows": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
