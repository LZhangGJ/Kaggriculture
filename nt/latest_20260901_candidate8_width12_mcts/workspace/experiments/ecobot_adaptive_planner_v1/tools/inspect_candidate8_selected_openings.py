#!/usr/bin/env python3
"""Decode the selected Day0/Day1 Candidate8 edits from a sequence receipt."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from audit_candidate8_multifuture_oracle import FAMILY_NAMES, load_genome
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


def decode(features: np.ndarray) -> dict:
    delta = {
        PROJECTS[index]: int(value)
        for index, value in enumerate(features[:8])
        if int(value) != 0
    }
    schedule = int(features[11])
    market = int(features[12])
    return {
        "target_delta": delta,
        "hand_delta": int(features[8]),
        "quadrant_delta": int(features[9]),
        "delay_days": int(features[10]),
        "schedule": SCHEDULES[schedule] if 0 <= schedule < len(SCHEDULES) else schedule,
        "market": MARKETS[market] if 0 <= market < len(MARKETS) else market,
        "recovery_profile": int(features[13]),
        "suffix_project": (
            PROJECTS[int(features[14])] if 0 <= int(features[14]) < 8 else None
        ),
        "market_item": int(features[15]),
        "recovery_issue": int(features[16]),
        "estimated_value": float(features[17]) / 100.0,
        "estimated_cash_cost": float(features[18]) / 100.0,
        "estimated_daily_action_load": float(features[19]) / 100.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--opponent", default="G001")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    rows = receipt["rows"]["8"]
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata, None)
    opponent = bundle.index(args.opponent)
    genome = load_genome(args.genomes, 0)
    output_rows = []

    for row in rows:
        chosen = []
        ranks = [int(value) for value in row["selected_ranks"]]
        for index, day in enumerate((0, 1)):
            result = bundle.adaptive_executor.candidate8_counterfactual(
                genome,
                opponent,
                int(row["seed"]),
                [int(row["seed"])],
                int(row["seat"]),
                day,
                False,
                64,
                [0][:index],
                ranks[:index],
            )
            rank = ranks[index]
            features = np.asarray(result["arm_features"], dtype=np.int32)
            families = np.asarray(result["arm_family"], dtype=np.int8)
            signatures = np.asarray(result["arm_signature"], dtype=np.uint64)
            chosen.append({
                "day": day,
                "rank": rank,
                "family": FAMILY_NAMES[int(families[rank])],
                "signature": int(signatures[rank]),
                **decode(features[rank]),
            })
        surviving_pairs = Counter(
            (int(sequence[0]), int(sequence[1]))
            for sequence in row.get("final_path_selected_ranks", [])
        )
        surviving_openings = []
        for (day0_rank, day1_rank), count in sorted(surviving_pairs.items()):
            day0_result = bundle.adaptive_executor.candidate8_counterfactual(
                genome, opponent, int(row["seed"]), [int(row["seed"])],
                int(row["seat"]), 0, False, 64, [], [],
            )
            day1_result = bundle.adaptive_executor.candidate8_counterfactual(
                genome, opponent, int(row["seed"]), [int(row["seed"])],
                int(row["seat"]), 1, False, 64, [0], [day0_rank],
            )
            decoded = []
            for day, rank, result in (
                (0, day0_rank, day0_result), (1, day1_rank, day1_result)
            ):
                features = np.asarray(result["arm_features"], dtype=np.int32)
                families = np.asarray(result["arm_family"], dtype=np.int8)
                decoded.append({
                    "day": day,
                    "rank": rank,
                    "family": FAMILY_NAMES[int(families[rank])],
                    **decode(features[rank]),
                })
            surviving_openings.append({"paths": count, "opening": decoded})

        output_rows.append({
            "seed": int(row["seed"]),
            "seat": int(row["seat"]),
            "terminal_cash": float(row["cash"]),
            "terminal_margin": float(row["margin"]),
            "opening": chosen,
            "surviving_openings": surviving_openings,
        })

    payload = {"schema": "candidate8-selected-openings.v1", "rows": output_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
