#!/usr/bin/env python3
"""Decode whether the winning Candidate8 MCTS path mirrors OceanMix."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_candidate8_multifuture_oracle import FAMILY_NAMES, load_genome
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


PROJECT_NAMES = [
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "GOOSE", "COW", "SHEEP"
]
MOVE = {"PASS", "NORTH", "SOUTH", "EAST", "WEST", "DIG"}


def commands(action: dict[str, Any]):
    farmer = action.get("farmer") or ["PASS"]
    if farmer and isinstance(farmer[0], list):
        farmer = farmer[0]
    yield farmer
    yield from action.get("hands") or []


def summarize(actions: list[dict[str, Any]]) -> dict[str, Any]:
    unit = Counter()
    market_orders = Counter()
    market_quantity = Counter()
    for action in actions:
        for command in commands(action):
            if not command or command[0] in MOVE:
                continue
            item = str(command[1]) if len(command) >= 2 else "_"
            unit[f"{command[0]}:{item}"] += 1
        for order in action.get("market") or []:
            if not order:
                continue
            item = str(order[1]) if len(order) >= 2 else "_"
            key = f"{order[0]}:{item}"
            market_orders[key] += 1
            quantity = order[2] if len(order) >= 3 and isinstance(order[2], (int, float)) else 1
            market_quantity[key] += float(quantity)
    return {
        "unit_action_counts": dict(sorted(unit.items())),
        "market_order_counts": dict(sorted(market_orders.items())),
        "market_quantities": dict(sorted(market_quantity.items())),
    }


def selected_arm_details(
    bundle: NativeTeammateBundle,
    genome: np.ndarray,
    opponent: int,
    seed: int,
    seat: int,
    days: list[int],
    ranks: list[int],
) -> list[dict[str, Any]]:
    output = []
    for index, (day, rank) in enumerate(zip(days, ranks, strict=True)):
        result = bundle.adaptive_executor.candidate8_counterfactual(
            genome,
            opponent,
            seed,
            [seed],
            seat,
            day,
            True,
            4096,
            days[:index],
            ranks[:index],
            False,
        )
        features = np.asarray(result["arm_features"], dtype=np.int64)
        families = np.asarray(result["arm_family"], dtype=np.int64)
        if rank >= features.shape[0]:
            raise RuntimeError(f"selected rank {rank} unavailable on day {day}")
        row = features[rank]
        output.append(
            {
                "day": day,
                "rank": rank,
                "family": FAMILY_NAMES[int(families[rank])],
                "target_delta": {
                    name: int(value)
                    for name, value in zip(PROJECT_NAMES, row[:8], strict=True)
                    if int(value) != 0
                },
                "hand_delta": int(row[8]),
                "quadrant_delta": int(row[9]),
                "effective_delay_days": int(row[10]),
                "schedule_profile": int(row[11]),
                "market_profile": int(row[12]),
                "recovery_profile": int(row[13]),
                "suffix_project": int(row[14]),
                "market_item": int(row[15]),
                "recovery_issue": int(row[16]),
                "estimated_value": float(row[17]) / 100.0,
                "estimated_cash_cost": float(row[18]) / 100.0,
                "estimated_daily_action_load": float(row[19]) / 100.0,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--mcts-receipt", required=True, type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    receipt = json.loads(args.mcts_receipt.read_text(encoding="utf-8"))
    winner = next(row for row in receipt["rows"] if row["win"])
    case = receipt["case"]
    cases = {
        row["route_id"]: row
        for row in json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    }
    full_case = cases[case["route_id"]]
    seat = int(full_case["candidate_seat"])
    seed = int(full_case["official_seed"])

    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    genome = load_genome(args.genomes, 0)
    opponent = bundle.index(full_case["family"])
    committed = bundle.adaptive_executor.candidate8_committed_sequence(
        genome,
        opponent,
        seed,
        winner["decision_days"],
        winner["selected_ranks"],
        seat,
        True,
        -1,
        0,
        True,
    )
    trace = list(committed["trace"])
    candidate_actions = [dict(joint[seat]) for joint in trace]
    opponent_actions = [dict(joint[1 - seat]) for joint in trace]

    payload = {
        "schema": "kaggriculture-oceanmix-mcts-strategy-similarity-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "rewards": [float(value) for value in committed["rewards"]],
        "selected_arm_details": selected_arm_details(
            bundle,
            genome,
            opponent,
            seed,
            seat,
            winner["decision_days"],
            winner["selected_ranks"],
        ),
        "day0": {
            "candidate": summarize(candidate_actions[:24]),
            "oceanmix": summarize(opponent_actions[:24]),
        },
        "full_game": {
            "candidate": summarize(candidate_actions),
            "oceanmix": summarize(opponent_actions),
        },
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["selected_arm_details"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["day0"], ensure_ascii=False, indent=2))
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
