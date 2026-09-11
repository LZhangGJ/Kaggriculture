#!/usr/bin/env python3
"""Generate controlled EcoBot-V6 search candidates around the Rank1 family.

Only economically interpretable thresholds and capacity envelopes are varied.
Rejected execution experiments stay fixed at the audited base values.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=23_100_001)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 16:
        raise ValueError("count must be at least 16")
    payload = json.loads(args.base.read_text(encoding="utf-8"))
    base = {str(k): float(v) for k, v in payload["base_values"].items()}
    rng = np.random.default_rng(args.seed)

    def uniform(lo: float, hi: float) -> float:
        return float(rng.uniform(lo, hi))

    def integer(lo: int, hi: int) -> int:
        return int(rng.integers(lo, hi + 1))

    fixed = {
        # Accepted generic execution capabilities.
        "future_structure_reservation_fraction": 0.75,
        "routine_cell_exclusivity": 1.0,
        "terminal_animal_economics": 1.0,
        "hard_latest_start_reservation": 0.25,
        # Rejected or not-yet-proven execution experiments remain disabled.
        "animal_flow_control": 0.0,
        "animal_flow_pressure_gate": 0.0,
        "animal_flow_executable_gate": 0.0,
        "crop_lane_layout_mode": 0.0,
        "region_ownership_scale": 0.0,
        "workload_region_mode": 0.0,
        "region_assignment_mode": 0.0,
        "prerequisite_chain_strength": 0.0,
        "local_candidate_order": 0.0,
        "weed_capacity_priority": 0.0,
        "hard_travel_slack_scale": 0.0,
        # The four semantic priors already encode the branch portfolio.  These
        # legacy release knobs would otherwise apply the branch twice.
        "yarn_animal_flex": 0.0,
        "pet_crop_flex": 0.0,
    }

    genomes: list[dict[str, object]] = [
        {"label": "all151_base", "values": fixed},
        {"label": "all151_no_relay", "values": {**fixed, "wheat_relay_capacity_fraction": 0.0}},
        {"label": "all151_never_sheep", "values": {**fixed, "yarn_opponent_sheep_gate": 0.0}},
        {"label": "all151_sheep_gate6", "values": {**fixed, "yarn_opponent_sheep_gate": 6.0}},
        {"label": "all151_sheep_gate12", "values": {**fixed, "yarn_opponent_sheep_gate": 12.0}},
        {"label": "all151_sheep_uncrowded", "values": {**fixed, "yarn_opponent_sheep_gate": 100.0}},
        {"label": "all151_relay01", "values": {**fixed, "wheat_relay_capacity_fraction": 0.1}},
        {"label": "all151_relay03", "values": {**fixed, "wheat_relay_capacity_fraction": 0.3}},
    ]

    relay_choices = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4])
    replan_choices = np.asarray([8, 12, 16, 24, 32, 48])
    hard_choices = np.asarray([0.0, 0.25, 0.5])
    while len(genomes) < args.count:
        max_cows = integer(10, 17)
        max_sheep = integer(12, 22)
        values: dict[str, float | int] = {
            **fixed,
            "cash_reserve": uniform(100, 650),
            "action_cost": uniform(0, 8),
            "move_cost": uniform(2, 12),
            "risk_multiplier": uniform(0.9, 2.3),
            "market_impact_weight": uniform(0.45, 1.65),
            "opponent_supply_weight": uniform(0.15, 1.6),
            "demand_drift_weight": uniform(0.15, 1.8),
            "sell_drop_limit": uniform(0.08, 0.55),
            "price_replan_fraction": uniform(0.04, 0.45),
            "task_stickiness": uniform(30, 150),
            "deadline_weight": uniform(80, 260),
            "fertilizer_value_fraction": uniform(0.6, 1.25),
            "max_hands": integer(12, 18),
            "max_total_animals": integer(max(max_cows, max_sheep), 28),
            "max_cows": max_cows,
            "max_sheep": max_sheep,
            "max_wheat": integer(45, 75),
            "max_carrot": integer(30, 70),
            "max_tomato": integer(12, 35),
            "max_strawberry": integer(38, 70),
            "max_melon": integer(10, 25),
            "min_wheat_buffer": integer(1, 7),
            "replan_interval_steps": int(rng.choice(replan_choices)),
            "routine_priority_scale": uniform(0.3, 4.0),
            "task_value_scale": uniform(0.1, 1.5),
            "plant_priority": uniform(470, 720),
            "wheat_relay_capacity_fraction": float(rng.choice(relay_choices)),
            "wheat_relay_opponent_risk": uniform(0, 2.5),
            "yarn_wool_price_threshold": uniform(175, 285),
            "yarn_opponent_sheep_gate": uniform(0, 22),
            "pet_carrot_price_threshold": uniform(30, 75),
            "hard_latest_start_reservation": float(rng.choice(hard_choices)),
        }
        genomes.append({
            "label": f"all151_search_{len(genomes):04d}",
            "values": values,
        })

    output = {
        "schema": "kaggriculture.native-adaptive-genomes.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "search_semantics": (
            "one identity-blind EcoBot V6 planner; all151 Rank1 branch family "
            "is a semantic prior and live project values remain authoritative"
        ),
        "base_values": base,
        "fixed_safety_values": fixed,
        "seed": args.seed,
        "genomes": genomes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "count": len(genomes),
        "seed": args.seed,
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
