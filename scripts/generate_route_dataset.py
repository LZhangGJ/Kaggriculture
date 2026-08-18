"""Generate paired counterfactual route outcomes from arbitrary local agents."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from kaggriculture_lab.route_learning import (
    read_counterfactual_records,
    validate_counterfactual_groups,
    write_counterfactual_records,
)
from kaggriculture_lab.route_planner import PlanSpec
from kaggriculture_lab.route_rollout import (
    RolloutScenario,
    RouteCandidate,
    collect_counterfactual_grid,
)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = {"prefix_agent", "opponents", "routes"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Missing configuration fields: {missing}")
    if not payload["opponents"]:
        raise ValueError("At least one opponent is required")
    if not payload["routes"]:
        raise ValueError("At least one route is required")
    return payload


def _candidates(payload: dict[str, Any]) -> list[RouteCandidate]:
    result = []
    for route in payload["routes"]:
        name = str(route["name"])
        plan = PlanSpec.from_mapping(name, route.get("plan"))
        result.append(RouteCandidate(name=name, agent=str(route["agent"]), plan=plan))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one shared prefix and branch several route agents at each decision state."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--num-seeds", type=int, default=64)
    parser.add_argument("--decision-step", type=int, action="append", dest="decision_steps")
    parser.add_argument("--one-seat", action="store_true", help="Evaluate only candidate seat 0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    payload = _load_config(args.config)
    candidates = _candidates(payload)
    decision_steps = args.decision_steps or list(payload.get("decision_steps", [168]))
    seats = (0,) if args.one_seat else (0, 1)
    seeds = range(args.seed_start, args.seed_start + args.num_seeds)
    scenarios = [
        RolloutScenario(
            seed=seed,
            seat=seat,
            decision_step=int(decision_step),
            prefix_agent=str(payload["prefix_agent"]),
            opponent=str(opponent),
        )
        for seed in seeds
        for seat in seats
        for opponent in payload["opponents"]
        for decision_step in decision_steps
    ]

    existing = read_counterfactual_records(args.output) if args.resume and args.output.exists() else []
    existing_keys = {(record.scenario_id, record.route_name) for record in existing}
    records = collect_counterfactual_grid(
        scenarios,
        candidates,
        configuration=payload.get("environment"),
        workers=args.workers,
    )
    new_records = [
        record
        for record in records
        if (record.scenario_id, record.route_name) not in existing_keys
    ]
    write_counterfactual_records(args.output, new_records, append=bool(existing))
    combined = [*existing, *new_records]
    groups = validate_counterfactual_groups(combined)

    status = Counter(record.candidate_status for record in combined)
    route_wins: dict[str, list[float]] = {}
    for record in combined:
        route_wins.setdefault(record.route_name, []).append(record.win)
    summary = {
        "records": len(combined),
        "new_records": len(new_records),
        "scenarios": len(groups),
        "routes": [candidate.name for candidate in candidates],
        "statuses": dict(status),
        "route_mean_win": {
            route: sum(values) / max(1, len(values))
            for route, values in route_wins.items()
        },
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"dataset={args.output}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
