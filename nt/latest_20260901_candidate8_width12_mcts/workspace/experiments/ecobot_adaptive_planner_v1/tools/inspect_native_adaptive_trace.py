#!/usr/bin/env python3
"""Inspect one native adaptive-planner game without dumping the full trace."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from fast_kaggriculture import (
    Config,
    FastEnv,
    adaptive_default_genome,
    adaptive_genome_names,
)
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def operation(action: object) -> str:
    if isinstance(action, (list, tuple)) and action:
        return str(action[0])
    return "PASS"


def tile_label(tile: object) -> str:
    if tile is None:
        return "EMPTY"
    if isinstance(tile, str):
        return tile
    if isinstance(tile, dict):
        for key in ("crop", "animal", "building", "type", "content"):
            if key in tile:
                value = tile[key]
                if isinstance(value, dict):
                    return f"{key}:{value.get('type', value.get('name', 'DICT'))}"
                return f"{key}:{value}"
        return "DICT:" + ",".join(sorted(str(key) for key in tile))
    if isinstance(tile, (list, tuple)) and tile:
        return str(tile[0])
    return type(tile).__name__


def contextual_operation(action: object, tile: object) -> str:
    """Label a unit action with the public crop/animal it operates on."""
    op = operation(action)
    if op in {"PLANT", "PICKUP", "PLACE"} and isinstance(action, (list, tuple)) and len(action) >= 2:
        return f"{op}:{action[1]}"
    if isinstance(tile, dict):
        if tile.get("crop") is not None and op in {"WATER", "HARVEST", "FERTILIZE"}:
            return f"{op}:{tile['crop']}"
        if tile.get("animal") is not None and op in {
            "FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER"
        }:
            return f"{op}:{tile['animal']}"
    return op


MOVE_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}
INVERSE_MOVE = {
    "NORTH": "SOUTH",
    "SOUTH": "NORTH",
    "EAST": "WEST",
    "WEST": "EAST",
}


def new_route_efficiency() -> list[dict[str, object]]:
    """Create compact per-day route diagnostics for one 30-day episode."""
    return [
        {
            "move_actions": 0,
            "work_actions": 0,
            "pass_actions": 0,
            "immediate_reversals": 0,
            "direction_changes": 0,
            "work_cells": set(),
        }
        for _ in range(30)
    ]


def record_route_efficiency(
    rows: list[dict[str, object]],
    history: dict[int, tuple[int, str]],
    day: int,
    step: int,
    unit: int,
    position: object,
    action: object,
) -> None:
    """Record movement churn without assuming any route or target semantics."""
    op = operation(action)
    row = rows[day]
    if op in MOVE_OPS:
        row["move_actions"] += 1
        previous = history.get(unit)
        if previous is not None and previous[0] == step - 1:
            previous_op = previous[1]
            if previous_op in MOVE_OPS and previous_op != op:
                row["direction_changes"] += 1
            if INVERSE_MOVE.get(previous_op) == op:
                row["immediate_reversals"] += 1
    elif op == "PASS":
        row["pass_actions"] += 1
    else:
        row["work_actions"] += 1
        if isinstance(position, (list, tuple)) and len(position) == 2:
            row["work_cells"].add((int(position[0]), int(position[1])))
    history[unit] = (step, op)


def render_route_efficiency(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rendered: list[dict[str, object]] = []
    for row in rows:
        moves = int(row["move_actions"])
        work = int(row["work_actions"])
        rendered.append(
            {
                "move_actions": moves,
                "work_actions": work,
                "pass_actions": int(row["pass_actions"]),
                "moves_per_work": round(moves / max(1, work), 6),
                "immediate_reversals": int(row["immediate_reversals"]),
                "direction_changes": int(row["direction_changes"]),
                "unique_work_cells": len(row["work_cells"]),
            }
        )
    return rendered


def snapshot(env: FastEnv, player: int) -> dict[str, object]:
    observation = env.observation(player)
    step = int(observation.get("step", env.step_count))
    farm = observation["farms"][player]
    private = observation["private"]
    tile_counts: collections.Counter[str] = collections.Counter()
    tile_samples: dict[str, object] = {}
    tile_positions: dict[str, list[list[int]]] = collections.defaultdict(list)
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            label = tile_label(tile)
            if label not in ("EMPTY", "LOCKED"):
                tile_counts[label] += 1
                tile_positions[label].append([x, y])
                if label not in tile_samples:
                    tile_samples[label] = tile
    inventories = private["inventories"]
    carried: collections.Counter[str] = collections.Counter()
    for inventory in inventories:
        for item, quantity in inventory.items():
            carried[str(item)] += int(quantity)
    return {
        "step": step,
        "day": int(observation.get("day", step // 24)),
        "hour": int(observation.get("hour", step % 24)),
        "money": float(farm["money"]),
        "hands": len(farm["hands"]),
        "hires_today": int(farm["hires_today"]),
        "unlocked_quadrants": list(farm["unlocked_quadrants"]),
        "shed": {key: int(value) for key, value in private["shed"].items() if value},
        "seeds": {key: int(value) for key, value in private["seeds"].items() if value},
        "carried": dict(carried),
        "market_inventory": dict(observation["market"]["inventory"]),
        "market_prices": dict(observation["market"]["prices"]),
        "tiles": dict(tile_counts),
        "tile_positions": dict(tile_positions),
        "tile_samples": tile_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--seed", type=int, default=1_943_001)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--opponent",
        default="PASSIVE",
        help="Route-library family used as the frozen opponent, or PASSIVE.",
    )
    parser.add_argument("--genome-receipt", type=Path)
    parser.add_argument("--genomes", type=Path)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--validation-rank", type=int, default=0)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override one genome field as NAME=VALUE; may be repeated.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--days",
        default="1,2,4,6,9,11,13,16,21,26,30",
        help="One-based days to snapshot at hour 23.",
    )
    parser.add_argument(
        "--timeline-days",
        default="",
        help="One-based days for per-step unit position/inventory diagnostics.",
    )
    args = parser.parse_args()

    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, args.backbone
    )
    genome = list(adaptive_default_genome())
    if args.genome_receipt is not None:
        payload = json.loads(args.genome_receipt.read_text(encoding="utf-8"))
        row = payload["validation"]["ranking"][args.validation_rank]
        genome = [float(row["values"][name]) for name in adaptive_genome_names()]
    elif args.genomes is not None:
        payload = json.loads(args.genomes.read_text(encoding="utf-8"))
        values = payload["genomes"][args.genome_index]["values"]
        defaults = dict(zip(adaptive_genome_names(), adaptive_default_genome(), strict=True))
        defaults.update({
            str(name): float(value)
            for name, value in payload.get("base_values", {}).items()
        })
        genome = [float(values.get(name, defaults[name])) for name in adaptive_genome_names()]
    names = list(adaptive_genome_names())
    for raw_override in args.override:
        name, raw_value = raw_override.split("=", 1)
        if name not in names:
            raise ValueError(f"unknown genome field: {name}")
        genome[names.index(name)] = float(raw_value)
    opponent_index = -1 if args.opponent.upper() == "PASSIVE" else bundle.index(args.opponent)
    result = bundle.adaptive_executor.play(
        genome, opponent_index, args.seed, args.seat, True
    )
    trace = result["trace"]
    env = FastEnv(Config(), args.seed)
    unit_attempts: collections.Counter[str] = collections.Counter()
    daily_unit_attempts = [collections.Counter() for _ in range(30)]
    daily_unit_item_attempts = [collections.Counter() for _ in range(30)]
    market_attempts: collections.Counter[str] = collections.Counter()
    daily_market_attempts = [collections.Counter() for _ in range(30)]
    market_quantities: collections.Counter[str] = collections.Counter()
    daily_market_item_quantities = [collections.Counter() for _ in range(30)]
    daily_market_item_fills = [collections.Counter() for _ in range(30)]
    daily_market_cash_delta = [0.0 for _ in range(30)]
    daily_market_sequences: list[list[dict[str, object]]] = [[] for _ in range(30)]
    opponent_unit_attempts: collections.Counter[str] = collections.Counter()
    opponent_market_attempts: collections.Counter[str] = collections.Counter()
    opponent_daily_market_item_quantities = [collections.Counter() for _ in range(30)]
    opponent_daily_market_item_fills = [collections.Counter() for _ in range(30)]
    opponent_daily_market_cash_delta = [0.0 for _ in range(30)]
    daily_route_efficiency = new_route_efficiency()
    route_history: dict[int, tuple[int, str]] = {}
    opponent_daily_route_efficiency = new_route_efficiency()
    opponent_route_history: dict[int, tuple[int, str]] = {}
    chosen_days = {int(value) for value in args.days.split(",") if value.strip()}
    timeline_days = {
        int(value) for value in args.timeline_days.split(",") if value.strip()
    }
    snapshots: list[dict[str, object]] = []
    opponent_snapshots: list[dict[str, object]] = []
    timeline: list[dict[str, object]] = []
    hard_events: list[dict[str, object]] = []
    end_overflow_events: list[dict[str, object]] = []
    samples: dict[str, object] = {}
    for step, joint_action in enumerate(trace):
        trace_day = min(29, step // 24)
        candidate_action = joint_action[args.seat]
        opponent_action = joint_action[1 - args.seat]
        before = env.observation(args.seat)
        before_farm = before["farms"][args.seat]
        before_positions = [before_farm["farmer"], *before_farm["hands"]]
        opponent_farm = before["farms"][1 - args.seat]
        opponent_positions = [
            opponent_farm["farmer"], *opponent_farm["hands"]
        ]
        candidate_unit_actions = [candidate_action["farmer"], *candidate_action["hands"]]
        for unit, action in enumerate(candidate_unit_actions):
            op = operation(action)
            unit_attempts[op] += 1
            daily_unit_attempts[trace_day][op] += 1
            tile = None
            if unit < len(before_positions):
                x, y = before_positions[unit]
                tile = before_farm["tiles"][y][x]
            daily_unit_item_attempts[trace_day][contextual_operation(action, tile)] += 1
            if op not in samples and op != "PASS":
                samples[op] = action
            if unit < len(before_positions):
                record_route_efficiency(
                    daily_route_efficiency,
                    route_history,
                    trace_day,
                    step,
                    unit,
                    before_positions[unit],
                    action,
                )
        for unit, action in enumerate(
            [opponent_action["farmer"], *opponent_action["hands"]]
        ):
            if unit < len(opponent_positions):
                record_route_efficiency(
                    opponent_daily_route_efficiency,
                    opponent_route_history,
                    trace_day,
                    step,
                    unit,
                    opponent_positions[unit],
                    action,
                )
        for action in candidate_action["market"]:
            op = operation(action)
            market_attempts[op] += 1
            daily_market_attempts[trace_day][op] += 1
            quantity = action[-1] if isinstance(action, (list, tuple)) and len(action) >= 3 else 1
            if isinstance(quantity, (int, float)):
                market_quantities[op] += int(quantity)
                item = str(action[1]) if len(action) >= 2 else ""
                daily_market_item_quantities[trace_day][f"{op}:{item}"] += int(quantity)
            if op not in samples:
                samples[op] = action
        if candidate_action["market"] or opponent_action["market"]:
            daily_market_sequences[trace_day].append({
                "step": step,
                "hour": step % 24,
                "orders": candidate_action["market"],
                "opponent_orders": opponent_action["market"],
                "market_inventory_before": dict(before["market"]["inventory"]),
                "market_prices_before": dict(before["market"]["prices"]),
            })
        for action in [opponent_action["farmer"], *opponent_action["hands"]]:
            opponent_unit_attempts[operation(action)] += 1
        for action in opponent_action["market"]:
            op = operation(action)
            opponent_market_attempts[op] += 1
            quantity = action[-1] if isinstance(action, (list, tuple)) and len(action) >= 3 else 1
            if isinstance(quantity, (int, float)):
                item = str(action[1]) if len(action) >= 2 else ""
                opponent_daily_market_item_quantities[trace_day][f"{op}:{item}"] += int(quantity)
        if int(before["day"]) + 1 in timeline_days:
            before_farm = before["farms"][args.seat]
            before_private = before["private"]
            positions = [before_farm["farmer"], *before_farm["hands"]]
            ops = [
                operation(candidate_action["farmer"]),
                *[operation(value) for value in candidate_action["hands"]],
            ]
            animals = []
            crops = []
            for y, row in enumerate(before_farm["tiles"]):
                for x, tile in enumerate(row):
                    if isinstance(tile, dict) and tile.get("animal"):
                        animals.append({
                            "position": [x, y],
                            "animal": tile.get("animal"),
                            "unfed": int(tile.get("consecutive_unfed", 0)),
                            "fed_today": bool(tile.get("fed_today")),
                        })
                    if isinstance(tile, dict) and tile.get("crop"):
                        crops.append({
                            "position": [x, y],
                            "crop": tile.get("crop"),
                            "unwatered": int(tile.get("consecutive_unwatered", 0)),
                            "watered_today": bool(tile.get("watered_today")),
                            "yield_units": int(tile.get("yield_units", 0)),
                        })
            timeline.append({
                "step": step,
                "day": int(before["day"]),
                "hour": int(before["hour"]),
                "money": float(before_farm["money"]),
                "shed": dict(before_private["shed"]),
                "market_inventory": dict(before["market"]["inventory"]),
                "market_prices": dict(before["market"]["prices"]),
                "market": candidate_action["market"],
                "animals": animals,
                "crops": crops,
                "units": [
                    {
                        "unit": unit,
                        "position": list(position),
                        "wheat": int(before_private["inventories"][unit].get("WHEAT", 0)),
                        "op": ops[unit],
                    }
                    for unit, position in enumerate(positions)
                ],
            })
        risk_crops: list[tuple[int, int]] = []
        risk_animals: list[tuple[int, int]] = []
        if int(before["hour"]) == 23:
            for y, row in enumerate(before["farms"][args.seat]["tiles"]):
                for x, tile in enumerate(row):
                    if not isinstance(tile, dict):
                        continue
                    if tile.get("kind") == "PLANT" and int(tile.get("consecutive_unwatered", 0)) >= 1 and not tile.get("watered_today"):
                        risk_crops.append((x, y))
                    if tile.get("animal") and int(tile.get("consecutive_unfed", 0)) >= 1 and not tile.get("fed_today"):
                        risk_animals.append((x, y))
        money_before = float(before_farm["money"])
        opponent_money_before = float(before["farms"][1 - args.seat]["money"])
        env.step(joint_action)
        overflow = int(env.last_end_of_day_overflow[args.seat])
        if overflow > 0:
            before_private = before["private"]
            before_shed = sum(int(value) for value in before_private["shed"].values())
            before_carried = sum(
                int(value)
                for inventory in before_private["inventories"]
                for value in inventory.values()
            )
            end_overflow_events.append({
                "step": step,
                "day": trace_day,
                "hour": step % 24,
                "quantity": overflow,
                "shed_before": before_shed,
                "carried_before": before_carried,
                "market_orders": candidate_action["market"],
            })
        own_fills = list(env.last_market_fills[args.seat])
        own_shortfalls = list(env.last_market_cash_shortfalls[args.seat])
        opponent_fills = list(env.last_market_fills[1 - args.seat])
        observation = env.observation(args.seat)
        money_after = float(observation["farms"][args.seat]["money"])
        opponent_money_after = float(observation["farms"][1 - args.seat]["money"])
        daily_market_cash_delta[trace_day] += money_after - money_before
        opponent_daily_market_cash_delta[trace_day] += opponent_money_after - opponent_money_before
        for order_index, action in enumerate(candidate_action["market"]):
            op = operation(action)
            item = str(action[1]) if len(action) >= 2 else ""
            fill = int(own_fills[order_index]) if order_index < len(own_fills) else 0
            daily_market_item_fills[trace_day][f"{op}:{item}"] += fill
        for order_index, action in enumerate(opponent_action["market"]):
            op = operation(action)
            item = str(action[1]) if len(action) >= 2 else ""
            fill = int(opponent_fills[order_index]) if order_index < len(opponent_fills) else 0
            opponent_daily_market_item_fills[trace_day][f"{op}:{item}"] += fill
        if candidate_action["market"] or opponent_action["market"]:
            daily_market_sequences[trace_day][-1].update({
                "fills": own_fills,
                "opponent_fills": opponent_fills,
                "cash_shortfalls": own_shortfalls,
                "money_before": money_before,
                "money_after": money_after,
                "cash_delta": money_after - money_before,
                "market_inventory_after": dict(observation["market"]["inventory"]),
                "market_prices_after": dict(observation["market"]["prices"]),
            })
        if int(before["day"]) + 1 in timeline_days and candidate_action["market"]:
            timeline[-1].update({
                "market_fills": own_fills,
                "market_cash_shortfalls": own_shortfalls,
                "money_after": money_after,
                "market_cash_delta": money_after - money_before,
            })
        if risk_crops or risk_animals:
            def kind_at(x: int, y: int) -> str:
                tile = observation["farms"][args.seat]["tiles"][y][x]
                return str(tile.get("kind", "")) if isinstance(tile, dict) else str(tile)

            lost_crops = [(x, y) for x, y in risk_crops if kind_at(x, y) == "WEED"]
            lost_animals = [
                (x, y) for x, y in risk_animals if kind_at(x, y) in ("PASTURE", "COOP")
            ]
            if lost_crops or lost_animals:
                private = before["private"]
                before_positions = [
                    before["farms"][args.seat]["farmer"],
                    *before["farms"][args.seat]["hands"],
                ]
                before_ops = [
                    operation(candidate_action["farmer"]),
                    *[operation(value) for value in candidate_action["hands"]],
                ]
                hard_events.append({
                    "step": step,
                    "day": int(before["day"]),
                    "hour": int(before["hour"]),
                    "money": float(before["farms"][args.seat]["money"]),
                    "lost_crops": lost_crops,
                    "lost_animals": lost_animals,
                    "wheat_shed": int(private["shed"]["WHEAT"]),
                    "wheat_carried": int(sum(inv.get("WHEAT", 0) for inv in private["inventories"])),
                    "unit_positions": before_positions,
                    "unit_wheat": [int(inv.get("WHEAT", 0)) for inv in private["inventories"]],
                    "unit_ops": before_ops,
                    "lost_animal_states": [
                        {
                            "position": [x, y],
                            "before": before["farms"][args.seat]["tiles"][y][x],
                            "after": observation["farms"][args.seat]["tiles"][y][x],
                            "units_here": [
                                {
                                    "unit": unit,
                                    "op": before_ops[unit],
                                    "wheat": int(private["inventories"][unit].get("WHEAT", 0)),
                                }
                                for unit, position in enumerate(before_positions)
                                if list(position) == [x, y]
                            ],
                        }
                        for x, y in lost_animals
                    ],
                    "market": candidate_action["market"],
                })
        if int(observation["hour"]) == 23 and int(observation["day"]) + 1 in chosen_days:
            snapshots.append(snapshot(env, args.seat))
            opponent_snapshots.append(snapshot(env, 1 - args.seat))
    # The native controller can replan several times during one day.  Keeping
    # every intermediate plan makes the diagnostic JSON needlessly huge; the
    # final plan for each day is the one that governs the next actions and is
    # the useful portfolio audit record.
    plan_by_day = {}
    for plan in result.get("plans", []):
        plan_by_day[int(plan["day"])] = plan

    payload = {
        "seed": args.seed,
        "seat": args.seat,
        "rewards": list(result["rewards"]),
        "diagnostics": {
            key: result[key]
            for key in (
                "replans",
                "override_actions",
                "avoidable_crop_losses",
                "avoidable_animal_losses",
                "end_overflow",
                "final_operating_prior_index",
                "operating_prior_switches",
                "operating_prior_trace",
            )
            if key in result
        },
        "trace_steps": len(trace),
        "unit_attempts": dict(unit_attempts),
        "daily_unit_attempts": [dict(row) for row in daily_unit_attempts],
        "daily_unit_item_attempts": [dict(row) for row in daily_unit_item_attempts],
        "daily_route_efficiency": render_route_efficiency(daily_route_efficiency),
        "opponent_daily_route_efficiency": render_route_efficiency(
            opponent_daily_route_efficiency
        ),
        "market_attempts": dict(market_attempts),
        "daily_market_attempts": [dict(row) for row in daily_market_attempts],
        "market_quantities": dict(market_quantities),
        "daily_market_item_quantities": [dict(row) for row in daily_market_item_quantities],
        "daily_market_item_fills": [dict(row) for row in daily_market_item_fills],
        "daily_market_cash_delta": daily_market_cash_delta,
        "daily_market_sequences": daily_market_sequences,
        "opponent_unit_attempts": dict(opponent_unit_attempts),
        "opponent_market_attempts": dict(opponent_market_attempts),
        "opponent_daily_market_item_quantities": [
            dict(row) for row in opponent_daily_market_item_quantities
        ],
        "opponent_daily_market_item_fills": [
            dict(row) for row in opponent_daily_market_item_fills
        ],
        "opponent_daily_market_cash_delta": opponent_daily_market_cash_delta,
        "action_samples": samples,
        "plans": [plan_by_day[day] for day in sorted(plan_by_day)],
        "hard_events": hard_events,
        "end_overflow_events": end_overflow_events,
        "timeline": timeline,
        "snapshots": snapshots,
        "opponent_snapshots": opponent_snapshots,
        "terminal": snapshot(env, args.seat),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
