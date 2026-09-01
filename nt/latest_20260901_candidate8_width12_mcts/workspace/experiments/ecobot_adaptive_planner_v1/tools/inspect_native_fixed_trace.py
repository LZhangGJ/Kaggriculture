#!/usr/bin/env python3
"""Inspect one frozen native route-vs-route game day by day.

This is an offline oracle diagnostic.  It intentionally executes a frozen raw
route tape so that its realized production and transaction flow can be compared
with the adaptive semantic planner on the same seed.  Raw actions produced by
this tool must not be imported by the final planner or submission.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from fast_kaggriculture import Config, FastEnv
from meta_agent.src.native_teammate_executor import NativeTeammateBundle

from inspect_native_adaptive_trace import (
    contextual_operation,
    new_route_efficiency,
    operation,
    record_route_efficiency,
    render_route_efficiency,
    snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--days",
        default=",".join(str(day) for day in range(1, 31)),
        help="One-based days to snapshot at hour 23.",
    )
    parser.add_argument(
        "--timeline-days",
        default="",
        help="One-based days for per-step unit position/inventory diagnostics.",
    )
    args = parser.parse_args()

    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    if args.seat == 0:
        result = bundle.play(
            args.left, args.right, args.seed, seat=0, capture_trace=True
        )
    else:
        result = bundle.play(
            args.right, args.left, args.seed, seat=1, capture_trace=True
        )
    trace = result["trace"]
    env = FastEnv(Config(), args.seed)
    chosen_days = {int(value) for value in args.days.split(",") if value.strip()}
    timeline_days = {
        int(value) for value in args.timeline_days.split(",") if value.strip()
    }

    daily_unit_attempts = [collections.Counter() for _ in range(30)]
    daily_unit_item_attempts = [collections.Counter() for _ in range(30)]
    daily_market_item_quantities = [collections.Counter() for _ in range(30)]
    daily_market_item_fills = [collections.Counter() for _ in range(30)]
    daily_market_cash_delta = [0.0 for _ in range(30)]
    daily_market_sequences: list[list[dict[str, object]]] = [[] for _ in range(30)]
    opponent_daily_market_item_quantities = [collections.Counter() for _ in range(30)]
    opponent_daily_market_item_fills = [collections.Counter() for _ in range(30)]
    opponent_daily_market_cash_delta = [0.0 for _ in range(30)]
    daily_route_efficiency = new_route_efficiency()
    route_history: dict[int, tuple[int, str]] = {}
    snapshots: list[dict[str, object]] = []
    opponent_snapshots: list[dict[str, object]] = []
    timeline: list[dict[str, object]] = []

    for step, joint_action in enumerate(trace):
        day = min(29, step // 24)
        own_action = joint_action[args.seat]
        opponent_action = joint_action[1 - args.seat]
        before = env.observation(args.seat)
        before_farm = before["farms"][args.seat]
        before_positions = [before_farm["farmer"], *before_farm["hands"]]
        if int(before["day"]) + 1 in timeline_days:
            before_private = before["private"]
            own_units = [own_action["farmer"], *own_action["hands"]]
            timeline.append({
                "step": step,
                "day": int(before["day"]),
                "hour": int(before["hour"]),
                "money": float(before_farm["money"]),
                "shed": dict(before_private["shed"]),
                "market_inventory": dict(before["market"]["inventory"]),
                "market_prices": dict(before["market"]["prices"]),
                "market": own_action["market"],
                "units": [
                    {
                        "unit": unit,
                        "position": list(position),
                        "inventory": dict(before_private["inventories"][unit]),
                        "action": own_units[unit],
                    }
                    for unit, position in enumerate(before_positions)
                ],
            })
        for unit, action in enumerate([own_action["farmer"], *own_action["hands"]]):
            daily_unit_attempts[day][operation(action)] += 1
            tile = None
            if unit < len(before_positions):
                x, y = before_positions[unit]
                tile = before_farm["tiles"][y][x]
            daily_unit_item_attempts[day][contextual_operation(action, tile)] += 1
            if unit < len(before_positions):
                record_route_efficiency(
                    daily_route_efficiency,
                    route_history,
                    day,
                    step,
                    unit,
                    before_positions[unit],
                    action,
                )
        if own_action["market"] or opponent_action["market"]:
            daily_market_sequences[day].append(
                {
                    "step": step,
                    "hour": step % 24,
                    "orders": own_action["market"],
                    "opponent_orders": opponent_action["market"],
                    "market_inventory_before": dict(before["market"]["inventory"]),
                    "market_prices_before": dict(before["market"]["prices"]),
                }
            )
        for action in own_action["market"]:
            op = operation(action)
            quantity = (
                action[-1]
                if isinstance(action, (list, tuple)) and len(action) >= 3
                else 1
            )
            if isinstance(quantity, (int, float)):
                item = str(action[1]) if len(action) >= 2 else ""
                daily_market_item_quantities[day][f"{op}:{item}"] += int(quantity)
        for action in opponent_action["market"]:
            op = operation(action)
            quantity = (
                action[-1]
                if isinstance(action, (list, tuple)) and len(action) >= 3
                else 1
            )
            if isinstance(quantity, (int, float)):
                item = str(action[1]) if len(action) >= 2 else ""
                opponent_daily_market_item_quantities[day][f"{op}:{item}"] += int(quantity)

        money_before = float(before_farm["money"])
        opponent_money_before = float(before["farms"][1 - args.seat]["money"])
        env.step(joint_action)
        own_fills = list(env.last_market_fills[args.seat])
        own_shortfalls = list(env.last_market_cash_shortfalls[args.seat])
        opponent_fills = list(env.last_market_fills[1 - args.seat])
        observation = env.observation(args.seat)
        money_after = float(observation["farms"][args.seat]["money"])
        opponent_money_after = float(observation["farms"][1 - args.seat]["money"])
        daily_market_cash_delta[day] += money_after - money_before
        opponent_daily_market_cash_delta[day] += opponent_money_after - opponent_money_before
        for order_index, action in enumerate(own_action["market"]):
            op = operation(action)
            item = str(action[1]) if len(action) >= 2 else ""
            fill = int(own_fills[order_index]) if order_index < len(own_fills) else 0
            daily_market_item_fills[day][f"{op}:{item}"] += fill
        for order_index, action in enumerate(opponent_action["market"]):
            op = operation(action)
            item = str(action[1]) if len(action) >= 2 else ""
            fill = int(opponent_fills[order_index]) if order_index < len(opponent_fills) else 0
            opponent_daily_market_item_fills[day][f"{op}:{item}"] += fill
        if own_action["market"] or opponent_action["market"]:
            daily_market_sequences[day][-1].update({
                "fills": own_fills,
                "opponent_fills": opponent_fills,
                "cash_shortfalls": own_shortfalls,
                "money_before": money_before,
                "money_after": money_after,
                "cash_delta": money_after - money_before,
                "market_inventory_after": dict(observation["market"]["inventory"]),
                "market_prices_after": dict(observation["market"]["prices"]),
            })
        if int(observation["hour"]) == 23 and int(observation["day"]) + 1 in chosen_days:
            snapshots.append(snapshot(env, args.seat))
            opponent_snapshots.append(snapshot(env, 1 - args.seat))

    payload = {
        "schema": "kaggriculture.native-fixed-trace-audit.v1",
        "warning": "offline raw-route oracle only; not a planner or submission input",
        "left": args.left,
        "right": args.right,
        "candidate_seat": args.seat,
        "seed": args.seed,
        "rewards": list(result["rewards"]),
        "trace_steps": len(trace),
        "daily_unit_attempts": [dict(row) for row in daily_unit_attempts],
        "daily_unit_item_attempts": [dict(row) for row in daily_unit_item_attempts],
        "daily_route_efficiency": render_route_efficiency(daily_route_efficiency),
        "daily_market_item_quantities": [
            dict(row) for row in daily_market_item_quantities
        ],
        "daily_market_item_fills": [
            dict(row) for row in daily_market_item_fills
        ],
        "daily_market_cash_delta": daily_market_cash_delta,
        "daily_market_sequences": daily_market_sequences,
        "opponent_daily_market_item_quantities": [
            dict(row) for row in opponent_daily_market_item_quantities
        ],
        "opponent_daily_market_item_fills": [
            dict(row) for row in opponent_daily_market_item_fills
        ],
        "opponent_daily_market_cash_delta": opponent_daily_market_cash_delta,
        "snapshots": snapshots,
        "opponent_snapshots": opponent_snapshots,
        "timeline": timeline,
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
