#!/usr/bin/env python3
"""Reconstruct realized public-product cash flow from a native trace audit.

The simulator records actual fill counts but intentionally keeps its hot loop
small.  This offline tool combines those fills with the official 1.32.7 quote
curve and order-slot interleaving to recover realized buy cost and sell revenue
per item.  It never feeds replay actions back into the planner.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
MARKET = (
    (25, 10000, 400, "SQRT", "LOG", 0.8, 0.2),
    (35, 10000, 450, "HINGE", "SQRT", 1.0, 0.7),
    (60, 10000, 200, "HINGE", "SQRT", 0.4, 0.6),
    (120, 10000, 100, "SQRT", "LINEAR", 0.7, 1.6),
    (250, 10000, 300, "LOG", "SQ", 0.2, 3.6),
    (50, 10000, 332, "HINGE", "LOG", 0.4, 0.2),
    (160, 10000, 122, "SQRT", "LINEAR", 0.6, 1.6),
    (200, 10000, 105, "LOG", "SQ", 0.2, 3.2),
    (100, 10000, 200, "LINEAR", "LINEAR", 0.4, 0.4),
)


def _shape(name: str, x: float, threshold: float) -> float:
    x = max(0.0, x)
    if name == "LINEAR":
        return x
    if name == "SQ":
        return x * x
    if name == "SQRT":
        return math.sqrt(x)
    if name == "LOG":
        return math.log1p(x)
    if name == "HINGE":
        unit = x / threshold
        return unit + 8.0 * max(0.0, unit - 1.0) ** 2
    raise ValueError(name)


def market_price(item: str, inventory: int) -> int:
    index = PRODUCTS.index(item)
    base, initial, threshold, below, above, below_scale, above_scale = MARKET[index]
    if inventory < initial:
        amplitude = below_scale * base / _shape(below, threshold, threshold)
        price = base + amplitude * _shape(below, initial - inventory, threshold)
    else:
        amplitude = above_scale * base / _shape(above, threshold, threshold)
        price = base - amplitude * _shape(above, inventory - initial, threshold)
    # All official prices in this environment are integral; the source uses
    # nearbyint under the default round-to-nearest-even mode.
    return max(1, int(round(price)))


def action_parts(action: list[Any]) -> tuple[str, str, int]:
    op = str(action[0]) if action else "PASS"
    item = str(action[1]) if len(action) >= 2 else ""
    quantity = int(action[2]) if len(action) >= 3 else 1
    return op, item, quantity


def add_metric(store: dict[str, dict[str, float]], item: str, key: str, value: float) -> None:
    store.setdefault(item, defaultdict(float))[key] += value


def analyze(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate_seat = int(payload.get("candidate_seat", payload.get("seat", 0)))
    totals = [dict(), dict()]
    daily = [[dict() for _ in range(30)] for _ in range(2)]

    for day, sequences in enumerate(payload["daily_market_sequences"]):
        for sequence in sequences:
            candidate_orders = sequence.get("orders", [])
            opponent_orders = sequence.get("opponent_orders", [])
            candidate_fills = sequence.get("fills", [])
            opponent_fills = sequence.get("opponent_fills", [])
            if candidate_seat == 0:
                orders = (candidate_orders, opponent_orders)
                fills = (candidate_fills, opponent_fills)
            else:
                orders = (opponent_orders, candidate_orders)
                fills = (opponent_fills, candidate_fills)
            inventory = {
                str(item): int(value)
                for item, value in sequence["market_inventory_before"].items()
            }
            max_orders = min(10, max(len(orders[0]), len(orders[1])))
            for order_index in range(max_orders):
                active = [False, False]
                remaining = [0, 0]
                parsed: list[tuple[str, str, int]] = [("PASS", "", 0)] * 2
                for player in (0, 1):
                    if order_index >= len(orders[player]):
                        continue
                    parsed[player] = action_parts(orders[player][order_index])
                    fill = int(fills[player][order_index]) if order_index < len(fills[player]) else 0
                    op, item, _ = parsed[player]
                    if fill > 0 and op in ("SELL", "BUY_PRODUCT") and item in PRODUCTS:
                        active[player] = True
                        remaining[player] = fill
                while active[0] or active[1]:
                    quotes: list[int | None] = [None, None]
                    for player in (0, 1):
                        if not active[player] or remaining[player] <= 0:
                            continue
                        op, item, _ = parsed[player]
                        quote_inventory = inventory[item] if op == "SELL" else inventory[item] - 1
                        quotes[player] = market_price(item, quote_inventory)
                    if quotes[0] is None and quotes[1] is None:
                        break
                    for player in (0, 1):
                        if quotes[player] is None:
                            continue
                        op, item, _ = parsed[player]
                        price = int(quotes[player])
                        if op == "SELL":
                            add_metric(totals[player], item, "sell_units", 1)
                            add_metric(totals[player], item, "sell_revenue", price)
                            add_metric(daily[player][day], item, "sell_units", 1)
                            add_metric(daily[player][day], item, "sell_revenue", price)
                            if price > 1:
                                inventory[item] += 1
                        else:
                            add_metric(totals[player], item, "buy_units", 1)
                            add_metric(totals[player], item, "buy_cost", price)
                            add_metric(daily[player][day], item, "buy_units", 1)
                            add_metric(daily[player][day], item, "buy_cost", price)
                            inventory[item] -= 1
                        remaining[player] -= 1
                        if remaining[player] <= 0:
                            active[player] = False

    def render(store: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for item, values in store.items():
            row = {key: float(value) for key, value in values.items()}
            row["net_cash"] = row.get("sell_revenue", 0.0) - row.get("buy_cost", 0.0)
            sell_units = row.get("sell_units", 0.0)
            buy_units = row.get("buy_units", 0.0)
            row["mean_sell_price"] = row.get("sell_revenue", 0.0) / sell_units if sell_units else 0.0
            row["mean_buy_price"] = row.get("buy_cost", 0.0) / buy_units if buy_units else 0.0
            out[item] = row
        return out

    own = render(totals[candidate_seat])
    opponent = render(totals[1 - candidate_seat])
    return {
        "schema": "kaggriculture.realized-market-cashflow-audit.v1",
        "source": str(path),
        "candidate_seat": candidate_seat,
        "rewards": payload.get("rewards"),
        "candidate": own,
        "opponent": opponent,
        "candidate_daily": [render(daily[candidate_seat][day]) for day in range(30)],
        "opponent_daily": [render(daily[1 - candidate_seat][day]) for day in range(30)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"traces": [analyze(path) for path in args.trace]}
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
