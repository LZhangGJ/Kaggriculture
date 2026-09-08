"""Complete final-turn frontier over every visible sellable product."""
from __future__ import annotations

import copy
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "agents"
if str(AGENTS) not in sys.path:
    sys.path.insert(0, str(AGENTS))

import e750a_place_funding_repair as PROJECTION
import late_bundle_diversifier as PARENT

ENGINE = PARENT.ENGINE
FINAL_EXECUTABLE_STEP = 718
PRODUCTS = tuple(ENGINE.PRODUCTS)
_LAST_DIAGNOSTIC: dict[int, dict[str, Any]] = {0: {}, 1: {}}


def _copy_action(action: Any) -> dict[str, Any]:
    return ENGINE._copy_action(action)


def _market_inventory(obs: Any, item: str) -> int:
    market = ENGINE._get(obs, "market", {}) or {}
    inventory = ENGINE._get(market, "inventory", {}) or {}
    return int(ENGINE._get(inventory, item, ENGINE.I0) or ENGINE.I0)


def _revenue(item: str, inventory: int, quantity: int) -> int:
    return sum(ENGINE._price(item, inventory + offset) for offset in range(quantity))


def _complete_terminal_frontier(
    obs: Any,
    seat: int,
    action: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sell all executable final inventory, optimizing scarce order slots."""
    changed = _copy_action(action)
    available, _pickup_reserve = PROJECTION._project_shed(obs, seat, changed)
    available = Counter(
        {
            str(item): max(0, int(quantity or 0))
            for item, quantity in available.items()
        }
    )

    # Re-cap inherited sells against the same engine-order shed projection.
    # Track one existing slot per item so its quantity can be expanded without
    # consuming a new market-order slot.
    capped: list[list[Any]] = []
    sold: Counter[str] = Counter()
    existing_slot: dict[str, int] = {}
    for raw in (changed.get("market") or [])[: ENGINE.MAX_MARKET_ORDERS]:
        order = list(raw)
        if len(order) >= 3 and order[0] == "SELL" and str(order[1]) in PRODUCTS:
            item = str(order[1])
            requested = max(0, int(order[2] or 0))
            quantity = min(requested, max(0, int(available[item])))
            if quantity <= 0:
                continue
            order[2] = quantity
            available[item] -= quantity
            sold[item] += quantity
            existing_slot[item] = len(capped)
        capped.append(order)
    changed["market"] = capped

    expanded: Counter[str] = Counter()
    for item, position in existing_slot.items():
        quantity = max(0, int(available[item]))
        if quantity <= 0 or position >= len(changed["market"]):
            continue
        changed["market"][position][2] += quantity
        available[item] = 0
        sold[item] += quantity
        expanded[item] += quantity

    candidates = []
    for item in PRODUCTS:
        quantity = max(0, int(available[item]))
        if quantity <= 0:
            continue
        inventory = _market_inventory(obs, item) + int(sold[item])
        value = _revenue(item, inventory, quantity)
        candidates.append((value, item, quantity))
    candidates.sort(reverse=True)

    room = max(0, ENGINE.MAX_MARKET_ORDERS - len(changed["market"]))
    appended = []
    for value, item, quantity in candidates[:room]:
        order = ["SELL", item, quantity]
        changed["market"].append(order)
        appended.append({"item": item, "quantity": quantity, "value": value})

    return changed, {
        "expanded": dict(expanded),
        "appended": appended,
        "remaining_candidates": max(0, len(candidates) - room),
    }


def agent(obs: Any, configuration: Any = None) -> dict[str, Any]:
    seat = ENGINE._seat(obs)
    step = max(0, min(ENGINE.MAX_STEPS - 1, int(ENGINE._get(obs, "step", 0) or 0)))
    parent_action = PARENT.agent(obs, configuration)
    if step == FINAL_EXECUTABLE_STEP:
        action, frontier = _complete_terminal_frontier(obs, seat, parent_action)
    else:
        action, frontier = parent_action, {
            "expanded": {},
            "appended": [],
            "remaining_candidates": 0,
        }
    _LAST_DIAGNOSTIC[seat] = {
        "step": step,
        "intervened": action != parent_action,
        "frontier": copy.deepcopy(frontier),
        "parent": copy.deepcopy(PARENT._LAST_DIAGNOSTIC.get(seat, {})),
    }
    return action


optimized_agent = agent
