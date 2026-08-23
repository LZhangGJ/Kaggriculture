#!/usr/bin/env python3
"""Build the self-contained CPU implementation of accepted JAX FC24B.

The deploy path deliberately wraps the already strict-parity FC15 CPU source.
It then applies only the accepted FC16/19/21/22/24B deltas in their JAX order:

* Moon V92 public-market observation, repayment and h4 preemption;
* eight opening WHEAT seeds;
* one-unit early bulk-CARROT seed deferral;
* public-state late animal-feed value guard with a wheat-credit ledger;
* last-feasible terminal crop salvage with the 2x carried-value gate.

The frozen Moon source is used only as an implementation library for the exact
market observer/repayment primitives.  Its production, movement and terminal
policy never owns an FC24B action.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import zlib

from build_fc15_cpu_submission import ROOT, render_fc15_cpu_source


TEMPLATE = r'''from __future__ import annotations

import base64
import copy
import types
import zlib


def _load_frozen(name, payload):
    module = types.ModuleType(name)
    module.__dict__["__name__"] = name
    source = zlib.decompress(base64.b85decode(payload)).decode("utf-8")
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    return module


_BASE = _load_frozen("_fc24b_base", __BASE__)
_MOON = _load_frozen("_fc24b_moon", __MOON__)
_PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
_CROP_FIRST_YIELD_DAY = {
    "WHEAT": 2,
    "CARROT": 2,
    "TOMATO": 8,
    "STRAWBERRY": 10,
    "MELON": 10,
}
# Exact CPython/JAX order frozen by high_potential_v20_gpu._EVAC_ACCESS.
_EVAC_ACCESS = ((4, 4), (5, 4), (5, 5), (4, 5))
_MOVE_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}
_ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
_STATE = {
    0: {
        "last_step": -1,
        "wheat_credit": 0,
        "salvage_active": False,
        "salvage_actor": -1,
        "salvage_target": (0, 0),
        "salvage_product": None,
        "salvage_quantity": 0,
    },
    1: {
        "last_step": -1,
        "wheat_credit": 0,
        "salvage_active": False,
        "salvage_actor": -1,
        "salvage_target": (0, 0),
        "salvage_product": None,
        "salvage_quantity": 0,
    },
}


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _seat(obs):
    return 1 if int(_get(obs, "player", 0) or 0) == 1 else 0


def _farm(obs, seat):
    farms = list(_get(obs, "farms", []) or [])
    return farms[seat] if seat < len(farms) else {}


def _private(obs):
    return _get(obs, "private", {}) or {}


def _market_prices(obs):
    return dict(_get(_get(obs, "market", {}) or {}, "prices", {}) or {})


def _copy_action(action):
    raw = copy.deepcopy(action or {})
    return {
        "farmer": list(raw.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (raw.get("hands") or [])],
        "market": [list(order) for order in (raw.get("market") or [])],
    }


def _reset_if_needed(seat, step):
    state = _STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state.update(
            last_step=step,
            wheat_credit=0,
            salvage_active=False,
            salvage_actor=-1,
            salvage_target=(0, 0),
            salvage_product=None,
            salvage_quantity=0,
        )
    else:
        state["last_step"] = step
    return state


def _base_sheep_pressure(seat):
    try:
        return bool(_BASE.__dict__["_STATE"][seat].get("sheep_pressure", False))
    except Exception:
        return False


def _observe_embedded_moon(obs, step, clone_distance):
    # Embedded FC16 learns horizons only while the public boards remain close.
    # The source observer must still run outside that regime so its previous
    # inventory/price/shop snapshot and 0.999 decay remain sequential.
    original = int(_MOON.__dict__.get("_PREEMPT_MIN_FUTURE_QUANTITY", 4))
    _MOON.__dict__["_ADAPT_MIN_EVIDENCE"] = 1.5
    if clone_distance > 6:
        _MOON.__dict__["_PREEMPT_MIN_FUTURE_QUANTITY"] = 10**9
    try:
        _MOON.__dict__["_observe_opponent_market"](obs, step)
    finally:
        _MOON.__dict__["_PREEMPT_MIN_FUTURE_QUANTITY"] = original


def _moon_h4_overlay(obs, action, step, seat):
    clone_distance = int(_MOON.__dict__["_clone_distance"](obs))
    _observe_embedded_moon(obs, step, clone_distance)
    action = _MOON.__dict__["_repay_shift"](obs, _copy_action(action), step)
    race = _MOON.__dict__["_race_state"](obs, step)
    if step >= 120 and clone_distance <= 2:
        horizons = race.setdefault("horizon", {})
        for product in _MOON.__dict__["_PREMIUM"]:
            horizons[product] = max(4, int(horizons.get(product, 1) or 1))
    if not _base_sheep_pressure(seat) and clone_distance <= 6:
        action = _MOON.__dict__["_preempt_shift"](obs, action, step)
    _MOON.__dict__["_record_own_sells"](obs, action, step)
    return action


def _opening_and_cash_guards(action, step):
    market = [list(order) for order in (action.get("market") or [])]
    for order in market:
        if len(order) < 3:
            continue
        if step == 0 and order[0] == "BUY_SEED" and order[1] == "WHEAT":
            order[2] = 8
        if (
            72 < step < 192
            and order[0] == "BUY_SEED"
            and order[1] == "CARROT"
            and int(order[2] or 0) >= 5
        ):
            order[2] = max(0, int(order[2] or 0) - 1)
    action["market"] = market[:10]
    return action


def _unit_positions(farm):
    return [
        list(_get(farm, "farmer", [0, 0]) or [0, 0]),
        *[list(value or [0, 0]) for value in (_get(farm, "hands", []) or [])],
    ]


def _unit_orders(action):
    return [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]


def _set_unit_order(action, actor, order):
    if actor == 0:
        action["farmer"] = list(order)
    elif 0 < actor <= len(action.get("hands") or []):
        action["hands"][actor - 1] = list(order)


def _tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError):
        return None


def _feed_value_guard(obs, action, step, seat, state):
    farm = _farm(obs, seat)
    prices = _market_prices(obs)
    positions = _unit_positions(farm)
    orders = _unit_orders(action)
    skipped = 0
    if step // 24 >= 10:
        for actor, order in enumerate(orders):
            if not isinstance(order, (list, tuple)) or not order or order[0] != "FEED":
                continue
            if actor >= len(positions):
                continue
            tile = _tile_at(farm, positions[actor])
            if not isinstance(tile, dict):
                continue
            animal = str(tile.get("animal") or "")
            product = _ANIMAL_PRODUCT.get(animal)
            if product is None:
                continue
            if bool(tile.get("fed_today", False)):
                continue
            if int(tile.get("consecutive_unfed", 0) or 0) != 0:
                continue
            pending = max(0, int(tile.get("pending_care_bonus", 0) or 0))
            product_value = int(prices.get(product, 0) or 0) * (1 + pending)
            wheat_price = int(prices.get("WHEAT", 0) or 0)
            if product_value < wheat_price:
                _set_unit_order(action, actor, ["PASS"])
                skipped += 1
    credit = max(0, int(state.get("wheat_credit", 0) or 0)) + skipped
    market = []
    for raw in action.get("market", []) or []:
        order = list(raw)
        if (
            credit > 0
            and len(order) >= 3
            and order[0] == "BUY_PRODUCT"
            and order[1] == "WHEAT"
        ):
            requested = max(0, int(order[2] or 0))
            take = min(requested, credit)
            requested -= take
            credit -= take
            if requested <= 0:
                continue
            order[2] = requested
        market.append(order)
    action["market"] = market[:10]
    state["wheat_credit"] = credit
    return action


def _nearest_evac(position):
    x, y = int(position[0]), int(position[1])
    distance = [abs(x - tx) + abs(y - ty) for tx, ty in _EVAC_ACCESS]
    index = min(range(len(distance)), key=lambda i: distance[i])
    return _EVAC_ACCESS[index], distance[index]


def _move_toward(position, target):
    x, y = int(position[0]), int(position[1])
    tx, ty = int(target[0]), int(target[1])
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def _planned_sales(action):
    out = {product: 0 for product in _PRODUCTS}
    for order in action.get("market", []) or []:
        if (
            isinstance(order, (list, tuple))
            and len(order) >= 3
            and order[0] == "SELL"
            and order[1] in out
        ):
            out[order[1]] += max(0, int(order[2] or 0))
    return out


def _append_or_merge_sale(action, product, quantity):
    market = [list(order) for order in (action.get("market") or [])]
    for order in market:
        if len(order) >= 3 and order[0] == "SELL" and order[1] == product:
            order[2] = int(order[2] or 0) + int(quantity)
            action["market"] = market[:10]
            return action
    if len(market) < 10:
        market.append(["SELL", product, int(quantity)])
    action["market"] = market[:10]
    return action


def _terminal_crop_salvage(obs, action, step, seat, state):
    farm = _farm(obs, seat)
    private = _private(obs)
    positions = _unit_positions(farm)
    orders = _unit_orders(action)
    inventories = list(_get(private, "inventories", []) or [])
    prices = _market_prices(obs)
    day = step // 24

    selected = None
    if not state.get("salvage_active", False) and step >= 696:
        for actor, (position, order) in enumerate(zip(positions, orders)):
            if not isinstance(order, (list, tuple)) or not order or order[0] not in _MOVE_OPS:
                continue
            tile = _tile_at(farm, position)
            if not isinstance(tile, dict):
                continue
            crop = str(tile.get("crop") or "")
            if crop not in _CROP_FIRST_YIELD_DAY:
                continue
            tile_yield = int(tile.get("yield_units", 0) or 0)
            planted_day = int(tile.get("planted_day", -1) or 0)
            if tile_yield <= 0 or day - planted_day < _CROP_FIRST_YIELD_DAY[crop]:
                continue
            target, distance = _nearest_evac(position)
            if distance + 2 != 719 - step:
                continue
            inventory = dict(inventories[actor] or {}) if actor < len(inventories) else {}
            carried_value = sum(
                max(0, int(inventory.get(product, 0) or 0))
                * int(prices.get(product, 0) or 0)
                for product in _PRODUCTS
            )
            crop_value = tile_yield * int(prices.get(crop, 0) or 0)
            if crop_value < 2 * carried_value:
                continue
            score = crop_value * 100 - actor
            if selected is None or score > selected[0]:
                selected = (score, actor, target, crop, tile_yield)

    if selected is not None:
        _, actor, target, product, quantity = selected
        state.update(
            salvage_active=True,
            salvage_actor=actor,
            salvage_target=tuple(target),
            salvage_product=product,
            salvage_quantity=quantity,
        )
        _set_unit_order(action, actor, ["HARVEST"])
    elif state.get("salvage_active", False):
        actor = int(state.get("salvage_actor", -1))
        target = tuple(state.get("salvage_target", (0, 0)))
        if 0 <= actor < len(positions):
            if tuple(positions[actor]) == target:
                _set_unit_order(action, actor, ["DROP"])
            else:
                _set_unit_order(action, actor, _move_toward(positions[actor], target))

    actor = int(state.get("salvage_actor", -1))
    drop = (
        state.get("salvage_active", False)
        and 0 <= actor < len(positions)
        and tuple(positions[actor]) == tuple(state.get("salvage_target", (0, 0)))
        and step == 718
    )
    if drop:
        projected = _MOON.__dict__["_projected_shed"](obs, action)
        planned = _planned_sales(action)
        product = state.get("salvage_product")
        quantity = max(0, int(state.get("salvage_quantity", 0) or 0))
        incremental = min(
            quantity,
            max(0, int(projected.get(product, 0) or 0) - planned.get(product, 0)),
        )
        if incremental > 0:
            action = _append_or_merge_sale(action, product, incremental)
    return action


def agent(obs, configuration=None):
    step = min(max(0, int(_get(obs, "step", 0) or 0)), 718)
    seat = _seat(obs)
    state = _reset_if_needed(seat, step)
    action = _copy_action(_BASE.__dict__["agent"](obs, configuration))
    action = _moon_h4_overlay(obs, action, step, seat)
    action = _opening_and_cash_guards(action, step)
    action = _feed_value_guard(obs, action, step, seat, state)
    action = _terminal_crop_salvage(obs, action, step, seat, state)
    return action


__version__ = "FC24B-fc22-plus-value-guarded-terminal-crop-salvage-cpu-v1"
'''


def _encode_text(source: str) -> str:
    return base64.b85encode(zlib.compress(source.encode("utf-8"), 9)).decode("ascii")


def render_fc24b_cpu_source() -> str:
    base = _encode_text(render_fc15_cpu_source())
    moon_path = (
        ROOT
        / "references/public_latest6_20260822/prvsiyan_moon_v92_latest/main.py"
    )
    moon = _encode_text(moon_path.read_text(encoding="utf-8"))
    return TEMPLATE.replace("__BASE__", repr(base)).replace("__MOON__", repr(moon))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rendered = render_fc24b_cpu_source()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
