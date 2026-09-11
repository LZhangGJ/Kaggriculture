#!/usr/bin/env python3
"""Build the CPU-only FC15 candidate with strict JAX semantic parity.

FC15 is the already accepted FC12G deploy path plus exactly two independently
validated changes:

* FC14's one-unit residual premium sale on the K320 near-mirror branch;
* actor-specific HIRE barriers for the X562 prefix and PRT suffix weed repair.

The generated file stays self-contained and reads only the live observation and
the candidate's own frozen route tapes.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import zlib

from build_fc12g_cpu_submission import _encode_fc12g_k320
from build_fc2a_cpu_submission import ROOT, TEMPLATE, encode_source


K320_FC14_RESIDUAL_SALE_OVERRIDE = r'''

# FC14/FC15 deploy-parity layer.  This is appended after the complete FC12G
# source so it wraps the final transformed K320 entry point.
_FC15_K320_BASE_AGENT = agent
_FC15_RESIDUAL_PRODUCTS = ("STRAWBERRY", "MILK", "WOOL")


def _fc15_pickup_reserve(action):
    reserve = {item: 0 for item in _FC15_RESIDUAL_PRODUCTS}
    orders = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    for order in orders:
        if not isinstance(order, (list, tuple)) or len(order) < 2:
            continue
        if order[0] != "PICKUP" or order[1] not in reserve:
            continue
        try:
            amount = int(order[2]) if len(order) >= 3 else 1
        except (TypeError, ValueError):
            amount = 0
        reserve[order[1]] += max(0, amount)
    return reserve


def _fc15_append_residual_sale(obs, action, product):
    planned = {item: 0 for item in _FC15_RESIDUAL_PRODUCTS}
    market = [list(order) for order in (action.get("market") or [])]
    for order in market:
        if (
            isinstance(order, list)
            and len(order) >= 3
            and order[0] == "SELL"
            and order[1] in planned
        ):
            try:
                amount = int(order[2])
            except (TypeError, ValueError):
                amount = 0
            planned[order[1]] += max(0, amount)
    reserve = _fc15_pickup_reserve(action)
    projected = _projected_shed(obs, action)
    available = max(
        0,
        int(projected.get(product, 0) or 0)
        - planned[product]
        - reserve[product],
    )
    quantity = min(available, 1)
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    current_price = int(_get(prices, product, 0) or 0)
    base_price = int(_MARKET_PARAMS[product][0])
    if quantity <= 0 or current_price < base_price:
        return action
    existing = next(
        (
            order
            for order in market
            if len(order) >= 3 and order[0] == "SELL" and order[1] == product
        ),
        None,
    )
    if existing is not None:
        existing[2] = int(existing[2] or 0) + quantity
    elif len(market) < 10:
        market.append(["SELL", product, quantity])
    else:
        return action
    action["market"] = market[:10]
    return action


def agent(obs):
    action = _FC15_K320_BASE_AGENT(obs)
    step = min(max(0, int(_get(obs, "step", 0) or 0)), 718)
    if step < 120 or _clone_distance(obs) > 6:
        return action
    for product in _FC15_RESIDUAL_PRODUCTS:
        action = _fc15_append_residual_sale(obs, action, product)
    return _align_hands(action, obs)


__version__ = "FC15-K320-FC14-residual-premium-sale"
'''


SPLIT_WEED_HIRE_GUARD_INSTALLATION = r'''

# FC15 actor-specific weed/HIRE synchronization.  The same implementation is
# installed into X562's embedded X540 namespace and the frozen PRT namespace.
def _install_fc15_split_weed_hire_guard(namespace):
    def future_hires(step, horizon):
        actions = list(namespace.get("_ACTIONS") or [])
        if not actions or horizon < 0:
            return 0
        last = len(actions) - 1
        total = 0
        for offset in range(13):
            if offset > horizon:
                continue
            trace = actions[min(max(int(step) + offset, 0), last)] or {}
            for order in list(trace.get("market") or []):
                if (
                    isinstance(order, (list, tuple))
                    and order
                    and order[0] == "HIRE"
                ):
                    total += 1
        return total

    def weed_repair_action(obs, action, step):
        action = namespace["_align_hands"](action, obs)
        seat = namespace["_seat"](obs)
        game = namespace["_WEED_STATE"][seat]
        if step == 0 or step < game.get("last_step", -1):
            game = {"last_step": step, "active": {}}
            namespace["_WEED_STATE"][seat] = game
        game["last_step"] = step
        farm = namespace["_farm"](obs, seat)
        positions = [
            namespace["_get"](farm, "farmer"),
            *list(namespace["_get"](farm, "hands", []) or []),
        ]
        unit_actions = [
            action.get("farmer", ["PASS"]),
            *list(action.get("hands") or []),
        ]
        active = game["active"]
        farmer_barrier = future_hires(step, 4) >= 5
        hand_barrier = future_hires(step, 2) >= 1

        for actor, transaction in list(active.items()):
            barrier = farmer_barrier if actor == "farmer" else hand_barrier
            if barrier:
                active.pop(actor, None)
                continue
            index = 0 if actor == "farmer" else int(actor) + 1
            if index >= len(unit_actions):
                active.pop(actor, None)
                continue
            age = step - transaction["start"]
            if age == 1:
                unit_actions[index] = list(transaction["intended"])
            elif 2 <= age <= 9:
                unit_actions[index] = namespace["_trace_actor_action"](
                    step - 1, actor
                )
            else:
                active.pop(actor, None)

        for index, (position, intended) in enumerate(zip(positions, unit_actions)):
            actor = "farmer" if index == 0 else index - 1
            barrier = farmer_barrier if actor == "farmer" else hand_barrier
            if barrier:
                continue
            if actor in active or not isinstance(intended, list) or not intended:
                continue
            if intended[0] not in ("BUILD_PASTURE", "PLANT"):
                continue
            tile = namespace["_tile_at"](farm, position)
            if not isinstance(tile, dict) or tile.get("kind") != "WEED":
                continue
            active[actor] = {"start": step, "intended": list(intended)}
            unit_actions[index] = ["DIG"]

        action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
        action["hands"] = unit_actions[1:]
        return namespace["_align_hands"](action, obs)

    namespace["_weed_repair_action"] = weed_repair_action


_install_fc15_split_weed_hire_guard(_X562.__dict__["_X540_NS"])
_install_fc15_split_weed_hire_guard(_PRT.__dict__)
# The accepted JAX route-rule maps every YARN-visible pressure opening to
# latest-bank route 7.  Route 7 is byte-identical to Boatlee V20's frozen
# ``legacy_6c8s_3q`` tape, while the public X562 wrapper normally maps YARN to
# its own route 13 (and has a special dominated-shop exception).  Install the
# exact route-7 tape and the exact any-YARN selector into X562's runtime so the
# CPU package executes the same action owner and feedback stack as JAX.
_FC15_X562_NS = _X562.__dict__["_X540_NS"]
_FC15_X562_NS["_E279_HIGH_ACTIONS"] = _LEGACY.__dict__["_LEGACY_ACTIONS_6C8S_3Q"]


def _fc15_x562_selected_expert(obs):
    namespace = _FC15_X562_NS
    seat = namespace["_seat"](obs)
    step = namespace["_e279_step"](obs)
    state = namespace["_E279_STATE"][seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "shops": (), "expert": None}
        namespace["_E279_STATE"][seat] = state
    state["last_step"] = step
    if step <= namespace["_E279_DECISION_STEP"]:
        state["shops"] = namespace["_e279_shops"](obs)
    if state.get("expert") is None and step >= namespace["_E279_DECISION_STEP"]:
        state["expert"] = (
            "high" if "YARN_STORE" in tuple(state.get("shops") or ()) else "low"
        )
    return str(state.get("expert") or "low")


_FC15_X562_NS["_e279_selected_expert"] = _fc15_x562_selected_expert
# FC15's accepted JAX pressure prefix intentionally calls the route-rule
# controller with ``enable_counters=False``.  The frozen public X562 source
# still contains its older R5/MD counter-sale hooks, so turn only those two
# hooks into identities in the CPU package.  Leaving them enabled adds sales
# that the accepted JAX policy never emits (first observed as an extra WOOL
# sale at action step 378).
_X562.__dict__["_X540_NS"]["_v17_r5_counter"] = (
    lambda obs, action, step: action
)
_X562.__dict__["_X540_NS"]["_v17_md_counter"] = (
    lambda obs, action, step: action
)
# The FC15 JAX PRT suffix deliberately starts from the frozen route tape and
# does not run PRT's dormant premium preemption layer.
_PRT.__dict__["_PREEMPT_ENABLED"] = False
'''


def _encode_fc15_k320(path: Path) -> str:
    fc12g_payload = _encode_fc12g_k320(path)
    source = zlib.decompress(base64.b85decode(fc12g_payload)).decode("utf-8")
    source += K320_FC14_RESIDUAL_SALE_OVERRIDE
    return base64.b85encode(zlib.compress(source.encode("utf-8"), 9)).decode("ascii")


def render_fc15_cpu_source() -> str:
    """Return the accepted self-contained FC15 Python source."""
    k320 = _encode_fc15_k320(
        ROOT / "submission/55655402_rayk_k320_adaptive_rank1/main.py"
    )
    x562 = encode_source(ROOT / "references/public_latest8_20260820/x562_latest/main.py")
    legacy = encode_source(
        ROOT / "references/public_high_potential_20260819/boatlee_v20_multi_route/main.py"
    )
    prt = encode_source(ROOT / "submission/55541953_prt_v6/main.py")
    return (
        TEMPLATE.replace("__K320__", repr(k320))
        .replace("__X562__", repr(x562))
        .replace("__LEGACY__", repr(legacy))
        .replace("__PRT__", repr(prt))
        .replace(
            '_WITH_CONFIGURATION = {',
            SPLIT_WEED_HIRE_GUARD_INSTALLATION + '\n\n_WITH_CONFIGURATION = {',
        )
        .replace(
            '__version__ = "FC2A-rank14-plus-anti-mirror-cpu-v2-shadow-prt"',
            '__version__ = "FC15-fc14-plus-x562-prt-split-weed-hire-guard-cpu-v1"',
        )
        .replace(
            '''    if state["route"] == "legacy":
        branch_action = legacy_action if legacy_action is not None else _call(_LEGACY, obs, configuration)
    else:
        branch_action = x562_action if x562_action is not None else _call(_X562, obs, configuration)
''',
            '''    # Both route 12 and route 7 are now owned by the patched X562
    # runtime, matching the single JAX pressure-prefix action owner.
    branch_action = x562_action if x562_action is not None else _call(_X562, obs, configuration)
''',
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rendered = render_fc15_cpu_source()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
