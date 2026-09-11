#!/usr/bin/env python3
"""Build the CPU-only FC12G mass-HIRE weed-guard candidate.

FC12G is FC2B plus one independently ablatable K320 correction.  A farmer
weed-repair insertion is cancelled/suppressed when the agent's own route tape
contains at least five HIRE orders in the current-to-next-two-step window.
This keeps deterministic hand spawn indices stable across the mass-HIRE
transaction.  No opponent identity or future environment state is used.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import zlib

from build_fc2a_cpu_submission import ROOT, TEMPLATE, encode_source


FC2B_REPLACEMENTS = (
    ("_PREEMPT_MAX_BATCH = 12", "_PREEMPT_MAX_BATCH = 32"),
    (
        '_PREMIUM = ("STRAWBERRY", "MELON", "MILK", "WOOL")',
        '_PREMIUM = ("STRAWBERRY", "MILK", "WOOL")',
    ),
    ("elif clone <= 6:\n        horizon = 2", "elif clone <= 6:\n        horizon = 4"),
)


WEED_GUARD_INSERTION = '''def _mass_hire_farmer_barrier(obs, step):
    """Use only our own route tape; offsets 0..2 match the JAX FC12G guard."""
    actions = _kawa_actions(obs)
    last = len(actions) - 1
    hires = 0
    for offset in range(3):
        trace = actions[min(max(int(step) + offset, 0), last)] or {}
        for order in list(trace.get("market") or []):
            if isinstance(order, (list, tuple)) and order and order[0] == "HIRE":
                hires += 1
    return hires >= 5


def _weed_repair_action(obs, action, step):
    action = _align_hands(action, obs)
    farmer_barrier = _mass_hire_farmer_barrier(obs, step)
'''


JAX_CLONE_AWARE_PREEMPT_OVERRIDE = r'''

# FC12G deploy-parity override.  The JAX policy uses the aggressive hedge only
# for an exact public mirror and a smaller broad hedge after the farms diverge.
# This function deliberately reads only the current public observation and our
# own frozen route tape; no opponent identity or future random event is used.
def _preempt_shift(obs, action, step):
    if not _PREEMPT_ENABLED:
        return action
    clone = _clone_distance(obs)
    exact_clone = clone == 0
    start_step = 120 if exact_clone else 216
    distance_limit = 6 if exact_clone else 100
    quantity_cap = 32 if exact_clone else 12
    products = (
        ("STRAWBERRY", "MILK", "WOOL")
        if exact_clone
        else ("STRAWBERRY", "MELON", "MILK", "WOOL")
    )
    if not (start_step <= step < 680) or clone > distance_limit:
        return action
    state = _shift_state(obs, step)
    if state.get("due"):
        return action
    actions = _kawa_actions(obs)
    future = {}
    for ahead in range(1, 5):
        if step + ahead >= len(actions):
            break
        for raw in (actions[step + ahead].get("market") or []):
            if len(raw) >= 3 and raw[0] == "SELL" and raw[1] in products:
                future[raw[1]] = future.get(raw[1], 0) + max(0, int(raw[2]))
    if not future:
        return action
    market = [list(raw) for raw in (action.get("market") or [])]
    if len(market) >= 10:
        return action
    remaining = _projected_shed(obs, action)
    for raw in market:
        if len(raw) >= 3 and raw[0] == "SELL":
            item = raw[1]
            remaining[item] = max(
                0, int(remaining.get(item, 0) or 0) - max(0, int(raw[2]))
            )
    shifted = {}
    for item in ("STRAWBERRY", "MELON", "MILK", "WOOL"):
        if item not in products:
            continue
        future_quantity = max(0, int(future.get(item, 0) or 0))
        if future_quantity < 4:
            continue
        quantity = min(
            max(0, int(remaining.get(item, 0) or 0)),
            future_quantity,
            quantity_cap,
        )
        if quantity <= 0 or len(market) >= 10:
            continue
        market.append(["SELL", item, quantity])
        remaining[item] = max(0, int(remaining.get(item, 0) or 0) - quantity)
        shifted[item] = quantity
    if shifted:
        action["market"] = market[:10]
        state["due_step"] = step + 4
        state["due"] = shifted
    return action
'''


def _encode_fc12g_k320(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    for old, new in FC2B_REPLACEMENTS:
        count = source.count(old)
        if count != 1:
            raise AssertionError(f"expected one FC2B replacement in {path}: {old!r}, got {count}")
        source = source.replace(old, new)

    function_start = '''def _weed_repair_action(obs, action, step):
    action = _align_hands(action, obs)
'''
    if source.count(function_start) != 1:
        raise AssertionError("unexpected _weed_repair_action function header")
    source = source.replace(function_start, WEED_GUARD_INSERTION)

    transaction_start = '''    for actor, transaction in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
'''
    transaction_guarded = '''    for actor, transaction in list(active.items()):
        if farmer_barrier and actor == "farmer":
            active.pop(actor, None)
            continue
        index = 0 if actor == "farmer" else int(actor) + 1
'''
    if source.count(transaction_start) != 1:
        raise AssertionError("unexpected active weed transaction loop")
    source = source.replace(transaction_start, transaction_guarded)

    trigger_start = '''        actor = "farmer" if index == 0 else index - 1
        if actor in active or not isinstance(intended, list) or not intended:
'''
    trigger_guarded = '''        actor = "farmer" if index == 0 else index - 1
        if farmer_barrier and actor == "farmer":
            continue
        if actor in active or not isinstance(intended, list) or not intended:
'''
    if source.count(trigger_start) != 1:
        raise AssertionError("unexpected weed trigger loop")
    source = source.replace(trigger_start, trigger_guarded)
    source += JAX_CLONE_AWARE_PREEMPT_OVERRIDE
    return base64.b85encode(zlib.compress(source.encode("utf-8"), 9)).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    k320 = _encode_fc12g_k320(ROOT / "submission/55655402_rayk_k320_adaptive_rank1/main.py")
    x562 = encode_source(ROOT / "references/public_latest8_20260820/x562_latest/main.py")
    legacy = encode_source(ROOT / "references/public_high_potential_20260819/boatlee_v20_multi_route/main.py")
    prt = encode_source(ROOT / "submission/55541953_prt_v6/main.py")
    rendered = (
        TEMPLATE.replace("__K320__", repr(k320))
        .replace("__X562__", repr(x562))
        .replace("__LEGACY__", repr(legacy))
        .replace("__PRT__", repr(prt))
        .replace(
            '__version__ = "FC2A-rank14-plus-anti-mirror-cpu-v2-shadow-prt"',
            '__version__ = "FC12G-fc2b-mass-hire-weed-guard-cpu-v2-deploy-parity"',
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
