#!/usr/bin/env python3
"""Replay saved NN ticks into FastEnv; compare eligible one-item market events with C++ projector.

Build: g++ -std=c++20 -O2 -Ipolicy/r1 experiments/sale_postprocess_one_item.cpp -o /tmp/sale_postprocess_one_item
Run:   PYTHONPATH=. /root/miniforge3/envs/torch-npu/bin/python experiments/audit_sale_postprocess_fastenv.py \
         --projector /tmp/sale_postprocess_one_item build/v306-kaggle-logs/*-replay.json
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import fast_kaggriculture


ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
SEEDS = dict(zip(ITEMS[:5], (10, 20, 50, 100, 80)))
FIB = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987)
LAND = (1000, 2000, 4000)


def quantity(order):
    return int(order[2]) if len(order) > 2 else 1


def fixed_costs(orders, farm):
    costs = []
    hires, land = farm["hires_today"], len(farm["unlocked_quadrants"]) - 1
    for order in orders:
        op = order[0]
        if op == "BUY_SEED" and order[1] in SEEDS:
            costs.extend([SEEDS[order[1]]] * quantity(order))
        elif op == "HIRE" and hires < len(FIB):
            costs.append(FIB[hires]); hires += 1
        elif op == "BUY_LAND" and land < len(LAND):
            costs.append(LAND[land]); land += 1
        else:
            return None
    return costs


def same_snapshot(actual, saved, seat):
    step = lambda o: o.get("step") if o.get("step") is not None else 24 * o["day"] + o["hour"]
    return (step(actual) == step(saved) and
            actual["market"]["inventory"] == saved["market"]["inventory"] and
            actual["farms"][seat]["money"] == saved["farms"][seat]["money"] and
            actual["private"]["shed"] == saved["private"]["shed"])


def audit_replay(path, end, fixtures, rejected, coverage):
    replay = json.loads(path.read_text())
    steps = replay["steps"]
    seat = replay["info"]["TeamNames"].index("QQ Farming")
    env = fast_kaggriculture.FastEnv(seed=int(replay["info"]["seed"]))
    checked_ticks = 0
    for tick in range(min(end, len(steps) - 1)):
        before = env.observation(seat)
        saved = steps[tick][seat]["observation"]
        if not same_snapshot(before, saved, seat):
            raise AssertionError(f"saved/FastEnv pre-state mismatch {path.name} step {tick}")
        actions = [steps[tick + 1][player]["action"] or {} for player in (0, 1)]
        candidate = None
        if tick >= 288:
            coverage["nn_ticks"] += 1
            market = actions[seat].get("market", [])
            if market and market[0][0] == "SELL":
                coverage["sell_ticks"] += 1
                if tick % 24 == 0:
                    coverage["day_boundary_sell_ticks"] += 1
                item = market[0][1]
                other = actions[1 - seat].get("market", [])
                if tick % 24 == 23:
                    rejected["end_of_day_shed_transfer"] += 1
                elif item not in ITEMS or quantity(market[0]) <= 0:
                    rejected["bad_sell"] += 1
                elif len(market) > 10:
                    rejected["order_slot_limit"] += 1
                elif (costs := fixed_costs(market[1:], before["farms"][seat])) is None:
                    rejected["multi_product_or_unsupported_fixed_order"] += 1
                elif any(len(x) > 1 and x[1] == item and x[0] in ("SELL", "BUY_PRODUCT")
                         for x in other[1:]):
                    rejected["rival_later_same_product_order"] += 1
                elif other and len(other[0]) > 1 and other[0][1] == item and other[0][0] != "SELL":
                    rejected["rival_same_slot_non_sell"] += 1
                else:
                    no_market = [{**a, "market": []} for a in actions]
                    probe = env.clone()
                    probe.step_raw(no_market)
                    probe_after = probe.observation(seat)
                    rival = (quantity(other[0]) if other and other[0][0] == "SELL"
                             and other[0][1] == item else 0)
                    candidate = {
                        "episode": path.stem, "tick": tick, "seat": seat, "item": item,
                        "costs": costs, "sell": quantity(market[0]), "rival": rival,
                        "unit_shed_delta": (probe_after["private"]["shed"][item]
                                            - before["private"]["shed"][item]),
                        "before": before, "town_demand": (before["market"]["inventory"][item]
                                                         - probe_after["market"]["inventory"][item]),
                    }
        env.step_raw(actions)
        after = env.observation(seat)
        if not same_snapshot(after, steps[tick + 1][seat]["observation"], seat):
            raise AssertionError(f"saved/FastEnv post-state mismatch {path.name} step {tick + 1}")
        checked_ticks += 1
        if candidate is not None:
            own_fills = env.last_market_fills[seat]
            rival_fills = env.last_market_fills[1 - seat]
            if own_fills[0] != candidate["sell"] or any(
                own_fills[i] != (1 if market[i][0] in ("HIRE", "BUY_LAND") else quantity(market[i]))
                for i in range(1, len(market))
            ):
                rejected["observed_partial_own_order"] += 1
            elif candidate["rival"] and rival_fills[0] != candidate["rival"]:
                rejected["observed_partial_rival_order"] += 1
            else:
                candidate["after"] = after
                candidate["fills"] = own_fills
                fixtures.append(candidate)
    return {"episode": path.stem, "replayed_ticks": checked_ticks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--projector", required=True, type=Path)
    parser.add_argument("--end", type=int, default=408, help="exclusive upper tick; no full games")
    args = parser.parse_args()
    fixtures, rejected, coverage = [], Counter(), Counter()
    sources = [audit_replay(path, args.end, fixtures, rejected, coverage) for path in args.replays]
    lines, less_one = [], []
    for x in fixtures:
        o, item = x["before"], x["item"]
        ints = (ITEMS.index(item), o["market"]["inventory"][item], o["private"]["shed"][item],
                int(o["farms"][x["seat"]]["money"]), x["sell"], x["rival"], x["town_demand"],
                len(x["costs"]), *x["costs"], x["unit_shed_delta"])
        lines.append(" ".join(map(str, ints)))
        less_one.append(" ".join(map(str, (*ints[:4], x["sell"] - 1, *ints[5:]))))
    result = subprocess.run([str(args.projector), "--project"], input="\n".join(lines) + "\n",
                            text=True, capture_output=True, check=True)
    outputs = result.stdout.splitlines()
    if len(outputs) != len(fixtures):
        raise AssertionError(f"projector returned {len(outputs)} lines for {len(fixtures)} events")
    alternatives = subprocess.run([str(args.projector), "--project"],
                                  input="\n".join(less_one) + "\n", text=True,
                                  capture_output=True, check=True).stdout.splitlines()
    if len(alternatives) != len(fixtures):
        raise AssertionError("projector returned wrong number of one-less-sale alternatives")
    passed, examples, accepted, cash_deltas, market_deltas = 0, [], Counter(), [], []
    for x, line, alternative in zip(fixtures, outputs, alternatives):
        words = line.split()
        if words[0] != "OK":
            raise AssertionError(f"projector rejected accepted event {x['episode']}:{x['tick']}: {line}")
        inv, shed, cash, sold, _, admitted, _, filled = map(int, words[1:])
        after, item = x["after"], x["item"]
        actual = (after["market"]["inventory"][item], after["private"]["shed"][item],
                  int(after["farms"][x["seat"]]["money"]), x["fills"][0], len(x["costs"]))
        projected = (inv, shed, cash, sold, filled)
        if projected != actual:
            raise AssertionError(f"market parity {x['episode']}:{x['tick']} {item}: {projected} != {actual}")
        alt = alternative.split()
        if alt[0] == "REJECT":
            raise AssertionError(f"one-less-sale alternative rejected {x['episode']}:{x['tick']}")
        cash_deltas.append(cash - int(alt[3]))
        market_deltas.append(inv - int(alt[1]))
        if alt[0] == "INCOMPLETE":
            accepted["one_less_sale_loses_fixed_purchase"] += 1
        passed += 1
        accepted["day_boundary" if x["tick"] % 24 == 0 else "intraday"] += 1
        if x["costs"]:
            accepted["with_fixed_purchase_units"] += 1
        if x["rival"]:
            accepted["same_slot_rival_sale"] += 1
        if x["unit_shed_delta"]:
            accepted["unit_phase_shed_delta"] += 1
        if admitted < sold:
            accepted["dollar_floor_sale"] += 1
        if len(examples) < 8:
            examples.append({"episode": x["episode"], "step": x["tick"], "seat": x["seat"],
                             "item": item, "sell": sold, "fixed_orders": len(x["costs"]),
                             "market_after": inv, "shed_after": shed, "cash_after": cash})
    print(json.dumps({"sources": sources, "coverage": coverage, "rejected": rejected,
                      "exact_events": passed, "accepted": accepted,
                      "one_less_sale_delta": {"cash_min": min(cash_deltas),
                                              "cash_median": sorted(cash_deltas)[len(cash_deltas) // 2],
                                              "cash_max": max(cash_deltas),
                                              "market_admitted": dict(Counter(market_deltas))},
                      "examples": examples},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
