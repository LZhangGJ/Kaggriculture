#!/usr/bin/env python3
"""Check offline full-order market projection against saved v306 ticks and one-less-sale clones."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import fast_kaggriculture


ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG",
         "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP")
OPS = {name: i for i, name in enumerate(("SELL", "BUY_PRODUCT", "BUY_SEED",
                                         "BUY_ANIMAL", "HIRE", "BUY_LAND"))}
FIRST_DAY = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}


def preview_unit_goods(obs, seat, action):
    """Only the official unit operations that can change shed or ordered worker bags."""
    farm, private = obs["farms"][seat], obs["private"]
    shed = dict(private["shed"])
    bags = [dict(x) for x in private["inventories"]]
    positions = [farm["farmer"], *farm["hands"]]
    moves = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    day = int(obs["day"])
    for idx, move in enumerate(moves[:len(positions)]):
        op = move[0]
        item = move[1] if len(move) > 1 else None
        quantity = max(0, int(move[2])) if len(move) > 2 else 1
        x, y = positions[idx]
        near = x in (4, 5) and y in (4, 5)
        tile = farm["tiles"][y][x]
        bag = bags[idx]
        def take(name, n):
            if bag.get(name, 0) < n or n <= 0:
                return False
            bag[name] -= n
            if not bag[name]:
                del bag[name]
            return True
        def add(name, n):
            if n > 0:
                bag[name] = bag.get(name, 0) + n
        if op == "DROP" and near:
            for name, count in bag.items():
                moved = min(count, 100 - sum(shed.values()))
                shed[name] += moved
            bag.clear()
        elif op == "PICKUP" and near and item in ITEMS and quantity > 0:
            moved = min(quantity, shed[item])
            shed[item] -= moved
            add(item, moved)
        elif op == "PLACE":
            if (item in ITEMS[9:] and isinstance(tile, dict)
                    and tile.get("kind") == ("COOP" if item == "GOOSE" else "PASTURE")
                    and not tile.get("animal")):
                take(item, 1)
            elif near and item in ITEMS and quantity > 0:
                moved = min(quantity, bag.get(item, 0), 100 - sum(shed.values()))
                if take(item, moved):
                    shed[item] += moved
        elif op == "HARVEST" and isinstance(tile, dict) and tile.get("yield_units", 0) > 0:
            if tile.get("kind") == "PLANT":
                crop = tile["crop"]
                if day - tile["planted_day"] >= FIRST_DAY[crop]:
                    add(crop, int(tile["yield_units"]))
            elif tile.get("animal") in ITEMS[9:]:
                add({"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}[tile["animal"]],
                    int(tile["yield_units"]))
        elif op == "FERTILIZE" and isinstance(tile, dict) and tile.get("kind") == "PLANT":
            take("FERTILIZER", 1)
        elif op == "FEED" and isinstance(tile, dict) and tile.get("animal") in ITEMS[9:] and not tile.get("fed_today"):
            take("WHEAT", 1)
        elif op == "COLLECT_FERTILIZER" and isinstance(tile, dict) and tile.get("animal") in ITEMS[9:] and tile.get("fertilizer_available"):
            add("FERTILIZER", 1)
    return shed, bags


def transfer_end_day(shed, bags):
    shed = dict(shed)
    overflow = 0
    for bag in bags:
        for item, count in bag.items():
            moved = min(count, 100 - sum(shed.values()))
            shed[item] += moved
            overflow += count - moved
    return shed, overflow


def quantity(order):
    return int(order[2]) if len(order) > 2 else 1


def snapshot(obs, seat):
    return (tuple(obs["market"]["inventory"][i] for i in ITEMS[:9]),
            int(obs["farms"][seat]["money"]),
            tuple(obs["private"]["shed"][i] for i in ITEMS))


def make_line(before, goods, actions, town, end_day):
    values = [*(before[0]["market"]["inventory"][i] for i in ITEMS[:9]), *town]
    for seat in (0, 1):
        farm, (shed, _) = before[seat]["farms"][seat], goods[seat]
        values.extend((int(farm["money"]), int(farm["hires_today"]),
                       len(farm["unlocked_quadrants"]) - 1))
        values.extend(shed[i] for i in ITEMS)
        orders = actions[seat].get("market", [])
        values.append(len(orders))
        for order in orders:
            values.extend((OPS.get(order[0], 6),
                           ITEMS.index(order[1]) if len(order) > 1 and order[1] in ITEMS else -1,
                           quantity(order)))
    values.append(int(end_day))
    for _, bags in goods:
        values.append(len(bags))
        for bag in bags:
            values.append(len(bag))
            for item, count in bag.items():
                values.extend((ITEMS.index(item), count))
    return " ".join(map(str, values))


def decode(line):
    numbers = list(map(int, line.split()))
    market, players, at = tuple(numbers[:9]), [], 9
    for _ in (0, 1):
        cash = numbers[at]; at += 1
        shed = tuple(numbers[at:at + 12]); at += 12
        count = numbers[at]; at += 1
        fills = tuple(numbers[at:at + count]); at += count
        overflow = numbers[at]; at += 1
        players.append((cash, shed, fills, overflow))
    if at != len(numbers):
        raise AssertionError("trailing projector fields")
    return market, players


def expected(env, after):
    market = tuple(after[0]["market"]["inventory"][i] for i in ITEMS[:9])
    players = []
    for seat in (0, 1):
        players.append((int(after[seat]["farms"][seat]["money"]),
                        tuple(after[seat]["private"]["shed"][i] for i in ITEMS),
                        tuple(env.last_market_fills[seat]),
                        int(env.last_end_of_day_overflow[seat])))
    return market, players


def day_suffix_pair(before_env, actions, alternative, steps, tick, seat):
    """Fixed future actions: an oracle diagnostic, not an online rival forecast."""
    original, changed = before_env.clone(), before_env.clone()
    original.step_raw(actions); changed.step_raw(alternative)
    stop = min(((tick // 24) + 1) * 24, len(steps) - 1)
    for future in range(tick + 1, stop):
        fixed = [steps[future + 1][p]["action"] or {} for p in (0, 1)]
        original.step_raw(fixed); changed.step_raw(fixed)
    a, b = original.observation(seat), changed.observation(seat)
    liquidate = [{"market": [["SELL", item, 100] for item in ITEMS[:9]]}
                 if p == seat else {} for p in (0, 1)]
    original.step_raw(liquidate); changed.step_raw(liquidate)
    la, lb = original.observation(seat), changed.observation(seat)
    return {
        "tick": tick, "end_tick": stop,
        "cash_delta": int(a["farms"][seat]["money"] - b["farms"][seat]["money"]),
        "margin_delta": int((a["farms"][seat]["money"] - a["farms"][1 - seat]["money"])
                            - (b["farms"][seat]["money"] - b["farms"][1 - seat]["money"])),
        "product_shed_delta": sum(a["private"]["shed"][i] - b["private"]["shed"][i]
                                  for i in ITEMS[:9]),
        "market_changed": a["market"]["inventory"] != b["market"]["inventory"],
        "liquidated_cash_delta": int(la["farms"][seat]["money"] - lb["farms"][seat]["money"]),
        "liquidated_margin_delta": int((la["farms"][seat]["money"] - la["farms"][1 - seat]["money"])
                                       - (lb["farms"][seat]["money"] - lb["farms"][1 - seat]["money"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--projector", required=True, type=Path)
    parser.add_argument("--end", type=int, default=480)
    parser.add_argument("--day-suffix", action="store_true",
                        help="first SELL of each NN day, fixed-action suffix and exact terminal liquidation")
    args = parser.parse_args()
    cases, coverage, day_suffix = [], Counter(), []
    for path in args.replays:
        replay = json.loads(path.read_text())
        seat = replay["info"]["TeamNames"].index("QQ Farming")
        env = fast_kaggriculture.FastEnv(seed=int(replay["info"]["seed"]))
        steps = replay["steps"]
        seen_day = set()
        for tick in range(min(args.end, len(steps) - 1)):
            before = [env.observation(p) for p in (0, 1)]
            for p in (0, 1):
                if snapshot(before[p], p) != snapshot(steps[tick][p]["observation"], p):
                    raise AssertionError(f"source replay pre-state {path.name}:{tick}:{p}")
            actions = [steps[tick + 1][p]["action"] or {} for p in (0, 1)]
            if tick >= 288:
                end_day = tick % 24 == 23
                no_market = [{**a, "market": []} for a in actions]
                probe = env.clone(); probe.step_raw(no_market)
                unit_after = [probe.observation(p) for p in (0, 1)]
                goods = [preview_unit_goods(before[p], p, actions[p]) for p in (0, 1)]
                for p in (0, 1):
                    shed, bags = goods[p]
                    if end_day:
                        shed, overflow = transfer_end_day(shed, bags)
                        if overflow != probe.last_end_of_day_overflow[p]:
                            raise AssertionError(f"unit preview overflow {path.name}:{tick}:{p}")
                    elif bags != unit_after[p]["private"]["inventories"]:
                        raise AssertionError(f"unit preview bags {path.name}:{tick}:{p}")
                    if shed != unit_after[p]["private"]["shed"]:
                        raise AssertionError(f"unit preview shed {path.name}:{tick}:{p}")
                town = [before[0]["market"]["inventory"][i]
                        - unit_after[0]["market"]["inventory"][i] for i in ITEMS[:9]]
                if any(unit_after[p]["farms"][p]["money"] != before[p]["farms"][p]["money"]
                       for p in (0, 1)):
                    raise AssertionError("unit/no-market phase changed cash")
                line = make_line(before, goods, actions, town, end_day)
                coverage["day_end_ticks" if end_day else "non_eod_ticks"] += 1
                sell_slots = [j for j, order in enumerate(actions[seat].get("market", []))
                              if order[0] == "SELL"]
                if sell_slots:
                    coverage["sell_ticks_any_slot"] += 1
                    coverage["sell_orders"] += len(sell_slots)
                    if sell_slots[0] > 0:
                        coverage["sell_ticks_first_sale_after_slot0"] += 1
                if actions[seat].get("market") and actions[seat]["market"][0][0] == "SELL":
                    alt = json.loads(json.dumps(actions))
                    alt[seat]["market"][0] = list(alt[seat]["market"][0])
                    alt[seat]["market"][0][2:] = [quantity(alt[seat]["market"][0]) - 1]
                    fork = env.clone(); fork.step_raw(alt)
                    alt_after = [fork.observation(p) for p in (0, 1)]
                    if args.day_suffix and tick // 24 not in seen_day:
                        seen_day.add(tick // 24)
                        day_suffix.append(day_suffix_pair(env, actions, alt, steps, tick, seat))
                    cases.append((path.name, tick, "less_one", seat,
                                  make_line(before, goods, alt, town, end_day), expected(fork, alt_after)))
                    coverage["one_less_sale_alternatives"] += 1
                if any(o[0] == "BUY_PRODUCT" for a in actions for o in a.get("market", [])):
                    coverage["buy_product_ticks"] += 1
                if len({o[1] for a in actions for o in a.get("market", [])
                        if len(o) > 1 and o[0] in ("SELL", "BUY_PRODUCT")}) > 1:
                    coverage["multi_product_trade_ticks"] += 1
                cases.append((path.name, tick, "actual", seat, line, None))
            env.step_raw(actions)
            after = [env.observation(p) for p in (0, 1)]
            for p in (0, 1):
                if snapshot(after[p], p) != snapshot(steps[tick + 1][p]["observation"], p):
                    raise AssertionError(f"source replay post-state {path.name}:{tick}:{p}")
            if tick >= 288:
                name, k, kind, _, line, _ = cases[-1]
                assert (name, k, kind) == (path.name, tick, "actual")
                cases[-1] = (name, k, kind, seat, line, expected(env, after))
    batch = "\n".join(case[4] for case in cases) + "\n"
    output = subprocess.run([str(args.projector)], input=batch, text=True,
                            capture_output=True, check=True).stdout.splitlines()
    if len(output) != len(cases):
        raise AssertionError(f"projector returned {len(output)} lines for {len(cases)} cases")
    by_tick = {}
    deltas, changed_fill_examples = [], []
    for case, line in zip(cases, output):
        name, tick, kind, seat, _, truth = case
        got = decode(line)
        if got != truth:
            raise AssertionError(f"market projection mismatch {name}:{tick}:{kind}: {got} != {truth}")
        coverage[kind + "_exact"] += 1
        if kind == "actual":
            alt = by_tick.pop((name, tick), None)
            if alt is not None:
                changed = got[1][seat][2][1:] != alt[1][seat][2][1:]
                deltas.append((got[1][seat][0] - alt[1][seat][0],
                               got[0] != alt[0], changed,
                               got[1][1 - seat][2] != alt[1][1 - seat][2]))
                if tick % 24 == 23:
                    coverage["day_end_less_one_exact"] += 1
                    if got[1][seat][3] != alt[1][seat][3]:
                        coverage["day_end_less_one_changes_overflow"] += 1
                if changed and len(changed_fill_examples) < 8:
                    changed_fill_examples.append({"episode": name, "tick": tick,
                                                  "cash_delta": deltas[-1][0],
                                                  "fills_actual": got[1][seat][2],
                                                  "fills_less_one": alt[1][seat][2]})
        else:
            by_tick[(name, tick)] = got
        if kind == "actual" and tick % 24 == 23:
            overflow = sum(p[3] for p in got[1])
            coverage["day_end_overflow_units"] += overflow
            if overflow:
                coverage["day_end_ticks_with_overflow"] += 1
    if by_tick:
        raise AssertionError("unpaired one-less-sale alternatives")
    money = sorted(x[0] for x in deltas)
    print(json.dumps({"sources": len(args.replays), "coverage": coverage,
                      "one_less_sale": {"cash_min": money[0],
                                        "cash_median": money[len(money) // 2],
                                        "cash_max": money[-1],
                                        "market_changed": sum(x[1] for x in deltas),
                                        "later_own_fills_changed": sum(x[2] for x in deltas),
                                        "rival_fills_changed": sum(x[3] for x in deltas),
                                        "changed_fill_examples": changed_fill_examples},
                      "day_suffix": {"count": len(day_suffix),
                                     "cash_delta": {"min": min((x["cash_delta"] for x in day_suffix), default=0),
                                                    "median": sorted(x["cash_delta"] for x in day_suffix)[len(day_suffix) // 2] if day_suffix else 0,
                                                    "max": max((x["cash_delta"] for x in day_suffix), default=0)},
                                     "liquidated_delta": {"min": min((x["liquidated_cash_delta"] for x in day_suffix), default=0),
                                                          "median": sorted(x["liquidated_cash_delta"] for x in day_suffix)[len(day_suffix) // 2] if day_suffix else 0,
                                                          "max": max((x["liquidated_cash_delta"] for x in day_suffix), default=0)},
                                     "liquidated_margin_delta": {"min": min((x["liquidated_margin_delta"] for x in day_suffix), default=0),
                                                                 "median": sorted(x["liquidated_margin_delta"] for x in day_suffix)[len(day_suffix) // 2] if day_suffix else 0,
                                                                 "max": max((x["liquidated_margin_delta"] for x in day_suffix), default=0)},
                                     "cash_vs_liquidation_sign_flip": sum(x["cash_delta"] * x["liquidated_cash_delta"] < 0 for x in day_suffix),
                                     "liquidation_prefers_less": sum(x["liquidated_cash_delta"] < 0 for x in day_suffix),
                                     "liquidation_prefers_more": sum(x["liquidated_cash_delta"] > 0 for x in day_suffix),
                                     "liquidation_ties": sum(x["liquidated_cash_delta"] == 0 for x in day_suffix),
                                     "margin_sign_flip": sum(x["margin_delta"] * x["liquidated_margin_delta"] < 0 for x in day_suffix),
                                     "market_changed": sum(x["market_changed"] for x in day_suffix),
                                     "examples": day_suffix[:8]}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
