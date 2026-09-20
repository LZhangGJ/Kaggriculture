#!/usr/bin/env python3
"""Export Thomas tapes/router and locate the first native-prefix mismatch."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path

from fast_kaggriculture import Config, FastEnv, NativeTeammateExecutor


ROOT = Path(__file__).resolve().parents[1]
THOMAS = ROOT / "opponents/thomas_2945/main.py"
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def load_thomas(tag: str):
    spec = importlib.util.spec_from_file_location(tag, THOMAS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_payload(module) -> dict:
    return {
        "schema": "thomas-native-prefix-input-v1",
        "source": str(THOMAS.relative_to(ROOT)),
        "router": {"switch_step": 144, "opening": 0, "new_default": 100},
        "route_ids": sorted(module._ROUTES),
        "routes": {str(key): module._ROUTES[key] for key in sorted(module._ROUTES)},
        "shop_routes": {
            "new": [[list(key), value] for key, value in module._R108_SHOP_ROUTES.items()],
            "old": [[list(key), value] for key, value in module._R110_OLD_SHOPS.items()],
            "v92": [[list(key), value] for key, value in module._V92_TABLE.items()],
        },
        "rival_overrides": [
            [list(key), value] for key, value in module._V93_ROUTE_BY_RIVAL.items()
        ],
    }


def select_route(payload: dict, shops: tuple[str, ...], rkey) -> int:
    tables = {
        name: {tuple(key): value for key, value in rows}
        for name, rows in payload["shop_routes"].items()
    }
    route = (tables["new"].get(shops, payload["router"]["new_default"])
             if "YARN_STORE" not in shops else tables["old"].get(shops, 0))
    route = tables["v92"].get(shops, route)
    return dict((tuple(key), value) for key, value in payload["rival_overrides"]).get(
        tuple(rkey) if rkey else None, route
    )


def normalize(action: dict) -> dict:
    # C++ renders an ignored quantity=1 on two-field unit actions.
    def unit(value):
        value = list(value or ["PASS"])
        return value + [1] if len(value) == 2 else value

    return {
        "farmer": unit(action.get("farmer")),
        "hands": [unit(value) for value in action.get("hands", [])],
        # Both spellings parse to the same no-op Action in FastEnv.
        # Zero-quantity market markers are execution no-ops. Across the 32-seed
        # Thomas audit, all 29 such first differences produced identical full
        # post-step observations and did not alter the later semantic action
        # stream (the next real differences were at steps 187/207).
        "market": [list(value) for value in action.get("market", [])
                   if value and value[0] != "PASS" and
                   (len(value) < 3 or int(value[2]) != 0)],
    }


def python_trace(seed: int, seat: int, steps: int, final: bool):
    module = load_thomas(f"thomas_prefix_{seed}_{seat}_{int(final)}")
    policy = module.agent if final else module._IMPL
    env = FastEnv(Config(), seed)
    observations = list(env.reset(seed))
    trace = []
    shops = ()
    for step in range(steps):
        observation = observations[seat]
        observation["step"] = step
        observation["player"] = seat
        if step == 144:
            shops = tuple(observation["town"]["unlocked_shops"][:2])
        action = policy(observation, {})
        trace.append(action)
        actions = [PASS, PASS]
        actions[seat] = action
        observations = list(env.step(actions))
    state = module._IMPL.chassis.players.get(seat, {})
    return module, trace, int(state.get("route", 0)), shops, state.get("router_state", {}).get("rkey")


def native_trace(module, seed: int, seat: int, steps: int, route_id: int,
                 shops: tuple[str, ...], thomas: bool):
    route_ids = sorted(module._ROUTES)
    tapes = [module._ROUTES[key] for key in route_ids]
    pass_tape = [PASS for _ in range(719)]
    pass_index = len(tapes)
    names = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
             "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
    predict_pairs = None
    if thomas and len(shops) == 2:
        pair = names.index(shops[0]) * 8 + names.index(shops[1])
        predict_pairs = {pair: [
            [[step, item, quantity] for (step, item), quantity in events.items()]
            for _, events in module._v92_p_pair(shops)
        ]}
    executor = NativeTeammateExecutor(
        tapes + [pass_tape], pass_tape, pass_tape,
        [pass_tape] * 5, [pass_tape] * 5, predict_pairs,
    )
    opening = route_ids.index(0)
    target = route_ids.index(route_id)
    args = [opening, pass_index] if seat == 0 else [pass_index, opening]
    switches = [144, target, -1, -1] if seat == 0 else [-1, -1, 144, target]
    result = executor.play(
        *args, seed, *switches, capture_trace=True,
        neutral_special_economy=False, stop_after_steps=steps,
        thomas_prefix_player=seat if thomas else -2,
    )
    return [joint[seat] for joint in result["trace"]]


def first_difference(left: list[dict], right: list[dict]):
    for step, (expected, actual) in enumerate(zip(left, right)):
        if normalize(expected) != normalize(actual):
            return {"step": step, "python": expected, "native": actual}
    return None


def difference_type(diff: dict | None) -> str:
    if diff is None:
        return "none"
    expected, actual = diff["python"], diff["native"]
    changed = [key for key in ("farmer", "hands", "market")
               if normalize(expected)[key] != normalize(actual)[key]]
    if changed != ["market"]:
        return "+".join(changed)
    left = expected.get("market", [])
    right = actual.get("market", [])
    if not left and right and all(len(order) >= 3 and int(order[2]) == 0 for order in right):
        return "market:extra_zero_markers"
    if (any(order and order[0] == "HIRE" for order in left) and
            any(order[:2] == ["BUY_ANIMAL", "SHEEP"] for order in left) and
            not any(order and order[0] == "HIRE" for order in right)):
        return "market:missing:sheep_commitment"
    signature = lambda order: (order[0], order[1] if len(order) > 1 else "")
    left_ops = Counter(signature(order) for order in left if order and order[0] != "PASS")
    right_ops = Counter(signature(order) for order in right if order and order[0] != "PASS")
    added = list((left_ops - right_ops).elements())
    removed = list((right_ops - left_ops).elements())
    if len(added) == len(removed) == 1 and added[0][0] == removed[0][0]:
        return f"market:{added[0][0].lower()}:{removed[0][1].lower()}->{added[0][1].lower()}"
    if len(added) == 1 and not removed:
        return f"market:missing:{added[0][0].lower()}:{added[0][1].lower()}"
    if len(removed) == 1 and not added:
        return f"market:extra:{removed[0][0].lower()}:{removed[0][1].lower()}"
    return "market:orders"


def evaluate_seed(seed: int, steps: int) -> dict:
    source = load_thomas(f"thomas_prefix_export_{seed}")
    payload = export_payload(source)
    rows = []
    for seat in (0, 1):
        base_module, base, route_id, shops, rkey = python_trace(seed, seat, steps, False)
        assert route_id == select_route(payload, shops, rkey)
        native = native_trace(base_module, seed, seat, steps, route_id, shops, False)
        thomas_native = native_trace(base_module, seed, seat, steps, route_id, shops, True)
        _, final, final_route_id, _, _ = python_trace(seed, seat, steps, True)
        assert route_id == final_route_id
        assert len(base) == len(final) == len(native) == len(thomas_native) == steps
        diff = first_difference(final, thomas_native)
        rows.append({
            "seat": seat,
            "shops": shops,
            "selected_route": route_id,
            "base_first_difference": first_difference(base, native),
            "final_first_difference": diff,
            "final_difference_type": difference_type(diff),
        })
    return {"seed": seed, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2609500000)
    parser.add_argument("--steps", type=int, default=288)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.steps <= 288:
        parser.error("--steps must be in 1..288")
    if args.seed_count < 1:
        parser.error("--seed-count must be positive")
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be in 1..32")

    source = load_thomas("thomas_prefix_export")
    payload = export_payload(source)
    assert len(payload["route_ids"]) == 41
    assert all(len(route) == 719 for route in payload["routes"].values())
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(payload, separators=(",", ":")) + "\n")

    seeds = list(range(args.seed, args.seed + args.seed_count))
    if args.workers == 1:
        games = [evaluate_seed(seed, args.steps) for seed in seeds]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, args.seed_count)) as pool:
            games = list(pool.map(evaluate_seed, seeds, [args.steps] * len(seeds)))
    rows = [row for game in games for row in game["rows"]]
    route_counts = Counter(str(row["selected_route"]) for row in rows)
    step_counts = Counter("none" if row["final_first_difference"] is None else
                          str(row["final_first_difference"]["step"]) for row in rows)
    type_counts = Counter(row["final_difference_type"] for row in rows)
    route9_rows = [row for row in rows if row["selected_route"] == 9]
    wool_rows = type_counts["market:missing:sell:wool"] + type_counts["market:sell:wheat->wool"]
    result = {
        "schema": "thomas-native-prefix-parity-batch-v1",
        "seed_start": args.seed,
        "seed_count": args.seed_count,
        "steps": args.steps,
        "both_seats": True,
        "routes": len(payload["route_ids"]),
        "export": str(args.export) if args.export else None,
        "summary": {
            "seat_runs": len(rows),
            "route_counts": dict(sorted(route_counts.items())),
            "route9_seed_count": sum(any(row["selected_route"] == 9 for row in game["rows"])
                                     for game in games),
            "route9_seat_runs": len(route9_rows),
            "route9_seat_fraction": len(route9_rows) / len(rows),
            "route9_exact_seat_runs": sum(row["final_first_difference"] is None
                                          for row in route9_rows),
            "exact_seed_count": sum(all(row["final_first_difference"] is None
                                        for row in game["rows"]) for game in games),
            "exact_seat_runs": step_counts["none"],
            "first_difference_step_counts": dict(sorted(step_counts.items())),
            "difference_type_counts": dict(sorted(type_counts.items())),
            "wool_sale_difference_seat_runs": wool_rows,
            "wool_sale_difference_fraction": wool_rows / len(rows),
        },
        "games": games,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
