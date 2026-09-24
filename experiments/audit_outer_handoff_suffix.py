#!/usr/bin/env python3
"""Paired FastEnv suffix audit of the best outer handoff bundle."""
import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import inspect
import json
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observation(state, player):
    value = json.loads(json.dumps(state[player]))
    value["step"] = value.get("day", 0) * 24 + value.get("hour", 0)
    value["player"] = player
    return value


def board(obs, seat):
    farm = obs["farms"][seat]
    tiles = [tile for row in farm["tiles"] for tile in row]
    return {"money": farm["money"], "rival_money": obs["farms"][1 - seat]["money"],
            "land": len(farm["unlocked_quadrants"]),
            "crops": sum(isinstance(tile, dict) and bool(tile.get("crop")) for tile in tiles),
            "animals": sum(isinstance(tile, dict) and bool(tile.get("animal")) for tile in tiles)}


def farm_snapshot(farm):
    crops, crop_yield, unwatered = Counter(), Counter(), Counter()
    animals, animal_yield, animal_cohorts, unfed, structures = Counter(), Counter(), Counter(), Counter(), Counter()
    empty = weeds = 0
    for row in farm["tiles"]:
        for tile in row:
            if tile is None:
                empty += 1
            elif isinstance(tile, dict):
                kind = tile.get("kind")
                crop, animal = tile.get("crop"), tile.get("animal")
                if kind:
                    structures[kind] += 1
                if kind == "WEED":
                    weeds += 1
                if crop:
                    crops[crop] += 1
                    crop_yield[crop] += tile.get("yield_units", 0)
                    unwatered[crop] += int(tile.get("consecutive_unwatered", 0) > 0)
                if animal:
                    animals[animal] += 1
                    animal_yield[animal] += tile.get("yield_units", 0)
                    animal_cohorts[f"{animal}@{tile.get('placed_day', -1)}"] += 1
                    unfed[animal] += int(tile.get("consecutive_unfed", 0) > 0)
    return {"money": farm["money"], "land": len(farm["unlocked_quadrants"]),
            "hands": len(farm.get("hands", [])), "empty": empty, "weeds": weeds,
            "crops": dict(crops), "crop_yield": dict(crop_yield), "unwatered": dict(unwatered),
            "animals": dict(animals), "animal_yield": dict(animal_yield),
            "animal_cohorts": dict(animal_cohorts), "unfed": dict(unfed),
            "structures": dict(structures)}


def daily_snapshot(obs, seat):
    private = obs.get("private", {})
    carried = Counter()
    for inventory in private.get("inventories", []):
        carried.update(inventory)
    market = obs.get("market", {})
    return {"step": obs["step"], "day": obs["day"], "hour": obs["hour"],
            "own": farm_snapshot(obs["farms"][seat]),
            "rival": farm_snapshot(obs["farms"][1 - seat]),
            "shed": dict(private.get("shed", {})), "carried": dict(carried),
            "seeds": dict(private.get("seeds", {})),
            "market_inventory": dict(market.get("inventory", {})),
            "market_prices": dict(market.get("prices", {}))}


def closed_loop_candidates(rows, label, limit):
    """Keep the baseline shortlist nested inside the outer shortlist."""
    normal = sorted((row for row in rows if not row.get("diagnostic")),
                    key=lambda row: row["score"], reverse=True)[:limit]
    if label == "baseline":
        return normal
    outer = sorted((row for row in rows if row.get("diagnostic", "").startswith("outer")),
                   key=lambda row: row["score"], reverse=True)[:limit]
    return list({row["index"]: row for row in normal + outer}.values())


def run(task):
    label, binary, bot, seed, seat, closed_loop_top = task
    os.environ["R1_BINARY_PATH"] = binary
    os.environ.pop("R1_CONFIG_OVERRIDES", None)
    main = load(ROOT / "agent/main.py", f"outer_suffix_{label}_{bot}_{seed}_{seat}")
    policy = main.create_agent(seat)
    rival = load(ROOT / f"opponents/{bot}/main.py", f"outer_suffix_rival_{label}_{bot}_{seed}_{seat}").agent
    with_config = len(inspect.signature(rival).parameters) > 1
    from fast_kaggriculture import Config, FastEnv
    env = FastEnv(Config(), seed)
    state = list(env.reset(seed))
    chosen = handoff_hash = post = None
    handoff_day = None
    trace = []
    market_orders = []
    try:
        while not env.done:
            obs = [observation(state, player) for player in (0, 1)]
            own = obs[seat]
            if chosen is None and policy.ready(own):
                rows = policy.dynamic.prepare_candidates(own)
                allowed = [row for row in rows if not row.get("diagnostic")]
                if label == "outer":
                    allowed += [row for row in rows if row.get("diagnostic", "").startswith("outer")]
                if closed_loop_top:
                    if not policy.dynamic._has_extended_horizon:
                        raise RuntimeError("--closed-loop-top requires a rebuilt diagnostic binary")
                    selected = closed_loop_candidates(rows, label, closed_loop_top)
                    for row in selected:
                        row["closed_loop_terminal"] = policy.dynamic.candidate_score_horizon(
                            own, row["index"], 30 - own["day"])
                    chosen = max(selected, key=lambda row: row["closed_loop_terminal"])
                else:
                    chosen = max(allowed, key=lambda row: row["score"])
                if policy.dynamic._has_extended_horizon:
                    chosen["closed_loop_scores"] = {str(days): policy.dynamic.candidate_score_horizon(
                        own, chosen["index"], days) for days in (1, 3, 5, 10, 18)}
                    chosen.setdefault("closed_loop_terminal", chosen["closed_loop_scores"]["18"])
                else:
                    chosen["closed_loop_scores"], chosen["closed_loop_terminal"] = {}, None
                policy.dynamic.install_candidate(chosen["index"])
                handoff_hash = hashlib.sha256(json.dumps(own, sort_keys=True).encode()).hexdigest()
                handoff_day = own["day"]
                trace.append(daily_snapshot(own, seat))
            if chosen is not None and post is None and own["day"] > handoff_day:
                post = board(own, seat)
            if chosen is not None and own["day"] > trace[-1]["day"]:
                trace.append(daily_snapshot(own, seat))
            actions = [None, None]
            actions[seat] = policy(own, {})
            if chosen is not None and own["hour"] == 0 and trace[-1]["day"] == own["day"]:
                trace[-1]["clock"] = policy.dynamic.clock_debug()
            other = obs[1 - seat]
            actions[1 - seat] = rival(other, {}) if with_config else rival(other)
            if chosen is not None:
                for player, side in ((seat, "own"), (1 - seat, "rival")):
                    for order in actions[player].get("market", []):
                        if len(order) < 2:
                            continue
                        item = order[1] if order[0] in ("SELL", "BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL") else None
                        market_orders.append({"step": own["step"], "side": side, "order": order,
                                              "inventory": own.get("market", {}).get("inventory", {}).get(item),
                                              "price": own.get("market", {}).get("prices", {}).get(item)})
            state = env.step(actions)
        if chosen is None:
            raise RuntimeError("handoff was not reached")
        terminal = daily_snapshot(observation(state, seat), seat)
        own, other = map(float, (env.rewards[seat], env.rewards[1 - seat]))
        return {"label": label, "bot": bot, "seed": seed, "seat": seat,
                "handoff_hash": handoff_hash, "chosen": {key: chosen[key] for key in
                    ("index", "id", "diagnostic", "score", "predicted",
                     "scenario_rollout_own_cash", "scenario_rollout_rival_cash",
                     "closed_loop_scores", "closed_loop_terminal")}, "post": post,
                "trace": trace, "market_orders": market_orders, "terminal": terminal,
                "cash": own, "opponent_cash": other,
                "margin": own - other, "error": None}
    except Exception as exc:
        return {"label": label, "bot": bot, "seed": seed, "seat": seat, "error": repr(exc)}
    finally:
        policy.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--outer", type=Path, required=True)
    parser.add_argument("--opponents", default="thomas_2945,melon_2749")
    parser.add_argument("--start", type=int, default=2609800100)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seats", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--closed-loop-top", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.closed_loop_top < 0:
        parser.error("--closed-loop-top must be non-negative")
    bots = [value for value in args.opponents.split(",") if value]
    seats = [int(value) for value in args.seats.split(",") if value]
    tasks = [(label, str(binary.resolve()), bot, seed, seat, args.closed_loop_top)
             for bot in bots for seed in range(args.start, args.start + args.seeds) for seat in seats
             for label, binary in (("baseline", args.baseline), ("outer", args.outer))]
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        rows = list(pool.map(run, tasks))
    pairs = []
    for bot in bots:
        for seed in range(args.start, args.start + args.seeds):
            for seat in seats:
                values = [row for row in rows if (row["bot"], row["seed"], row["seat"]) == (bot, seed, seat)]
                if len(values) != 2 or any(row["error"] for row in values):
                    raise RuntimeError(f"incomplete pair: {(bot, seed, seat, values)}")
                arms = {row["label"]: row for row in values}
                if arms["baseline"]["handoff_hash"] != arms["outer"]["handoff_hash"]:
                    raise RuntimeError(f"handoff mismatch: {(bot, seed, seat)}")
                pairs.append({"bot": bot, "seed": seed, "seat": seat,
                              "margin_delta": arms["outer"]["margin"] - arms["baseline"]["margin"],
                              "win_delta": int(arms["outer"]["margin"] > 0) - int(arms["baseline"]["margin"] > 0),
                              "baseline_choice": arms["baseline"]["chosen"], "outer_choice": arms["outer"]["chosen"]})
    result = {"scope": "same warm prefix; forced handoff candidate; FastEnv suffix",
              "binaries": {"baseline": str(args.baseline), "outer": str(args.outer)},
              "seed_range": [args.start, args.start + args.seeds], "pairs": pairs, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pairs": len(pairs), "win_delta": sum(row["win_delta"] for row in pairs),
                      "margin_delta": sum(row["margin_delta"] for row in pairs), "output": str(args.output)}))


if __name__ == "__main__":
    main()
