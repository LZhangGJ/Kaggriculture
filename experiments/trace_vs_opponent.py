#!/usr/bin/env python3
"""Per-day trace of one policy against one opponent.

Records both farms at every day boundary so a loss can be localised in time
(pre-handoff deficit vs post-handoff collapse) instead of only being scored at
the end.  Same-seed, both-seat, one game per child process, as in run_strong_ab.
"""
import argparse
import concurrent.futures as cf
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path

from run_strong_ab import BOTS, entry, load

ROOT = Path(__file__).resolve().parent


def snapshot(observation, player, sold):
    farm = observation["farms"][player]
    tiles = [tile for row in farm["tiles"] for tile in row]
    private = observation["private"]
    shed = private.get("shed", {}) or {}
    inventories = private.get("inventories", []) or []
    prices = (observation.get("market", {}) or {}).get("prices", {}) or {}
    inventory = (observation.get("market", {}) or {}).get("inventory", {}) or {}
    def kinds(field):
        result = {}
        for tile in tiles:
            if isinstance(tile, dict) and tile.get(field):
                kind = tile[field]
                result[kind] = result.get(kind, 0) + 1
        return result
    return {
        "money": float(farm["money"]),
        "land": len(farm["unlocked_quadrants"]),
        "hands": len(farm["hands"]),
        "animals": sum(1 for t in tiles if isinstance(t, dict) and t.get("animal")),
        "crops": sum(1 for t in tiles if isinstance(t, dict) and t.get("crop")),
        "weeds": sum(1 for t in tiles if t == "WEED" or (isinstance(t, dict) and t.get("kind") == "WEED")),
        "empty": sum(1 for t in tiles if t == "EMPTY" or (isinstance(t, dict) and t.get("kind") in (None, "EMPTY"))),
        "shed": int(sum(float(v or 0) for v in shed.values())),
        "shed_by_item": {k: int(v or 0) for k, v in shed.items() if v},
        "shed_value": sum(float(v or 0) * float(prices.get(k, 0) or 0) for k, v in shed.items()),
        "sold": dict(sold),
        "sold_total": sum(sold.values()),
        "carried": int(sum(float(v or 0) for bag in inventories for v in (bag or {}).values())),
        "hires_today": int(farm.get("hires_today", 0) or 0),
        "unfed": sum(1 for t in tiles if isinstance(t, dict) and int(t.get("consecutive_unfed", 0) or 0) > 0),
        "unwatered": sum(1 for t in tiles if isinstance(t, dict) and int(t.get("consecutive_unwatered", 0) or 0) > 0),
        "crop_yield": int(sum(float(t.get("yield_units", 0) or 0) for t in tiles if isinstance(t, dict) and t.get("crop"))),
        "crops_by_kind": kinds("crop"),
        "animals_by_kind": kinds("animal"),
        "market_inventory": {k: int(v or 0) for k, v in inventory.items()},
        "market_prices": {k: int(v or 0) for k, v in prices.items()},
        "shops": list((observation.get("town", {}) or {}).get("unlocked_shops", []) or []),
    }


def add_sales(total, observation, action):
    """Count executable SELL units; selling is limited only by current shed stock."""
    left = dict(((observation.get("private", {}) or {}).get("shed", {}) or {}))
    for atom in (action or {}).get("market", []):
        if atom and atom[0] == "SELL" and len(atom) > 1:
            item = atom[1]
            quantity = min(int(atom[2] if len(atom) > 2 else 1), int(left.get(item, 0) or 0))
            if quantity > 0:
                total[item] = total.get(item, 0) + quantity
                left[item] = int(left.get(item, 0) or 0) - quantity


def play(task):
    policy_path, opponent, bot_path, seed, seat, opening, days, engine = task
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    if opening:
        os.environ["REPLAY_FORCED_OPENING"] = opening
    module = load(policy_path, f"trace_{opponent}_{seed}_{seat}")
    policy = module.create_agent(seat) if hasattr(module, "create_agent") else module.agent
    rival = load(bot_path, f"rival_{opponent}_{seed}_{seat}").agent
    with_config = len(inspect.signature(rival).parameters) > 1
    if engine == "fast":
        from fast_kaggriculture import Config, FastEnv
        env = FastEnv(Config(), seed)
        state = list(env.reset(seed))
        configuration = {}
    else:
        from kaggle_environments import make
        env = make("kaggriculture", configuration={"seed": seed}, debug=True)
        state = env.reset()
        configuration = env.configuration
    daily = []
    sold = [{}, {}]
    market_failures = []
    land_events = []
    error = None
    try:
        while not env.done:
            observations = []
            for player in (0, 1):
                raw = state[player] if engine == "fast" else state[player].observation
                obs = json.loads(json.dumps(raw))
                obs["step"] = obs.get("step", obs.get("day", 0) * 24 + obs.get("hour", 0))
                obs["player"] = player
                observations.append(obs)
            if observations[0]["step"] % 24 == 12 and observations[0]["step"] // 24 in days:
                daily.append({
                    "day": observations[0]["step"] // 24,
                    "hour": observations[0]["step"] % 24,
                    "own": snapshot(observations[seat], seat, sold[seat]),
                    "rival": snapshot(observations[1 - seat], 1 - seat, sold[1 - seat]),
                })
            actions = [policy(obs, configuration) if p == seat else
                       (rival(obs, configuration) if with_config else rival(obs))
                       for p, obs in enumerate(observations)]
            for player in (0, 1):
                add_sales(sold[player], observations[player], actions[player])
            step = observations[0]["step"]
            state = env.step(actions)
            if engine == "fast":
                fills = env.last_market_fills[seat]
                shortfalls = env.last_market_cash_shortfalls[seat]
                for slot, order in enumerate((actions[seat] or {}).get("market", [])):
                    if not order or order[0] not in {
                            "HIRE", "BUY_LAND", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT"}:
                        continue
                    requested = 1 if order[0] in {"HIRE", "BUY_LAND"} else max(
                        0, int(order[2] if len(order) > 2 else 1))
                    filled = int(fills[slot]) if slot < len(fills) else 0
                    if order[0] == "BUY_LAND":
                        land_events.append({
                            "step": step, "filled": filled,
                            "cash_shortfall": (float(shortfalls[slot])
                                               if slot < len(shortfalls) else 0.0),
                            "money_before": float(observations[seat]["farms"][seat]["money"]),
                            "land_before": len(observations[seat]["farms"][seat]["unlocked_quadrants"]),
                            "market": list((actions[seat] or {}).get("market", [])),
                        })
                    if filled < requested:
                        market_failures.append({
                            "step": step, "op": order[0],
                            "item": order[1] if len(order) > 1 else None,
                            "requested": requested, "filled": filled,
                            "cash_shortfall": (float(shortfalls[slot])
                                               if slot < len(shortfalls) else 0.0),
                            "money_before": float(observations[seat]["farms"][seat]["money"]),
                            "shed_before": dict((observations[seat].get("private") or {}).get("shed") or {}),
                            "market": list((actions[seat] or {}).get("market", [])),
                        })
        debug = policy.debug() if hasattr(policy, "debug") else {}
        if engine == "fast":
            own, other = map(float, (env.rewards[seat], env.rewards[1 - seat]))
        else:
            final = json.loads(json.dumps(state[0].observation))["farms"]
            own, other = final[seat]["money"], final[1 - seat]["money"]
        return {"opponent": opponent, "seed": seed, "seat": seat, "opening_override": opening,
                "own": own, "rival": other,
                "margin": own - other, "daily": daily,
                "handoff_step": debug.get("dynamic_handoff_step"),
                "handoff_land": debug.get("dynamic_handoff_land"),
                "replay_opening": debug.get("replay_opening"),
                "replay_current": debug.get("replay_current"),
                "replay_switched": debug.get("replay_switched"),
                "replay_switch_step": debug.get("replay_switch_step"),
                "market_failures": market_failures, "land_events": land_events, "error": None}
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return {"opponent": opponent, "seed": seed, "seat": seat, "opening_override": opening,
                "daily": daily, "error": repr(exc)}
    finally:
        close = getattr(policy, "close", None)
        if close:
            try:
                close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=entry, default=ROOT.parent / "agent" / "main.py")
    parser.add_argument("--opponents", default="thomas_2945,melon_2749,demand_preserving")
    parser.add_argument("--start", type=int, default=2609600000)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--engine", choices=("official", "fast"), default="official")
    parser.add_argument("--days", default="11,15,20,25,29")
    parser.add_argument("--openings", help="comma-separated route families to force")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [name.strip() for name in args.opponents.split(",") if name.strip()]
    unknown = [name for name in names if name not in BOTS]
    if unknown:
        parser.error(f"unknown opponents: {unknown}")
    if args.seeds < 1:
        parser.error("seeds must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    openings = [value.strip() for value in (args.openings or "").split(",") if value.strip()] or [None]
    days = {int(value) for value in args.days.split(",") if value.strip()}
    if not days or min(days) < 0 or max(days) > 29:
        parser.error("days must be comma-separated values in 0..29")
    tasks = [(str(args.policy), name, BOTS[name], seed, seat, opening, days, args.engine)
             for opening in openings for name in names
             for seed in range(args.start, args.start + args.seeds) for seat in (0, 1)]
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                max_tasks_per_child=1) as pool:
        rows = list(pool.map(play, tasks))
    result = {
        "policy": str(args.policy), "engine": args.engine, "opening_overrides": openings,
        "seed_range": [args.start, args.start + args.seeds], "both_seats": True,
        "snapshot_days": sorted(days), "snapshot_hour": 12,
        "opponents": {name: BOTS[name] for name in names}, "games": len(rows), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name in names:
        valid = [r for r in rows if r["opponent"] == name and not r["error"]]
        wins = sum(1 for r in valid if r["margin"] > 0)
        print(json.dumps({"opponent": name, "games": len(valid), "wins": wins,
                          "errors": sum(1 for r in rows if r["opponent"] == name) - len(valid),
                          "mean_margin": sum(r["margin"] for r in valid) / len(valid) if valid else None}))
    if len(openings) > 1:
        by_opening = {}
        for row in rows:
            if row["error"]:
                continue
            entry_ = by_opening.setdefault(row.get("opening_override"), {"wins": 0, "games": 0, "margin": 0.0})
            entry_["wins"] += int(row["margin"] > 0)
            entry_["games"] += 1
            entry_["margin"] += row["margin"]
        print(json.dumps(sorted(((k, v["wins"], v["games"], v["margin"] / v["games"])
                                 for k, v in by_opening.items()),
                                key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
