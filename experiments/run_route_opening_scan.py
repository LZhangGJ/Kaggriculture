#!/usr/bin/env python3
"""Compare fixed opening families with their existing replay-switch trees."""
import argparse
import concurrent.futures as cf
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path

from run_strong_ab import BOTS, load

ROOT = Path(__file__).resolve().parent
POLICY = ROOT.parent / "agent" / "main.py"


def play(task):
    opening, bot, bot_path, seed, seat, disable_switch, handoff_step = task
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["REPLAY_FORCED_OPENING"] = opening
    os.environ["REPLAY_HANDOFF_STEP"] = str(handoff_step)
    from kaggle_environments import make

    policy = load(POLICY, f"route_{opening}_{bot}_{seed}_{seat}").create_agent()
    if disable_switch:
        policy.replay.controller.nodes = {}
    opponent = load(bot_path, f"opponent_{opening}_{bot}_{seed}_{seat}").agent
    with_config = len(inspect.signature(opponent).parameters) > 1
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    state = env.reset()
    try:
        while not env.done:
            actions = []
            for player in (0, 1):
                obs = json.loads(json.dumps(state[player].observation))
                obs["step"] = obs.get("step", obs.get("day", 0) * 24 + obs.get("hour", 0))
                obs["player"] = player
                actions.append(policy(obs, env.configuration) if player == seat else
                               opponent(obs, env.configuration) if with_config else opponent(obs))
            state = env.step(actions)
        farms = json.loads(json.dumps(state[0].observation))["farms"]
        own, rival = farms[seat]["money"], farms[1 - seat]["money"]
        return {"opening": opening, "bot": bot, "seed": seed, "seat": seat,
                "cash": own, "opponent_cash": rival, "margin": own - rival, "error": None}
    except Exception as exc:
        return {"opening": opening, "bot": bot, "seed": seed, "seat": seat, "error": repr(exc)}
    finally:
        policy.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openings", default="G001,G210,G275,G379,G411")
    parser.add_argument("--start", type=int, default=2609500000)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--workers", type=int, default=192)
    parser.add_argument("--disable-switch", action="store_true")
    parser.add_argument("--handoff-day", type=int, default=None,
                        help="hand the game to the dynamic policy at this day "
                             "(default: never, i.e. the route tree plays the whole game)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    openings = args.openings.split(",")
    if len(openings) != len(set(openings)) or args.seeds < 1:
        parser.error("openings must be unique and seeds positive")
    if args.handoff_day is not None and not 0 <= args.handoff_day <= 30:
        parser.error("handoff-day must be in 0..30")
    handoff_step = 720 if args.handoff_day is None else args.handoff_day * 24
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    tasks = [(opening, bot, path, seed, seat, args.disable_switch, handoff_step)
             for opening in openings for bot, path in BOTS.items()
             for seed in range(args.start, args.start + args.seeds) for seat in (0, 1)]
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                max_tasks_per_child=1) as pool:
        rows = list(pool.map(play, tasks))
    summary = {}
    for opening in openings:
        selected = [r for r in rows if r["opening"] == opening]
        valid = [r for r in selected if not r["error"]]
        by_bot = {}
        for bot in BOTS:
            values = [r for r in valid if r["bot"] == bot]
            by_bot[bot] = {"games": len(values), "wins": sum(r["margin"] > 0 for r in values),
                           "mean_margin": sum(r["margin"] for r in values) / len(values) if values else None}
        summary[opening] = {"games": len(valid), "wins": sum(r["margin"] > 0 for r in valid),
                            "mean_margin": sum(r["margin"] for r in valid) / len(valid) if valid else None,
                            "errors": len(selected) - len(valid), "by_bot": by_bot}
    result = {"seed_range": [args.start, args.start + args.seeds], "both_seats": True,
              "openings": openings, "route_switch": not args.disable_switch,
              "handoff_day": args.handoff_day, "handoff_step": handoff_step,
              "opponents": BOTS, "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
