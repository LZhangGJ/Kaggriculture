#!/usr/bin/env python3
"""Scan fixed replay-to-R1 handoff days against the seven strong bots."""
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
    day, bot, bot_path, seed, seat = task
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["REPLAY_FORCED_OPENING"] = "G001"
    os.environ["REPLAY_HANDOFF_STEP"] = str(day * 24)
    from kaggle_environments import make

    policy = load(POLICY, f"handoff_{day}_{bot}_{seed}_{seat}").create_agent()
    opponent = load(bot_path, f"opponent_{day}_{bot}_{seed}_{seat}").agent
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
        return {"day": day, "bot": bot, "seed": seed, "seat": seat,
                "cash": own, "opponent_cash": rival, "margin": own - rival, "error": None}
    except Exception as exc:
        return {"day": day, "bot": bot, "seed": seed, "seat": seat, "error": repr(exc)}
    finally:
        policy.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", default="6,7,9,10,12,15,18,21,24,27,30")
    parser.add_argument("--start", type=int, default=2609500000)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--workers", type=int, default=192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    days = [int(x) for x in args.days.split(",")]
    if len(days) != len(set(days)) or any(day < 0 or day > 30 for day in days):
        parser.error("days must be unique values in 0..30")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.seeds < 1:
        parser.error("seeds must be positive")
    tasks = [(day, bot, path, seed, seat) for day in days for bot, path in BOTS.items()
             for seed in range(args.start, args.start + args.seeds) for seat in (0, 1)]
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                max_tasks_per_child=1) as pool:
        rows = list(pool.map(play, tasks))
    summary = {}
    for day in days:
        selected = [r for r in rows if r["day"] == day]
        valid = [r for r in selected if not r["error"]]
        by_bot = {}
        for bot in BOTS:
            bot_rows = [r for r in valid if r["bot"] == bot]
            by_bot[bot] = {"games": len(bot_rows), "wins": sum(r["margin"] > 0 for r in bot_rows),
                           "mean_margin": sum(r["margin"] for r in bot_rows) / len(bot_rows) if bot_rows else None}
        summary[str(day)] = {"games": len(valid), "wins": sum(r["margin"] > 0 for r in valid),
                             "mean_margin": sum(r["margin"] for r in valid) / len(valid) if valid else None,
                             "errors": len(selected) - len(valid), "by_bot": by_bot}
    result = {"seed_range": [args.start, args.start + args.seeds], "both_seats": True, "days": days, "opponents": BOTS,
              "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
