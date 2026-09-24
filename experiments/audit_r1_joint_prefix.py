#!/usr/bin/env python3
"""Read-only warm-prefix audit of deployed R1 normal/joint choices."""
import argparse
import functools
import inspect
import json
import os
from pathlib import Path

from audit_r1_greedy_regret import ROOT, load, observation


def run(binary, bot, seed, seat, stop_day):
    os.environ["R1_BINARY_PATH"] = str(binary)
    os.environ.pop("R1_CONFIG_OVERRIDES", None)
    main = load(ROOT / "agent/main.py", f"joint_main_{seed}_{seat}")
    policy = main.create_agent(seat)
    rival = load(ROOT / f"opponents/{bot}/main.py", f"joint_bot_{seed}_{seat}").agent
    with_config = len(inspect.signature(rival).parameters) > 1
    from fast_kaggriculture import Config, FastEnv
    env = FastEnv(Config(), seed)
    state = list(env.reset(seed))
    days = []
    try:
        while not env.done:
            obs = [observation(state, p) for p in (0, 1)]
            own, other = obs[seat], obs[1-seat]
            if own["day"] >= stop_day:
                break
            own_action = policy(own, {})
            if own["hour"] == 0 and policy.ready(own):
                debug = policy.dynamic.debug()
                days.append({"day": own["day"], "normal_choices": debug.get("search_choices"),
                             "joint_searches": debug.get("joint_searches"),
                             "joint_switches": debug.get("joint_switches"),
                             "joint_rejected": debug.get("joint_rejected"),
                             "joint_gain": debug.get("joint_gain"),
                             "joint_last": debug.get("joint_last")})
            other_action = rival(other, {}) if with_config else rival(other)
            actions = [None, None]
            actions[seat], actions[1-seat] = own_action, other_action
            state = env.step(actions)
        if not days:
            raise RuntimeError("no dynamic day observed")
        return {"bot": bot, "seed": seed, "seat": seat, "stop_day": stop_day,
                "handoff_day": days[0]["day"], "days": days}
    finally:
        policy.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--opponent", default="thomas_2945")
    p.add_argument("--start", type=int, default=2609800100)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--seat", type=int, default=0)
    p.add_argument("--stop-day", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error(f"refusing to overwrite {args.output}")
    from meta_agent.src import teammate_expanded_routes
    teammate_expanded_routes.load_action_tapes = functools.lru_cache(maxsize=1)(
        teammate_expanded_routes.load_action_tapes)
    rows = [run(args.binary, args.opponent, args.start + i, args.seat, args.stop_day)
            for i in range(args.seeds)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": "deployed-online-prefix-only", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
