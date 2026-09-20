#!/usr/bin/env python3
"""Force each R1 proposal for one day and measure its real official suffix."""
import argparse
import concurrent.futures as cf
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path

from run_strong_ab import BOTS, load

ROOT = Path(__file__).resolve().parents[1]


def board(observation, seat):
    farm = observation["farms"][seat]
    tiles = [tile for row in farm["tiles"] for tile in row]
    return {"money": farm["money"], "land": len(farm["unlocked_quadrants"]),
            "crops": sum(isinstance(t, dict) and bool(t.get("crop")) for t in tiles),
            "animals": sum(isinstance(t, dict) and bool(t.get("animal")) for t in tiles),
            "weeds": sum(t == "WEED" or isinstance(t, dict) and t.get("kind") == "WEED" for t in tiles)}


def run(task):
    bot, seed, seat, target, candidate, *options = task
    with_features = bool(options and options[0])
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    hybrid = load(ROOT / "agent/main.py", f"oracle_{bot}_{seed}_{seat}_{target}_{candidate}")
    policy = hybrid.create_agent(seat)
    rival = load(BOTS[bot], f"oracle_rival_{bot}_{seed}_{seat}_{target}_{candidate}").agent
    with_config = len(inspect.signature(rival).parameters) > 1
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    state = env.reset()
    chosen = post = execution = auto_debug = None
    actual_day = None
    try:
        while not env.done:
            observations = []
            for player in (0, 1):
                obs = json.loads(json.dumps(state[player].observation))
                obs["step"] = obs.get("day", 0) * 24 + obs.get("hour", 0)
                obs["player"] = player
                observations.append(obs)
            step = observations[0]["step"]
            trigger = (target == "handoff" and policy.ready(observations[seat])) or \
                      (target != "handoff" and step == int(target) * 24)
            capture_auto = False
            if chosen is None and trigger:
                actual_day = observations[seat].get("day", step // 24)
                proposals = policy.dynamic.prepare_candidates(observations[seat], with_features)
                if candidate is None:
                    return {"bot": bot, "seed": seed, "seat": seat, "target": target,
                            "day": actual_day,
                            "proposals": proposals, "error": None}
                if candidate == "auto":
                    chosen = max((row for row in proposals if not row.get("diagnostic")),
                                 key=lambda row: row["score"])
                    capture_auto = True
                else:
                    chosen = proposals[candidate]
                    policy.dynamic.install_candidate(candidate)
            if candidate is not None and actual_day is not None and step == (actual_day + 1) * 24:
                post = board(observations[seat], seat)
                execution = policy.dynamic.obligations(observations[seat])
            actions = [policy(observations[p], env.configuration) if p == seat else
                       (rival(observations[p], env.configuration) if with_config else rival(observations[p]))
                       for p in (0, 1)]
            if capture_auto:
                auto_debug = policy.dynamic.debug()
            state = env.step(actions)
        farms = json.loads(json.dumps(state[0].observation))["farms"]
        own, other = farms[seat]["money"], farms[1 - seat]["money"]
        if candidate is not None and chosen is None:
            raise RuntimeError(f"decision target not reached: {target}")
        return {"bot": bot, "seed": seed, "seat": seat, "target": target, "day": actual_day,
                "variant": "auto" if candidate == "auto" else chosen.get("diagnostic"),
                "candidate": chosen, "auto_debug": auto_debug, "post": post,
                "execution": execution, "margin": own - other, "error": None}
    except Exception as exc:
        return {"bot": bot, "seed": seed, "seat": seat, "target": target,
                "candidate_index": candidate, "error": repr(exc)}
    finally:
        policy.close()


def cases_from_trace(path):
    rows = json.loads(path.read_text())["rows"]
    cases = []
    for bot in ("thomas_2945", "melon_2749"):
        selected = [row for row in rows if row["opponent"] == bot and row["seat"] == 0]
        for win in (True, False):
            matches = [row for row in selected if (row["margin"] > 0) == win][:2]
            cases.extend((bot, row["seed"], row["seat"], "win" if win else "loss") for row in matches)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=ROOT / "work/trace-current-delay1-thomas-melon-8seed-2609800000.json")
    parser.add_argument("--days", default="11,15")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    days = [int(value) for value in args.days.split(",")]
    cases = cases_from_trace(args.trace)
    context = mp.get_context("spawn")
    discovery_tasks = [(bot, seed, seat, day, None) for bot, seed, seat, _ in cases for day in days]
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=context, max_tasks_per_child=1) as pool:
        discovered = list(pool.map(run, discovery_tasks))
    labels = {(bot, seed, seat): label for bot, seed, seat, label in cases}
    tasks = [(row["bot"], row["seed"], row["seat"], row["day"], proposal["index"])
             for row in discovered if not row["error"] for proposal in row["proposals"]]
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=context, max_tasks_per_child=1) as pool:
        rows = list(pool.map(run, tasks))
    for row in rows:
        row["source_outcome"] = labels[(row["bot"], row["seed"], row["seat"])]
    groups = []
    for discovered_row in discovered:
        key = tuple(discovered_row[name] for name in ("bot", "seed", "seat", "day"))
        values = [row for row in rows if tuple(row[name] for name in ("bot", "seed", "seat", "day")) == key and not row["error"]]
        if not values:
            continue
        predicted = max(values, key=lambda row: row["candidate"]["score"])
        actual = max(values, key=lambda row: row["margin"])
        groups.append({"bot": key[0], "seed": key[1], "seat": key[2], "day": key[3],
                       "source_outcome": labels[key[:3]], "predicted_index": predicted["candidate"]["index"],
                       "actual_index": actual["candidate"]["index"], "top_match": predicted is actual,
                       "regret": actual["margin"] - predicted["margin"]})
    result = {"trace": str(args.trace), "days": days, "cases": cases, "discovered": discovered,
              "games": len(rows), "errors": sum(bool(row["error"]) for row in rows),
              "groups": groups, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "games": len(rows), "errors": result["errors"],
                      "groups": len(groups), "top_matches": sum(row["top_match"] for row in groups),
                      "mean_regret": sum(row["regret"] for row in groups) / len(groups)}))


if __name__ == "__main__":
    main()
