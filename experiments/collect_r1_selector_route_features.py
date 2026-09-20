#!/usr/bin/env python3
"""Re-run selector prefixes and collect aligned 147-D route-switch features."""
import argparse
import concurrent.futures as cf
import copy
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from meta_agent.src.route_switch_features import route_switch_feature_names, route_switch_vector
from run_strong_ab import BOTS, load

ROOT = Path(__file__).resolve().parents[1]


def source_rows(path, limit=None):
    with np.load(path, allow_pickle=False) as data:
        required = ("bots", "seeds", "seats", "days", "targets")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"selector NPZ missing keys: {missing}")
        arrays = {name: np.asarray(data[name]) for name in required}
    sizes = {len(value) for value in arrays.values()}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
        raise ValueError("selector row keys must have one shared non-zero length")
    count = next(iter(sizes)) if limit is None else min(next(iter(sizes)), limit)
    rows = [(index, str(arrays["bots"][index]), int(arrays["seeds"][index]),
             int(arrays["seats"][index]), str(arrays["targets"][index]),
             int(arrays["days"][index])) for index in range(count)]
    return rows, arrays


def collect(task):
    index, bot, seed, seat, target, expected_day = task
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    hybrid = load(ROOT / "agent/main.py", f"route_features_{index}_{seed}_{seat}")
    policy = hybrid.create_agent(seat)
    rival = load(BOTS[bot], f"route_features_rival_{index}_{seed}_{seat}").agent
    with_config = len(inspect.signature(rival).parameters) > 1
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    state = env.reset()
    try:
        while not env.done:
            observations = []
            for player in (0, 1):
                observation = json.loads(json.dumps(state[player].observation))
                observation["step"] = observation.get("day", 0) * 24 + observation.get("hour", 0)
                observation["player"] = player
                observations.append(observation)
            observation = observations[seat]
            step = observations[0]["step"]
            trigger = ((target == "handoff" and policy.ready(observation)) or
                       (target != "handoff" and step == int(target) * 24))
            if trigger:
                day = int(observation.get("day", step // 24))
                if day != expected_day:
                    raise ValueError(f"row {index}: trigger day {day} != selector day {expected_day}")
                controller = policy.replay.controller
                history = copy.deepcopy(controller.history)
                history.update(observation)
                route_id = controller.route_by_family[controller.current]
                route = policy.replay.expanded_agent.action_tapes[route_id]
                vector = route_switch_vector(observation, history, route)
                return index, step, controller.current, vector
            actions = [policy(observations[player], env.configuration) if player == seat else
                       (rival(observations[player], env.configuration) if with_config
                        else rival(observations[player])) for player in (0, 1)]
            state = env.step(actions)
        raise RuntimeError(f"row {index}: target {target} not reached")
    finally:
        policy.close()


def self_check():
    assert len(route_switch_feature_names()) == 147
    sample = {
        "bots": np.asarray(["thomas_2945", "melon_2749"]),
        "seeds": np.asarray([7, 8]), "seats": np.asarray([0, 1]),
        "days": np.asarray([11, 15]), "targets": np.asarray(["handoff", "15"]),
    }
    rows = [(i, str(sample["bots"][i]), int(sample["seeds"][i]),
             int(sample["seats"][i]), str(sample["targets"][i]), int(sample["days"][i]))
            for i in range(2)]
    assert rows == [(0, "thomas_2945", 7, 0, "handoff", 11),
                    (1, "melon_2749", 8, 1, "15", 15)]
    print(json.dumps({"status": "PASS", "feature_dim": 147, "rows": len(rows)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-npz", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.selector_npz or not args.selector_npz.is_file():
        parser.error("--selector-npz must be an existing NPZ")
    if not args.output or args.output.exists():
        parser.error("--output must be a new NPZ path")
    if not 1 <= args.workers <= 64 or args.limit is not None and args.limit < 1:
        parser.error("workers must be 1..64 and limit must be positive")
    rows, arrays = source_rows(args.selector_npz, args.limit)
    unknown = sorted({row[1] for row in rows} - set(BOTS))
    if unknown:
        parser.error(f"unknown bots: {unknown}")
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        results = list(pool.map(collect, rows))
    if [row[0] for row in results] != list(range(len(rows))):
        raise RuntimeError("worker results lost selector row alignment")
    features = np.asarray([row[3] for row in results], dtype=np.float32)
    if features.shape != (len(rows), 147) or not np.isfinite(features).all():
        raise RuntimeError(f"invalid route feature matrix: {features.shape}")
    count = len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, features=features,
        feature_names=np.asarray(route_switch_feature_names()),
        bots=arrays["bots"][:count], seeds=arrays["seeds"][:count],
        seats=arrays["seats"][:count], days=arrays["days"][:count],
        targets=arrays["targets"][:count],
        steps=np.asarray([row[1] for row in results], dtype=np.int16),
        route_families=np.asarray([row[2] for row in results]),
    )
    print(json.dumps({"output": str(args.output), "shape": list(features.shape),
                      "rows": count, "workers": args.workers}))


if __name__ == "__main__":
    main()
