#!/usr/bin/env python3
"""Measure which label actually predicts the value of a replay-route switch.

The shipped tree is labelled by *tape-vs-tape* payoff over the whole remaining
game (``NativeTeammateExecutor::switch_search`` -> ``play(opening, opponent,
seed, checkpoint, target, -1, -1)`` -- no dynamic policy anywhere).  The
deployment, however, runs the tape for only one or two more days and then hands
the board to the R1 dynamic policy.  So the tree optimises 18 days it never
plays.

This probe samples (opening, checkpoint, target, opponent, seed, seat) cells,
runs the **real hybrid** for each, and records:

* ``true_margin``  -- realised margin of the actual deployment configuration,
* ``tape_label``   -- the tape-vs-tape payoff the tree was trained on,
* ``board``        -- the day-11 board shape at the checkpoint.

It then reports how well each candidate label correlates with ``true_margin``
across cells, so the next training run can use a measured label instead of an
assumed one.
"""
import argparse
import concurrent.futures as cf
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))


def load(path, name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def board_shape(observation, seat):
    farm = observation["farms"][seat]
    tiles = [tile for row in farm["tiles"] for tile in row]
    return {
        "land": len(farm["unlocked_quadrants"]),
        "crops": sum(1 for t in tiles if isinstance(t, dict) and t.get("crop")),
        "animals": sum(1 for t in tiles if isinstance(t, dict) and t.get("animal")),
        "money": float(farm["money"]),
    }


def play(task):
    """One real hybrid game with a forced switch at `checkpoint` to `target`."""
    (opening, target, opponent, seed, seat, checkpoint,
     opening_index, target_index, opponent_index, seed_index) = task
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["REPLAY_FORCED_OPENING"] = opening
    os.environ.pop("REPLAY_HANDOFF_LAND", None)
    os.environ.pop("REPLAY_HANDOFF_MIN_STEP", None)
    os.environ["REPLAY_HANDOFF_STEP"] = "288"
    from kaggle_environments import make

    hybrid = load(ROOT / "agent/main.py", f"probe_{opening}_{target}_{seed}_{seat}")
    policy = hybrid.create_agent(seat)
    route = policy.replay
    route.controller.nodes = {}  # the forced schedule below is the intervention
    route.expanded_agent.select_schedule(
        ((0, route.controller.route_by_family[opening]),
         (checkpoint, route.controller.route_by_family[target]))
    )
    rival = load(ROOT / "opponents" / f"{opponent}" / "main.py", f"rival_{seed}_{seat}").agent
    with_config = len(inspect.signature(rival).parameters) > 1
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    state = env.reset()
    board = None
    error = None
    try:
        while not env.done:
            actions = []
            for player in (0, 1):
                obs = json.loads(json.dumps(state[player].observation))
                obs["step"] = obs.get("step", obs.get("day", 0) * 24 + obs.get("hour", 0))
                obs["player"] = player
                if player == seat and obs["step"] == checkpoint:
                    board = board_shape(obs, seat)
                actions.append(policy(obs, env.configuration) if player == seat else
                               (rival(obs, env.configuration) if with_config else rival(obs)))
            state = env.step(actions)
        farms = json.loads(json.dumps(state[0].observation))["farms"]
        own, other = farms[seat]["money"], farms[1 - seat]["money"]
        return {"opening": opening, "target": target, "opponent": opponent, "seed": seed,
                "seat": seat, "checkpoint": checkpoint, "true_margin": own - other,
                "own": own, "rival": other, "board": board,
                "opening_index": opening_index, "target_index": target_index,
                "opponent_index": opponent_index, "seed_index": seed_index, "error": None}
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return {"opening": opening, "target": target, "opponent": opponent, "seed": seed,
                "seat": seat, "checkpoint": checkpoint, "error": repr(exc)}
    finally:
        close = getattr(policy, "close", None)
        if close:
            try:
                close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openings", default="G001,G275,G379")
    parser.add_argument("--targets", default="G001,G210,G275,G379")
    parser.add_argument("--opponents", default="thomas_2945,melon_2749,demand_preserving,ahmed_v47")
    parser.add_argument("--checkpoints", default="240,264")
    parser.add_argument("--seeds", default="2609600000:2609600004")
    parser.add_argument("--label-npz", type=Path,
                        default=ROOT / "data/artifacts/switch-fine-late-240-264-26x128.npz")
    parser.add_argument("--start-seed", type=int, default=2609220000,
                        help="first seed of the label npz grid")
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    start, _, stop = args.seeds.partition(":")
    seed_values = list(range(int(start), int(stop))) if stop else [int(start)]
    openings = args.openings.split(",")
    targets = args.targets.split(",")
    opponents = args.opponents.split(",")
    checkpoints = [int(x) for x in args.checkpoints.split(",")]

    with np.load(args.label_npz, mmap_mode="r") as saved:
        npz_openings = saved["openings"].astype(str).tolist()
        npz_targets = saved["targets"].astype(str).tolist()
        npz_opponents = saved["opponents"].astype(str).tolist()
        npz_checkpoints = saved["checkpoints"].astype(int).tolist()
        npz_seeds = saved["seeds"].astype(np.int64).tolist()
        outcome = np.asarray(saved["outcome"], dtype=np.float32) * 0.5
        margin = np.asarray(saved["margin"], dtype=np.float32)

    tasks, keys = [], []
    for opening in openings:
        for target in targets:
            for opponent in opponents:
                for seed in seed_values:
                    if seed not in npz_seeds:
                        continue
                    for seat in (0, 1):
                        for checkpoint in checkpoints:
                            tasks.append((opening, target, opponent, seed, seat, checkpoint,
                                          npz_openings.index(opening), npz_targets.index(target),
                                          npz_opponents.index(opponent), npz_seeds.index(seed)))
                            keys.append((npz_openings.index(opening), npz_checkpoints.index(checkpoint),
                                         npz_targets.index(target), npz_opponents.index(opponent),
                                         npz_seeds.index(seed), seat))
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                max_tasks_per_child=1) as pool:
        rows = list(pool.map(play, tasks))
    for row, key in zip(rows, keys):
        if row["error"]:
            continue
        row["tape_label_win"] = float(outcome[key])
        row["tape_label_margin"] = float(margin[key])
    rows = [r for r in rows if not r["error"] and "tape_label_win" in r]

    def correlate(field):
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        truth = np.asarray([row["true_margin"] for row in rows], dtype=np.float64)
        if values.std() == 0 or truth.std() == 0:
            return None
        return float(np.corrcoef(values, truth)[0, 1])

    board_fields = ("board_land", "board_crops", "board_animals", "board_money")
    for row in rows:
        board = row.get("board") or {}
        for name in board_fields:
            row[name] = float(board.get(name[len("board_"):], float("nan")))

    def correlate_field(name):
        values = np.asarray([r[name] for r in rows], dtype=np.float64)
        truth = np.asarray([r["true_margin"] for r in rows], dtype=np.float64)
        good = np.isfinite(values) & np.isfinite(truth)
        if good.sum() < 3 or values[good].std() == 0 or truth[good].std() == 0:
            return None
        return float(np.corrcoef(values[good], truth[good])[0, 1])

    true_wins = sum(1 for r in rows if r["true_margin"] > 0)
    result = {
        "cells": len(rows), "true_wins": true_wins,
        "true_win_rate": true_wins / len(rows) if rows else None,
        "correlation_with_true_margin": {
            "tape_label_margin": correlate("tape_label_margin"),
            "tape_label_win": correlate("tape_label_win"),
            **{name: correlate_field(name) for name in board_fields},
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
