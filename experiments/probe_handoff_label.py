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
import copy
import hashlib
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

from meta_agent.src.route_switch_features import (  # noqa: E402
    OpponentSaleHistory,
    opponent_sale_feature_names,
    route_switch_feature_names,
    route_switch_vector,
)
from run_strong_ab import BOTS  # noqa: E402


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
     opening_index, target_index, opponent_index, seed_index, deployed_prefix) = task
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["REPLAY_FORCED_OPENING"] = opening
    os.environ["REPLAY_HANDOFF_LAND"] = "3"
    os.environ["REPLAY_HANDOFF_LAND_DELAY"] = "1"
    os.environ.pop("REPLAY_HANDOFF_MIN_STEP", None)
    os.environ.pop("REPLAY_HANDOFF_SELECTOR", None)
    os.environ.pop("R1_CONFIG_OVERRIDES", None)
    os.environ["REPLAY_HANDOFF_STEP"] = "288"
    from kaggle_environments import make

    hybrid = load(ROOT / "agent/main.py", f"probe_{opening}_{target}_{seed}_{seat}")
    policy = hybrid.create_agent(seat)
    route = policy.replay
    if not deployed_prefix:
        route.controller.nodes = {}  # the forced schedule below is the intervention
    rival = load(ROOT / "opponents" / f"{opponent}" / "main.py", f"rival_{seed}_{seat}").agent
    with_config = len(inspect.signature(rival).parameters) > 1
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    state = env.reset()
    board = None
    features = None
    sale_features = None
    prefix_hash = None
    sale_history = OpponentSaleHistory()
    error = None
    try:
        while not env.done:
            actions = []
            for player in (0, 1):
                obs = json.loads(json.dumps(state[player].observation))
                obs["step"] = int(obs.get("day", 0)) * 24 + int(obs.get("hour", 0))
                obs["player"] = player
                if player == seat:
                    sale_history.update(obs, env.configuration)
                if player == seat and obs["step"] == checkpoint:
                    prefix_current = route.controller.current
                    if deployed_prefix:
                        route.controller.nodes = {}
                    board = board_shape(obs, seat)
                    history = copy.deepcopy(route.controller.history)
                    history.update(obs)
                    route_id = route.controller.route_by_family[route.controller.current]
                    features = route_switch_vector(
                        obs, history, route.expanded_agent.action_tapes[route_id]
                    ).tolist()
                    sale_features = sale_history.vector().tolist()
                    prefix_hash = hashlib.sha256(json.dumps(
                        {"observation": obs, "history": vars(history)},
                        sort_keys=True, separators=(",", ":"),
                    ).encode()).hexdigest()
                    route.expanded_agent.select_schedule((
                        *route.expanded_agent.schedule,
                        (checkpoint, route.controller.route_by_family[target]),
                    ))
                action = (policy(obs, env.configuration) if player == seat else
                          (rival(obs, env.configuration) if with_config else rival(obs)))
                if player == seat:
                    sale_history.record_action(obs, action, env.configuration)
                actions.append(action)
            state = env.step(actions)
        farms = json.loads(json.dumps(state[0].observation))["farms"]
        own, other = farms[seat]["money"], farms[1 - seat]["money"]
        return {"opening": opening, "target": target, "opponent": opponent, "seed": seed,
                "seat": seat, "checkpoint": checkpoint, "true_margin": own - other,
                "own": own, "rival": other, "board": board,
                "features": features,
                "sale_features": sale_features,
                "prefix_hash": prefix_hash,
                "prefix_current": prefix_current,
                "installed_correct": route.expanded_agent.schedule[-1][1] ==
                                     route.controller.route_by_family[target],
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
    parser.add_argument("--openings", default="G275")
    parser.add_argument("--targets", default="G275,G114,G019,G113,G001,G024,G316,G267,G195")
    parser.add_argument("--opponents", default=",".join(BOTS))
    parser.add_argument("--checkpoints", default="144,168")
    parser.add_argument("--seeds", default="2610800000:2610800032")
    parser.add_argument("--seats", default="0,1")
    parser.add_argument("--label-npz", type=Path)
    parser.add_argument("--start-seed", type=int, default=2609220000,
                        help="first seed of the label npz grid")
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--deployed-prefix", action="store_true",
                        help="run the shipped route tree before the forced checkpoint switch")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    start, _, stop = args.seeds.partition(":")
    seed_values = list(range(int(start), int(stop))) if stop else [int(start)]
    openings = args.openings.split(",")
    targets = args.targets.split(",")
    opponents = args.opponents.split(",")
    checkpoints = [int(x) for x in args.checkpoints.split(",")]
    seats = [int(x) for x in args.seats.split(",")]
    grids = (openings, targets, opponents, checkpoints, seed_values, seats)
    if any(not values or len(values) != len(set(values)) for values in grids) or \
            any(seat not in (0, 1) for seat in seats):
        parser.error("all grids must be non-empty and unique; seats must be 0 and/or 1")

    labels = None
    if args.label_npz:
        with np.load(args.label_npz, mmap_mode="r") as saved:
            labels = {
                "openings": saved["openings"].astype(str).tolist(),
                "targets": saved["targets"].astype(str).tolist(),
                "opponents": saved["opponents"].astype(str).tolist(),
                "checkpoints": saved["checkpoints"].astype(int).tolist(),
                "seeds": saved["seeds"].astype(np.int64).tolist(),
                "outcome": np.asarray(saved["outcome"], dtype=np.float32) * 0.5,
                "margin": np.asarray(saved["margin"], dtype=np.float32),
            }

    tasks, keys = [], []
    for opening in openings:
        for target in targets:
            for opponent in opponents:
                for seed in seed_values:
                    for seat in seats:
                        for checkpoint in checkpoints:
                            tasks.append((opening, target, opponent, seed, seat, checkpoint,
                                          openings.index(opening), targets.index(target),
                                          opponents.index(opponent), seed_values.index(seed),
                                          args.deployed_prefix))
                            if labels and all(value in labels[name] for name, value in (
                                ("openings", opening), ("targets", target),
                                ("opponents", opponent), ("checkpoints", checkpoint),
                                ("seeds", seed),
                            )):
                                keys.append((labels["openings"].index(opening),
                                             labels["checkpoints"].index(checkpoint),
                                             labels["targets"].index(target),
                                             labels["opponents"].index(opponent),
                                             labels["seeds"].index(seed), seat))
                            else:
                                keys.append(None)
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                max_tasks_per_child=1) as pool:
        rows = list(pool.map(play, tasks))
    for row, key in zip(rows, keys):
        if row["error"]:
            continue
        if key is not None:
            row["tape_label_win"] = float(labels["outcome"][key])
            row["tape_label_margin"] = float(labels["margin"][key])
    rows = [r for r in rows if not r["error"]]
    if (len(rows) != len(tasks) or any(r["features"] is None for r in rows) or
            any(r["sale_features"] is None for r in rows) or
            any(not r["installed_correct"] for r in rows)):
        raise RuntimeError("missing warm-label rows or checkpoint features")
    paired = {}
    target_sets = {}
    for row in rows:
        key = (row["opening"], row["opponent"], row["seed"], row["seat"], row["checkpoint"])
        vector = np.asarray(row["features"], dtype=np.float32)
        if vector.shape != (147,) or not np.isfinite(vector).all():
            raise RuntimeError(f"invalid route features for {key}")
        sales = np.asarray(row["sale_features"], dtype=np.float32)
        if sales.shape != (7,) or not np.isfinite(sales).all():
            raise RuntimeError(f"invalid opponent-sale features for {key}")
        signature = (vector, sales, row["prefix_hash"])
        if key in paired and (not np.array_equal(paired[key][0], vector) or
                              not np.array_equal(paired[key][1], sales) or
                              paired[key][2] != row["prefix_hash"]):
            raise RuntimeError(f"counterfactual states differ for {key}")
        paired[key] = signature
        target_sets.setdefault(key, []).append(row["target"])
    expected_targets = sorted(targets)
    if any(sorted(values) != expected_targets for values in target_sets.values()):
        raise RuntimeError("counterfactual target grid is incomplete or duplicated")

    if args.output.suffix == ".npz":
        games = np.zeros((len(rows), 12), dtype=np.float64)
        states = np.asarray([row.pop("features") for row in rows], dtype=np.float32)
        sale_states = np.asarray([row.pop("sale_features") for row in rows], dtype=np.float32)
        for index, row in enumerate(rows):
            games[index] = (
                index, row["opening_index"], row["target_index"], row["opponent_index"],
                row["checkpoint"], row["seed"], row["seat"], row["own"], row["rival"],
                row["true_margin"], float(row["true_margin"] > 0) +
                .5 * float(row["true_margin"] == 0), 0,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output, games=games, states=states,
            opponent_sale_states=sale_states,
            prefix_routes=np.asarray([row["prefix_current"] for row in rows]),
            openings=np.asarray(openings), targets=np.asarray(targets),
            opponents=np.asarray(opponents), checkpoints=np.asarray(checkpoints, dtype=np.int16),
            seeds=np.asarray(seed_values, dtype=np.int64),
            seats=np.asarray(seats, dtype=np.int8),
            feature_names=np.asarray(route_switch_feature_names()),
            opponent_sale_feature_names=np.asarray(opponent_sale_feature_names()),
            engine=np.asarray("official warm replay-to-R1 suffix"),
        )
        print(json.dumps({"output": str(args.output), "games": len(rows),
                          "states": list(states.shape), "errors": 0}))
        return

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
            **({"tape_label_margin": correlate("tape_label_margin"),
                "tape_label_win": correlate("tape_label_win")}
               if rows and "tape_label_win" in rows[0] else {}),
            **{name: correlate_field(name) for name in board_fields},
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
