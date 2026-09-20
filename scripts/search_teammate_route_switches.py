#!/usr/bin/env python3
"""Counterfactual one-switch search with paired C++ games and state capture."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


LEFT: Any = None
RIGHT: Any = None
FEATURE_DIM = 0


def _worker_init(source_path: str, tapes_path: str) -> None:
    global LEFT, RIGHT, FEATURE_DIM
    for variable in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
    ):
        os.environ[variable] = "1"
    from meta_agent.src.teammate_expanded_routes import (
        TeammateExpandedRouteAgent,
        load_action_tapes,
    )

    source = Path(source_path).read_text(encoding="utf-8")
    tapes = load_action_tapes(tapes_path)
    LEFT = TeammateExpandedRouteAgent(source, tapes, f"switch_left_{os.getpid()}")
    RIGHT = TeammateExpandedRouteAgent(source, tapes, f"switch_right_{os.getpid()}")
    from meta_agent.src.route_switch_features import ROUTE_SWITCH_DIM
    FEATURE_DIM = ROUTE_SWITCH_DIM


def _features(observation: dict[str, Any], history: Any, route_actions: Any) -> np.ndarray:
    from meta_agent.src.route_switch_features import route_switch_vector

    return route_switch_vector(observation, history, route_actions).astype(np.float32)


def _play(task: tuple[int, int, int, int, str, str, str, int, int, int]):
    task_id, opening_index, target_index, opponent_index, opening, target, opponent, checkpoint, seed, seat = task
    try:
        from fast_kaggriculture import Config, FastEnv
        from meta_agent.src.route_switch_features import RouteSwitchHistory

        LEFT.select_schedule(((0, opening), (checkpoint, target)))
        RIGHT.select(opponent)
        agents = [LEFT, RIGHT] if seat == 0 else [RIGHT, LEFT]
        env = FastEnv(Config(), seed)
        observations = list(env.reset(seed))
        history = RouteSwitchHistory()
        state = None
        while not env.done:
            step = int(env.step_count)
            for player, observation in enumerate(observations):
                observation["player"] = player
                observation["step"] = step
            history.update(observations[seat])
            if step == checkpoint:
                state = _features(observations[seat], history, LEFT.action_tapes[opening])
            actions = [agents[player](observations[player], {}) for player in range(2)]
            observations = list(env.step(actions))
        if state is None:
            raise RuntimeError("switch checkpoint was not observed")
        rewards = [float(value) for value in env.rewards]
        own, other = rewards[seat], rewards[1 - seat]
        margin = own - other
        row = (
            task_id, opening_index, target_index, opponent_index, checkpoint,
            seed, seat, own, other, margin,
            float(margin > 0) + 0.5 * float(margin == 0), 0.0,
        )
        return row, state
    except Exception:
        return (
            task_id, opening_index, target_index, opponent_index, checkpoint,
            seed, seat, 0.0, 0.0, 0.0, 0.0, 1.0,
        ), np.zeros(FEATURE_DIM, dtype=np.float32)


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _ints(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    return tuple(range(int(start), int(stop))) if separator else tuple(
        int(part) for part in value.split(",")
    )


def _opening_checkpoints(value: str) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for raw in value.split(","):
        family, separator, checkpoints = raw.partition(":")
        if not separator:
            raise argparse.ArgumentTypeError("expected FAMILY:CHECKPOINT[+CHECKPOINT]")
        result[family.strip()] = tuple(int(item) for item in checkpoints.split("+") if item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--openings", type=_csv, required=True)
    parser.add_argument("--targets", type=_csv, required=True)
    parser.add_argument("--checkpoints", type=_ints, required=True)
    parser.add_argument(
        "--opening-checkpoints", type=_opening_checkpoints,
        help="Optional per-opening restriction, e.g. G1:120,G26:72,G41:72.",
    )
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(192, os.cpu_count() or 1))
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--opponent-limit", type=int)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    all_routes = {str(value["family"]): str(value["route_id"]) for value in metadata["opponent_routes"]}
    openings = [(family, all_routes[family]) for family in args.openings]
    targets = [(family, all_routes[family]) for family in args.targets]
    opponents = [
        (str(value["family"]), str(value["route_id"]))
        for value in metadata["opponent_routes"]
    ]
    if args.opponent_limit is not None:
        opponents = opponents[: args.opponent_limit]
    tasks = []
    task_id = 0
    for opening_index, (_, opening) in enumerate(openings):
        opening_family = openings[opening_index][0]
        local_checkpoints = (
            args.opening_checkpoints.get(opening_family, args.checkpoints)
            if args.opening_checkpoints else args.checkpoints
        )
        for target_index, (_, target) in enumerate(targets):
            for opponent_index, (_, opponent) in enumerate(opponents):
                for checkpoint in local_checkpoints:
                    for seed in args.seeds:
                        for seat in (0, 1):
                            tasks.append((
                                task_id, opening_index, target_index, opponent_index,
                                opening, target, opponent, checkpoint, seed, seat,
                            ))
                            task_id += 1
    started = time.perf_counter()
    games = np.zeros((len(tasks), 12), dtype=np.float64)
    from meta_agent.src.route_switch_features import ROUTE_SWITCH_DIM
    states = np.zeros((len(tasks), ROUTE_SWITCH_DIM), dtype=np.float32)
    context = mp.get_context("fork")
    with context.Pool(
        min(args.workers, len(tasks)), initializer=_worker_init,
        initargs=(str(args.source.resolve()), str(args.actions.resolve())),
    ) as pool:
        for count, (row, state) in enumerate(pool.imap_unordered(_play, tasks, chunksize=1), start=1):
            index = int(row[0])
            games[index] = row
            states[index] = state
            if count % 100 == 0 or count == len(tasks):
                rate = count / max(time.perf_counter() - started, 1e-9)
                print(f"completed {count}/{len(tasks)} games ({rate:.1f} games/s)", flush=True)
            if count % args.checkpoint_every == 0:
                np.savez_compressed(args.output, games=games, states=states)
    np.savez_compressed(
        args.output,
        games=games,
        states=states,
        openings=np.asarray([value[0] for value in openings]),
        targets=np.asarray([value[0] for value in targets]),
        opponents=np.asarray([value[0] for value in opponents]),
        checkpoints=np.asarray(sorted(set(int(row[7]) for row in tasks)), dtype=np.int16),
        seeds=np.asarray(args.seeds, dtype=np.int64),
    )
    print(json.dumps({
        "output": str(args.output), "games": len(games),
        "errors": int(np.count_nonzero(games[:, -1])), "state_shape": list(states.shape),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
