#!/usr/bin/env python3
"""Mine low-cash, weed-hit and asset-loss states before targeted switch search."""

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


def _init(source_path: str, actions_path: str) -> None:
    global LEFT, RIGHT
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    from meta_agent.src.teammate_expanded_routes import TeammateExpandedRouteAgent, load_action_tapes

    source = Path(source_path).read_text(encoding="utf-8")
    tapes = load_action_tapes(actions_path)
    LEFT = TeammateExpandedRouteAgent(source, tapes, f"mine_left_{os.getpid()}")
    RIGHT = TeammateExpandedRouteAgent(source, tapes, f"mine_right_{os.getpid()}")


def _play(task: tuple[int, int, int, str, str, int, int, tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:
    task_id, opening_index, opponent_index, opening, opponent, seed, seat, checkpoints = task
    from meta_agent.src.route_switch_features import ROUTE_SWITCH_DIM
    states = np.zeros((len(checkpoints), ROUTE_SWITCH_DIM), dtype=np.float32)
    try:
        from fast_kaggriculture import Config, FastEnv
        from meta_agent.src.route_switch_features import RouteSwitchHistory, route_switch_vector

        LEFT.select(opening)
        RIGHT.select(opponent)
        agents = [LEFT, RIGHT] if seat == 0 else [RIGHT, LEFT]
        env = FastEnv(Config(), seed)
        observations = list(env.reset(seed))
        history = RouteSwitchHistory()
        checkpoint_index = {step: index for index, step in enumerate(checkpoints)}
        while not env.done:
            step = int(env.step_count)
            for player, observation in enumerate(observations):
                observation["player"] = player
                observation["step"] = step
            history.update(observations[seat])
            if step in checkpoint_index:
                states[checkpoint_index[step]] = route_switch_vector(
                    observations[seat], history, LEFT.action_tapes[opening]
                )
            actions = [agents[player](observations[player], {}) for player in range(2)]
            observations = list(env.step(actions))
        rewards = [float(value) for value in env.rewards]
        own, other = rewards[seat], rewards[1 - seat]
        margin = own - other
        game = np.asarray([
            task_id, opening_index, opponent_index, seed, seat, own, other, margin,
            float(margin > 0) + 0.5 * float(margin == 0), 0.0,
        ], dtype=np.float64)
        return game, states
    except Exception:
        return np.asarray([task_id, opening_index, opponent_index, seed, seat, 0, 0, 0, 0, 1], dtype=np.float64), states


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _ints(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    return tuple(range(int(start), int(stop))) if separator else tuple(int(item) for item in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--openings", type=_csv, required=True)
    parser.add_argument("--checkpoints", type=_ints, required=True)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(192, os.cpu_count() or 1))
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    route_by_family = {str(row["family"]): str(row["route_id"]) for row in metadata["opponent_routes"]}
    openings = [(family, route_by_family[family]) for family in args.openings]
    opponents = [(str(row["family"]), str(row["route_id"])) for row in metadata["opponent_routes"]]
    tasks = []
    for opening_index, (_, opening) in enumerate(openings):
        for opponent_index, (_, opponent) in enumerate(opponents):
            for seed in args.seeds:
                for seat in (0, 1):
                    tasks.append((len(tasks), opening_index, opponent_index, opening, opponent, seed, seat, args.checkpoints))
    from meta_agent.src.route_switch_features import ROUTE_SWITCH_DIM, route_switch_feature_names
    games = np.zeros((len(tasks), 10), dtype=np.float64)
    states = np.zeros((len(tasks), len(args.checkpoints), ROUTE_SWITCH_DIM), dtype=np.float32)
    started = time.perf_counter()
    with mp.get_context("fork").Pool(
        min(args.workers, len(tasks)), initializer=_init,
        initargs=(str(args.source.resolve()), str(args.actions.resolve())),
    ) as pool:
        for count, (game, state) in enumerate(pool.imap_unordered(_play, tasks, chunksize=1), start=1):
            index = int(game[0]); games[index] = game; states[index] = state
            if count % 100 == 0 or count == len(tasks):
                print(f"completed {count}/{len(tasks)} ({count / max(time.perf_counter() - started, 1e-9):.1f} games/s)", flush=True)
            if count % 1000 == 0:
                np.savez_compressed(args.output, games=games, states=states)
    np.savez_compressed(
        args.output, games=games, states=states,
        openings=np.asarray([row[0] for row in openings]),
        opponents=np.asarray([row[0] for row in opponents]),
        checkpoints=np.asarray(args.checkpoints, dtype=np.int16),
        seeds=np.asarray(args.seeds, dtype=np.int64),
        feature_names=np.asarray(route_switch_feature_names()),
    )
    print(json.dumps({"output": str(args.output), "games": len(games), "errors": int(np.count_nonzero(games[:, -1])), "state_shape": list(states.shape)}, indent=2))


if __name__ == "__main__":
    main()
