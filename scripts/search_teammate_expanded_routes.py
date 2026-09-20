#!/usr/bin/env python3
"""Fully paired C++ screening of expanded routes under teammate overlays."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


SOURCE = ""
TAPES: dict[str, list[dict[str, Any]]] = {}
LEFT: Any = None
RIGHT: Any = None
CONFIGURATION: dict[str, Any] = {}


def _worker_init(source_path: str, tapes_path: str) -> None:
    global SOURCE, TAPES, LEFT, RIGHT
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    from meta_agent.src.teammate_expanded_routes import (
        TeammateExpandedRouteAgent,
        load_action_tapes,
    )

    SOURCE = Path(source_path).read_text(encoding="utf-8")
    TAPES = load_action_tapes(tapes_path)
    LEFT = TeammateExpandedRouteAgent(SOURCE, TAPES, f"expanded_left_{os.getpid()}")
    RIGHT = TeammateExpandedRouteAgent(SOURCE, TAPES, f"expanded_right_{os.getpid()}")


def _play(task: tuple[int, int, int, str, str, int, int]) -> tuple[float, ...]:
    task_id, left_index, right_index, left_route, right_route, seed, left_seat = task
    try:
        from fast_kaggriculture import Config, FastEnv

        LEFT.select(left_route)
        RIGHT.select(right_route)
        agents = [LEFT, RIGHT] if left_seat == 0 else [RIGHT, LEFT]
        env = FastEnv(Config(), int(seed))
        observations = list(env.reset(int(seed)))
        while not env.done:
            step = int(env.step_count)
            for player, observation in enumerate(observations):
                observation["player"] = player
                observation["step"] = step
            actions = [agents[player](observations[player], CONFIGURATION) for player in range(2)]
            observations = list(env.step(actions))
        rewards = [float(value) for value in env.rewards]
        own = rewards[left_seat]
        other = rewards[1 - left_seat]
        margin = own - other
        return (
            task_id,
            left_index,
            right_index,
            seed,
            left_seat,
            own,
            other,
            margin,
            float(margin > 0) + 0.5 * float(margin == 0),
            0.0,
        )
    except Exception as error:
        return (task_id, left_index, right_index, seed, left_seat, 0, 0, 0, 0, 1.0)


def _seeds(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    return tuple(range(int(start), int(stop))) if separator else tuple(
        int(part) for part in value.split(",")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seeds", type=_seeds, default=tuple(range(20260824, 20260826)))
    parser.add_argument("--workers", type=int, default=min(192, os.cpu_count() or 1))
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--all-opponent-routes",
        action="store_true",
        help="Use all 56 family representatives as columns, including zero-win own drops.",
    )
    parser.add_argument(
        "--opponent-families",
        help="Optional comma-separated opponent family subset, for rectangular follow-up shards.",
    )
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    routes = [str(value["route_id"]) for value in metadata["selected"]]
    families = [str(value["family"]) for value in metadata["selected"]]
    opponent_entries = (
        metadata.get("opponent_routes", metadata["selected"])
        if args.all_opponent_routes
        else metadata["selected"]
    )
    opponent_routes = [str(value["route_id"]) for value in opponent_entries]
    opponent_families = [str(value["family"]) for value in opponent_entries]
    if args.opponent_families:
        wanted_opponents = {
            value.strip() for value in args.opponent_families.split(",") if value.strip()
        }
        filtered = [
            (route, family)
            for route, family in zip(opponent_routes, opponent_families)
            if family in wanted_opponents
        ]
        opponent_routes = [value[0] for value in filtered]
        opponent_families = [value[1] for value in filtered]
    if args.limit is not None:
        routes = routes[: args.limit]
        families = families[: args.limit]
        opponent_routes = opponent_routes[: args.limit]
        opponent_families = opponent_families[: args.limit]
    tasks = []
    task_id = 0
    for left_index, left_route in enumerate(routes):
        for right_index, right_route in enumerate(opponent_routes):
            for seed in args.seeds:
                for seat in (0, 1):
                    tasks.append(
                        (task_id, left_index, right_index, left_route, right_route, seed, seat)
                    )
                    task_id += 1

    columns = (
        "task_id",
        "left_index",
        "right_index",
        "seed",
        "left_seat",
        "left_reward",
        "right_reward",
        "margin",
        "score",
        "error",
    )
    completed: dict[int, tuple[float, ...]] = {}
    if args.output.exists():
        with np.load(args.output) as saved:
            previous = saved["games"]
        completed = {int(row[0]): tuple(row) for row in previous}
    pending = [task for task in tasks if task[0] not in completed]
    started = time.perf_counter()
    context = mp.get_context("fork")
    with context.Pool(
        min(args.workers, max(1, len(pending))),
        initializer=_worker_init,
        initargs=(str(args.source.resolve()), str(args.actions.resolve())),
    ) as pool:
        for offset, row in enumerate(pool.imap_unordered(_play, pending, chunksize=1), start=1):
            completed[int(row[0])] = row
            if offset % 100 == 0 or offset == len(pending):
                elapsed = time.perf_counter() - started
                rate = offset / max(elapsed, 1e-9)
                print(
                    f"completed {len(completed)}/{len(tasks)} games "
                    f"({rate:.1f} games/s)",
                    flush=True,
                )
            if offset % args.checkpoint_every == 0:
                values = np.asarray([completed[key] for key in sorted(completed)], dtype=np.float64)
                np.savez_compressed(args.output, games=values)

    games = np.asarray([completed[key] for key in sorted(completed)], dtype=np.float64)
    np.savez_compressed(args.output, games=games)
    errors = int(np.count_nonzero(games[:, -1]))
    matrix = np.full((len(routes), len(opponent_routes)), np.nan, dtype=np.float64)
    margins = np.full_like(matrix, np.nan)
    for left in range(len(routes)):
        for right in range(len(opponent_routes)):
            subset = games[(games[:, 1] == left) & (games[:, 2] == right) & (games[:, -1] == 0)]
            if len(subset):
                matrix[left, right] = float(np.mean(subset[:, 8]))
                margins[left, right] = float(np.mean(subset[:, 7]))
    route_scores = []
    for index, (route_id, family) in enumerate(zip(routes, families)):
        route_scores.append(
            {
                "route_id": route_id,
                "family": family,
                "mean_score": float(np.nanmean(matrix[index])),
                "worst_family_score": float(np.nanmin(matrix[index])),
                "mean_margin": float(np.nanmean(margins[index])),
            }
        )
    route_scores.sort(key=lambda value: (-value["mean_score"], -value["worst_family_score"]))
    payload = {
        "schema_version": 1,
        "engine": "fast_kaggriculture C++ FastEnv",
        "paired": "every route pair, seed, and both seats",
        "seeds": list(args.seeds),
        "games": len(games),
        "errors": errors,
        "columns": columns,
        "routes": routes,
        "families": families,
        "opponent_routes": opponent_routes,
        "opponent_families": opponent_families,
        "score_matrix": matrix.tolist(),
        "margin_matrix": margins.tolist(),
        "ranking": route_scores,
    }
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"games": len(games), "errors": errors, "top": route_scores[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
