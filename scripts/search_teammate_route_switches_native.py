#!/usr/bin/env python3
"""Pure-C++ batched counterfactual route-switch search and state capture."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _ints(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    return tuple(range(int(start), int(stop))) if separator else tuple(
        int(part) for part in value.split(",")
    )


def _opening_checkpoints(value: str) -> dict[str, tuple[int, ...]]:
    result = {}
    for raw in value.split(","):
        family, separator, checkpoints = raw.partition(":")
        if not separator:
            raise argparse.ArgumentTypeError("expected FAMILY:STEP[+STEP]")
        result[family.strip()] = tuple(int(item) for item in checkpoints.split("+"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--openings", type=_csv, required=True)
    parser.add_argument("--targets", type=_csv, required=True)
    parser.add_argument("--checkpoints", type=_ints, required=True)
    parser.add_argument("--opening-checkpoints", type=_opening_checkpoints)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    setup_started = time.perf_counter()
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    openings = tuple(args.openings)
    targets = tuple(args.targets)
    opponents = tuple(bundle.families)
    for family in (*openings, *targets):
        bundle.index(family)

    task_rows: list[tuple[int, ...]] = []
    game_tasks: list[tuple[int, ...]] = []
    state_tasks: list[tuple[int, ...]] = []
    state_keys: dict[tuple[int, int, int, int, int], int] = {}
    state_indices: list[int] = []
    task_id = 0
    for opening_index, opening in enumerate(openings):
        checkpoints = (
            args.opening_checkpoints.get(opening, args.checkpoints)
            if args.opening_checkpoints else args.checkpoints
        )
        opening_route = bundle.index(opening)
        for target_index, target in enumerate(targets):
            target_route = bundle.index(target)
            for opponent_index, opponent in enumerate(opponents):
                opponent_route = bundle.index(opponent)
                for checkpoint in checkpoints:
                    for seed in args.seeds:
                        for seat in (0, 1):
                            task_rows.append((
                                task_id, opening_index, target_index, opponent_index,
                                checkpoint, seed, seat,
                            ))
                            if seat == 0:
                                game_tasks.append((
                                    opening_route, opponent_route, seed,
                                    checkpoint, target_route, -1, -1,
                                ))
                                feature_task = (
                                    opening_route, opponent_route, seed,
                                    checkpoint, 0, opening_route,
                                )
                            else:
                                game_tasks.append((
                                    opponent_route, opening_route, seed,
                                    -1, -1, checkpoint, target_route,
                                ))
                                feature_task = (
                                    opponent_route, opening_route, seed,
                                    checkpoint, 1, opening_route,
                                )
                            key = (opening_index, opponent_index, checkpoint, seed, seat)
                            state_index = state_keys.get(key)
                            if state_index is None:
                                state_index = len(state_tasks)
                                state_keys[key] = state_index
                                state_tasks.append(feature_task)
                            state_indices.append(state_index)
                            task_id += 1

    build_elapsed = time.perf_counter() - setup_started
    games_array = np.asarray(game_tasks, dtype=np.int64)
    states_array = np.asarray(state_tasks, dtype=np.int64)
    print(json.dumps({
        "games": len(game_tasks), "unique_states": len(state_tasks),
        "seeds": len(args.seeds), "setup_seconds": build_elapsed,
    }, ensure_ascii=False), flush=True)

    started = time.perf_counter()
    rewards = np.asarray(bundle.executor.play_batch(games_array), dtype=np.float64)
    game_elapsed = time.perf_counter() - started
    print(f"native games: {len(game_tasks) / game_elapsed:,.1f} games/s "
          f"({game_elapsed:.3f}s)", flush=True)

    state_started = time.perf_counter()
    unique_states = np.asarray(
        bundle.executor.features_batch(states_array), dtype=np.float32
    )
    feature_elapsed = time.perf_counter() - state_started
    print(f"native states: {len(state_tasks) / feature_elapsed:,.1f} states/s "
          f"({feature_elapsed:.3f}s)", flush=True)

    rows = np.zeros((len(task_rows), 12), dtype=np.float64)
    rows[:, :7] = np.asarray(task_rows, dtype=np.float64)
    seats = rows[:, 6].astype(np.int64)
    own = rewards[np.arange(len(rewards)), seats]
    other = rewards[np.arange(len(rewards)), 1 - seats]
    margin = own - other
    rows[:, 7] = own
    rows[:, 8] = other
    rows[:, 9] = margin
    rows[:, 10] = (margin > 0).astype(np.float64) + .5 * (margin == 0)
    states = unique_states[np.asarray(state_indices, dtype=np.int64)]

    save_started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        games=rows,
        states=states,
        openings=np.asarray(openings),
        targets=np.asarray(targets),
        opponents=np.asarray(opponents),
        checkpoints=np.asarray(sorted(set(rows[:, 4].astype(int))), dtype=np.int16),
        seeds=np.asarray(args.seeds, dtype=np.int64),
        engine=np.asarray("native C++ teammate executor v1"),
    )
    save_elapsed = time.perf_counter() - save_started
    print(json.dumps({
        "output": str(args.output), "games": len(rows), "errors": 0,
        "state_shape": list(states.shape), "game_seconds": game_elapsed,
        "feature_seconds": feature_elapsed, "save_seconds": save_elapsed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
