#!/usr/bin/env python3
"""Search fixed routes and one-switch schedules against public trace agents."""

from __future__ import annotations

import argparse
import json
import runpy
import time
from pathlib import Path

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def _ints(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    return tuple(range(int(start), int(stop))) if separator else tuple(
        int(part) for part in value.split(",") if part.strip()
    )


def _opponent(value: str) -> tuple[str, str]:
    family, separator, trace_spec = value.partition("=")
    if not separator or not family or not trace_spec:
        raise argparse.ArgumentTypeError(
            "opponent must be FAMILY=PATH or FAMILY=PATH::VARIABLE::KEY"
        )
    return family, trace_spec


def _trace(trace_spec: str) -> list[dict]:
    parts = trace_spec.rsplit("::", 2)
    path = Path(parts[0]).resolve()
    namespace = runpy.run_path(str(path))
    if len(parts) == 1:
        trace = namespace.get("_ACTIONS", namespace.get("_TRACE"))
        description = "_ACTIONS/_TRACE"
    elif len(parts) == 3:
        variable, key = parts[1:]
        routes = namespace.get(variable)
        if not isinstance(routes, dict):
            raise ValueError(f"{variable} is not a route dictionary in {path}")
        trace = routes.get(key)
        description = f"{variable}[{key!r}]"
    else:
        raise ValueError(
            "trace spec must be PATH or PATH::VARIABLE::KEY: " + trace_spec
        )
    if not isinstance(trace, list) or len(trace) < 719:
        raise ValueError(f"no 719-step {description} in {path}")
    return [dict(value or {}) for value in trace[:719]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--opponent", type=_opponent, action="append", required=True,
        help="FAMILY=PATH or FAMILY=PATH::VARIABLE::KEY for bundled route dictionaries.",
    )
    parser.add_argument("--opening", default="G001")
    parser.add_argument("--checkpoints", default="fixed,144,168,216")
    parser.add_argument("--candidates", help="Optional comma-separated candidate families.")
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-output", type=Path)
    args = parser.parse_args()

    additional = {family: _trace(path) for family, path in args.opponent}
    bundle = NativeTeammateBundle(
        args.source, args.actions, args.metadata, additional_routes=additional
    )
    opponent_families = tuple(additional)
    base_families = bundle.families[: len(bundle.families) - len(additional)]
    candidates = (
        tuple(value for value in args.candidates.split(",") if value)
        if args.candidates else base_families
    )
    missing = sorted(set(candidates) - set(base_families))
    if missing:
        raise KeyError(f"unknown candidate family: {missing[0]}")
    schedules: list[tuple[str, int, str]] = []
    for raw in args.checkpoints.split(","):
        if raw == "fixed":
            schedules.extend((f"fixed:{family}", -1, family) for family in candidates)
        else:
            checkpoint = int(raw)
            schedules.extend(
                (f"{checkpoint}:{family}", checkpoint, family) for family in candidates
            )

    tasks = np.empty(
        (len(schedules) * len(opponent_families) * len(args.seeds) * 2, 7),
        dtype=np.int64,
    )
    sample_rows = []
    row = 0
    opening_index = bundle.index(args.opening)
    for strategy_index, (_label, checkpoint, target) in enumerate(schedules):
        target_index = bundle.index(target)
        for opponent_index, opponent in enumerate(opponent_families):
            opponent_route = bundle.index(opponent)
            for seed_index, seed in enumerate(args.seeds):
                for candidate_seat in (0, 1):
                    if checkpoint < 0:
                        left = target_index if candidate_seat == 0 else opponent_route
                        right = opponent_route if candidate_seat == 0 else target_index
                        tasks[row] = (left, right, seed, -1, -1, -1, -1)
                    elif candidate_seat == 0:
                        tasks[row] = (
                            opening_index, opponent_route, seed,
                            checkpoint, target_index, -1, -1,
                        )
                    else:
                        tasks[row] = (
                            opponent_route, opening_index, seed,
                            -1, -1, checkpoint, target_index,
                        )
                    sample_rows.append((
                        strategy_index, opponent_index, seed_index, candidate_seat
                    ))
                    row += 1

    started = time.perf_counter()
    rewards = np.asarray(bundle.executor.play_batch(tasks), dtype=np.float64)
    elapsed = time.perf_counter() - started
    margins = np.empty(
        (len(schedules), len(opponent_families), len(args.seeds), 2),
        dtype=np.float64,
    )
    for index, (strategy_index, opponent_index, seed_index, seat) in enumerate(sample_rows):
        margins[strategy_index, opponent_index, seed_index, seat] = (
            rewards[index, seat] - rewards[index, 1 - seat]
        )

    ranking = []
    for strategy_index, (label, checkpoint, target) in enumerate(schedules):
        block = margins[strategy_index]
        opponent_rows = []
        for opponent_index, opponent in enumerate(opponent_families):
            values = block[opponent_index]
            opponent_rows.append({
                "opponent": opponent,
                "raw_win_rate": float(np.mean(values > 0)),
                "both_seats_win_rate": float(np.mean(np.all(values > 0, axis=1))),
                "mean_margin": float(np.mean(values)),
                "minimum_margin": float(np.min(values)),
            })
        ranking.append({
            "strategy": label,
            "checkpoint": checkpoint,
            "target": target,
            "minimum_opponent_raw_win_rate": min(
                value["raw_win_rate"] for value in opponent_rows
            ),
            "combined_raw_win_rate": float(np.mean(block > 0)),
            "minimum_opponent_mean_margin": min(
                value["mean_margin"] for value in opponent_rows
            ),
            "combined_mean_margin": float(np.mean(block)),
            "opponents": opponent_rows,
        })
    ranking.sort(key=lambda value: (
        -value["minimum_opponent_raw_win_rate"],
        -value["combined_raw_win_rate"],
        -value["minimum_opponent_mean_margin"],
        -value["combined_mean_margin"],
        value["strategy"],
    ))
    payload = {
        "schema": "native-public-trace-counter-search-v1",
        "opponents": [
            {"family": family, "trace_spec": trace_spec}
            for family, trace_spec in args.opponent
        ],
        "opening": args.opening,
        "checkpoints": args.checkpoints.split(","),
        "candidate_count": len(candidates),
        "strategy_count": len(schedules),
        "seed_count": len(args.seeds),
        "games": len(tasks),
        "elapsed_seconds": elapsed,
        "games_per_second": len(tasks) / elapsed,
        "ranking": ranking,
    }
    if args.matrix_output:
        args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.matrix_output,
            strategies=np.asarray([value[0] for value in schedules]),
            opponents=np.asarray(opponent_families),
            seeds=np.asarray(args.seeds, dtype=np.int64),
            margins=margins,
        )
        payload["matrix_output"] = str(args.matrix_output.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "games": payload["games"],
        "elapsed_seconds": elapsed,
        "games_per_second": payload["games_per_second"],
        "top": ranking[:20],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
