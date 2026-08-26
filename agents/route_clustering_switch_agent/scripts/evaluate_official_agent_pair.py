#!/usr/bin/env python3
"""Paired dual-seat evaluation of two submissions in the official simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping


AGENT_A: Path | None = None
AGENT_B: Path | None = None


def _ints(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    return tuple(range(int(start), int(stop))) if separator else tuple(
        int(part) for part in value.split(",") if part.strip()
    )


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    return getter(key, default) if callable(getter) else getattr(value, key, default)


def _init_worker(agent_a: str, agent_b: str, workspace_deps: str) -> None:
    global AGENT_A, AGENT_B
    if workspace_deps and workspace_deps not in sys.path:
        sys.path.insert(0, workspace_deps)
    AGENT_A, AGENT_B = Path(agent_a), Path(agent_b)


def _run(task: tuple[int, int]) -> dict[str, Any]:
    from kaggle_environments import make

    seed, seat_a = task
    assert AGENT_A is not None and AGENT_B is not None
    started = time.perf_counter()
    try:
        namespace_a = runpy.run_path(str(AGENT_A))
        namespace_b = runpy.run_path(str(AGENT_B))
        agent_a = namespace_a.get("agent")
        agent_b = namespace_b.get("agent")
        if not callable(agent_a) or not callable(agent_b):
            raise ValueError("both submissions must expose callable agent")
        agents = [agent_b, agent_b]
        agents[seat_a] = agent_a
        environment = make(
            "kaggriculture",
            configuration={"episodeSteps": 720},
            info={"seed": int(seed)},
            debug=True,
        )
        environment.run(agents)
        reward_a = float(_get(environment.state[seat_a], "reward", 0.0) or 0.0)
        reward_b = float(_get(environment.state[1 - seat_a], "reward", 0.0) or 0.0)
        statuses = [str(_get(value, "status", "")) for value in environment.state]
        return {
            "seed": int(seed),
            "seat_a": int(seat_a),
            "reward_a": reward_a,
            "reward_b": reward_b,
            "margin_a": reward_a - reward_b,
            "score_a": 1.0 if reward_a > reward_b else 0.5 if reward_a == reward_b else 0.0,
            "completed": len(environment.steps) == 720,
            "steps": int(len(environment.steps)),
            "statuses": statuses,
            "error": None,
            "seconds": time.perf_counter() - started,
        }
    except Exception as error:  # preserve failures in the durable audit
        return {
            "seed": int(seed), "seat_a": int(seat_a), "completed": False,
            "error": f"{type(error).__name__}: {error}",
            "seconds": time.perf_counter() - started,
        }


def _lower(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return values[0], 0.0
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    # Cornish-Fisher expansion of the one-sided 95% Student-t critical value.
    # This keeps the official evaluator independent of the native-search scipy env.
    degrees = len(values) - 1
    z = 1.6448536269514722
    critical = (
        z
        + (z**3 + z) / (4 * degrees)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * degrees**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * degrees**3)
    )
    lower = statistics.fmean(values) - critical * standard_error
    return lower, standard_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-a", type=Path, required=True)
    parser.add_argument("--agent-b", type=Path, required=True)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--workspace-deps", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    agent_a = args.agent_a.resolve()
    agent_b = args.agent_b.resolve()
    workspace_deps = (
        args.workspace_deps.resolve()
        if args.workspace_deps else Path(__file__).resolve().parents[4] / ".ke"
    )
    tasks = [(seed, seat) for seed in args.seeds for seat in (0, 1)]
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(agent_a), str(agent_b), str(workspace_deps)),
    ) as pool:
        rows = list(pool.map(_run, tasks, chunksize=1))
    elapsed = time.perf_counter() - started
    valid = [row for row in rows if row["error"] is None]
    per_seed_scores = [
        statistics.fmean(row["score_a"] for row in valid if row["seed"] == seed)
        for seed in args.seeds
        if sum(row["seed"] == seed for row in valid) == 2
    ]
    per_seed_margins = [
        statistics.fmean(row["margin_a"] for row in valid if row["seed"] == seed)
        for seed in args.seeds
        if sum(row["seed"] == seed for row in valid) == 2
    ]
    score_lower, score_se = _lower(per_seed_scores) if per_seed_scores else (0.0, 0.0)
    margin_lower, margin_se = _lower(per_seed_margins) if per_seed_margins else (0.0, 0.0)
    payload = {
        "schema_version": 1,
        "engine": "kaggle-environments official local simulator",
        "pairing": "identical seeds and both seats",
        "agent_a": str(agent_a),
        "agent_b": str(agent_b),
        "agent_a_sha256": hashlib.sha256(agent_a.read_bytes()).hexdigest(),
        "agent_b_sha256": hashlib.sha256(agent_b.read_bytes()).hexdigest(),
        "seeds": list(args.seeds),
        "workers": args.workers,
        "games": len(rows),
        "valid_games": len(valid),
        "completed_games": sum(bool(row.get("completed")) for row in valid),
        "errors": [row for row in rows if row["error"] is not None],
        "score_a": statistics.fmean(row["score_a"] for row in valid) if valid else 0.0,
        "win_rate_a": statistics.fmean(row["score_a"] == 1.0 for row in valid) if valid else 0.0,
        "tie_rate": statistics.fmean(row["score_a"] == 0.5 for row in valid) if valid else 0.0,
        "mean_margin_a": statistics.fmean(row["margin_a"] for row in valid) if valid else 0.0,
        "paired_seed_score_lower_95pct": score_lower,
        "paired_seed_score_standard_error": score_se,
        "paired_seed_margin_lower_95pct": margin_lower,
        "paired_seed_margin_standard_error": margin_se,
        "elapsed_seconds": elapsed,
        "games_per_second": len(rows) / elapsed,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in (
        "agent_a_sha256", "agent_b_sha256", "games", "valid_games",
        "completed_games", "score_a", "win_rate_a", "tie_rate",
        "mean_margin_a", "paired_seed_score_lower_95pct",
        "paired_seed_margin_lower_95pct", "elapsed_seconds", "games_per_second",
        "errors",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
