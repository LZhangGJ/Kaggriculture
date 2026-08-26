#!/usr/bin/env python3
"""Evaluate one submission against each current top-level public agent."""

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


AGENT: Path | None = None


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


def _inventory(root: Path) -> list[dict[str, str]]:
    rows = []
    for directory in sorted(root.iterdir(), key=lambda value: value.name.lower()):
        if not directory.is_dir() or directory.name in {"sources", "by_sha256"}:
            continue
        candidates = sorted(
            (
                path for path in directory.rglob("*.py")
                if path.name in {"main.py", "submission.py"}
            ),
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"no agent entrypoint under {directory}")
        entrypoint = candidates[0].resolve()
        rows.append(
            {
                "opponent": directory.name,
                "entrypoint": str(entrypoint),
                "sha256": hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
            }
        )
    return rows


def _init_worker(agent: str, workspace_deps: str) -> None:
    global AGENT
    if workspace_deps and workspace_deps not in sys.path:
        sys.path.insert(0, workspace_deps)
    AGENT = Path(agent)


def _run(task: tuple[str, str, int, int]) -> dict[str, Any]:
    opponent, opponent_path, seed, seat = task
    started = time.perf_counter()
    try:
        from kaggle_environments import make

        assert AGENT is not None
        own = runpy.run_path(str(AGENT)).get("agent")
        other = runpy.run_path(opponent_path).get("agent")
        if not callable(own) or not callable(other):
            raise ValueError("both submissions must expose callable agent")
        agents = [other, other]
        agents[seat] = own
        environment = make(
            "kaggriculture",
            configuration={"episodeSteps": 720},
            info={"seed": int(seed)},
            debug=True,
        )
        environment.run(agents)
        own_reward = float(_get(environment.state[seat], "reward", 0.0) or 0.0)
        other_reward = float(_get(environment.state[1 - seat], "reward", 0.0) or 0.0)
        score = 1.0 if own_reward > other_reward else 0.5 if own_reward == other_reward else 0.0
        return {
            "opponent": opponent, "seed": seed, "seat": seat,
            "reward": own_reward, "opponent_reward": other_reward,
            "margin": own_reward - other_reward, "score": score,
            "win": score == 1.0, "tie": score == 0.5,
            "completed": len(environment.steps) == 720,
            "steps": len(environment.steps), "error": None,
            "seconds": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "opponent": opponent, "seed": seed, "seat": seat,
            "completed": False, "error": f"{type(error).__name__}: {error}",
            "seconds": time.perf_counter() - started,
        }


def _lower(values: list[float]) -> float:
    if len(values) < 2:
        return values[0]
    degrees = len(values) - 1
    z = 1.6448536269514722
    critical = (
        z + (z**3 + z) / (4 * degrees)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * degrees**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * degrees**3)
    )
    return statistics.fmean(values) - critical * statistics.stdev(values) / math.sqrt(len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--workspace-deps", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    agent = args.agent.resolve()
    inventory = _inventory(args.pool_root.resolve())
    if not inventory:
        raise ValueError("no current top-level public agents found")
    workspace_deps = (
        args.workspace_deps.resolve()
        if args.workspace_deps else Path(__file__).resolve().parents[4] / ".ke"
    )
    tasks = [
        (row["opponent"], row["entrypoint"], seed, seat)
        for row in inventory for seed in args.seeds for seat in (0, 1)
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(agent), str(workspace_deps)),
    ) as pool:
        games = list(pool.map(_run, tasks, chunksize=1))
    elapsed = time.perf_counter() - started

    opponents = []
    for inventory_row in inventory:
        name = inventory_row["opponent"]
        rows = [row for row in games if row["opponent"] == name]
        valid = [row for row in rows if row["error"] is None]
        complete_seeds = [
            seed for seed in args.seeds
            if sum(row["seed"] == seed for row in valid) == 2
        ]
        paired_scores = [
            statistics.fmean(row["score"] for row in valid if row["seed"] == seed)
            for seed in complete_seeds
        ]
        opponents.append(
            {
                **inventory_row,
                "games": len(rows),
                "valid_games": len(valid),
                "completed_games": sum(bool(row.get("completed")) for row in valid),
                "wins": sum(bool(row["win"]) for row in valid),
                "ties": sum(bool(row["tie"]) for row in valid),
                "losses": sum(not row["win"] and not row["tie"] for row in valid),
                "win_rate": statistics.fmean(row["win"] for row in valid) if valid else 0.0,
                "score": statistics.fmean(row["score"] for row in valid) if valid else 0.0,
                "mean_margin": statistics.fmean(row["margin"] for row in valid) if valid else 0.0,
                "paired_seed_score_lower_95pct": _lower(paired_scores) if paired_scores else 0.0,
                "errors": [row for row in rows if row["error"] is not None],
            }
        )
    opponents.sort(key=lambda row: (row["win_rate"], row["score"], row["mean_margin"], row["opponent"]))
    payload = {
        "schema_version": 1,
        "engine": "kaggle-environments official local simulator",
        "scope": (
            f"{len(inventory)} top-level current public agents; "
            "sources and by_sha256 excluded"
        ),
        "agent": str(agent),
        "agent_sha256": hashlib.sha256(agent.read_bytes()).hexdigest(),
        "pool_root": str(args.pool_root.resolve()),
        "seeds": list(args.seeds),
        "workers": args.workers,
        "opponent_count": len(opponents),
        "games": len(games),
        "valid_games": sum(row["error"] is None for row in games),
        "completed_games": sum(bool(row.get("completed")) for row in games),
        "minimum_win_rate": min(row["win_rate"] for row in opponents),
        "opponents_at_or_above_80pct": sum(row["win_rate"] >= 0.8 for row in opponents),
        "elapsed_seconds": elapsed,
        "games_per_second": len(games) / elapsed,
        "opponents": opponents,
        "games_detail": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "agent_sha256": payload["agent_sha256"],
        "opponent_count": payload["opponent_count"],
        "games": payload["games"],
        "valid_games": payload["valid_games"],
        "completed_games": payload["completed_games"],
        "minimum_win_rate": payload["minimum_win_rate"],
        "opponents_at_or_above_80pct": payload["opponents_at_or_above_80pct"],
        "elapsed_seconds": elapsed,
        "games_per_second": payload["games_per_second"],
        "ranking": [
            {key: row[key] for key in (
                "opponent", "wins", "ties", "losses", "win_rate", "score",
                "mean_margin", "completed_games", "valid_games",
            )}
            for row in opponents
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
