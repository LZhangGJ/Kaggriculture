#!/usr/bin/env python3
"""Extract visible checkpoint features against official Python opponents.

The environment is advanced only through the requested checkpoint.  This keeps
the expensive official interpreter in the loop without paying for an unused
terminal reward calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np


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
        rows.append({
            "opponent": directory.name,
            "entrypoint": str(entrypoint),
            "sha256": hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
        })
    return rows


def _init_worker(agent: str, workspace_deps: str) -> None:
    global AGENT
    if workspace_deps and workspace_deps not in sys.path:
        sys.path.insert(0, workspace_deps)
    AGENT = Path(agent)


def _controller(agent: Any) -> Any:
    namespace = getattr(agent, "__globals__", {})
    for name in ("_S_CONTROLLER", "_CONTROLLER"):
        controller = namespace.get(name)
        if controller is not None:
            return controller
    raise RuntimeError("candidate agent does not expose its route controller")


def _run(task: tuple[str, str, int, int, int]) -> dict[str, Any]:
    opponent, opponent_path, seed, candidate_seat, checkpoint = task
    from kaggle_environments.agent import Agent as OfficialAgent
    from kaggle_environments import make

    assert AGENT is not None
    started = time.perf_counter()
    try:
        own = runpy.run_path(str(AGENT)).get("agent")
        other = runpy.run_path(opponent_path).get("agent")
        if not callable(own) or not callable(other):
            raise ValueError("both submissions must expose callable agent")
        agents = [other, other]
        agents[candidate_seat] = own
        environment = make(
            "kaggriculture",
            configuration={"episodeSteps": 720},
            info={"seed": int(seed)},
            debug=True,
        )
        environment.reset(2)
        runners = [OfficialAgent(agent, environment) for agent in agents]
        while True:
            shared_states = [
                environment._Environment__get_shared_state(index)
                for index in range(len(runners))
            ]
            step = int(_get(_get(shared_states[0], "observation", {}), "step", 0) or 0)
            actions = [
                runner.act(_get(shared_states[index], "observation", {}))[0]
                if str(_get(shared_states[index], "status", "")) == "ACTIVE" else None
                for index, runner in enumerate(runners)
            ]
            if step >= checkpoint:
                break
            environment.step(actions)
        trace = [
            row for row in list(_controller(own).decision_trace)
            if int(row["checkpoint"]) == checkpoint
        ]
        if len(trace) != 1:
            raise RuntimeError(
                f"expected one route decision at checkpoint {checkpoint}, got {len(trace)}"
            )
        return {
            "opponent": opponent,
            "seed": int(seed),
            "seat": int(candidate_seat),
            "checkpoint": int(checkpoint),
            "prediction": str(trace[0]["prediction"]),
            "features": [float(value) for value in trace[0]["vector"]],
            "steps": len(environment.steps),
            "error": None,
            "seconds": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "opponent": opponent,
            "seed": int(seed),
            "seat": int(candidate_seat),
            "checkpoint": int(checkpoint),
            "error": f"{type(error).__name__}: {error}",
            "seconds": time.perf_counter() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--checkpoint", type=int, default=144)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--workspace-deps", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    agent = args.agent.resolve()
    inventory = _inventory(args.pool_root.resolve())
    workspace_deps = (
        args.workspace_deps.resolve()
        if args.workspace_deps else Path(__file__).resolve().parents[4] / ".ke"
    )
    tasks = [
        (row["opponent"], row["entrypoint"], seed, seat, args.checkpoint)
        for row in inventory for seed in args.seeds for seat in (0, 1)
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(agent), str(workspace_deps)),
    ) as pool:
        rows = list(pool.map(_run, tasks, chunksize=1))
    elapsed = time.perf_counter() - started
    valid = [row for row in rows if row["error"] is None]
    feature_dim = len(valid[0]["features"]) if valid else 0
    payload = {
        "schema": "official-opponent-checkpoint-features-v1",
        "agent": str(agent),
        "agent_sha256": hashlib.sha256(agent.read_bytes()).hexdigest(),
        "pool_root": str(args.pool_root.resolve()),
        "opponents": inventory,
        "seeds": list(args.seeds),
        "checkpoint": args.checkpoint,
        "games": len(rows),
        "valid_games": len(valid),
        "feature_dim": feature_dim,
        "elapsed_seconds": elapsed,
        "games_per_second": len(rows) / elapsed,
        "errors": [row for row in rows if row["error"] is not None],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: payload[key] for key in (
        "agent_sha256", "checkpoint", "games", "valid_games", "feature_dim",
        "elapsed_seconds", "games_per_second", "errors",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
