#!/usr/bin/env python3
"""Paired official-engine evaluation of two standalone Kaggriculture agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
import types
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence


for _variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_variable] = "1"

from kaggle_environments import make


_LEFT: dict[str, Any] | None = None
_RIGHT: dict[str, Any] | None = None


def _load(path: str, name: str) -> dict[str, Any]:
    source = Path(path).read_text(encoding="utf-8")
    parent = str(Path(path).resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = types.ModuleType(name)
    module.__file__ = path
    sys.modules[name] = module
    exec(compile(source, path, "exec"), module.__dict__)
    if not callable(module.__dict__.get("agent")):
        raise RuntimeError(f"standalone agent() missing from {path}")
    return module.__dict__


def _init_worker(left: str, right: str) -> None:
    global _LEFT, _RIGHT
    identity = os.getpid()
    _LEFT = _load(left, f"paired_left_{identity}")
    _RIGHT = _load(right, f"paired_right_{identity}")


def _reward(state: Any) -> float:
    reward = getattr(state, "reward", None)
    if reward is None:
        raise RuntimeError(f"missing reward for status={getattr(state, 'status', None)}")
    return float(reward)


def play(task: tuple[int, int]) -> dict[str, Any]:
    assert _LEFT is not None and _RIGHT is not None
    seed, left_seat = task
    left = _LEFT["agent"]
    right = _RIGHT["agent"]
    agents = [left, right] if left_seat == 0 else [right, left]
    environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
    started = time.perf_counter()
    environment.run(agents)
    states = list(environment.state)
    statuses = [str(getattr(state, "status", "")) for state in states]
    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(
            f"bad statuses seed={seed} left_seat={left_seat}: {statuses}"
        )
    rewards = [_reward(state) for state in states]
    own, other = rewards[left_seat], rewards[1 - left_seat]
    margin = own - other
    return {
        "seed": seed,
        "left_seat": left_seat,
        "left_reward": own,
        "right_reward": other,
        "margin": margin,
        "result": "win" if margin > 0 else "loss" if margin < 0 else "draw",
        "elapsed_seconds": time.perf_counter() - started,
    }


def _wilson(wins: int, draws: int, games: int) -> list[float]:
    p = (wins + 0.5 * draws) / games
    z = 1.959963984540054
    denominator = 1.0 + z * z / games
    centre = (p + z * z / (2.0 * games)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / games + z * z / (4.0 * games * games)
    ) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(row["result"] == "win" for row in rows)
    draws = sum(row["result"] == "draw" for row in rows)
    losses = len(rows) - wins - draws
    margins = sorted(float(row["margin"]) for row in rows)
    return {
        "games": len(rows),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_rate": (wins + 0.5 * draws) / len(rows),
        "score_rate_wilson_95": _wilson(wins, draws, len(rows)),
        "mean_margin": sum(margins) / len(margins),
        "median_margin": (margins[(len(margins) - 1) // 2] + margins[len(margins) // 2]) / 2,
        "worst_margin": margins[0],
        "best_margin": margins[-1],
        "mean_left_reward": sum(float(row["left_reward"]) for row in rows) / len(rows),
        "mean_right_reward": sum(float(row["right_reward"]) for row in rows) / len(rows),
    }


def _seed_range(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("seed range must be START:STOP")
    result = tuple(range(int(start), int(stop)))
    if not result:
        raise argparse.ArgumentTypeError("empty seed range")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, default=root / "G304.py")
    parser.add_argument(
        "--right", type=Path,
        default=root / "teammate_meta_route_submission_v1/main.py",
    )
    parser.add_argument(
        "--seeds", type=_seed_range,
        default=tuple(range(2026082700, 2026083724)),
    )
    parser.add_argument("--workers", type=int, default=192)
    parser.add_argument(
        "--output", type=Path,
        default=root / "evaluation/G304-vs-fixed-G001-official-1024seeds.json",
    )
    args = parser.parse_args(argv)
    left = args.left.resolve()
    right = args.right.resolve()
    for path in (left, right):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    tasks = [(seed, seat) for seed in args.seeds for seat in (0, 1)]
    workers = min(max(1, args.workers), len(tasks), os.cpu_count() or 1)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with mp.get_context("spawn").Pool(
        workers, initializer=_init_worker, initargs=(str(left), str(right))
    ) as pool:
        for row in pool.imap_unordered(play, tasks, chunksize=1):
            rows.append(row)
            if len(rows) % 64 == 0 or len(rows) == len(tasks):
                print(f"[{len(rows):04d}/{len(tasks):04d}]", flush=True)
    rows.sort(key=lambda row: (row["seed"], row["left_seat"]))
    by_seat = {
        str(seat): _summary([row for row in rows if row["left_seat"] == seat])
        for seat in (0, 1)
    }
    report = {
        "environment": "kaggle_environments.make('kaggriculture')",
        "kaggle_environments_version": version("kaggle-environments"),
        "configuration": {"official_defaults": True, "marketParams": {}},
        "pairing": "same seeds, both seats; all rates and margins are from the left agent's perspective",
        "left": {"path": str(left), "sha256": hashlib.sha256(left.read_bytes()).hexdigest()},
        "right": {"path": str(right), "sha256": hashlib.sha256(right.read_bytes()).hexdigest()},
        "seeds": list(args.seeds),
        "workers": workers,
        "wall_seconds": time.perf_counter() - started,
        "summary": _summary(rows),
        "by_seat": by_seat,
        "games": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": report["summary"], "by_seat": by_seat,
        "workers": workers, "wall_seconds": report["wall_seconds"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
