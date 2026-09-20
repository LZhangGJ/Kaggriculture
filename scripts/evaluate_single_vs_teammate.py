#!/usr/bin/env python3
"""Paired official-engine evaluation of the packed agent vs teammate original."""

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


_CANDIDATE: dict[str, Any] | None = None
_OPPONENT: dict[str, Any] | None = None


def _load(path: str, name: str) -> dict[str, Any]:
    source = Path(path).read_text(encoding="utf-8")
    module = types.ModuleType(name)
    module.__file__ = path
    sys.modules[name] = module
    exec(compile(source, path, "exec"), module.__dict__)
    return module.__dict__


def _init_worker(candidate: str, opponent: str) -> None:
    global _CANDIDATE, _OPPONENT
    identity = os.getpid()
    _CANDIDATE = _load(candidate, f"packed_candidate_{identity}")
    _OPPONENT = _load(opponent, f"teammate_original_{identity}")


def _opening(seed: int) -> str:
    assert _CANDIDATE is not None
    digest = hashlib.blake2b(
        f"paired-nash-opening-v1:{seed}".encode(), digest_size=8
    ).digest()
    value = int.from_bytes(digest, "big") / float(1 << 64)
    cumulative = 0.0
    weights = _CANDIDATE["_S_CONTROLLER"].opening_weights
    for family, weight in weights:
        cumulative += float(weight)
        if value < cumulative:
            return str(family)
    return str(weights[-1][0])


def _reward(state: Any) -> float:
    reward = getattr(state, "reward", None)
    if reward is None:
        raise RuntimeError(f"missing reward for status={getattr(state, 'status', None)}")
    return float(reward)


def play(task: tuple[int, int]) -> dict[str, Any]:
    assert _CANDIDATE is not None and _OPPONENT is not None
    seed, seat = task
    opening = _opening(seed)
    controller = _CANDIDATE["_S_CONTROLLER"]
    controller.forced_opening = opening
    candidate = _CANDIDATE["agent"]
    opponent = _OPPONENT["agent"]
    agents = [candidate, opponent] if seat == 0 else [opponent, candidate]
    environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
    started = time.perf_counter()
    environment.run(agents)
    states = list(environment.state)
    statuses = [str(getattr(state, "status", "")) for state in states]
    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(f"bad statuses seed={seed} seat={seat}: {statuses}")
    rewards = [_reward(state) for state in states]
    own, other = rewards[seat], rewards[1 - seat]
    margin = own - other
    return {
        "seed": seed,
        "candidate_seat": seat,
        "opening": opening,
        "final_route": str(controller.current),
        "switched": bool(controller.switched),
        "candidate_reward": own,
        "teammate_reward": other,
        "margin": margin,
        "result": "win" if margin > 0 else "loss" if margin < 0 else "draw",
        "elapsed_seconds": time.perf_counter() - started,
    }


def _wilson(wins: int, draws: int, games: int) -> list[float]:
    # Treat a draw as half a win and report a descriptive Wilson interval.
    p = (wins + 0.5 * draws) / games
    z = 1.959963984540054
    denominator = 1.0 + z * z / games
    centre = (p + z * z / (2.0 * games)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / games + z * z / (4.0 * games * games)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(row["result"] == "win" for row in rows)
    draws = sum(row["result"] == "draw" for row in rows)
    losses = len(rows) - wins - draws
    return {
        "games": len(rows),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_rate": (wins + 0.5 * draws) / len(rows),
        "score_rate_wilson_95": _wilson(wins, draws, len(rows)),
        "mean_margin": sum(row["margin"] for row in rows) / len(rows),
        "switch_rate": sum(bool(row["switched"]) for row in rows) / len(rows),
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
    parser.add_argument("--candidate", type=Path, default=root / "teammate_meta_route_175_single.py")
    parser.add_argument(
        "--opponent", type=Path,
        default=root / "experiments/teammate-route-meta/original/main.py",
    )
    parser.add_argument("--seeds", type=_seed_range, default=tuple(range(2026082500, 2026082564)))
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--output", type=Path,
        default=root / "evaluation/packed-175-vs-teammate-original-official-64seeds.json",
    )
    args = parser.parse_args(argv)
    candidate = args.candidate.resolve()
    opponent = args.opponent.resolve()
    for path in (candidate, opponent):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    tasks = [(seed, seat) for seed in args.seeds for seat in (0, 1)]
    workers = min(max(1, args.workers), len(tasks), os.cpu_count() or 1)
    rows = []
    started = time.perf_counter()
    with mp.get_context("spawn").Pool(
        workers, initializer=_init_worker, initargs=(str(candidate), str(opponent))
    ) as pool:
        for row in pool.imap_unordered(play, tasks, chunksize=1):
            rows.append(row)
            if len(rows) % 16 == 0 or len(rows) == len(tasks):
                print(f"[{len(rows):03d}/{len(tasks):03d}]", flush=True)
    rows.sort(key=lambda row: (row["seed"], row["candidate_seat"]))
    by_seat = {
        str(seat): _summary([row for row in rows if row["candidate_seat"] == seat])
        for seat in (0, 1)
    }
    by_opening = {
        opening: _summary([row for row in rows if row["opening"] == opening])
        for opening in sorted({row["opening"] for row in rows})
    }
    report = {
        "environment": "kaggle_environments.make('kaggriculture')",
        "kaggle_environments_version": version("kaggle-environments"),
        "official_default_market_params": True,
        "candidate": {
            "path": str(candidate),
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        },
        "opponent": {
            "path": str(opponent),
            "sha256": hashlib.sha256(opponent.read_bytes()).hexdigest(),
            "description": "exact teammate original high-score standalone script",
        },
        "pairing": "same seed, both seats, same deterministic Nash opening per seed",
        "seeds": list(args.seeds),
        "workers": workers,
        "wall_seconds": time.perf_counter() - started,
        "summary": _summary(rows),
        "by_seat": by_seat,
        "by_opening": by_opening,
        "games": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": report["summary"],
        "by_seat": by_seat,
        "by_opening": by_opening,
        "wall_seconds": report["wall_seconds"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
