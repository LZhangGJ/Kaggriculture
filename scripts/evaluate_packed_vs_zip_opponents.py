#!/usr/bin/env python3
"""Paired official-engine league against public agents extracted from the ZIP."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import random
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


_CANDIDATE_CODE: Any = None
_CANDIDATE_PATH = ""
_OPPONENT_CODES: dict[str, tuple[Any, str]] = {}
_FORCED_OPENING = "nash"


class LoadedAgent:
    def __init__(self, code: Any, path: str, name: str) -> None:
        module = types.ModuleType(name)
        module.__file__ = path
        sys.modules[name] = module
        exec(code, module.__dict__)
        self.namespace = module.__dict__
        self.agent = module.__dict__["agent"]
        parameters = inspect.signature(self.agent).parameters.values()
        self.takes_configuration = any(
            parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
            or parameter.name in {"config", "configuration"}
            for parameter in parameters
        ) or len(inspect.signature(self.agent).parameters) >= 2

    def __call__(self, observation: Any, configuration: Any = None):
        if self.takes_configuration:
            return self.agent(observation, configuration)
        return self.agent(observation)


def _init_worker(candidate: str, opponents: dict[str, str], forced_opening: str) -> None:
    global _CANDIDATE_CODE, _CANDIDATE_PATH, _OPPONENT_CODES, _FORCED_OPENING
    _CANDIDATE_PATH = candidate
    _CANDIDATE_CODE = compile(Path(candidate).read_text(encoding="utf-8"), candidate, "exec")
    _OPPONENT_CODES = {
        slug: (compile(Path(path).read_text(encoding="utf-8"), path, "exec"), path)
        for slug, path in opponents.items()
    }
    _FORCED_OPENING = forced_opening


def _opening(seed: int, controller: Any) -> str:
    if _FORCED_OPENING == "original":
        return str(controller.forced_opening)
    if _FORCED_OPENING != "nash":
        available = {str(family) for family, _ in controller.opening_weights}
        if _FORCED_OPENING not in available:
            raise ValueError(f"forced opening {_FORCED_OPENING!r} is not in {sorted(available)}")
        return _FORCED_OPENING
    digest = hashlib.blake2b(
        f"zip-opponents-paired-opening-v1:{seed}".encode(), digest_size=8
    ).digest()
    value = int.from_bytes(digest, "big") / float(1 << 64)
    cumulative = 0.0
    for family, weight in controller.opening_weights:
        cumulative += float(weight)
        if value < cumulative:
            return str(family)
    return str(controller.opening_weights[-1][0])


def _reward(state: Any) -> float:
    reward = getattr(state, "reward", None)
    if reward is None:
        raise RuntimeError(f"missing reward; status={getattr(state, 'status', None)}")
    return float(reward)


def play(task: tuple[str, int, int]) -> dict[str, Any]:
    slug, seed, seat = task
    identity = f"{os.getpid()}_{slug}_{seed}_{seat}"
    candidate = LoadedAgent(_CANDIDATE_CODE, _CANDIDATE_PATH, f"candidate_{identity}")
    opponent_code, opponent_path = _OPPONENT_CODES[slug]
    opponent = LoadedAgent(opponent_code, opponent_path, f"opponent_{identity}")
    controller = candidate.namespace["_S_CONTROLLER"]
    opening = _opening(seed, controller)
    if _FORCED_OPENING != "original":
        controller.forced_opening = opening
    agents = [candidate, opponent] if seat == 0 else [opponent, candidate]
    environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
    started = time.perf_counter()
    environment.run(agents)
    states = list(environment.state)
    statuses = [str(getattr(state, "status", "")) for state in states]
    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(f"bad statuses {slug}/{seed}/{seat}: {statuses}")
    rewards = [_reward(state) for state in states]
    own, other = rewards[seat], rewards[1 - seat]
    margin = own - other
    return {
        "opponent": slug,
        "seed": seed,
        "candidate_seat": seat,
        "opening": opening,
        "final_route": str(controller.current),
        "switched": bool(controller.switched),
        "candidate_reward": own,
        "opponent_reward": other,
        "margin": margin,
        "result": "win" if margin > 0 else "loss" if margin < 0 else "draw",
        "elapsed_seconds": time.perf_counter() - started,
    }


def _score(row: dict[str, Any]) -> float:
    return 1.0 if row["result"] == "win" else 0.5 if row["result"] == "draw" else 0.0


def _seed_bootstrap(rows: Sequence[dict[str, Any]], samples: int = 20_000) -> list[float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), []).append(_score(row))
    units = [sum(values) / len(values) for values in grouped.values()]
    rng = random.Random(20260826 + len(rows))
    estimates = sorted(
        sum(rng.choice(units) for _ in units) / len(units) for _ in range(samples)
    )
    return [estimates[int(0.025 * samples)], estimates[int(0.975 * samples) - 1]]


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
        "paired_seed_bootstrap_95": _seed_bootstrap(rows),
        "mean_margin": sum(float(row["margin"]) for row in rows) / len(rows),
        "switch_rate": sum(bool(row["switched"]) for row in rows) / len(rows),
    }


def _seed_range(value: str) -> tuple[int, ...]:
    if "," in value:
        result = tuple(int(seed) for seed in value.split(",") if seed)
        if result:
            return result
    start, separator, stop = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("expected START:STOP or comma-separated seeds")
    result = tuple(range(int(start), int(stop)))
    if not result:
        raise argparse.ArgumentTypeError("empty seed range")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=root / "teammate_meta_route_175_single.py")
    parser.add_argument("--opponents", type=Path, default=root / "external/kaggriculture-public-opponents")
    parser.add_argument("--seeds", type=_seed_range, default=tuple(range(2026082600, 2026082664)))
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument(
        "--forced-opening", choices=("original", "nash", "G001", "G136"), default="nash",
        help="Keep the candidate's deployed opening, use the Nash draw, or force one opening.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "evaluation/packed-175-vs-kaggriculture-zip-strong-64seeds.json",
    )
    args = parser.parse_args(argv)
    candidate = args.candidate.resolve()
    opponents_dir = args.opponents.resolve()
    manifest = json.loads((opponents_dir / "manifest.json").read_text(encoding="utf-8"))
    opponents = {
        str(row["slug"]): str((opponents_dir / str(row["file"])).resolve())
        for row in manifest["agents"]
    }
    tasks = [
        (slug, seed, seat)
        for slug in opponents
        for seed in args.seeds
        for seat in (0, 1)
    ]
    workers = min(max(1, args.workers), len(tasks), os.cpu_count() or 1)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with mp.get_context("spawn").Pool(
        workers, initializer=_init_worker,
        initargs=(str(candidate), opponents, args.forced_opening),
    ) as pool:
        for row in pool.imap_unordered(play, tasks, chunksize=1):
            rows.append(row)
            if len(rows) % 32 == 0 or len(rows) == len(tasks):
                print(f"[{len(rows):04d}/{len(tasks):04d}]", flush=True)
    rows.sort(key=lambda row: (row["opponent"], row["seed"], row["candidate_seat"]))
    versus = {
        slug: _summary([row for row in rows if row["opponent"] == slug])
        for slug in opponents
    }
    by_seat = {
        str(seat): _summary([row for row in rows if row["candidate_seat"] == seat])
        for seat in (0, 1)
    }
    by_opening = {
        opening: _summary([row for row in rows if row["opening"] == opening])
        for opening in sorted({str(row["opening"]) for row in rows})
    }
    report = {
        "environment": "kaggle_environments.make('kaggriculture')",
        "kaggle_environments_version": version("kaggle-environments"),
        "official_default_market_params": True,
        "pairing": (
            "same seeds, both seats, candidate's original deployed opening"
            if args.forced_opening == "original" else
            f"same seeds, both seats, forced opening {args.forced_opening}"
            if args.forced_opening != "nash" else
            "same seeds, both seats, identical deterministic Nash opening per seed"
        ),
        "forced_opening": args.forced_opening,
        "candidate": {
            "path": str(candidate),
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        },
        "opponent_manifest": manifest,
        "seeds": list(args.seeds),
        "workers": workers,
        "wall_seconds": time.perf_counter() - started,
        "summary": _summary(rows),
        "versus": versus,
        "by_seat": by_seat,
        "by_opening": by_opening,
        "games": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": report["summary"], "versus": versus,
        "by_seat": by_seat, "by_opening": by_opening,
        "wall_seconds": report["wall_seconds"], "output": str(args.output),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
