#!/usr/bin/env python3
"""Paired official-engine league against NT's reconstructed leaderboard top 40."""

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

from meta_agent.src.nt_front40_routes import NtFront40TraceAgent


_CANDIDATE_CODE: Any = None
_CANDIDATE_PATH = ""
_SPECS: dict[str, dict[str, Any]] = {}
_SOURCE_CODES: dict[str, tuple[Any, str]] = {}
_FORCED_OPENING: str = "G001"
_TRACE_CACHE: dict[str, NtFront40TraceAgent] = {}


class LoadedAgent:
    def __init__(self, code: Any, path: str, name: str) -> None:
        module = types.ModuleType(name)
        module.__file__ = path
        sys.modules[name] = module
        self.name = name
        self.module = module
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

    def close(self) -> None:
        """Release the per-game module and its large embedded route state."""
        self.agent = None
        self.namespace = {}
        if sys.modules.get(self.name) is self.module:
            del sys.modules[self.name]
        self.module = None


def _init_worker(candidate: str, specs: dict[str, dict[str, Any]], forced: str) -> None:
    global _CANDIDATE_CODE, _CANDIDATE_PATH, _SPECS, _SOURCE_CODES, _FORCED_OPENING
    _CANDIDATE_PATH = candidate
    _CANDIDATE_CODE = compile(Path(candidate).read_text(encoding="utf-8"), candidate, "exec")
    _SPECS = specs
    _FORCED_OPENING = forced
    _SOURCE_CODES = {}
    for slug, spec in specs.items():
        if spec["type"] == "source":
            path = str(spec["source"])
            _SOURCE_CODES[slug] = (
                compile(Path(path).read_text(encoding="utf-8"), path, "exec"), path
            )


def _opening(seed: int, controller: Any) -> str:
    if _FORCED_OPENING == "original":
        return str(controller.forced_opening)
    if _FORCED_OPENING != "nash":
        return _FORCED_OPENING
    digest = hashlib.blake2b(f"nt-front40-paired-opening-v1:{seed}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / float(1 << 64)
    cumulative = 0.0
    for family, weight in controller.opening_weights:
        cumulative += float(weight)
        if value < cumulative:
            return str(family)
    return str(controller.opening_weights[-1][0])


def _opponent(slug: str, identity: str) -> Any:
    spec = _SPECS[slug]
    if spec["type"] == "source":
        code, path = _SOURCE_CODES[slug]
        return LoadedAgent(code, path, f"opponent_{identity}")
    if slug not in _TRACE_CACHE:
        _TRACE_CACHE.clear()
        _TRACE_CACHE[slug] = NtFront40TraceAgent(
            spec["bank"], spec["first_route_ids"],
            second_route_ids=spec.get("second_route_ids"), manage_market=True,
        )
    return _TRACE_CACHE[slug]


def _reward(state: Any) -> float:
    reward = getattr(state, "reward", None)
    if reward is None:
        raise RuntimeError(f"missing reward; status={getattr(state, 'status', None)}")
    return float(reward)


def play(task: tuple[str, int, int]) -> dict[str, Any]:
    slug, seed, seat = task
    identity = f"{os.getpid()}_{slug}_{seed}_{seat}"
    candidate = LoadedAgent(_CANDIDATE_CODE, _CANDIDATE_PATH, f"candidate_{identity}")
    opponent = _opponent(slug, identity)
    controller = candidate.namespace["_S_CONTROLLER"]
    opening = _opening(seed, controller)
    if _FORCED_OPENING != "original":
        controller.forced_opening = opening
    agents = [candidate, opponent] if seat == 0 else [opponent, candidate]
    try:
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
            "opponent": slug, "seed": seed, "candidate_seat": seat,
            "opening": opening, "final_route": str(controller.current),
            "switched": bool(controller.switched), "candidate_reward": own,
            "opponent_reward": other, "margin": margin,
            "result": "win" if margin > 0 else "loss" if margin < 0 else "draw",
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        candidate.close()
        if isinstance(opponent, LoadedAgent):
            opponent.close()


def _score(row: dict[str, Any]) -> float:
    return 1.0 if row["result"] == "win" else 0.5 if row["result"] == "draw" else 0.0


def _seed_bootstrap(rows: Sequence[dict[str, Any]], samples: int = 20_000) -> list[float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), []).append(_score(row))
    units = [sum(values) / len(values) for values in grouped.values()]
    rng = random.Random(20260826 + len(rows))
    estimates = sorted(sum(rng.choice(units) for _ in units) / len(units) for _ in range(samples))
    return [estimates[int(.025 * samples)], estimates[int(.975 * samples) - 1]]


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(row["result"] == "win" for row in rows)
    draws = sum(row["result"] == "draw" for row in rows)
    losses = len(rows) - wins - draws
    return {
        "games": len(rows), "wins": wins, "draws": draws, "losses": losses,
        "score_rate": (wins + .5 * draws) / len(rows),
        "paired_seed_bootstrap_95": _seed_bootstrap(rows),
        "mean_margin": sum(float(row["margin"]) for row in rows) / len(rows),
        "switch_rate": sum(bool(row["switched"]) for row in rows) / len(rows),
    }


def _seed_range(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("expected START:STOP")
    result = tuple(range(int(start), int(stop)))
    if not result:
        raise argparse.ArgumentTypeError("empty seed range")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=root / "main.py")
    parser.add_argument("--manifest", type=Path, default=root / "external/kaggriculture-nt-front40-opponents/manifest.json")
    parser.add_argument("--seeds", type=_seed_range, default=tuple(range(2026082600, 2026082664)))
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument(
        "--max-tasks-per-child", type=int, default=8,
        help="Recycle workers to bound memory retained by the simulator and agent modules.",
    )
    parser.add_argument(
        "--forced-opening", choices=("original", "nash", "G001", "G136"),
        default="G001",
        help="Use original to preserve the standalone candidate's deployed opening.",
    )
    parser.add_argument("--output", type=Path, default=root / "evaluation/fixed-G001-vs-nt-front40-official-64seeds.json")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Progress file; existing compatible rows are resumed automatically.",
    )
    args = parser.parse_args(argv)
    candidate = args.candidate.resolve()
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    checkpoint = args.checkpoint or Path(str(args.output) + ".partial.json")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    specs: dict[str, dict[str, Any]] = {}
    for row in manifest["agents"]:
        spec = dict(row)
        if spec["type"] == "trace_bank":
            spec["bank"] = str((root / spec["bank"]).resolve())
        else:
            spec["source"] = str((root / spec["source"]).resolve())
        specs[str(spec["slug"])] = spec
    all_tasks = [(slug, seed, seat) for slug in specs for seed in args.seeds for seat in (0, 1)]
    rows: list[dict[str, Any]] = []
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        expected = {
            "candidate_sha256": candidate_sha256,
            "seeds": list(args.seeds),
            "forced_opening": args.forced_opening,
            "opponents": list(specs),
        }
        actual = {key: saved.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(f"incompatible checkpoint {checkpoint}: {actual!r}")
        rows = list(saved.get("games", []))
    completed = {
        (str(row["opponent"]), int(row["seed"]), int(row["candidate_seat"]))
        for row in rows
    }
    tasks = [task for task in all_tasks if task not in completed]
    workers = min(max(1, args.workers), max(1, len(tasks)), os.cpu_count() or 1)

    def save_checkpoint() -> None:
        payload = {
            "candidate_sha256": candidate_sha256,
            "seeds": list(args.seeds),
            "forced_opening": args.forced_opening,
            "opponents": list(specs),
            "complete": len(rows) == len(all_tasks),
            "games": rows,
        }
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(checkpoint) + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.replace(checkpoint)

    started = time.perf_counter()
    if tasks:
        with mp.get_context("spawn").Pool(
            workers, initializer=_init_worker,
            initargs=(str(candidate), specs, args.forced_opening),
            maxtasksperchild=max(1, args.max_tasks_per_child),
        ) as pool:
            since_checkpoint = 0
            for row in pool.imap_unordered(play, tasks, chunksize=1):
                rows.append(row)
                since_checkpoint += 1
                if since_checkpoint >= 40 or len(rows) == len(all_tasks):
                    save_checkpoint()
                    since_checkpoint = 0
                    print(f"[{len(rows):04d}/{len(all_tasks):04d}]", flush=True)
            if since_checkpoint:
                save_checkpoint()
    rows.sort(key=lambda row: (row["opponent"], row["seed"], row["candidate_seat"]))
    versus = {slug: _summary([row for row in rows if row["opponent"] == slug]) for slug in specs}
    report = {
        "environment": "kaggle_environments.make('kaggriculture')",
        "kaggle_environments_version": version("kaggle-environments"),
        "official_default_market_params": True,
        "pairing": f"same seeds, both seats, forced opening {args.forced_opening}",
        "opponent_fidelity": manifest["warning"],
        "candidate": {"path": str(candidate), "sha256": candidate_sha256},
        "opponent_manifest": manifest, "seeds": list(args.seeds), "workers": workers,
        "wall_seconds": time.perf_counter() - started, "summary": _summary(rows),
        "versus": versus,
        "by_seat": {str(seat): _summary([row for row in rows if row["candidate_seat"] == seat]) for seat in (0, 1)},
        "games": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_checkpoint()
    print(json.dumps({"summary": report["summary"], "versus": versus, "by_seat": report["by_seat"], "wall_seconds": report["wall_seconds"], "output": str(args.output)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
