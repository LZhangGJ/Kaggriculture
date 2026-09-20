"""Evaluate the bundled submission against audited public opponents.

This harness deliberately uses the installed official Kaggle interpreter.  The
bundled C++ simulator is for training throughput and is not the scoring source
for this report.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence


for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_variable] = "1"

from kaggle_environments import make

import main
from kaggrl import CommunityAgent, community_agent_specs


STRONG_OPPONENTS = (
    "public_b85",
    "schedule_93311715",
    "frontier_soil",
    "adaptive_farming",
    "kaito_v35",
)
DEFAULT_SEEDS = (
    20260824,
    20260825,
    20260826,
    20260827,
    20260828,
    20260829,
    20260830,
    20260831,
    20260832,
    20260833,
    20260834,
    20260835,
    20260836,
    20260837,
    20260838,
    20260839,
)


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _csv_names(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one opponent")
    available = {spec.name for spec in community_agent_specs()}
    unknown = sorted(set(result) - available)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown opponents: {', '.join(unknown)}; available: {', '.join(sorted(available))}"
        )
    return result


def _reward(state: Any) -> float:
    value = getattr(state, "reward", None)
    if value is None:
        raise RuntimeError(f"episode ended without a reward; status={getattr(state, 'status', None)!r}")
    return float(value)


def play(opponent_name: str, seed: int, submission_seat: int) -> dict[str, Any]:
    opponent = CommunityAgent(opponent_name)
    agents = [main.agent, opponent] if submission_seat == 0 else [opponent, main.agent]
    environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
    started = time.perf_counter()
    environment.run(agents)
    elapsed = time.perf_counter() - started

    states = list(environment.state)
    statuses = [str(getattr(state, "status", "")) for state in states]
    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(
            f"official episode did not finish cleanly: opponent={opponent_name}, "
            f"seed={seed}, seat={submission_seat}, statuses={statuses}"
        )
    rewards = [_reward(state) for state in states]
    own = rewards[submission_seat]
    other = rewards[1 - submission_seat]
    margin = own - other
    selector = getattr(main, "_SELECTOR", None)
    return {
        "opponent": opponent_name,
        "seed": seed,
        "submission_seat": submission_seat,
        "opening_salt": int(selector.mixture_salt) if selector is not None else None,
        "opening_route_id": selector.opening_route_id if selector is not None else None,
        "submission_reward": own,
        "opponent_reward": other,
        "margin": margin,
        "result": "win" if margin > 0 else "loss" if margin < 0 else "draw",
        "elapsed_seconds": elapsed,
    }


def _play_task(task: tuple[str, int, int]) -> dict[str, Any]:
    return play(*task)


def summarize(games: Sequence[dict[str, Any]]) -> dict[str, Any]:
    versus: dict[str, Any] = {}
    for opponent in sorted({str(game["opponent"]) for game in games}):
        rows = [game for game in games if game["opponent"] == opponent]
        wins = sum(game["result"] == "win" for game in rows)
        draws = sum(game["result"] == "draw" for game in rows)
        versus[opponent] = {
            "games": len(rows),
            "wins": wins,
            "draws": draws,
            "losses": len(rows) - wins - draws,
            "score_rate": (wins + 0.5 * draws) / len(rows),
            "mean_margin": sum(float(game["margin"]) for game in rows) / len(rows),
        }
    wins = sum(game["result"] == "win" for game in games)
    draws = sum(game["result"] == "draw" for game in games)
    return {
        "games": len(games),
        "wins": wins,
        "draws": draws,
        "losses": len(games) - wins - draws,
        "score_rate": (wins + 0.5 * draws) / len(games),
        "mean_margin": sum(float(game["margin"]) for game in games) / len(games),
        "versus": versus,
    }


def main_cli(argv: Sequence[str] | None = None) -> None:
    defaults = ",".join(STRONG_OPPONENTS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponents", type=_csv_names, default=_csv_names(defaults))
    parser.add_argument("--seeds", type=_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--seats", type=_csv_ints, default=(0, 1))
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Spawned worker processes; 0 uses min(CPU count, task count).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/community-official-hard-16seeds.json"),
    )
    args = parser.parse_args(argv)
    if any(seat not in (0, 1) for seat in args.seats):
        parser.error("--seats may contain only 0 and 1")
    if args.workers < 0:
        parser.error("--workers must be non-negative")

    games: list[dict[str, Any]] = []
    tasks = [
        (opponent, seed, seat)
        for opponent in args.opponents
        for seed in args.seeds
        for seat in args.seats
    ]
    total = len(tasks)
    suite_started = time.perf_counter()
    context = mp.get_context("spawn")
    requested_workers = args.workers or (os.cpu_count() or 1)
    worker_count = min(requested_workers, total)
    with context.Pool(processes=worker_count) as pool:
        for game in pool.imap_unordered(_play_task, tasks, chunksize=1):
            games.append(game)
            print(
                f"[{len(games):03d}/{total:03d}] {game['opponent']} "
                f"seed={game['seed']} seat={game['submission_seat']} "
                f"{game['result']} {game['submission_reward']:.0f}:"
                f"{game['opponent_reward']:.0f} margin={game['margin']:+.0f} "
                f"time={game['elapsed_seconds']:.2f}s",
                flush=True,
            )

    report = {
        "environment": "kaggle_environments.make('kaggriculture')",
        "kaggle_environments_version": version("kaggle-environments"),
        "configuration": {"marketParams": {}, "official_defaults": True},
        "seeds": list(args.seeds),
        "seats": list(args.seats),
        "workers": worker_count,
        "wall_seconds": time.perf_counter() - suite_started,
        "summary": summarize(games),
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(
        f"summary games={summary['games']} W/D/L={summary['wins']}/"
        f"{summary['draws']}/{summary['losses']} score_rate={summary['score_rate']:.1%} "
        f"mean_margin={summary['mean_margin']:+.1f} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main_cli()
