"""Paired official-engine A/B for the current and opponent-aware SELL rankers."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence


for _variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_variable] = "1"

from kaggle_environments import make

import main
from kaggrl import CommunityAgent, community_agent_specs
from meta_agent.src.opponent_sell_ranker import CausalOpponentSellRanker


VARIANTS = ("current", "opponent_aware")
STRONG_OPPONENTS = (
    "public_b85", "schedule_93311715", "frontier_soil",
    "adaptive_farming", "kaito_v35",
)
DEFAULT_SEEDS = tuple(range(20260824, 20260840))
_RANKER = CausalOpponentSellRanker()


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _csv_names(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    available = {spec.name for spec in community_agent_specs()}
    unknown = sorted(set(result) - available)
    if not result or unknown:
        raise argparse.ArgumentTypeError(f"unknown or empty opponents: {unknown}")
    return result


def _paired_salt(opponent: str, seed: int, seat: int) -> int:
    payload = f"sell-ranker-ab-v1:{opponent}:{seed}:{seat}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _candidate_agent(observation: Any, configuration: Any = None) -> dict[str, Any]:
    action = main.agent(observation, configuration)
    return _RANKER.rerank_action(observation, action)


def _reward(state: Any) -> float:
    value = getattr(state, "reward", None)
    if value is None:
        raise RuntimeError(f"episode ended without reward: {getattr(state, 'status', None)}")
    return float(value)


def play(task: tuple[str, str, int, int]) -> dict[str, Any]:
    variant, opponent_name, seed, submission_seat = task
    salt = _paired_salt(opponent_name, seed, submission_seat)
    main._SELECTOR.fixed_mixture_salt = salt
    main._SELECTOR.mixture_salt = salt
    opponent = CommunityAgent(opponent_name)
    submission = main.agent if variant == "current" else _candidate_agent
    agents = [submission, opponent] if submission_seat == 0 else [opponent, submission]
    environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
    started = time.perf_counter()
    environment.run(agents)
    elapsed = time.perf_counter() - started
    states = list(environment.state)
    statuses = [str(getattr(state, "status", "")) for state in states]
    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(
            f"bad statuses variant={variant} opponent={opponent_name} "
            f"seed={seed} seat={submission_seat}: {statuses}"
        )
    rewards = [_reward(state) for state in states]
    own, other = rewards[submission_seat], rewards[1 - submission_seat]
    margin = own - other
    return {
        "variant": variant,
        "opponent": opponent_name,
        "seed": seed,
        "submission_seat": submission_seat,
        "opening_salt": salt,
        "opening_route_id": main._SELECTOR.opening_route_id,
        "submission_reward": own,
        "opponent_reward": other,
        "margin": margin,
        "result": "win" if margin > 0 else "loss" if margin < 0 else "draw",
        "ranker": _RANKER.stats() if variant == "opponent_aware" else {},
        "elapsed_seconds": elapsed,
    }


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(row["result"] == "win" for row in rows)
    draws = sum(row["result"] == "draw" for row in rows)
    by_opponent = {}
    for opponent in sorted({row["opponent"] for row in rows}):
        subset = [row for row in rows if row["opponent"] == opponent]
        local_wins = sum(row["result"] == "win" for row in subset)
        local_draws = sum(row["result"] == "draw" for row in subset)
        by_opponent[opponent] = {
            "games": len(subset),
            "wins": local_wins,
            "draws": local_draws,
            "losses": len(subset) - local_wins - local_draws,
            "score_rate": (local_wins + 0.5 * local_draws) / len(subset),
            "mean_margin": sum(row["margin"] for row in subset) / len(subset),
        }
    return {
        "games": len(rows),
        "wins": wins,
        "draws": draws,
        "losses": len(rows) - wins - draws,
        "score_rate": (wins + 0.5 * draws) / len(rows),
        "mean_margin": sum(row["margin"] for row in rows) / len(rows),
        "versus": by_opponent,
    }


def _paired(games: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for game in games:
        key = (game["opponent"], game["seed"], game["submission_seat"])
        grouped.setdefault(key, {})[game["variant"]] = game
    deltas = []
    flips = {"loss_to_win": 0, "win_to_loss": 0, "other": 0}
    route_mismatches = 0
    changed_games = 0
    eligible_games = 0
    changed_turns = 0
    eligible_turns = 0
    for pair in grouped.values():
        if set(pair) != set(VARIANTS):
            raise RuntimeError(f"incomplete pair: {pair.keys()}")
        baseline, candidate = pair["current"], pair["opponent_aware"]
        route_mismatches += int(
            baseline["opening_route_id"] != candidate["opening_route_id"]
        )
        delta = candidate["margin"] - baseline["margin"]
        deltas.append(delta)
        stats = candidate["ranker"]
        changed_games += int(stats["changed_turns"] > 0)
        eligible_games += int(stats["eligible_turns"] > 0)
        changed_turns += stats["changed_turns"]
        eligible_turns += stats["eligible_turns"]
        before, after = baseline["result"], candidate["result"]
        if before == "loss" and after == "win":
            flips["loss_to_win"] += 1
        elif before == "win" and after == "loss":
            flips["win_to_loss"] += 1
        elif before != after:
            flips["other"] += 1
    values = sorted(float(value) for value in deltas)
    n = len(values)
    return {
        "pairs": n,
        "route_mismatches": route_mismatches,
        "mean_margin_delta": sum(values) / max(1, n),
        "median_margin_delta": values[n // 2] if n else 0.0,
        "improved_pairs": sum(value > 0 for value in values),
        "tied_pairs": sum(value == 0 for value in values),
        "worsened_pairs": sum(value < 0 for value in values),
        "result_flips": flips,
        "eligible_games": eligible_games,
        "changed_games": changed_games,
        "eligible_turns": eligible_turns,
        "changed_turns": changed_turns,
    }


def main_cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponents", type=_csv_names, default=STRONG_OPPONENTS)
    parser.add_argument("--seeds", type=_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--seats", type=_csv_ints, default=(0, 1))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("evaluation/sell-ranker-ab-official-hard-16seeds.json"),
    )
    args = parser.parse_args(argv)
    if any(seat not in (0, 1) for seat in args.seats):
        parser.error("--seats may contain only 0 and 1")
    tasks = [
        (variant, opponent, seed, seat)
        for opponent in args.opponents
        for seed in args.seeds
        for seat in args.seats
        for variant in VARIANTS
    ]
    context = mp.get_context("spawn")
    worker_count = min(args.workers or (os.cpu_count() or 1), len(tasks))
    games: list[dict[str, Any]] = []
    started = time.perf_counter()
    with context.Pool(worker_count) as pool:
        for game in pool.imap_unordered(play, tasks, chunksize=1):
            games.append(game)
            print(
                f"[{len(games):03d}/{len(tasks):03d}] {game['variant']} "
                f"{game['opponent']} seed={game['seed']} seat={game['submission_seat']} "
                f"{game['result']} margin={game['margin']:+.0f}",
                flush=True,
            )
    report = {
        "environment": "kaggle_environments.make('kaggriculture')",
        "kaggle_environments_version": version("kaggle-environments"),
        "configuration": {"marketParams": {}, "official_defaults": True},
        "variants": list(VARIANTS),
        "opponents": list(args.opponents),
        "seeds": list(args.seeds),
        "seats": list(args.seats),
        "workers": worker_count,
        "wall_seconds": time.perf_counter() - started,
        "summary": {
            variant: _summarize([game for game in games if game["variant"] == variant])
            for variant in VARIANTS
        },
        "paired": _paired(games),
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": report["summary"], "paired": report["paired"]}, indent=2))
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main_cli()
