"""Re-estimate forced-opening payoffs and solve the empirical maximin mix."""

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
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np
from kaggle_environments import make
from scipy.optimize import linprog

from kaggrl import CommunityAgent, community_agent_specs
from meta_agent.src.equilibrium_selector import EquilibriumOpeningSelector
from meta_agent.src.replay_trie_agent import ReplayTrieAgent


ROOT = Path(__file__).resolve().parents[1]
MIXTURE = ROOT / "opening-mixture-v8.json"
DEFAULT_CHECKPOINT = ROOT / "recurrent-route-best.pt"
STRONG_OPPONENTS = (
    "public_b85", "schedule_93311715", "frontier_soil",
    "adaptive_farming", "kaito_v35",
)
DEFAULT_SEEDS = tuple(range(20261001, 20261033))
_POLICIES: dict[str, tuple[ReplayTrieAgent, EquilibriumOpeningSelector]] = {}


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected integers")
    return result


def _csv_names(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    available = {spec.name for spec in community_agent_specs()}
    unknown = sorted(set(result) - available)
    if not result or unknown:
        raise argparse.ArgumentTypeError(f"unknown or empty opponents: {unknown}")
    return result


def _policy(route_id: str, checkpoint: str) -> tuple[ReplayTrieAgent, EquilibriumOpeningSelector]:
    cached = _POLICIES.get(route_id)
    if cached is not None:
        return cached
    selector = EquilibriumOpeningSelector(
        str(ROOT / "route-runtime-v8"),
        str(ROOT / "branch-descriptors-v8.pkl"),
        tree_q_model_path=None,
        recurrent_model_path=checkpoint,
        residual_weight=1.0,
        enable_trading=False,
        epsilon=0.0,
        top_k=16,
        prior_weight=0.0,
        seed=0,
        mixture_path=str(MIXTURE),
        mixture_salt=0,
    )
    if route_id not in selector.selector.route_offsets:
        raise ValueError(f"route missing from runtime: {route_id}")
    selector.opening_support = [(route_id, 1.0)]
    policy = ReplayTrieAgent(
        ROOT / "route-runtime-v8",
        selector_override=selector,
        manage_sells=True,
        lead_sells=True,
        lead_turns=5,
        lead_batch=20,
        lead_max_distance=8,
        repair_weeds=True,
        weed_replay_steps=8,
    )
    cached = policy, selector
    _POLICIES[route_id] = cached
    return cached


def _reward(state: Any) -> float:
    value = getattr(state, "reward", None)
    if value is None:
        raise RuntimeError("episode ended without reward")
    return float(value)


def play(task: tuple[str, str, str, int, int]) -> dict[str, Any]:
    route_id, checkpoint, opponent_name, seed, seat = task
    submission, selector = _policy(route_id, checkpoint)
    opponent = CommunityAgent(opponent_name)
    agents = [submission, opponent] if seat == 0 else [opponent, submission]
    environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
    started = time.perf_counter()
    environment.run(agents)
    states = list(environment.state)
    statuses = [str(getattr(state, "status", "")) for state in states]
    if statuses != ["DONE", "DONE"] or selector.opening_route_id != route_id:
        raise RuntimeError(
            f"invalid forced-route game {route_id}/{opponent_name}/{seed}/{seat}: "
            f"statuses={statuses}, selected={selector.opening_route_id}"
        )
    rewards = [_reward(state) for state in states]
    own, other = rewards[seat], rewards[1 - seat]
    margin = own - other
    return {
        "route_id": route_id, "opponent": opponent_name,
        "seed": seed, "submission_seat": seat,
        "submission_reward": own, "opponent_reward": other,
        "margin": margin,
        "result": "win" if margin > 0 else "loss" if margin < 0 else "draw",
        "elapsed_seconds": time.perf_counter() - started,
    }


def _maximin(payoffs: np.ndarray) -> tuple[np.ndarray, float]:
    routes, opponents = payoffs.shape
    objective = np.zeros(routes + 1, dtype=np.float64)
    objective[-1] = -1.0
    # For every opponent j: sum_i w_i * payoff[i,j] >= value.
    constraints = np.concatenate(
        (-payoffs.T.astype(np.float64), np.ones((opponents, 1))), axis=1
    )
    equality = np.ones((1, routes + 1), dtype=np.float64)
    equality[0, -1] = 0.0
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=np.zeros(opponents),
        A_eq=equality,
        b_eq=np.ones(1),
        bounds=[(0.0, 1.0)] * routes + [(0.0, 1.0)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"maximin solve failed: {result.message}")
    return result.x[:-1], float(result.x[-1])


def main_cli(argv: Sequence[str] | None = None) -> None:
    source = json.loads(MIXTURE.read_text())
    route_rows = list(source["support"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--opponents", type=_csv_names, default=STRONG_OPPONENTS)
    parser.add_argument("--seeds", type=_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--seats", type=_csv_ints, default=(0, 1))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("evaluation/opening-maximin-best-official-32seeds.json"),
    )
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"missing checkpoint: {checkpoint}")
    routes = [str(row["route_id"]) for row in route_rows]
    tasks = [
        (route, str(checkpoint), opponent, seed, seat)
        for route in routes
        for opponent in args.opponents
        for seed in args.seeds
        for seat in args.seats
    ]
    worker_count = min(args.workers or (os.cpu_count() or 1), len(tasks))
    games = []
    started = time.perf_counter()
    with mp.get_context("spawn").Pool(worker_count) as pool:
        for game in pool.imap_unordered(play, tasks, chunksize=1):
            games.append(game)
            if len(games) % 40 == 0 or len(games) == len(tasks):
                print(f"[{len(games):04d}/{len(tasks):04d}]", flush=True)

    payoffs = np.zeros((len(routes), len(args.opponents)), dtype=np.float64)
    cells: dict[str, Any] = {}
    for i, route in enumerate(routes):
        cells[route] = {}
        for j, opponent in enumerate(args.opponents):
            rows = [
                game for game in games
                if game["route_id"] == route and game["opponent"] == opponent
            ]
            wins = sum(row["result"] == "win" for row in rows)
            draws = sum(row["result"] == "draw" for row in rows)
            score = (wins + 0.5 * draws) / len(rows)
            payoffs[i, j] = score
            cells[route][opponent] = {
                "games": len(rows), "wins": wins, "draws": draws,
                "losses": len(rows) - wins - draws, "score_rate": score,
                "mean_margin": sum(row["margin"] for row in rows) / len(rows),
            }
    weights, value = _maximin(payoffs)
    report = {
        "schema_version": 1,
        "environment": "kaggle_environments.make('kaggriculture')",
        "kaggle_environments_version": version("kaggle-environments"),
        "checkpoint": str(checkpoint),
        "opponents": list(args.opponents), "seeds": list(args.seeds),
        "seats": list(args.seats), "workers": worker_count,
        "wall_seconds": time.perf_counter() - started,
        "routes": routes, "payoff_matrix": payoffs.tolist(), "cells": cells,
        "empirical_maximin_score": value,
        "support": [
            {
                "route_id": route,
                "old_weight": float(route_rows[index]["weight"]),
                "new_weight": float(weights[index]),
                "payoff_vector": payoffs[index].tolist(),
            }
            for index, route in enumerate(routes)
            if weights[index] > 1e-10 or float(route_rows[index]["weight"]) > 0.0
        ],
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "empirical_maximin_score": value,
        "support": report["support"],
    }, indent=2))
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main_cli()
