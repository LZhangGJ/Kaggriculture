#!/usr/bin/env python3
"""Paired unseen-seed evaluation of searched switching versus staying."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


CANDIDATE: Any = None
OPPONENT: Any = None
DYNAMIC: Any = None
BASELINE: Any = None
ROUTE_BY_FAMILY: dict[str, str] = {}


def _init(source_path: str, actions_path: str, metadata_path: str, policy_path: str, nash_path: str) -> None:
    global CANDIDATE, OPPONENT, DYNAMIC, BASELINE, ROUTE_BY_FAMILY
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    from meta_agent.src.search_route_policy import SearchRouteController, SearchRoutedTeammateAgent
    from meta_agent.src.teammate_expanded_routes import TeammateExpandedRouteAgent, load_action_tapes

    source = Path(source_path).read_text(encoding="utf-8")
    tapes = load_action_tapes(actions_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    ROUTE_BY_FAMILY = {
        str(value["family"]): str(value["route_id"])
        for value in metadata["opponent_routes"]
    }
    nash = json.loads(Path(nash_path).read_text(encoding="utf-8"))
    weights = [(str(value["family"]), float(value["weight"])) for value in nash["opening_support"]]
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    empty = {**policy, "nodes": []}
    CANDIDATE = TeammateExpandedRouteAgent(source, tapes, f"eval_candidate_{os.getpid()}")
    OPPONENT = TeammateExpandedRouteAgent(source, tapes, f"eval_opponent_{os.getpid()}")
    dynamic_controller = SearchRouteController(policy, ROUTE_BY_FAMILY, weights, rng_seed=1)
    baseline_controller = SearchRouteController(empty, ROUTE_BY_FAMILY, weights, rng_seed=1)
    DYNAMIC = SearchRoutedTeammateAgent(CANDIDATE, dynamic_controller, policy.get("targets", []))
    BASELINE = SearchRoutedTeammateAgent(CANDIDATE, baseline_controller, policy.get("targets", []))


def _play(task: tuple[int, int, str, str, int, int, int]) -> tuple[Any, ...]:
    task_id, variant, opening, opponent_family, seed, seat, opponent_index = task
    try:
        from fast_kaggriculture import Config, FastEnv

        candidate = DYNAMIC if variant else BASELINE
        candidate.controller.forced_opening = opening
        OPPONENT.select(ROUTE_BY_FAMILY[opponent_family])
        agents = [candidate, OPPONENT] if seat == 0 else [OPPONENT, candidate]
        env = FastEnv(Config(), seed)
        observations = list(env.reset(seed))
        while not env.done:
            step = int(env.step_count)
            for player, observation in enumerate(observations):
                observation["player"] = player
                observation["step"] = step
            actions = [agents[player](observations[player], {}) for player in range(2)]
            observations = list(env.step(actions))
        rewards = [float(value) for value in env.rewards]
        own, other = rewards[seat], rewards[1 - seat]
        margin = own - other
        return (
            task_id, variant, opening, opponent_family, opponent_index, seed, seat,
            own, other, margin, float(margin > 0) + 0.5 * float(margin == 0),
            candidate.controller.current, float(candidate.controller.switched), 0.0,
        )
    except Exception:
        return (task_id, variant, opening, opponent_family, opponent_index, seed, seat, 0, 0, 0, 0, "", 0, 1.0)


def _seeds(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    return tuple(range(int(start), int(stop))) if separator else tuple(int(item) for item in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--nash", type=Path, required=True)
    parser.add_argument("--seeds", type=_seeds, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(192, os.cpu_count() or 1))
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    opponents = [str(value["family"]) for value in metadata["opponent_routes"]]
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    openings = [
        str(value["selected"]["opening"])
        for value in policy["nodes"] if value["selected"].get("enabled", True)
    ]
    tasks = []
    for opening in openings:
        for opponent_index, opponent in enumerate(opponents):
            for seed in args.seeds:
                for seat in (0, 1):
                    for variant in (0, 1):
                        tasks.append((len(tasks), variant, opening, opponent, seed, seat, opponent_index))
    started = time.perf_counter()
    rows = []
    with mp.get_context("fork").Pool(
        min(args.workers, len(tasks)), initializer=_init,
        initargs=tuple(str(value.resolve()) for value in (args.source, args.actions, args.metadata, args.policy, args.nash)),
    ) as pool:
        for count, row in enumerate(pool.imap_unordered(_play, tasks, chunksize=1), start=1):
            rows.append(row)
            if count % 100 == 0 or count == len(tasks):
                print(f"completed {count}/{len(tasks)} ({count / max(time.perf_counter() - started, 1e-9):.1f} games/s)", flush=True)
    rows.sort(key=lambda row: row[0])
    errors = sum(int(row[-1]) for row in rows)
    paired = {}
    for row in rows:
        key = (row[2], row[3], row[5], row[6])
        paired.setdefault(key, {})[int(row[1])] = row
    differences = [values[1][10] - values[0][10] for values in paired.values() if 0 in values and 1 in values]
    payload = {
        "schema_version": 1,
        "policy": str(args.policy), "seeds": list(args.seeds),
        "games": len(rows), "errors": errors,
        "openings": openings, "opponents": opponents,
        "baseline_score": float(np.mean([row[10] for row in rows if row[1] == 0])),
        "dynamic_score": float(np.mean([row[10] for row in rows if row[1] == 1])),
        "paired_improvement": float(np.mean(differences)),
        "switch_rate": float(np.mean([row[12] for row in rows if row[1] == 1])),
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("games", "errors", "baseline_score", "dynamic_score", "paired_improvement", "switch_rate")}, indent=2))


if __name__ == "__main__":
    main()
