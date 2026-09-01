#!/usr/bin/env python3
"""Validate frozen complete-route choices on independent seeds."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def score(rewards: np.ndarray, seats: np.ndarray) -> np.ndarray:
    own = rewards[np.arange(len(rewards)), seats]
    other = rewards[np.arange(len(rewards)), 1 - seats]
    return np.where(own > other, 1.0, np.where(own == other, 0.5, 0.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    families = list(bundle.families)
    route_index = {name: index for index, name in enumerate(families)}
    opponents = [row["opponent"] for row in oracle["opponents"]]
    full_choice = {row["opponent"]: row["best_route"] for row in oracle["opponents"]}
    compact_routes = [
        row["route"] for row in oracle["greedy_cover_90pct"]["selected_routes"]
    ]

    # Reconstruct the training matrix only to freeze the best choice inside the
    # already-selected compact route set.  Holdout rewards are never used for
    # route selection.
    with np.load(Path(oracle["inputs"]["matrix"]), allow_pickle=False) as saved:
        train_families = [str(value) for value in saved["families"]]
        train_score = np.asarray(saved["score"], dtype=np.float64)
        train_games = np.asarray(saved["games"], dtype=np.int64)
    train_index = {name: index for index, name in enumerate(train_families)}
    for index in range(len(train_families)):
        if train_games[index, index] == 0:
            train_score[index, index] = 0.5
    compact_choice: dict[str, str] = {}
    for opponent in opponents:
        opponent_index = train_index[opponent]
        compact_choice[opponent] = max(
            compact_routes,
            key=lambda route: train_score[train_index[route], opponent_index],
        )

    policies = {
        "g001_fixed": {opponent: "G001" for opponent in opponents},
        "compact10_identity_oracle": compact_choice,
        "full178_identity_oracle": full_choice,
    }
    tasks: list[tuple[int, int, int, int, int, int, int]] = []
    task_meta: list[tuple[str, str, int]] = []
    for policy, choices in policies.items():
        for opponent in opponents:
            candidate = choices[opponent]
            for seed in range(args.seed_start, args.seed_start + args.seed_count):
                tasks.append((
                    route_index[candidate], route_index[opponent], seed,
                    -1, -1, -1, -1,
                ))
                task_meta.append((policy, opponent, 0))
                tasks.append((
                    route_index[opponent], route_index[candidate], seed,
                    -1, -1, -1, -1,
                ))
                task_meta.append((policy, opponent, 1))

    started = time.perf_counter()
    rewards = np.asarray(
        bundle.executor.play_batch(np.asarray(tasks, dtype=np.int64)),
        dtype=np.float64,
    )
    simulation_seconds = time.perf_counter() - started
    seats = np.asarray([row[2] for row in task_meta], dtype=np.int64)
    scores = score(rewards, seats)
    margins = (
        rewards[np.arange(len(rewards)), seats]
        - rewards[np.arange(len(rewards)), 1 - seats]
    )

    summaries: dict[str, dict] = {}
    per_opponent: list[dict] = []
    offset = 0
    games_per_opponent = args.seed_count * 2
    for policy in policies:
        policy_scores: list[np.ndarray] = []
        policy_margins: list[np.ndarray] = []
        for opponent in opponents:
            stop = offset + games_per_opponent
            values = scores[offset:stop]
            delta = margins[offset:stop]
            policy_scores.append(values)
            policy_margins.append(delta)
            per_opponent.append({
                "policy": policy,
                "opponent": opponent,
                "selected_route": policies[policy][opponent],
                "games": games_per_opponent,
                "score_rate": float(values.mean()),
                "mean_margin": float(delta.mean()),
            })
            offset = stop
        route_rates = np.asarray([values.mean() for values in policy_scores])
        all_scores = np.concatenate(policy_scores)
        all_margins = np.concatenate(policy_margins)
        summaries[policy] = {
            "games": int(len(all_scores)),
            "score_rate": float(all_scores.mean()),
            "mean_margin": float(all_margins.mean()),
            "opponents_at_least_90pct": int((route_rates >= 0.90).sum()),
            "opponents_at_least_50pct": int((route_rates >= 0.50).sum()),
            "worst_opponent_score": float(route_rates.min()),
        }

    result = {
        "schema": "kaggriculture.complete-route-portfolio-holdout.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "Identity-aware choices are offline upper bounds. Choices were "
            "frozen from the 8-seed matrix and tested on independent seeds."
        ),
        "config": {
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "seats": [0, 1],
            "opponents": len(opponents),
            "compact_routes": compact_routes,
        },
        "summary": summaries,
        "simulation": {
            "games": len(tasks),
            "seconds": simulation_seconds,
            "games_per_second": len(tasks) / simulation_seconds,
        },
        "opponents": per_opponent,
        "inputs": {
            "source": str(args.source),
            "actions": str(args.actions),
            "metadata": str(args.metadata),
            "oracle": str(args.oracle),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "summary": summaries,
        "simulation": result["simulation"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
