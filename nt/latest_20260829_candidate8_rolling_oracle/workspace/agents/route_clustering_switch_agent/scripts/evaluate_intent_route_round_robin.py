#!/usr/bin/env python3
"""Fully paired, dual-seat native round robin for intent-route carriers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def _seeds(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    if separator:
        return tuple(range(int(start), int(stop)))
    return tuple(int(item) for item in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seeds", type=_seeds, default=tuple(range(64)))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    setup = time.perf_counter()
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    families = tuple(bundle.families)
    tasks = []
    keys = []
    for left in range(len(families)):
        for right in range(left + 1, len(families)):
            for seed in args.seeds:
                tasks.append((left, right, seed, -1, -1, -1, -1))
                keys.append((left, right, seed, 0))
                tasks.append((right, left, seed, -1, -1, -1, -1))
                keys.append((left, right, seed, 1))
    setup_seconds = time.perf_counter() - setup
    started = time.perf_counter()
    rewards, audit = bundle.executor.play_audit_batch(np.asarray(tasks, dtype=np.int64))
    rewards = np.asarray(rewards)
    audit = np.asarray(audit)
    elapsed = time.perf_counter() - started
    keys_array = np.asarray(keys, dtype=np.int64)
    scores = np.zeros((len(families), len(families)), dtype=np.float64)
    margins = np.zeros_like(scores)
    games = np.zeros_like(scores, dtype=np.int64)
    route_audits = [[] for _ in families]
    for key, reward, game_audit in zip(keys, rewards, audit):
        left, right, _seed, swapped = key
        own = reward[swapped]
        other = reward[1 - swapped]
        score = float(own > other) + .5 * float(own == other)
        scores[left, right] += score
        scores[right, left] += 1 - score
        margins[left, right] += own - other
        margins[right, left] += other - own
        games[left, right] += 1
        games[right, left] += 1
        route_audits[left].append(game_audit[swapped])
        route_audits[right].append(game_audit[1 - swapped])
    mean_score = np.divide(scores, games, out=np.zeros_like(scores), where=games > 0)
    mean_margin = np.divide(margins, games, out=np.zeros_like(margins), where=games > 0)
    overall = np.divide(scores.sum(axis=1), games.sum(axis=1))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    details = {entry["family"]: entry for entry in metadata["opponent_routes"]}
    ranking = sorted(range(len(families)), key=lambda index: (-overall[index], families[index]))
    summary = {
        "schema_version": 1,
        "engine": "native teammate execution stack",
        "pairing": "every unordered route pair, identical seeds, both seats",
        "seed_count": len(args.seeds),
        "games": len(tasks),
        "setup_seconds": setup_seconds,
        "simulation_seconds": elapsed,
        "games_per_second": len(tasks) / elapsed,
        "ranking": [
            {
                "rank": rank,
                "family": families[index],
                "alias": details[families[index]].get("alias"),
                "score": float(overall[index]),
                "games": int(games[index].sum()),
                "selected": bool(details[families[index]].get("selected")),
                "zero_macro_failure_rate": float(np.mean(
                    np.asarray(route_audits[index])[:, :2].sum(axis=1) == 0
                )),
                "mean_macro_unit_failures": float(np.mean(
                    np.asarray(route_audits[index])[:, 0]
                )),
                "mean_macro_market_failures": float(np.mean(
                    np.asarray(route_audits[index])[:, 1]
                )),
            }
            for rank, index in enumerate(ranking, 1)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, families=np.asarray(families), keys=keys_array,
        rewards=rewards, audit=audit, mean_score=mean_score, mean_margin=mean_margin,
        games=games, overall_score=overall, seeds=np.asarray(args.seeds),
    )
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "games": len(tasks), "seconds": elapsed,
        "games_per_second": len(tasks) / elapsed,
        "top10": summary["ranking"][:10],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
