#!/usr/bin/env python3
"""Thin serializer for the all-C++ route round-robin kernel."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def _seeds(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    return tuple(range(int(start), int(stop))) if separator else tuple(
        int(item) for item in value.split(",")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seeds", type=_seeds, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    setup = time.perf_counter()
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    setup_seconds = time.perf_counter() - setup
    started = time.perf_counter()
    result = bundle.executor.round_robin(args.seeds)
    simulation_seconds = time.perf_counter() - started
    score = np.asarray(result["score"])
    margin = np.asarray(result["margin"])
    games = np.asarray(result["games"])
    unit = np.asarray(result["mean_unit_failures"])
    market = np.asarray(result["mean_market_failures"])
    total_games = games.sum(axis=1)
    overall = np.divide((score * games).sum(axis=1), total_games)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    details = {entry["family"]: entry for entry in metadata["opponent_routes"]}
    ranking = sorted(range(len(bundle.families)), key=lambda i: (-overall[i], bundle.families[i]))
    payload = {
        "schema_version": 1,
        "engine": "C++ NativeTeammateExecutor.round_robin",
        "routes": len(bundle.families),
        "seed_count": len(args.seeds),
        "pairing": "all unordered pairs, identical seeds, both seats",
        "games": int(games.sum() // 2),
        "setup_seconds": setup_seconds,
        "simulation_seconds": simulation_seconds,
        "games_per_second": int(games.sum() // 2) / simulation_seconds,
        "ranking": [
            {
                "rank": rank,
                "family": bundle.families[index],
                "alias": details[bundle.families[index]].get("alias"),
                "score": float(overall[index]),
                "games": int(total_games[index]),
                "mean_unit_failures": float(np.average(unit[index], weights=games[index])),
                "mean_market_failures": float(np.average(market[index], weights=games[index])),
            }
            for rank, index in enumerate(ranking, 1)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, families=np.asarray(bundle.families), seeds=np.asarray(args.seeds),
        score=score, margin=margin, games=games, overall_score=overall,
        mean_unit_failures=unit, mean_market_failures=market,
    )
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "routes": payload["routes"], "games": payload["games"],
        "seconds": simulation_seconds, "games_per_second": payload["games_per_second"],
        "top10": payload["ranking"][:10],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
