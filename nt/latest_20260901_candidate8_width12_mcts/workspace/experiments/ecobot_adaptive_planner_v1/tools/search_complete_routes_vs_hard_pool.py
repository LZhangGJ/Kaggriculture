#!/usr/bin/env python3
"""Search complete-route counters for hard opponents with a seed holdout."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--merged-receipt", required=True, type=Path)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--train-seeds", type=int, default=32)
    parser.add_argument("--holdout-seeds", type=int, default=32)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    merged = json.loads(args.merged_receipt.read_text(encoding="utf-8"))
    targets = list(merged["route_bands"]["oracle_below_50pct"])
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    families = list(bundle.families)
    route_index = {name: index for index, name in enumerate(families)}
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    details = {row["family"]: row for row in metadata["opponent_routes"]}
    missing = [target for target in targets if target not in route_index]
    if missing:
        raise ValueError(f"hard routes absent from route library: {missing}")

    total_seeds = args.train_seeds + args.holdout_seeds
    tasks: list[tuple[int, int, int, int, int, int, int]] = []
    seats: list[int] = []
    for target in targets:
        opponent = route_index[target]
        for candidate in range(len(families)):
            for seed in range(args.seed_start, args.seed_start + total_seeds):
                tasks.append((candidate, opponent, seed, -1, -1, -1, -1))
                seats.append(0)
                tasks.append((opponent, candidate, seed, -1, -1, -1, -1))
                seats.append(1)

    started = time.perf_counter()
    rewards = np.asarray(
        bundle.executor.play_batch(np.asarray(tasks, dtype=np.int64)),
        dtype=np.float64,
    )
    simulation_seconds = time.perf_counter() - started
    seats_array = np.asarray(seats, dtype=np.int64)
    own = rewards[np.arange(len(rewards)), seats_array]
    other = rewards[np.arange(len(rewards)), 1 - seats_array]
    scores = np.where(own > other, 1.0, np.where(own == other, 0.5, 0.0))
    margins = own - other
    shape = (len(targets), len(families), total_seeds, 2)
    scores = scores.reshape(shape)
    margins = margins.reshape(shape)

    train_score = scores[:, :, :args.train_seeds].mean(axis=(2, 3))
    holdout_score = scores[:, :, args.train_seeds:].mean(axis=(2, 3))
    train_margin = margins[:, :, :args.train_seeds].mean(axis=(2, 3))
    holdout_margin = margins[:, :, args.train_seeds:].mean(axis=(2, 3))
    selected = np.argmax(train_score, axis=1)

    rows: list[dict] = []
    for target_index, target in enumerate(targets):
        candidate = int(selected[target_index])
        holdout_oracle = int(np.argmax(holdout_score[target_index]))
        rows.append({
            "opponent": target,
            "selected_route": families[candidate],
            "selected_alias": details.get(families[candidate], {}).get("alias"),
            "train_score_rate": float(train_score[target_index, candidate]),
            "train_mean_margin": float(train_margin[target_index, candidate]),
            "holdout_score_rate": float(holdout_score[target_index, candidate]),
            "holdout_mean_margin": float(holdout_margin[target_index, candidate]),
            "holdout_posthoc_best_route": families[holdout_oracle],
            "holdout_posthoc_best_score_rate": float(
                holdout_score[target_index, holdout_oracle]
            ),
            "routes_train_at_least_90pct": int(
                (train_score[target_index] >= 0.90).sum()
            ),
            "routes_holdout_at_least_90pct": int(
                (holdout_score[target_index] >= 0.90).sum()
            ),
            "routes_holdout_at_least_50pct": int(
                (holdout_score[target_index] >= 0.50).sum()
            ),
        })

    holdout_rates = np.asarray([row["holdout_score_rate"] for row in rows])
    result = {
        "schema": "kaggriculture.complete-route-hard-pool-search.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "Offline opponent-identity-aware route selection. The holdout is "
            "independent in RNG only and is not a deployable router."
        ),
        "config": {
            "candidate_routes": len(families),
            "targets": targets,
            "seed_start": args.seed_start,
            "train_seeds": args.train_seeds,
            "holdout_seeds": args.holdout_seeds,
            "both_seats": True,
        },
        "summary": {
            "targets": len(targets),
            "holdout_mean_score_rate": float(holdout_rates.mean()),
            "targets_holdout_at_least_90pct": int((holdout_rates >= 0.90).sum()),
            "targets_holdout_at_least_50pct": int((holdout_rates >= 0.50).sum()),
            "holdout_worst_score_rate": float(holdout_rates.min()),
            "games": len(tasks),
            "simulation_seconds": simulation_seconds,
            "games_per_second": len(tasks) / simulation_seconds,
        },
        "opponents": rows,
        "inputs": {
            "source": str(args.source),
            "actions": str(args.actions),
            "metadata": str(args.metadata),
            "merged_receipt": str(args.merged_receipt),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "summary": result["summary"],
        "opponents": rows,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
