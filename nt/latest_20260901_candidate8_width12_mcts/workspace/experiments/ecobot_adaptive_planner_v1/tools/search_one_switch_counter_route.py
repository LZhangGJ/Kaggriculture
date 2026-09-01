#!/usr/bin/env python3
"""Search one-switch complete-route counters against one frozen opponent."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def parse_ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--opening", default="G001")
    parser.add_argument("--opponent", required=True)
    parser.add_argument(
        "--checkpoints", type=parse_ints,
        default=parse_ints("24,48,72,96,120,144,168,192,216,240,288,360"),
    )
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--train-seeds", type=int, default=32)
    parser.add_argument("--holdout-seeds", type=int, default=32)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    families = list(bundle.families)
    opening = bundle.index(args.opening)
    opponent = bundle.index(args.opponent)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    details = {row["family"]: row for row in metadata["opponent_routes"]}
    total_seeds = args.train_seeds + args.holdout_seeds

    variants = [(-1, opening)] + [
        (checkpoint, target)
        for checkpoint in args.checkpoints
        for target in range(len(families))
    ]
    tasks: list[tuple[int, int, int, int, int, int, int]] = []
    seats: list[int] = []
    for checkpoint, target in variants:
        for seed in range(args.seed_start, args.seed_start + total_seeds):
            tasks.append((
                opening, opponent, seed,
                checkpoint, target, -1, -1,
            ))
            seats.append(0)
            tasks.append((
                opponent, opening, seed,
                -1, -1, checkpoint, target,
            ))
            seats.append(1)

    started = time.perf_counter()
    rewards = np.asarray(
        bundle.executor.play_batch(np.asarray(tasks, dtype=np.int64)),
        dtype=np.float64,
    )
    simulation_seconds = time.perf_counter() - started
    seat_array = np.asarray(seats, dtype=np.int64)
    own = rewards[np.arange(len(rewards)), seat_array]
    other = rewards[np.arange(len(rewards)), 1 - seat_array]
    scores = np.where(own > other, 1.0, np.where(own == other, 0.5, 0.0))
    margins = own - other
    scores = scores.reshape(len(variants), total_seeds, 2)
    margins = margins.reshape(len(variants), total_seeds, 2)
    train_score = scores[:, :args.train_seeds].mean(axis=(1, 2))
    train_margin = margins[:, :args.train_seeds].mean(axis=(1, 2))
    holdout_score = scores[:, args.train_seeds:].mean(axis=(1, 2))
    holdout_margin = margins[:, args.train_seeds:].mean(axis=(1, 2))

    selected = max(
        range(len(variants)),
        key=lambda index: (train_score[index], train_margin[index], -index),
    )
    holdout_best = max(
        range(len(variants)),
        key=lambda index: (holdout_score[index], holdout_margin[index], -index),
    )
    order = sorted(
        range(len(variants)),
        key=lambda index: (-train_score[index], -train_margin[index], index),
    )

    def row(index: int) -> dict:
        checkpoint, target = variants[index]
        return {
            "checkpoint": checkpoint,
            "target_route": families[target],
            "target_alias": details.get(families[target], {}).get("alias"),
            "train_score_rate": float(train_score[index]),
            "train_mean_margin": float(train_margin[index]),
            "holdout_score_rate": float(holdout_score[index]),
            "holdout_mean_margin": float(holdout_margin[index]),
        }

    result = {
        "schema": "kaggriculture.one-switch-counter-route-search.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "Opponent-specific offline counter search. The chosen switch is not "
            "deployable until triggered by public-state evidence."
        ),
        "config": {
            "opening": args.opening,
            "opponent": args.opponent,
            "target_routes": len(families),
            "checkpoints": args.checkpoints,
            "seed_start": args.seed_start,
            "train_seeds": args.train_seeds,
            "holdout_seeds": args.holdout_seeds,
            "both_seats": True,
        },
        "summary": {
            "baseline": row(0),
            "train_selected": row(selected),
            "holdout_posthoc_best": row(holdout_best),
            "variants": len(variants),
            "variants_train_at_least_90pct": int((train_score >= 0.90).sum()),
            "variants_holdout_at_least_90pct": int((holdout_score >= 0.90).sum()),
            "games": len(tasks),
            "simulation_seconds": simulation_seconds,
            "games_per_second": len(tasks) / simulation_seconds,
        },
        "top_train_variants": [row(index) for index in order[:50]],
        "inputs": {
            "source": str(args.source),
            "actions": str(args.actions),
            "metadata": str(args.metadata),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
