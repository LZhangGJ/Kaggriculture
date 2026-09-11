#!/usr/bin/env python3
"""Evaluate a public-state one-switch route policy on the clean opponent pool."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fast_kaggriculture import native_tree_predict
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def win_score(rewards: np.ndarray, seats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(len(seats))
    own = rewards[rows, seats]
    other = rewards[rows, 1 - seats]
    return (
        np.where(own > other, 1.0, np.where(own == other, 0.5, 0.0)),
        own - other,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--merged-receipt", required=True, type=Path)
    parser.add_argument("--opening", default="G001")
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    opening = bundle.index(args.opening)
    merged = json.loads(args.merged_receipt.read_text(encoding="utf-8"))
    opponents = [row["opponent"] for row in merged["opponents"]]
    opponent_routes = np.asarray([bundle.index(name) for name in opponents], dtype=np.int64)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    nodes = [
        row["selected"] for row in policy["nodes"]
        if row["selected"].get("enabled", True)
        and row["selected"]["opening"] == args.opening
    ]
    nodes.sort(key=lambda row: int(row["checkpoint"]))
    if not nodes:
        raise ValueError(f"no enabled nodes for opening {args.opening}")

    seeds = np.arange(args.seed_start, args.seed_start + args.seed_count, dtype=np.int64)
    opponent_grid = np.broadcast_to(
        opponent_routes[:, None, None], (len(opponents), len(seeds), 2)
    ).reshape(-1)
    seed_grid = np.broadcast_to(
        seeds[None, :, None], (len(opponents), len(seeds), 2)
    ).reshape(-1)
    seat_grid = np.broadcast_to(
        np.arange(2, dtype=np.int64)[None, None, :],
        (len(opponents), len(seeds), 2),
    ).reshape(-1)
    samples = len(seat_grid)

    switch_step = np.full(samples, -1, dtype=np.int64)
    target_route = np.full(samples, opening, dtype=np.int64)
    feature_seconds = 0.0
    prediction_counts: Counter[str] = Counter()
    for node in nodes:
        checkpoint = int(node["checkpoint"])
        tasks = np.empty((samples, 6), dtype=np.int64)
        tasks[:, 0] = np.where(seat_grid == 0, opening, opponent_grid)
        tasks[:, 1] = np.where(seat_grid == 0, opponent_grid, opening)
        tasks[:, 2] = seed_grid
        tasks[:, 3] = checkpoint
        tasks[:, 4] = seat_grid
        tasks[:, 5] = opening
        started = time.perf_counter()
        features = np.asarray(bundle.executor.features_batch(tasks))
        tree = node["tree"]
        leaf_class = np.argmax(np.asarray(tree["value"]), axis=1).astype(np.int32)
        predicted_class = np.asarray(native_tree_predict(
            np.asarray(tree["left"], dtype=np.int32),
            np.asarray(tree["right"], dtype=np.int32),
            np.asarray(tree["feature"], dtype=np.int32),
            np.asarray(tree["threshold"], dtype=np.float64),
            leaf_class,
            np.ascontiguousarray(features, dtype=np.float32),
        ))
        class_names = np.asarray(tree["classes"], dtype=str)
        predicted_names = class_names[predicted_class]
        prediction_counts.update(map(str, predicted_names))
        predicted_routes = np.asarray(
            [bundle.index(str(value)) for value in predicted_names], dtype=np.int64
        )
        changed = (switch_step < 0) & (predicted_routes != opening)
        switch_step[changed] = checkpoint
        target_route[changed] = predicted_routes[changed]
        feature_seconds += time.perf_counter() - started

    baseline_tasks = np.empty((samples, 7), dtype=np.int64)
    baseline_tasks[:, 0] = np.where(seat_grid == 0, opening, opponent_grid)
    baseline_tasks[:, 1] = np.where(seat_grid == 0, opponent_grid, opening)
    baseline_tasks[:, 2] = seed_grid
    baseline_tasks[:, 3:] = -1
    dynamic_tasks = baseline_tasks.copy()
    dynamic_tasks[:, 3] = np.where(seat_grid == 0, switch_step, -1)
    dynamic_tasks[:, 4] = np.where(seat_grid == 0, target_route, -1)
    dynamic_tasks[:, 5] = np.where(seat_grid == 1, switch_step, -1)
    dynamic_tasks[:, 6] = np.where(seat_grid == 1, target_route, -1)

    started = time.perf_counter()
    rewards = np.asarray(
        bundle.executor.play_batch(np.concatenate([baseline_tasks, dynamic_tasks])),
        dtype=np.float64,
    )
    simulation_seconds = time.perf_counter() - started
    baseline_rewards, dynamic_rewards = np.split(rewards, 2)
    baseline_score, baseline_margin = win_score(baseline_rewards, seat_grid)
    dynamic_score, dynamic_margin = win_score(dynamic_rewards, seat_grid)
    baseline_cube = baseline_score.reshape(len(opponents), len(seeds), 2)
    dynamic_cube = dynamic_score.reshape(len(opponents), len(seeds), 2)
    baseline_margin_cube = baseline_margin.reshape(len(opponents), len(seeds), 2)
    dynamic_margin_cube = dynamic_margin.reshape(len(opponents), len(seeds), 2)
    route_switch = (switch_step >= 0).reshape(len(opponents), len(seeds), 2)

    rows: list[dict] = []
    for index, opponent in enumerate(opponents):
        rows.append({
            "opponent": opponent,
            "baseline_score_rate": float(baseline_cube[index].mean()),
            "dynamic_score_rate": float(dynamic_cube[index].mean()),
            "score_gain": float(
                dynamic_cube[index].mean() - baseline_cube[index].mean()
            ),
            "baseline_mean_margin": float(baseline_margin_cube[index].mean()),
            "dynamic_mean_margin": float(dynamic_margin_cube[index].mean()),
            "switch_rate": float(route_switch[index].mean()),
        })
    dynamic_rates = dynamic_cube.mean(axis=(1, 2))
    result = {
        "schema": "kaggriculture.public-route-switch-policy-evaluation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "Deployable-information diagnostic: tree features contain own state "
            "and public opponent/market state, not opponent identity. C++ parity "
            "with the final Python package is not asserted by this experiment."
        ),
        "config": {
            "opening": args.opening,
            "checkpoints": [int(node["checkpoint"]) for node in nodes],
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "opponents": len(opponents),
            "both_seats": True,
        },
        "summary": {
            "games_per_policy": samples,
            "baseline_score_rate": float(baseline_score.mean()),
            "dynamic_score_rate": float(dynamic_score.mean()),
            "score_gain": float(dynamic_score.mean() - baseline_score.mean()),
            "baseline_mean_margin": float(baseline_margin.mean()),
            "dynamic_mean_margin": float(dynamic_margin.mean()),
            "switch_rate": float((switch_step >= 0).mean()),
            "opponents_at_least_90pct": int((dynamic_rates >= 0.90).sum()),
            "opponents_at_least_50pct": int((dynamic_rates >= 0.50).sum()),
            "worst_opponent_score": float(dynamic_rates.min()),
            "feature_seconds": feature_seconds,
            "simulation_seconds": simulation_seconds,
            "games_per_second": (samples * 2) / simulation_seconds,
        },
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "opponents": rows,
        "inputs": {
            "source": str(args.source),
            "actions": str(args.actions),
            "metadata": str(args.metadata),
            "policy": str(args.policy),
            "merged_receipt": str(args.merged_receipt),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "summary": result["summary"],
        "worst": sorted(rows, key=lambda row: row["dynamic_score_rate"])[:20],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
