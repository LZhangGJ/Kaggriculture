#!/usr/bin/env python3
"""Rank fixed mature routes against an existing searched switch policy."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from fast_kaggriculture import native_tree_predict
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def _ints(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    if separator:
        return tuple(range(int(start), int(stop)))
    return tuple(int(part) for part in value.split(",") if part.strip())


def _predictions(executor, tree: dict, features: np.ndarray) -> np.ndarray:
    leaf_class = np.argmax(np.asarray(tree["value"]), axis=1).astype(np.int32)
    indices = np.asarray(native_tree_predict(
        np.asarray(tree["left"], dtype=np.int32),
        np.asarray(tree["right"], dtype=np.int32),
        np.asarray(tree["feature"], dtype=np.int32),
        np.asarray(tree["threshold"], dtype=np.float64),
        leaf_class,
        np.ascontiguousarray(features, dtype=np.float32),
    ))
    return np.asarray(tree["classes"], dtype=str)[indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--opening", default="G001")
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    nodes = sorted(
        (
            value["selected"] for value in policy["nodes"]
            if value["selected"].get("enabled", True)
            and str(value["selected"]["opening"]) == args.opening
        ),
        key=lambda value: int(value["checkpoint"]),
    )
    if not nodes:
        raise ValueError(f"no enabled policy nodes for {args.opening}")
    opening_index = bundle.index(args.opening)

    # Samples are ordered candidate, seed, candidate seat.  The policy is the
    # other player and sees the same no-switch history at each later node only
    # when no earlier node has switched, matching SearchRouteController.
    sample_rows = [
        (candidate_index, int(seed), candidate_seat)
        for candidate_index in range(len(bundle.families))
        for seed in args.seeds
        for candidate_seat in (0, 1)
    ]
    switch_step = np.full(len(sample_rows), -1, dtype=np.int64)
    switch_target = np.full(len(sample_rows), opening_index, dtype=np.int64)
    predicted_at: list[np.ndarray] = []
    feature_seconds = 0.0

    for node in nodes:
        checkpoint = int(node["checkpoint"])
        tasks = np.empty((len(sample_rows), 6), dtype=np.int64)
        for row_index, (candidate_index, seed, candidate_seat) in enumerate(sample_rows):
            if candidate_seat == 0:
                tasks[row_index] = (
                    candidate_index, opening_index, seed, checkpoint, 1, opening_index,
                )
            else:
                tasks[row_index] = (
                    opening_index, candidate_index, seed, checkpoint, 0, opening_index,
                )
        feature_started = time.perf_counter()
        features = np.asarray(bundle.executor.features_batch(tasks))
        predictions = _predictions(bundle.executor, node["tree"], features)
        feature_seconds += time.perf_counter() - feature_started
        predicted_at.append(predictions)
        for row_index, prediction in enumerate(predictions):
            if switch_step[row_index] >= 0 or prediction == args.opening:
                continue
            switch_step[row_index] = checkpoint
            switch_target[row_index] = bundle.index(str(prediction))

    game_tasks = np.empty((len(sample_rows), 7), dtype=np.int64)
    for row_index, (candidate_index, seed, candidate_seat) in enumerate(sample_rows):
        if candidate_seat == 0:
            game_tasks[row_index] = (
                candidate_index, opening_index, seed,
                -1, -1, switch_step[row_index], switch_target[row_index],
            )
        else:
            game_tasks[row_index] = (
                opening_index, candidate_index, seed,
                switch_step[row_index], switch_target[row_index], -1, -1,
            )
    game_started = time.perf_counter()
    rewards = np.asarray(bundle.executor.play_batch(game_tasks), dtype=np.float64)
    game_seconds = time.perf_counter() - game_started

    rows = []
    block_size = len(args.seeds) * 2
    for candidate_index, family in enumerate(bundle.families):
        start = candidate_index * block_size
        stop = start + block_size
        block_rewards = rewards[start:stop]
        block_samples = sample_rows[start:stop]
        candidate_seats = np.asarray([value[2] for value in block_samples], dtype=np.int64)
        indices = np.arange(block_size)
        own = block_rewards[indices, candidate_seats]
        other = block_rewards[indices, 1 - candidate_seats]
        margins = own - other
        scores = (margins > 0).astype(np.float64) + 0.5 * (margins == 0)
        paired_margins = margins.reshape(len(args.seeds), 2).mean(axis=1)
        paired_scores = scores.reshape(len(args.seeds), 2).mean(axis=1)
        if len(args.seeds) > 1:
            margin_se = float(np.std(paired_margins, ddof=1) / math.sqrt(len(args.seeds)))
            margin_lower = float(
                np.mean(paired_margins)
                - student_t.ppf(.95, len(args.seeds) - 1) * margin_se
            )
        else:
            margin_se = 0.0
            margin_lower = float(paired_margins[0])
        route_targets = [
            bundle.families[int(index)] for index in switch_target[start:stop]
        ]
        rows.append({
            "family": family,
            "mean_score": float(np.mean(scores)),
            "mean_margin": float(np.mean(margins)),
            "minimum_margin": float(np.min(margins)),
            "minimum_paired_seed_margin": float(np.min(paired_margins)),
            "one_sided_95pct_margin_lower": margin_lower,
            "paired_margin_standard_error": margin_se,
            "mean_reward": float(np.mean(own)),
            "minimum_reward": float(np.min(own)),
            "switch_rate": float(np.mean(switch_step[start:stop] >= 0)),
            "policy_target_counts": dict(sorted(Counter(route_targets).items())),
            "scenario_count": int(block_size),
            "seed_count": len(args.seeds),
            "paired_seed_scores": paired_scores.tolist(),
            "paired_seed_margins": paired_margins.tolist(),
        })
    rows.sort(key=lambda row: (
        -float(row["mean_score"]),
        -float(row["one_sided_95pct_margin_lower"]),
        -float(row["mean_margin"]),
        str(row["family"]),
    ))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "native-fixed-route-vs-searched-policy-v1",
        "engine": "C++ NativeTeammateExecutor with exact first-switch policy replay",
        "source": str(args.source.resolve()),
        "actions": str(args.actions.resolve()),
        "metadata": str(args.metadata.resolve()),
        "policy": str(args.policy.resolve()),
        "opening": args.opening,
        "checkpoints": [int(value["checkpoint"]) for value in nodes],
        "seeds": list(args.seeds),
        "candidate_count": len(rows),
        "games": len(sample_rows),
        "setup_and_feature_seconds": feature_seconds,
        "game_seconds": game_seconds,
        "games_per_second": len(sample_rows) / game_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "ranking": rows,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "games": payload["games"],
        "games_per_second": payload["games_per_second"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "top10": [
            {key: row[key] for key in (
                "rank", "family", "mean_score", "mean_margin",
                "minimum_paired_seed_margin", "one_sided_95pct_margin_lower",
                "mean_reward", "switch_rate", "policy_target_counts",
            )}
            for row in rows[:10]
        ],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
