#!/usr/bin/env python3
"""Exhaustively evaluate one-switch checkpoint sequences in the C++ executor."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from fast_kaggriculture import native_tree_predict
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def _ints(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    return tuple(range(int(start), int(stop))) if separator else tuple(
        int(part) for part in value.split(",")
    )


def _scores(rewards: np.ndarray, seats: np.ndarray) -> np.ndarray:
    rows = np.arange(len(seats))
    own, other = rewards[rows, seats], rewards[rows, 1 - seats]
    return (own > other).astype(np.float64) + 0.5 * (own == other)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--opening", required=True)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-policy-output", type=Path, required=True)
    parser.add_argument(
        "--fixed-all", action="store_true",
        help="Evaluate exactly the full supplied sequence instead of searching subsets.",
    )
    args = parser.parse_args()

    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    wrapped_nodes = [
        value for value in policy["nodes"]
        if value["selected"].get("enabled", True)
        and str(value["selected"]["opening"]) == args.opening
    ]
    wrapped_nodes.sort(key=lambda value: int(value["selected"]["checkpoint"]))
    if not wrapped_nodes:
        raise ValueError(f"no enabled nodes for {args.opening}")
    opening_route = bundle.index(args.opening)
    opponents = np.arange(len(bundle.families), dtype=np.int64)
    opponent_grid = np.broadcast_to(
        opponents[:, None, None], (len(opponents), len(args.seeds), 2)
    ).reshape(-1)
    seed_grid = np.broadcast_to(
        np.asarray(args.seeds)[None, :, None], (len(opponents), len(args.seeds), 2)
    ).reshape(-1)
    seat_grid = np.broadcast_to(
        np.arange(2)[None, None, :], (len(opponents), len(args.seeds), 2)
    ).reshape(-1)
    samples = len(seat_grid)

    predictions = []
    feature_seconds = 0.0
    for wrapped in wrapped_nodes:
        selected = wrapped["selected"]
        checkpoint = int(selected["checkpoint"])
        state_tasks = np.empty((samples, 6), dtype=np.int64)
        state_tasks[:, 0] = np.where(seat_grid == 0, opening_route, opponent_grid)
        state_tasks[:, 1] = np.where(seat_grid == 0, opponent_grid, opening_route)
        state_tasks[:, 2] = seed_grid
        state_tasks[:, 3] = checkpoint
        state_tasks[:, 4] = seat_grid
        state_tasks[:, 5] = opening_route
        started = time.perf_counter()
        features = np.asarray(bundle.executor.features_batch(state_tasks))
        tree = selected["tree"]
        leaf_class = np.argmax(np.asarray(tree["value"]), axis=1).astype(np.int32)
        predicted = np.asarray(native_tree_predict(
            np.asarray(tree["left"], dtype=np.int32),
            np.asarray(tree["right"], dtype=np.int32),
            np.asarray(tree["feature"], dtype=np.int32),
            np.asarray(tree["threshold"], dtype=np.float64),
            leaf_class,
            np.ascontiguousarray(features, dtype=np.float32),
        ))
        classes = np.asarray(tree["classes"], dtype=str)
        predictions.append(classes[predicted])
        feature_seconds += time.perf_counter() - started

    checkpoints = [int(value["selected"]["checkpoint"]) for value in wrapped_nodes]
    subset_masks = (
        [(1 << len(checkpoints)) - 1]
        if args.fixed_all else list(range(1, 1 << len(checkpoints)))
    )
    dynamic_tasks = np.empty((len(subset_masks), samples, 7), dtype=np.int64)
    for mi, mask in enumerate(subset_masks):
        target = np.full(samples, args.opening, dtype=f"U{max(map(len, bundle.families))}")
        switch_step = np.full(samples, -1, dtype=np.int64)
        for ni, checkpoint in enumerate(checkpoints):
            if not (mask & (1 << ni)):
                continue
            changed = (switch_step < 0) & (predictions[ni] != args.opening)
            target[changed] = predictions[ni][changed]
            switch_step[changed] = checkpoint
        target_route = np.asarray([bundle.index(value) for value in target], dtype=np.int64)
        block = dynamic_tasks[mi]
        block[:, 0] = np.where(seat_grid == 0, opening_route, opponent_grid)
        block[:, 1] = np.where(seat_grid == 0, opponent_grid, opening_route)
        block[:, 2] = seed_grid
        block[:, 3] = np.where(seat_grid == 0, switch_step, -1)
        block[:, 4] = np.where(seat_grid == 0, target_route, -1)
        block[:, 5] = np.where(seat_grid == 1, switch_step, -1)
        block[:, 6] = np.where(seat_grid == 1, target_route, -1)

    baseline = dynamic_tasks[0].copy()
    baseline[:, 3:] = -1
    all_tasks = np.concatenate((baseline, dynamic_tasks.reshape(-1, 7)))
    started = time.perf_counter()
    rewards = np.asarray(bundle.executor.play_batch(all_tasks), dtype=np.float64)
    game_seconds = time.perf_counter() - started
    baseline_score = _scores(rewards[:samples], seat_grid)
    dynamic_rewards = rewards[samples:].reshape(len(subset_masks), samples, 2)
    rows = []
    for mi, mask in enumerate(subset_masks):
        score = _scores(dynamic_rewards[mi], seat_grid)
        difference = score - baseline_score
        seed_values = difference.reshape(len(opponents), len(args.seeds), 2).mean(axis=(0, 2))
        standard_error = float(np.std(seed_values, ddof=1) / math.sqrt(len(seed_values)))
        lower = float(
            np.mean(seed_values)
            - student_t.ppf(0.95, len(seed_values) - 1) * standard_error
        )
        used = [checkpoints[index] for index in range(len(checkpoints)) if mask & (1 << index)]
        switch_step = dynamic_tasks[mi, :, 3] + dynamic_tasks[mi, :, 5] + 1
        rows.append({
            "mask": mask,
            "checkpoints": used,
            "node_count": len(used),
            "baseline_score": float(baseline_score.mean()),
            "dynamic_score": float(score.mean()),
            "improvement": float(difference.mean()),
            "one_sided_95pct_lower": lower,
            "seed_standard_error": standard_error,
            "switch_rate": float(np.mean(switch_step >= 0)),
        })
    rows.sort(key=lambda row: (-row["one_sided_95pct_lower"], row["node_count"], row["checkpoints"]))
    best = rows[0]
    selected_checkpoints = set(best["checkpoints"])
    selected_policy = dict(policy)
    selected_policy["nodes"] = [
        value for value in policy["nodes"]
        if (
            str(value["selected"]["opening"]) != args.opening
            or int(value["selected"]["checkpoint"]) in selected_checkpoints
        )
    ]
    selected_policy["sequence_selection"] = {
        "source": str(args.policy), "selection_seeds": list(args.seeds), **best
    }
    args.selected_policy_output.write_text(
        json.dumps(selected_policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload = {
        "schema_version": 1,
        "engine": "C++ features_batch, native_tree_predict, play_batch",
        "policy": str(args.policy),
        "opening": args.opening,
        "seeds": list(args.seeds),
        "opponents": list(bundle.families),
        "candidate_checkpoints": checkpoints,
        "subsets": len(subset_masks),
        "games": int(len(all_tasks)),
        "feature_seconds": feature_seconds,
        "game_seconds": game_seconds,
        "best": best,
        "ranking": rows,
        "selected_policy": str(args.selected_policy_output),
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "games": payload["games"], "feature_seconds": feature_seconds,
        "game_seconds": game_seconds, "best": best, "top10": rows[:10],
        "selected_policy": str(args.selected_policy_output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
