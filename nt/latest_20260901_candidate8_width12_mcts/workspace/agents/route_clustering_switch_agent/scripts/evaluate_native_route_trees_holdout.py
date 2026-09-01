#!/usr/bin/env python3
"""Large unseen-seed holdout for native searched route-switch trees."""

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


def _win_score(rewards: np.ndarray, seats: np.ndarray) -> np.ndarray:
    rows = np.arange(len(rewards))
    own = rewards[rows, seats]
    other = rewards[rows, 1 - seats]
    return (own > other).astype(np.float64) + .5 * (own == other)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    summaries = []
    for wrapped in policy["nodes"]:
        node = wrapped["selected"]
        if not node["enabled"]:
            continue
        opening = str(node["opening"])
        checkpoint = int(node["checkpoint"])
        opening_route = bundle.index(opening)
        state_tasks = []
        metadata = []
        for opponent_index, opponent in enumerate(bundle.families):
            opponent_route = bundle.index(opponent)
            for seed in args.seeds:
                state_tasks.append((
                    opening_route, opponent_route, seed, checkpoint, 0, opening_route,
                ))
                metadata.append((opponent_index, seed, 0))
                state_tasks.append((
                    opponent_route, opening_route, seed, checkpoint, 1, opening_route,
                ))
                metadata.append((opponent_index, seed, 1))
        state_tasks_array = np.asarray(state_tasks, dtype=np.int64)
        state_started = time.perf_counter()
        features = np.asarray(bundle.executor.features_batch(state_tasks_array))
        tree = node["tree"]
        values = np.asarray(tree["value"], dtype=np.float64)
        leaf_class = np.argmax(values, axis=1).astype(np.int32)
        predictions = np.asarray(native_tree_predict(
            np.asarray(tree["left"], dtype=np.int32),
            np.asarray(tree["right"], dtype=np.int32),
            np.asarray(tree["feature"], dtype=np.int32),
            np.asarray(tree["threshold"], dtype=np.float64),
            leaf_class,
            np.ascontiguousarray(features, dtype=np.float32),
        ))
        classes = tuple(str(value) for value in tree["classes"])
        targets = tuple(classes[int(index)] for index in predictions)
        feature_elapsed = time.perf_counter() - state_started

        dynamic_tasks = []
        baseline_tasks = []
        for (opponent_index, seed, seat), target in zip(metadata, targets):
            opponent_route = bundle.index(bundle.families[opponent_index])
            target_route = bundle.index(target)
            if seat == 0:
                baseline_tasks.append((opening_route, opponent_route, seed, -1, -1, -1, -1))
                dynamic_tasks.append((opening_route, opponent_route, seed, checkpoint, target_route, -1, -1))
            else:
                baseline_tasks.append((opponent_route, opening_route, seed, -1, -1, -1, -1))
                dynamic_tasks.append((opponent_route, opening_route, seed, -1, -1, checkpoint, target_route))
        all_tasks = np.asarray([*baseline_tasks, *dynamic_tasks], dtype=np.int64)
        game_started = time.perf_counter()
        rewards = np.asarray(bundle.executor.play_batch(all_tasks), dtype=np.float64)
        baseline_rewards, dynamic_rewards = np.split(rewards, 2)
        seats = np.asarray([value[2] for value in metadata], dtype=np.int64)
        baseline_score = _win_score(baseline_rewards, seats)
        dynamic_score = _win_score(dynamic_rewards, seats)
        difference = dynamic_score - baseline_score
        cube = difference.reshape(len(bundle.families), len(args.seeds), 2)
        seed_values = cube.mean(axis=(0, 2))
        standard_error = float(np.std(seed_values, ddof=1) / math.sqrt(len(seed_values)))
        lower = float(np.mean(seed_values) - student_t.ppf(.95, len(seed_values) - 1) * standard_error)
        per_opponent = {}
        for index, opponent in enumerate(bundle.families):
            per_opponent[opponent] = {
                "games": int(len(args.seeds) * 2),
                "baseline": float(baseline_score.reshape(len(bundle.families), -1)[index].mean()),
                "dynamic": float(dynamic_score.reshape(len(bundle.families), -1)[index].mean()),
                "improvement": float(cube[index].mean()),
            }
        summaries.append({
            "opening": opening,
            "checkpoint": checkpoint,
            "states": len(metadata),
            "seed_count": len(args.seeds),
            "baseline_score": float(np.mean(baseline_score)),
            "dynamic_score": float(np.mean(dynamic_score)),
            "improvement": float(np.mean(difference)),
            "one_sided_95pct_lower": lower,
            "seed_standard_error": standard_error,
            "switch_rate": float(np.mean(np.asarray(targets) != opening)),
            "target_frequencies": {
                target: int(targets.count(target)) for target in sorted(set(targets))
            },
            "feature_seconds": feature_elapsed,
            "game_seconds": time.perf_counter() - game_started,
            "per_opponent": per_opponent,
        })
        print(json.dumps({key: summaries[-1][key] for key in (
            "opening", "states", "baseline_score", "dynamic_score", "improvement",
            "one_sided_95pct_lower", "switch_rate", "target_frequencies",
        )}, ensure_ascii=False), flush=True)
    payload = {
        "kind": "native_route_tree_unseen_seed_holdout",
        "policy": str(args.policy),
        "seeds": list(args.seeds),
        "summaries": summaries,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
