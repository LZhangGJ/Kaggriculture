#!/usr/bin/env python3
"""Greedily select evolved leaf overrides by end-to-end policy rollouts."""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def _ints(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    return tuple(range(int(start), int(stop))) if separator else tuple(
        int(part) for part in value.split(",") if part.strip()
    )


def _tree_leaves_and_predictions(
    tree: dict[str, Any], matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(tree["left"], dtype=np.int32)
    right = np.asarray(tree["right"], dtype=np.int32)
    feature = np.asarray(tree["feature"], dtype=np.int32)
    threshold = np.asarray(tree["threshold"], dtype=np.float64)
    value = np.asarray(tree["value"], dtype=np.float64)
    classes = np.asarray(tree["classes"], dtype=str)
    leaves = np.empty(len(matrix), dtype=np.int32)
    predictions = np.empty(len(matrix), dtype=f"U{max(map(len, classes))}")
    for row_index, vector in enumerate(matrix):
        node = 0
        while left[node] >= 0:
            node = left[node] if vector[feature[node]] <= threshold[node] else right[node]
        leaves[row_index] = node
        predictions[row_index] = classes[int(np.argmax(value[node]))]
    return leaves, predictions


def _score(rewards: np.ndarray, seats: np.ndarray) -> np.ndarray:
    rows = np.arange(len(seats))
    own = rewards[rows, seats]
    other = rewards[rows, 1 - seats]
    return (own > other).astype(np.float64) + 0.5 * (own == other)


def _confidence(diff: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    values = np.asarray(
        [float(np.mean(diff[groups == group])) for group in np.unique(groups)],
        dtype=np.float64,
    )
    mean = float(np.mean(values))
    standard_error = (
        float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    )
    lower = (
        mean - float(student_t.ppf(0.95, len(values) - 1)) * standard_error
        if len(values) > 1 else mean
    )
    return {
        "mean": mean,
        "standard_error": standard_error,
        "lower": float(lower),
        "groups": int(len(values)),
    }


def _tasks_for_predictions(
    predictions: dict[int, np.ndarray],
    checkpoints: list[int],
    opening_name: str,
    opening_route: int,
    opponent_grid: np.ndarray,
    seed_grid: np.ndarray,
    seat_grid: np.ndarray,
    bundle: NativeTeammateBundle,
) -> np.ndarray:
    samples = len(seat_grid)
    target = np.full(samples, opening_name, dtype=f"U{max(map(len, bundle.families))}")
    switch_step = np.full(samples, -1, dtype=np.int64)
    for checkpoint in checkpoints:
        changed = (switch_step < 0) & (predictions[checkpoint] != opening_name)
        target[changed] = predictions[checkpoint][changed]
        switch_step[changed] = checkpoint
    target_route = np.asarray([bundle.index(value) for value in target], dtype=np.int64)
    tasks = np.empty((samples, 7), dtype=np.int64)
    tasks[:, 0] = np.where(seat_grid == 0, opening_route, opponent_grid)
    tasks[:, 1] = np.where(seat_grid == 0, opponent_grid, opening_route)
    tasks[:, 2] = seed_grid
    tasks[:, 3] = np.where(seat_grid == 0, switch_step, -1)
    tasks[:, 4] = np.where(seat_grid == 0, target_route, -1)
    tasks[:, 5] = np.where(seat_grid == 1, switch_step, -1)
    tasks[:, 6] = np.where(seat_grid == 1, target_route, -1)
    return tasks


def _expand_and_replace(tree: dict[str, Any], leaf: int, target: str) -> None:
    classes = [str(value) for value in tree["classes"]]
    values = np.asarray(tree["value"], dtype=np.float64)
    if target not in classes:
        values = np.pad(values, ((0, 0), (0, 1)))
        classes.append(target)
    values[leaf] = 0.0
    values[leaf, classes.index(target)] = 1.0
    tree["classes"] = classes
    tree["value"] = values.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--opening", required=True)
    parser.add_argument("--seeds", type=_ints, required=True)
    parser.add_argument("--min-confidence-lower", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    base_policy = json.loads(args.base_policy.read_text(encoding="utf-8"))
    proposal_payload = json.loads(args.proposals.read_text(encoding="utf-8"))
    proposals = [
        {
            "checkpoint": int(row["checkpoint"]),
            "leaf": int(row["leaf"]),
            "target": str(row["selected"]["candidate"]),
            "training_audit": row,
        }
        for row in proposal_payload["residual_route_augmentation"]["audits"]
        if row["accepted"]
    ]
    nodes = {
        int(row["selected"]["checkpoint"]): row["selected"]["tree"]
        for row in base_policy["nodes"]
        if row["selected"].get("enabled", True)
        and str(row["selected"]["opening"]) == args.opening
    }
    checkpoints = sorted(nodes)
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
    opening_route = bundle.index(args.opening)

    base_predictions: dict[int, np.ndarray] = {}
    leaves_by_checkpoint: dict[int, np.ndarray] = {}
    feature_seconds = 0.0
    for checkpoint in checkpoints:
        state_tasks = np.empty((samples, 6), dtype=np.int64)
        state_tasks[:, 0] = np.where(seat_grid == 0, opening_route, opponent_grid)
        state_tasks[:, 1] = np.where(seat_grid == 0, opponent_grid, opening_route)
        state_tasks[:, 2] = seed_grid
        state_tasks[:, 3] = checkpoint
        state_tasks[:, 4] = seat_grid
        state_tasks[:, 5] = opening_route
        started = time.perf_counter()
        matrix = np.asarray(bundle.executor.features_batch(state_tasks), dtype=np.float32)
        leaves, predictions = _tree_leaves_and_predictions(nodes[checkpoint], matrix)
        feature_seconds += time.perf_counter() - started
        leaves_by_checkpoint[checkpoint] = leaves
        base_predictions[checkpoint] = predictions

    def predictions_for(selected: set[int]) -> dict[int, np.ndarray]:
        family_width = max(map(len, bundle.families))
        result = {
            checkpoint: np.asarray(values, dtype=f"U{family_width}")
            for checkpoint, values in base_predictions.items()
        }
        for index in selected:
            proposal = proposals[index]
            checkpoint = proposal["checkpoint"]
            mask = leaves_by_checkpoint[checkpoint] == proposal["leaf"]
            result[checkpoint][mask] = proposal["target"]
        return result

    def tasks_for(selected: set[int]) -> np.ndarray:
        return _tasks_for_predictions(
            predictions_for(selected), checkpoints, args.opening, opening_route,
            opponent_grid, seed_grid, seat_grid, bundle,
        )

    selected: set[int] = set()
    started = time.perf_counter()
    current_rewards = np.asarray(bundle.executor.play_batch(tasks_for(selected)), dtype=np.float64)
    current_score = _score(current_rewards, seat_grid)
    game_seconds = time.perf_counter() - started
    base_score = current_score.copy()
    rounds = []
    remaining = set(range(len(proposals)))
    while remaining:
        candidates = sorted(remaining)
        batch_tasks = np.concatenate([tasks_for(selected | {index}) for index in candidates])
        started = time.perf_counter()
        rewards = np.asarray(bundle.executor.play_batch(batch_tasks), dtype=np.float64)
        game_seconds += time.perf_counter() - started
        rewards = rewards.reshape(len(candidates), samples, 2)
        rows = []
        candidate_scores: dict[int, np.ndarray] = {}
        for candidate_index, candidate_rewards in zip(candidates, rewards):
            score = _score(candidate_rewards, seat_grid)
            candidate_scores[candidate_index] = score
            difference = score - current_score
            seed_confidence = _confidence(difference, seed_grid)
            opponent_confidence = _confidence(difference, opponent_grid)
            rows.append(
                {
                    "proposal_index": candidate_index,
                    "proposal": {
                        key: proposals[candidate_index][key]
                        for key in ("checkpoint", "leaf", "target")
                    },
                    "score": float(np.mean(score)),
                    "improvement": float(np.mean(difference)),
                    "seed_confidence": seed_confidence,
                    "opponent_confidence": opponent_confidence,
                    "robust_lower": float(
                        min(seed_confidence["lower"], opponent_confidence["lower"])
                    ),
                }
            )
        rows.sort(key=lambda row: (-row["robust_lower"], -row["improvement"], row["proposal_index"]))
        best = rows[0]
        accepted = bool(
            best["improvement"] > 0
            and best["robust_lower"] >= args.min_confidence_lower
        )
        rounds.append({"accepted": accepted, "ranking": rows})
        if not accepted:
            break
        winner = int(best["proposal_index"])
        selected.add(winner)
        remaining.remove(winner)
        current_score = candidate_scores[winner]

    output_policy = deepcopy(base_policy)
    output_nodes = {
        int(row["selected"]["checkpoint"]): row["selected"]["tree"]
        for row in output_policy["nodes"]
        if row["selected"].get("enabled", True)
        and str(row["selected"]["opening"]) == args.opening
    }
    for index in sorted(selected):
        proposal = proposals[index]
        _expand_and_replace(
            output_nodes[proposal["checkpoint"]], proposal["leaf"], proposal["target"]
        )
    evolved_targets = sorted({proposals[index]["target"] for index in selected})
    output_policy["targets"] = [str(value) for value in output_policy.get("targets", ())] + [
        value for value in evolved_targets if value not in output_policy.get("targets", ())
    ]
    output_policy["residual_route_augmentation"] = {
        "method": "training-confidence proposals plus end-to-end greedy paired validation",
        "base_policy": str(args.base_policy),
        "proposal_source": str(args.proposals),
        "selection_seeds": list(args.seeds),
        "opponents": list(bundle.families),
        "selected_proposal_indices": sorted(selected),
        "selected_proposals": [proposals[index] for index in sorted(selected)],
        "base_score": float(np.mean(base_score)),
        "selected_score": float(np.mean(current_score)),
        "improvement": float(np.mean(current_score - base_score)),
        "min_confidence_lower": args.min_confidence_lower,
    }
    args.output.write_text(
        json.dumps(output_policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "engine": "C++ features_batch and play_batch",
        "samples_per_policy": samples,
        "games": int(samples * (1 + sum(len(row["ranking"]) for row in rounds))),
        "feature_seconds": feature_seconds,
        "game_seconds": game_seconds,
        "base_score": float(np.mean(base_score)),
        "selected_score": float(np.mean(current_score)),
        "improvement": float(np.mean(current_score - base_score)),
        "selected_proposal_indices": sorted(selected),
        "selected_proposals": [proposals[index] for index in sorted(selected)],
        "rounds": rounds,
        "output": str(args.output),
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "games", "feature_seconds", "game_seconds", "base_score",
        "selected_score", "improvement", "selected_proposal_indices",
        "selected_proposals", "output",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
