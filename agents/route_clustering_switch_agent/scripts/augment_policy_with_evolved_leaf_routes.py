#!/usr/bin/env python3
"""Conservatively replace base-policy leaves with robust evolved routes.

The base tree topology and every unqualified leaf remain unchanged.  An evolved
route may replace a leaf only when its paired score improvement clears both
seed-grouped and opponent-grouped one-sided confidence bounds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _predict_nodes(tree: dict[str, Any], matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(tree["left"], dtype=np.int32)
    right = np.asarray(tree["right"], dtype=np.int32)
    feature = np.asarray(tree["feature"], dtype=np.int32)
    threshold = np.asarray(tree["threshold"], dtype=np.float64)
    values = np.asarray(tree["value"], dtype=np.float64)
    classes = np.asarray(tree["classes"], dtype=str)
    leaves = np.empty(len(matrix), dtype=np.int32)
    predictions = np.empty(len(matrix), dtype=f"U{max(map(len, classes))}")
    for row_index, vector in enumerate(matrix):
        node = 0
        while left[node] >= 0:
            node = left[node] if vector[feature[node]] <= threshold[node] else right[node]
        leaves[row_index] = node
        predictions[row_index] = classes[int(np.argmax(values[node]))]
    return leaves, predictions


def _group_confidence(diff: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    values = np.asarray(
        [float(np.mean(diff[groups == group])) for group in np.unique(groups)],
        dtype=np.float64,
    )
    mean = float(np.mean(values))
    if len(values) < 2:
        return {"mean": mean, "standard_error": 0.0, "lower": mean, "groups": len(values)}
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    lower = mean - float(student_t.ppf(0.95, len(values) - 1)) * standard_error
    return {
        "mean": mean,
        "standard_error": standard_error,
        "lower": float(lower),
        "groups": int(len(values)),
    }


def _expand_tree_classes(tree: dict[str, Any], additions: tuple[str, ...]) -> None:
    old_classes = [str(value) for value in tree["classes"]]
    classes = old_classes + [value for value in additions if value not in old_classes]
    if classes == old_classes:
        return
    old_values = np.asarray(tree["value"], dtype=np.float64)
    values = np.zeros((len(old_values), len(classes)), dtype=np.float64)
    for old_index, target in enumerate(old_classes):
        values[:, classes.index(target)] = old_values[:, old_index]
    tree["classes"] = classes
    tree["value"] = values.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--evolved-targets", type=_csv, required=True)
    parser.add_argument("--min-samples", type=int, default=512)
    parser.add_argument("--min-mean-improvement", type=float, default=0.005)
    parser.add_argument("--min-confidence-lower", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads(args.base_policy.read_text(encoding="utf-8"))
    with np.load(args.search) as saved:
        outcomes = saved["outcome"].astype(np.float32) * 0.5
        states = saved["states"].astype(np.float32)
        openings = saved["openings"].astype(str).tolist()
        targets = saved["targets"].astype(str).tolist()
        opponents = saved["opponents"].astype(str)
        checkpoints = saved["checkpoints"].astype(int).tolist()
        seeds = saved["seeds"].astype(np.int64)

    missing = sorted(set(args.evolved_targets) - set(targets))
    if missing:
        raise ValueError(f"evolved targets absent from search: {missing}")
    target_index = {target: index for index, target in enumerate(targets)}
    opponent_grid = np.broadcast_to(
        opponents[:, None, None], (len(opponents), len(seeds), 2)
    ).reshape(-1)
    seed_grid = np.broadcast_to(
        seeds[None, :, None], (len(opponents), len(seeds), 2)
    ).reshape(-1)
    audits: list[dict[str, Any]] = []

    for wrapped in policy["nodes"]:
        selected = wrapped["selected"]
        if not bool(selected.get("enabled", True)):
            continue
        opening = str(selected["opening"])
        checkpoint = int(selected["checkpoint"])
        opening_index = openings.index(opening)
        checkpoint_index = checkpoints.index(checkpoint)
        matrix = states[opening_index, checkpoint_index].reshape(-1, states.shape[-1])
        node_outcomes = outcomes[opening_index, checkpoint_index].reshape(len(targets), -1)
        tree = selected["tree"]
        leaves, base_predictions = _predict_nodes(tree, matrix)
        _expand_tree_classes(tree, args.evolved_targets)
        tree_values = np.asarray(tree["value"], dtype=np.float64)
        classes = [str(value) for value in tree["classes"]]

        for leaf in sorted(np.unique(leaves).tolist()):
            indices = np.flatnonzero(leaves == leaf)
            base_targets = np.unique(base_predictions[indices])
            if len(base_targets) != 1:
                raise RuntimeError("a base leaf predicted more than one route")
            base_target = str(base_targets[0])
            base_scores = node_outcomes[target_index[base_target], indices]
            rows = []
            for candidate in args.evolved_targets:
                candidate_scores = node_outcomes[target_index[candidate], indices]
                difference = candidate_scores - base_scores
                seed_confidence = _group_confidence(difference, seed_grid[indices])
                opponent_confidence = _group_confidence(difference, opponent_grid[indices])
                rows.append(
                    {
                        "candidate": candidate,
                        "mean_improvement": float(np.mean(difference)),
                        "seed_confidence": seed_confidence,
                        "opponent_confidence": opponent_confidence,
                        "robust_lower": float(
                            min(seed_confidence["lower"], opponent_confidence["lower"])
                        ),
                    }
                )
            rows.sort(
                key=lambda row: (-row["robust_lower"], -row["mean_improvement"], row["candidate"])
            )
            best = rows[0]
            accepted = bool(
                len(indices) >= args.min_samples
                and best["mean_improvement"] >= args.min_mean_improvement
                and best["robust_lower"] >= args.min_confidence_lower
            )
            if accepted:
                tree_values[leaf] = 0.0
                tree_values[leaf, classes.index(best["candidate"])] = 1.0
            audits.append(
                {
                    "opening": opening,
                    "checkpoint": checkpoint,
                    "leaf": int(leaf),
                    "samples": int(len(indices)),
                    "sample_fraction": float(len(indices) / len(matrix)),
                    "base_target": base_target,
                    "accepted": accepted,
                    "selected": best,
                    "candidates": rows,
                }
            )
        tree["value"] = tree_values.tolist()

    old_targets = [str(value) for value in policy.get("targets", ())]
    policy["targets"] = old_targets + [
        value for value in args.evolved_targets if value not in old_targets
    ]
    policy["residual_route_augmentation"] = {
        "method": "paired leaf-level conservative policy improvement",
        "base_policy": str(args.base_policy),
        "counterfactual_search": str(args.search),
        "evolved_targets": list(args.evolved_targets),
        "min_samples": args.min_samples,
        "min_mean_improvement": args.min_mean_improvement,
        "min_confidence_lower": args.min_confidence_lower,
        "paired_by": ["seed", "opponent", "seat", "state"],
        "confidence": "one-sided 95% Student-t lower bound",
        "accepted_leaf_count": int(sum(row["accepted"] for row in audits)),
        "audits": audits,
    }
    args.output.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted_leaf_count": policy["residual_route_augmentation"]["accepted_leaf_count"],
                "accepted": [
                    {
                        key: row[key]
                        for key in ("checkpoint", "leaf", "samples", "base_target", "selected")
                    }
                    for row in audits
                    if row["accepted"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
