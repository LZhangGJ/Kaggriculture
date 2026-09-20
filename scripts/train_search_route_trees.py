#!/usr/bin/env python3
"""Distill counterfactual route-switch search into shallow Python trees."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from meta_agent.src.recurrent_meta import (
    ANIMALS,
    CROPS,
    MARKET_DIM,
    PRIVATE_DIM,
    PUBLIC_DIM,
    QUADRANTS,
    STRUCTURES,
)


def _public_names(prefix: str) -> list[str]:
    names = [
        f"{prefix}_money_log", f"{prefix}_hands", f"{prefix}_land",
        f"{prefix}_hires_today", f"{prefix}_farmer_x", f"{prefix}_farmer_y",
    ]
    crop_fields = ("count", "yield", "needs_water", "unwatered", "harvestable", "fertilized")
    animal_fields = ("count", "yield", "needs_feed", "needs_care", "unfed", "harvestable")
    for item in CROPS:
        names.extend(f"{prefix}_{item.lower()}_{field}" for field in crop_fields)
    for item in ANIMALS:
        names.extend(f"{prefix}_{item.lower()}_{field}" for field in animal_fields)
    names.extend(f"{prefix}_{item.lower()}_count" for item in STRUCTURES)
    quadrant_fields = ("unlocked", "empty", "crops", "animals", "workers")
    for quadrant in QUADRANTS:
        names.extend(f"{prefix}_{quadrant.lower()}_{field}" for field in quadrant_fields)
    if len(names) != PUBLIC_DIM:
        raise RuntimeError(f"public feature names {len(names)} != {PUBLIC_DIM}")
    return names


def _feature_names() -> list[str]:
    return [
        *_public_names("self"),
        *_public_names("opponent"),
        *[f"private_{index}" for index in range(PRIVATE_DIM)],
        *[f"market_town_{index}" for index in range(MARKET_DIM)],
    ]


def _tree_payload(model: DecisionTreeClassifier) -> dict[str, Any]:
    tree = model.tree_
    return {
        "classes": [str(value) for value in model.classes_],
        "left": tree.children_left.astype(int).tolist(),
        "right": tree.children_right.astype(int).tolist(),
        "feature": tree.feature.astype(int).tolist(),
        "threshold": tree.threshold.astype(float).tolist(),
        "value": tree.value[:, 0, :].astype(float).tolist(),
    }


def _group_samples(
    games: np.ndarray, states: np.ndarray, opening: int, checkpoint: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[int, np.ndarray]]]:
    mask = (
        (games[:, 1].astype(int) == opening)
        & (games[:, 4].astype(int) == checkpoint)
        & (games[:, -1] == 0)
    )
    indices = np.flatnonzero(mask)
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index in indices:
        key = (int(games[index, 3]), int(games[index, 5]), int(games[index, 6]))
        groups[key].append(int(index))
    matrix = []
    labels = []
    seeds = []
    outcomes = []
    for (_, seed, _), rows in sorted(groups.items()):
        by_target = {int(games[index, 2]): games[index] for index in rows}
        if not by_target:
            continue
        best = max(
            by_target,
            key=lambda target: (float(by_target[target][10]), float(by_target[target][9])),
        )
        reference = rows[0]
        if any(np.max(np.abs(states[index].astype(np.float32) - states[reference].astype(np.float32))) > 1e-3 for index in rows):
            raise RuntimeError("counterfactual switch states are not paired")
        matrix.append(states[reference].astype(np.float32))
        labels.append(best)
        seeds.append(seed)
        outcomes.append(by_target)
    return (
        np.asarray(matrix, dtype=np.float32),
        np.asarray(labels, dtype=np.int16),
        np.asarray(seeds, dtype=np.int64),
        outcomes,
    )


def _score_predictions(predictions: np.ndarray, outcomes: list[dict[int, np.ndarray]]) -> dict[str, float]:
    selected = []
    oracle = []
    for prediction, rows in zip(predictions, outcomes):
        selected.append(float(rows[int(prediction)][10]))
        oracle.append(max(float(row[10]) for row in rows.values()))
    return {
        "score": float(np.mean(selected)),
        "oracle_score": float(np.mean(oracle)),
        "oracle_capture": float(np.mean(selected) / max(np.mean(oracle), 1e-9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-leaf", type=int, default=8)
    parser.add_argument(
        "--min-improvement", type=float, default=0.01,
        help="Disable an opening unless cross-validated score beats staying by this amount.",
    )
    args = parser.parse_args()

    with np.load(args.search) as saved:
        games = saved["games"]
        states = saved["states"]
        openings = saved["openings"].astype(str).tolist()
        targets = saved["targets"].astype(str).tolist()
        checkpoints = saved["checkpoints"].astype(int).tolist()
    nodes = []
    for opening_index, opening in enumerate(openings):
        candidates = []
        for checkpoint in checkpoints:
            matrix, labels, seed_groups, outcomes = _group_samples(
                games, states, opening_index, checkpoint
            )
            unique_seeds = np.unique(seed_groups)
            folds = min(5, len(unique_seeds))
            predictions = np.empty(len(labels), dtype=np.int16)
            if folds >= 2:
                splitter = GroupKFold(n_splits=folds)
                for train, valid in splitter.split(matrix, labels, seed_groups):
                    model = DecisionTreeClassifier(
                        max_depth=args.max_depth,
                        min_samples_leaf=args.min_leaf,
                        random_state=20260824,
                    ).fit(matrix[train], labels[train])
                    predictions[valid] = model.predict(matrix[valid]).astype(np.int16)
            else:
                model = DecisionTreeClassifier(
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_leaf,
                    random_state=20260824,
                ).fit(matrix, labels)
                predictions[:] = model.predict(matrix).astype(np.int16)
            metrics = _score_predictions(predictions, outcomes)
            stay_index = targets.index(opening) if opening in targets else None
            if stay_index is not None:
                metrics["stay_score"] = float(np.mean([
                    rows[stay_index][10] for rows in outcomes
                ]))
                metrics["improvement"] = metrics["score"] - metrics["stay_score"]
            else:
                metrics["stay_score"] = None
                metrics["improvement"] = None
            final = DecisionTreeClassifier(
                max_depth=args.max_depth,
                min_samples_leaf=args.min_leaf,
                random_state=20260824,
            ).fit(matrix, labels)
            candidates.append({
                "opening": opening,
                "checkpoint": checkpoint,
                "samples": len(labels),
                "metrics": metrics,
                "tree": _tree_payload(final),
            })
        candidates.sort(
            key=lambda value: (
                -(value["metrics"]["improvement"] if value["metrics"]["improvement"] is not None else -1),
                -value["metrics"]["score"],
                value["checkpoint"],
            )
        )
        selected = candidates[0]
        improvement = selected["metrics"]["improvement"]
        selected["enabled"] = bool(
            improvement is not None and improvement >= args.min_improvement
        )
        nodes.append({"selected": selected, "checkpoint_trials": candidates})
    payload = {
        "schema_version": 1,
        "kind": "counterfactual_search_route_trees",
        "feature_names": _feature_names(),
        "openings": openings,
        "targets": targets,
        "nodes": nodes,
        "training": {
            "max_depth": args.max_depth,
            "min_leaf": args.min_leaf,
            "min_improvement": args.min_improvement,
        },
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "selected": [
            {
                "opening": node["selected"]["opening"],
                "checkpoint": node["selected"]["checkpoint"],
                "enabled": node["selected"]["enabled"],
                **node["selected"]["metrics"],
            }
            for node in nodes
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
