#!/usr/bin/env python3
"""Train a probabilistic all-family opponent recognizer from public replay state."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-cache", type=Path, required=True)
    parser.add_argument("--macro-cache", type=Path, required=True)
    parser.add_argument("--global-clusters", type=Path, required=True)
    parser.add_argument(
        "--family-names",
        type=Path,
        help="Optional NPZ with family_names aligned to macro-cache rows.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--trees", type=int, default=192)
    parser.add_argument("--cv-trees", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-leaf", type=int, default=3)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def _global_names(rows: list[dict[str, Any]], labels: np.ndarray) -> np.ndarray:
    sizes = Counter(labels.tolist())
    medians = {
        label: float(np.median([
            float(row["reward"]) for row, value in zip(rows, labels)
            if int(value) == int(label)
        ]))
        for label in sizes
    }
    ordered = sorted(sizes, key=lambda label: (-sizes[label], -medians[label]))
    names = {label: f"G{index + 1}" for index, label in enumerate(ordered)}
    return np.asarray([names[int(label)] for label in labels])


def _dataset(
    selector_cache: Path, macro_cache: Path, global_clusters: Path,
    family_names: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], str]:
    with np.load(selector_cache, allow_pickle=True) as cached:
        selector_rows = list(cached["rows"])
    with np.load(macro_cache, allow_pickle=True) as cached:
        macro_rows = list(cached["rows"])
    with np.load(global_clusters) as cached:
        labels = cached["labels"].astype(int)
    if len(macro_rows) != len(labels):
        raise ValueError("global labels do not align with macro rows")
    if family_names is None:
        global_names = _global_names(macro_rows, labels)
    else:
        with np.load(family_names) as named:
            global_names = named["family_names"].astype(str)
        if len(global_names) != len(macro_rows):
            raise ValueError("family names do not align with macro rows")
    label_by_key = {
        (int(row["episode_id"]), int(row["player_index"]), str(row["team"])): str(label)
        for row, label in zip(macro_rows, global_names)
    }
    rich_states = bool(selector_rows and "states" in selector_rows[0])
    if rich_states:
        first = next(iter(selector_rows[0]["states"].values()))
        public_size = len(first["self_public"])
        market_size = len(first["market_town"])
        names = [
            *[f"public_{index}" for index in range(public_size)],
            *[f"market_{index}" for index in range(market_size)],
        ]
        feature_schema = "public_market_v1"
    else:
        names = sorted({
            name
            for row in selector_rows
            for values in row["features"].values()
            for name in values
            if name.startswith(("own_", "shop_", "market_"))
        })
        feature_schema = "simple_public_v1"
    feature_names = [*names, "route_checkpoint"]
    samples: list[list[float]] = []
    targets: list[str] = []
    episode_groups: list[str] = []
    team_groups: list[str] = []
    checkpoints: list[int] = []
    for row in selector_rows:
        key = (int(row["episode_id"]), int(row["player_index"]), str(row["team"]))
        label = label_by_key.get(key)
        if not label:
            continue
        state_values = row["states"] if rich_states else row["features"]
        for checkpoint, values in sorted(state_values.items()):
            checkpoint = int(checkpoint)
            if rich_states:
                vector = np.concatenate([
                    np.asarray(values["self_public"], dtype=np.float32),
                    np.asarray(values["market_town"], dtype=np.float32),
                ]).tolist()
            else:
                vector = [float(values.get(name, 0.0)) for name in names]
            samples.append([*vector, checkpoint / 719.0])
            targets.append(label)
            episode_groups.append(f"{key[0]}:{key[1]}")
            team_groups.append(key[2])
            checkpoints.append(checkpoint)
    return (
        np.asarray(samples, dtype=np.float32), np.asarray(targets),
        np.asarray(episode_groups), np.asarray(team_groups),
        np.asarray(checkpoints, dtype=np.int16), feature_names, feature_schema,
    )


def _new_model(args: argparse.Namespace, trees: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=max(1, int(trees)), max_depth=max(1, args.max_depth),
        min_samples_leaf=max(1, args.min_leaf), max_features="sqrt",
        class_weight="balanced", random_state=20260824,
        n_jobs=args.workers,
    )


def _aligned_probabilities(
    model: ExtraTreesClassifier, matrix: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    raw = model.predict_proba(matrix)
    result = np.zeros((len(matrix), len(classes)), dtype=np.float64)
    index = {str(label): i for i, label in enumerate(classes)}
    for source, label in enumerate(model.classes_):
        result[:, index[str(label)]] = raw[:, source]
    return result


def _metrics(
    probabilities: np.ndarray, targets: np.ndarray, classes: np.ndarray,
    checkpoints: np.ndarray,
) -> dict[str, Any]:
    class_index = {str(label): i for i, label in enumerate(classes)}
    truth = np.asarray([class_index[str(label)] for label in targets], dtype=np.int64)
    order = np.argsort(-probabilities, axis=1)
    result: dict[str, Any] = {
        "samples": len(targets),
        "top1": float(np.mean(order[:, 0] == truth)),
        "top3": float(np.mean([truth[i] in order[i, :3] for i in range(len(truth))])),
        "top5": float(np.mean([truth[i] in order[i, :5] for i in range(len(truth))])),
        "nll": float(np.mean(-np.log(np.maximum(1e-9, probabilities[np.arange(len(truth)), truth])))),
    }
    result["by_checkpoint"] = {}
    for checkpoint in sorted(set(checkpoints.tolist())):
        mask = checkpoints == checkpoint
        local_order = order[mask]
        local_truth = truth[mask]
        result["by_checkpoint"][str(checkpoint)] = {
            "samples": int(mask.sum()),
            "top1": float(np.mean(local_order[:, 0] == local_truth)),
            "top3": float(np.mean([
                local_truth[i] in local_order[i, :3]
                for i in range(len(local_truth))
            ])),
            "top5": float(np.mean([
                local_truth[i] in local_order[i, :5]
                for i in range(len(local_truth))
            ])),
        }
    return result


def _cross_validate(
    args: argparse.Namespace, matrix: np.ndarray, targets: np.ndarray,
    groups: np.ndarray, checkpoints: np.ndarray, classes: np.ndarray,
) -> dict[str, Any]:
    unique = len(set(groups.tolist()))
    folds = min(max(2, int(args.folds)), unique)
    probabilities = np.zeros((len(targets), len(classes)), dtype=np.float64)
    seen = np.zeros(len(targets), dtype=bool)
    splitter = GroupKFold(n_splits=folds)
    for fold, (train, valid) in enumerate(splitter.split(matrix, targets, groups), 1):
        model = _new_model(args, args.cv_trees)
        model.fit(matrix[train], targets[train])
        probabilities[valid] = _aligned_probabilities(model, matrix[valid], classes)
        seen[valid] = True
        print(f"fold {fold}/{folds}: train={len(train)} valid={len(valid)}", flush=True)
    if not seen.all():
        raise RuntimeError("cross-validation did not cover every sample")
    return _metrics(probabilities, targets, classes, checkpoints)


def _export_tree(model: ExtraTreesClassifier) -> list[dict[str, np.ndarray]]:
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = np.asarray(tree.value[:, 0, :], dtype=np.float32)
        trees.append({
            "left": np.asarray(tree.children_left, dtype=np.int32),
            "right": np.asarray(tree.children_right, dtype=np.int32),
            "feature": np.asarray(tree.feature, dtype=np.int16),
            "threshold": np.asarray(tree.threshold, dtype=np.float32),
            "value": values,
        })
    return trees


def main() -> None:
    args = arguments()
    matrix, targets, episode_groups, team_groups, checkpoints, feature_names, feature_schema = _dataset(
        args.selector_cache, args.macro_cache, args.global_clusters, args.family_names
    )
    classes, class_counts = np.unique(targets, return_counts=True)
    print(json.dumps({
        "samples": len(targets), "episodes": len(set(episode_groups)),
        "teams": len(set(team_groups)), "classes": len(classes),
        "features": len(feature_names),
    }, ensure_ascii=False), flush=True)
    replay_cv = _cross_validate(
        args, matrix, targets, episode_groups, checkpoints, classes
    )
    team_cv = _cross_validate(
        args, matrix, targets, team_groups, checkpoints, classes
    )
    model = _new_model(args, args.trees)
    model.fit(matrix, targets)
    priors = np.zeros(len(model.classes_), dtype=np.float32)
    count_by_class = dict(zip(classes.tolist(), class_counts.tolist()))
    for index, label in enumerate(model.classes_):
        priors[index] = count_by_class[str(label)]
    priors /= priors.sum()
    payload = {
        "schema_version": 1,
        "model_kind": "global_opponent_route_forest",
        "feature_names": feature_names,
        "feature_schema": feature_schema,
        "classes": [str(value) for value in model.classes_],
        "priors": priors,
        "trees": _export_tree(model),
        "checkpoints": sorted(set(checkpoints.tolist())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pickle.dumps(payload, protocol=5))
    report_path = args.report or args.output.with_suffix(".metrics.json")
    report = {
        "schema_version": 1,
        "model": {
            "kind": payload["model_kind"], "trees": args.trees,
            "max_depth": args.max_depth, "min_leaf": args.min_leaf,
            "features": len(feature_names), "classes": len(classes),
        },
        "dataset": {
            "samples": len(targets), "route_sides": len(set(episode_groups)),
            "teams": len(set(team_groups)), "checkpoints": payload["checkpoints"],
            "class_support": dict(sorted(count_by_class.items(), key=lambda item: int(item[0][1:]))),
        },
        "validation": {
            "unseen_replay_side": replay_cv,
            "unseen_team": team_cv,
            "warning": "team-held-out is the honest estimate for previously unseen opponent scripts",
        },
        "artifact": {"path": str(args.output), "bytes": args.output.stat().st_size},
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output), "bytes": args.output.stat().st_size,
        "report": str(report_path), "replay_top1": replay_cv["top1"],
        "team_top1": team_cv["top1"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
