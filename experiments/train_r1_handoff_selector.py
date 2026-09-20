#!/usr/bin/env python3
"""Train a conservative shallow handoff selector on paired suffix outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from meta_agent.src.route_switch_features import route_switch_feature_names
from scripts.train_robust_search_route_trees import _tree_payload


def aligned(labels, features):
    keys = ("bots", "seeds", "seats", "days", "targets")
    return all(np.array_equal(labels[name], features[name]) for name in keys)


def candidate_diff(features, classes):
    lookup = {str(name): index for index, name in enumerate(classes)}
    order = ("competitive_sale", "crop_succession")
    if set(lookup) != {"default", *order} or features.ndim != 3 or features.shape[1:] != (3, 356):
        raise ValueError("candidate_diff requires three 356-D selector candidates")
    default = features[:, lookup["default"]]
    matrix = np.concatenate([features[:, lookup[name]] - default for name in order], axis=1)
    names = [f"{name}_minus_default_{index:03d}" for name in order for index in range(356)]
    return matrix.astype(np.float32), names, list(order)


def outcomes(prediction, classes, margins, bots):
    lookup = {name: index for index, name in enumerate(classes)}
    selected = margins[np.arange(len(margins)), [lookup[str(value)] for value in prediction]]
    default = margins[:, lookup["default"]]
    result = {
        "wins": int(np.sum(selected > 0)), "default_wins": int(np.sum(default > 0)),
        "win_delta": int(np.sum(selected > 0) - np.sum(default > 0)),
        "mean_margin": float(np.mean(selected)),
        "mean_margin_delta": float(np.mean(selected - default)), "per_bot": {},
    }
    for bot in sorted(set(bots)):
        use = bots == bot
        result["per_bot"][str(bot)] = {
            "win_delta": int(np.sum(selected[use] > 0) - np.sum(default[use] > 0)),
            "mean_margin_delta": float(np.mean(selected[use] - default[use])),
        }
    result["min_bot_win_delta"] = min(row["win_delta"] for row in result["per_bot"].values())
    result["median_bot_margin_delta"] = float(np.median([
        row["mean_margin_delta"] for row in result["per_bot"].values()
    ]))
    return result


def cross_validate(matrix, labels, groups, depth, leaf, folds=5):
    prediction = np.empty(len(labels), dtype=labels.dtype)
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    for fold, (train, valid) in enumerate(splitter.split(matrix, labels, groups)):
        model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                       random_state=20260920 + fold)
        model.fit(matrix[train], labels[train])
        prediction[valid] = model.predict(matrix[valid])
    return prediction


def self_check():
    classes = np.asarray(["default", "competitive_sale", "crop_succession"])
    margins = np.asarray([[1, -1, -2], [-1, 2, -3], [-2, -1, 3]], dtype=float)
    summary = outcomes(classes, classes, margins, np.asarray(["a", "a", "b"]))
    assert summary["wins"] == 3 and summary["default_wins"] == 1 and summary["win_delta"] == 2
    matrix = np.zeros((12, 147), dtype=np.float32)
    matrix[:, 0] = np.arange(12) % 2
    labels = np.where(matrix[:, 0] == 0, "default", "competitive_sale")
    prediction = cross_validate(matrix, labels, np.repeat(np.arange(3), 4), 1, 2)
    assert prediction.shape == labels.shape and set(prediction) <= set(classes)
    shuffled = np.asarray(["crop_succession", "default", "competitive_sale"])
    cube = np.zeros((2, 3, 356), dtype=np.float32)
    cube[:, 0] = 7; cube[:, 1] = 2; cube[:, 2] = 5
    diff, names, order = candidate_diff(cube, shuffled)
    assert diff.shape == (2, 712) and np.all(diff[:, :356] == 3)
    assert np.all(diff[:, 356:] == 5) and order == ["competitive_sale", "crop_succession"]
    assert names[0] == "competitive_sale_minus_default_000"
    assert names[356] == "crop_succession_minus_default_000"
    print(json.dumps({"status": "PASS"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--route-features", type=Path)
    parser.add_argument("--feature-mode", choices=("route", "candidate_diff"), default="route")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    if not args.labels or not args.output or args.feature_mode == "route" and not args.route_features:
        parser.error("--labels, --output and route mode's --route-features are required")
    if not args.labels.is_file() or args.feature_mode == "route" and not args.route_features.is_file() or args.output.exists():
        parser.error("inputs must exist and output must be new")

    with np.load(args.labels, allow_pickle=False) as source:
        classes = source["classes"].astype(str)
        all_targets = source["targets"].astype(str)
        use = all_targets == "handoff"
        feature_order = None
        if args.feature_mode == "candidate_diff":
            matrix, names, feature_order = candidate_diff(source["features"][use], classes)
            feature_schema = "r1_candidate_diff_v1"
        else:
            with np.load(args.route_features, allow_pickle=False) as route:
                if not aligned(source, route):
                    raise ValueError("label and route-feature rows are not aligned")
                matrix = route["features"][use].astype(np.float32)
                names = route["feature_names"].astype(str).tolist()
            feature_schema = "semantic_route_switch_v1"
        margins = source["margins"][use].astype(np.float64)
        labels = classes[source["labels"][use].astype(int)]
        bots = source["bots"][use].astype(str)
        seeds = source["seeds"][use].astype(np.int64)
        recomputed = classes[np.asarray([
            max(range(len(classes)), key=lambda i: (row[i] > 0, row[i])) for row in margins
        ])]
        expected = 712 if args.feature_mode == "candidate_diff" else 147
        if matrix.shape != (int(np.sum(use)), expected) or len(names) != expected or not np.isfinite(matrix).all():
            raise ValueError(f"invalid handoff feature matrix {matrix.shape}")
        keys = list(zip(bots.tolist(), seeds.tolist(), source["seats"][use].astype(int).tolist()))
        if len(matrix) != 224 or len(set(keys)) != 224 or len(set(bots)) != 7 or \
                len(set(seeds)) != 16 or not np.isfinite(margins).all():
            raise ValueError("handoff rows must be the complete 7x16x2 paired grid")
        if args.feature_mode == "route" and names != route_switch_feature_names():
            raise ValueError("route feature names do not match the canonical 147-D schema")
        if not np.array_equal(labels, recomputed) or set(classes) != {
                "default", "competitive_sale", "crop_succession"}:
            raise ValueError("invalid selector labels/classes")

    oracle = outcomes(labels, classes, margins, bots)
    trials = []
    for depth in (1, 2, 3):
        for leaf in (8, 12, 16):
            by_seed = outcomes(cross_validate(matrix, labels, seeds, depth, leaf),
                               classes, margins, bots)
            by_bot = outcomes(cross_validate(matrix, labels, bots, depth, leaf,
                                              folds=len(np.unique(bots))),
                              classes, margins, bots)
            trials.append({"depth": depth, "min_leaf": leaf,
                           "robust_win_delta": min(by_seed["win_delta"], by_bot["win_delta"]),
                           "robust_min_bot_win_delta": min(by_seed["min_bot_win_delta"],
                                                           by_bot["min_bot_win_delta"]),
                           "robust_margin_delta": min(by_seed["mean_margin_delta"],
                                                      by_bot["mean_margin_delta"]),
                           "robust_median_bot_margin_delta": min(
                               by_seed["median_bot_margin_delta"],
                               by_bot["median_bot_margin_delta"]),
                           "seed_group_cv": by_seed, "bot_group_cv": by_bot})
    best = max(trials, key=lambda row: (row["robust_min_bot_win_delta"],
                                        row["robust_win_delta"],
                                        row["robust_median_bot_margin_delta"],
                                        -row["depth"], row["min_leaf"]))
    model = DecisionTreeClassifier(max_depth=best["depth"], min_samples_leaf=best["min_leaf"],
                                   random_state=20260920).fit(matrix, labels)
    tree = _tree_payload(model)
    tree["class_kind"] = "handoff_config"
    payload = {
        "schema": "r1-handoff-selector-v1", "eligible": bool(
            best["robust_win_delta"] > 0 and best["robust_min_bot_win_delta"] >= 0 and
            best["robust_margin_delta"] > 0 and best["robust_median_bot_margin_delta"] > 0),
        "samples": len(matrix), "classes": classes.tolist(), "oracle": oracle,
        "selected": best, "trials": trials, "tree": tree,
        "feature_mode": args.feature_mode, "feature_schema": feature_schema,
        "feature_order": feature_order, "feature_names": list(names),
        "feature_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "samples": len(matrix),
                      "eligible": payload["eligible"], "selected": best,
                      "oracle": oracle}))


if __name__ == "__main__":
    main()
