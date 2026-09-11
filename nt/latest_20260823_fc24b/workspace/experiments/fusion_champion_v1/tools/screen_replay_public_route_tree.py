#!/usr/bin/env python3
"""Screen whether an expert's route family is predictable from public state.

This is an observational audit, not a policy promotion tool.  The target is a
route family reconstructed from the completed Replay, while every feature is
restricted to one prospective decision frame.  Results are reported with
ordinary stratified OOF and opponent-submission-grouped OOF so a tree cannot
win merely by memorising one recurring opponent matchup.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text


def feature_families(frame: int, columns: list[str]) -> dict[str, list[str]]:
    prefix = f"f{frame}_"
    numeric = [
        column
        for column in columns
        if column.startswith(prefix)
        and column not in {f"{prefix}latest_shop"}
        and not column.endswith("_signature")
    ]
    shops = [column for column in numeric if column.startswith(f"{prefix}shop_")]
    market = [
        column
        for column in numeric
        if column.startswith(f"{prefix}market_inventory_")
        or column.startswith(f"{prefix}market_price_")
    ]
    opponent = [column for column in numeric if column.startswith(f"{prefix}opp_")]
    own = [column for column in numeric if column.startswith(f"{prefix}own_")]
    return {
        "shop_only": shops,
        "shop_market": sorted(set(shops + market)),
        "shop_opponent": sorted(set(shops + opponent)),
        "shop_market_opponent": sorted(set(shops + market + opponent)),
        "all_public": sorted(set(shops + market + opponent + own)),
    }


def oof_predictions(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    depth: int,
    min_leaf: int,
    grouped: bool,
    seed: int,
) -> tuple[np.ndarray, int]:
    if grouped:
        unique_groups = np.unique(groups)
        splits = min(5, len(unique_groups))
        if splits < 2:
            raise ValueError("fewer than two opponent groups")
        splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
        iterator = splitter.split(x, y, groups)
    else:
        smallest_class = min(Counter(y.tolist()).values())
        splits = min(5, smallest_class)
        if splits < 2:
            raise ValueError("a route class has fewer than two examples")
        splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        iterator = splitter.split(x, y)

    prediction = np.empty_like(y, dtype=object)
    seen = np.zeros(len(y), dtype=bool)
    for train, valid in iterator:
        model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=min_leaf,
            random_state=seed,
            class_weight="balanced",
        )
        model.fit(x[train], y[train])
        prediction[valid] = model.predict(x[valid])
        seen[valid] = True
    if not np.all(seen):
        raise RuntimeError("OOF splitter did not cover every row")
    return prediction, splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tree-output", type=Path, required=True)
    parser.add_argument("--decision-frame", type=int, default=24)
    parser.add_argument("--target", default="capital_signature")
    parser.add_argument(
        "--target-missing",
        default="__MISSING__",
        help=(
            "Explicit label used when the target column is empty.  Pass 0 for "
            "sparse count targets whose omitted value semantically means zero."
        ),
    )
    parser.add_argument(
        "--feature-family",
        choices=(
            "shop_only",
            "shop_market",
            "shop_opponent",
            "shop_market_opponent",
            "all_public",
        ),
        default="",
        help="Optionally restrict the screen to one auditable feature family.",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="Optional exact row filter; may be repeated.",
    )
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    frame = args.decision_frame
    data = pd.read_csv(args.episode_csv)
    applied_filters: dict[str, str] = {}
    for expression in args.filter:
        if "=" not in expression:
            raise ValueError(f"invalid --filter {expression!r}; expected COLUMN=VALUE")
        column, value = expression.split("=", 1)
        if column not in data:
            raise KeyError(f"filter column missing: {column}")
        data = data[data[column].astype(str) == value].copy()
        applied_filters[column] = value
    if data.empty:
        raise RuntimeError("filters selected no rows")
    if args.target not in data:
        raise KeyError(f"target missing: {args.target}")
    if "opponent_submission_id" not in data:
        raise KeyError("opponent_submission_id missing")

    # Pandas may keep a mixture of float NaN and strings in sparse Replay
    # columns.  Normalize before converting to an object array so label sorting
    # and sklearn both see one stable type.  The caller must choose whether an
    # omitted field means zero or a genuinely unknown label.
    y = (
        data[args.target]
        .fillna(args.target_missing)
        .astype(str)
        .to_numpy(dtype=str)
    )
    groups = data["opponent_submission_id"].fillna(-1).astype(str).to_numpy(dtype=object)
    labels = sorted(set(y.tolist()))
    families = feature_families(frame, list(data.columns))
    if args.feature_family:
        families = {args.feature_family: families[args.feature_family]}
    rows: list[dict[str, Any]] = []

    for family, features in families.items():
        if not features:
            continue
        x = data[features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float64)
        for depth in range(1, 7):
            for min_leaf in (1, 2, 3, 4):
                record: dict[str, Any] = {
                    "feature_family": family,
                    "feature_count": len(features),
                    "max_depth": depth,
                    "min_samples_leaf": min_leaf,
                }
                for name, grouped in (("stratified", False), ("opponent_grouped", True)):
                    try:
                        prediction, splits = oof_predictions(
                            x, y, groups, depth, min_leaf, grouped, args.seed
                        )
                        record[f"{name}_splits"] = splits
                        record[f"{name}_accuracy"] = float(accuracy_score(y, prediction))
                        record[f"{name}_balanced_accuracy"] = float(
                            balanced_accuracy_score(y, prediction)
                        )
                        record[f"{name}_confusion"] = confusion_matrix(
                            y, prediction, labels=labels
                        ).tolist()
                    except ValueError as exc:
                        record[f"{name}_error"] = str(exc)
                rows.append(record)

    valid = [row for row in rows if "opponent_grouped_balanced_accuracy" in row]
    if not valid:
        raise RuntimeError("no valid grouped OOF configuration")
    best = max(
        valid,
        key=lambda row: (
            row["opponent_grouped_balanced_accuracy"],
            row["opponent_grouped_accuracy"],
            -row["max_depth"],
            -row["feature_count"],
        ),
    )
    best_features = families[str(best["feature_family"])]
    best_x = (
        data[best_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(np.float64)
    )
    model = DecisionTreeClassifier(
        max_depth=int(best["max_depth"]),
        min_samples_leaf=int(best["min_samples_leaf"]),
        random_state=args.seed,
        class_weight="balanced",
    )
    model.fit(best_x, y)
    tree_text = export_text(model, feature_names=best_features, decimals=2)

    result = {
        "schema": "kaggriculture.replay-public-route-tree-screen.v1",
        "source_boundary": (
            "observational replay audit only; target is reconstructed from the "
            "future completed route and is not a deployable causal label"
        ),
        "decision_frame": frame,
        "target": args.target,
        "target_missing": args.target_missing,
        "filters": applied_filters,
        "episodes": len(data),
        "labels": labels,
        "label_counts": dict(Counter(y.tolist())),
        "opponent_groups": int(len(set(groups.tolist()))),
        "best": best,
        "best_features": best_features,
        "full_fit_tree": tree_text,
        "all_screens": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.tree_output.write_text(tree_text, encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("episodes", "labels", "label_counts", "opponent_groups", "best")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
