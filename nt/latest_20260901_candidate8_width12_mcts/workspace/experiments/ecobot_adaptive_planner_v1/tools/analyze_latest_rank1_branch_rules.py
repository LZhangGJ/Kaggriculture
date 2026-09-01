#!/usr/bin/env python3
"""Extract auditable public-state branch rules from the latest Rank-1 replays.

The labels describe observable route outcomes.  They are not claims about the
hidden source code.  Only public state available at the checkpoint is used as
input; author identity, episode id, future events, and raw action coordinates
are excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text


SHOPS = (
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
    "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET",
)
PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--submission-id", type=int, default=55714246)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def number(row: dict[str, str], name: str) -> float:
    raw = row.get(name, "")
    return float(raw) if raw not in ("", None) else 0.0


def public_features(row: dict[str, str], day: int) -> tuple[list[str], list[float]]:
    prefix = f"D{day}:"
    names = ["seat", "own:money", "own:land", "own:visible_yield",
             "opp:money", "opp:land", "opp:visible_yield"]
    names += [f"price:{item}" for item in PRODUCTS]
    names += [f"own:crop:{crop}" for crop in CROPS]
    names += [f"own:animal:{animal}" for animal in ANIMALS]
    names += [f"opp:crop:{crop}" for crop in CROPS]
    names += [f"opp:animal:{animal}" for animal in ANIMALS]
    names += [f"shop:{shop}" for shop in SHOPS]
    return names, [number(row, prefix + name) for name in names]


def total(row: dict[str, str], suffix: str, phases: range = range(6)) -> float:
    return sum(number(row, f"P{phase}:{suffix}") for phase in phases)


def fit_rule(rows: list[dict[str, str]], labels: np.ndarray, day: int) -> dict:
    feature_names, _ = public_features(rows[0], day)
    x = np.asarray([public_features(row, day)[1] for row in rows], dtype=np.float64)
    counts = Counter(int(value) for value in labels)
    folds = min(5, min(counts.values()))
    model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=8,
        random_state=82828,
        class_weight="balanced",
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=82828)
    prediction = cross_val_predict(model, x, labels, cv=cv)
    model.fit(x, labels)
    importances = sorted(
        ((feature_names[index], float(value))
         for index, value in enumerate(model.feature_importances_) if value > 0),
        key=lambda pair: -pair[1],
    )
    return {
        "checkpoint_day": day,
        "samples": len(rows),
        "label_counts": dict(sorted(counts.items())),
        "balanced_accuracy_cv": float(balanced_accuracy_score(labels, prediction)),
        "confusion_matrix_cv": confusion_matrix(labels, prediction).tolist(),
        "top_features": [
            {"feature": name, "importance": value}
            for name, value in importances
        ],
        "tree_text": export_text(model, feature_names=feature_names),
    }


def main() -> int:
    args = parse_args()
    with args.features.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if int(row["submission_id"]) == args.submission_id]
    if not rows:
        raise SystemExit(f"no rows for submission {args.submission_id}")

    # Reuse the audited two-cluster macro-family label, but assign its semantic
    # direction from the observed investment totals rather than assuming a
    # fixed numeric KMeans label.  The cluster itself was fitted only from
    # coarse phase-level investment totals, not raw coordinates.
    clusters = sorted({int(row["cluster"]) for row in rows})
    sheep_mean = {
        cluster: float(np.mean([
            total(row, "BUY_ANIMAL:SHEEP")
            for row in rows if int(row["cluster"]) == cluster
        ]))
        for cluster in clusters
    }
    sheep_cluster = max(sheep_mean, key=sheep_mean.get)
    animal_labels = np.asarray([
        int(int(row["cluster"]) == sheep_cluster) for row in rows
    ], dtype=np.int32)

    # Observable late suffix: at least one carrot seed is purchased in days
    # 20-29.  The latest Rank-1 audit identifies this as the carrot suffix; the
    # other completed branch remains wheat-heavy.  This label uses future
    # actions only as the supervised target, never as a selector input.
    late_carrot = np.asarray([
        total(row, "BUY_SEED:CARROT", range(4, 6)) for row in rows
    ], dtype=np.float64)
    late_wheat = np.asarray([
        total(row, "BUY_SEED:WHEAT", range(4, 6)) for row in rows
    ], dtype=np.float64)
    # Ten or more late carrot seeds reproduces the 93/58 carrot-vs-wheat split
    # reported by the frozen latest-submission audit.  Tiny 1-9 unit repairs do
    # not constitute a different macro suffix.
    suffix_labels = (late_carrot >= 10).astype(np.int32)

    payload = {
        "schema": "kaggriculture.rank1-latest-public-branch-rules.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "submission_id": args.submission_id,
        "evidence_boundary": (
            "Observable route outcomes predicted only from public checkpoint "
            "state; not a claim about hidden source-code conditions."
        ),
        "episodes": len(rows),
        "labels": {
            "animal": "1=sheep-heavy, 0=cow/mixed",
            "late_suffix": "1=at least 10 late carrot seeds, 0=wheat-heavy suffix",
            "late_carrot_quantity": {
                "mean": float(late_carrot.mean()),
                "positive_episodes": int(np.count_nonzero(late_carrot)),
            },
            "late_wheat_quantity": {
                "mean": float(late_wheat.mean()),
                "positive_episodes": int(np.count_nonzero(late_wheat)),
            },
        },
        "animal_rules": [fit_rule(rows, animal_labels, day) for day in (4, 9)],
        "late_suffix_rules": [fit_rule(rows, suffix_labels, day) for day in (19, 24)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
