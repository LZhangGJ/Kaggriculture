#!/usr/bin/env python3
"""Train a groupwise KEEP/SWITCH ranker with untouched-route evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from train_switch_value_regressor import engineered_features, load, training_weight


VARIANTS = [
    {"num_leaves": 15, "max_depth": 6, "min_child_samples": 15,
     "reg_lambda": 10.0, "reg_alpha": 2.0},
    {"num_leaves": 31, "max_depth": 7, "min_child_samples": 20,
     "reg_lambda": 15.0, "reg_alpha": 3.0},
    {"num_leaves": 31, "max_depth": 8, "min_child_samples": 12,
     "reg_lambda": 8.0, "reg_alpha": 1.0},
    {"num_leaves": 63, "max_depth": 9, "min_child_samples": 12,
     "reg_lambda": 12.0, "reg_alpha": 2.0},
]
KEEP_MARGINS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_order(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(
        data.get("source_dataset_index", np.zeros(len(data["prefix_seed"]))),
        dtype=np.int64,
    )
    group_key = (
        source * 10**10
        + np.asarray(data["prefix_seed"], dtype=np.int64) * 2
        + np.asarray(data["seat"], dtype=np.int64)
    )
    order = np.argsort(group_key, kind="stable")
    _, sizes = np.unique(group_key[order], return_counts=True)
    return order, sizes


def relevance_labels(
    actual: np.ndarray, data: dict[str, np.ndarray], minimum_gap: float = 0.0
) -> np.ndarray:
    source = np.asarray(
        data.get("source_dataset_index", np.zeros(len(data["prefix_seed"]))),
        dtype=np.int64,
    )
    group_key = (
        source * 10**10
        + np.asarray(data["prefix_seed"], dtype=np.int64) * 2
        + np.asarray(data["seat"], dtype=np.int64)
    )
    labels = np.zeros(len(actual), dtype=np.int32)
    for key in np.unique(group_key):
        rows = np.flatnonzero(group_key == key)
        # Equal counterfactual values are the same relevance class.  The old
        # arange assignment silently imposed an arbitrary order on exact ties
        # (especially KEEP and no-effect edits), teaching the ranker false
        # preferences.  minimum_gap optionally merges differences smaller
        # than the Monte-Carlo noise scale; it is selected on validation only.
        order = np.argsort(actual[rows], kind="stable")
        relevance = 0
        anchor = float(actual[rows[order[0]]])
        labels[rows[order[0]]] = relevance
        required = max(1e-9, float(minimum_gap))
        for position in order[1:]:
            value = float(actual[rows[position]])
            if value - anchor >= required:
                relevance += 1
                anchor = value
            labels[rows[position]] = relevance
    return labels


def ranking_metrics(
    prediction: np.ndarray,
    actual: np.ndarray,
    data: dict[str, np.ndarray],
    keep_margin: float,
) -> dict[str, float | int]:
    seed = np.asarray(data["prefix_seed"], dtype=np.int64)
    seat = np.asarray(data["seat"], dtype=np.int64)
    rank = np.asarray(data["candidate_rank"], dtype=np.int64)
    group_key = seed * 2 + seat
    pair_correct = pair_total = 0
    gap_correct = gap_total = 0
    top1 = top3 = switch_count = 0
    chosen: list[float] = []
    for key in np.unique(group_key):
        rows = np.flatnonzero(group_key == key)
        scores = np.asarray(prediction[rows], dtype=np.float64).copy()
        values = np.asarray(actual[rows], dtype=np.float64)
        ranks = rank[rows]
        keep = np.flatnonzero(ranks < 0)
        if keep.size != 1:
            raise ValueError(f"group {key} has {keep.size} KEEP rows")
        scores[keep[0]] += keep_margin
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                difference = values[left] - values[right]
                if difference == 0:
                    continue
                correct = np.sign(difference) == np.sign(
                    scores[left] - scores[right]
                )
                pair_total += 1
                pair_correct += int(correct)
                if abs(difference) >= 2000.0:
                    gap_total += 1
                    gap_correct += int(correct)
        actual_best = int(np.argmax(values))
        predicted_order = np.argsort(-scores, kind="stable")
        top1 += actual_best == int(predicted_order[0])
        top3 += actual_best in predicted_order[:3]
        selected = int(predicted_order[0])
        chosen.append(float(values[selected]))
        switch_count += int(ranks[selected] >= 0)
    selected_values = np.asarray(chosen, dtype=np.float64)
    groups = len(np.unique(group_key))
    return {
        "groups": int(groups),
        "keep_margin": float(keep_margin),
        "pairwise_accuracy": pair_correct / pair_total if pair_total else 0.0,
        "pairwise_comparisons": pair_total,
        "pairwise_gap2000_accuracy": (
            gap_correct / gap_total if gap_total else 0.0
        ),
        "pairwise_gap2000_comparisons": gap_total,
        "top1_oracle_recall_with_keep": top1 / groups if groups else 0.0,
        "top3_oracle_recall_with_keep": top3 / groups if groups else 0.0,
        "mean_realized_delta": float(np.mean(selected_values)),
        "median_realized_delta": float(np.median(selected_values)),
        "p10_realized_delta": float(np.quantile(selected_values, 0.10)),
        "positive_choice_rate": float(np.mean(selected_values > 0)),
        "negative_choice_rate": float(np.mean(selected_values < 0)),
        "switch_rate": switch_count / groups if groups else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()

    train = load(args.train)
    validation = load(args.validation)
    test = load(args.test)
    for name, data in (("validation", validation), ("test", test)):
        if not np.array_equal(train["feature_names"], data["feature_names"]):
            raise ValueError(f"train/{name} feature schema mismatch")
    train_x, expanded_names, categorical = engineered_features(
        train["features"], train["feature_names"]
    )
    validation_x, _, _ = engineered_features(
        validation["features"], validation["feature_names"]
    )
    test_x, _, _ = engineered_features(test["features"], test["feature_names"])
    train_y = np.asarray(train["expected_delta"], dtype=np.float64)
    validation_y = np.asarray(validation["expected_delta"], dtype=np.float64)
    test_y = np.asarray(test["expected_delta"], dtype=np.float64)
    labels = relevance_labels(train_y, train)
    weights = training_weight(train)
    train_order, train_group_sizes = grouped_order(train)

    audits: list[dict[str, object]] = []
    models: list[lgb.LGBMRanker] = []
    best_key: tuple[float, float, float, float] | None = None
    best_index = -1
    best_margin = 0.0
    for index, variant in enumerate(VARIANTS):
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            lambdarank_truncation_level=5,
            n_estimators=500,
            learning_rate=0.025,
            verbosity=-1,
            n_jobs=16,
            random_state=2400 + index,
            **variant,
        )
        model.fit(
            train_x[train_order],
            labels[train_order],
            group=train_group_sizes.tolist(),
            sample_weight=weights[train_order],
            categorical_feature=categorical,
        )
        prediction = model.predict(validation_x)
        margin_audits = [
            ranking_metrics(prediction, validation_y, validation, margin)
            for margin in KEEP_MARGINS
        ]
        local_best = max(
            margin_audits,
            key=lambda item: (
                float(item["mean_realized_delta"]),
                float(item["p10_realized_delta"]),
                -float(item["negative_choice_rate"]),
                float(item["top3_oracle_recall_with_keep"]),
            ),
        )
        key = (
            float(local_best["mean_realized_delta"]),
            float(local_best["p10_realized_delta"]),
            -float(local_best["negative_choice_rate"]),
            float(local_best["top3_oracle_recall_with_keep"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_index = index
            best_margin = float(local_best["keep_margin"])
        audits.append({
            "variant_index": index,
            "parameters": variant,
            "keep_margin_audits": margin_audits,
            "selected_validation_metrics": local_best,
        })
        models.append(model)

    model = models[best_index]
    test_prediction = model.predict(test_x)
    test_margin_audits = [
        ranking_metrics(test_prediction, test_y, test, margin)
        for margin in KEEP_MARGINS
    ]
    test_metrics = next(
        row for row in test_margin_audits
        if float(row["keep_margin"]) == best_margin
    )
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.model_output))
    importance = model.feature_importances_
    order = np.argsort(-importance)[:50]
    payload = {
        "schema": "kaggriculture.switch-lambdarank-independent-test.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train": {"path": str(args.train), "sha256": sha256(args.train),
                  "rows": int(len(train_y))},
        "validation": {
            "path": str(args.validation), "sha256": sha256(args.validation),
            "rows": int(len(validation_y)), "used_for_selection": True,
        },
        "test": {
            "path": str(args.test), "sha256": sha256(args.test),
            "rows": int(len(test_y)), "used_for_selection": False,
        },
        "raw_feature_dim": int(train["features"].shape[1]),
        "expanded_feature_dim": int(train_x.shape[1]),
        "variants": audits,
        "selected_variant_index": best_index,
        "selected_keep_margin": best_margin,
        "selected_validation_metrics": audits[best_index][
            "selected_validation_metrics"
        ],
        "selected_test_metrics": test_metrics,
        "test_keep_margin_audits_not_used_for_selection": test_margin_audits,
        "model": {"path": str(args.model_output), "sha256": sha256(args.model_output)},
        "feature_importance": [
            {"feature": expanded_names[i], "importance": float(importance[i])}
            for i in order
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_variant_index": best_index,
        "selected_keep_margin": best_margin,
        "validation": payload["selected_validation_metrics"],
        "test": test_metrics,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
