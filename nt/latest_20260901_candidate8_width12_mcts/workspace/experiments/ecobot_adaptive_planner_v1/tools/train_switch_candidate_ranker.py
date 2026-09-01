#!/usr/bin/env python3
"""Train and audit an offline public-state ranker for KEEP/SWITCH candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np


def add_keep_rows(
    features: np.ndarray,
    delta: np.ndarray,
    seed: np.ndarray,
    seat: np.ndarray,
    rank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    group_key = seed * 2 + seat
    keep_features = []
    keep_seed = []
    keep_seat = []
    for key in np.unique(group_key):
        source = features[np.flatnonzero(group_key == key)[0]].copy()
        source[11:15] = [-1, -1, 0, 0]
        source[16] = source[15]
        source[18] = source[17]
        source[20] = source[19]
        source[45:53] = source[37:45]
        source[72:80] = 0
        source[99] = source[98]
        source[101] = source[100]
        source[103] = source[102]
        source[105] = source[104]
        keep_features.append(source)
        keep_seed.append(key // 2)
        keep_seat.append(key % 2)
    return (
        np.concatenate([features, np.asarray(keep_features)], axis=0),
        np.concatenate([delta, np.zeros(len(keep_features))]),
        np.concatenate([seed, np.asarray(keep_seed, dtype=seed.dtype)]),
        np.concatenate([seat, np.asarray(keep_seat, dtype=seat.dtype)]),
        np.concatenate([rank, -np.ones(len(keep_features), dtype=rank.dtype)]),
    )


def relevance_labels(actual_delta: np.ndarray, group: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(actual_delta), dtype=np.int32)
    for key in np.unique(group):
        indices = np.flatnonzero(group == key)
        order = np.argsort(actual_delta[indices], kind="stable")
        labels[indices[order]] = np.arange(len(indices), dtype=np.int32)
    return labels


def ranking_metrics(
    prediction: np.ndarray,
    actual_delta: np.ndarray,
    group: np.ndarray,
) -> dict[str, float | int]:
    pairwise_correct = 0
    pairwise_total = 0
    top1 = 0
    top3 = 0
    chosen_delta = []
    groups = np.unique(group)
    for key in groups:
        indices = np.flatnonzero(group == key)
        actual = actual_delta[indices]
        predicted = prediction[indices]
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                actual_sign = np.sign(actual[left] - actual[right])
                if actual_sign == 0:
                    continue
                pairwise_total += 1
                pairwise_correct += actual_sign == np.sign(
                    predicted[left] - predicted[right]
                )
        actual_best = int(np.argmax(actual))
        predicted_order = np.argsort(-predicted, kind="stable")
        top1 += actual_best == int(predicted_order[0])
        top3 += actual_best in predicted_order[:3]
        chosen_delta.append(float(actual[int(predicted_order[0])]))
    selected = np.asarray(chosen_delta, dtype=np.float64)
    return {
        "groups": int(len(groups)),
        "pairwise_comparisons": pairwise_total,
        "pairwise_accuracy": (
            pairwise_correct / pairwise_total if pairwise_total else 0.0
        ),
        "top1_oracle_recall": top1 / len(groups) if len(groups) else 0.0,
        "top3_oracle_recall": top3 / len(groups) if len(groups) else 0.0,
        "mean_realized_delta": float(selected.mean()) if selected.size else 0.0,
        "positive_choice_rate": float(np.mean(selected > 0)) if selected.size else 0.0,
        "p10_realized_delta": (
            float(np.quantile(selected, 0.10)) if selected.size else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.dataset)
    features = np.asarray(data["features"], dtype=np.float64)
    expected_value_dataset = "expected_delta" in data.files
    actual_delta = np.asarray(
        data["expected_delta" if expected_value_dataset else "actual_delta"],
        dtype=np.float64,
    )
    rank = np.asarray(data["candidate_rank"], dtype=np.int16)
    seed = np.asarray(
        data["prefix_seed" if expected_value_dataset else "seed"],
        dtype=np.int64,
    )
    seat = np.asarray(data["seat"], dtype=np.int8)
    names = [str(value) for value in data["feature_names"]]
    # Every expected-value row is a SWITCH arm whose target is already measured
    # relative to KEEP.  KEEP is therefore a real candidate with delta zero,
    # not an implicit post-hoc threshold.  Omitting it made the old audit always
    # choose some switch, even when every available edit lost money, and greatly
    # overstated the deployability of the learned ranker.
    features, actual_delta, seed, seat, rank = add_keep_rows(
        features, actual_delta, seed, seat, rank
    )
    group = seed * 2 + seat
    labels = relevance_labels(actual_delta, group)
    analytic_prediction = features[:, 16].copy()
    ranker_oof = np.zeros(len(actual_delta), dtype=np.float64)
    regressor_oof = np.zeros(len(actual_delta), dtype=np.float64)
    ranker_importance = np.zeros(features.shape[1], dtype=np.float64)
    regressor_importance = np.zeros(features.shape[1], dtype=np.float64)

    for fold in range(5):
        training = seed % 5 != fold
        validation = ~training
        train_order = np.argsort(group[training], kind="stable")
        validation_order = np.flatnonzero(validation)
        train_indices = np.flatnonzero(training)[train_order]
        train_group = group[train_indices]
        _, train_group_sizes = np.unique(train_group, return_counts=True)

        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=240,
            num_leaves=15,
            max_depth=6,
            min_child_samples=30,
            learning_rate=0.025,
            reg_lambda=30.0,
            reg_alpha=8.0,
            verbosity=-1,
            n_jobs=16,
            random_state=300 + fold,
        )
        ranker.fit(
            features[train_indices],
            labels[train_indices],
            group=train_group_sizes.tolist(),
        )
        ranker_oof[validation_order] = ranker.predict(features[validation_order])
        ranker_importance += ranker.feature_importances_

        regressor = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=260,
            num_leaves=15,
            max_depth=6,
            min_child_samples=30,
            learning_rate=0.025,
            reg_lambda=30.0,
            reg_alpha=8.0,
            verbosity=-1,
            n_jobs=16,
            random_state=400 + fold,
        )
        regressor.fit(features[training], actual_delta[training])
        regressor_oof[validation_order] = regressor.predict(features[validation_order])
        regressor_importance += regressor.feature_importances_

    def importance(values: np.ndarray):
        order = np.argsort(-values)[:24]
        return [
            {"feature": names[index], "importance": float(values[index])}
            for index in order
        ]

    payload = {
        "schema": "kaggriculture.switch-candidate-ranker-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "candidate_rows_with_keep": int(len(actual_delta)),
        "groups": int(len(np.unique(group))),
        "analytic": ranking_metrics(
            analytic_prediction, actual_delta, group
        ),
        "lambdarank_oof": ranking_metrics(ranker_oof, actual_delta, group),
        "regression_l1_oof": ranking_metrics(
            regressor_oof, actual_delta, group
        ),
        "ranker_feature_importance": importance(ranker_importance),
        "regressor_feature_importance": importance(regressor_importance),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "groups": payload["groups"],
        "analytic": payload["analytic"],
        "lambdarank_oof": payload["lambdarank_oof"],
        "regression_l1_oof": payload["regression_l1_oof"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
