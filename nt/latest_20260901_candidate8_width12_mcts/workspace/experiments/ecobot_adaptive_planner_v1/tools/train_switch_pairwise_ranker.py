#!/usr/bin/env python3
"""Train a pairwise KEEP/SWITCH ranker on expected continuation values."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

try:
    from catboost import CatBoostClassifier
except ImportError:  # CatBoost is an optional audit arm, never a hard dependency.
    CatBoostClassifier = None


def pair_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([left, right, left - right], axis=-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.dataset)
    features = np.asarray(data["features"], dtype=np.float64)
    expected_delta = np.asarray(data["expected_delta"], dtype=np.float64)
    prefix_seed = np.asarray(data["prefix_seed"], dtype=np.int64)
    seat = np.asarray(data["seat"], dtype=np.int8)
    group = prefix_seed * 2 + seat
    groups = np.unique(group)

    pair_left = []
    pair_right = []
    pair_group = []
    pair_difference = []
    for key in groups:
        indices = np.flatnonzero(group == key)
        for left_position in range(len(indices)):
            for right_position in range(left_position + 1, len(indices)):
                left = int(indices[left_position])
                right = int(indices[right_position])
                difference = expected_delta[left] - expected_delta[right]
                if difference == 0:
                    continue
                pair_left.append(left)
                pair_right.append(right)
                pair_group.append(key)
                pair_difference.append(difference)
    pair_left = np.asarray(pair_left, dtype=np.int64)
    pair_right = np.asarray(pair_right, dtype=np.int64)
    pair_group = np.asarray(pair_group, dtype=np.int64)
    pair_difference = np.asarray(pair_difference, dtype=np.float64)

    oof_probability = np.zeros(len(pair_left), dtype=np.float64)
    catboost_oof_probability = np.zeros(len(pair_left), dtype=np.float64)
    importance = np.zeros(features.shape[1] * 3, dtype=np.float64)
    catboost_importance = np.zeros(features.shape[1] * 3, dtype=np.float64)
    for fold in range(5):
        training = (pair_group // 2) % 5 != fold
        validation = ~training
        left = pair_features(features[pair_left[training]], features[pair_right[training]])
        right = pair_features(features[pair_right[training]], features[pair_left[training]])
        train_matrix = np.concatenate([left, right], axis=0)
        labels = np.concatenate([
            pair_difference[training] > 0,
            pair_difference[training] < 0,
        ]).astype(np.int8)
        weights_one_way = np.clip(
            1.0 + np.abs(pair_difference[training]) / 2000.0,
            1.0,
            8.0,
        )
        weights = np.concatenate([weights_one_way, weights_one_way])
        model = lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=20,
            max_depth=7,
            min_child_samples=20,
            learning_rate=0.025,
            reg_lambda=25.0,
            reg_alpha=5.0,
            verbosity=-1,
            n_jobs=16,
            random_state=500 + fold,
        )
        model.fit(train_matrix, labels, sample_weight=weights)
        validation_matrix = pair_features(
            features[pair_left[validation]], features[pair_right[validation]]
        )
        oof_probability[validation] = model.predict_proba(validation_matrix)[:, 1]
        importance += model.feature_importances_

        if CatBoostClassifier is not None:
            catboost = CatBoostClassifier(
                iterations=450,
                depth=6,
                learning_rate=0.025,
                l2_leaf_reg=20.0,
                loss_function="Logloss",
                random_seed=600 + fold,
                thread_count=16,
                verbose=False,
                allow_writing_files=False,
            )
            catboost.fit(train_matrix, labels, sample_weight=weights)
            catboost_oof_probability[validation] = catboost.predict_proba(
                validation_matrix
            )[:, 1]
            catboost_importance += catboost.feature_importances_

    def pair_accuracy(
        probability: np.ndarray, minimum_gap: float
    ) -> dict[str, float | int]:
        selected = np.abs(pair_difference) >= minimum_gap
        correct = (probability[selected] >= 0.5) == (
            pair_difference[selected] > 0
        )
        return {
            "minimum_expected_value_gap": minimum_gap,
            "pairs": int(selected.sum()),
            "accuracy": float(correct.mean()) if correct.size else 0.0,
        }

    def evaluate(probability: np.ndarray) -> dict[str, object]:
        top1 = 0
        top3 = 0
        chosen_delta = []
        for key in groups:
            indices = np.flatnonzero(group == key)
            scores = np.zeros(len(indices), dtype=np.float64)
            for left_position in range(len(indices)):
                for right_position in range(left_position + 1, len(indices)):
                    left = int(indices[left_position])
                    right = int(indices[right_position])
                    matches = np.flatnonzero(
                        (pair_group == key) &
                        (pair_left == left) &
                        (pair_right == right)
                    )
                    if not matches.size:
                        continue
                    chance = probability[int(matches[0])]
                    scores[left_position] += chance
                    scores[right_position] += 1.0 - chance
            actual = expected_delta[indices]
            actual_best = int(np.argmax(actual))
            predicted_order = np.argsort(-scores, kind="stable")
            top1 += actual_best == int(predicted_order[0])
            top3 += actual_best in predicted_order[:3]
            chosen_delta.append(float(actual[int(predicted_order[0])]))
        chosen = np.asarray(chosen_delta, dtype=np.float64)
        return {
            "pairwise": [
                pair_accuracy(probability, 0.0),
                pair_accuracy(probability, 250.0),
                pair_accuracy(probability, 500.0),
                pair_accuracy(probability, 1000.0),
                pair_accuracy(probability, 2000.0),
            ],
            "top1_oracle_recall": top1 / len(groups) if len(groups) else 0.0,
            "top3_oracle_recall": top3 / len(groups) if len(groups) else 0.0,
            "mean_realized_delta": float(chosen.mean()) if chosen.size else 0.0,
            "positive_choice_rate": (
                float(np.mean(chosen > 0)) if chosen.size else 0.0
            ),
            "p10_realized_delta": (
                float(np.quantile(chosen, 0.10)) if chosen.size else 0.0
            ),
        }

    feature_names = [str(value) for value in data["feature_names"]]
    expanded_names = (
        [f"left:{name}" for name in feature_names] +
        [f"right:{name}" for name in feature_names] +
        [f"delta:{name}" for name in feature_names]
    )
    order = np.argsort(-importance)[:32]
    catboost_order = np.argsort(-catboost_importance)[:32]
    payload = {
        "schema": "kaggriculture.switch-pairwise-ranker-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "groups": int(len(groups)),
        "lightgbm": evaluate(oof_probability),
        "catboost": (
            evaluate(catboost_oof_probability)
            if CatBoostClassifier is not None
            else {"status": "SKIPPED", "reason": "catboost_not_installed"}
        ),
        "lightgbm_feature_importance": [
            {"feature": expanded_names[index], "importance": float(importance[index])}
            for index in order
        ],
        "catboost_feature_importance": (
            [
                {
                    "feature": expanded_names[index],
                    "importance": float(catboost_importance[index]),
                }
                for index in catboost_order
            ]
            if CatBoostClassifier is not None
            else []
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "groups": payload["groups"],
        "lightgbm": payload["lightgbm"],
        "catboost": payload["catboost"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
