#!/usr/bin/env python3
"""Train a direct pairwise project-choice policy with an untouched test set.

Unlike the scalar/LambdaRank arms, this model learns the actual decision the
planner has to make: given two complete project edits in the same state, which
one has the larger counterfactual continuation value?  KEEP is an ordinary
candidate.  At inference time candidates receive a probability-Borda score,
and an edit is allowed only when its direct probability of beating KEEP clears
the validation-selected safety threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from train_switch_safety_gate import selection_key
from train_switch_value_regressor import generalized_engineered_features, load


MODEL_VARIANTS = [
    {"num_leaves": 15, "max_depth": 6, "min_child_samples": 30,
     "reg_lambda": 15.0, "reg_alpha": 3.0},
    {"num_leaves": 31, "max_depth": 7, "min_child_samples": 24,
     "reg_lambda": 20.0, "reg_alpha": 4.0},
    {"num_leaves": 31, "max_depth": 8, "min_child_samples": 16,
     "reg_lambda": 12.0, "reg_alpha": 2.0},
    {"num_leaves": 63, "max_depth": 9, "min_child_samples": 20,
     "reg_lambda": 20.0, "reg_alpha": 4.0},
]
EXPANDED_MODEL_VARIANTS = MODEL_VARIANTS + [
    {
        "num_leaves": 7, "max_depth": 4, "min_child_samples": 50,
        "reg_lambda": 40.0, "reg_alpha": 8.0,
        "n_estimators": 800, "learning_rate": 0.015,
        "feature_fraction": 0.75,
    },
    {
        "num_leaves": 15, "max_depth": 5, "min_child_samples": 40,
        "reg_lambda": 30.0, "reg_alpha": 6.0,
        "n_estimators": 700, "learning_rate": 0.020,
        "feature_fraction": 0.80,
    },
    {
        "num_leaves": 31, "max_depth": 6, "min_child_samples": 40,
        "reg_lambda": 30.0, "reg_alpha": 6.0,
        "n_estimators": 600, "learning_rate": 0.020,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.85, "bagging_freq": 1,
    },
    {
        "num_leaves": 31, "max_depth": 8, "min_child_samples": 50,
        "reg_lambda": 40.0, "reg_alpha": 8.0,
        "n_estimators": 800, "learning_rate": 0.015,
        "feature_fraction": 0.85,
    },
    {
        "num_leaves": 63, "max_depth": 8, "min_child_samples": 40,
        "reg_lambda": 30.0, "reg_alpha": 5.0,
        "n_estimators": 600, "learning_rate": 0.020,
        "feature_fraction": 0.80,
    },
    {
        "num_leaves": 127, "max_depth": 10, "min_child_samples": 30,
        "reg_lambda": 35.0, "reg_alpha": 7.0,
        "n_estimators": 500, "learning_rate": 0.020,
        "feature_fraction": 0.70,
    },
    {
        "num_leaves": 31, "max_depth": 7, "min_child_samples": 80,
        "reg_lambda": 50.0, "reg_alpha": 10.0,
        "n_estimators": 800, "learning_rate": 0.015,
        "feature_fraction": 0.90,
    },
    {
        "num_leaves": 63, "max_depth": 9, "min_child_samples": 64,
        "reg_lambda": 50.0, "reg_alpha": 10.0,
        "n_estimators": 700, "learning_rate": 0.015,
        "feature_fraction": 0.85,
    },
]
TRAIN_MINIMUM_GAPS = [0.0, 500.0, 1000.0, 2000.0]
KEEP_WIN_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
KEEP_BONUSES = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]


def model_features(
    features: np.ndarray,
    names: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Build a backward-compatible or generalizing feature representation.

    Treating the complete shop bitmask as one categorical ID lets a tree
    memorize exact shop combinations and gives no similarity to a new mask.
    shop_bits keeps precisely the same public information while exposing each
    product shop as an independent reusable fact.
    """
    return generalized_engineered_features(features, names, mode)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def group_keys(data: dict[str, np.ndarray]) -> np.ndarray:
    source = np.asarray(
        data.get("source_dataset_index", np.zeros(len(data["prefix_seed"]))),
        dtype=np.int64,
    )
    seed = np.asarray(data["prefix_seed"], dtype=np.int64)
    seat = np.asarray(data["seat"], dtype=np.int64)
    return source * 10**10 + seed * 2 + seat


def pair_structure(
    data: dict[str, np.ndarray], include_exact_ties: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return every unordered within-state pair and its true value gap."""
    keys = group_keys(data)
    actual = np.asarray(data["expected_delta"], dtype=np.float64)
    left: list[int] = []
    right: list[int] = []
    pair_key: list[int] = []
    difference: list[float] = []
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        for left_position in range(len(rows)):
            for right_position in range(left_position + 1, len(rows)):
                left_row = int(rows[left_position])
                right_row = int(rows[right_position])
                gap = float(actual[left_row] - actual[right_row])
                if not include_exact_ties and gap == 0.0:
                    continue
                left.append(left_row)
                right.append(right_row)
                pair_key.append(int(key))
                difference.append(gap)
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(pair_key, dtype=np.int64),
        np.asarray(difference, dtype=np.float64),
    )


def pair_matrix(
    features: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    left_x = np.asarray(features[left], dtype=np.float32)
    right_x = np.asarray(features[right], dtype=np.float32)
    return np.concatenate([left_x, right_x, left_x - right_x], axis=1)


def row_standard_error(data: dict[str, np.ndarray]) -> np.ndarray:
    std = np.asarray(data["future_std"], dtype=np.float64)
    count = np.asarray(
        data.get(
            "future_sample_count",
            np.full(len(std), data["future_delta_samples"].shape[1]),
        ),
        dtype=np.float64,
    )
    return std / np.sqrt(np.maximum(1.0, count))


def paired_standard_error(
    data: dict[str, np.ndarray], left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Standard error for common-random-number counterfactual pairs.

    Every arm within a decision group is continued with the same ordered
    future seeds.  Estimating each arm independently discards that covariance
    and misstates label confidence, so use the observed paired differences.
    """
    if "future_delta_samples" not in data:
        row_error = row_standard_error(data)
        return np.sqrt(np.square(row_error[left]) + np.square(row_error[right]))
    samples = np.asarray(data["future_delta_samples"], dtype=np.float64)
    differences = samples[left] - samples[right]
    count = differences.shape[1]
    if count <= 1:
        return np.zeros(len(left), dtype=np.float64)
    return np.std(differences, axis=1, ddof=1) / np.sqrt(count)


def training_pairs(
    data: dict[str, np.ndarray],
    features: np.ndarray,
    minimum_gap: float,
    minimum_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    left, right, _, difference = pair_structure(data, include_exact_ties=False)
    pair_error = paired_standard_error(data, left, right)
    z_score = np.abs(difference) / np.maximum(pair_error, 1e-9)
    selected = (
        (np.abs(difference) >= max(1e-9, minimum_gap))
        & (z_score >= minimum_z)
    )
    left = left[selected]
    right = right[selected]
    difference = difference[selected]
    pair_error = pair_error[selected]
    forward = pair_matrix(features, left, right)
    backward = pair_matrix(features, right, left)
    matrix = np.concatenate([forward, backward], axis=0)
    label = np.concatenate([difference > 0.0, difference < 0.0]).astype(np.int8)
    confidence = 1.0 / (1.0 + np.square(pair_error / 2500.0))
    importance = 1.0 + np.minimum(np.abs(difference) / 3000.0, 3.0)
    one_way_weight = np.clip(confidence * importance, 0.05, 4.0)
    weight = np.concatenate([one_way_weight, one_way_weight]).astype(np.float32)
    return matrix, label, weight, len(left)


def predict_pairs(
    model: object,
    data: dict[str, np.ndarray],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left, right, pair_key, difference = pair_structure(data)
    matrix = pair_matrix(features, left, right)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(matrix)[:, 1]
    else:
        probability = model.predict(matrix)
    return left, right, pair_key, difference, np.asarray(probability, dtype=np.float64)


def pair_accuracy(
    difference: np.ndarray, probability: np.ndarray, minimum_gap: float
) -> dict[str, float | int]:
    selected = np.abs(difference) >= max(1e-9, minimum_gap)
    correct = (probability[selected] >= 0.5) == (difference[selected] > 0.0)
    return {
        "minimum_expected_value_gap": float(minimum_gap),
        "pairs": int(np.sum(selected)),
        "accuracy": float(np.mean(correct)) if len(correct) else 0.0,
    }


def w3_selection_key(
    pairwise: list[dict[str, float | int]],
    raw_screening: dict[str, object],
) -> tuple[float, ...]:
    """Validation-only key for the W3 project-ranking model.

    Model selection and deployment safety are separate questions.  W3 first
    needs a ranker that preserves the oracle candidate and orders complete
    projects correctly.  KEEP thresholds are calibrated only after choosing
    that ranker, so a conservative always-KEEP policy cannot select the model.
    """
    topk = raw_screening["topk_oracle_recall_with_keep"]
    top4 = float(topk["4"])
    top5 = float(topk["5"])
    return (
        float(top4 >= 0.90),
        float(pairwise[0]["accuracy"]),
        float(pairwise[2]["accuracy"]),
        top4,
        top5,
        float(raw_screening["mean_realized_delta"]),
    )


def policy_metrics(
    data: dict[str, np.ndarray],
    pair_prediction: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    keep_win_threshold: float,
    keep_bonus: float,
) -> dict[str, float | int]:
    left, right, pair_key, difference, probability = pair_prediction
    keys = group_keys(data)
    actual = np.asarray(data["expected_delta"], dtype=np.float64)
    candidate_rank = np.asarray(data["candidate_rank"], dtype=np.int64)
    chosen: list[float] = []
    topk_hits = np.zeros(5, dtype=np.int64)
    switch_count = 0
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        local_index = {int(row): position for position, row in enumerate(rows)}
        keep_rows = np.flatnonzero(candidate_rank[rows] < 0)
        if len(keep_rows) != 1:
            raise ValueError(f"group {key} has {len(keep_rows)} KEEP rows")
        keep_position = int(keep_rows[0])
        scores = np.zeros(len(rows), dtype=np.float64)
        beat_keep = np.zeros(len(rows), dtype=np.float64)
        beat_keep[keep_position] = 1.0
        matches = np.flatnonzero(pair_key == key)
        for pair_row in matches:
            left_position = local_index[int(left[pair_row])]
            right_position = local_index[int(right[pair_row])]
            chance = float(probability[pair_row])
            scores[left_position] += chance
            scores[right_position] += 1.0 - chance
            if right_position == keep_position:
                beat_keep[left_position] = chance
            elif left_position == keep_position:
                beat_keep[right_position] = 1.0 - chance
        scores[keep_position] += keep_bonus
        allowed = (candidate_rank[rows] < 0) | (beat_keep >= keep_win_threshold)
        scores[~allowed] = -np.inf
        predicted_order = np.argsort(-scores, kind="stable")
        selected = int(predicted_order[0])
        values = actual[rows]
        actual_best = int(np.argmax(values))
        finite_order = [int(i) for i in predicted_order if np.isfinite(scores[i])]
        # Top-K is the screening recall of candidates that would actually be
        # admitted by the KEEP safety filter.  Do not count filtered rows just
        # because argsort places their -inf scores at the tail.
        for top_k in range(1, 6):
            topk_hits[top_k - 1] += actual_best in finite_order[:top_k]
        chosen.append(float(values[selected]))
        switch_count += int(candidate_rank[rows[selected]] >= 0)
    selected_value = np.asarray(chosen, dtype=np.float64)
    groups = len(selected_value)
    return {
        "groups": int(groups),
        "keep_win_threshold": float(keep_win_threshold),
        "keep_bonus": float(keep_bonus),
        "mean_realized_delta": float(np.mean(selected_value)),
        "median_realized_delta": float(np.median(selected_value)),
        "p10_realized_delta": float(np.quantile(selected_value, 0.10)),
        "min_realized_delta": float(np.min(selected_value)),
        "positive_choice_rate": float(np.mean(selected_value > 0.0)),
        "negative_choice_rate": float(np.mean(selected_value < 0.0)),
        "switch_rate": switch_count / groups if groups else 0.0,
        "top1_oracle_recall_with_keep": topk_hits[0] / groups if groups else 0.0,
        "top3_oracle_recall_with_keep": topk_hits[2] / groups if groups else 0.0,
        "topk_oracle_recall_with_keep": {
            str(top_k): topk_hits[top_k - 1] / groups if groups else 0.0
            for top_k in range(1, 6)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument(
        "--feature-mode",
        choices=(
            "legacy", "shop_bits", "shop_bits_scale",
            "shop_bits_scale_catfix", "shop_bits_scale_marginal",
            "shop_bits_scale_forecast",
            "shop_bits_scale_forecast_catfix",
            "shop_bits_scale_forecast_marginal_catfix",
        ),
        default="legacy",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("safety", "w3"),
        default="safety",
        help=(
            "safety preserves legacy deployment-oriented selection; w3 "
            "selects the ranking model by pair accuracy/oracle recall and "
            "calibrates the KEEP safety policy only afterwards"
        ),
    )
    parser.add_argument(
        "--variant-profile", choices=("default", "expanded"), default="default",
        help="Expanded adds stronger regularisation and row/feature subsampling arms.",
    )
    parser.add_argument(
        "--minimum-gaps", default=",".join(str(value) for value in TRAIN_MINIMUM_GAPS),
        help="Comma-separated training label-gap thresholds.",
    )
    parser.add_argument(
        "--minimum-zs", default="0",
        help=(
            "Comma-separated paired label-confidence thresholds.  Z uses the "
            "common-future difference standard error, not independent-arm error."
        ),
    )
    args = parser.parse_args()

    model_variants = (
        MODEL_VARIANTS if args.variant_profile == "default"
        else EXPANDED_MODEL_VARIANTS
    )
    minimum_gaps = [
        float(value) for value in args.minimum_gaps.split(",") if value.strip()
    ]
    if not minimum_gaps:
        raise ValueError("minimum-gaps must not be empty")
    minimum_zs = [
        float(value) for value in args.minimum_zs.split(",") if value.strip()
    ]
    if not minimum_zs:
        raise ValueError("minimum-zs must not be empty")

    train = load(args.train)
    validation = load(args.validation)
    test = load(args.test)
    for name, data in (("validation", validation), ("test", test)):
        if not np.array_equal(train["feature_names"], data["feature_names"]):
            raise ValueError(f"train/{name} feature schema mismatch")

    train_x, expanded_names, categorical = model_features(
        train["features"], train["feature_names"], args.feature_mode
    )
    validation_x, _, _ = model_features(
        validation["features"], validation["feature_names"], args.feature_mode
    )
    test_x, _, _ = model_features(
        test["features"], test["feature_names"], args.feature_mode
    )
    pair_categorical = categorical + [len(expanded_names) + index for index in categorical]

    model_audits: list[dict[str, object]] = []
    models: list[lgb.LGBMClassifier] = []
    best_key: tuple[float, ...] | None = None
    best_model_index = -1
    best_policy: tuple[float, float] = (0.5, 0.0)
    for minimum_gap in minimum_gaps:
        for minimum_z in minimum_zs:
            matrix, label, weight, pair_count = training_pairs(
                train, train_x, minimum_gap, minimum_z
            )
            for variant_index, variant in enumerate(model_variants):
                parameters = {
                    "objective": "binary",
                    "n_estimators": 500,
                    "learning_rate": 0.025,
                    "verbosity": -1,
                    "n_jobs": 16,
                    "random_state": (
                        6100 + int(minimum_gap) + int(round(100 * minimum_z))
                        + variant_index
                    ),
                }
                parameters.update(variant)
                model = lgb.LGBMClassifier(**parameters)
                model.fit(
                    matrix,
                    label,
                    sample_weight=weight,
                    categorical_feature=pair_categorical,
                )
                validation_prediction = predict_pairs(model, validation, validation_x)
                policy_grid = [
                    policy_metrics(validation, validation_prediction, threshold, bonus)
                    for threshold in KEEP_WIN_THRESHOLDS
                    for bonus in KEEP_BONUSES
                ]
                selected_policy = max(policy_grid, key=selection_key)
                _, _, _, validation_difference, validation_probability = validation_prediction
                validation_pairwise = [
                    pair_accuracy(validation_difference, validation_probability, gap)
                    for gap in (0.0, 500.0, 1000.0, 2000.0)
                ]
                raw_screening_policy = policy_metrics(
                    validation, validation_prediction, 0.0, 0.0
                )
                key = (
                    selection_key(selected_policy)
                    if args.selection_mode == "safety"
                    else w3_selection_key(validation_pairwise, raw_screening_policy)
                )
                model_index = len(models)
                if best_key is None or key > best_key:
                    best_key = key
                    best_model_index = model_index
                    best_policy = (
                        float(selected_policy["keep_win_threshold"]),
                        float(selected_policy["keep_bonus"]),
                    )
                model_audits.append({
                    "model_index": model_index,
                    "train_minimum_gap": float(minimum_gap),
                    "train_minimum_z": float(minimum_z),
                    "variant_index": variant_index,
                    "parameters": parameters,
                    "training_unordered_pairs": int(pair_count),
                    "validation_pairwise": validation_pairwise,
                    "raw_validation_screening_policy": raw_screening_policy,
                    "selected_validation_policy": selected_policy,
                })
                models.append(model)

    selected_model = models[best_model_index]
    threshold, bonus = best_policy
    test_prediction = predict_pairs(selected_model, test, test_x)
    _, _, _, test_difference, test_probability = test_prediction
    test_metrics = policy_metrics(test, test_prediction, threshold, bonus)
    test_pairwise = [
        pair_accuracy(test_difference, test_probability, gap)
        for gap in (0.0, 500.0, 1000.0, 2000.0)
    ]
    selected_audit = model_audits[best_model_index]

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    selected_model.booster_.save_model(str(args.model_output))
    importance = np.asarray(selected_model.feature_importances_, dtype=np.float64)
    pair_names = (
        [f"left:{name}" for name in expanded_names]
        + [f"right:{name}" for name in expanded_names]
        + [f"delta:{name}" for name in expanded_names]
    )
    importance_order = np.argsort(-importance)[:50]
    payload = {
        "schema": "kaggriculture.switch-pairwise-policy-independent-test.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "invariants": {
            "test_used_for_selection": False,
            "keep_is_ordinary_candidate": True,
            "candidate_requires_direct_win_over_keep": True,
            "source_in_group_key": True,
        },
        "train": {"path": str(args.train), "sha256": sha256(args.train),
                  "rows": int(len(train["expected_delta"]))},
        "validation": {"path": str(args.validation), "sha256": sha256(args.validation),
                       "rows": int(len(validation["expected_delta"]))},
        "test": {"path": str(args.test), "sha256": sha256(args.test),
                 "rows": int(len(test["expected_delta"]))},
        "raw_feature_dim": int(train["features"].shape[1]),
        "expanded_feature_dim": int(train_x.shape[1]),
        "pair_feature_dim": int(train_x.shape[1] * 3),
        "feature_mode": args.feature_mode,
        "selection_mode": args.selection_mode,
        "variant_profile": args.variant_profile,
        "minimum_gaps": minimum_gaps,
        "minimum_zs": minimum_zs,
        "model_audits": model_audits,
        "selected_model_index": int(best_model_index),
        "selected_model": selected_audit,
        "selected_keep_win_threshold": threshold,
        "selected_keep_bonus": bonus,
        "selected_test_pairwise_not_used_for_selection": test_pairwise,
        "selected_test_policy_not_used_for_selection": test_metrics,
        "model": {"path": str(args.model_output), "sha256": sha256(args.model_output)},
        "feature_importance": [
            {"feature": pair_names[index], "importance": float(importance[index])}
            for index in importance_order
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_model_index": best_model_index,
        "selected_train_minimum_gap": selected_audit["train_minimum_gap"],
        "selected_train_minimum_z": selected_audit["train_minimum_z"],
        "selected_variant_index": selected_audit["variant_index"],
        "selected_validation_policy": selected_audit["selected_validation_policy"],
        "test_pairwise": test_pairwise,
        "test_policy": test_metrics,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
