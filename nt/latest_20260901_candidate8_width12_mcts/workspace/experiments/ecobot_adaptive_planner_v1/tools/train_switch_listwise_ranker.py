#!/usr/bin/env python3
"""Train a listwise project ranker on complete KEEP/SWITCH candidate groups."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from train_switch_pairwise_policy import group_keys, model_features
from train_switch_value_regressor import load


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_groups(data: dict[str, np.ndarray]) -> tuple[np.ndarray, list[int]]:
    keys = group_keys(data)
    order = np.argsort(keys, kind="stable")
    _, sizes = np.unique(keys[order], return_counts=True)
    return order, [int(value) for value in sizes]


def relevance_labels(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Use tie-aware within-state ordinal relevance, never route identity."""
    labels = np.zeros(len(values), dtype=np.int32)
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        unique = np.unique(values[rows])
        labels[rows] = np.searchsorted(unique, values[rows]).astype(np.int32)
    return labels


def row_weights(data: dict[str, np.ndarray]) -> np.ndarray:
    count = np.asarray(
        data.get(
            "future_sample_count",
            np.full(len(data["future_std"]), data["future_delta_samples"].shape[1]),
        ),
        dtype=np.float64,
    )
    standard_error = np.asarray(data["future_std"], dtype=np.float64) / np.sqrt(
        np.maximum(count, 1.0)
    )
    return np.clip(1.0 / (1.0 + np.square(standard_error / 2500.0)), 0.05, 1.0)


def ranking_metrics(
    score: np.ndarray,
    data: dict[str, np.ndarray],
) -> dict[str, object]:
    keys = group_keys(data)
    actual = np.asarray(data["expected_delta"], dtype=np.float64)
    pair_correct = {gap: 0 for gap in (0.0, 500.0, 1000.0, 2000.0)}
    pair_total = {gap: 0 for gap in pair_correct}
    top_hits = {k: 0 for k in range(1, 6)}
    selected_values: list[float] = []
    groups = np.unique(keys)
    for key in groups:
        rows = np.flatnonzero(keys == key)
        value = actual[rows]
        prediction = score[rows]
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                difference = float(value[left] - value[right])
                if difference == 0.0:
                    continue
                correct = np.sign(prediction[left] - prediction[right]) == np.sign(
                    difference
                )
                for gap in pair_total:
                    if abs(difference) >= max(1e-9, gap):
                        pair_total[gap] += 1
                        pair_correct[gap] += int(correct)
        oracle = int(np.argmax(value))
        ranked = np.argsort(-prediction, kind="stable")
        for k in top_hits:
            top_hits[k] += int(oracle in ranked[:k])
        selected_values.append(float(value[int(ranked[0])]))
    selected = np.asarray(selected_values, dtype=np.float64)
    return {
        "groups": int(len(groups)),
        "pairwise": [
            {
                "minimum_expected_value_gap": gap,
                "pairs": pair_total[gap],
                "accuracy": (
                    pair_correct[gap] / pair_total[gap] if pair_total[gap] else 0.0
                ),
            }
            for gap in pair_total
        ],
        "topk_oracle_recall_with_keep": {
            str(k): top_hits[k] / len(groups) if len(groups) else 0.0
            for k in top_hits
        },
        "mean_realized_delta": float(np.mean(selected)) if len(selected) else 0.0,
        "p10_realized_delta": (
            float(np.quantile(selected, 0.10)) if len(selected) else 0.0
        ),
        "negative_choice_rate": (
            float(np.mean(selected < 0.0)) if len(selected) else 0.0
        ),
    }


def selection_key(metrics: dict[str, object]) -> tuple[float, ...]:
    pair = float(metrics["pairwise"][0]["accuracy"])
    top4 = float(metrics["topk_oracle_recall_with_keep"]["4"])
    top5 = float(metrics["topk_oracle_recall_with_keep"]["5"])
    # W3 needs both gates.  Before either passes, maximize the weaker normalized
    # gate instead of allowing pair accuracy to hide poor oracle preservation.
    joint_progress = min(pair / 0.75, top4 / 0.90)
    return (
        float(pair >= 0.75 and top4 >= 0.90),
        joint_progress,
        top4,
        pair,
        top5,
        float(metrics["mean_realized_delta"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument(
        "--feature-mode",
        default="shop_bits_scale_forecast_marginal_catfix",
    )
    args = parser.parse_args()

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
    train_keys = group_keys(train)
    labels = relevance_labels(
        np.asarray(train["expected_delta"], dtype=np.float64), train_keys
    )
    order, group_sizes = ordered_groups(train)
    weight = row_weights(train)

    variants = [
        {"objective": "lambdarank", "num_leaves": 15, "max_depth": 6,
         "min_child_samples": 30, "reg_lambda": 20.0, "reg_alpha": 4.0},
        {"objective": "lambdarank", "num_leaves": 31, "max_depth": 7,
         "min_child_samples": 24, "reg_lambda": 20.0, "reg_alpha": 4.0},
        {"objective": "lambdarank", "num_leaves": 63, "max_depth": 9,
         "min_child_samples": 30, "reg_lambda": 30.0, "reg_alpha": 6.0},
        {"objective": "rank_xendcg", "num_leaves": 15, "max_depth": 6,
         "min_child_samples": 30, "reg_lambda": 20.0, "reg_alpha": 4.0},
        {"objective": "rank_xendcg", "num_leaves": 31, "max_depth": 7,
         "min_child_samples": 24, "reg_lambda": 20.0, "reg_alpha": 4.0},
        {"objective": "rank_xendcg", "num_leaves": 63, "max_depth": 9,
         "min_child_samples": 30, "reg_lambda": 30.0, "reg_alpha": 6.0},
    ]
    models: list[lgb.LGBMRanker] = []
    audits: list[dict[str, object]] = []
    best_index = -1
    best_key: tuple[float, ...] | None = None
    for index, variant in enumerate(variants):
        model = lgb.LGBMRanker(
            n_estimators=500,
            learning_rate=0.025,
            verbosity=-1,
            n_jobs=16,
            random_state=7300 + index,
            **variant,
        )
        model.fit(
            train_x[order],
            labels[order],
            group=group_sizes,
            sample_weight=weight[order],
            categorical_feature=categorical,
        )
        validation_metrics = ranking_metrics(model.predict(validation_x), validation)
        key = selection_key(validation_metrics)
        if best_key is None or key > best_key:
            best_key = key
            best_index = index
        audits.append({
            "variant_index": index,
            "parameters": variant,
            "validation_metrics": validation_metrics,
        })
        models.append(model)

    selected = models[best_index]
    test_metrics = ranking_metrics(selected.predict(test_x), test)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    selected.booster_.save_model(str(args.model_output))
    importance = selected.feature_importances_
    importance_order = np.argsort(-importance)[:50]
    payload = {
        "schema": "kaggriculture.switch-listwise-ranker-independent-test.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "invariants": {
            "test_used_for_selection": False,
            "keep_is_ordinary_candidate": True,
            "route_or_opponent_identity_used": False,
            "source_in_group_key": True,
        },
        "train": {"path": str(args.train), "sha256": sha256(args.train),
                  "rows": int(len(train["features"]))},
        "validation": {"path": str(args.validation),
                       "sha256": sha256(args.validation),
                       "rows": int(len(validation["features"]))},
        "test": {"path": str(args.test), "sha256": sha256(args.test),
                 "rows": int(len(test["features"]))},
        "feature_mode": args.feature_mode,
        "raw_feature_dim": int(train["features"].shape[1]),
        "expanded_feature_dim": int(train_x.shape[1]),
        "variants": audits,
        "selected_variant_index": best_index,
        "selected_validation_metrics": audits[best_index]["validation_metrics"],
        "selected_test_metrics_not_used_for_selection": test_metrics,
        "model": {"path": str(args.model_output)},
        "feature_importance": [
            {
                "feature": expanded_names[index],
                "importance": float(importance[index]),
            }
            for index in importance_order
        ],
    }
    payload["model"]["sha256"] = sha256(args.model_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_variant_index": best_index,
        "validation": payload["selected_validation_metrics"],
        "test": test_metrics,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
