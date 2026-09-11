#!/usr/bin/env python3
"""Train a two-stage safe SWITCH policy and audit untouched route families.

The first model ranks KEEP and local project edits.  The second model answers a
different question: whether an edit has enough evidence of positive expected
value to be allowed at all.  Model/threshold selection uses only the validation
route families; the test route families are read once after selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from train_switch_lambdarank import (
    VARIANTS as RANKER_VARIANTS,
    grouped_order,
    relevance_labels,
)
from train_switch_value_regressor import (
    generalized_engineered_features,
    load,
    training_weight,
)


GATE_VARIANTS = [
    {"num_leaves": 15, "max_depth": 5, "min_child_samples": 20,
     "reg_lambda": 15.0, "reg_alpha": 3.0},
    {"num_leaves": 31, "max_depth": 6, "min_child_samples": 20,
     "reg_lambda": 20.0, "reg_alpha": 4.0},
    {"num_leaves": 31, "max_depth": 7, "min_child_samples": 12,
     "reg_lambda": 12.0, "reg_alpha": 2.0},
]
LABEL_Z = [0.0, 0.5, 1.0]
GATE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
KEEP_MARGINS = [0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
RANK_LABEL_MINIMUM_GAPS = [0.0, 500.0, 1000.0, 2000.0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def group_keys(data: dict[str, np.ndarray]) -> np.ndarray:
    # Prefix seeds are currently disjoint across source corpora.  Keep the
    # source index in the key anyway so future corpus mergers cannot collide.
    source = np.asarray(
        data.get("source_dataset_index", np.zeros(len(data["prefix_seed"]))),
        dtype=np.int64,
    )
    seed = np.asarray(data["prefix_seed"], dtype=np.int64)
    seat = np.asarray(data["seat"], dtype=np.int64)
    return source * 10**10 + seed * 2 + seat


def source_name(data: dict[str, np.ndarray], index: int) -> str:
    names = data.get("source_dataset_names")
    if names is None or index < 0 or index >= len(names):
        return f"source_{index}"
    return str(names[index])


def gate_labels(data: dict[str, np.ndarray], z_value: float) -> np.ndarray:
    mean = np.asarray(data["expected_delta"], dtype=np.float64)
    std = np.asarray(data["future_std"], dtype=np.float64)
    sample_count = np.asarray(
        data.get(
            "future_sample_count",
            np.full(len(mean), data["future_delta_samples"].shape[1]),
        ),
        dtype=np.float64,
    )
    standard_error = std / np.sqrt(np.maximum(1.0, sample_count))
    return (mean - z_value * standard_error > 0.0).astype(np.int8)


def binary_metrics(
    probability: np.ndarray,
    actual: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    label = np.asarray(actual > 0.0, dtype=np.int8)
    prediction = np.asarray(probability >= threshold, dtype=np.int8)
    tp = int(np.sum((prediction == 1) & (label == 1)))
    fp = int(np.sum((prediction == 1) & (label == 0)))
    tn = int(np.sum((prediction == 0) & (label == 0)))
    fn = int(np.sum((prediction == 0) & (label == 1)))
    return {
        "rows": int(len(label)),
        "threshold": float(threshold),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "positive_rate": float(np.mean(prediction)) if len(prediction) else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def policy_metrics(
    rank_score: np.ndarray,
    gate_probability: np.ndarray,
    data: dict[str, np.ndarray],
    gate_threshold: float,
    keep_margin: float,
) -> dict[str, object]:
    actual = np.asarray(data["expected_delta"], dtype=np.float64)
    rank = np.asarray(data["candidate_rank"], dtype=np.int64)
    source = np.asarray(
        data.get("source_dataset_index", np.zeros(len(actual))), dtype=np.int64
    )
    keys = group_keys(data)
    chosen_rows: list[int] = []
    oracle_top1 = oracle_top3 = 0
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        keep = rows[rank[rows] < 0]
        if keep.size != 1:
            raise ValueError(f"group {key} has {keep.size} KEEP rows")
        scores = np.asarray(rank_score[rows], dtype=np.float64).copy()
        local_rank = rank[rows]
        allowed = (local_rank < 0) | (
            np.asarray(gate_probability[rows]) >= gate_threshold
        )
        scores[~allowed] = -np.inf
        scores[local_rank < 0] += keep_margin
        order = np.argsort(-scores, kind="stable")
        selected_local = int(order[0])
        chosen_rows.append(int(rows[selected_local]))
        actual_order = np.argsort(-actual[rows], kind="stable")
        oracle_best = int(actual_order[0])
        oracle_top1 += selected_local == oracle_best
        top3_allowed = [int(i) for i in order if np.isfinite(scores[i])][:3]
        oracle_top3 += oracle_best in top3_allowed

    chosen_rows_array = np.asarray(chosen_rows, dtype=np.int64)
    values = actual[chosen_rows_array]
    switches = rank[chosen_rows_array] >= 0

    def summarize(mask: np.ndarray) -> dict[str, float | int]:
        selected = values[mask]
        selected_switch = switches[mask]
        if not len(selected):
            return {"groups": 0}
        return {
            "groups": int(len(selected)),
            "mean_realized_delta": float(np.mean(selected)),
            "median_realized_delta": float(np.median(selected)),
            "p10_realized_delta": float(np.quantile(selected, 0.10)),
            "min_realized_delta": float(np.min(selected)),
            "positive_choice_rate": float(np.mean(selected > 0)),
            "negative_choice_rate": float(np.mean(selected < 0)),
            "switch_rate": float(np.mean(selected_switch)),
        }

    groups = len(chosen_rows_array)
    result: dict[str, object] = {
        "gate_threshold": float(gate_threshold),
        "keep_margin": float(keep_margin),
        **summarize(np.ones(groups, dtype=bool)),
        "top1_oracle_recall_with_keep": oracle_top1 / groups if groups else 0.0,
        "top3_oracle_recall_with_keep": oracle_top3 / groups if groups else 0.0,
    }
    per_source: list[dict[str, object]] = []
    chosen_source = source[chosen_rows_array]
    for index in np.unique(chosen_source):
        row = {
            "source_index": int(index),
            "source_name": source_name(data, int(index)),
        }
        row.update(summarize(chosen_source == index))
        per_source.append(row)
    result["per_source"] = per_source
    result["gate_binary"] = binary_metrics(
        gate_probability[rank >= 0], actual[rank >= 0], gate_threshold
    )
    return result


def selection_key(metrics: dict[str, object]) -> tuple[float, ...]:
    """Validation-only lexicographic key with a non-trivial safety gate."""
    switch_rate = float(metrics["switch_rate"])
    negative_rate = float(metrics["negative_choice_rate"])
    p10 = float(metrics["p10_realized_delta"])
    mean = float(metrics["mean_realized_delta"])
    top3 = float(metrics["top3_oracle_recall_with_keep"])
    # Never let always-KEEP win by appearing perfectly safe.  Prefer settings
    # that switch in at least 10% of states and meet the strong safety target.
    strong_safe = switch_rate >= 0.10 and negative_rate <= 0.10 and p10 >= 0.0
    weak_safe = switch_rate >= 0.10 and negative_rate <= 0.15 and p10 >= -500.0
    return (
        float(strong_safe),
        float(weak_safe),
        mean,
        p10,
        -negative_rate,
        top3,
        switch_rate,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranker-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument(
        "--feature-mode",
        choices=("legacy", "shop_bits", "shop_bits_scale"),
        default="legacy",
    )
    args = parser.parse_args()

    train = load(args.train)
    validation = load(args.validation)
    test = load(args.test)
    for name, data in (("validation", validation), ("test", test)):
        if not np.array_equal(train["feature_names"], data["feature_names"]):
            raise ValueError(f"train/{name} feature schema mismatch")
    train_x, expanded_names, categorical = generalized_engineered_features(
        train["features"], train["feature_names"], args.feature_mode
    )
    validation_x, _, _ = generalized_engineered_features(
        validation["features"], validation["feature_names"], args.feature_mode
    )
    test_x, _, _ = generalized_engineered_features(
        test["features"], test["feature_names"], args.feature_mode
    )
    train_y = np.asarray(train["expected_delta"], dtype=np.float64)
    weights = training_weight(train)
    candidate_train = np.asarray(train["candidate_rank"] >= 0)
    train_order, train_group_sizes = grouped_order(train)
    rankers: list[tuple[float, int, lgb.LGBMRanker]] = []
    ranker_predictions: list[tuple[np.ndarray, np.ndarray]] = []
    for minimum_gap in RANK_LABEL_MINIMUM_GAPS:
        ordinal = relevance_labels(train_y, train, minimum_gap)
        for index, variant in enumerate(RANKER_VARIANTS):
            model = lgb.LGBMRanker(
                objective="lambdarank", metric="ndcg",
                lambdarank_truncation_level=5,
                n_estimators=500, learning_rate=0.025, verbosity=-1,
                n_jobs=16,
                random_state=3400 + int(minimum_gap) + index,
                **variant,
            )
            model.fit(
                train_x[train_order], ordinal[train_order],
                group=train_group_sizes.tolist(),
                sample_weight=weights[train_order],
                categorical_feature=categorical,
            )
            rankers.append((minimum_gap, index, model))
            ranker_predictions.append(
                (model.predict(validation_x), model.predict(test_x))
            )

    gates: list[tuple[float, int, lgb.LGBMClassifier]] = []
    gate_predictions: list[tuple[np.ndarray, np.ndarray]] = []
    for z_value in LABEL_Z:
        labels = gate_labels(train, z_value)[candidate_train]
        for index, variant in enumerate(GATE_VARIANTS):
            model = lgb.LGBMClassifier(
                objective="binary", n_estimators=400, learning_rate=0.025,
                verbosity=-1, n_jobs=16,
                random_state=4400 + int(z_value * 100) + index,
                **variant,
            )
            model.fit(
                train_x[candidate_train], labels,
                sample_weight=weights[candidate_train],
                categorical_feature=categorical,
            )
            # KEEP probability is irrelevant but zero makes the policy logic
            # explicit and avoids accidental KEEP gating.
            validation_probability = model.predict_proba(validation_x)[:, 1]
            test_probability = model.predict_proba(test_x)[:, 1]
            gates.append((z_value, index, model))
            gate_predictions.append((validation_probability, test_probability))

    best_key: tuple[float, ...] | None = None
    best_spec: tuple[int, int, float, float] | None = None
    audits: list[dict[str, object]] = []
    for ranker_index, (validation_rank, _) in enumerate(ranker_predictions):
        for gate_index, (validation_gate, _) in enumerate(gate_predictions):
            z_value, gate_variant, _ = gates[gate_index]
            for gate_threshold in GATE_THRESHOLDS:
                for keep_margin in KEEP_MARGINS:
                    metrics = policy_metrics(
                        validation_rank, validation_gate, validation,
                        gate_threshold, keep_margin,
                    )
                    key = selection_key(metrics)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_spec = (
                            ranker_index, gate_index,
                            gate_threshold, keep_margin,
                        )
                    audits.append({
                        "ranker_model_index": ranker_index,
                        "ranker_variant_index": rankers[ranker_index][1],
                        "ranker_label_minimum_gap": rankers[ranker_index][0],
                        "gate_variant_index": gate_variant,
                        "gate_label_z": z_value,
                        "metrics": metrics,
                    })

    if best_spec is None:
        raise RuntimeError("no safety-gate candidate evaluated")
    ranker_index, gate_index, gate_threshold, keep_margin = best_spec
    validation_metrics = policy_metrics(
        ranker_predictions[ranker_index][0], gate_predictions[gate_index][0],
        validation, gate_threshold, keep_margin,
    )
    test_metrics = policy_metrics(
        ranker_predictions[ranker_index][1], gate_predictions[gate_index][1],
        test, gate_threshold, keep_margin,
    )
    z_value, gate_variant_index, gate_model = gates[gate_index]
    ranker_minimum_gap, ranker_variant_index, ranker_model = rankers[
        ranker_index
    ]

    args.ranker_output.parent.mkdir(parents=True, exist_ok=True)
    args.gate_output.parent.mkdir(parents=True, exist_ok=True)
    ranker_model.booster_.save_model(str(args.ranker_output))
    gate_model.booster_.save_model(str(args.gate_output))
    payload = {
        "schema": "kaggriculture.switch-two-stage-safety-gate.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "invariants": {
            "runtime_opponent_identity_feature": False,
            "validation_used_for_selection": True,
            "test_used_for_selection": False,
            "always_keep_disallowed_as_trivial_winner": True,
        },
        "train": {"path": str(args.train), "sha256": sha256(args.train),
                  "rows": int(len(train_y))},
        "validation": {"path": str(args.validation),
                       "sha256": sha256(args.validation),
                       "rows": int(len(validation["expected_delta"]))},
        "test": {"path": str(args.test), "sha256": sha256(args.test),
                 "rows": int(len(test["expected_delta"]))},
        "raw_feature_dim": int(train["features"].shape[1]),
        "expanded_feature_dim": int(train_x.shape[1]),
        "feature_mode": args.feature_mode,
        "selected": {
            "ranker_model_index": ranker_index,
            "ranker_variant_index": ranker_variant_index,
            "ranker_label_minimum_gap": ranker_minimum_gap,
            "ranker_parameters": RANKER_VARIANTS[ranker_variant_index],
            "gate_label_z": z_value,
            "gate_variant_index": gate_variant_index,
            "gate_parameters": GATE_VARIANTS[gate_variant_index],
            "gate_threshold": gate_threshold,
            "keep_margin": keep_margin,
            "selection_key": list(best_key or ()),
        },
        "selected_validation_metrics": validation_metrics,
        "selected_test_metrics": test_metrics,
        "validation_grid_summary": {
            "evaluated_settings": len(audits),
            "strong_safe_settings": int(sum(
                selection_key(row["metrics"])[0] > 0 for row in audits
            )),
            "top_by_selection_key": sorted(
                audits, key=lambda row: selection_key(row["metrics"]),
                reverse=True,
            )[:20],
        },
        "models": {
            "ranker": {"path": str(args.ranker_output),
                       "sha256": sha256(args.ranker_output)},
            "gate": {"path": str(args.gate_output),
                     "sha256": sha256(args.gate_output)},
        },
        "feature_importance": {
            "ranker": [
                {"feature": expanded_names[i], "importance": float(
                    ranker_model.feature_importances_[i]
                )}
                for i in np.argsort(-ranker_model.feature_importances_)[:40]
            ],
            "gate": [
                {"feature": expanded_names[i], "importance": float(
                    gate_model.feature_importances_[i]
                )}
                for i in np.argsort(-gate_model.feature_importances_)[:40]
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected": payload["selected"],
        "validation": validation_metrics,
        "untouched_test": test_metrics,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
