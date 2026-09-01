#!/usr/bin/env python3
"""Audit SWITCH value ranking while holding out an entire opponent route.

The ordinary train/validation split uses new seeds but the same frozen opponent
routes.  That is useful for stochastic generalisation, but it does not prove
that the planner can react to an unseen opponent.  This script therefore:

1. removes one opponent from the noisy training corpus;
2. tunes model/KEEP threshold only on the remaining opponents' high-precision
   validation states;
3. evaluates once on the held-out opponent's high-precision states;
4. repeats for every opponent and reports pooled out-of-fold performance.

Opponent identity is used only to construct the audit split.  It is never part
of the model input, which remains restricted to public state and generic
candidate-plan features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from train_switch_value_regressor import (
    engineered_features,
    load,
    metrics,
    training_weight,
)


VARIANTS = [
    {
        "objective": "regression_l1", "num_leaves": 15,
        "max_depth": 6, "min_child_samples": 15,
        "reg_lambda": 5.0, "reg_alpha": 1.0,
    },
    {
        "objective": "regression_l1", "num_leaves": 31,
        "max_depth": 7, "min_child_samples": 20,
        "reg_lambda": 10.0, "reg_alpha": 2.0,
    },
    {
        "objective": "regression", "num_leaves": 15,
        "max_depth": 6, "min_child_samples": 20,
        "reg_lambda": 10.0, "reg_alpha": 2.0,
    },
    {
        "objective": "regression", "num_leaves": 31,
        "max_depth": 8, "min_child_samples": 20,
        "reg_lambda": 15.0, "reg_alpha": 3.0,
    },
    {
        "objective": "regression", "num_leaves": 63,
        "max_depth": 9, "min_child_samples": 12,
        "reg_lambda": 5.0, "reg_alpha": 1.0,
    },
    {
        "objective": "huber", "num_leaves": 31,
        "max_depth": 8, "min_child_samples": 20,
        "reg_lambda": 5.0, "reg_alpha": 1.0,
    },
]
THRESHOLDS = [0.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0]
PROJECT_NAMES = np.asarray([
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "GOOSE", "COW", "SHEEP",
])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def opponent_tokens(data: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    names = [str(value) for value in data["source_dataset_names"]]
    tokens: list[str] = []
    for name in names:
        match = re.search(r"(?:^|_)G(\d+)(?:_|\.)", name, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"cannot extract opponent token from {name!r}")
        tokens.append(f"G{int(match.group(1)):03d}")
    row_tokens = np.asarray(tokens, dtype="U16")[
        np.asarray(data["source_dataset_index"], dtype=np.int64)
    ]
    return row_tokens, tokens


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    categorical: list[int],
    variant: dict[str, object],
    seed: int,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.025,
        verbosity=-1,
        n_jobs=16,
        random_state=seed,
        **variant,
    )
    model.fit(x, y, sample_weight=weights, categorical_feature=categorical)
    return model


def select_variant_and_threshold(
    train_x: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    categorical: list[int],
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    calibration: dict[str, np.ndarray],
    fold_index: int,
) -> tuple[lgb.LGBMRegressor, int, float, dict[str, object]]:
    best_key: tuple[float, float, float, float] | None = None
    best: tuple[lgb.LGBMRegressor, int, float, dict[str, object]] | None = None
    for variant_index, variant in enumerate(VARIANTS):
        model = fit_model(
            train_x,
            train_y,
            weights,
            categorical,
            variant,
            seed=1700 + fold_index * 20 + variant_index,
        )
        prediction = model.predict(calibration_x)
        for threshold in THRESHOLDS:
            audit = metrics(
                prediction,
                calibration_y,
                calibration["prefix_seed"],
                calibration["seat"],
                calibration["candidate_rank"],
                threshold,
            )
            # Selection uses only non-held-out opponents.  Mean value is the
            # primary goal, followed by tail safety and false-switch control.
            key = (
                float(audit["mean_realized_delta"]),
                float(audit["p10_realized_delta"]),
                -float(audit["negative_choice_rate"]),
                float(audit["pairwise_gap2000_accuracy"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = (model, variant_index, threshold, audit)
    assert best is not None
    return best


def chosen_values(
    prediction: np.ndarray,
    actual: np.ndarray,
    seed: np.ndarray,
    seat: np.ndarray,
    candidate_rank: np.ndarray,
    threshold: float,
) -> np.ndarray:
    group = seed.astype(np.int64) * 2 + seat.astype(np.int64)
    selected: list[float] = []
    for key in np.unique(group):
        rows = np.flatnonzero(group == key)
        scores = np.asarray(prediction[rows], dtype=np.float64).copy()
        ranks = np.asarray(candidate_rank[rows], dtype=np.int64)
        keep_rows = np.flatnonzero(ranks < 0)
        if keep_rows.size != 1:
            raise ValueError(
                f"group {key} must contain exactly one KEEP row, got {keep_rows.size}"
            )
        scores[keep_rows[0]] = threshold
        selected.append(float(actual[rows][int(np.argmax(scores))]))
    return np.asarray(selected, dtype=np.float64)


def family_diagnostics(
    features: np.ndarray,
    feature_names: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, object]]:
    at = {str(name): index for index, name in enumerate(feature_names)}
    source = features[:, at["source_project"]].astype(np.int64)
    destination = features[:, at["destination_project"]].astype(np.int64)
    families = source * 16 + destination
    rows: list[dict[str, object]] = []
    for family in np.unique(families):
        selected = families == family
        source_id = int(source[selected][0])
        destination_id = int(destination[selected][0])
        if not (0 <= source_id < len(PROJECT_NAMES)):
            source_name = "NONE"
        else:
            source_name = str(PROJECT_NAMES[source_id])
        if not (0 <= destination_id < len(PROJECT_NAMES)):
            destination_name = "NONE"
        else:
            destination_name = str(PROJECT_NAMES[destination_id])
        error = prediction[selected] - actual[selected]
        rows.append({
            "family": f"{source_name}->{destination_name}",
            "rows": int(np.sum(selected)),
            "actual_mean": float(np.mean(actual[selected])),
            "prediction_mean": float(np.mean(prediction[selected])),
            "mae": float(np.mean(np.abs(error))),
            "sign_accuracy": float(np.mean(
                np.sign(prediction[selected]) == np.sign(actual[selected])
            )),
            "positive_rate": float(np.mean(actual[selected] > 0)),
        })
    return sorted(rows, key=lambda item: (-int(item["rows"]), str(item["family"])))


def weighted_average(
    folds: list[dict[str, object]], metric: str, weight: str
) -> float:
    denominator = sum(float(fold["held_out_metrics"][weight]) for fold in folds)
    if denominator == 0:
        return 0.0
    return sum(
        float(fold["held_out_metrics"][metric])
        * float(fold["held_out_metrics"][weight])
        for fold in folds
    ) / denominator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    train = load(args.train)
    validation = load(args.validation)
    if not np.array_equal(train["feature_names"], validation["feature_names"]):
        raise ValueError("train/validation feature schema mismatch")
    train_row_opponent, train_opponents = opponent_tokens(train)
    validation_row_opponent, validation_opponents = opponent_tokens(validation)
    opponents = sorted(set(train_opponents) & set(validation_opponents))
    if len(opponents) < 3:
        raise ValueError("at least three matched opponents are required")

    train_x, expanded_names, categorical = engineered_features(
        train["features"], train["feature_names"]
    )
    validation_x, _, _ = engineered_features(
        validation["features"], validation["feature_names"]
    )
    train_y = np.clip(
        np.asarray(train["expected_delta"], dtype=np.float64), -30000.0, 30000.0
    )
    validation_y = np.asarray(validation["expected_delta"], dtype=np.float64)
    all_weights = training_weight(train)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    folds: list[dict[str, object]] = []
    oof_prediction = np.full(len(validation_y), np.nan, dtype=np.float64)
    oof_threshold = np.full(len(validation_y), np.nan, dtype=np.float64)
    for fold_index, opponent in enumerate(opponents):
        train_mask = train_row_opponent != opponent
        calibration_mask = validation_row_opponent != opponent
        held_mask = validation_row_opponent == opponent
        calibration = {
            key: np.asarray(value)[calibration_mask]
            for key, value in validation.items()
            if np.asarray(value).ndim > 0 and len(np.asarray(value)) == len(validation_y)
        }
        model, variant_index, threshold, calibration_audit = (
            select_variant_and_threshold(
                train_x[train_mask],
                train_y[train_mask],
                all_weights[train_mask],
                categorical,
                validation_x[calibration_mask],
                validation_y[calibration_mask],
                calibration,
                fold_index,
            )
        )
        held_prediction = model.predict(validation_x[held_mask])
        held_audit = metrics(
            held_prediction,
            validation_y[held_mask],
            validation["prefix_seed"][held_mask],
            validation["seat"][held_mask],
            validation["candidate_rank"][held_mask],
            threshold,
        )
        oof_prediction[held_mask] = held_prediction
        oof_threshold[held_mask] = threshold
        model_path = args.model_dir / f"switch_value_loo_{opponent}.txt"
        model.booster_.save_model(str(model_path))
        folds.append({
            "held_out_opponent": opponent,
            "train_rows": int(np.sum(train_mask)),
            "calibration_rows": int(np.sum(calibration_mask)),
            "held_out_rows": int(np.sum(held_mask)),
            "variant_index": variant_index,
            "parameters": VARIANTS[variant_index],
            "selected_threshold": threshold,
            "calibration_metrics": calibration_audit,
            "held_out_metrics": held_audit,
            "model": {"path": str(model_path), "sha256": sha256(model_path)},
        })

    if np.any(~np.isfinite(oof_prediction)):
        raise RuntimeError("some validation rows did not receive an OOF prediction")
    chosen = np.concatenate([
        chosen_values(
            oof_prediction[validation_row_opponent == opponent],
            validation_y[validation_row_opponent == opponent],
            validation["prefix_seed"][validation_row_opponent == opponent],
            validation["seat"][validation_row_opponent == opponent],
            validation["candidate_rank"][validation_row_opponent == opponent],
            float(np.unique(oof_threshold[validation_row_opponent == opponent])[0]),
        )
        for opponent in opponents
    ])
    pooled = {
        "groups": int(sum(int(fold["held_out_metrics"]["groups"]) for fold in folds)),
        "pairwise_accuracy": weighted_average(
            folds, "pairwise_accuracy", "pairwise_comparisons"
        ),
        "pairwise_gap2000_accuracy": weighted_average(
            folds, "pairwise_gap2000_accuracy", "pairwise_gap2000_comparisons"
        ),
        "top1_oracle_recall_with_keep": weighted_average(
            folds, "top1_oracle_recall_with_keep", "groups"
        ),
        "top3_oracle_recall_with_keep": weighted_average(
            folds, "top3_oracle_recall_with_keep", "groups"
        ),
        "mean_realized_delta": float(np.mean(chosen)),
        "median_realized_delta": float(np.median(chosen)),
        "p10_realized_delta": float(np.quantile(chosen, 0.10)),
        "positive_choice_rate": float(np.mean(chosen > 0)),
        "negative_choice_rate": float(np.mean(chosen < 0)),
    }
    payload = {
        "schema": "kaggriculture.switch-value-leave-one-opponent-out.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_contract": {
            "split_key_used_only_for_audit": "frozen opponent route token",
            "opponent_identity_in_model_features": False,
            "variant_and_threshold_selected_without_held_out_opponent": True,
        },
        "train": {"path": str(args.train), "sha256": sha256(args.train)},
        "validation": {
            "path": str(args.validation), "sha256": sha256(args.validation)
        },
        "opponents": opponents,
        "raw_feature_dim": int(train["features"].shape[1]),
        "expanded_feature_dim": int(train_x.shape[1]),
        "folds": folds,
        "pooled_out_of_opponent_metrics": pooled,
        "family_diagnostics": family_diagnostics(
            validation["features"],
            validation["feature_names"],
            validation_y,
            oof_prediction,
        ),
        "expanded_feature_names": expanded_names,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "opponents": opponents,
        "pooled_out_of_opponent_metrics": pooled,
        "fold_metrics": [
            {
                "held_out_opponent": fold["held_out_opponent"],
                "selected_threshold": fold["selected_threshold"],
                "metrics": fold["held_out_metrics"],
            }
            for fold in folds
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
