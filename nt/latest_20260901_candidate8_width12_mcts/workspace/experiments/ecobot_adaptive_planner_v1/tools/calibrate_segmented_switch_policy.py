#!/usr/bin/env python3
"""Calibrate one frozen pairwise ranker by a public-state segment.

The ranker remains shared across all states.  Only the conservative KEEP gate
is calibrated independently for each segment (for example, 1/2/3/4 unlocked
quadrants).  This is deliberately smaller than training one isolated model per
segment and therefore preserves cross-segment statistical strength.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from train_switch_pairwise_policy import (
    KEEP_BONUSES,
    KEEP_WIN_THRESHOLDS,
    group_keys,
    model_features,
    pair_accuracy,
    policy_metrics,
    predict_pairs,
    sha256,
)
from train_switch_safety_gate import selection_key
from train_switch_value_regressor import load


def subset_rows(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    """Keep row-shaped arrays while preserving feature/source metadata."""
    rows = len(data["expected_delta"])
    return {
        name: value[mask] if value.ndim > 0 and value.shape[0] == rows else value
        for name, value in data.items()
    }


def segment_rows(
    data: dict[str, np.ndarray], feature_index: int, value: int
) -> np.ndarray:
    raw = np.asarray(data["features"][:, feature_index], dtype=np.float64)
    return np.isclose(raw, float(value))


def assert_group_atomic(
    data: dict[str, np.ndarray], feature_index: int, label: str
) -> None:
    keys = group_keys(data)
    raw = np.asarray(data["features"][:, feature_index], dtype=np.float64)
    for key in np.unique(keys):
        values = np.unique(raw[keys == key])
        if len(values) != 1:
            raise ValueError(
                f"{label}: segment feature is not group-atomic for group {key}: "
                f"{values.tolist()}"
            )


def summarize_weighted(parts: list[dict[str, object]]) -> dict[str, object]:
    groups = sum(int(part["groups"]) for part in parts)
    if groups == 0:
        return {"groups": 0}
    weighted = lambda name: sum(
        int(part["groups"]) * float(part[name]) for part in parts
    ) / groups
    return {
        "groups": groups,
        "mean_realized_delta": weighted("mean_realized_delta"),
        "positive_choice_rate": weighted("positive_choice_rate"),
        "negative_choice_rate": weighted("negative_choice_rate"),
        "switch_rate": weighted("switch_rate"),
        "top1_oracle_recall_with_keep": weighted(
            "top1_oracle_recall_with_keep"
        ),
        "top3_oracle_recall_with_keep": weighted(
            "top3_oracle_recall_with_keep"
        ),
        # Quantiles cannot be reconstructed from per-segment summaries.  Keep
        # the conservative worst segment diagnostics explicit instead.
        "worst_segment_p10_realized_delta": min(
            float(part["p10_realized_delta"]) for part in parts
        ),
        "worst_segment_min_realized_delta": min(
            float(part["min_realized_delta"]) for part in parts
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-receipt", type=Path, required=True)
    parser.add_argument("--segment-feature", required=True)
    parser.add_argument("--segment-values", default="1,2,3,4")
    parser.add_argument("--default-threshold", type=float, required=True)
    parser.add_argument("--default-bonus", type=float, default=0.0)
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration = load(args.calibration)
    test = load(args.test)
    if not np.array_equal(calibration["feature_names"], test["feature_names"]):
        raise ValueError("calibration/test feature schema mismatch")
    names = [str(value) for value in calibration["feature_names"]]
    if args.segment_feature not in names:
        raise ValueError(f"unknown segment feature: {args.segment_feature}")
    feature_index = names.index(args.segment_feature)
    assert_group_atomic(calibration, feature_index, "calibration")
    assert_group_atomic(test, feature_index, "test")

    model = lgb.Booster(model_file=str(args.model))
    segment_values = [
        int(value) for value in args.segment_values.split(",") if value.strip()
    ]
    calibrated: list[dict[str, object]] = []
    test_parts: list[dict[str, object]] = []

    for value in segment_values:
        calibration_mask = segment_rows(calibration, feature_index, value)
        test_mask = segment_rows(test, feature_index, value)
        if np.any(calibration_mask):
            part = subset_rows(calibration, calibration_mask)
            features, _, _ = model_features(
                part["features"], part["feature_names"], args.feature_mode
            )
            prediction = predict_pairs(model, part, features)
            grid = [
                policy_metrics(part, prediction, threshold, bonus)
                for threshold in KEEP_WIN_THRESHOLDS
                for bonus in KEEP_BONUSES
            ]
            selected = max(grid, key=selection_key)
            threshold = float(selected["keep_win_threshold"])
            bonus = float(selected["keep_bonus"])
            _, _, _, difference, probability = prediction
            calibration_pairwise = [
                pair_accuracy(difference, probability, gap)
                for gap in (0.0, 500.0, 1000.0, 2000.0)
            ]
        else:
            selected = None
            threshold = args.default_threshold
            bonus = args.default_bonus
            calibration_pairwise = []
        calibrated.append({
            "segment_value": value,
            "calibration_groups": (
                int(selected["groups"]) if selected is not None else 0
            ),
            "keep_win_threshold": threshold,
            "keep_bonus": bonus,
            "calibration_pairwise": calibration_pairwise,
            "calibration_policy": selected,
            "used_default": selected is None,
        })

        if np.any(test_mask):
            part = subset_rows(test, test_mask)
            features, _, _ = model_features(
                part["features"], part["feature_names"], args.feature_mode
            )
            prediction = predict_pairs(model, part, features)
            _, _, _, difference, probability = prediction
            metrics = policy_metrics(part, prediction, threshold, bonus)
            test_parts.append({
                "segment_value": value,
                "pairwise": [
                    pair_accuracy(difference, probability, gap)
                    for gap in (0.0, 500.0, 1000.0, 2000.0)
                ],
                **metrics,
            })

    payload = {
        "schema": "kaggriculture.switch-pairwise-segmented-gate.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "invariants": {
            "shared_ranker_across_segments": True,
            "model_frozen_before_test": True,
            "test_not_used_for_threshold_selection": True,
            "segment_feature_group_atomic": True,
            "opponent_identity_not_a_segment": True,
        },
        "calibration": {
            "path": str(args.calibration), "sha256": sha256(args.calibration)
        },
        "test": {"path": str(args.test), "sha256": sha256(args.test)},
        "model": {"path": str(args.model), "sha256": sha256(args.model)},
        "model_receipt": {
            "path": str(args.model_receipt), "sha256": sha256(args.model_receipt)
        },
        "feature_mode": args.feature_mode,
        "segment_feature": args.segment_feature,
        "default_policy": {
            "keep_win_threshold": args.default_threshold,
            "keep_bonus": args.default_bonus,
        },
        "calibrated_segments": calibrated,
        "test_segments": test_parts,
        "test_weighted_summary": summarize_weighted(test_parts),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
