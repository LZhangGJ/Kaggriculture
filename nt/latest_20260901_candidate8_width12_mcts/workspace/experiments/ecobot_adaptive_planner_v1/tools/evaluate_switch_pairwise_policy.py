#!/usr/bin/env python3
"""Evaluate one frozen pairwise project-choice policy on a new corpus."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from train_switch_pairwise_policy import (
    model_features,
    pair_accuracy,
    policy_metrics,
    predict_pairs,
    sha256,
)
from train_switch_value_regressor import load


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--keep-win-threshold", type=float, required=True)
    parser.add_argument("--keep-bonus", type=float, required=True)
    parser.add_argument("--policy-receipt", type=Path, required=True)
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

    data = load(args.corpus)
    features, _, _ = model_features(
        data["features"], data["feature_names"], args.feature_mode
    )
    model = lgb.Booster(model_file=str(args.model))
    prediction = predict_pairs(model, data, features)
    _, _, _, difference, probability = prediction
    payload = {
        "schema": "kaggriculture.switch-pairwise-policy-frozen-evaluation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "invariants": {
            "model_frozen_before_corpus_evaluation": True,
            "corpus_not_used_for_model_or_threshold_selection": True,
        },
        "corpus": {"path": str(args.corpus), "sha256": sha256(args.corpus),
                   "rows": int(len(data["expected_delta"]))},
        "model": {"path": str(args.model), "sha256": sha256(args.model)},
        "feature_mode": args.feature_mode,
        "policy_receipt": {
            "path": str(args.policy_receipt), "sha256": sha256(args.policy_receipt)
        },
        "pairwise": [
            pair_accuracy(difference, probability, gap)
            for gap in (0.0, 500.0, 1000.0, 2000.0)
        ],
        "policy": policy_metrics(
            data,
            prediction,
            args.keep_win_threshold,
            args.keep_bonus,
        ),
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
