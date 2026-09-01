#!/usr/bin/env python3
"""Evaluate one frozen SWITCH safety policy on a never-used corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from train_switch_lambdarank import ranking_metrics
from train_switch_safety_gate import policy_metrics
from train_switch_value_regressor import generalized_engineered_features, load


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--gate-threshold", type=float, required=True)
    parser.add_argument("--keep-margin", type=float, required=True)
    parser.add_argument("--policy-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-mode",
        choices=("legacy", "shop_bits", "shop_bits_scale"),
        default="legacy",
    )
    args = parser.parse_args()

    data = load(args.corpus)
    x, expanded_names, _ = generalized_engineered_features(
        data["features"], data["feature_names"], args.feature_mode
    )
    ranker = lgb.Booster(model_file=str(args.ranker))
    gate = lgb.Booster(model_file=str(args.gate))
    rank_score = ranker.predict(x)
    gate_probability = gate.predict(x)
    actual = np.asarray(data["expected_delta"], dtype=np.float64)
    safety = policy_metrics(
        rank_score, gate_probability, data,
        args.gate_threshold, args.keep_margin,
    )
    raw_ranking = ranking_metrics(
        rank_score, actual, data, args.keep_margin
    )
    payload = {
        "schema": "kaggriculture.switch-safety-policy-frozen-evaluation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_contract": {
            "policy_and_thresholds_frozen_before_corpus_open": True,
            "corpus_used_for_model_or_threshold_selection": False,
            "opponent_identity_runtime_feature": False,
        },
        "corpus": {"path": str(args.corpus), "sha256": sha256(args.corpus),
                   "rows": int(len(actual))},
        "ranker": {"path": str(args.ranker), "sha256": sha256(args.ranker)},
        "gate": {"path": str(args.gate), "sha256": sha256(args.gate)},
        "policy_receipt": (
            None if args.policy_receipt is None else {
                "path": str(args.policy_receipt),
                "sha256": sha256(args.policy_receipt),
            }
        ),
        "gate_threshold": args.gate_threshold,
        "keep_margin": args.keep_margin,
        "raw_feature_dim": int(data["features"].shape[1]),
        "expanded_feature_dim": int(x.shape[1]),
        "feature_mode": args.feature_mode,
        "expanded_feature_names_sha256": hashlib.sha256(
            "\n".join(expanded_names).encode("utf-8")
        ).hexdigest(),
        "safety_policy_metrics": safety,
        "raw_ranker_metrics": raw_ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "safety_policy": safety,
        "raw_ranker": raw_ranking,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
