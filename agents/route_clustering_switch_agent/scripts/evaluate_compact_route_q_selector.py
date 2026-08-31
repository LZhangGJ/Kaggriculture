#!/usr/bin/env python3
"""Evaluate a frozen compact route-Q selector on an independent C++ matrix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from train_compact_route_q_selector import (
    load_node, opponent_raw_win_rates, policy_metrics, zero_feature_prefixes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    if bundle.get("schema") != "compact-route-q-model-v1":
        raise ValueError("unsupported compact route-Q model schema")
    checkpoint = int(bundle["checkpoints"][0])
    node = load_node(args.search, checkpoint)
    if node["opening"] != bundle["opening"]:
        raise ValueError("model and evaluation openings differ")
    if node["targets"].tolist() != list(bundle["targets"]):
        raise ValueError("model and evaluation targets differ")
    features = zero_feature_prefixes(
        node["features"], list(bundle["feature_names"]),
        tuple(bundle.get("zero_feature_prefixes", ())),
    )
    predictions = np.argmax(bundle["models"][0].predict(features), axis=1)
    opening_index = int(np.flatnonzero(node["targets"] == node["opening"])[0])
    baseline = policy_metrics(
        node["scores"], node["margins"],
        np.full(len(features), opening_index), node["sample_shape"],
    )
    selected = policy_metrics(
        node["scores"], node["margins"], predictions, node["sample_shape"],
    )
    selector_rates = opponent_raw_win_rates(
        node["scores"], predictions, node["sample_shape"]
    )
    oracle_scores = np.max(node["scores"], axis=1)
    oracle_rates = (oracle_scores == 1.0).reshape(
        node["sample_shape"]
    ).mean(axis=(1, 2))
    payload = {
        "schema": "compact-route-q-evaluation-v1",
        "model": str(args.model.resolve()),
        "source": str(args.search.resolve()),
        "checkpoint": checkpoint,
        "seed_count": len(node["seeds"]),
        "opponents": node["opponents"].tolist(),
        "baseline": baseline,
        "selector": selected,
        "opponent_raw_win_rates": dict(zip(
            node["opponents"].tolist(), selector_rates.tolist()
        )),
        "route_oracle": {
            "raw_win_rate": float(np.mean(oracle_scores == 1.0)),
            "minimum_opponent_raw_win_rate": float(np.min(oracle_rates)),
            "opponent_raw_win_rates": dict(zip(
                node["opponents"].tolist(), oracle_rates.tolist()
            )),
        },
        "prediction_counts": dict(Counter(node["targets"][predictions].tolist())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
