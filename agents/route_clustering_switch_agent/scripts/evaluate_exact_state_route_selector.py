#!/usr/bin/env python3
"""Evaluate a frozen exact-state selector, falling back on unseen states."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from train_compact_route_q_selector import load_node, policy_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.model, allow_pickle=False) as saved:
        states = saved["states"]
        actions = saved["actions"]
        targets = saved["targets"].astype(str)
        opening = str(saved["opening"])
        checkpoint = int(saved["checkpoint"])
    node = load_node(args.search, checkpoint)
    if node["opening"] != opening or node["targets"].tolist() != targets.tolist():
        raise ValueError("model and evaluation route schemas differ")
    lookup = {row.tobytes(): int(action) for row, action in zip(states, actions)}
    fallback = int(np.flatnonzero(targets == opening)[0])
    predictions = np.asarray([
        lookup.get(row.tobytes(), fallback) for row in node["features"]
    ], dtype=np.int32)
    metrics = policy_metrics(
        node["scores"], node["margins"], predictions, node["sample_shape"]
    )
    payload = {
        "schema": "exact-public-state-route-selector-evaluation-v1",
        "model": str(args.model.resolve()),
        "source": str(args.search.resolve()),
        "checkpoint": checkpoint,
        "seed_count": len(node["seeds"]),
        "opponents": node["opponents"].tolist(),
        "selector": metrics,
        "seen_state_rate": float(np.mean([
            row.tobytes() in lookup for row in node["features"]
        ])),
        "prediction_counts": dict(Counter(targets[predictions].tolist())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
