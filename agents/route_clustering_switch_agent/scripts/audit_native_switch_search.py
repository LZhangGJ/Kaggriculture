#!/usr/bin/env python3
"""Audit compact native switch-search tensors and best-target label balance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("search", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.search) as saved:
        outcome_codes = saved["outcome"]
        outcomes = outcome_codes.astype(np.float32) * 0.5
        margins = saved["margin"].astype(np.float32)
        states = saved["states"]
        targets = saved["targets"].astype(str)
        checkpoints = saved["checkpoints"].astype(int)
        seeds = saved["seeds"].astype(int)
        opponents = saved["opponents"].astype(str)

    values, counts = np.unique(outcome_codes, return_counts=True)
    nodes = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        node_scores = outcomes[0, checkpoint_index].reshape(len(targets), -1).T
        node_margins = margins[0, checkpoint_index].reshape(len(targets), -1).T
        quality = node_scores.astype(np.float64) * 1e12 + node_margins
        labels = np.argmax(quality, axis=1)
        target_indices, target_counts = np.unique(labels, return_counts=True)
        distribution = sorted(
            (
                {
                    "target": str(targets[index]),
                    "count": int(count),
                    "rate": float(count / len(labels)),
                }
                for index, count in zip(target_indices, target_counts)
            ),
            key=lambda row: (-row["count"], row["target"]),
        )
        ev_count = sum(
            row["count"] for row in distribution if row["target"].startswith("EV")
        )
        nodes.append(
            {
                "checkpoint": int(checkpoint),
                "samples": int(len(labels)),
                "evolved_label_count": int(ev_count),
                "evolved_label_rate": float(ev_count / len(labels)),
                "target_distribution": distribution,
            }
        )

    payload = {
        "schema_version": 1,
        "search": str(args.search),
        "outcome_shape": list(outcomes.shape),
        "margin_shape": list(margins.shape),
        "state_shape": list(states.shape),
        "seeds": [int(value) for value in seeds],
        "opponent_count": int(len(opponents)),
        "target_count": int(len(targets)),
        "outcome_code_counts": {
            str(int(value)): int(count) for value, count in zip(values, counts)
        },
        "nodes": nodes,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
