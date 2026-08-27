#!/usr/bin/env python3
"""Evaluate an exported route-Q policy on saved counterfactual matrices."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "src"))

from meta_agent.src.search_route_policy import QuantizedExtraTreesRouteQ


def _load_margins(paths: list[Path]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    targets = None
    seeds = None
    margins = []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            current_targets = saved["targets"].astype(str)
            current_seeds = saved["seeds"].astype(np.int64)
            current_margins = saved["margins"].astype(np.float64)
        if targets is None:
            targets, seeds = current_targets, current_seeds
        elif not np.array_equal(targets, current_targets) or not np.array_equal(
            seeds, current_seeds
        ):
            raise ValueError("counterfactual matrix panels are not aligned")
        margins.append(current_margins)
    assert targets is not None and seeds is not None
    return targets, margins, seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--matrices", nargs=3, type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    model = QuantizedExtraTreesRouteQ(policy)
    targets, margins, seeds = _load_margins(args.matrices)
    if targets.tolist() != list(model.targets):
        raise ValueError("policy and evaluation targets differ")
    checkpoints = [int(value) for value in policy["checkpoints"]]
    with np.load(args.features, allow_pickle=False) as saved:
        features = [
            np.ascontiguousarray(
                saved[f"features_{checkpoint}"].reshape(-1, len(saved["feature_names"])),
                dtype=np.float32,
            )
            for checkpoint in checkpoints
        ]

    opening = str(policy["openings"][0])
    lookup = {str(target): index for index, target in enumerate(targets)}
    scenarios = features[0].shape[0]
    active = np.ones(scenarios, dtype=bool)
    result = np.zeros(scenarios, dtype=np.float64)
    prediction_counts = {}
    for node_index, checkpoint in enumerate(checkpoints):
        predictions = np.asarray([
            model.predict(checkpoint, vector) for vector in features[node_index]
        ])
        prediction_counts[str(checkpoint)] = dict(Counter(predictions.tolist()))
        indices = np.asarray([lookup[value] for value in predictions], dtype=np.int64)
        values = margins[node_index].reshape(len(targets), -1)[
            indices, np.arange(scenarios)
        ]
        switch = active & (predictions != opening)
        result[switch] = values[switch]
        active[switch] = False
        if node_index == len(checkpoints) - 1:
            result[active] = values[active]

    paired = result.reshape(len(seeds), 2)
    paired_win = (paired > 0).mean(axis=1)
    paired_margin = paired.mean(axis=1)
    critical = float(student_t.ppf(0.95, len(seeds) - 1))
    win_se = float(np.std(paired_win, ddof=1) / math.sqrt(len(seeds)))
    margin_se = float(np.std(paired_margin, ddof=1) / math.sqrt(len(seeds)))
    metrics = {
        "raw_win_rate": float(np.mean(result > 0)),
        "mean_margin": float(np.mean(result)),
        "minimum_margin": float(np.min(result)),
        "both_seats_win_rate": float(np.mean(np.all(paired > 0, axis=1))),
        "p10_paired_margin": float(np.quantile(paired_margin, 0.10)),
        "paired_seed_raw_win_lower_95pct": float(
            np.mean(paired_win) - critical * win_se
        ),
        "paired_seed_margin_lower_95pct": float(
            np.mean(paired_margin) - critical * margin_se
        ),
        "completion_rate": 1.0,
    }
    payload = {
        "schema": "quantized-counter-q-blind-evaluation-v1",
        "policy": str(args.policy.resolve()),
        "matrices": [str(path.resolve()) for path in args.matrices],
        "features": str(args.features.resolve()),
        "seed_count": len(seeds),
        "scenario_count": scenarios,
        "metrics": metrics,
        "prediction_counts": prediction_counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
