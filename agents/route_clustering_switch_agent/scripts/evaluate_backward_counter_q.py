#!/usr/bin/env python3
"""Evaluate a frozen backward route-Q selector on an untouched matrix panel."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import t as student_t

from benchmark_backward_counter_q import _evaluate
from train_backward_counter_selector import _load_features, _load_matrices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--matrices", nargs=3, type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    targets, margins, seeds = _load_matrices(args.matrices)
    if targets.tolist() != list(bundle["targets"]):
        raise ValueError("model and evaluation targets differ")
    features, _ = _load_features(args.features, seeds)
    metrics, counts = _evaluate(
        bundle["models"], features, margins, targets, bundle["opening"]
    )

    scenarios = features[0].shape[0]
    active = np.ones(scenarios, dtype=bool)
    result = np.zeros(scenarios, dtype=np.float64)
    lookup = {str(target): index for index, target in enumerate(targets)}
    for node_index, model in enumerate(bundle["models"]):
        predictions = targets[np.argmax(model.predict(features[node_index]), axis=1)]
        indices = np.asarray([lookup[str(value)] for value in predictions])
        values = margins[node_index].reshape(len(targets), -1)[
            indices, np.arange(scenarios)
        ]
        switch = active & (predictions != bundle["opening"])
        result[switch] = values[switch]
        active[switch] = False
        if node_index == 2:
            result[active] = values[active]
    paired_win = (result > 0).reshape(len(seeds), 2).mean(axis=1)
    paired_margin = result.reshape(len(seeds), 2).mean(axis=1)
    win_se = float(np.std(paired_win, ddof=1) / math.sqrt(len(seeds)))
    margin_se = float(np.std(paired_margin, ddof=1) / math.sqrt(len(seeds)))
    critical = float(student_t.ppf(0.95, len(seeds) - 1))
    metrics["paired_seed_raw_win_lower_95pct"] = float(
        np.mean(paired_win) - critical * win_se
    )
    metrics["paired_seed_margin_lower_95pct"] = float(
        np.mean(paired_margin) - critical * margin_se
    )
    metrics["paired_seed_win_standard_error"] = win_se
    metrics["paired_seed_margin_standard_error"] = margin_se
    payload = {
        "schema": "backward-counter-q-blind-evaluation-v1",
        "model": str(args.model.resolve()),
        "matrices": [str(path.resolve()) for path in args.matrices],
        "features": str(args.features.resolve()),
        "seed_count": len(seeds),
        "scenario_count": scenarios,
        "metrics": metrics,
        "prediction_counts": counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
