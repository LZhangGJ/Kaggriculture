#!/usr/bin/env python3
"""Train a seed-grouped route-Q selector directly from a compact C++ search."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import t as student_t
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from meta_agent.src.route_switch_features import route_switch_feature_names


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def _csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part.strip())


def load_node(path: Path, checkpoint: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        openings = saved["openings"].astype(str)
        checkpoints = saved["checkpoints"].astype(int)
        opponents = saved["opponents"].astype(str)
        seeds = saved["seeds"].astype(np.int64)
        targets = saved["targets"].astype(str)
        if len(openings) != 1:
            raise ValueError("compact route-Q training requires one opening")
        matches = np.flatnonzero(checkpoints == checkpoint)
        if len(matches) != 1:
            raise ValueError(f"checkpoint {checkpoint} is absent or duplicated")
        ci = int(matches[0])
        features = saved["states"][0, ci].reshape(-1, saved["states"].shape[-1])
        scores = (
            saved["outcome"][0, ci].astype(np.float32).reshape(len(targets), -1).T
            * 0.5
        )
        margins = saved["margin"][0, ci].astype(np.float32).reshape(len(targets), -1).T
    groups = np.broadcast_to(
        seeds[None, :, None], (len(opponents), len(seeds), 2)
    ).reshape(-1)
    return {
        "opening": str(openings[0]), "checkpoint": int(checkpoint),
        "targets": targets, "opponents": opponents, "seeds": seeds,
        "features": np.asarray(features, dtype=np.float32),
        "scores": scores, "margins": margins, "groups": groups,
        "sample_shape": (len(opponents), len(seeds), 2),
    }


def zero_feature_prefixes(
    matrix: np.ndarray, names: list[str], prefixes: tuple[str, ...],
) -> np.ndarray:
    result = matrix.copy()
    if prefixes:
        indices = [
            index for index, name in enumerate(names)
            if any(name.startswith(prefix) for prefix in prefixes)
        ]
        result[:, indices] = 0
    return result


def selected_values(values: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return values[np.arange(len(values)), predictions]


def opponent_raw_win_rates(
    scores: np.ndarray, predictions: np.ndarray,
    sample_shape: tuple[int, int, int],
) -> np.ndarray:
    wins = selected_values(scores, predictions) == 1.0
    return wins.reshape(sample_shape).mean(axis=(1, 2))


def policy_metrics(
    scores: np.ndarray, margins: np.ndarray, predictions: np.ndarray,
    sample_shape: tuple[int, int, int],
) -> dict[str, float]:
    chosen_scores = selected_values(scores, predictions)
    chosen_margins = selected_values(margins, predictions)
    wins = chosen_scores == 1.0
    paired = wins.reshape(sample_shape)
    per_seed = paired.mean(axis=(0, 2))
    if len(per_seed) >= 2:
        standard_error = float(np.std(per_seed, ddof=1) / math.sqrt(len(per_seed)))
        lower = float(
            np.mean(per_seed)
            - student_t.ppf(0.95, len(per_seed) - 1) * standard_error
        )
    else:
        standard_error = 0.0
        lower = float(np.mean(per_seed))
    opponent_rates = opponent_raw_win_rates(scores, predictions, sample_shape)
    return {
        "raw_win_rate": float(np.mean(wins)),
        "score_rate": float(np.mean(chosen_scores)),
        "both_seats_win_rate": float(np.mean(np.all(paired, axis=2))),
        "minimum_opponent_raw_win_rate": float(np.min(opponent_rates)),
        "mean_margin": float(np.mean(chosen_margins)),
        "minimum_margin": float(np.min(chosen_margins)),
        "paired_seed_raw_win_lower_95pct": lower,
        "paired_seed_win_standard_error": standard_error,
    }


def fit_model(
    features: np.ndarray, scores: np.ndarray, margins: np.ndarray,
    *, estimators: int, leaf: int, max_features: float,
    margin_weight: float, random_state: int,
) -> ExtraTreesRegressor:
    utility = scores + margin_weight * np.tanh(margins / 10000.0)
    return ExtraTreesRegressor(
        n_estimators=estimators,
        min_samples_leaf=leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=random_state,
    ).fit(features, utility)


def trial_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    """Rank selector trials by the deployment constraint before their mean."""

    metrics = row["cross_validation"]
    return (
        -metrics["minimum_opponent_raw_win_rate"],
        -metrics["paired_seed_raw_win_lower_95pct"],
        -metrics["raw_win_rate"],
        -metrics["both_seats_win_rate"],
        -metrics["mean_margin"],
        row["min_leaf"], row["max_features"], row["margin_weight"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--estimators", type=int, default=256)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-leaves", default="1,2,4,8,16,32")
    parser.add_argument("--max-features", default="0.25,0.5,1.0")
    parser.add_argument("--margin-weights", default="0,0.02")
    parser.add_argument("--zero-feature-prefix", action="append", default=[])
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    node = load_node(args.search, args.checkpoint)
    names = route_switch_feature_names()
    if node["features"].shape[1] != len(names):
        raise ValueError("compact feature schema differs from runtime schema")
    prefixes = tuple(args.zero_feature_prefix)
    features = zero_feature_prefixes(node["features"], names, prefixes)
    scores, margins = node["scores"], node["margins"]
    opening_index = int(np.flatnonzero(node["targets"] == node["opening"])[0])
    baseline = policy_metrics(
        scores, margins, np.full(len(features), opening_index), node["sample_shape"]
    )
    splits = list(GroupKFold(args.folds).split(
        features, groups=node["groups"]
    ))
    trials = []
    for leaf in _csv_ints(args.min_leaves):
        for max_features in _csv_floats(args.max_features):
            for margin_weight in _csv_floats(args.margin_weights):
                predictions = np.empty(len(features), dtype=np.int32)
                for fold, (train, valid) in enumerate(splits):
                    model = fit_model(
                        features[train], scores[train], margins[train],
                        estimators=args.estimators, leaf=leaf,
                        max_features=max_features, margin_weight=margin_weight,
                        random_state=20260827 + fold,
                    )
                    predictions[valid] = np.argmax(model.predict(features[valid]), axis=1)
                metrics = policy_metrics(
                    scores, margins, predictions, node["sample_shape"]
                )
                trials.append({
                    "min_leaf": leaf, "max_features": max_features,
                    "margin_weight": margin_weight, "cross_validation": metrics,
                })
    trials.sort(key=trial_sort_key)
    selected = trials[0]
    model = fit_model(
        features, scores, margins, estimators=args.estimators,
        leaf=int(selected["min_leaf"]),
        max_features=float(selected["max_features"]),
        margin_weight=float(selected["margin_weight"]),
        random_state=20260827,
    )
    training_predictions = np.argmax(model.predict(features), axis=1)
    training = policy_metrics(
        scores, margins, training_predictions, node["sample_shape"]
    )
    bundle = {
        "schema": "compact-route-q-model-v1",
        "opening": node["opening"], "targets": node["targets"].tolist(),
        "checkpoints": [args.checkpoint], "models": [model],
        "feature_names": names,
        "zero_feature_prefixes": list(prefixes),
        "selected": selected,
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_model, compress=3)
    report = {
        "schema": "compact-route-q-training-v1",
        "source": str(args.search.resolve()),
        "checkpoint": args.checkpoint,
        "seed_count": len(node["seeds"]),
        "opponents": node["opponents"].tolist(),
        "targets": node["targets"].tolist(),
        "feature_ablation": {"zero_feature_prefixes": list(prefixes)},
        "selection_objective": "minimum_opponent_raw_win_rate",
        "baseline": baseline,
        "selected": selected,
        "training": training,
        "model_bytes": args.output_model.stat().st_size,
        "trials": trials,
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "model": str(args.output_model.resolve()),
        "report": str(args.output_report.resolve()),
        "baseline": baseline, "selected": selected, "training": training,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
