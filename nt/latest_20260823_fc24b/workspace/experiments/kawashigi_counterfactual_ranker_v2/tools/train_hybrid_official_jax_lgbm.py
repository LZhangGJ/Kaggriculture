#!/usr/bin/env python3
"""Train and OOF-validate the official+JAX KAWASHIGI route-value ranker."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_official_counterfactual_lgbm import (  # noqa: E402
    ROUTES,
    actual_policy,
    evaluate,
    folds_by_opponent,
    folds_by_seed,
    load_contexts,
)


THRESHOLDS = (0.0, 20000.0, 40000.0, 80000.0, 120000.0, 160000.0)
MODEL_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 120,
    "learning_rate": 0.035,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_alpha": 10.0,
    "reg_lambda": 30.0,
    "random_state": 20260816,
    "n_jobs": 1,
    "verbosity": -1,
}
OFFICIAL_WEIGHT = 4.0
JAX_WEIGHT = 0.25
FEATURE_SOURCE = ROOT / "experiments/gold_adaptive_rule_v2/agents/gold_imitations_current_20260816_0955_v2/rank01_team/main.py"


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("kawashigi_centroid_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def centroid_routes(features: np.ndarray) -> list[str]:
    module = load_module(FEATURE_SOURCE)
    mean = np.asarray(module._CGR_FEATURE_MEAN, dtype=np.float64)
    scale = np.asarray(module._CGR_FEATURE_SCALE, dtype=np.float64)
    normalized = (features[:, :48] - mean) / scale
    routes = list(module._CGR_CENTROIDS)
    centroids = np.asarray([module._CGR_CENTROIDS[route] for route in routes], dtype=np.float64)
    distance = np.sum((normalized[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return [routes[index] for index in np.argmin(distance, axis=1)]


def load_jax(features_path: Path, outcomes_path: Path) -> dict[str, Any]:
    with np.load(features_path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float64)
        feature_names = [str(value) for value in data["feature_names"].tolist()]
        opponents = np.asarray(data["opponent_ids"], dtype=np.int16)
        seeds = np.asarray(data["seeds"], dtype=np.int32)
        decision_step = int(data["decision_step"])
    with np.load(outcomes_path, allow_pickle=False) as data:
        candidate_ids = np.asarray(data["candidate_ids"], dtype=np.int16)
        outcome_opponents = np.asarray(data["opponent_ids"], dtype=np.int16)
        outcome_seeds = np.asarray(data["seeds"], dtype=np.int32)
        wins = np.asarray(data["wins"], dtype=bool)
        margins = np.asarray(data["margins"], dtype=np.float64)
    if decision_step != 145 or features.shape[-1] != 66:
        raise ValueError((decision_step, features.shape))
    if candidate_ids.tolist() != [0, 1, 2, 3]:
        raise ValueError(candidate_ids.tolist())
    if not np.array_equal(opponents, outcome_opponents) or not np.array_equal(seeds, outcome_seeds):
        raise ValueError("JAX feature/outcome panel mismatch")
    expected = (4, 2, 1, len(opponents), len(seeds))
    if wins.shape != expected or margins.shape != expected:
        raise ValueError((wins.shape, margins.shape, expected))
    x = features.reshape(-1, 66)
    win_matrix = np.stack([wins[index, :, 0].reshape(-1) for index in range(4)], axis=1)
    margin_matrix = np.stack([margins[index, :, 0].reshape(-1) for index in range(4)], axis=1)
    utility = win_matrix.astype(np.float64) * 200000.0 + np.clip(margin_matrix, -50000.0, 50000.0)
    flat_opponents = np.tile(np.repeat(opponents, len(seeds)), 2)
    flat_seeds = np.tile(seeds, 2 * len(opponents))
    seats = np.repeat(np.arange(2, dtype=np.int8), len(opponents) * len(seeds))
    return {
        "x": x,
        "feature_names": feature_names,
        "wins": win_matrix,
        "margins": margin_matrix,
        "utility": utility,
        "opponents": flat_opponents,
        "seeds": flat_seeds,
        "seats": seats,
        "fallback": centroid_routes(x),
    }


def fit_one(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    model.fit(x, y, sample_weight=weight)
    return model


def oof_predictions(
    contexts: list[dict[str, Any]],
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    jax_data: dict[str, Any] | None,
) -> np.ndarray:
    official_x = np.asarray([context["features"] for context in contexts], dtype=np.float64)
    predictions = np.full((len(contexts), len(ROUTES)), np.nan, dtype=np.float64)
    for train_index, valid_index in splits:
        for route_index, route in enumerate(ROUTES):
            official_y = np.asarray([context["outcomes"][route]["utility"] for context in contexts], dtype=np.float64)
            x_train = official_x[train_index]
            y_train = official_y[train_index]
            weights = np.full(len(train_index), OFFICIAL_WEIGHT, dtype=np.float64)
            if jax_data is not None:
                x_train = np.concatenate((jax_data["x"], x_train), axis=0)
                y_train = np.concatenate((jax_data["utility"][:, route_index], y_train), axis=0)
                weights = np.concatenate((np.full(len(jax_data["x"]), JAX_WEIGHT), weights))
            model = fit_one(x_train, y_train, weights)
            predictions[valid_index, route_index] = model.predict(official_x[valid_index])
    if np.isnan(predictions).any():
        raise AssertionError("incomplete OOF")
    return predictions


def compact_metric(metric: dict[str, Any]) -> dict[str, Any]:
    result = dict(metric)
    result.pop("rows", None)
    return result


def evaluate_thresholds(contexts: list[dict[str, Any]], predictions: np.ndarray) -> list[dict[str, Any]]:
    return [compact_metric(evaluate(contexts, predictions, threshold)) for threshold in THRESHOLDS]


def select_candidate(
    variants: dict[str, dict[str, list[dict[str, Any]]]],
    fallback: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for source, values in variants.items():
        for seed_metric, opponent_metric in zip(values["seed"], values["opponent"]):
            rows.append({
                "source": source,
                "threshold": seed_metric["threshold"],
                "worst_oof_win_rate": min(seed_metric["win_rate"], opponent_metric["win_rate"]),
                "worst_oof_minimum_opponent_win_rate": min(seed_metric["minimum_opponent_win_rate"], opponent_metric["minimum_opponent_win_rate"]),
                "mean_oof_margin": (seed_metric["mean_margin"] + opponent_metric["mean_margin"]) / 2.0,
                "seed_oof_wins": seed_metric["wins"],
                "opponent_oof_wins": opponent_metric["wins"],
                "mean_switches": (seed_metric["switches"] + opponent_metric["switches"]) / 2.0,
            })
    for row in rows:
        row["opponent_safety_pass"] = bool(
            row["worst_oof_minimum_opponent_win_rate"]
            >= fallback["minimum_opponent_win_rate"]
        )
    selected = max(
        rows,
        key=lambda row: (
            row["opponent_safety_pass"],
            row["worst_oof_win_rate"],
            row["worst_oof_minimum_opponent_win_rate"],
            row["mean_oof_margin"],
            -row["mean_switches"],
        ),
    )
    selected["accepted"] = bool(
        selected["opponent_safety_pass"]
        and selected["worst_oof_win_rate"] >= fallback["win_rate"] + 0.05
        and selected["mean_oof_margin"] > fallback["mean_margin"]
    )
    return selected, rows


def jax_metrics(data: dict[str, Any], chosen: np.ndarray) -> dict[str, Any]:
    row = np.arange(len(chosen))
    wins = data["wins"][row, chosen]
    margins = data["margins"][row, chosen]
    by_opponent = {}
    for opponent in sorted(set(int(value) for value in data["opponents"])):
        mask = data["opponents"] == opponent
        by_opponent[str(opponent)] = {
            "games": int(mask.sum()),
            "wins": int(wins[mask].sum()),
            "win_rate": float(wins[mask].mean()),
            "mean_margin": float(margins[mask].mean()),
        }
    return {
        "games": len(chosen),
        "wins": int(wins.sum()),
        "win_rate": float(wins.mean()),
        "mean_margin": float(margins.mean()),
        "minimum_opponent_win_rate": min(value["win_rate"] for value in by_opponent.values()),
        "route_counts": dict(Counter(ROUTES[int(index)] for index in chosen)),
        "by_opponent_id": by_opponent,
    }


def tree_value(node: dict[str, Any], values: np.ndarray) -> float:
    while "leaf_value" not in node:
        feature = int(node["split_feature"])
        node = node["left_child"] if values[feature] <= float(node["threshold"]) else node["right_child"]
    return float(node["leaf_value"])


def portable_predict(model_dump: dict[str, Any], values: np.ndarray) -> np.ndarray:
    return np.asarray([
        sum(tree_value(tree["tree_structure"], row) for tree in model_dump["tree_info"])
        for row in values
    ])


def train_final(
    contexts: list[dict[str, Any]],
    jax_data: dict[str, Any],
    source: str,
    feature_names: list[str],
    model_output: Path,
) -> dict[str, Any]:
    official_x = np.asarray([context["features"] for context in contexts], dtype=np.float64)
    dumps = {}
    max_parity_error = 0.0
    for route_index, route in enumerate(ROUTES):
        official_y = np.asarray([context["outcomes"][route]["utility"] for context in contexts], dtype=np.float64)
        x_train = official_x
        y_train = official_y
        weights = np.full(len(contexts), OFFICIAL_WEIGHT, dtype=np.float64)
        if source == "official_plus_jax":
            x_train = np.concatenate((jax_data["x"], official_x), axis=0)
            y_train = np.concatenate((jax_data["utility"][:, route_index], official_y), axis=0)
            weights = np.concatenate((np.full(len(jax_data["x"]), JAX_WEIGHT), weights))
        model = fit_one(x_train, y_train, weights)
        dump = model.booster_.dump_model()
        probe = official_x[: min(32, len(official_x))]
        max_parity_error = max(max_parity_error, float(np.max(np.abs(model.predict(probe) - portable_predict(dump, probe)))))
        dumps[route] = dump
    payload = {
        "schema": "kawashigi-hybrid-official-jax-lgbm-v1",
        "routes": list(ROUTES),
        "feature_names": feature_names,
        "training_source": source,
        "model_params": MODEL_PARAMS,
        "official_weight": OFFICIAL_WEIGHT,
        "jax_weight": JAX_WEIGHT,
        "portable_max_abs_error": max_parity_error,
        "models": dumps,
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"path": str(model_output), "sha256": sha256(model_output), "bytes": model_output.stat().st_size, "portable_max_abs_error": max_parity_error}


def main() -> int:
    global JAX_WEIGHT
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-panel", type=Path, required=True)
    parser.add_argument("--jax-features", type=Path, required=True)
    parser.add_argument("--jax-outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument(
        "--jax-weight",
        type=float,
        default=JAX_WEIGHT,
        help="Per-row weight for parity-admitted JAX contexts.",
    )
    args = parser.parse_args()
    if not 0.0 < args.jax_weight <= OFFICIAL_WEIGHT:
        raise ValueError("jax-weight must be positive and no larger than official weight")
    JAX_WEIGHT = float(args.jax_weight)
    paths = {
        name: resolve(value)
        for name, value in vars(args).items()
        if isinstance(value, Path)
    }
    panel = json.loads(paths["official_panel"].read_text(encoding="utf-8"))
    contexts = load_contexts(panel)
    jax_data = load_jax(paths["jax_features"], paths["jax_outcomes"])
    official_feature_names = [str(value) for value in panel["feature_names"]]
    if official_feature_names != jax_data["feature_names"]:
        raise ValueError("official/JAX feature-name mismatch")

    variants = {}
    for source, jax_training in (("official_only", None), ("official_plus_jax", jax_data)):
        seed_prediction = oof_predictions(contexts, folds_by_seed(contexts), jax_training)
        opponent_prediction = oof_predictions(contexts, folds_by_opponent(contexts), jax_training)
        variants[source] = {
            "seed": evaluate_thresholds(contexts, seed_prediction),
            "opponent": evaluate_thresholds(contexts, opponent_prediction),
        }
    baselines = {
        "fallback": actual_policy(contexts, "fallback"),
        "oracle": actual_policy(contexts, "oracle"),
        **{f"fixed_{route}": actual_policy(contexts, route) for route in ROUTES},
    }
    selected, selection_rows = select_candidate(variants, baselines["fallback"])
    model_receipt = train_final(
        contexts,
        jax_data,
        selected["source"],
        official_feature_names,
        paths["model_output"],
    )

    fallback_index = np.asarray([ROUTES.index(route) for route in jax_data["fallback"]], dtype=np.int16)
    oracle_index = np.argmax(jax_data["utility"], axis=1).astype(np.int16)
    fixed_metrics = {route: jax_metrics(jax_data, np.full(len(fallback_index), index, dtype=np.int16)) for index, route in enumerate(ROUTES)}
    jax_summary = {
        "truth_boundary": "training/calibration panel only; not independent evidence",
        "contexts": len(fallback_index),
        "fallback": jax_metrics(jax_data, fallback_index),
        "oracle": jax_metrics(jax_data, oracle_index),
        "fixed_routes": fixed_metrics,
    }
    result = {
        "schema": "kawashigi-hybrid-official-jax-lgbm-training-v1",
        "status": "OOF_PASS_READY_FOR_FRESH_OFFICIAL" if selected["accepted"] else "OOF_WEAK_NOT_PROMOTED",
        "official_contexts": len(contexts),
        "jax_contexts": len(jax_data["x"]),
        "feature_count": 66,
        "information_boundary": ["actor-visible public state only", "no opponent identity", "no seed", "no future events", "no terminal label at inference"],
        "weights": {"official": OFFICIAL_WEIGHT, "jax": JAX_WEIGHT},
        "model_params": MODEL_PARAMS,
        "baselines": baselines,
        "oof_variants": variants,
        "selection_rows": selection_rows,
        "selected": selected,
        "acceptance_rule": "weaker of seed-group and leave-one-opponent-out improves fallback by at least 5 percentage points, improves mean margin, and does not reduce the minimum opponent win rate",
        "jax_calibration": jax_summary,
        "model": model_receipt,
        "sources": {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items() if key not in {"output", "model_output"}},
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "official_contexts": len(contexts),
        "jax_contexts": len(jax_data["x"]),
        "fallback": baselines["fallback"],
        "oracle": baselines["oracle"],
        "selected": selected,
        "jax_fallback": jax_summary["fallback"],
        "jax_oracle": jax_summary["oracle"],
        "model": model_receipt,
        "output": str(paths["output"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
