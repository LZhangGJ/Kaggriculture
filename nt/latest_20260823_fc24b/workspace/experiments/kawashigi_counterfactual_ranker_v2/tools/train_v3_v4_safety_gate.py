#!/usr/bin/env python3
"""Train a public-state safety gate that chooses between the V3 and V4 routers."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_hybrid_official_jax_lgbm import (  # noqa: E402
    OFFICIAL_WEIGHT,
    ROUTES,
    folds_by_opponent,
    folds_by_seed,
    load_contexts,
    load_jax,
    portable_predict,
)


THRESHOLDS = (0.0, 1000.0, 2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0)
PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 80,
    "learning_rate": 0.04,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 40,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_alpha": 20.0,
    "reg_lambda": 40.0,
    "random_state": 20260816,
    "n_jobs": 1,
    "verbosity": -1,
}


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def choose(payload: dict[str, Any], x: np.ndarray) -> np.ndarray:
    scores = np.stack(
        [portable_predict(payload["models"][route], x) for route in ROUTES],
        axis=1,
    )
    return np.argmax(scores, axis=1).astype(np.int16)


def fit(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(x, y, sample_weight=weight)
    return model


def metric(
    contexts: list[dict[str, Any]],
    baseline: np.ndarray,
    challenger: np.ndarray,
    gate_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    use_challenger = gate_score >= threshold
    chosen = np.where(use_challenger, challenger, baseline).astype(np.int16)
    wins, margins = [], []
    by_opponent: dict[str, dict[str, list[float]]] = {}
    for index, context in enumerate(contexts):
        outcome = context["outcomes"][ROUTES[int(chosen[index])]]
        win = int(bool(outcome["win"]))
        margin = float(outcome["margin"])
        wins.append(win)
        margins.append(margin)
        bucket = by_opponent.setdefault(str(context["opponent"]), {"wins": [], "margins": []})
        bucket["wins"].append(win)
        bucket["margins"].append(margin)
    opponent_metrics = {
        name: {
            "games": len(row["wins"]),
            "wins": int(sum(row["wins"])),
            "win_rate": float(np.mean(row["wins"])),
            "mean_margin": float(np.mean(row["margins"])),
        }
        for name, row in by_opponent.items()
    }
    return {
        "threshold": threshold,
        "games": len(contexts),
        "wins": int(sum(wins)),
        "win_rate": float(np.mean(wins)),
        "mean_margin": float(np.mean(margins)),
        "minimum_opponent_win_rate": min(row["win_rate"] for row in opponent_metrics.values()),
        "challenger_uses": int(np.sum(use_challenger & (challenger != baseline))),
        "route_counts": dict(Counter(ROUTES[int(value)] for value in chosen)),
        "by_opponent": opponent_metrics,
    }


def oof(
    contexts: list[dict[str, Any]],
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    official_x: np.ndarray,
    official_delta: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
    jax_x: np.ndarray,
    jax_delta: np.ndarray,
    jax_baseline: np.ndarray,
    jax_challenger: np.ndarray,
    jax_row_weight: np.ndarray,
) -> list[dict[str, Any]]:
    predictions = np.full(len(contexts), np.nan, dtype=np.float64)
    jax_disagreement = jax_baseline != jax_challenger
    for train_index, valid_index in splits:
        official_train = train_index[baseline[train_index] != challenger[train_index]]
        x = np.concatenate((jax_x[jax_disagreement], official_x[official_train]), axis=0)
        y = np.concatenate((jax_delta[jax_disagreement], official_delta[official_train]), axis=0)
        weight = np.concatenate((
            jax_row_weight[jax_disagreement],
            np.full(len(official_train), OFFICIAL_WEIGHT, dtype=np.float64),
        ))
        model = fit(x, y, weight)
        predictions[valid_index] = model.predict(official_x[valid_index])
    if np.isnan(predictions).any():
        raise AssertionError("incomplete safety-gate OOF")
    return [metric(contexts, baseline, challenger, predictions, threshold) for threshold in THRESHOLDS]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-panel", type=Path, required=True)
    parser.add_argument("--jax-features", type=Path, required=True)
    parser.add_argument("--jax-outcomes", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--challenger-model", type=Path, required=True)
    parser.add_argument("--jax-weight", type=float, default=0.03)
    parser.add_argument("--hard-jax-opponent-ids", default="")
    parser.add_argument("--hard-jax-multiplier", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: resolve(value)
        for name, value in vars(args).items()
        if isinstance(value, Path)
    }
    panel = json.loads(paths["official_panel"].read_text(encoding="utf-8"))
    contexts = load_contexts(panel)
    data = load_jax(paths["jax_features"], paths["jax_outcomes"])
    baseline_payload = json.loads(paths["baseline_model"].read_text(encoding="utf-8"))
    challenger_payload = json.loads(paths["challenger_model"].read_text(encoding="utf-8"))
    feature_names = [str(value) for value in panel["feature_names"]]
    if feature_names != data["feature_names"]:
        raise ValueError("official/JAX feature mismatch")

    official_x = np.asarray([row["features"] for row in contexts], dtype=np.float64)
    official_utility = np.asarray([
        [row["outcomes"][route]["utility"] for route in ROUTES]
        for row in contexts
    ], dtype=np.float64)
    official_baseline = choose(baseline_payload, official_x)
    official_challenger = choose(challenger_payload, official_x)
    row = np.arange(len(contexts))
    official_delta = official_utility[row, official_challenger] - official_utility[row, official_baseline]

    jax_baseline = choose(baseline_payload, data["x"])
    jax_challenger = choose(challenger_payload, data["x"])
    jax_row = np.arange(len(data["x"]))
    jax_delta = data["utility"][jax_row, jax_challenger] - data["utility"][jax_row, jax_baseline]
    hard_opponent_ids = {
        int(value.strip())
        for value in args.hard_jax_opponent_ids.split(",")
        if value.strip()
    }
    if args.hard_jax_multiplier <= 0:
        raise ValueError("hard-jax-multiplier must be positive")
    jax_row_weight = np.full(len(data["x"]), args.jax_weight, dtype=np.float64)
    if hard_opponent_ids:
        hard_mask = np.isin(data["opponents"], np.asarray(sorted(hard_opponent_ids)))
        jax_row_weight[hard_mask] *= args.hard_jax_multiplier

    variants = {
        "seed": oof(
            contexts, folds_by_seed(contexts), official_x, official_delta,
            official_baseline, official_challenger,
            data["x"], jax_delta, jax_baseline, jax_challenger, jax_row_weight,
        ),
        "opponent": oof(
            contexts, folds_by_opponent(contexts), official_x, official_delta,
            official_baseline, official_challenger,
            data["x"], jax_delta, jax_baseline, jax_challenger, jax_row_weight,
        ),
    }
    baseline_metric = metric(
        contexts,
        official_baseline,
        official_challenger,
        np.full(len(contexts), -np.inf),
        0.0,
    )
    rows = []
    for seed_metric, opponent_metric in zip(variants["seed"], variants["opponent"]):
        rows.append({
            "threshold": seed_metric["threshold"],
            "worst_oof_win_rate": min(seed_metric["win_rate"], opponent_metric["win_rate"]),
            "worst_oof_minimum_opponent_win_rate": min(
                seed_metric["minimum_opponent_win_rate"],
                opponent_metric["minimum_opponent_win_rate"],
            ),
            "mean_oof_margin": (seed_metric["mean_margin"] + opponent_metric["mean_margin"]) / 2.0,
            "seed_oof_wins": seed_metric["wins"],
            "opponent_oof_wins": opponent_metric["wins"],
            "mean_challenger_uses": (seed_metric["challenger_uses"] + opponent_metric["challenger_uses"]) / 2.0,
        })
    for candidate in rows:
        candidate["safety_pass"] = bool(
            candidate["worst_oof_win_rate"] >= baseline_metric["win_rate"]
            and candidate["worst_oof_minimum_opponent_win_rate"] >= baseline_metric["minimum_opponent_win_rate"]
            and candidate["mean_oof_margin"] >= baseline_metric["mean_margin"]
        )
        candidate["nontrivial_pass"] = bool(
            candidate["safety_pass"] and candidate["mean_challenger_uses"] > 0
        )
    selected = max(rows, key=lambda item: (
        item["nontrivial_pass"],
        item["safety_pass"],
        item["worst_oof_win_rate"],
        item["worst_oof_minimum_opponent_win_rate"],
        item["mean_oof_margin"],
        -item["mean_challenger_uses"],
    ))

    jax_disagreement = jax_baseline != jax_challenger
    official_disagreement = official_baseline != official_challenger
    x = np.concatenate((data["x"][jax_disagreement], official_x[official_disagreement]), axis=0)
    y = np.concatenate((jax_delta[jax_disagreement], official_delta[official_disagreement]), axis=0)
    weight = np.concatenate((
        jax_row_weight[jax_disagreement],
        np.full(int(np.sum(official_disagreement)), OFFICIAL_WEIGHT, dtype=np.float64),
    ))
    final = fit(x, y, weight)
    dump = final.booster_.dump_model()
    probe = official_x[:32]
    parity_error = float(np.max(np.abs(final.predict(probe) - portable_predict(dump, probe))))
    model_payload = {
        "schema": "kawashigi-v3-v4-public-state-safety-gate-v1",
        "routes": list(ROUTES),
        "feature_names": feature_names,
        "threshold": selected["threshold"],
        "gate_params": PARAMS,
        "official_weight": OFFICIAL_WEIGHT,
        "jax_weight": args.jax_weight,
        "hard_jax_opponent_ids": sorted(hard_opponent_ids),
        "hard_jax_multiplier": args.hard_jax_multiplier,
        "training_scope": "baseline/challenger disagreement contexts only",
        "portable_max_abs_error": parity_error,
        "baseline": baseline_payload,
        "challenger": challenger_payload,
        "gate_model": dump,
    }
    paths["model_output"].parent.mkdir(parents=True, exist_ok=True)
    paths["model_output"].write_text(
        json.dumps(model_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    result = {
        "schema": "kawashigi-v3-v4-safety-gate-training-v1",
        "status": "OOF_NONTRIVIAL_SAFETY_PASS" if selected["nontrivial_pass"] else "OOF_NO_NONTRIVIAL_SAFETY_PASS",
        "information_boundary": [
            "actor-visible public state only",
            "no opponent identity",
            "no seed",
            "no future event at inference",
        ],
        "official_contexts": len(contexts),
        "jax_contexts": len(data["x"]),
        "official_disagreement_contexts": int(np.sum(official_disagreement)),
        "jax_disagreement_contexts": int(np.sum(jax_disagreement)),
        "baseline_metric": baseline_metric,
        "selection_rows": rows,
        "selected": selected,
        "oof": variants,
        "training_label": "utility(challenger route) - utility(baseline route)",
        "model": {
            "path": str(paths["model_output"]),
            "sha256": sha256(paths["model_output"]),
            "bytes": paths["model_output"].stat().st_size,
            "portable_max_abs_error": parity_error,
        },
        "sources": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
            if name not in {"output", "model_output"}
        },
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "baseline": baseline_metric,
        "selected": selected,
        "model": result["model"],
        "output": str(paths["output"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
