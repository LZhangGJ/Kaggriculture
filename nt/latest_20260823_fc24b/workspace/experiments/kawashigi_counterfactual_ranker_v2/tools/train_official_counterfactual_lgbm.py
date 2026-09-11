#!/usr/bin/env python3
"""Train route-specific LightGBM value models from official counterfactuals."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


ROOT = Path(__file__).resolve().parents[3]
ROUTES = ("10C-4S-75L", "8C-6S-75L", "6C-12S-100L", "6C-8S-75L")
THRESHOLDS = (0.0, 2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0)
MODEL_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 80,
    "learning_rate": 0.04,
    "num_leaves": 5,
    "max_depth": 3,
    "min_child_samples": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "reg_alpha": 10.0,
    "reg_lambda": 25.0,
    "random_state": 20260816,
    "n_jobs": 1,
    "verbosity": -1,
}


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    return parser.parse_args()


def load_contexts(panel: dict[str, Any]) -> list[dict[str, Any]]:
    if not panel.get("all_done") or not panel.get("same_state_pass"):
        raise ValueError("panel did not pass official completion and same-state gates")
    raw_rows = panel["rows"]
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in raw_rows:
        key = (str(row["opponent"]), int(row["seed"]), int(row["candidate_seat"]))
        grouped[key][str(row["candidate"])] = row
    contexts: list[dict[str, Any]] = []
    for (opponent, seed, seat), by_route in sorted(grouped.items()):
        if tuple(sorted(by_route)) != tuple(sorted(ROUTES)):
            raise ValueError((opponent, seed, seat, sorted(by_route)))
        feature_sets = {tuple(float(value) for value in by_route[route]["features"]) for route in ROUTES}
        if len(feature_sets) != 1:
            raise ValueError(f"counterfactual state mismatch: {(opponent, seed, seat)}")
        fallback_routes = {str(by_route[route]["fallback_route"]) for route in ROUTES}
        if len(fallback_routes) != 1:
            raise ValueError(f"fallback mismatch: {(opponent, seed, seat)}")
        contexts.append({
            "context_id": f"{opponent}|{seed}|{seat}",
            "opponent": opponent,
            "seed": seed,
            "seat": seat,
            "features": list(feature_sets.pop()),
            "fallback_route": fallback_routes.pop(),
            "outcomes": {
                route: {
                    "utility": float(by_route[route]["utility"]),
                    "win": bool(by_route[route]["win"]),
                    "margin": float(by_route[route]["margin"]),
                }
                for route in ROUTES
            },
        })
    return contexts


def folds_by_seed(contexts: list[dict[str, Any]]) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([context["seed"] for context in contexts])
    return GroupKFold(n_splits=min(4, len(set(groups)))).split(np.zeros(len(contexts)), groups=groups)


def folds_by_opponent(contexts: list[dict[str, Any]]) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([context["opponent"] for context in contexts])
    return LeaveOneGroupOut().split(np.zeros(len(contexts)), groups=groups)


def cross_validated_predictions(
    contexts: list[dict[str, Any]],
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    x = np.asarray([context["features"] for context in contexts], dtype=np.float64)
    predictions = np.full((len(contexts), len(ROUTES)), np.nan, dtype=np.float64)
    for train_index, valid_index in splits:
        for route_index, route in enumerate(ROUTES):
            y = np.asarray([context["outcomes"][route]["utility"] for context in contexts], dtype=np.float64)
            model = lgb.LGBMRegressor(**MODEL_PARAMS)
            model.fit(x[train_index], y[train_index])
            predictions[valid_index, route_index] = model.predict(x[valid_index])
    if np.isnan(predictions).any():
        raise AssertionError("incomplete OOF predictions")
    return predictions


def evaluate(
    contexts: list[dict[str, Any]],
    predictions: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    chosen_rows: list[dict[str, Any]] = []
    for index, context in enumerate(contexts):
        fallback = str(context["fallback_route"])
        fallback_index = ROUTES.index(fallback)
        best_index = int(np.argmax(predictions[index]))
        gain = float(predictions[index, best_index] - predictions[index, fallback_index])
        selected = ROUTES[best_index] if gain >= threshold else fallback
        outcome = context["outcomes"][selected]
        chosen_rows.append({
            "opponent": context["opponent"],
            "seed": context["seed"],
            "seat": context["seat"],
            "fallback": fallback,
            "selected": selected,
            "switched": selected != fallback,
            "predicted_gain": gain,
            "win": bool(outcome["win"]),
            "margin": float(outcome["margin"]),
        })
    by_opponent = {}
    for opponent in sorted({row["opponent"] for row in chosen_rows}):
        group = [row for row in chosen_rows if row["opponent"] == opponent]
        by_opponent[opponent] = {
            "games": len(group),
            "wins": sum(row["win"] for row in group),
            "win_rate": sum(row["win"] for row in group) / len(group),
            "mean_margin": sum(row["margin"] for row in group) / len(group),
        }
    wins = sum(row["win"] for row in chosen_rows)
    return {
        "threshold": threshold,
        "games": len(chosen_rows),
        "wins": wins,
        "win_rate": wins / len(chosen_rows),
        "mean_margin": sum(row["margin"] for row in chosen_rows) / len(chosen_rows),
        "minimum_opponent_win_rate": min(value["win_rate"] for value in by_opponent.values()),
        "switches": sum(row["switched"] for row in chosen_rows),
        "selected_route_counts": dict(Counter(row["selected"] for row in chosen_rows)),
        "by_opponent": by_opponent,
        "rows": chosen_rows,
    }


def actual_policy(contexts: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    rows = []
    for context in contexts:
        if policy == "fallback":
            route = context["fallback_route"]
        elif policy == "oracle":
            route = max(ROUTES, key=lambda name: (context["outcomes"][name]["utility"], context["outcomes"][name]["margin"]))
        else:
            route = policy
        outcome = context["outcomes"][route]
        rows.append({"opponent": context["opponent"], "route": route, **outcome})
    by_opponent = {}
    for opponent in sorted({row["opponent"] for row in rows}):
        group = [row for row in rows if row["opponent"] == opponent]
        by_opponent[opponent] = {
            "win_rate": sum(row["win"] for row in group) / len(group),
            "mean_margin": sum(row["margin"] for row in group) / len(group),
        }
    wins = sum(row["win"] for row in rows)
    return {
        "policy": policy,
        "games": len(rows),
        "wins": wins,
        "win_rate": wins / len(rows),
        "mean_margin": sum(row["margin"] for row in rows) / len(rows),
        "minimum_opponent_win_rate": min(value["win_rate"] for value in by_opponent.values()),
        "route_counts": dict(Counter(row["route"] for row in rows)),
        "by_opponent": by_opponent,
    }


def dump_models(contexts: list[dict[str, Any]], model_output: Path) -> dict[str, Any]:
    x = np.asarray([context["features"] for context in contexts], dtype=np.float64)
    models = {}
    for route in ROUTES:
        y = np.asarray([context["outcomes"][route]["utility"] for context in contexts], dtype=np.float64)
        model = lgb.LGBMRegressor(**MODEL_PARAMS)
        model.fit(x, y)
        models[route] = model.booster_.dump_model()
    payload = {
        "schema": "kawashigi-official-counterfactual-lgbm-model-v1",
        "routes": list(ROUTES),
        "feature_names": list(contexts[0].keys()) if False else None,
        "model_params": MODEL_PARAMS,
        "models": models,
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"path": str(model_output), "sha256": sha256(model_output), "bytes": model_output.stat().st_size}


def main() -> int:
    args = parse_args()
    panel_path = resolve(args.panel)
    output = resolve(args.output)
    model_output = resolve(args.model_output)
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    contexts = load_contexts(panel)
    seed_predictions = cross_validated_predictions(contexts, folds_by_seed(contexts))
    opponent_predictions = cross_validated_predictions(contexts, folds_by_opponent(contexts))
    seed_metrics = [evaluate(contexts, seed_predictions, threshold) for threshold in THRESHOLDS]
    opponent_metrics = [evaluate(contexts, opponent_predictions, threshold) for threshold in THRESHOLDS]
    baselines = {
        "fallback": actual_policy(contexts, "fallback"),
        "oracle": actual_policy(contexts, "oracle"),
        **{f"fixed_{route}": actual_policy(contexts, route) for route in ROUTES},
    }
    # Select one conservative threshold on the weaker of the two OOF regimes.
    candidates = []
    for seed_metric, opponent_metric in zip(seed_metrics, opponent_metrics):
        candidates.append({
            "threshold": seed_metric["threshold"],
            "worst_oof_win_rate": min(seed_metric["win_rate"], opponent_metric["win_rate"]),
            "worst_oof_minimum_opponent_win_rate": min(seed_metric["minimum_opponent_win_rate"], opponent_metric["minimum_opponent_win_rate"]),
            "mean_oof_margin": (seed_metric["mean_margin"] + opponent_metric["mean_margin"]) / 2.0,
            "mean_switches": (seed_metric["switches"] + opponent_metric["switches"]) / 2.0,
        })
    selected = max(
        candidates,
        key=lambda row: (
            row["worst_oof_win_rate"],
            row["worst_oof_minimum_opponent_win_rate"],
            row["mean_oof_margin"],
            -row["mean_switches"],
        ),
    )
    model_receipt = dump_models(contexts, model_output)
    fallback_rate = baselines["fallback"]["win_rate"]
    fallback_margin = baselines["fallback"]["mean_margin"]
    acceptance = bool(
        selected["worst_oof_win_rate"] >= fallback_rate + 0.05
        and selected["mean_oof_margin"] > fallback_margin
    )
    result = {
        "schema": "kawashigi-official-counterfactual-lgbm-training-v1",
        "status": "OOF_PASS" if acceptance else "OOF_WEAK_NOT_PROMOTED",
        "panel": str(panel_path),
        "panel_sha256": sha256(panel_path),
        "context_count": len(contexts),
        "row_count": len(contexts) * len(ROUTES),
        "feature_count": len(contexts[0]["features"]),
        "information_boundary": ["actor-visible public state only", "no opponent identity", "no seed", "no future events", "no terminal label at inference"],
        "model_params": MODEL_PARAMS,
        "thresholds": list(THRESHOLDS),
        "baselines": baselines,
        "seed_group_oof": seed_metrics,
        "leave_one_opponent_out": opponent_metrics,
        "threshold_selection": candidates,
        "selected_threshold": selected,
        "acceptance_rule": "the weaker of seed-group OOF and leave-one-opponent-out must improve frozen fallback by at least 5 percentage points and improve mean margin",
        "accepted": acceptance,
        "model": model_receipt,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "contexts": len(contexts),
        "fallback": baselines["fallback"],
        "best_fixed": max((value for key, value in baselines.items() if key.startswith("fixed_")), key=lambda row: (row["win_rate"], row["mean_margin"])),
        "oracle": baselines["oracle"],
        "selected_threshold": selected,
        "seed_oof": next(row for row in seed_metrics if row["threshold"] == selected["threshold"]),
        "opponent_oof": next(row for row in opponent_metrics if row["threshold"] == selected["threshold"]),
        "model": model_receipt,
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
