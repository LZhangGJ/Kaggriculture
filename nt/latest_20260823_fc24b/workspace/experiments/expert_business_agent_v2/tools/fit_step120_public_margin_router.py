#!/usr/bin/env python3
"""Fit a prefix-safe public-state terminal-margin router at step 120.

Every admitted route has an identical action prefix through step 119.  The
model therefore observes one real public state at the beginning of step 120
and predicts the terminal margin of each legal continuation.  Opponent name,
seed, seat, future events and terminal outcomes are audit metadata/labels only
and are never runtime inputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def make_model(spec: dict[str, Any], seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=spec["n_estimators"],
        learning_rate=spec["learning_rate"],
        num_leaves=spec["num_leaves"],
        max_depth=spec["max_depth"],
        min_child_samples=spec["min_child_samples"],
        reg_lambda=spec["reg_lambda"],
        colsample_bytree=spec["colsample_bytree"],
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )


def summarize(
    margins: np.ndarray,
    opponent_index: np.ndarray,
    opponent_names: list[str],
) -> dict[str, Any]:
    per_opponent = {}
    for index, name in enumerate(opponent_names):
        values = margins[opponent_index == index]
        per_opponent[name] = {
            "games": int(values.size),
            "wins": int(np.sum(values > 0)),
            "ties": int(np.sum(values == 0)),
            "win_rate": float(np.mean(values > 0)),
            "mean_margin": float(np.mean(values)),
        }
    return {
        "games": int(margins.size),
        "wins": int(np.sum(margins > 0)),
        "ties": int(np.sum(margins == 0)),
        "win_rate": float(np.mean(margins > 0)),
        "mean_margin": float(np.mean(margins)),
        "minimum_opponent_win_rate": float(
            min(row["win_rate"] for row in per_opponent.values())
        ),
        "opponents_below_90pct": sorted(
            name for name, row in per_opponent.items() if row["win_rate"] < 0.90
        ),
        "per_opponent": per_opponent,
    }


def score_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["minimum_opponent_win_rate"]),
        float(row["win_rate"]),
        float(row["mean_margin"]),
    )


def select_with_threshold(
    predictions: np.ndarray,
    margins: np.ndarray,
    baseline_index: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    best = np.argmax(predictions, axis=0)
    columns = np.arange(predictions.shape[1])
    improvement = predictions[best, columns] - predictions[baseline_index]
    selected = np.where(improvement > threshold, best, baseline_index)
    return margins[selected, columns], selected


def route_frequency(selected: np.ndarray, route_names: list[str]) -> list[dict[str, Any]]:
    rows = [
        {
            "route_index": index,
            "route": route,
            "games": int(np.sum(selected == index)),
            "fraction": float(np.mean(selected == index)),
        }
        for index, route in enumerate(route_names)
        if np.any(selected == index)
    ]
    return sorted(rows, key=lambda row: (-row["games"], row["route"]))


def portable_tree_value(node: dict[str, Any], values: np.ndarray) -> float:
    while "leaf_value" not in node:
        feature = int(node["split_feature"])
        threshold = float(node["threshold"])
        node = node["left_child"] if float(values[feature]) <= threshold else node["right_child"]
    return float(node["leaf_value"])


def portable_predict(model_dump: dict[str, Any], values: np.ndarray) -> np.ndarray:
    return np.asarray([
        sum(
            portable_tree_value(tree["tree_structure"], row)
            for tree in model_dump["tree_info"]
        )
        for row in values
    ], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--prefix-audit", type=Path, required=True)
    parser.add_argument("--prefix-anchor-route", default="route04_rank02_10c_4s_75l")
    parser.add_argument(
        "--routes",
        default="",
        help=(
            "Optional comma-separated subset of routes from the anchor's step-120 "
            "prefix group. This supports a train-only pruning pass before blind "
            "evaluation and keeps the deployable model compact."
        ),
    )
    parser.add_argument("--opponents", required=True)
    parser.add_argument("--train-seeds", type=int, default=128)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()

    features_path = resolve(args.features)
    outcomes_path = resolve(args.outcomes)
    bank_path = resolve(args.bank_receipt)
    prefix_path = resolve(args.prefix_audit)
    model_path = resolve(args.model_output)
    receipt_path = resolve(args.receipt_output)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
    all_route_names = [str(value) for value in bank["candidate_names"]]
    all_opponent_names = [str(row["name"]) for row in bank["opponents"]]
    opponent_names = [value.strip() for value in args.opponents.split(",") if value.strip()]
    missing_opponents = sorted(set(opponent_names) - set(all_opponent_names))
    if missing_opponents:
        raise ValueError(f"unknown opponents: {missing_opponents}")
    opponent_ids = [all_opponent_names.index(name) for name in opponent_names]

    groups = prefix["groups"]["120"]
    group = next(
        (row for row in groups if args.prefix_anchor_route in row["routes"]),
        None,
    )
    if group is None:
        raise ValueError(f"anchor route not present at step 120: {args.prefix_anchor_route}")
    route_names = [str(value) for value in group["routes"]]
    requested_routes = [
        value.strip() for value in args.routes.split(",") if value.strip()
    ]
    if requested_routes:
        missing_routes = sorted(set(requested_routes) - set(route_names))
        if missing_routes:
            raise ValueError(
                f"routes outside the step-120 prefix group: {missing_routes}"
            )
        if len(requested_routes) != len(set(requested_routes)):
            raise ValueError("duplicate route in --routes")
        route_names = requested_routes
    route_ids = [all_route_names.index(name) for name in route_names]

    with np.load(features_path, allow_pickle=False) as data:
        features = np.asarray(data["features120"], dtype=np.float32)
        feature_names = [str(value) for value in data["feature_names120"]]
        seeds = np.asarray(data["seeds"], dtype=np.int32)
        feature_step = int(data["feature_step120"])
    with np.load(outcomes_path, allow_pickle=False) as data:
        all_margins = np.asarray(data["margins"], dtype=np.float32)[:, :, 0]
    if feature_step != 120 or features.shape[-1] != len(feature_names):
        raise ValueError(
            f"invalid feature payload: step={feature_step} shape={features.shape} names={len(feature_names)}"
        )
    if args.train_seeds <= 0 or args.train_seeds >= seeds.size:
        raise ValueError("train-seeds must leave a non-empty blind seed block")

    # [seat, selected opponent, seed, feature] and
    # [route, seat, selected opponent, seed].
    features = features[:, opponent_ids]
    margins = all_margins[route_ids][:, :, opponent_ids, :]
    seat_count, opponent_count, seed_count, feature_count = features.shape
    x = features.reshape(-1, feature_count)
    margins = margins.reshape(len(route_ids), -1)
    seat_grid, opponent_grid, seed_grid = np.meshgrid(
        np.arange(seat_count, dtype=np.int8),
        np.arange(opponent_count, dtype=np.int16),
        np.arange(seed_count, dtype=np.int16),
        indexing="ij",
    )
    opponent_index = opponent_grid.reshape(-1)
    seed_index = seed_grid.reshape(-1)
    event_seed = seeds[seed_index]
    train_mask = seed_index < args.train_seeds
    blind_mask = ~train_mask
    x_train = x[train_mask]
    train_groups = event_seed[train_mask]
    train_margins = margins[:, train_mask]

    # Select the fallback route only from the training partition.
    fixed_rows = []
    for route_index, route in enumerate(route_names):
        row = summarize(train_margins[route_index], opponent_index[train_mask], opponent_names)
        row.update({"route_index": route_index, "route": route})
        fixed_rows.append(row)
    fixed_rows.sort(key=score_key, reverse=True)
    baseline_index = int(fixed_rows[0]["route_index"])

    specs = [
        {
            "name": "tiny_depth2",
            "n_estimators": 32,
            "learning_rate": 0.05,
            "num_leaves": 4,
            "max_depth": 2,
            "min_child_samples": 64,
            "reg_lambda": 40.0,
            "colsample_bytree": 0.80,
        },
        {
            "name": "small_depth3",
            "n_estimators": 64,
            "learning_rate": 0.04,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 64,
            "reg_lambda": 30.0,
            "colsample_bytree": 0.85,
        },
        {
            "name": "medium_depth4",
            "n_estimators": 96,
            "learning_rate": 0.03,
            "num_leaves": 12,
            "max_depth": 4,
            "min_child_samples": 80,
            "reg_lambda": 40.0,
            "colsample_bytree": 0.85,
        },
    ]
    thresholds = (0.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0)
    splitter = GroupKFold(n_splits=4)
    cv_rows = []
    best: tuple[tuple[float, float, float], dict[str, Any], float, np.ndarray] | None = None
    for spec_index, spec in enumerate(specs):
        oof = np.zeros_like(train_margins, dtype=np.float32)
        for fold, (fit_index, validation_index) in enumerate(
            splitter.split(x_train, groups=train_groups)
        ):
            for route_index in range(len(route_ids)):
                model = make_model(spec, 22000 + spec_index * 1000 + fold * 100 + route_index)
                target = np.clip(train_margins[route_index, fit_index], -20000.0, 20000.0)
                model.fit(x_train[fit_index], target)
                oof[route_index, validation_index] = model.predict(x_train[validation_index])
        for threshold in thresholds:
            selected_margin, selected = select_with_threshold(
                oof, train_margins, baseline_index, threshold
            )
            summary = summarize(selected_margin, opponent_index[train_mask], opponent_names)
            row = {
                "model": spec["name"],
                "threshold": threshold,
                "baseline_route": route_names[baseline_index],
                "route_count_used": len(np.unique(selected)),
                **{key: value for key, value in summary.items() if key != "per_opponent"},
            }
            cv_rows.append(row)
            key = score_key(summary)
            if best is None or key > best[0]:
                best = (key, spec, threshold, oof.copy())
    assert best is not None
    _, selected_spec, selected_threshold, selected_oof = best

    model_dumps = []
    blind_predictions = np.zeros((len(route_ids), int(np.sum(blind_mask))), dtype=np.float32)
    gain = np.zeros(feature_count, dtype=np.float64)
    portable_max_error = 0.0
    probe = x[blind_mask][: min(128, int(np.sum(blind_mask)))]
    for route_index, route in enumerate(route_names):
        model = make_model(selected_spec, 26000 + route_index)
        target = np.clip(train_margins[route_index], -20000.0, 20000.0)
        model.fit(x_train, target)
        blind_predictions[route_index] = model.predict(x[blind_mask]).astype(np.float32)
        dump = model.booster_.dump_model()
        model_dumps.append({
            "route_index": route_index,
            "route": route,
            "source_candidate_id": route_ids[route_index],
            "model": dump,
        })
        gain += model.booster_.feature_importance(importance_type="gain")
        portable_max_error = max(
            portable_max_error,
            float(np.max(np.abs(portable_predict(dump, probe) - model.predict(probe)))),
        )

    def partition(mask: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
        partition_margins = margins[:, mask]
        routed_margin, selected = select_with_threshold(
            prediction, partition_margins, baseline_index, selected_threshold
        )
        result = {
            "baseline": summarize(
                partition_margins[baseline_index], opponent_index[mask], opponent_names
            ),
            "router": summarize(routed_margin, opponent_index[mask], opponent_names),
            "hindsight_oracle": summarize(
                np.max(partition_margins, axis=0), opponent_index[mask], opponent_names
            ),
            "route_frequency": route_frequency(selected, route_names),
        }
        return result

    train_result = partition(train_mask, selected_oof)
    blind_result = partition(blind_mask, blind_predictions)
    importance = sorted(
        (
            {"feature": name, "gain": float(value)}
            for name, value in zip(feature_names, gain, strict=True)
        ),
        key=lambda row: -row["gain"],
    )

    model_payload = {
        "schema": "kaggriculture-step120-public-margin-router-model-v1",
        "decision_step": 120,
        "feature_names": feature_names,
        "route_names": route_names,
        "baseline_route_index": baseline_index,
        "baseline_route": route_names[baseline_index],
        "switch_threshold": selected_threshold,
        "model_spec": selected_spec,
        "models": model_dumps,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(
        json.dumps(model_payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema": "kaggriculture-step120-public-margin-router-fit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "JAX_SCREEN_ONLY",
        "decision_step": 120,
        "common_prefix_actions": 120,
        "prefix_anchor_route": args.prefix_anchor_route,
        "route_selection": (
            "explicit_train_only_pruned_subset"
            if requested_routes
            else "all_routes_in_prefix_group"
        ),
        "route_count": len(route_names),
        "route_names": route_names,
        "selected_opponents": opponent_names,
        "forbidden_runtime_inputs": [
            "opponent_identity",
            "seed",
            "seat",
            "future_events",
            "terminal_outcome",
            "private_opponent_state",
        ],
        "target": "clipped_terminal_margin_per_route",
        "split": {
            "type": "contiguous_disjoint_event_seed_blocks",
            "train_seed_start": int(seeds[0]),
            "train_seed_end": int(seeds[args.train_seeds - 1]),
            "blind_seed_start": int(seeds[args.train_seeds]),
            "blind_seed_end": int(seeds[-1]),
            "train_seed_count": args.train_seeds,
            "blind_seed_count": int(seeds.size - args.train_seeds),
        },
        "selected_model": selected_spec,
        "selected_threshold": selected_threshold,
        "baseline_route": route_names[baseline_index],
        "cv_candidates": cv_rows,
        "train_oof": train_result,
        "blind": blind_result,
        "top_feature_gain": importance[:20],
        "portable_tree_max_abs_error": portable_max_error,
        "model_output": str(model_path),
        "model_sha256": sha256(model_path),
        "sources": {
            "features": str(features_path),
            "features_sha256": sha256(features_path),
            "outcomes": str(outcomes_path),
            "outcomes_sha256": sha256(outcomes_path),
            "bank_receipt": str(bank_path),
            "bank_receipt_sha256": sha256(bank_path),
            "prefix_audit": str(prefix_path),
            "prefix_audit_sha256": sha256(prefix_path),
        },
        "truth_boundary": (
            "Blind JAX screening against three stepwise-exact opponents. The route "
            "choice is prefix-safe and public-state-only, but official Python 1.32.7 "
            "independent-seed validation remains mandatory before promotion."
        ),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "route_count": result["route_count"],
        "selected_model": selected_spec["name"],
        "selected_threshold": selected_threshold,
        "baseline_route": result["baseline_route"],
        "train_oof": train_result,
        "blind": blind_result,
        "portable_tree_max_abs_error": portable_max_error,
        "model": str(model_path),
        "receipt": str(receipt_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
