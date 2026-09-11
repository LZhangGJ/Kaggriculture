#!/usr/bin/env python3
"""Fit a public-state Route37/Route38 terminal-margin router.

The split is by event seed, never by individual game, so the same random shop
calendar cannot appear in both train and blind partitions.  Opponent id, seed,
seat, future events and terminal outcomes are used only for grouping/evaluation
and are never model inputs.
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def summarize(
    selected_margin: np.ndarray,
    opponent_names: list[str],
    opponent_index: np.ndarray,
) -> dict[str, Any]:
    win = selected_margin > 0
    tie = selected_margin == 0
    per_opponent: dict[str, Any] = {}
    for index, name in enumerate(opponent_names):
        mask = opponent_index == index
        values = selected_margin[mask]
        per_opponent[name] = {
            "games": int(mask.sum()),
            "wins": int(np.sum(values > 0)),
            "ties": int(np.sum(values == 0)),
            "win_rate": float(np.mean(values > 0)),
            "mean_margin": float(np.mean(values)),
        }
    return {
        "games": int(selected_margin.size),
        "wins": int(np.sum(win)),
        "ties": int(np.sum(tie)),
        "win_rate": float(np.mean(win)),
        "mean_margin": float(np.mean(selected_margin)),
        "minimum_opponent_win_rate": float(
            min(row["win_rate"] for row in per_opponent.values())
        ),
        "opponents_below_90pct": sorted(
            name for name, row in per_opponent.items() if row["win_rate"] < 0.90
        ),
        "per_opponent": per_opponent,
    }


def flatten_panel(features_path: Path, outcomes_path: Path, receipt_path: Path):
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    opponent_names = [row["name"] for row in receipt["opponents"]]
    with np.load(features_path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        feature_names = [str(value) for value in data["feature_names"]]
        seeds = np.asarray(data["seeds"], dtype=np.int32)
        opponent_ids = np.asarray(data["opponent_ids"], dtype=np.int16)
    with np.load(outcomes_path, allow_pickle=False) as data:
        margins = np.asarray(data["margins"], dtype=np.float32)[:, :, 0]
        candidate_ids = np.asarray(data["candidate_ids"], dtype=np.int16)
    if features.shape[:3] != margins.shape[1:]:
        raise ValueError(
            f"feature/outcome panel mismatch: {features.shape} vs {margins.shape}"
        )
    if margins.shape[0] != 2:
        raise ValueError("exactly two candidate routes are required")

    # Source tensors are [seat, opponent, seed, feature] and
    # [candidate, seat, opponent, seed].  Keep opponent/seed/seat only as
    # audit metadata and flatten in the same deterministic order.
    seat_grid, opponent_grid, seed_grid = np.meshgrid(
        np.arange(features.shape[0], dtype=np.int8),
        np.arange(features.shape[1], dtype=np.int16),
        np.arange(features.shape[2], dtype=np.int16),
        indexing="ij",
    )
    x = features.reshape(-1, features.shape[-1])
    route37 = margins[0].reshape(-1)
    route38 = margins[1].reshape(-1)
    return {
        "x": x,
        "route37": route37,
        "route38": route38,
        "delta": route38 - route37,
        "seat": seat_grid.reshape(-1),
        "opponent": opponent_grid.reshape(-1),
        "seed_index": seed_grid.reshape(-1),
        "seed": seeds[seed_grid.reshape(-1)],
        "seeds": seeds,
        "opponent_ids": opponent_ids,
        "opponent_names": opponent_names,
        "feature_names": feature_names,
        "candidate_ids": candidate_ids,
    }


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


def score_key(summary: dict[str, Any]) -> tuple[float, float, float]:
    return (
        summary["minimum_opponent_win_rate"],
        summary["win_rate"],
        summary["mean_margin"],
    )


def portable_tree_value(node: dict[str, Any], values: np.ndarray) -> float:
    while "leaf_value" not in node:
        feature = int(node["split_feature"])
        threshold = float(node["threshold"])
        node = (
            node["left_child"]
            if float(values[feature]) <= threshold
            else node["right_child"]
        )
    return float(node["leaf_value"])


def portable_predict(model_dump: dict[str, Any], values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            sum(
                portable_tree_value(tree["tree_structure"], row)
                for tree in model_dump["tree_info"]
            )
            for row in values
        ],
        dtype=np.float64,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--train-seeds", type=int, default=160)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()

    panel = flatten_panel(args.features, args.outcomes, args.bank_receipt)
    if args.train_seeds <= 0 or args.train_seeds >= len(panel["seeds"]):
        raise ValueError("train-seeds must leave a non-empty blind seed block")
    train_mask = panel["seed_index"] < args.train_seeds
    blind_mask = ~train_mask
    x_train = panel["x"][train_mask]
    y_train = panel["delta"][train_mask]
    groups = panel["seed"][train_mask]

    specs = [
        {
            "name": "tiny_depth2",
            "n_estimators": 64,
            "learning_rate": 0.05,
            "num_leaves": 4,
            "max_depth": 2,
            "min_child_samples": 96,
            "reg_lambda": 20.0,
            "colsample_bytree": 0.75,
        },
        {
            "name": "small_depth3",
            "n_estimators": 96,
            "learning_rate": 0.04,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 96,
            "reg_lambda": 20.0,
            "colsample_bytree": 0.80,
        },
        {
            "name": "small_depth4",
            "n_estimators": 128,
            "learning_rate": 0.03,
            "num_leaves": 12,
            "max_depth": 4,
            "min_child_samples": 128,
            "reg_lambda": 30.0,
            "colsample_bytree": 0.80,
        },
    ]
    thresholds = (-5000.0, -3000.0, -2000.0, -1000.0, 0.0, 1000.0, 2000.0, 3000.0, 5000.0)
    splitter = GroupKFold(n_splits=4)
    cv_rows = []
    best: tuple[tuple[float, float, float], dict[str, Any], float, np.ndarray] | None = None
    for spec_index, spec in enumerate(specs):
        oof = np.zeros_like(y_train, dtype=np.float32)
        for fold, (fit_index, validation_index) in enumerate(
            splitter.split(x_train, y_train, groups=groups)
        ):
            model = make_model(spec, 17000 + 100 * spec_index + fold)
            model.fit(x_train[fit_index], y_train[fit_index])
            oof[validation_index] = model.predict(x_train[validation_index])
        train_route37 = panel["route37"][train_mask]
        train_route38 = panel["route38"][train_mask]
        train_opponent = panel["opponent"][train_mask]
        for threshold in thresholds:
            choose38 = oof > threshold
            selected = np.where(choose38, train_route38, train_route37)
            summary = summarize(
                selected, panel["opponent_names"], train_opponent
            )
            row = {
                "model": spec["name"],
                "threshold": threshold,
                "route38_fraction": float(np.mean(choose38)),
                **{key: value for key, value in summary.items() if key != "per_opponent"},
            }
            cv_rows.append(row)
            key = score_key(summary)
            if best is None or key > best[0]:
                best = (key, spec, threshold, oof.copy())
    assert best is not None
    _, selected_spec, selected_threshold, selected_oof = best

    model = make_model(selected_spec, 18001)
    model.fit(x_train, y_train)
    blind_prediction = model.predict(panel["x"][blind_mask]).astype(np.float32)

    def partition(mask: np.ndarray, prediction: np.ndarray | None) -> dict[str, Any]:
        r37 = panel["route37"][mask]
        r38 = panel["route38"][mask]
        opponent = panel["opponent"][mask]
        oracle = np.maximum(r37, r38)
        result = {
            "route37": summarize(r37, panel["opponent_names"], opponent),
            "route38": summarize(r38, panel["opponent_names"], opponent),
            "oracle_two_route": summarize(
                oracle, panel["opponent_names"], opponent
            ),
        }
        if prediction is not None:
            choose38 = prediction > selected_threshold
            routed = np.where(choose38, r38, r37)
            result["router"] = summarize(
                routed, panel["opponent_names"], opponent
            )
            result["router"]["route38_fraction"] = float(np.mean(choose38))
            result["router"]["delta_prediction_mae"] = float(
                np.mean(np.abs(prediction - panel["delta"][mask]))
            )
            result["router"]["delta_prediction_correlation"] = float(
                np.corrcoef(prediction, panel["delta"][mask])[0, 1]
            )
        return result

    train_result = partition(train_mask, selected_oof)
    blind_result = partition(blind_mask, blind_prediction)
    importance = sorted(
        (
            {
                "feature": name,
                "gain": float(value),
            }
            for name, value in zip(
                panel["feature_names"],
                model.booster_.feature_importance(importance_type="gain"),
                strict=True,
            )
        ),
        key=lambda row: -row["gain"],
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model_dump = model.booster_.dump_model()
    probe = panel["x"][blind_mask][: min(2048, int(np.sum(blind_mask)))]
    portable_error = float(
        np.max(
            np.abs(
                portable_predict(model_dump, probe)
                - model.predict(probe).astype(np.float64)
            )
        )
    )
    args.model_output.write_text(
        json.dumps(model_dump, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema": "route37-route38-jax-margin-router-fit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "JAX_SCREEN_ONLY",
        "forbidden_runtime_inputs": [
            "opponent_identity",
            "seed",
            "seat",
            "future_events",
            "terminal_outcome",
            "private_opponent_state",
        ],
        "target": "terminal_margin_route38_minus_route37",
        "split": {
            "type": "contiguous_disjoint_event_seed_blocks",
            "train_seed_start": int(panel["seeds"][0]),
            "train_seed_end": int(panel["seeds"][args.train_seeds - 1]),
            "blind_seed_start": int(panel["seeds"][args.train_seeds]),
            "blind_seed_end": int(panel["seeds"][-1]),
            "train_seed_count": args.train_seeds,
            "blind_seed_count": len(panel["seeds"]) - args.train_seeds,
        },
        "selected_model": selected_spec,
        "selected_threshold": selected_threshold,
        "cv_candidates": cv_rows,
        "train_oof": train_result,
        "blind": blind_result,
        "top_feature_gain": importance[:20],
        "model_output": str(args.model_output.resolve()),
        "model_sha256": sha256(args.model_output),
        "portable_tree_max_abs_error": portable_error,
        "sources": {
            "features": str(args.features.resolve()),
            "features_sha256": sha256(args.features),
            "outcomes": str(args.outcomes.resolve()),
            "outcomes_sha256": sha256(args.outcomes),
            "bank_receipt": str(args.bank_receipt.resolve()),
            "bank_receipt_sha256": sha256(args.bank_receipt),
        },
        "truth_boundary": (
            "GPU screening against a stage-1 24-opponent bank. Boatlee is "
            "stepwise exact; the other 23 retain outcome-parity rather than "
            "stepwise-exact status. Official Python 1.32.7 holdout is mandatory."
        ),
    }
    args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    concise = {
        "status": result["status"],
        "selected_model": selected_spec["name"],
        "selected_threshold": selected_threshold,
        "train_oof": {
            name: {
                key: value
                for key, value in row.items()
                if key in {"win_rate", "mean_margin", "minimum_opponent_win_rate", "route38_fraction"}
            }
            for name, row in train_result.items()
        },
        "blind": {
            name: {
                key: value
                for key, value in row.items()
                if key in {"win_rate", "mean_margin", "minimum_opponent_win_rate", "route38_fraction", "delta_prediction_correlation"}
            }
            for name, row in blind_result.items()
        },
        "receipt": str(args.receipt_output.resolve()),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
