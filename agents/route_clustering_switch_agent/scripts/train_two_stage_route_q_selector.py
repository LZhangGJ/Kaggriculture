#!/usr/bin/env python3
"""Train and validate a public-state Route-Q policy with two real switches."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from train_compact_route_q_selector import fit_model, policy_metrics


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def _csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part.strip())


def load_matrix(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        early_targets = saved["early_targets"].astype(str)
        late_targets = saved["late_targets"].astype(str)
        seeds = saved["seeds"].astype(np.int64)
        early_features = saved["early_states"].reshape(
            -1, saved["early_states"].shape[-1]
        ).astype(np.float32)
        late_features = saved["late_states"].reshape(
            len(early_targets), -1, saved["late_states"].shape[-1]
        ).astype(np.float32)
        outcome = saved["outcome"].reshape(
            len(early_targets), len(late_targets), -1
        )
        margins = saved["margin"].reshape(outcome.shape).astype(np.float32)
        return {
            "path": path,
            "opening": str(saved["opening"]),
            "checkpoints": saved["checkpoints"].astype(np.int16),
            "early_targets": early_targets,
            "late_targets": late_targets,
            "seeds": seeds,
            "early_features": early_features,
            "late_features": late_features,
            "scores": outcome.astype(np.float32) * 0.5,
            "margins": margins,
        }


def fit_policy(
    data: dict[str, Any], rows: np.ndarray, *, estimators: int, leaf: int,
    max_features: float, margin_weight: float, random_state: int,
) -> dict[str, Any]:
    scores = data["scores"]
    margins = data["margins"]
    late_models = []
    effective_scores = np.empty((len(rows), len(data["early_targets"])), np.float32)
    effective_margins = np.empty_like(effective_scores)
    local_rows = np.arange(len(rows))
    for early in range(len(data["early_targets"])):
        model = fit_model(
            data["late_features"][early, rows], scores[early, :, rows],
            margins[early, :, rows], estimators=estimators, leaf=leaf,
            max_features=max_features, margin_weight=margin_weight,
            random_state=random_state + early,
        )
        predictions = np.argmax(
            model.predict(data["late_features"][early, rows]), axis=1
        )
        effective_scores[:, early] = scores[early, predictions, rows]
        effective_margins[:, early] = margins[early, predictions, rows]
        late_models.append(model)
    early_model = fit_model(
        data["early_features"][rows], effective_scores, effective_margins,
        estimators=estimators, leaf=leaf, max_features=max_features,
        margin_weight=margin_weight, random_state=random_state + 1000,
    )
    return {"early": early_model, "late": late_models}


def select(
    data: dict[str, Any], policy: dict[str, Any], rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    early = np.argmax(
        policy["early"].predict(data["early_features"][rows]), axis=1
    )
    late = np.empty(len(rows), dtype=np.int32)
    for early_action, model in enumerate(policy["late"]):
        mask = early == early_action
        if np.any(mask):
            late[mask] = np.argmax(
                model.predict(data["late_features"][early_action, rows[mask]]),
                axis=1,
            )
    local = np.arange(len(rows))
    return (
        data["scores"][early, late, rows],
        data["margins"][early, late, rows],
        early,
        late,
    )


def metrics(scores: np.ndarray, margins: np.ndarray, seed_count: int) -> dict[str, float]:
    return policy_metrics(
        scores[:, None], margins[:, None], np.zeros(len(scores), dtype=np.int32),
        (1, seed_count, 2),
    )


def evaluate(data: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    rows = np.arange(len(data["early_features"]))
    scores, margins, early, late = select(data, policy, rows)
    paired = scores.reshape(len(data["seeds"]), 2)
    return {
        **metrics(scores, margins, len(data["seeds"])),
        "early_target_counts": dict(Counter(
            data["early_targets"][early].tolist()
        )),
        "late_action_counts": dict(Counter(
            data["late_targets"][late].tolist()
        )),
        "hidden_pair_oracle_raw_win_rate": float(
            np.max(data["scores"], axis=(0, 1)).mean()
        ),
        "seat_win_rates": [float(scores[0::2].mean()), float(scores[1::2].mean())],
        "both_seats_win_rate": float(np.all(paired == 1.0, axis=1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--estimators", type=int, default=64)
    parser.add_argument("--min-leaves", default="4,8,16,32")
    parser.add_argument("--max-features", default="0.25,0.5")
    parser.add_argument("--margin-weights", default="0,0.01")
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    data = load_matrix(args.matrix)
    groups = np.repeat(data["seeds"], 2)
    splits = list(GroupKFold(args.folds).split(data["early_features"], groups=groups))
    trials = []
    for leaf in _csv_ints(args.min_leaves):
        for max_features in _csv_floats(args.max_features):
            for margin_weight in _csv_floats(args.margin_weights):
                cv_scores = np.empty(len(groups), dtype=np.float32)
                cv_margins = np.empty(len(groups), dtype=np.float32)
                for fold, (train, valid) in enumerate(splits):
                    policy = fit_policy(
                        data, train, estimators=args.estimators, leaf=leaf,
                        max_features=max_features, margin_weight=margin_weight,
                        random_state=20260827 + 10000 * fold,
                    )
                    cv_scores[valid], cv_margins[valid], _, _ = select(
                        data, policy, valid
                    )
                trial_metrics = metrics(cv_scores, cv_margins, len(data["seeds"]))
                trials.append({
                    "min_leaf": leaf,
                    "max_features": max_features,
                    "margin_weight": margin_weight,
                    "cross_validation": trial_metrics,
                })
    trials.sort(key=lambda row: (
        -row["cross_validation"]["paired_seed_raw_win_lower_95pct"],
        -row["cross_validation"]["raw_win_rate"],
        -row["cross_validation"]["mean_margin"],
        row["min_leaf"], row["max_features"], row["margin_weight"],
    ))
    selected = trials[0]
    all_rows = np.arange(len(groups))
    policy = fit_policy(
        data, all_rows, estimators=args.estimators,
        leaf=int(selected["min_leaf"]),
        max_features=float(selected["max_features"]),
        margin_weight=float(selected["margin_weight"]), random_state=20260827,
    )
    bundle = {
        "schema": "two-stage-route-q-model-v1",
        "opening": data["opening"],
        "checkpoints": data["checkpoints"].tolist(),
        "early_targets": data["early_targets"].tolist(),
        "late_targets": data["late_targets"].tolist(),
        "early_model": policy["early"],
        "late_models": policy["late"],
        "selected": selected,
    }
    validation = None
    if args.validation:
        holdout = load_matrix(args.validation)
        for key in ("opening", "early_targets", "late_targets", "checkpoints"):
            if not np.array_equal(data[key], holdout[key]):
                raise ValueError(f"validation {key} does not match training")
        validation = evaluate(holdout, policy)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_model, compress=3)
    report = {
        "schema": "two-stage-route-q-training-v1",
        "source": str(args.matrix.resolve()),
        "validation_source": str(args.validation.resolve()) if args.validation else None,
        "seed_count": len(data["seeds"]),
        "selected": selected,
        "training": evaluate(data, policy),
        "validation": validation,
        "model_bytes": args.output_model.stat().st_size,
        "trials": trials,
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "model": str(args.output_model.resolve()),
        "selected": selected,
        "training": report["training"],
        "validation": validation,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
