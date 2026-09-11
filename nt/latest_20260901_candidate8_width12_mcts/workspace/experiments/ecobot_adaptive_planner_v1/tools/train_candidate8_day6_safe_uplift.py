#!/usr/bin/env python3
"""Train a conservative two-head Candidate8-vs-KEEP Day6 selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_strings(raw: str) -> set[str]:
    return {value.strip() for value in raw.split(",") if value.strip()}


def make_features(
    raw: np.ndarray, family: np.ndarray, seat: np.ndarray, names: list[str],
) -> tuple[np.ndarray, list[str]]:
    values = raw.astype(np.float32)
    for index, name in enumerate(names):
        if "x100" in name:
            values[:, index] /= 100.0
    one_hot = np.eye(9, dtype=np.float32)[family.astype(np.int64)]
    return np.concatenate([
        values,
        one_hot,
        seat.astype(np.float32).reshape(-1, 1),
    ], axis=1), [
        *names,
        *(f"family_{index}" for index in range(9)),
        "candidate_seat",
    ]


def paired_targets(
    state_id: np.ndarray, own: np.ndarray, rival: np.ndarray, z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_target = np.zeros(len(state_id), dtype=np.float64)
    margin_lcb_target = np.zeros(len(state_id), dtype=np.float64)
    cash_lcb_target = np.zeros(len(state_id), dtype=np.float64)
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        state_own = own[indices]
        state_rival = rival[indices]
        keep_own = state_own[0:1]
        keep_rival = state_rival[0:1]
        score = np.where(
            state_own > state_rival,
            1.0,
            np.where(state_own == state_rival, 0.5, 0.0),
        )
        keep_score = np.where(
            keep_own > keep_rival,
            1.0,
            np.where(keep_own == keep_rival, 0.5, 0.0),
        )
        score_delta = score - keep_score
        margin_delta = (state_own - state_rival) - (keep_own - keep_rival)
        cash_delta = state_own - keep_own
        score_target[indices] = score_delta.mean(axis=1)
        for target, values in (
            (margin_lcb_target, margin_delta),
            (cash_lcb_target, cash_delta),
        ):
            mean = values.mean(axis=1)
            standard_error = values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])
            target[indices] = mean - z * standard_error
    return score_target, margin_lcb_target, cash_lcb_target


def paired_mean_margin_target(
    state_id: np.ndarray, own: np.ndarray, rival: np.ndarray,
) -> np.ndarray:
    target = np.zeros(len(state_id), dtype=np.float64)
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        margin = own[indices] - rival[indices]
        target[indices] = (margin - margin[0:1]).mean(axis=1)
    return target


def select_rows(
    states: set[int], state_id: np.ndarray,
    predicted_score: np.ndarray, predicted_margin: np.ndarray,
    score_target: np.ndarray, margin_target: np.ndarray, cash_target: np.ndarray,
    score_threshold: float, margin_threshold: float,
    predicted_rank_margin: np.ndarray | None = None,
    margin_mean_target: np.ndarray | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for state in sorted(states):
        indices = np.flatnonzero(state_id == state)
        if not len(indices):
            continue
        keep = int(indices[0])
        eligible = [
            int(index) for index in indices[1:]
            if predicted_score[index] >= score_threshold
            and predicted_margin[index] >= margin_threshold
        ]
        selected = keep if not eligible else max(
            eligible,
            key=lambda index: (
                float(predicted_score[index]),
                float(
                    predicted_margin[index]
                    if predicted_rank_margin is None
                    else predicted_rank_margin[index]
                ),
                -int(index),
            ),
        )
        rows.append({
            "state_id": int(state),
            "activated": selected != keep,
            "score_delta": float(score_target[selected]),
            "margin_mean_delta": float(
                margin_target[selected]
                if margin_mean_target is None
                else margin_mean_target[selected]
            ),
            "margin_lcb_delta": float(margin_target[selected]),
            "cash_lcb_delta": float(cash_target[selected]),
            "selected_local_rank": int(np.flatnonzero(indices == selected)[0]),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    active = [row for row in rows if row["activated"]]
    def mean(rows_: list[dict], key: str) -> float:
        return float(np.mean([row[key] for row in rows_])) if rows_ else 0.0
    return {
        "states": len(rows),
        "activation_rate": len(active) / len(rows) if rows else 0.0,
        "mean_score_delta_all": mean(rows, "score_delta"),
        "mean_margin_delta_all": mean(rows, "margin_mean_delta"),
        "mean_margin_lcb_delta_all": mean(rows, "margin_lcb_delta"),
        "mean_cash_lcb_delta_all": mean(rows, "cash_lcb_delta"),
        "mean_score_delta_active": mean(active, "score_delta"),
        "mean_margin_delta_active": mean(active, "margin_mean_delta"),
        "mean_margin_lcb_delta_active": mean(active, "margin_lcb_delta"),
        "active_score_regression_rate": float(np.mean([
            row["score_delta"] < 0 for row in active
        ])) if active else 0.0,
        "active_margin_negative_rate": float(np.mean([
            row["margin_lcb_delta"] < 0 for row in active
        ])) if active else 0.0,
    }


def passes(metrics: dict) -> bool:
    return (
        metrics["activation_rate"] >= 0.05
        and metrics["active_score_regression_rate"] <= 0.10
        and metrics["active_margin_negative_rate"] <= 0.20
        and metrics["mean_score_delta_active"] >= 0.0
        and metrics["mean_margin_lcb_delta_active"] > 0.0
    )


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0:
        return 0.0
    if np.std(left) == 0 or np.std(right) == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--holdout-opponents", type=parse_strings, default=parse_strings("G031,G038,G092"))
    parser.add_argument("--holdout-seed-count", type=int, default=2)
    parser.add_argument("--calibration-seed-count", type=int, default=2)
    parser.add_argument("--z", type=float, default=1.0)
    parser.add_argument(
        "--raw-feature-count",
        type=int,
        help="Optional prefix width for a controlled feature ablation.",
    )
    parser.add_argument(
        "--score-nonzero-weight",
        type=float,
        default=1.0,
        help=(
            "Sample weight for candidate-vs-KEEP score targets that are not "
            "zero. This counters the large mass of candidates that leave the "
            "win/loss result unchanged without changing the continuous target."
        ),
    )
    parser.add_argument(
        "--blind-future-split",
        action="store_true",
        help=(
            "Use even future columns as Bank A for fit/calibration and odd "
            "columns as an untouched Bank B for all held-out evaluation."
        ),
    )
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"])
        opponents = np.asarray(data["opponent"])
        seeds = np.asarray(data["prefix_seed"])
        seat = np.asarray(data["seat"])
        family = np.asarray(data["family"])
        raw_features = np.asarray(data["features"])
        feature_names = [str(value) for value in data["feature_names"]]
        own = np.asarray(data["future_own_cash"], dtype=np.float64)
        rival = np.asarray(data["future_opponent_cash"], dtype=np.float64)

    if args.raw_feature_count is not None:
        if not 1 <= args.raw_feature_count <= raw_features.shape[1]:
            raise ValueError("raw feature count is outside the dataset width")
        raw_features = raw_features[:, :args.raw_feature_count]
        feature_names = feature_names[:args.raw_feature_count]
    x, model_feature_names = make_features(raw_features, family, seat, feature_names)
    if args.blind_future_split:
        if own.shape[1] < 8 or own.shape[1] % 2:
            raise ValueError(
                "blind future split requires an even number of at least 8 futures"
            )
        bank_a = np.arange(0, own.shape[1], 2, dtype=np.int64)
        bank_b = np.arange(1, own.shape[1], 2, dtype=np.int64)
        score_target_a, margin_target_a, cash_target_a = paired_targets(
            state_id, own[:, bank_a], rival[:, bank_a], args.z
        )
        margin_mean_target_a = paired_mean_margin_target(
            state_id, own[:, bank_a], rival[:, bank_a]
        )
        score_target_b, margin_target_b, cash_target_b = paired_targets(
            state_id, own[:, bank_b], rival[:, bank_b], args.z
        )
        margin_mean_target_b = paired_mean_margin_target(
            state_id, own[:, bank_b], rival[:, bank_b]
        )
    else:
        bank_a = np.arange(own.shape[1], dtype=np.int64)
        bank_b = bank_a.copy()
        score_target_a, margin_target_a, cash_target_a = paired_targets(
            state_id, own, rival, args.z
        )
        margin_mean_target_a = paired_mean_margin_target(state_id, own, rival)
        score_target_b = score_target_a
        margin_target_b = margin_target_a
        margin_mean_target_b = margin_mean_target_a
        cash_target_b = cash_target_a
    unique_seeds = sorted(int(value) for value in np.unique(seeds))
    test_seeds = set(unique_seeds[-args.holdout_seed_count:])
    calibration_start = -(args.holdout_seed_count + args.calibration_seed_count)
    calibration_seeds = set(unique_seeds[calibration_start:-args.holdout_seed_count])
    train_seeds = set(unique_seeds[:calibration_start])
    all_opponents = {str(value) for value in np.unique(opponents)}
    test_opponents = all_opponents & args.holdout_opponents
    train_opponents = all_opponents - test_opponents

    state_meta: dict[int, tuple[str, int]] = {}
    for state in np.unique(state_id):
        index = int(np.flatnonzero(state_id == state)[0])
        state_meta[int(state)] = (str(opponents[index]), int(seeds[index]))
    def states_for(opponent_set: set[str], seed_set: set[int]) -> set[int]:
        return {
            state for state, (opponent, seed) in state_meta.items()
            if opponent in opponent_set and seed in seed_set
        }
    partitions = {
        "fit": states_for(train_opponents, train_seeds),
        "calibration": states_for(train_opponents, calibration_seeds),
        "unseen_seeds": states_for(train_opponents, test_seeds),
        "unseen_opponents": states_for(test_opponents, train_seeds),
        "joint_unseen": states_for(test_opponents, test_seeds),
    }
    fit_indices = np.flatnonzero(np.isin(state_id, list(partitions["fit"])))

    common = dict(
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=15,
        max_depth=5,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=20260830,
        n_jobs=16,
        verbosity=-1,
    )
    score_model = lgb.LGBMRegressor(**common)
    margin_model = lgb.LGBMRegressor(**common)
    margin_mean_model = lgb.LGBMRegressor(**common)
    score_weights = np.ones(len(fit_indices), dtype=np.float64)
    score_weights[np.abs(score_target_a[fit_indices]) > 1e-12] = (
        args.score_nonzero_weight
    )
    score_model.fit(
        x[fit_indices], score_target_a[fit_indices], sample_weight=score_weights
    )
    margin_model.fit(x[fit_indices], margin_target_a[fit_indices])
    margin_mean_model.fit(x[fit_indices], margin_mean_target_a[fit_indices])
    predicted_score = score_model.predict(x)
    predicted_margin = margin_model.predict(x)
    predicted_margin_mean = margin_mean_model.predict(x)

    prediction_diagnostics = {}
    for name, states in partitions.items():
        indices = np.flatnonzero(np.isin(state_id, list(states)))
        evaluation_score_target = (
            score_target_a
            if name in {"fit", "calibration"}
            else score_target_b
        )
        evaluation_margin_target = (
            margin_target_a
            if name in {"fit", "calibration"}
            else margin_target_b
        )
        evaluation_margin_mean_target = (
            margin_mean_target_a
            if name in {"fit", "calibration"}
            else margin_mean_target_b
        )
        prediction_diagnostics[name] = {
            "rows": int(len(indices)),
            "score_correlation": correlation(
                predicted_score[indices], evaluation_score_target[indices]
            ),
            "margin_lcb_correlation": correlation(
                predicted_margin[indices], evaluation_margin_target[indices]
            ),
            "score_mae": float(np.mean(np.abs(
                predicted_score[indices] - evaluation_score_target[indices]
            ))) if len(indices) else 0.0,
            "margin_lcb_mae": float(np.mean(np.abs(
                predicted_margin[indices] - evaluation_margin_target[indices]
            ))) if len(indices) else 0.0,
            "margin_mean_correlation": correlation(
                predicted_margin_mean[indices],
                evaluation_margin_mean_target[indices],
            ),
            "margin_mean_mae": float(np.mean(np.abs(
                predicted_margin_mean[indices]
                - evaluation_margin_mean_target[indices]
            ))) if len(indices) else 0.0,
        }

    calibration_indices = np.flatnonzero(np.isin(
        state_id, list(partitions["calibration"])
    ))
    score_grid = np.unique(np.concatenate([
        np.asarray([0.0, np.inf]),
        np.maximum(0.0, np.quantile(predicted_score[calibration_indices], np.linspace(0.50, 0.98, 20))),
    ]))
    margin_grid = np.unique(np.concatenate([
        np.asarray([0.0, np.inf]),
        np.maximum(0.0, np.quantile(predicted_margin[calibration_indices], np.linspace(0.50, 0.98, 20))),
    ]))
    calibration_candidates = []
    for score_threshold in score_grid:
        for margin_threshold in margin_grid:
            rows = select_rows(
                partitions["calibration"], state_id,
                predicted_score, predicted_margin,
                score_target_a, margin_target_a, cash_target_a,
                float(score_threshold), float(margin_threshold),
                predicted_margin_mean,
                margin_mean_target_a,
            )
            metrics = summarize(rows)
            calibration_candidates.append({
                "score_threshold": float(score_threshold),
                "margin_threshold": float(margin_threshold),
                "metrics": metrics,
                "passed": passes(metrics),
            })
    eligible = [row for row in calibration_candidates if row["passed"]]
    active_calibration_candidates = [
        row for row in calibration_candidates
        if row["metrics"]["activation_rate"] >= 0.05
    ]
    best_nonpassing = max(
        active_calibration_candidates,
        key=lambda row: (
            -row["metrics"]["active_score_regression_rate"],
            -row["metrics"]["active_margin_negative_rate"],
            row["metrics"]["mean_score_delta_all"],
            row["metrics"]["mean_margin_lcb_delta_all"],
        ),
    ) if active_calibration_candidates else None
    if eligible:
        chosen = max(eligible, key=lambda row: (
            row["metrics"]["mean_score_delta_all"],
            row["metrics"]["mean_margin_lcb_delta_all"],
            -row["metrics"]["active_score_regression_rate"],
        ))
    else:
        chosen = {
            "score_threshold": float("inf"),
            "margin_threshold": float("inf"),
            "metrics": summarize(select_rows(
                partitions["calibration"], state_id,
                predicted_score, predicted_margin,
                score_target_a, margin_target_a, cash_target_a,
                float("inf"), float("inf"),
                predicted_margin_mean,
                margin_mean_target_a,
            )),
            "passed": False,
        }

    metrics = {}
    selected_rows = {}
    for name, states in partitions.items():
        evaluation_targets = (
            (score_target_a, margin_target_a, cash_target_a)
            if name in {"fit", "calibration"}
            else (score_target_b, margin_target_b, cash_target_b)
        )
        rows = select_rows(
            states, state_id, predicted_score, predicted_margin,
            *evaluation_targets,
            chosen["score_threshold"], chosen["margin_threshold"],
            predicted_margin_mean,
            (
                margin_mean_target_a
                if name in {"fit", "calibration"}
                else margin_mean_target_b
            ),
        )
        selected_rows[name] = rows
        metrics[name] = summarize(rows)

    joint_passed = passes(metrics["joint_unseen"])
    deployment_gate_passed = all(
        passes(metrics[name])
        for name in (
            "calibration", "unseen_seeds", "unseen_opponents", "joint_unseen"
        )
    )
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "score_model": score_model,
        "margin_model": margin_model,
        "margin_mean_model": margin_mean_model,
        "feature_names": model_feature_names,
        "score_threshold": chosen["score_threshold"],
        "margin_threshold": chosen["margin_threshold"],
        "z": args.z,
    }, args.model_output)
    payload = {
        "schema": "kaggriculture.candidate8-day6-safe-uplift.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "Public-state regressors predict paired candidate-vs-KEEP score delta, "
            "mean margin delta, and margin LCB. The first two are the policy value "
            "outputs; the LCB is only a safety gate. Opponent identity and seed are "
            "split keys only. "
            "When blind_future_split is enabled, Bank B is never used for fit or "
            "threshold calibration."
        ),
        "future_banks": {
            "blind_future_split": args.blind_future_split,
            "bank_a_columns": bank_a.tolist(),
            "bank_b_columns": bank_b.tolist(),
            "bank_a_usage": "fit and threshold calibration",
            "bank_b_usage": "unseen-seed/opponent evaluation only",
        },
        "training": {
            "raw_feature_count": int(raw_features.shape[1]),
            "score_nonzero_weight": args.score_nonzero_weight,
            "score_nonzero_fit_rows": int(np.count_nonzero(
                np.abs(score_target_a[fit_indices]) > 1e-12
            )),
            "score_fit_rows": int(len(fit_indices)),
        },
        "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
        "split": {
            "train_opponents": sorted(train_opponents),
            "test_opponents": sorted(test_opponents),
            "train_seeds": sorted(train_seeds),
            "calibration_seeds": sorted(calibration_seeds),
            "test_seeds": sorted(test_seeds),
        },
        "selected_thresholds": chosen,
        "best_nonpassing_thresholds": best_nonpassing,
        "prediction_diagnostics": prediction_diagnostics,
        "metrics": metrics,
        "selected_rows": selected_rows,
        "joint_gate_passed": joint_passed,
        "deployment_gate_passed": deployment_gate_passed,
        "model": {"path": str(args.model_output), "sha256": sha256(args.model_output)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_thresholds": chosen,
        "best_nonpassing_thresholds": best_nonpassing,
        "prediction_diagnostics": prediction_diagnostics,
        "metrics": metrics,
        "joint_gate_passed": joint_passed,
        "deployment_gate_passed": deployment_gate_passed,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
