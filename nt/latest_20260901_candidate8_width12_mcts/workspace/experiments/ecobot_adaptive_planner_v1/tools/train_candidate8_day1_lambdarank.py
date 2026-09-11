#!/usr/bin/env python3
"""Train and strictly validate the public-state Candidate8 Day1 ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np


DEFAULT_HOLDOUT_OPPONENTS = "G001,G003,G049,G245"
SCORE_WEIGHT = 10_000_000.0
MARGIN_WEIGHT = 1.0
CASH_WEIGHT = 0.001


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
    result = np.concatenate([
        values,
        one_hot,
        seat.astype(np.float32).reshape(-1, 1),
    ], axis=1)
    return result, [
        *names,
        *(f"family_{index}" for index in range(9)),
        "candidate_seat",
    ]


def utility(score: np.ndarray, margin: np.ndarray, cash: np.ndarray) -> np.ndarray:
    return SCORE_WEIGHT * score + MARGIN_WEIGHT * margin + CASH_WEIGHT * cash


def state_order(
    indices: np.ndarray,
    score: np.ndarray,
    margin: np.ndarray,
    cash: np.ndarray,
    signature: np.ndarray,
) -> np.ndarray:
    return np.asarray(sorted(
        indices,
        key=lambda index: (
            -float(score[index]),
            -float(margin[index]),
            -float(cash[index]),
            int(signature[index]),
        ),
    ), dtype=np.int64)


def relevance_labels(
    state_id: np.ndarray,
    score: np.ndarray,
    margin: np.ndarray,
    cash: np.ndarray,
    signature: np.ndarray,
) -> np.ndarray:
    labels = np.zeros(len(state_id), dtype=np.int32)
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        order = state_order(indices, score, margin, cash, signature)
        keep = int(indices[0])
        keep_key = (float(score[keep]), float(margin[keep]), float(cash[keep]))
        labels[order[0]] = 4
        labels[order[1:4]] = np.maximum(labels[order[1:4]], 3)
        labels[order[4:8]] = np.maximum(labels[order[4:8]], 2)
        for index in indices:
            key = (float(score[index]), float(margin[index]), float(cash[index]))
            if key > keep_key:
                labels[index] = max(labels[index], 1)
        # Counterfactual means are often exactly tied.  Equal outcomes must
        # receive equal relevance rather than learning the signature tie-break.
        tied: dict[tuple[float, float, float], list[int]] = {}
        for index in indices:
            key = (float(score[index]), float(margin[index]), float(cash[index]))
            tied.setdefault(key, []).append(int(index))
        for tied_indices in tied.values():
            tied_label = int(labels[tied_indices].max())
            labels[tied_indices] = tied_label
    return labels


def half_bank_label_stability(
    state_id: np.ndarray, own: np.ndarray, rival: np.ndarray,
) -> dict:
    if own.ndim != 2 or own.shape != rival.shape or own.shape[1] < 4:
        return {"available": False}
    half = own.shape[1] // 2
    correlations = []
    exact = []
    recall_at_8 = []
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        utility_halves = []
        for columns in (slice(0, half), slice(half, own.shape[1])):
            own_mean = own[indices, columns].mean(axis=1)
            rival_mean = rival[indices, columns].mean(axis=1)
            margin_mean = own_mean - rival_mean
            score_mean = np.where(
                own[indices, columns] > rival[indices, columns],
                1.0,
                np.where(
                    own[indices, columns] == rival[indices, columns], 0.5, 0.0
                ),
            ).mean(axis=1)
            utility_halves.append(utility(score_mean, margin_mean, own_mean))
        first, second = utility_halves
        if np.std(first) == 0 or np.std(second) == 0:
            correlation = 1.0 if np.array_equal(first, second) else 0.0
        else:
            correlation = float(np.corrcoef(first, second)[0, 1])
        first_order = np.argsort(-first, kind="stable")
        second_order = np.argsort(-second, kind="stable")
        correlations.append(correlation)
        exact.append(int(first_order[0] == second_order[0]))
        recall_at_8.append(int(first_order[0] in set(second_order[:8].tolist())))
    return {
        "available": True,
        "future_count": int(own.shape[1]),
        "half_count": int(half),
        "states": len(correlations),
        "mean_candidate_utility_correlation": float(np.mean(correlations)),
        "half_bank_exact_oracle_agreement": float(np.mean(exact)),
        "half_bank_oracle_recall_at_8": float(np.mean(recall_at_8)),
    }


def subset_by_states(
    state_id: np.ndarray, states: set[int], x: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    indices = np.flatnonzero(np.isin(state_id, list(states)))
    groups = [int(np.sum(state_id[indices] == state)) for state in sorted(states)]
    # Rows are generated state-by-state.  Reorder explicitly to make the
    # LightGBM query grouping invariant to future dataset concatenation.
    ordered = np.concatenate([
        indices[state_id[indices] == state] for state in sorted(states)
    ])
    return x[ordered], y[ordered], groups


def select_rows(
    states: set[int], state_id: np.ndarray, prediction: np.ndarray,
    score: np.ndarray, margin: np.ndarray, cash: np.ndarray,
    signature: np.ndarray, threshold: float | None,
) -> list[dict]:
    actual_utility = utility(score, margin, cash)
    rows: list[dict] = []
    for state in sorted(states):
        indices = np.flatnonzero(state_id == state)
        if not len(indices):
            continue
        keep = int(indices[0])
        oracle = int(state_order(
            indices, score, margin, cash, signature
        )[0])
        predicted_order = np.asarray(sorted(
            indices,
            key=lambda index: (-float(prediction[index]), int(signature[index])),
        ), dtype=np.int64)
        raw = int(predicted_order[0])
        uplift = float(prediction[raw] - prediction[keep])
        selected = keep if threshold is not None and uplift <= threshold else raw
        top8 = set(int(value) for value in predicted_order[:8])
        rows.append({
            "state_id": int(state),
            "activated": selected != keep,
            "predicted_uplift": uplift,
            "oracle_exact": selected == oracle,
            "oracle_in_top8": oracle in top8,
            "score_gain": float(score[selected] - score[keep]),
            "margin_gain": float(margin[selected] - margin[keep]),
            "cash_gain": float(cash[selected] - cash[keep]),
            "utility_gain": float(actual_utility[selected] - actual_utility[keep]),
            "raw_utility_gain": float(actual_utility[raw] - actual_utility[keep]),
            "selected_signature": int(signature[selected]),
            "oracle_signature": int(signature[oracle]),
        })
    return rows


def summarize(rows: list[dict], threshold: float | None) -> dict:
    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows])) if rows else 0.0
    return {
        "states": len(rows),
        "safe_threshold": threshold,
        "activation_rate": mean("activated"),
        "exact_oracle_recall": mean("oracle_exact"),
        "oracle_recall_at_8": mean("oracle_in_top8"),
        "mean_score_gain_vs_keep": mean("score_gain"),
        "mean_margin_gain_vs_keep": mean("margin_gain"),
        "mean_cash_gain_vs_keep": mean("cash_gain"),
        "mean_utility_gain_vs_keep": mean("utility_gain"),
        "negative_utility_rate": float(np.mean([
            row["utility_gain"] < 0 for row in rows
        ])) if rows else 0.0,
        "score_regression_rate": float(np.mean([
            row["score_gain"] < 0 for row in rows
        ])) if rows else 0.0,
    }


def calibrate_threshold(rows: list[dict]) -> tuple[float, dict]:
    predicted = np.asarray([row["predicted_uplift"] for row in rows])
    realized = np.asarray([row["raw_utility_gain"] for row in rows])
    thresholds = np.unique(np.concatenate([
        np.asarray([0.0, np.inf]),
        np.quantile(predicted, np.linspace(0.0, 0.98, 80)),
    ]))
    candidates = []
    for threshold in thresholds:
        active = predicted > threshold
        gains = np.where(active, realized, 0.0)
        candidates.append({
            "threshold": float(threshold),
            "activation_rate": float(active.mean()),
            "mean_utility_gain": float(gains.mean()),
            "negative_utility_rate": float(np.mean(gains < 0)),
        })
    eligible = [
        row for row in candidates
        if row["negative_utility_rate"] <= 0.05
        and row["activation_rate"] >= 0.05
    ]
    best = max(
        eligible or candidates,
        key=lambda row: (
            row["mean_utility_gain"],
            -row["negative_utility_rate"],
            row["activation_rate"],
        ),
    )
    return float(best["threshold"]), best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--holdout-opponents", type=parse_strings,
        default=parse_strings(DEFAULT_HOLDOUT_OPPONENTS),
    )
    parser.add_argument("--holdout-seed-count", type=int, default=4)
    parser.add_argument("--calibration-seed-count", type=int, default=2)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=20260830)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"])
        opponents = np.asarray(data["opponent"])
        seeds = np.asarray(data["prefix_seed"])
        seat = np.asarray(data["seat"])
        day = np.asarray(data["decision_day"])
        signature = np.asarray(data["signature"])
        family = np.asarray(data["family"])
        raw_features = np.asarray(data["features"])
        score = np.asarray(data["expected_score_rate"])
        margin = np.asarray(data["expected_margin"])
        cash = np.asarray(data["expected_own_cash"])
        future_own = np.asarray(data["future_own_cash"])
        future_rival = np.asarray(data["future_opponent_cash"])
        feature_names = [str(value) for value in data["feature_names"]]
    if np.any(day != 1):
        raise ValueError("Day1 ranker requires a Day1-only dataset")

    unique_seeds = np.unique(seeds)
    if (
        args.holdout_seed_count <= 0
        or args.calibration_seed_count <= 0
        or args.holdout_seed_count + args.calibration_seed_count >= len(unique_seeds)
    ):
        raise ValueError("seed split must leave fit, calibration, and test seeds")
    test_seeds = {int(value) for value in unique_seeds[-args.holdout_seed_count:]}
    calibration_start = -(
        args.holdout_seed_count + args.calibration_seed_count
    )
    calibration_end = -args.holdout_seed_count
    calibration_seeds = {
        int(value) for value in unique_seeds[calibration_start:calibration_end]
    }
    train_seeds = {
        int(value) for value in unique_seeds[:calibration_start]
    }
    all_opponents = {str(value) for value in np.unique(opponents)}
    test_opponents = all_opponents & args.holdout_opponents
    train_opponents = all_opponents - test_opponents
    if not train_seeds or not test_seeds or not train_opponents or not test_opponents:
        raise ValueError("strict split must leave all four partitions non-empty")

    state_meta = {}
    for state in np.unique(state_id):
        index = int(np.flatnonzero(state_id == state)[0])
        state_meta[int(state)] = (str(opponents[index]), int(seeds[index]))

    def states_for(opponent_set: set[str], seed_set: set[int]) -> set[int]:
        return {
            state for state, (opponent, seed_value) in state_meta.items()
            if opponent in opponent_set and seed_value in seed_set
        }

    fit_states = states_for(train_opponents, train_seeds)
    calibration_states = states_for(train_opponents, calibration_seeds)
    seed_test_states = states_for(train_opponents, test_seeds)
    opponent_test_states = states_for(test_opponents, train_seeds)
    joint_test_states = states_for(test_opponents, test_seeds)

    x, final_feature_names = make_features(
        raw_features, family, seat, feature_names
    )
    y = relevance_labels(state_id, score, margin, cash, signature)
    x_fit, y_fit, fit_groups = subset_by_states(state_id, fit_states, x, y)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.estimators,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_samples=20,
        reg_lambda=1.0,
        lambdarank_truncation_level=8,
        label_gain=[0, 1, 3, 7, 15],
        n_jobs=16,
        random_state=args.random_state,
        verbosity=-1,
    )
    model.fit(x_fit, y_fit, group=fit_groups)
    prediction = model.predict(x)

    calibration_raw = select_rows(
        calibration_states, state_id, prediction, score, margin, cash,
        signature, None,
    )
    threshold, threshold_summary = calibrate_threshold(calibration_raw)
    partitions = {
        "fit": fit_states,
        "calibration": calibration_states,
        "test_unseen_seeds": seed_test_states,
        "test_unseen_opponents": opponent_test_states,
        "test_joint_unseen": joint_test_states,
    }
    metrics = {
        name: summarize(select_rows(
            states, state_id, prediction, score, margin, cash,
            signature, threshold,
        ), threshold)
        for name, states in partitions.items()
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": final_feature_names,
        "safe_threshold": threshold,
        "holdout_opponents": sorted(test_opponents),
        "train_opponents": sorted(train_opponents),
        "train_seeds": sorted(train_seeds),
        "calibration_seeds": sorted(calibration_seeds),
        "test_seeds": sorted(test_seeds),
        "boundary": "public features plus candidate seat; no opponent ID or seed",
    }, args.model_output, compress=3)

    joint = metrics["test_joint_unseen"]
    opponent_test = metrics["test_unseen_opponents"]
    payload = {
        "schema": "kaggriculture.candidate8-day1-lambdarank.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(args.dataset),
            "sha256": sha256(args.dataset),
            "rows": int(len(state_id)),
            "states": int(len(np.unique(state_id))),
        },
        "split": {
            "train_opponents": sorted(train_opponents),
            "test_opponents": sorted(test_opponents),
            "train_seeds": sorted(train_seeds),
            "calibration_seeds": sorted(calibration_seeds),
            "test_seeds": sorted(test_seeds),
        },
        "model": {
            "type": "LightGBM_LambdaRank",
            "parameters": model.get_params(),
            "artifact": str(args.model_output),
            "sha256": sha256(args.model_output),
            "feature_count": len(final_feature_names),
            "feature_boundary": (
                "Candidate and public-state features, family one-hot, and seat. "
                "Opponent identity and random seed are excluded."
            ),
        },
        "safe_threshold_calibration": threshold_summary,
        "label_stability": half_bank_label_stability(
            state_id, future_own, future_rival
        ),
        "metrics": metrics,
        "gate": {
            "joint_score_gain_positive": joint["mean_score_gain_vs_keep"] > 0,
            "joint_margin_gain_positive": joint["mean_margin_gain_vs_keep"] > 0,
            "joint_negative_utility_le_10pct": joint["negative_utility_rate"] <= 0.10,
            "unseen_opponent_margin_gain_positive": (
                opponent_test["mean_margin_gain_vs_keep"] > 0
            ),
            "joint_oracle_recall_at_8_ge_50pct": joint["oracle_recall_at_8"] >= 0.50,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "model": str(args.model_output),
        "output": str(args.output),
        "safe_threshold": threshold,
        "metrics": metrics,
        "gate": payload["gate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
