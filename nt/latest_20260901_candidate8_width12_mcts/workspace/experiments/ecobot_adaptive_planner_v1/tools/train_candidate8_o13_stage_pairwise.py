#!/usr/bin/env python3
"""Train stage-specific Candidate8 pairwise rankers and downside-risk gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from train_candidate8_day6_safe_uplift import (
    make_features,
    paired_mean_margin_target,
    paired_targets,
    parse_strings,
)


STAGES = {"EARLY": (6,), "MID": (12,), "LATE": (18,)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def auc(target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) == 0 or len(np.unique(target)) < 2:
        return 0.5
    return float(roc_auc_score(target, prediction))


def compare(
    left: int,
    right: int,
    score: np.ndarray,
    lcb: np.ndarray,
    mean: np.ndarray,
) -> int:
    left_key = (float(score[left]), float(lcb[left]), float(mean[left]))
    right_key = (float(score[right]), float(lcb[right]), float(mean[right]))
    return (left_key > right_key) - (left_key < right_key)


def pair_features(
    x: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    lhs = np.asarray(x[left], dtype=np.float32)
    rhs = np.asarray(x[right], dtype=np.float32)
    return np.concatenate([lhs, rhs, lhs - rhs], axis=1)


def build_pair_training(
    states: set[int],
    state_id: np.ndarray,
    x: np.ndarray,
    score: np.ndarray,
    lcb: np.ndarray,
    mean: np.ndarray,
    extreme_count: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    left: list[int] = []
    right: list[int] = []
    labels: list[int] = []
    candidate_keep_pairs = 0
    extreme_pairs = 0
    tied_pairs = 0
    for state in sorted(states):
        indices = np.flatnonzero(state_id == state)
        keep = int(indices[0])
        unordered: set[tuple[int, int]] = set()
        for candidate in indices[1:]:
            candidate = int(candidate)
            if compare(candidate, keep, score, lcb, mean) == 0:
                tied_pairs += 1
                continue
            unordered.add((min(candidate, keep), max(candidate, keep)))
            candidate_keep_pairs += 1
        ordered = sorted(
            (int(index) for index in indices),
            key=lambda index: (
                float(score[index]), float(lcb[index]), float(mean[index]),
                -index,
            ),
        )
        bottom = ordered[:extreme_count]
        top = ordered[-extreme_count:]
        for lower in bottom:
            for upper in top:
                if lower == upper:
                    continue
                if compare(lower, upper, score, lcb, mean) == 0:
                    tied_pairs += 1
                    continue
                pair = (min(lower, upper), max(lower, upper))
                if pair not in unordered:
                    extreme_pairs += 1
                unordered.add(pair)
        for first, second in sorted(unordered):
            relation = compare(first, second, score, lcb, mean)
            if relation == 0:
                continue
            left.extend((first, second))
            right.extend((second, first))
            labels.extend((int(relation > 0), int(relation < 0)))
    left_array = np.asarray(left, dtype=np.int64)
    right_array = np.asarray(right, dtype=np.int64)
    label_array = np.asarray(labels, dtype=np.int8)
    return pair_features(x, left_array, right_array), label_array, {
        "states": len(states),
        "oriented_pairs": int(len(label_array)),
        "candidate_keep_unordered_pairs": int(candidate_keep_pairs),
        "extra_extreme_unordered_pairs": int(extreme_pairs),
        "skipped_tied_pairs": int(tied_pairs),
        "positive_rate": float(np.mean(label_array)) if len(label_array) else 0.0,
    }


def predict_candidate_vs_keep(
    model: lgb.LGBMClassifier,
    x: np.ndarray,
    state_id: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    keep_lookup: dict[int, int] = {
        int(state): int(np.flatnonzero(state_id == state)[0])
        for state in np.unique(state_id[indices])
    }
    keep_indices = np.asarray(
        [keep_lookup[int(state_id[index])] for index in indices],
        dtype=np.int64,
    )
    return model.predict_proba(pair_features(x, indices, keep_indices))[:, 1]


def select(
    states: set[int],
    state_id: np.ndarray,
    p_state: np.ndarray,
    p_pair: np.ndarray,
    p_positive: np.ndarray,
    p_negative: np.ndarray,
    p_score_up: np.ndarray,
    p_score_down: np.ndarray,
    actual_state_effect: np.ndarray,
    actual_action_effect: np.ndarray,
    score: np.ndarray,
    margin_mean: np.ndarray,
    margin_lcb: np.ndarray,
    thresholds: dict,
) -> list[dict]:
    rows: list[dict] = []
    for state in sorted(states):
        indices = np.flatnonzero(state_id == state)
        keep = int(indices[0])
        eligible = [
            int(index) for index in indices[1:]
            if p_state[index] >= thresholds["state_effect_threshold"]
            and p_pair[index] >= thresholds["pair_better_threshold"]
            and p_positive[index] >= thresholds["positive_lcb_threshold"]
            and p_negative[index] <= thresholds["negative_lcb_ceiling"]
            and p_score_up[index] >= thresholds["score_improve_threshold"]
            and p_score_down[index] <= thresholds["score_regress_ceiling"]
        ]
        chosen = keep if not eligible else max(
            eligible,
            key=lambda index: (
                float(p_pair[index]),
                float(p_score_up[index] - p_score_down[index]),
                float(p_positive[index] - p_negative[index]),
                -index,
            ),
        )
        candidate_indices = indices[1:]
        safe_opportunity = np.any(
            actual_state_effect[candidate_indices]
            & (score[candidate_indices] >= 0.0)
            & (margin_lcb[candidate_indices] > 0.0)
        )
        win_opportunity = np.any(
            actual_state_effect[candidate_indices]
            & (score[candidate_indices] > 0.0)
        )
        rows.append({
            "state_id": int(state),
            "activated": chosen != keep,
            "selected_local_rank": int(np.flatnonzero(indices == chosen)[0]),
            "state_effect": bool(actual_state_effect[chosen]),
            "action_effect": bool(actual_action_effect[chosen]),
            "score_delta": float(score[chosen]),
            "margin_mean_delta": float(margin_mean[chosen]),
            "margin_lcb_delta": float(margin_lcb[chosen]),
            "safe_opportunity": bool(safe_opportunity),
            "win_opportunity": bool(win_opportunity),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    active = [row for row in rows if row["activated"]]

    def mean(rows_: list[dict], key: str) -> float:
        return float(np.mean([row[key] for row in rows_])) if rows_ else 0.0

    opportunity = mean(rows, "safe_opportunity")
    return {
        "states": len(rows),
        "safe_opportunity_rate": opportunity,
        "win_opportunity_rate": mean(rows, "win_opportunity"),
        "required_activation_rate": min(0.05, 0.25 * opportunity),
        "activation_rate": len(active) / len(rows) if rows else 0.0,
        "active_state_effect_rate": mean(active, "state_effect"),
        "active_action_effect_rate": mean(active, "action_effect"),
        "mean_score_delta_all": mean(rows, "score_delta"),
        "mean_margin_delta_all": mean(rows, "margin_mean_delta"),
        "mean_margin_lcb_delta_all": mean(rows, "margin_lcb_delta"),
        "mean_score_delta_active": mean(active, "score_delta"),
        "mean_margin_delta_active": mean(active, "margin_mean_delta"),
        "mean_margin_lcb_delta_active": mean(active, "margin_lcb_delta"),
        "active_score_regression_rate": float(np.mean([
            row["score_delta"] < 0.0 for row in active
        ])) if active else 0.0,
        "active_margin_lcb_negative_rate": float(np.mean([
            row["margin_lcb_delta"] < 0.0 for row in active
        ])) if active else 0.0,
    }


def passes(metrics: dict) -> bool:
    if metrics["activation_rate"] < metrics["required_activation_rate"]:
        return False
    if metrics["activation_rate"] == 0.0:
        return metrics["safe_opportunity_rate"] == 0.0
    return (
        metrics["active_state_effect_rate"] >= 0.90
        and metrics["active_score_regression_rate"] <= 0.10
        and metrics["active_margin_lcb_negative_rate"] <= 0.20
        and metrics["mean_score_delta_active"] >= 0.0
        and metrics["mean_margin_delta_active"] > 0.0
        and metrics["mean_margin_lcb_delta_active"] > 0.0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--early-extra-dataset", type=Path)
    parser.add_argument("--effect-model", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--holdout-opponents", type=parse_strings,
        default=parse_strings("G001,G003,G049,G245"),
    )
    parser.add_argument("--holdout-seed-count", type=int, default=2)
    parser.add_argument("--calibration-seed-count", type=int, default=2)
    parser.add_argument("--extreme-count", type=int, default=8)
    parser.add_argument("--z", type=float, default=1.0)
    args = parser.parse_args()

    frozen_effect = joblib.load(args.effect_model)
    def load_dataset(path: Path) -> dict:
        with np.load(path, allow_pickle=False) as data:
            return {
                "state_id": np.asarray(data["state_id"]),
                "opponents": np.asarray(data["opponent"]),
                "seeds": np.asarray(data["prefix_seed"]),
                "day": np.asarray(data["decision_day"]),
                "seat": np.asarray(data["seat"]),
                "family": np.asarray(data["family"]),
                "raw_features": np.asarray(data["features"]),
                "feature_names": [str(value) for value in data["feature_names"]],
                "own": np.asarray(data["future_own_cash"], dtype=np.float64),
                "rival": np.asarray(
                    data["future_opponent_cash"], dtype=np.float64
                ),
                "state_effect": (
                    np.asarray(data["state_change_count_full"]) > 0
                ),
                "action_effect": (
                    np.asarray(data["action_change_count_full"]) > 0
                ),
            }

    base = load_dataset(args.dataset)
    base_rows = len(base["state_id"])
    if args.early_extra_dataset:
        extra = load_dataset(args.early_extra_dataset)
        if extra["feature_names"] != base["feature_names"]:
            raise ValueError("extra EARLY feature contract differs")
        if extra["own"].shape[1] != base["own"].shape[1]:
            raise ValueError("extra EARLY future count differs")
        if set(int(value) for value in np.unique(extra["day"])) != {6}:
            raise ValueError("extra training dataset must contain only Day6")
        offset = int(np.max(base["state_id"])) + 1
        extra["state_id"] = extra["state_id"] + offset
        combined = {}
        for key in (
            "state_id", "opponents", "seeds", "day", "seat", "family",
            "raw_features", "own", "rival", "state_effect", "action_effect",
        ):
            combined[key] = np.concatenate([base[key], extra[key]], axis=0)
    else:
        combined = base
    state_id = combined["state_id"]
    opponents = combined["opponents"]
    seeds = combined["seeds"]
    day = combined["day"]
    seat = combined["seat"]
    family = combined["family"]
    raw_features = combined["raw_features"]
    feature_names = base["feature_names"]
    own = combined["own"]
    rival = combined["rival"]
    state_effect = combined["state_effect"]
    action_effect = combined["action_effect"]
    is_extra = np.arange(len(state_id)) >= base_rows

    x, model_feature_names = make_features(
        raw_features, family, seat, feature_names
    )
    x = np.asarray(x, dtype=np.float32)
    if model_feature_names != list(frozen_effect["feature_names"]):
        raise ValueError("feature contract differs from frozen O1.2 model")
    bank_a = np.arange(0, own.shape[1], 2)
    bank_b = np.arange(1, own.shape[1], 2)
    score_a, lcb_a, _ = paired_targets(
        state_id, own[:, bank_a], rival[:, bank_a], args.z
    )
    score_b, lcb_b, _ = paired_targets(
        state_id, own[:, bank_b], rival[:, bank_b], args.z
    )
    mean_a = paired_mean_margin_target(
        state_id, own[:, bank_a], rival[:, bank_a]
    )
    mean_b = paired_mean_margin_target(
        state_id, own[:, bank_b], rival[:, bank_b]
    )

    unique_seeds = sorted(int(value) for value in np.unique(seeds[~is_extra]))
    test_seeds = set(unique_seeds[-args.holdout_seed_count:])
    calibration_start = -(args.holdout_seed_count + args.calibration_seed_count)
    calibration_seeds = set(
        unique_seeds[calibration_start:-args.holdout_seed_count]
    )
    train_seeds = set(unique_seeds[:calibration_start])
    all_opponents = {str(value) for value in np.unique(opponents)}
    test_opponents = all_opponents & args.holdout_opponents
    train_opponents = all_opponents - test_opponents
    state_meta: dict[int, tuple[str, int, int]] = {}
    for state in np.unique(state_id):
        index = int(np.flatnonzero(state_id == state)[0])
        state_meta[int(state)] = (
            str(opponents[index]), int(seeds[index]), int(day[index])
        )

    def states_for(
        opponent_set: set[str], seed_set: set[int], stage_days: tuple[int, ...]
    ) -> set[int]:
        return {
            state for state, (opponent, seed, state_day) in state_meta.items()
            if opponent in opponent_set and seed in seed_set
            and state_day in stage_days
        }

    partitions_by_stage: dict[str, dict[str, set[int]]] = {}
    for stage, days in STAGES.items():
        partitions_by_stage[stage] = {
            "fit": states_for(train_opponents, train_seeds, days),
            "calibration": states_for(
                train_opponents, calibration_seeds, days
            ),
            "unseen_seeds": states_for(train_opponents, test_seeds, days),
            "unseen_opponents": states_for(
                test_opponents, train_seeds, days
            ),
            "joint_unseen": states_for(test_opponents, test_seeds, days),
        }
    if args.early_extra_dataset:
        extra_states = {
            int(state) for state in np.unique(state_id[is_extra])
            if state_meta[int(state)][0] in train_opponents
        }
        partitions_by_stage["EARLY"]["fit"].update(extra_states)

    non_keep = np.ones(len(state_id), dtype=bool)
    for state in np.unique(state_id):
        non_keep[np.flatnonzero(state_id == state)[0]] = False
    p_state = frozen_effect["state_effect_model"].predict_proba(x)[:, 1]
    stage_models = {}
    stage_predictions = {}
    pair_receipts = {}
    prediction_diagnostics = {}
    common = dict(
        n_estimators=350, learning_rate=0.035, num_leaves=15,
        max_depth=5, min_child_samples=40, subsample=0.85,
        colsample_bytree=0.80, reg_lambda=2.0,
        random_state=20260831, n_jobs=16, verbosity=-1,
    )

    for stage, days in STAGES.items():
        partitions = partitions_by_stage[stage]
        pair_x, pair_y, pair_receipt = build_pair_training(
            partitions["fit"], state_id, x, score_a, lcb_a, mean_a,
            args.extreme_count,
        )
        pair_model = lgb.LGBMClassifier(**common, class_weight="balanced")
        pair_model.fit(pair_x, pair_y)
        fit_mask = (
            np.isin(state_id, list(partitions["fit"])) & non_keep
        )
        fit_indices = np.flatnonzero(fit_mask)
        positive_model = lgb.LGBMClassifier(**common, class_weight="balanced")
        negative_model = lgb.LGBMClassifier(**common, class_weight="balanced")
        score_up_model = lgb.LGBMClassifier(**common, class_weight="balanced")
        score_down_model = lgb.LGBMClassifier(**common, class_weight="balanced")
        positive_model.fit(x[fit_indices], lcb_a[fit_indices] > 0.0)
        negative_model.fit(x[fit_indices], lcb_a[fit_indices] < 0.0)
        score_up_model.fit(x[fit_indices], score_a[fit_indices] > 1e-12)
        score_down_model.fit(x[fit_indices], score_a[fit_indices] < -1e-12)

        stage_indices = np.flatnonzero(np.isin(day, days))
        p_pair = np.zeros(len(state_id), dtype=np.float64)
        p_positive = np.zeros(len(state_id), dtype=np.float64)
        p_negative = np.zeros(len(state_id), dtype=np.float64)
        p_score_up = np.zeros(len(state_id), dtype=np.float64)
        p_score_down = np.zeros(len(state_id), dtype=np.float64)
        p_pair[stage_indices] = predict_candidate_vs_keep(
            pair_model, x, state_id, stage_indices
        )
        p_positive[stage_indices] = positive_model.predict_proba(
            x[stage_indices]
        )[:, 1]
        p_negative[stage_indices] = negative_model.predict_proba(
            x[stage_indices]
        )[:, 1]
        p_score_up[stage_indices] = score_up_model.predict_proba(
            x[stage_indices]
        )[:, 1]
        p_score_down[stage_indices] = score_down_model.predict_proba(
            x[stage_indices]
        )[:, 1]
        stage_models[stage] = {
            "days": days,
            "pair_model": pair_model,
            "positive_lcb_model": positive_model,
            "negative_lcb_model": negative_model,
            "score_improve_model": score_up_model,
            "score_regress_model": score_down_model,
        }
        stage_predictions[stage] = {
            "p_pair": p_pair,
            "p_positive": p_positive,
            "p_negative": p_negative,
            "p_score_up": p_score_up,
            "p_score_down": p_score_down,
        }
        pair_receipts[stage] = pair_receipt
        prediction_diagnostics[stage] = {}
        for name, states in partitions.items():
            indices = np.flatnonzero(
                np.isin(state_id, list(states)) & non_keep
            )
            test_bank = name not in {"fit", "calibration"}
            score_target = score_b if test_bank else score_a
            lcb_target = lcb_b if test_bank else lcb_a
            prediction_diagnostics[stage][name] = {
                "rows": int(len(indices)),
                "state_effect_auc": auc(
                    state_effect[indices], p_state[indices]
                ),
                "positive_lcb_auc": auc(
                    lcb_target[indices] > 0.0, p_positive[indices]
                ),
                "negative_lcb_auc": auc(
                    lcb_target[indices] < 0.0, p_negative[indices]
                ),
                "score_improve_auc": auc(
                    score_target[indices] > 1e-12, p_score_up[indices]
                ),
                "score_regress_auc": auc(
                    score_target[indices] < -1e-12, p_score_down[indices]
                ),
                "pair_better_auc": auc(
                    (
                        (score_target[indices] > 1e-12)
                        | (
                            (np.abs(score_target[indices]) <= 1e-12)
                            & (lcb_target[indices] > 0.0)
                        )
                    ),
                    p_pair[indices],
                ),
            }

    thresholds_by_stage = {}
    threshold_diagnostics = {}
    for stage, partitions in partitions_by_stage.items():
        pred = stage_predictions[stage]
        candidates = []
        pair_grid = (
            (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
            if stage == "EARLY" else (0.50, 0.60, 0.70, 0.80, 0.90)
        )
        positive_grid = (
            (0.20, 0.30, 0.40, 0.55, 0.70)
            if stage == "EARLY" else (0.40, 0.55, 0.70, 0.80, 0.90)
        )
        negative_grid = (
            (0.20, 0.35, 0.50, 0.65, 0.80)
            if stage == "EARLY" else (0.10, 0.20, 0.35, 0.50)
        )
        regress_grid = (
            (0.10, 0.25, 0.40, 0.60, 0.80)
            if stage == "EARLY" else (0.05, 0.10, 0.25, 0.40)
        )
        for state_threshold in (0.85, 0.95, 0.99):
            for pair_threshold in pair_grid:
                for positive_threshold in positive_grid:
                    for negative_ceiling in negative_grid:
                        for improve_threshold in (0.0, 0.05, 0.15, 0.30):
                            for regress_ceiling in regress_grid:
                                thresholds = {
                                    "state_effect_threshold": state_threshold,
                                    "pair_better_threshold": pair_threshold,
                                    "positive_lcb_threshold": positive_threshold,
                                    "negative_lcb_ceiling": negative_ceiling,
                                    "score_improve_threshold": improve_threshold,
                                    "score_regress_ceiling": regress_ceiling,
                                }
                                rows = select(
                                    partitions["calibration"], state_id,
                                    p_state, pred["p_pair"],
                                    pred["p_positive"], pred["p_negative"],
                                    pred["p_score_up"], pred["p_score_down"],
                                    state_effect, action_effect, score_a,
                                    mean_a, lcb_a, thresholds,
                                )
                                metrics = summarize(rows)
                                candidates.append({
                                    "thresholds": thresholds,
                                    "metrics": metrics,
                                    "passed": passes(metrics),
                                })
        eligible = [item for item in candidates if item["passed"]]
        structurally_valid = [
            item for item in candidates
            if item["metrics"]["activation_rate"]
            >= item["metrics"]["required_activation_rate"]
            and item["metrics"]["active_state_effect_rate"] >= 0.90
        ]
        pool = eligible if eligible else structurally_valid
        if pool:
            chosen = max(pool, key=lambda item: (
                item["metrics"]["mean_score_delta_all"],
                item["metrics"]["mean_margin_lcb_delta_all"],
                item["metrics"]["mean_margin_delta_all"],
            ))
            chosen["research_only"] = not bool(eligible)
        else:
            chosen = {
                "thresholds": {
                    "state_effect_threshold": float("inf"),
                    "pair_better_threshold": float("inf"),
                    "positive_lcb_threshold": float("inf"),
                    "negative_lcb_ceiling": float("-inf"),
                    "score_improve_threshold": float("inf"),
                    "score_regress_ceiling": float("-inf"),
                },
                "metrics": summarize([]),
                "passed": False,
                "research_only": True,
            }
        thresholds_by_stage[stage] = chosen
        threshold_diagnostics[stage] = {
            "candidate_count": len(candidates),
            "passed_count": len(eligible),
            "structurally_valid_count": len(structurally_valid),
            "best_activation_candidate": max(
                [
                    item for item in candidates
                    if item["metrics"]["activation_rate"] > 0.0
                ],
                key=lambda item: (
                    item["metrics"]["activation_rate"],
                    item["metrics"]["active_state_effect_rate"],
                ),
                default=None,
            ),
            "best_nonempty_safety_candidate": max(
                [
                    item for item in candidates
                    if item["metrics"]["activation_rate"] > 0.0
                ],
                key=lambda item: (
                    item["metrics"]["active_state_effect_rate"],
                    -item["metrics"]["active_score_regression_rate"],
                    -item["metrics"]["active_margin_lcb_negative_rate"],
                    item["metrics"]["mean_margin_lcb_delta_all"],
                ),
                default=None,
            ),
            "selected": chosen,
        }

    metrics = {}
    selected_rows = {}
    for partition_name in (
        "fit", "calibration", "unseen_seeds", "unseen_opponents",
        "joint_unseen",
    ):
        combined_rows = []
        metrics[partition_name] = {"by_stage": {}}
        selected_rows[partition_name] = []
        for stage, partitions in partitions_by_stage.items():
            pred = stage_predictions[stage]
            test_bank = partition_name not in {"fit", "calibration"}
            rows = select(
                partitions[partition_name], state_id, p_state,
                pred["p_pair"], pred["p_positive"], pred["p_negative"],
                pred["p_score_up"], pred["p_score_down"], state_effect,
                action_effect, score_b if test_bank else score_a,
                mean_b if test_bank else mean_a,
                lcb_b if test_bank else lcb_a,
                thresholds_by_stage[stage]["thresholds"],
            )
            metrics[partition_name]["by_stage"][stage] = summarize(rows)
            for row in rows:
                row["stage"] = stage
            combined_rows.extend(rows)
        metrics[partition_name]["overall"] = summarize(combined_rows)
        selected_rows[partition_name] = combined_rows

    calibration_stage_pass = all(
        thresholds_by_stage[stage]["passed"] for stage in STAGES
    )
    internal_pass = all(
        passes(metrics[name]["overall"])
        for name in ("unseen_seeds", "unseen_opponents", "joint_unseen")
    )
    deployment_gate = calibration_stage_pass and internal_pass
    model_payload = {
        "schema": "kaggriculture.candidate8-o13-stage-pairwise-model.v1",
        "state_effect_model": frozen_effect["state_effect_model"],
        "stage_models": stage_models,
        "thresholds_by_stage": thresholds_by_stage,
        "feature_names": model_feature_names,
        "pair_feature_names": (
            [f"lhs:{name}" for name in model_feature_names]
            + [f"rhs:{name}" for name in model_feature_names]
            + [f"delta:{name}" for name in model_feature_names]
        ),
        "z": args.z,
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, args.model_output)
    payload = {
        "schema": "kaggriculture.candidate8-o13-stage-pairwise.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "O1.2 state-effect filter is frozen. EARLY/MID/LATE models use "
            "public features only. Pairwise training contains candidate-vs-"
            "KEEP and deterministic top-vs-bottom within-state comparisons."
        ),
        "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
        "early_extra_dataset": (
            {
                "path": str(args.early_extra_dataset),
                "sha256": sha256(args.early_extra_dataset),
                "rows": int(np.sum(is_extra)),
                "states": int(len(np.unique(state_id[is_extra]))),
                "fit_only": True,
            }
            if args.early_extra_dataset else None
        ),
        "effect_model": {
            "path": str(args.effect_model), "sha256": sha256(args.effect_model)
        },
        "model": {"path": str(args.model_output), "sha256": sha256(args.model_output)},
        "split": {
            "train_opponents": sorted(train_opponents),
            "test_opponents": sorted(test_opponents),
            "train_seeds": sorted(train_seeds),
            "calibration_seeds": sorted(calibration_seeds),
            "test_seeds": sorted(test_seeds),
            "bank_a_columns": bank_a.tolist(),
            "bank_b_columns": bank_b.tolist(),
        },
        "pair_training": pair_receipts,
        "prediction_diagnostics": prediction_diagnostics,
        "threshold_search": threshold_diagnostics,
        "thresholds_by_stage": thresholds_by_stage,
        "metrics": metrics,
        "selected_rows": selected_rows,
        "gates": {
            "all_stage_calibration_passed": calibration_stage_pass,
            "all_internal_unseen_passed": internal_pass,
            "deployment_gate_passed": deployment_gate,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "pair_training": pair_receipts,
        "thresholds_by_stage": thresholds_by_stage,
        "prediction_diagnostics": prediction_diagnostics,
        "metrics": metrics,
        "gates": payload["gates"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
