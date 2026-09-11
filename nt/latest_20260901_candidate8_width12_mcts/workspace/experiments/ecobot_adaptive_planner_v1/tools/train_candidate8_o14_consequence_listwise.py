#!/usr/bin/env python3
"""Train O1.4 stage-wise listwise Candidate8 selectors with consequences."""

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

from candidate8_consequence_features import CONTRACT, compile_consequence_features
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


def load_dataset(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        result = {
            "state_id": np.asarray(data["state_id"]),
            "opponents": np.asarray(data["opponent"]),
            "seeds": np.asarray(data["prefix_seed"]),
            "day": np.asarray(data["decision_day"]),
            "seat": np.asarray(data["seat"]),
            "family": np.asarray(data["family"]),
            "raw_features": np.asarray(data["features"]),
            "feature_names": [str(value) for value in data["feature_names"]],
            "own": np.asarray(data["future_own_cash"], dtype=np.float64),
            "rival": np.asarray(data["future_opponent_cash"], dtype=np.float64),
            "state_effect": np.asarray(data["state_change_count_full"]) > 0,
            "action_effect": np.asarray(data["action_change_count_full"]) > 0,
        }
        if "consequence_features" in data.files:
            result["exact_consequence"] = np.asarray(
                data["consequence_features"], dtype=np.float32
            )
            result["exact_consequence_names"] = [
                str(value) for value in data["consequence_feature_names"]
            ]
        if "response_scenario_names" in data.files:
            result["response_scenario_names"] = [
                str(value) for value in data["response_scenario_names"]
            ]
            result["response_scenario_outcomes"] = np.asarray(
                data["response_scenario_outcomes"], dtype=np.float32
            )
            result["response_scenario_outcome_names"] = [
                str(value) for value in data["response_scenario_outcome_names"]
            ]
        return result


def combine(base: dict, extra_path: Path | None) -> tuple[dict, np.ndarray, int]:
    base_rows = len(base["state_id"])
    if extra_path is None:
        return base, np.zeros(base_rows, dtype=bool), base_rows
    extra = load_dataset(extra_path)
    if extra["feature_names"] != base["feature_names"]:
        raise ValueError("extra EARLY feature contract differs")
    if extra["own"].shape[1] != base["own"].shape[1]:
        raise ValueError("extra EARLY future count differs")
    if set(int(value) for value in np.unique(extra["day"])) != {6}:
        raise ValueError("extra training dataset must contain only Day6")
    extra["state_id"] = extra["state_id"] + int(np.max(base["state_id"])) + 1
    merged = {"feature_names": base["feature_names"]}
    for key in (
        "state_id", "opponents", "seeds", "day", "seat", "family",
        "raw_features", "own", "rival", "state_effect", "action_effect",
    ):
        merged[key] = np.concatenate([base[key], extra[key]], axis=0)
    if "exact_consequence" in base or "exact_consequence" in extra:
        if "exact_consequence" not in base or "exact_consequence" not in extra:
            raise ValueError("both base and EARLY extra must carry exact consequences")
        if base["exact_consequence_names"] != extra["exact_consequence_names"]:
            raise ValueError("exact consequence contracts differ")
        merged["exact_consequence"] = np.concatenate(
            [base["exact_consequence"], extra["exact_consequence"]], axis=0
        )
        merged["exact_consequence_names"] = base["exact_consequence_names"]
    if "response_scenario_names" in base or "response_scenario_names" in extra:
        if base.get("response_scenario_names") != extra.get("response_scenario_names"):
            raise ValueError("response scenario contracts differ")
        merged["response_scenario_names"] = base["response_scenario_names"]
        if (
            base.get("response_scenario_outcome_names")
            != extra.get("response_scenario_outcome_names")
        ):
            raise ValueError("response scenario outcome contracts differ")
        merged["response_scenario_outcome_names"] = base[
            "response_scenario_outcome_names"
        ]
        merged["response_scenario_outcomes"] = np.concatenate([
            base["response_scenario_outcomes"],
            extra["response_scenario_outcomes"],
        ], axis=0)
    is_extra = np.arange(len(merged["state_id"])) >= base_rows
    return merged, is_extra, base_rows


def relevance_labels(
    states: set[int], state_id: np.ndarray, score: np.ndarray,
    lcb: np.ndarray, mean: np.ndarray, maximum_grade: int = 15,
) -> np.ndarray:
    labels = np.zeros(len(state_id), dtype=np.int32)
    for state in states:
        indices = np.flatnonzero(state_id == state)
        keys = [(float(score[i]), float(lcb[i]), float(mean[i])) for i in indices]
        unique = sorted(set(keys))
        if len(unique) == 1:
            continue
        grade = {
            key: int(round(rank * maximum_grade / (len(unique) - 1)))
            for rank, key in enumerate(unique)
        }
        labels[indices] = [grade[key] for key in keys]
    return labels


def response_teacher_relevance_labels(
    states: set[int], state_id: np.ndarray, outcomes: np.ndarray,
    maximum_grade: int = 15,
) -> np.ndarray:
    """Rank arms by fixed public-response scenarios, never true futures."""
    if outcomes.ndim != 3 or outcomes.shape[1:] != (5, 9):
        raise ValueError(f"unexpected response teacher shape {outcomes.shape}")
    labels = np.zeros(len(state_id), dtype=np.int32)
    terminal_margin = outcomes[:, :, 8]
    for state in states:
        indices = np.flatnonzero(state_id == state)
        keep_margin = terminal_margin[indices[0]]
        keys = []
        for index in indices:
            margin = terminal_margin[index]
            delta = margin - keep_margin
            score = float(np.mean(np.where(
                margin > 0, 1.0, np.where(margin == 0, 0.5, 0.0)
            )))
            keys.append((
                score,
                float(np.quantile(delta, 0.25)),
                float(np.min(delta)),
                float(np.mean(delta)),
                -float(np.std(delta)),
            ))
        unique = sorted(set(keys))
        if len(unique) == 1:
            continue
        grade = {
            key: int(round(rank * maximum_grade / (len(unique) - 1)))
            for rank, key in enumerate(unique)
        }
        labels[indices] = [grade[key] for key in keys]
    return labels


def rank_training_rows(
    states: set[int], state_id: np.ndarray, x: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    indices = np.flatnonzero(np.isin(state_id, list(states)))
    order = np.lexsort((indices, state_id[indices]))
    indices = indices[order]
    groups = [int(np.sum(state_id[indices] == state)) for state in sorted(states)]
    weights = np.asarray([
        1.0 / max(1, np.sum(state_id[indices] == state_id[index]))
        for index in indices
    ], dtype=np.float32)
    return x[indices], labels[indices], groups, weights


def predict_probability(model: lgb.LGBMClassifier, x: np.ndarray) -> np.ndarray:
    values = model.predict_proba(x)
    if values.shape[1] == 1:
        return np.full(len(x), float(model.classes_[0] == 1), dtype=np.float64)
    return values[:, int(np.flatnonzero(model.classes_ == 1)[0])]


def rank_advantage(
    rank_score: np.ndarray, state_id: np.ndarray, indices: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(state_id), dtype=np.float64)
    for state in np.unique(state_id[indices]):
        rows = np.flatnonzero(state_id == state)
        result[rows] = rank_score[rows] - rank_score[rows[0]]
    return result


def select(
    states: set[int], state_id: np.ndarray, rank_score: np.ndarray,
    rank_delta: np.ndarray, p_state: np.ndarray, p_positive: np.ndarray,
    p_negative: np.ndarray, p_score_up: np.ndarray, p_score_down: np.ndarray,
    actual_state_effect: np.ndarray, actual_action_effect: np.ndarray,
    score: np.ndarray, margin_mean: np.ndarray, margin_lcb: np.ndarray,
    thresholds: dict,
) -> list[dict]:
    rows: list[dict] = []
    for state in sorted(states):
        indices = np.flatnonzero(state_id == state)
        keep = int(indices[0])
        eligible = [
            int(index) for index in indices[1:]
            if p_state[index] >= thresholds["state_effect_threshold"]
            and rank_delta[index] >= thresholds["rank_advantage_threshold"]
            and p_positive[index] >= thresholds["positive_lcb_threshold"]
            and p_negative[index] <= thresholds["negative_lcb_ceiling"]
            and p_score_up[index] >= thresholds["score_improve_threshold"]
            and p_score_down[index] <= thresholds["score_regress_ceiling"]
        ]
        chosen = keep if not eligible else max(
            eligible,
            key=lambda index: (
                float(rank_score[index]),
                float(p_score_up[index] - p_score_down[index]),
                float(p_positive[index] - p_negative[index]),
                -index,
            ),
        )
        candidates = indices[1:]
        rows.append({
            "state_id": int(state),
            "activated": chosen != keep,
            "selected_local_rank": int(np.flatnonzero(indices == chosen)[0]),
            "state_effect": bool(actual_state_effect[chosen]),
            "action_effect": bool(actual_action_effect[chosen]),
            "score_delta": float(score[chosen]),
            "margin_mean_delta": float(margin_mean[chosen]),
            "margin_lcb_delta": float(margin_lcb[chosen]),
            "safe_opportunity": bool(np.any(
                actual_state_effect[candidates]
                & (score[candidates] >= 0.0)
                & (margin_lcb[candidates] > 0.0)
            )),
            "win_opportunity": bool(np.any(
                actual_state_effect[candidates] & (score[candidates] > 0.0)
            )),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    active = [row for row in rows if row["activated"]]

    def mean(source: list[dict], key: str) -> float:
        return float(np.mean([row[key] for row in source])) if source else 0.0

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


def top1_diagnostics(
    states: set[int], state_id: np.ndarray, rank_score: np.ndarray,
    score: np.ndarray, lcb: np.ndarray, mean: np.ndarray,
) -> dict:
    exact = 0
    score_regress = 0
    negative_lcb = 0
    regrets = []
    for state in states:
        indices = np.flatnonzero(state_id == state)
        oracle = max(indices, key=lambda i: (score[i], lcb[i], mean[i], -i))
        chosen = max(indices, key=lambda i: (rank_score[i], -i))
        exact += int(chosen == oracle)
        score_regress += int(score[chosen] < 0.0)
        negative_lcb += int(lcb[chosen] < 0.0)
        regrets.append(float(mean[oracle] - mean[chosen]))
    count = len(states)
    return {
        "states": count,
        "exact_oracle_top1_rate": exact / count if count else 0.0,
        "top1_score_regression_rate": score_regress / count if count else 0.0,
        "top1_negative_lcb_rate": negative_lcb / count if count else 0.0,
        "mean_margin_regret": float(np.mean(regrets)) if regrets else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--early-extra-dataset", type=Path)
    parser.add_argument("--effect-model", required=True, type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--holdout-opponents", type=parse_strings,
        default=parse_strings("G001,G003,G049,G245"),
    )
    parser.add_argument("--holdout-seed-count", type=int, default=2)
    parser.add_argument("--calibration-seed-count", type=int, default=2)
    parser.add_argument("--z", type=float, default=1.0)
    parser.add_argument(
        "--only-variant",
        help=(
            "Optional research speed gate. Train exactly one named variant "
            "instead of every ablation; the frozen deployment variant remains "
            "unchanged and the name must match it."
        ),
    )
    args = parser.parse_args()

    frozen_effect = joblib.load(args.effect_model)
    base = load_dataset(args.dataset)
    data, is_extra, base_rows = combine(base, args.early_extra_dataset)
    state_id = data["state_id"]
    opponents = data["opponents"]
    seeds = data["seeds"]
    day = data["day"]
    seat = data["seat"]
    family = data["family"]
    raw = data["raw_features"]
    own = data["own"]
    rival = data["rival"]
    state_effect = data["state_effect"]
    action_effect = data["action_effect"]
    source_names = base["feature_names"]
    response_scenario_names = base.get("response_scenario_names")
    response_scenario_outcomes = data.get("response_scenario_outcomes")

    raw_x, raw_model_names = make_features(raw, family, seat, source_names)
    raw_x = np.asarray(raw_x, dtype=np.float32)
    if raw_model_names != list(frozen_effect["feature_names"]):
        raise ValueError("O1.2 state-effect feature contract differs")
    consequence, consequence_names = compile_consequence_features(raw, source_names)
    augmented_x = np.concatenate([raw_x, consequence], axis=1).astype(np.float32)
    augmented_names = [*raw_model_names, *consequence_names]
    exact_consequence = data.get("exact_consequence")
    exact_augmented_x = None
    exact_augmented_names = None
    exact_delta_x = None
    exact_delta_names = None
    if exact_consequence is not None:
        exact_augmented_x = np.concatenate(
            [raw_x, exact_consequence], axis=1
        ).astype(np.float32)
        exact_augmented_names = [
            *raw_model_names, *data["exact_consequence_names"]
        ]
        exact_delta = np.empty_like(exact_consequence, dtype=np.float32)
        for state in np.unique(state_id):
            rows = np.flatnonzero(state_id == state)
            exact_delta[rows] = exact_consequence[rows] - exact_consequence[rows[0]]
        exact_delta_x = np.concatenate([raw_x, exact_delta], axis=1).astype(
            np.float32
        )
        exact_delta_names = [
            *raw_model_names,
            *[f"delta_to_keep_{name}" for name in data["exact_consequence_names"]],
        ]

    bank_a = np.arange(0, own.shape[1], 2)
    bank_b = np.arange(1, own.shape[1], 2)
    score_a, lcb_a, _ = paired_targets(
        state_id, own[:, bank_a], rival[:, bank_a], args.z
    )
    score_b, lcb_b, _ = paired_targets(
        state_id, own[:, bank_b], rival[:, bank_b], args.z
    )
    mean_a = paired_mean_margin_target(state_id, own[:, bank_a], rival[:, bank_a])
    mean_b = paired_mean_margin_target(state_id, own[:, bank_b], rival[:, bank_b])

    unique_seeds = sorted(int(value) for value in np.unique(seeds[~is_extra]))
    test_seeds = set(unique_seeds[-args.holdout_seed_count:])
    cal_start = -(args.holdout_seed_count + args.calibration_seed_count)
    calibration_seeds = set(unique_seeds[cal_start:-args.holdout_seed_count])
    train_seeds = set(unique_seeds[:cal_start])
    all_opponents = {str(value) for value in np.unique(opponents)}
    test_opponents = all_opponents & args.holdout_opponents
    train_opponents = all_opponents - test_opponents
    state_meta = {}
    for state in np.unique(state_id):
        index = int(np.flatnonzero(state_id == state)[0])
        state_meta[int(state)] = (
            str(opponents[index]), int(seeds[index]), int(day[index])
        )

    def states_for(opponent_set: set[str], seed_set: set[int], days: tuple[int, ...]):
        return {
            state for state, (opponent, seed, state_day) in state_meta.items()
            if opponent in opponent_set and seed in seed_set and state_day in days
        }

    partitions_by_stage = {}
    for stage, days in STAGES.items():
        partitions_by_stage[stage] = {
            "fit": states_for(train_opponents, train_seeds, days),
            "calibration": states_for(train_opponents, calibration_seeds, days),
            "unseen_seeds": states_for(train_opponents, test_seeds, days),
            "unseen_opponents": states_for(test_opponents, train_seeds, days),
            "joint_unseen": states_for(test_opponents, test_seeds, days),
        }
    if args.early_extra_dataset:
        partitions_by_stage["EARLY"]["fit"].update(
            int(state) for state in np.unique(state_id[is_extra])
            if state_meta[int(state)][0] in train_opponents
        )

    non_keep = np.ones(len(state_id), dtype=bool)
    for state in np.unique(state_id):
        non_keep[np.flatnonzero(state_id == state)[0]] = False
    p_state = predict_probability(frozen_effect["state_effect_model"], raw_x)

    variants = {
        "RAW_LISTWISE": (raw_x, raw_model_names),
        "CONSEQUENCE_LISTWISE": (augmented_x, augmented_names),
    }
    if exact_augmented_x is not None:
        variants["EXACT_PREVIEW_LISTWISE"] = (
            exact_augmented_x, exact_augmented_names
        )
        # Same consequence inputs, but rank candidates directly by predicted
        # competitive upside and downside instead of asking LambdaRank to
        # reconstruct the entire within-state ordering. This is a prespecified
        # ablation motivated by the O1.3/O1.4 class imbalance: the practical
        # question is candidate versus KEEP, not exact rank 1..64.
        variants["EXACT_PREVIEW_UTILITY"] = (
            exact_augmented_x, exact_augmented_names
        )
        variants["EXACT_PREVIEW_DELTA_LISTWISE"] = (
            exact_delta_x, exact_delta_names
        )
        variants["EXACT_PREVIEW_DELTA_UTILITY"] = (
            exact_delta_x, exact_delta_names
        )
    if response_scenario_outcomes is not None:
        # Deployable student: scenario rollouts create listwise teacher labels
        # during training, but inference receives only public raw features.
        variants["RESPONSE_DISTILLED_LISTWISE"] = (
            raw_x, raw_model_names
        )
    if args.only_variant:
        if args.only_variant not in variants:
            raise ValueError(
                f"unknown --only-variant {args.only_variant}; "
                f"available={sorted(variants)}"
            )
        variants = {args.only_variant: variants[args.only_variant]}
    common_ranker = dict(
        objective="lambdarank", metric="ndcg", n_estimators=450,
        learning_rate=0.03, num_leaves=15, max_depth=5,
        min_child_samples=30, subsample=0.85, colsample_bytree=0.75,
        reg_lambda=3.0, random_state=20260831, n_jobs=16, verbosity=-1,
    )
    common_classifier = dict(
        n_estimators=350, learning_rate=0.035, num_leaves=15,
        max_depth=5, min_child_samples=40, subsample=0.85,
        colsample_bytree=0.75, reg_lambda=3.0, random_state=20260831,
        n_jobs=16, verbosity=-1,
    )

    trained = {}
    ablation = {}
    for variant, (x, names) in variants.items():
        trained[variant] = {}
        ablation[variant] = {}
        for stage, days in STAGES.items():
            partitions = partitions_by_stage[stage]
            labels = (
                response_teacher_relevance_labels(
                    partitions["fit"], state_id, response_scenario_outcomes
                )
                if variant == "RESPONSE_DISTILLED_LISTWISE"
                else relevance_labels(
                    partitions["fit"], state_id, score_a, lcb_a, mean_a
                )
            )
            train_x, train_y, groups, weights = rank_training_rows(
                partitions["fit"], state_id, x, labels
            )
            ranker = lgb.LGBMRanker(**common_ranker)
            ranker.fit(train_x, train_y, group=groups, sample_weight=weights)
            fit_indices = np.flatnonzero(
                np.isin(state_id, list(partitions["fit"])) & non_keep
            )
            positive = lgb.LGBMClassifier(
                **common_classifier, class_weight="balanced"
            ).fit(x[fit_indices], lcb_a[fit_indices] > 0.0)
            negative = lgb.LGBMClassifier(
                **common_classifier, class_weight="balanced"
            ).fit(x[fit_indices], lcb_a[fit_indices] < 0.0)
            score_up = lgb.LGBMClassifier(
                **common_classifier, class_weight="balanced"
            ).fit(x[fit_indices], score_a[fit_indices] > 1e-12)
            score_down = lgb.LGBMClassifier(
                **common_classifier, class_weight="balanced"
            ).fit(x[fit_indices], score_a[fit_indices] < -1e-12)
            stage_indices = np.flatnonzero(np.isin(day, days))
            rank_score = np.zeros(len(state_id), dtype=np.float64)
            rank_score[stage_indices] = ranker.predict(x[stage_indices])
            rank_delta = rank_advantage(rank_score, state_id, stage_indices)
            predictions = {
                "rank_score": rank_score,
                "rank_delta": rank_delta,
                "p_positive": np.zeros(len(state_id)),
                "p_negative": np.zeros(len(state_id)),
                "p_score_up": np.zeros(len(state_id)),
                "p_score_down": np.zeros(len(state_id)),
            }
            predictions["p_positive"][stage_indices] = predict_probability(
                positive, x[stage_indices]
            )
            predictions["p_negative"][stage_indices] = predict_probability(
                negative, x[stage_indices]
            )
            predictions["p_score_up"][stage_indices] = predict_probability(
                score_up, x[stage_indices]
            )
            predictions["p_score_down"][stage_indices] = predict_probability(
                score_down, x[stage_indices]
            )
            if variant in {
                "EXACT_PREVIEW_UTILITY", "EXACT_PREVIEW_DELTA_UTILITY"
            }:
                rank_score[stage_indices] = (
                    4.0 * (
                        predictions["p_score_up"][stage_indices]
                        - predictions["p_score_down"][stage_indices]
                    )
                    + predictions["p_positive"][stage_indices]
                    - predictions["p_negative"][stage_indices]
                )
                rank_delta = rank_advantage(
                    rank_score, state_id, stage_indices
                )
                predictions["rank_score"] = rank_score
                predictions["rank_delta"] = rank_delta

            threshold_candidates = []
            cal_nonkeep = np.flatnonzero(
                np.isin(state_id, list(partitions["calibration"])) & non_keep
            )
            positive_rank = rank_delta[cal_nonkeep]
            rank_grid = sorted(set([
                0.0,
                *[max(0.0, float(np.quantile(positive_rank, q)))
                  for q in (0.25, 0.50, 0.70, 0.85, 0.95)],
            ]))
            for state_threshold in (0.85, 0.95, 0.99):
                for rank_threshold in rank_grid:
                    for positive_threshold in (0.0, 0.25, 0.45, 0.65):
                        for negative_ceiling in (0.15, 0.30, 0.50, 0.70):
                            for improve_threshold in (0.0, 0.05, 0.15):
                                for regress_ceiling in (0.05, 0.15, 0.30, 0.50):
                                    thresholds = {
                                        "state_effect_threshold": state_threshold,
                                        "rank_advantage_threshold": rank_threshold,
                                        "positive_lcb_threshold": positive_threshold,
                                        "negative_lcb_ceiling": negative_ceiling,
                                        "score_improve_threshold": improve_threshold,
                                        "score_regress_ceiling": regress_ceiling,
                                    }
                                    rows = select(
                                        partitions["calibration"], state_id,
                                        rank_score, rank_delta, p_state,
                                        predictions["p_positive"],
                                        predictions["p_negative"],
                                        predictions["p_score_up"],
                                        predictions["p_score_down"],
                                        state_effect, action_effect, score_a,
                                        mean_a, lcb_a, thresholds,
                                    )
                                    metrics = summarize(rows)
                                    threshold_candidates.append({
                                        "thresholds": thresholds,
                                        "metrics": metrics,
                                        "passed": passes(metrics),
                                    })
            passing = [row for row in threshold_candidates if row["passed"]]
            structural = [
                row for row in threshold_candidates
                if row["metrics"]["activation_rate"]
                >= row["metrics"]["required_activation_rate"]
                and row["metrics"]["active_state_effect_rate"] >= 0.90
            ]
            pool = passing if passing else structural
            if pool:
                chosen = max(pool, key=lambda row: (
                    row["metrics"]["mean_score_delta_all"],
                    row["metrics"]["mean_margin_lcb_delta_all"],
                    row["metrics"]["mean_margin_delta_all"],
                    -row["metrics"]["active_score_regression_rate"],
                ))
                chosen["research_only"] = not bool(passing)
            else:
                chosen = {
                    "thresholds": {
                        "state_effect_threshold": float("inf"),
                        "rank_advantage_threshold": float("inf"),
                        "positive_lcb_threshold": float("inf"),
                        "negative_lcb_ceiling": float("-inf"),
                        "score_improve_threshold": float("inf"),
                        "score_regress_ceiling": float("-inf"),
                    },
                    "metrics": summarize([]), "passed": False,
                    "research_only": True,
                }

            metrics = {}
            diagnostics = {}
            selected = {}
            for partition_name, states in partitions.items():
                use_b = partition_name not in {"fit", "calibration"}
                target_score = score_b if use_b else score_a
                target_lcb = lcb_b if use_b else lcb_a
                target_mean = mean_b if use_b else mean_a
                rows = select(
                    states, state_id, rank_score, rank_delta, p_state,
                    predictions["p_positive"], predictions["p_negative"],
                    predictions["p_score_up"], predictions["p_score_down"],
                    state_effect, action_effect, target_score, target_mean,
                    target_lcb, chosen["thresholds"],
                )
                selected[partition_name] = rows
                metrics[partition_name] = summarize(rows)
                indices = np.flatnonzero(
                    np.isin(state_id, list(states)) & non_keep
                )
                diagnostics[partition_name] = {
                    "rows": int(len(indices)),
                    "rank_better_than_keep_auc": auc(
                        (
                            (target_score[indices] > 1e-12)
                            | ((np.abs(target_score[indices]) <= 1e-12)
                               & (target_lcb[indices] > 0.0))
                        ),
                        rank_delta[indices],
                    ),
                    "positive_lcb_auc": auc(
                        target_lcb[indices] > 0.0,
                        predictions["p_positive"][indices],
                    ),
                    "negative_lcb_auc": auc(
                        target_lcb[indices] < 0.0,
                        predictions["p_negative"][indices],
                    ),
                    "score_improve_auc": auc(
                        target_score[indices] > 1e-12,
                        predictions["p_score_up"][indices],
                    ),
                    "score_regress_auc": auc(
                        target_score[indices] < -1e-12,
                        predictions["p_score_down"][indices],
                    ),
                    "ungated_top1": top1_diagnostics(
                        states, state_id, rank_score, target_score,
                        target_lcb, target_mean,
                    ),
                }

            trained[variant][stage] = {
                "days": days,
                "ranker": ranker,
                "positive_lcb_model": positive,
                "negative_lcb_model": negative,
                "score_improve_model": score_up,
                "score_regress_model": score_down,
                "thresholds": chosen,
                "feature_names": names,
            }
            ablation[variant][stage] = {
                "threshold_search": {
                    "candidate_count": len(threshold_candidates),
                    "passed_count": len(passing),
                    "structurally_valid_count": len(structural),
                    "rank_grid": rank_grid,
                    "selected": chosen,
                },
                "metrics": metrics,
                "prediction_diagnostics": diagnostics,
                "selected_rows": selected,
            }

    # O1.5 must freeze the public-only distilled student. Exact response
    # rollouts are an offline teacher and cannot be recomputed in the Kaggle
    # Python one-second action budget. O1.4 keeps its prior frozen variant.
    frozen_variant = (
        "RESPONSE_DISTILLED_LISTWISE"
        if response_scenario_outcomes is not None else (
            "EXACT_PREVIEW_DELTA_UTILITY"
            if exact_augmented_x is not None else "CONSEQUENCE_LISTWISE"
        )
    )
    if args.only_variant and args.only_variant != frozen_variant:
        raise ValueError(
            "--only-variant must equal the frozen variant "
            f"{frozen_variant}, got {args.only_variant}"
        )
    frozen_stages = trained[frozen_variant]
    calibration_pass = all(
        frozen_stages[stage]["thresholds"]["passed"] for stage in STAGES
    )
    internal_pass = all(
        passes(ablation[frozen_variant][stage]["metrics"][partition])
        for stage in STAGES
        for partition in ("unseen_seeds", "unseen_opponents", "joint_unseen")
    )
    exact_contract = None
    if exact_augmented_x is not None:
        exact_contract = (
            {
                "kind": "public_response_scenario_ensemble",
                "horizons": [24, 48, "terminal"],
                "width": int(exact_consequence.shape[1]),
                "future_seed": "fixed synthetic contract constant",
                "scenarios": response_scenario_names,
                "opponent_private_state": "public-history belief replacement",
                "true_future_information_used": False,
            }
            if response_scenario_names else {
                "kind": "canonical_executor_preview",
                "horizons": [24, 48, "terminal"],
                "width": int(exact_consequence.shape[1]),
                "future_seed": "fixed contract constant",
                "opponent": "PASS",
                "true_future_information_used": False,
            }
        )
    model_uses_response_rollouts = (
        frozen_variant != "RESPONSE_DISTILLED_LISTWISE"
        and response_scenario_names is not None
    )
    payload_model = {
        "schema": "kaggriculture.candidate8-o14-consequence-listwise-model.v1",
        "variant": frozen_variant,
        "state_effect_model": frozen_effect["state_effect_model"],
        "state_effect_feature_names": raw_model_names,
        "stage_models": frozen_stages,
        "feature_names": frozen_stages["EARLY"]["feature_names"],
        "source_feature_names": source_names,
        "consequence_feature_names": (
            [] if frozen_variant == "RESPONSE_DISTILLED_LISTWISE" else (
                data["exact_consequence_names"]
                if exact_augmented_x is not None else consequence_names
            )
        ),
        "consequence_contract": (
            {
                "kind": "distilled_public_only",
                "teacher": "public_response_scenario_ensemble",
                "scenarios": response_scenario_names,
                "runtime_rollouts_required": False,
                "true_future_information_used": False,
            }
            if frozen_variant == "RESPONSE_DISTILLED_LISTWISE" else (
                exact_contract if exact_augmented_x is not None else CONTRACT.__dict__
            )
        ),
        "z": args.z,
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload_model, args.model_output)

    receipt = {
        "schema": (
            "kaggriculture.candidate8-o15-response-listwise.v1"
            if response_scenario_names
            else "kaggriculture.candidate8-o14-consequence-listwise.v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "O1.5 response features use five fixed public-history rival "
            "scenarios at 24/48/terminal horizons with a fixed synthetic "
            "future. They never use route identity, hidden rival inventory or "
            "the true future label bank. Thresholds use Bank A calibration; "
            "unseen sets use independent Bank B labels."
            if response_scenario_names else
            "When present, exact preview features use 24/48/terminal canonical "
            "executor previews with a fixed synthetic future and PASS opponent. "
            "They never use the true future label bank. Thresholds use Bank A "
            "calibration; unseen sets use independent Bank B labels."
        ),
        "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
        "early_extra_dataset": (
            {"path": str(args.early_extra_dataset),
             "sha256": sha256(args.early_extra_dataset),
             "rows": int(np.sum(is_extra)), "fit_only": True}
            if args.early_extra_dataset else None
        ),
        "effect_model": {
            "path": str(args.effect_model), "sha256": sha256(args.effect_model)
        },
        "model": {"path": str(args.model_output), "sha256": sha256(args.model_output)},
        "feature_contract": {
            "raw_width": int(raw_x.shape[1]),
            "consequence_width": int(
                exact_consequence.shape[1]
                if exact_consequence is not None else consequence.shape[1]
            ),
            "augmented_width": int(
                exact_augmented_x.shape[1]
                if exact_augmented_x is not None else augmented_x.shape[1]
            ),
            "frozen_model_width": int(
                len(frozen_stages["EARLY"]["feature_names"])
            ),
            "runtime_response_rollouts_required": model_uses_response_rollouts,
            "future_information_used": False,
        },
        "split": {
            "train_opponents": sorted(train_opponents),
            "test_opponents": sorted(test_opponents),
            "train_seeds": sorted(train_seeds),
            "calibration_seeds": sorted(calibration_seeds),
            "test_seeds": sorted(test_seeds),
            "bank_a_columns": bank_a.tolist(),
            "bank_b_columns": bank_b.tolist(),
        },
        "ablation": ablation,
        "frozen_variant": frozen_variant,
        "gates": {
            "all_stage_calibration_passed": calibration_pass,
            "all_internal_unseen_stage_passed": internal_pass,
            "deployment_gate_passed": calibration_pass and internal_pass,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({
        "feature_contract": receipt["feature_contract"],
        "ablation": {
            variant: {
                stage: {
                    "calibration": values["metrics"]["calibration"],
                    "unseen_seeds": values["metrics"]["unseen_seeds"],
                    "unseen_opponents": values["metrics"]["unseen_opponents"],
                    "joint_unseen": values["metrics"]["joint_unseen"],
                    "passed_thresholds": values["threshold_search"]["passed_count"],
                }
                for stage, values in stages.items()
            }
            for variant, stages in ablation.items()
        },
        "gates": receipt["gates"],
        "receipt": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
