#!/usr/bin/env python3
"""Train-only action-support prior -> state residual -> KEEP sanity screen."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

import train_phase_challenger_outcome_pairwise_residual_A_v0 as pair_v0


SCHEMA = "phase-challenger-action-prior-residual-A-v1"
FORMAL_SEEDS, SEED_FOLDS = pair_v0.FORMAL_SEEDS, pair_v0.SEED_FOLDS
REGRESSION, NEUTRAL, WIN_GAIN = (
    pair_v0.REGRESSION, pair_v0.NEUTRAL, pair_v0.WIN_GAIN,
)
PAIR_WIDTH = pair_v0.PAIR_WIDTH
PAIR_FEATURE_NAMES = pair_v0.PAIR_FEATURE_NAMES
STATE_SHUFFLE_WIDTH = 907
PRIOR_STRENGTH = 4.0
RESIDUAL_LOG_ODDS_WEIGHT = 1.0
MAX_RISK_PROBABILITY = 0.5
PROBABILITY_EPSILON = 1e-5
ALL_KNOWN_SAFE_SCORE = -2.0
KEEP_ONLY_SAFE_SCORE = 2.0


def seed_fold_map() -> dict[int, int]:
    return pair_v0.seed_fold_map()


def _subset(arrays: Mapping[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[rows] for key, value in arrays.items()}


def outcome_arrays(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    return pair_v0.outcome_arrays(arrays)


def build_pairs(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    pairs = pair_v0.attach_pair_targets(
        arrays, pair_v0.build_pair_features(arrays),
    )
    candidate = np.asarray(pairs["candidate_row"], np.int64)
    pairs["relative_outcome"] = np.asarray(
        arrays["relative_outcome"], np.int8,
    )[candidate]
    pairs["target_regression"] = (
        pairs["relative_outcome"] < 0
    ).astype(np.int8)
    if not np.array_equal(
        pairs["target_regression"],
        (np.asarray(pairs["target_class"]) == REGRESSION).astype(np.int8),
    ):
        raise AssertionError("binary risk target diverged from outcome regression")
    return pairs


def classifier_params(trees: int, random_seed: int) -> dict[str, Any]:
    if trees < 2:
        raise ValueError("action-prior residual screen requires at least two trees")
    return {
        "n_estimators": int(trees), "max_depth": 6,
        "min_samples_leaf": 2, "max_features": 0.5,
        "bootstrap": False, "class_weight": "balanced",
        "n_jobs": -1, "random_state": int(random_seed),
    }


def regressor_params(trees: int, random_seed: int) -> dict[str, Any]:
    return {
        key: value for key, value in classifier_params(trees, random_seed).items()
        if key != "class_weight"
    }


def _fit_binary(
    features: np.ndarray, target: np.ndarray, weights: np.ndarray,
    trees: int, random_seed: int,
) -> tuple[Any, dict[str, Any]]:
    target = np.asarray(target, np.int8)
    classes = np.unique(target)
    counts = Counter(map(int, target))
    if len(classes) == 1:
        model: Any = {"constant_binary": int(classes[0])}
        kind = "constant_missing-class-safe"
    else:
        model = ExtraTreesClassifier(**classifier_params(trees, random_seed))
        model.fit(features, target, sample_weight=weights)
        kind = "class-balanced-shallow-ExtraTreesClassifier"
    return model, {
        "kind": kind, "negative_rows": counts[0], "positive_rows": counts[1],
        "class_weight": "balanced", "pair_cell_sample_weight_used": True,
    }


def _predict_binary(model: Any, features: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        value = float(model["constant_binary"])
        return np.full(len(features), value, np.float64)
    raw = np.asarray(model.predict_proba(features), np.float64)
    result = np.zeros(len(features), np.float64)
    if 1 in model.classes_:
        result = raw[:, list(model.classes_).index(1)]
    return result


def _fit_margin(
    features: np.ndarray, pairs: Mapping[str, np.ndarray], weights: np.ndarray,
    trees: int, random_seed: int,
) -> tuple[Any, dict[str, Any]]:
    # This is deliberately stricter than target_class == NEUTRAL: future
    # loss->tie rows are outcome upgrades, not margin tie-break examples.
    neutral = np.flatnonzero(np.asarray(pairs["relative_outcome"]) == 0)
    target = np.asarray(pairs["target_neutral_margin"], np.float64)
    if not len(neutral):
        model: Any = {"constant_margin": 0.0}
        kind = "constant-no-same-outcome-safe"
    elif np.all(target[neutral] == target[neutral[0]]):
        model = {"constant_margin": float(target[neutral[0]])}
        kind = "constant-same-outcome-margin"
    else:
        model = ExtraTreesRegressor(**regressor_params(trees, random_seed))
        model.fit(features[neutral], target[neutral], sample_weight=weights[neutral])
        kind = "same-outcome-only-shallow-ExtraTreesRegressor"
    return model, {
        "kind": kind, "training_rows": int(len(neutral)),
        "training_filter": "relative_outcome == 0",
        "loss_to_tie_rows_admitted": 0,
    }


def _predict_margin(model: Any, features: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        return np.full(len(features), float(model["constant_margin"]), np.float64)
    return np.asarray(model.predict(features), np.float64)


def _smoothed_rate(positive: float, support: float, global_rate: float) -> float:
    return float(
        (positive + PRIOR_STRENGTH * global_rate)
        / (support + PRIOR_STRENGTH)
    )


def fit_action_prior(pairs: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    fingerprints = np.asarray(pairs["action_delta_fingerprint"])
    gain = np.asarray(pairs["target_win_gain"], np.int8)
    risk = np.asarray(pairs["target_regression"], np.int8)
    decision = np.asarray(pairs["decision"], np.int64)
    if not len(fingerprints):
        raise ValueError("action prior requires candidate-vs-KEEP pairs")
    seen_decision_action: set[tuple[int, bytes]] = set()
    counts: dict[bytes, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for fingerprint, row_gain, row_risk, row_decision in zip(
        fingerprints, gain, risk, decision, strict=True,
    ):
        key = bytes(fingerprint)
        observation = (int(row_decision), key)
        if observation in seen_decision_action:
            raise ValueError("one decision repeated an exact executable action delta")
        seen_decision_action.add(observation)
        counts[key][0] += 1.0
        counts[key][1] += int(row_gain)
        counts[key][2] += int(row_risk)
    n = float(len(fingerprints))
    global_gain = float((gain.sum() + 0.5) / (n + 1.0))
    global_risk = float((risk.sum() + 0.5) / (n + 1.0))
    table = {
        key: {
            "support": int(value[0]), "win_gain_count": int(value[1]),
            "regression_count": int(value[2]),
            "win_gain_rate": _smoothed_rate(value[1], value[0], global_gain),
            "regression_rate": _smoothed_rate(value[2], value[0], global_risk),
        }
        for key, value in counts.items()
    }
    return {
        "table": table, "global_win_gain_rate": global_gain,
        "global_regression_rate": global_risk,
    }, {
        "exact_action_deltas": len(table), "observations": int(n),
        "prior_strength": PRIOR_STRENGTH,
        "global_win_gain_rate": global_gain,
        "global_regression_rate": global_risk,
        "support_unit": "unique (decision, exact action-delta) observation",
        "unknown_action_policy": "ineligible_then_KEEP",
        "labels_source": "train split only",
    }


def shuffled_state_features(
    pairs: Mapping[str, np.ndarray], random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Shuffle shared state by decision within (seat, step), preserving actions."""

    features = np.asarray(pairs["features"], np.float32)
    shuffled = features.copy()
    slices = pair_v0.outcome_v1.base.decision_slices(
        np.asarray(pairs["decision"], np.int64),
    )
    cells: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for rows in slices:
        seats = set(map(int, np.asarray(pairs["seat"])[rows]))
        steps = set(map(int, np.asarray(pairs["step"])[rows]))
        if len(seats) != 1 or len(steps) != 1:
            raise ValueError("state-shuffle decision crossed seat or step")
        cells[(seats.pop(), steps.pop())].append(rows)
    rng = np.random.default_rng(int(random_seed))
    digest = hashlib.sha256()
    moved = 0
    for cell in sorted(cells):
        groups = cells[cell]
        permutation = rng.permutation(len(groups))
        digest.update(np.asarray(cell, np.int64).tobytes())
        digest.update(np.asarray(permutation, np.int64).tobytes())
        snapshots = [features[rows[0], :STATE_SHUFFLE_WIDTH].copy() for rows in groups]
        for target, source_at in zip(groups, permutation, strict=True):
            shuffled[target, :STATE_SHUFFLE_WIDTH] = snapshots[int(source_at)]
            moved += int(not np.array_equal(target, groups[int(source_at)]))
    if not np.array_equal(
        shuffled[:, STATE_SHUFFLE_WIDTH:], features[:, STATE_SHUFFLE_WIDTH:],
    ):
        raise AssertionError("state shuffle changed seat or action residual fields")
    return shuffled, {
        "scheme": "decision-state permutation within (seat, step)",
        "shuffled_feature_prefix": f"pair_features[0:{STATE_SHUFFLE_WIDTH}]",
        "absolute_seat_and_action_delta_preserved": True,
        "decision_groups": len(slices), "moved_decision_groups": moved,
        "permutation_sha256": digest.hexdigest(),
        "labels_used_to_construct_permutation": False,
    }


def fit_hierarchy(
    pairs: Mapping[str, np.ndarray], trees: int, random_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    features = np.asarray(pairs["features"], np.float32)
    weights, weight_audit = pair_v0.pair_weights(pairs)
    prior, prior_audit = fit_action_prior(pairs)
    gain_model, gain_audit = _fit_binary(
        features, np.asarray(pairs["target_win_gain"]), weights,
        trees, random_seed + 11,
    )
    risk_model, risk_audit = _fit_binary(
        features, np.asarray(pairs["target_regression"]), weights,
        trees, random_seed + 23,
    )
    margin_model, margin_audit = _fit_margin(
        features, pairs, weights, trees, random_seed + 37,
    )
    shuffled, shuffle_audit = shuffled_state_features(pairs, random_seed + 51)
    shuffle_gain_model, shuffle_gain_audit = _fit_binary(
        shuffled, np.asarray(pairs["target_win_gain"]), weights,
        trees, random_seed + 67,
    )
    shuffle_risk_model, shuffle_risk_audit = _fit_binary(
        shuffled, np.asarray(pairs["target_regression"]), weights,
        trees, random_seed + 79,
    )
    return {
        "prior": prior, "gain_model": gain_model, "risk_model": risk_model,
        "margin_model": margin_model,
        "shuffle_gain_model": shuffle_gain_model,
        "shuffle_risk_model": shuffle_risk_model,
    }, {
        "action_prior": prior_audit, "gain_head": gain_audit,
        "risk_head": risk_audit, "neutral_margin_head": margin_audit,
        "state_shuffle": shuffle_audit,
        "state_shuffle_gain_head": shuffle_gain_audit,
        "state_shuffle_risk_head": shuffle_risk_audit,
        "pair_weight": weight_audit,
        "fusion": (
            "smoothed exact-action prior log-odds plus fixed-weight "
            "class-balanced state-action evidence"
        ),
        "residual_log_odds_weight": RESIDUAL_LOG_ODDS_WEIGHT,
        "outer_labels_choose_fusion_weight": False,
    }


def _clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(
        np.asarray(values, np.float64),
        PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON,
    )


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = _clip_probability(values)
    return np.log(clipped) - np.log1p(-clipped)


def _expit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def lookup_action_prior(
    prior: Mapping[str, Any], pairs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    n = len(pairs["decision"])
    known = np.zeros(n, np.bool_)
    support = np.zeros(n, np.int32)
    gain = np.full(n, float(prior["global_win_gain_rate"]), np.float64)
    risk = np.full(n, float(prior["global_regression_rate"]), np.float64)
    table = prior["table"]
    for row, fingerprint in enumerate(pairs["action_delta_fingerprint"]):
        record = table.get(bytes(fingerprint))
        if record is None:
            continue
        known[row] = True
        support[row] = int(record["support"])
        gain[row] = float(record["win_gain_rate"])
        risk[row] = float(record["regression_rate"])
    return {
        "known_action": known, "action_support": support,
        "gain_probability": gain, "risk_probability": risk,
        "safe_score": gain - risk,
        "neutral_margin_prediction": np.zeros(n, np.float64),
    }


def _combine_prior_and_evidence(
    prior_probability: np.ndarray, balanced_probability: np.ndarray,
) -> np.ndarray:
    # A balanced classifier has a 0.5 training prior; its log-odds are fixed
    # weight evidence, not a second empirical base rate.
    return _expit(
        _logit(prior_probability)
        + RESIDUAL_LOG_ODDS_WEIGHT * _logit(balanced_probability)
    )


def predict_hierarchy(
    hierarchy: Mapping[str, Any], pairs: Mapping[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    features = np.asarray(pairs["features"], np.float32)
    support = lookup_action_prior(hierarchy["prior"], pairs)
    gain = _combine_prior_and_evidence(
        support["gain_probability"],
        _predict_binary(hierarchy["gain_model"], features),
    )
    risk = _combine_prior_and_evidence(
        support["risk_probability"],
        _predict_binary(hierarchy["risk_model"], features),
    )
    shuffle_gain = _combine_prior_and_evidence(
        support["gain_probability"],
        _predict_binary(hierarchy["shuffle_gain_model"], features),
    )
    shuffle_risk = _combine_prior_and_evidence(
        support["risk_probability"],
        _predict_binary(hierarchy["shuffle_risk_model"], features),
    )
    common = {
        "known_action": support["known_action"],
        "action_support": support["action_support"],
    }
    return {
        "primary": {
            **common, "gain_probability": gain, "risk_probability": risk,
            "safe_score": gain - risk,
            "neutral_margin_prediction": _predict_margin(
                hierarchy["margin_model"], features,
            ),
        },
        "support_only": support,
        "state_shuffle": {
            **common, "gain_probability": shuffle_gain,
            "risk_probability": shuffle_risk,
            "safe_score": shuffle_gain - shuffle_risk,
            "neutral_margin_prediction": np.zeros(len(features), np.float64),
        },
    }


def empty_pair_predictions(rows: int) -> dict[str, dict[str, np.ndarray]]:
    def mode() -> dict[str, np.ndarray]:
        return {
            "known_action": np.zeros(rows, np.bool_),
            "action_support": np.zeros(rows, np.int32),
            "gain_probability": np.full(rows, np.nan, np.float64),
            "risk_probability": np.full(rows, np.nan, np.float64),
            "safe_score": np.full(rows, np.nan, np.float64),
            "neutral_margin_prediction": np.full(rows, np.nan, np.float64),
        }
    return {
        "primary": mode(), "support_only": mode(), "state_shuffle": mode(),
    }


def inner_seed_fold_oof(
    pairs: Mapping[str, np.ndarray], groups: np.ndarray,
    trees: int, random_seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    groups = np.asarray(groups, np.int8)
    unique = tuple(sorted(set(map(int, groups))))
    if len(unique) < 2:
        raise ValueError("inner seed-fold OOF requires at least two groups")
    predictions = empty_pair_predictions(len(groups))
    audits = []
    for offset, group in enumerate(unique):
        train = np.flatnonzero(groups != group)
        valid = np.flatnonzero(groups == group)
        hierarchy, fit_audit = fit_hierarchy(
            _subset(pairs, train), trees,
            random_seed + offset * 10_007,
        )
        fold_predictions = predict_hierarchy(
            hierarchy, _subset(pairs, valid),
        )
        for mode, fields in fold_predictions.items():
            for key, values in fields.items():
                predictions[mode][key][valid] = values
        train_seeds = set(map(int, np.asarray(pairs["seed"])[train]))
        valid_seeds = set(map(int, np.asarray(pairs["seed"])[valid]))
        audits.append({
            "left_out_seed_fold": group,
            "train_seeds": sorted(train_seeds), "valid_seeds": sorted(valid_seeds),
            "train_valid_seed_overlap": sorted(train_seeds & valid_seeds),
            "train_valid_decision_overlap": sorted(
                set(map(int, np.asarray(pairs["decision"])[train]))
                & set(map(int, np.asarray(pairs["decision"])[valid]))
            ),
            **fit_audit,
        })
    for mode in predictions.values():
        if (
            not np.isfinite(mode["gain_probability"]).all()
            or not np.isfinite(mode["risk_probability"]).all()
            or not np.isfinite(mode["safe_score"]).all()
            or not np.isfinite(mode["neutral_margin_prediction"]).all()
        ):
            raise RuntimeError("inner seed-fold OOF predictions are incomplete")
    return predictions, audits


def candidate_scores(
    arrays: Mapping[str, np.ndarray], pairs: Mapping[str, np.ndarray],
    pair_scores: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    candidate = np.asarray(pairs["candidate_row"], np.int64)
    if len(candidate) != len(np.unique(candidate)):
        raise ValueError("candidate-vs-KEEP mapping is not one-to-one")
    rows = len(arrays["decision"])
    result = {
        "known_action": np.zeros(rows, np.bool_),
        "action_support": np.zeros(rows, np.int32),
        "gain_probability": np.zeros(rows, np.float64),
        "risk_probability": np.ones(rows, np.float64),
        "safe_score": np.full(rows, -np.inf, np.float64),
        "neutral_margin_prediction": np.zeros(rows, np.float64),
    }
    for key in result:
        result[key][candidate] = np.asarray(pair_scores[key])
    return result


def decision_top_rows(
    arrays: Mapping[str, np.ndarray], scores: Mapping[str, np.ndarray],
) -> np.ndarray:
    edit = np.asarray(arrays["edit"])
    known = np.asarray(scores["known_action"], np.bool_)
    risk = np.asarray(scores["risk_probability"], np.float64)
    safe = np.asarray(scores["safe_score"], np.float64)
    gain = np.asarray(scores["gain_probability"], np.float64)
    margin = np.asarray(scores["neutral_margin_prediction"], np.float64)
    result = []
    for rows in pair_v0.outcome_v1.base.decision_slices(
        np.asarray(arrays["decision"], np.int64),
    ):
        keep = rows[edit[rows] == pair_v0.outcome_v1.base.KEEP_EDIT]
        if len(keep) != 1:
            raise ValueError("selection requires one KEEP per decision")
        candidates = rows[
            (edit[rows] != pair_v0.outcome_v1.base.KEEP_EDIT)
            & known[rows] & (risk[rows] < MAX_RISK_PROBABILITY)
        ]
        chosen = int(keep[0])
        if len(candidates):
            chosen = max(map(int, candidates), key=lambda row: (
                safe[row], gain[row], -risk[row], margin[row], -row,
            ))
        result.append(chosen)
    return np.asarray(result, np.int64)


def threshold_breakpoints(
    arrays: Mapping[str, np.ndarray], scores: Mapping[str, np.ndarray],
) -> tuple[float, ...]:
    top = decision_top_rows(arrays, scores)
    edit = np.asarray(arrays["edit"])
    values = sorted(set(map(
        float, np.asarray(scores["safe_score"])[
            top[edit[top] != pair_v0.outcome_v1.base.KEEP_EDIT]
        ],
    )))
    if values and (values[0] < -1.0 or values[-1] > 1.0):
        raise AssertionError("gain-minus-risk score escaped [-1, 1]")
    return (ALL_KNOWN_SAFE_SCORE, *values, KEEP_ONLY_SAFE_SCORE)


def select_choices(
    arrays: Mapping[str, np.ndarray], scores: Mapping[str, np.ndarray],
    min_safe_score: float,
) -> np.ndarray:
    top = decision_top_rows(arrays, scores)
    safe = np.asarray(scores["safe_score"], np.float64)
    edit = np.asarray(arrays["edit"])
    result = top.copy()
    slices = pair_v0.outcome_v1.base.decision_slices(
        np.asarray(arrays["decision"], np.int64),
    )
    for at, (rows, choice) in enumerate(zip(slices, top, strict=True)):
        if (
            edit[choice] == pair_v0.outcome_v1.base.KEEP_EDIT
            or safe[choice] < float(min_safe_score)
        ):
            result[at] = int(rows[edit[rows] == pair_v0.outcome_v1.base.KEEP_EDIT][0])
    return result


def choice_metrics(
    arrays: Mapping[str, np.ndarray], choices: np.ndarray,
) -> dict[str, Any]:
    return pair_v0.choice_metrics(arrays, choices)


def calibrate(
    arrays: Mapping[str, np.ndarray], scores: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray]:
    breakpoints = threshold_breakpoints(arrays, scores)
    trials = []
    for threshold in breakpoints:
        choices = select_choices(arrays, scores, threshold)
        metrics = choice_metrics(arrays, choices)
        trials.append(({
            "min_safe_score": threshold,
            "threshold_is_KEEP_only": threshold == KEEP_ONLY_SAFE_SCORE,
            **metrics,
        }, choices))
    allowed = [item for item in trials if (
        item[0]["wins_lost"] == 0
        and item[0]["outcome_regressions"] == 0
        and not any(item[0]["opponent_outcome_regressions"].values())
        and item[0]["same_outcome_margin_delta_sum"] >= 0.0
    )]
    if not allowed:
        raise RuntimeError("decision-top calibration lost KEEP-only fallback")
    chosen, choices = max(allowed, key=lambda item: (
        item[0]["raw_win_delta"], item[0]["wins_gained"],
        item[0]["outcome_delta_sum"],
        item[0]["same_outcome_margin_delta_sum"],
        -item[0]["fires"], item[0]["min_safe_score"],
    ))
    return {
        "min_safe_score": chosen["min_safe_score"],
        "inner_seed_fold_OOF_metrics": chosen,
        "candidate_thresholds_tested": len(breakpoints),
        "threshold_source": "unique inner-OOF decision-top safe-score breakpoints",
        "KEEP_only_included": True,
        "candidate_row_quantiles_used": False,
    }, choices


def rank_metrics(
    arrays: Mapping[str, np.ndarray], scores: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return pair_v0._rank_metrics(
        arrays, np.asarray(scores["safe_score"], np.float64),
    )


def action_support_audit(
    arrays: Mapping[str, np.ndarray], scores: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    edit = np.asarray(arrays["edit"])
    candidate = edit != pair_v0.outcome_v1.base.KEEP_EDIT
    known = np.asarray(scores["known_action"], np.bool_)
    decisions_known = 0
    slices = pair_v0.outcome_v1.base.decision_slices(
        np.asarray(arrays["decision"], np.int64),
    )
    for rows in slices:
        decisions_known += int(np.any(candidate[rows] & known[rows]))
    return {
        "candidate_rows": int(np.count_nonzero(candidate)),
        "known_candidate_rows": int(np.count_nonzero(candidate & known)),
        "unknown_candidate_rows": int(np.count_nonzero(candidate & ~known)),
        "decisions_with_known_action": decisions_known,
        "decisions": len(slices),
        "unknown_action_policy": "ineligible_then_KEEP",
    }


def lopo_coverage_only(pairs: Mapping[str, np.ndarray]) -> dict[str, Any]:
    gain = np.asarray(pairs["target_win_gain"]) == 1
    opponents = sorted(set(map(int, np.asarray(pairs["opponent"])[gain])))
    return {
        "mode": "cheap_train_only_LOPO_coverage_no_model_fit",
        "win_gain_opponents": opponents,
        "coverage_supported": len(opponents) >= 2,
        "models_fit": 0, "predictions_discarded": 0,
        "used_for_choice_or_threshold": False,
        "unknown_opponent_policy": "abstain_to_KEEP_until_supported",
    }


def _aggregate_rank(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = int(sum(row["upgrade_decisions"] for row in rows))
    if not decisions:
        return {
            "upgrade_decisions": 0, "top1_win_recall": 0.0,
            "mean_reciprocal_rank_first_win": 0.0,
        }
    top1 = sum(
        row["top1_win_recall"] * row["upgrade_decisions"] for row in rows
    )
    reciprocal = sum(
        row["mean_reciprocal_rank_first_win"] * row["upgrade_decisions"]
        for row in rows
    )
    return {
        "upgrade_decisions": decisions,
        "top1_win_recall": float(top1 / decisions),
        "mean_reciprocal_rank_first_win": float(reciprocal / decisions),
    }


def _rank_strictly_better(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        left["top1_win_recall"] > right["top1_win_recall"]
        or left["mean_reciprocal_rank_first_win"]
        > right["mean_reciprocal_rank_first_win"]
    )


def evaluate(
    arrays: Mapping[str, np.ndarray], *, trees: int = 48,
    random_seed: int = 20260829,
) -> dict[str, Any]:
    input_audit = pair_v0.outcome_v1.base.validate_arrays(arrays)
    a, a_source = outcome_arrays(arrays)
    if tuple(sorted(set(map(int, a["seed"])))) != FORMAL_SEEDS:
        raise ValueError("action-prior residual v1 requires all eight formal seeds")
    fold_map = seed_fold_map()
    row_folds = np.asarray([fold_map[int(seed)] for seed in a["seed"]], np.int8)
    choices_by_mode: dict[str, dict[int, int]] = {
        mode: {} for mode in ("primary", "support_only", "state_shuffle")
    }
    folds = []
    for outer_fold in range(4):
        train_rows = np.flatnonzero(row_folds != outer_fold)
        valid_rows = np.flatnonzero(row_folds == outer_fold)
        train, valid = _subset(a, train_rows), _subset(a, valid_rows)
        train_pairs, valid_pairs = build_pairs(train), build_pairs(valid)
        inner_groups = np.asarray(
            [fold_map[int(seed)] for seed in train_pairs["seed"]], np.int8,
        )
        oof_pair_scores, inner_audit = inner_seed_fold_oof(
            train_pairs, inner_groups, trees,
            random_seed + outer_fold * 100_003 + 101,
        )
        calibration: dict[str, dict[str, Any]] = {}
        for mode in oof_pair_scores:
            oof_scores = candidate_scores(
                train, train_pairs, oof_pair_scores[mode],
            )
            calibration[mode], _ = calibrate(train, oof_scores)

        # Inner OOF is only for calibration. The outer model is freshly fit on
        # every outer-train seed before it can see outer-valid features.
        hierarchy, outer_fit_audit = fit_hierarchy(
            train_pairs, trees,
            random_seed + outer_fold * 100_003 + 50_003,
        )
        valid_pair_scores = predict_hierarchy(hierarchy, valid_pairs)
        valid_scores = {
            mode: candidate_scores(valid, valid_pairs, values)
            for mode, values in valid_pair_scores.items()
        }
        valid_choices = {
            mode: select_choices(
                valid, scores, calibration[mode]["min_safe_score"],
            )
            for mode, scores in valid_scores.items()
        }
        slices = pair_v0.outcome_v1.base.decision_slices(valid["decision"])
        for mode, mode_choices in valid_choices.items():
            for rows, choice in zip(slices, mode_choices, strict=True):
                decision = int(valid["decision"][rows[0]])
                choices_by_mode[mode][decision] = int(valid_rows[int(choice)])
        oracle = pair_v0.outcome_v1.oracle_headroom(valid)
        ranks = {
            mode: rank_metrics(valid, scores)
            for mode, scores in valid_scores.items()
        }
        folds.append({
            "left_out_seed_fold": outer_fold,
            "train_seeds": sorted(set(map(int, train["seed"]))),
            "valid_seeds": sorted(set(map(int, valid["seed"]))),
            "train_valid_seed_overlap": [],
            "outer_labels_used_for_model_or_threshold": False,
            "inner_seed_fold_OOF_used_for_calibration_only": True,
            "outer_full_train_refit": True,
            "outer_refit_train_seed_count": len(set(map(int, train["seed"]))),
            "outer_refit_audit": outer_fit_audit,
            "inner_OOF_audit": inner_audit,
            "calibration": calibration,
            "outer_metrics": {
                mode: choice_metrics(valid, values)
                for mode, values in valid_choices.items()
            },
            "outer_oracle_headroom": oracle,
            "no_threshold_rank": ranks,
            "primary_action_support": action_support_audit(
                valid, valid_scores["primary"],
            ),
            "LOPO": lopo_coverage_only(train_pairs),
            "outer_pair_weight_audit": pair_v0.pair_weights(valid_pairs)[1],
            "outer_pair_rows_by_step": pair_v0.pair_rows_by_step(valid_pairs),
        })

    slices = pair_v0.outcome_v1.base.decision_slices(a["decision"])
    decision_order = [int(a["decision"][rows[0]]) for rows in slices]
    ordered_choices: dict[str, np.ndarray] = {}
    for mode, mapping in choices_by_mode.items():
        if set(mapping) != set(decision_order):
            raise RuntimeError(f"{mode} outer folds lost decision coverage")
        ordered_choices[mode] = np.asarray(
            [mapping[decision] for decision in decision_order], np.int64,
        )
    metrics = {
        mode: choice_metrics(a, choices)
        for mode, choices in ordered_choices.items()
    }
    positive_folds = [
        fold for fold in folds
        if fold["outer_oracle_headroom"]["oracle_raw_win_gain"] > 0
    ]
    aggregate_rank = {
        mode: _aggregate_rank([
            fold["no_threshold_rank"][mode] for fold in positive_folds
        ])
        for mode in ordered_choices
    }
    improving_folds = sum(
        fold["outer_metrics"]["primary"]["raw_win_delta"] > 0
        for fold in positive_folds
    )
    primary = metrics["primary"]
    acceptance = {
        "at_least_two_headroom_outer_folds_add_wins": improving_folds >= 2,
        "headroom_outer_folds_adding_wins": int(improving_folds),
        "wins_lost_zero": primary["wins_lost"] == 0,
        "outcome_regressions_zero": primary["outcome_regressions"] == 0,
        "same_outcome_margin_nonnegative": (
            primary["same_outcome_margin_delta_sum"] >= 0.0
        ),
        "primary_rank_strictly_beats_support_only": _rank_strictly_better(
            aggregate_rank["primary"], aggregate_rank["support_only"],
        ),
        "primary_rank_strictly_beats_state_shuffle": _rank_strictly_better(
            aggregate_rank["primary"], aggregate_rank["state_shuffle"],
        ),
    }
    acceptance["passed"] = all(
        acceptance[key] for key in (
            "at_least_two_headroom_outer_folds_add_wins",
            "wins_lost_zero", "outcome_regressions_zero",
            "same_outcome_margin_nonnegative",
            "primary_rank_strictly_beats_support_only",
            "primary_rank_strictly_beats_state_shuffle",
        )
    )
    pairs_by_step: Counter[int] = Counter()
    for fold in folds:
        pairs_by_step.update({
            int(step): int(count)
            for step, count in fold["outer_pair_rows_by_step"].items()
        })
    return {
        "schema": SCHEMA,
        "status": (
            "action_prior_state_residual_sanity_screen_passed"
            if acceptance["passed"]
            else "action_prior_state_residual_sanity_screen_failed"
        ),
        "evidence_class": "train_only_repair_on_dev_screen_not_final_model",
        "seed_folds": [list(values) for values in SEED_FOLDS],
        "outer_labels_used_for_model_or_threshold": False,
        "outer_full_train_refit": True,
        "inner_seed_fold_OOF_used_for_calibration_only": True,
        "feature_schema": {
            **{
                key: value for key, value in pair_v0.FEATURE_SCHEMA.items()
                if key not in {
                    "shared_indices", "relative_indices", "pair_feature_names",
                }
            },
            "pair_width": PAIR_WIDTH,
            "schema_dependency": pair_v0.SCHEMA,
        },
        "model_contract": {
            "classifier": classifier_params(trees, random_seed),
            "gain_and_risk_heads_class_balanced": True,
            "action_prior_strength": PRIOR_STRENGTH,
            "residual_log_odds_weight": RESIDUAL_LOG_ODDS_WEIGHT,
            "max_risk_probability": MAX_RISK_PROBABILITY,
            "fusion_weight_selected_without_outer_labels": True,
            "unknown_action_fallback": "KEEP",
            "margin_training_filter": "relative_outcome == 0",
        },
        "folds": folds, "acceptance_gate": acceptance,
        "positive_fold_aggregate_rank": aggregate_rank,
        "all_decision_metrics": metrics,
        "all_decision_oracle_headroom": pair_v0.outcome_v1.oracle_headroom(a),
        "A_choices": ordered_choices["primary"],
        "support_only_A_choices": ordered_choices["support_only"],
        "state_shuffle_A_choices": ordered_choices["state_shuffle"],
        "A_source_indices": a_source,
        "input_audit": input_audit,
        "A_rows": int(len(a["decision"])), "A_decisions": len(slices),
        "A_candidate_KEEP_pairs": int(sum(
            fold["outer_pair_weight_audit"]["pair_rows"] for fold in folds
        )),
        "A_candidate_KEEP_pairs_by_step": {
            str(step): pairs_by_step[step] for step in sorted(pairs_by_step)
        },
        "objective_contract": {
            "stage_1": "exact action support and empirical gain/risk prior",
            "stage_2": "class-balanced state-action residual evidence",
            "stage_3": "KEEP for unknown, unsafe, or below-threshold actions",
            "thresholds": "unique decision-top OOF safe-score breakpoints",
            "tie_break_only": "relative_outcome==0 margin head",
            "support_only_and_state_shuffle_are_required_ablations": True,
        },
    }
