#!/usr/bin/env python3
"""Pure nested seed-fold evaluator for one A/R0 outcome classifier."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

import run_farmer_augment_union_mpc_v1 as gate
import train_phase_challenger_residual_gate_v1 as base


SCHEMA = "phase-challenger-outcome-multiclass-A-nested-v1"
REGRESSION, NEUTRAL, WIN_GAIN = 0, 1, 2
FORMAL_SEEDS = tuple(range(2026086300, 2026086308))
SEED_FOLDS = (
    (2026086300, 2026086304),
    (2026086301, 2026086305),
    (2026086302, 2026086306),
    (2026086303, 2026086307),
)


def seed_fold_map() -> dict[int, int]:
    result = {
        seed: fold for fold, seeds in enumerate(SEED_FOLDS) for seed in seeds
    }
    if tuple(sorted(result)) != FORMAL_SEEDS:
        raise AssertionError("pre-registered seed folds changed")
    return result


def outcome_arrays(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Filter the canonical R0 prefix and label every row versus its KEEP."""

    if "outcome" not in arrays:
        raise ValueError("outcome-aware A requires raw outcome labels")
    a, source = base.r0_arrays(arrays)
    outcome = np.asarray(arrays["outcome"], np.int8)[source]
    if not set(map(int, outcome)) <= {0, 1, 2}:
        raise ValueError("outcome must be 0=loss, 1=tie, 2=win")
    fallback = np.empty(len(outcome), np.int8)
    for rows in base.decision_slices(a["decision"]):
        keep = rows[np.asarray(a["edit"])[rows] == base.KEEP_EDIT]
        if len(keep) != 1:
            raise ValueError("A decision lost its unique KEEP")
        fallback[rows] = outcome[int(keep[0])]
    relative = np.sign(outcome.astype(np.int16) - fallback).astype(np.int8)
    win_gain = (outcome == 2) & (fallback < 2)
    target = np.full(len(outcome), NEUTRAL, np.int8)
    target[relative < 0] = REGRESSION
    target[win_gain] = WIN_GAIN
    result = {
        **a,
        "outcome": outcome,
        "fallback_outcome": fallback,
        "relative_outcome": relative,
        "target_win_gain": win_gain.astype(np.int8),
        "target_class": target,
    }
    for key in ("seat", "step"):
        if key in arrays:
            result[key] = np.asarray(arrays[key])[source]
    return result, source


def model_params(trees: int, random_seed: int) -> dict[str, Any]:
    if trees < 2:
        raise ValueError("multiclass evaluation requires at least two trees")
    return {
        "n_estimators": int(trees),
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": 0.5,
        "class_weight": "balanced",
        "bootstrap": False,
        "n_jobs": -1,
        "random_state": int(random_seed),
    }


def _class_counts(target: np.ndarray) -> dict[str, int]:
    counts = Counter(map(int, target))
    return {
        "regression": counts[REGRESSION],
        "neutral_including_nonwin_ordinal_upgrade": counts[NEUTRAL],
        "win_gain": counts[WIN_GAIN],
    }


def _subset(arrays: Mapping[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[rows] for key, value in arrays.items()}


def fit_model(
    arrays: Mapping[str, np.ndarray], trees: int, random_seed: int,
) -> tuple[Any, dict[str, Any]]:
    target = np.asarray(arrays["target_class"], np.int8)
    counts = _class_counts(target)
    classes = np.unique(target)
    if len(classes) == 1:
        return {"constant_class": int(classes[0])}, {
            "class_counts": counts,
            "model_kind": "constant_missing-class-safe",
        }
    model = ExtraTreesClassifier(**model_params(trees, random_seed))
    model.fit(
        np.asarray(arrays["features"], np.float32), target,
        sample_weight=gate._decision_weights(
            np.asarray(arrays["decision"], np.int64)
        ),
    )
    return model, {
        "class_counts": counts,
        "model_kind": "single_three_class_ExtraTreesClassifier",
    }


def predict_proba3(model: Any, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, np.float32)
    result = np.zeros((len(values), 3), np.float32)
    if isinstance(model, dict):
        result[:, int(model["constant_class"])] = 1.0
        return result
    raw = np.asarray(model.predict_proba(values), np.float32)
    result[:, np.asarray(model.classes_, np.int64)] = raw
    if not np.allclose(result.sum(axis=1), 1.0):
        raise RuntimeError("three-class probability alignment failed")
    return result


def group_oof_view(
    arrays: Mapping[str, np.ndarray], groups: np.ndarray, view: str,
    trees: int, random_seed: int,
) -> tuple[np.ndarray, list[Any], list[dict[str, Any]]]:
    groups = np.asarray(groups)
    unique = tuple(sorted(set(map(int, groups))))
    if len(unique) < 2:
        raise ValueError(f"{view} OOF requires at least two groups")
    probabilities = np.full((len(groups), 3), np.nan, np.float32)
    models = []
    audit = []
    for offset, group in enumerate(unique):
        train = np.flatnonzero(groups != group)
        valid = np.flatnonzero(groups == group)
        model, model_audit = fit_model(
            _subset(arrays, train), trees, random_seed + offset * 1009,
        )
        probabilities[valid] = predict_proba3(model, arrays["features"][valid])
        models.append(model)
        audit.append({
            "view": view,
            "left_out_group": group,
            "train_seeds": sorted(set(map(int, arrays["seed"][train]))),
            "valid_seeds": sorted(set(map(int, arrays["seed"][valid]))),
            "train_valid_seed_overlap": sorted(
                set(map(int, arrays["seed"][train]))
                & set(map(int, arrays["seed"][valid]))
            ),
            "train_valid_decision_overlap": sorted(
                set(map(int, arrays["decision"][train]))
                & set(map(int, arrays["decision"][valid]))
            ),
            "valid_class_counts": _class_counts(
                np.asarray(arrays["target_class"])[valid]
            ),
            **model_audit,
        })
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"{view} OOF probabilities are incomplete")
    return probabilities, models, audit


def ensemble_view_mean(models: list[Any], features: np.ndarray) -> np.ndarray:
    if not models:
        raise ValueError("view ensemble is empty")
    return np.mean(
        [predict_proba3(model, features) for model in models], axis=0,
        dtype=np.float64,
    ).astype(np.float32)


def lopo_coverage_diagnostic(
    arrays: Mapping[str, np.ndarray], probabilities: np.ndarray,
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    target = np.asarray(arrays["target_win_gain"], np.int8)
    opponent = np.asarray(arrays["opponent"], np.int64)
    positive = np.flatnonzero(target == 1)
    positive_opponents = sorted(set(map(int, opponent[positive])))
    cross_supported = len(positive_opponents) >= 2 and all(
        row["class_counts"]["win_gain"] > 0
        for row in audit if row["valid_class_counts"]["win_gain"] > 0
    )
    return {
        "mode": "unknown_opponent_lopo_coverage_diagnostic_only",
        "status": (
            "coverage_supported_diagnostic_only"
            if cross_supported else "unsupported_abstain"
        ),
        "win_gain_rows": int(len(positive)),
        "win_gain_opponents": positive_opponents,
        "positive_row_mean_predicted_win_gain": (
            float(np.mean(probabilities[positive, WIN_GAIN]))
            if len(positive) else 0.0
        ),
        "used_for_known_pool_choice_or_threshold": False,
        "audit": audit,
    }


def select_choices(
    arrays: Mapping[str, np.ndarray], probabilities: np.ndarray,
    params: Mapping[str, float],
) -> np.ndarray:
    """Select without reading outcome or margin labels."""

    probability = np.asarray(probabilities, np.float64)
    if probability.shape != (len(arrays["decision"]), 3):
        raise ValueError("three-class probabilities do not align with rows")
    risk_cap = float(params["max_regression_probability"])
    win_floor = float(params["min_win_gain_probability"])
    edit = np.asarray(arrays["edit"])
    choices = []
    for rows in base.decision_slices(np.asarray(arrays["decision"], np.int64)):
        keep = rows[edit[rows] == base.KEEP_EDIT]
        if len(keep) != 1:
            raise ValueError("selection requires exactly one KEEP per decision")
        chosen = int(keep[0])
        candidates = rows[edit[rows] != base.KEEP_EDIT]
        eligible = candidates[
            (probability[candidates, REGRESSION] <= risk_cap)
            & (probability[candidates, WIN_GAIN] > win_floor)
        ]
        if len(eligible):
            chosen = max(map(int, eligible), key=lambda row: (
                float(probability[row, WIN_GAIN] - probability[row, REGRESSION]),
                float(probability[row, WIN_GAIN]),
                -float(probability[row, REGRESSION]),
                -row,
            ))
        choices.append(chosen)
    return np.asarray(choices, np.int64)


def choice_metrics(
    arrays: Mapping[str, np.ndarray], choices: np.ndarray,
) -> dict[str, Any]:
    slices = base.decision_slices(np.asarray(arrays["decision"], np.int64))
    choices = np.asarray(choices, np.int64)
    if len(slices) != len(choices):
        raise ValueError("choices do not cover every decision")
    outcome_delta = []
    win_delta = []
    margins = []
    same_margins = []
    fired = []
    opponents = []
    for rows, choice in zip(slices, choices, strict=True):
        if not np.any(rows == choice):
            raise ValueError("choice escaped its own decision")
        keep = rows[np.asarray(arrays["edit"])[rows] == base.KEEP_EDIT]
        keep_at = int(keep[0])
        keep_outcome = int(arrays["outcome"][keep_at])
        chosen_outcome = int(arrays["outcome"][choice])
        delta = chosen_outcome - keep_outcome
        is_fired = int(arrays["edit"][choice]) != int(base.KEEP_EDIT)
        margin = float(arrays["delta_margin"][choice])
        outcome_delta.append(delta)
        win_delta.append(int(chosen_outcome == 2) - int(keep_outcome == 2))
        margins.append(margin)
        same_margins.append(margin if is_fired and delta == 0 else 0.0)
        fired.append(is_fired)
        opponents.append(int(arrays["opponent"][choice]))
    outcome_values = np.asarray(outcome_delta, np.int16)
    win_values = np.asarray(win_delta, np.int16)
    opponent_regressions = {
        str(opponent): int(sum(
            value < 0 for value, group in zip(outcome_values, opponents, strict=True)
            if group == opponent
        )) for opponent in sorted(set(opponents))
    }
    return {
        "decisions": len(slices),
        "KEEP_candidates_evaluated": len(slices),
        "fires": int(sum(fired)),
        "raw_win_delta": int(win_values.sum()),
        "wins_gained": int(np.count_nonzero(win_values > 0)),
        "wins_lost": int(np.count_nonzero(win_values < 0)),
        "outcome_delta_sum": int(outcome_values.sum()),
        "outcome_upgrades": int(np.count_nonzero(outcome_values > 0)),
        "outcome_regressions": int(np.count_nonzero(outcome_values < 0)),
        "same_outcome_margin_delta_sum": float(np.sum(same_margins)),
        "selected_margin_delta_sum": float(np.sum(margins)),
        "opponent_outcome_regressions": opponent_regressions,
        "all_decisions_evaluated": True,
        "choices_within_source_panels": True,
    }


def _thresholds(values: np.ndarray, *, risk: bool) -> tuple[float, ...]:
    data = np.asarray(values, np.float64)
    points = [0.0, 1.0]
    if len(data):
        points.extend(map(float, np.quantile(data, [0.25, 0.5, 0.75, 0.9])))
        if not risk:
            points.append(float(np.max(data)))
    return tuple(sorted(set(points)))


def calibrate(
    arrays: Mapping[str, np.ndarray], probabilities: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    edit = np.asarray(arrays["edit"])
    candidate = edit != base.KEEP_EDIT
    trials = []
    choices_by_trial = []
    for risk_cap in _thresholds(probabilities[candidate, REGRESSION], risk=True):
        for win_floor in _thresholds(
            probabilities[candidate, WIN_GAIN], risk=False,
        ):
            params = {
                "max_regression_probability": risk_cap,
                "min_win_gain_probability": win_floor,
            }
            choices = select_choices(arrays, probabilities, params)
            metrics = {**params, **choice_metrics(arrays, choices)}
            trials.append(metrics)
            choices_by_trial.append(choices)
    allowed = [index for index, row in enumerate(trials) if (
        row["wins_lost"] == 0
        and row["outcome_regressions"] == 0
        and not any(row["opponent_outcome_regressions"].values())
    )]
    if not allowed:
        raise RuntimeError("calibration lost the KEEP-only fallback")
    best = max(allowed, key=lambda index: (
        trials[index]["raw_win_delta"],
        trials[index]["wins_gained"],
        trials[index]["outcome_delta_sum"],
        trials[index]["outcome_upgrades"],
        trials[index]["same_outcome_margin_delta_sum"],
        trials[index]["selected_margin_delta_sum"],
        -trials[index]["fires"],
        -trials[index]["max_regression_probability"],
        trials[index]["min_win_gain_probability"],
    ))
    chosen = trials[best]
    return {
        "max_regression_probability": chosen["max_regression_probability"],
        "min_win_gain_probability": chosen["min_win_gain_probability"],
        "outer_train_inner_OOF_metrics": chosen,
    }, choices_by_trial[best], trials


def oracle_headroom(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    upgrades = raw_win_gain = outcome_gain = same_margin = 0.0
    slices = base.decision_slices(np.asarray(arrays["decision"], np.int64))
    for rows in slices:
        keep = rows[np.asarray(arrays["edit"])[rows] == base.KEEP_EDIT]
        keep_at = int(keep[0])
        best = max(map(int, rows), key=lambda row: (
            int(arrays["outcome"][row]), float(arrays["delta_margin"][row]), -row,
        ))
        keep_outcome = int(arrays["outcome"][keep_at])
        best_outcome = int(arrays["outcome"][best])
        upgrades += int(best_outcome > keep_outcome)
        raw_win_gain += int(best_outcome == 2) - int(keep_outcome == 2)
        outcome_gain += best_outcome - keep_outcome
        if best_outcome == keep_outcome:
            same_margin += max(0.0, float(arrays["delta_margin"][best]))
    return {
        "decisions": len(slices),
        "decisions_with_outcome_upgrade": int(upgrades),
        "oracle_raw_win_gain": int(raw_win_gain),
        "oracle_outcome_delta_sum": int(outcome_gain),
        "same_outcome_positive_margin_headroom_sum": float(same_margin),
    }


def evaluate(
    arrays: Mapping[str, np.ndarray], *, trees: int = 48,
    random_seed: int = 20260829,
) -> dict[str, Any]:
    input_audit = base.validate_arrays(arrays)
    a, a_source = outcome_arrays(arrays)
    if tuple(sorted(set(map(int, a["seed"])))) != FORMAL_SEEDS:
        raise ValueError("v1 requires all eight pre-registered train seeds")
    mapping = seed_fold_map()
    row_folds = np.asarray([mapping[int(seed)] for seed in a["seed"]], np.int8)
    chosen_by_decision: dict[int, int] = {}
    folds = []
    for outer_fold in range(4):
        train_rows = np.flatnonzero(row_folds != outer_fold)
        valid_rows = np.flatnonzero(row_folds == outer_fold)
        train = _subset(a, train_rows)
        valid = _subset(a, valid_rows)
        train_seed_folds = np.asarray([
            mapping[int(seed)] for seed in train["seed"]
        ], np.int8)
        lopo_oof, _, lopo_audit = group_oof_view(
            train, np.asarray(train["opponent"]), "LOPO", trees,
            random_seed + outer_fold * 100_003 + 101,
        )
        seed_oof, seed_models, seed_audit = group_oof_view(
            train, train_seed_folds, "leave-one-seed-fold-out", trees,
            random_seed + outer_fold * 100_003 + 10_101,
        )
        calibration, _, _ = calibrate(train, seed_oof)
        valid_probability = ensemble_view_mean(seed_models, valid["features"])
        params = {
            "max_regression_probability": calibration["max_regression_probability"],
            "min_win_gain_probability": calibration["min_win_gain_probability"],
        }
        valid_choices = select_choices(valid, valid_probability, params)
        outer_metrics = choice_metrics(valid, valid_choices)
        valid_slices = base.decision_slices(valid["decision"])
        for rows, choice in zip(valid_slices, valid_choices, strict=True):
            decision = int(valid["decision"][rows[0]])
            chosen_by_decision[decision] = int(valid_rows[int(choice)])
        folds.append({
            "left_out_seed_fold": outer_fold,
            "train_seeds": sorted(set(map(int, train["seed"]))),
            "valid_seeds": sorted(set(map(int, valid["seed"]))),
            "train_valid_seed_overlap": [],
            "outer_labels_used_for_threshold_calibration": False,
            "threshold_source": (
                "outer-train leave-one-seed-fold-out OOF model means only"
            ),
            "inner_OOF_audit": {
                "leave_one_seed_fold_out": seed_audit,
                "individual_tree_probabilities_treated_as_independent": False,
            },
            "unknown_opponent_lopo": lopo_coverage_diagnostic(
                train, lopo_oof, lopo_audit,
            ),
            "chosen_thresholds": calibration,
            "outer_metrics_all_decisions": outer_metrics,
            "outer_oracle_headroom_all_decisions": oracle_headroom(valid),
            "outer_prediction_view_models": {
                "seed_fold_models": len(seed_models),
                "within_model_tree_probabilities_reduced_by": "mean",
                "mode": "known_pool_loso_primary",
            },
        })
    slices = base.decision_slices(a["decision"])
    decision_order = [int(a["decision"][rows[0]]) for rows in slices]
    if set(decision_order) != set(chosen_by_decision):
        raise RuntimeError("outer folds did not choose every A decision exactly once")
    choices = np.asarray([chosen_by_decision[value] for value in decision_order], np.int64)
    if sum(fold["outer_metrics_all_decisions"]["decisions"] for fold in folds) != len(slices):
        raise AssertionError("outer evaluation lost full decision coverage")
    return {
        "schema": SCHEMA,
        "status": "strict_four_seed_fold_outcome_multiclass_A_complete",
        "scheme": (
            "one shared three-class A/R0 model: regression/neutral/win_gain "
            "versus KEEP; known-pool seed-fold generalization is primary; "
            "unknown-opponent LOPO is coverage diagnostic only"
        ),
        "seed_folds": [list(values) for values in SEED_FOLDS],
        "outer_labels_used_for_threshold_calibration": False,
        "phase_enabled": False,
        "primary_mode": "known_pool_loso",
        "unknown_opponent_mode": "lopo_coverage_diagnostic_or_abstain",
        "model_params": model_params(trees, random_seed),
        "folds": folds,
        "all_decision_metrics": choice_metrics(a, choices),
        "all_decision_oracle_headroom": oracle_headroom(a),
        "A_choices": choices,
        "A_source_indices": a_source,
        "input_audit": input_audit,
        "A_rows": int(len(a["decision"])),
        "A_decisions": len(slices),
        "objective_contract": {
            "primary": "target_win_gain and raw win delta",
            "secondary": (
                "ordinal outcome delta, then same-outcome margin delta"
            ),
            "fallback": "KEEP",
            "oracle_upgrade_subset_used_for_evaluation": False,
        },
    }
