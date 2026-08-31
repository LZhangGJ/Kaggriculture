#!/usr/bin/env python3
"""Locked v2.1 formal integration for the uniform top-outcome objective.

The frozen v2 evaluator remains authoritative for folds, inner-OOF threshold
calibration, controls, acceptance, and outer-label boundaries.  Only its model
fit and the explicit no-credible-gain fail-safe are replaced, inside an
exception-safe scoped adapter.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import train_phase_challenger_bilinear_listwise_A_v2 as frozen
import train_phase_challenger_bilinear_listwise_optimizer_stability_v2_1_locked as locked


SCHEMA = "phase-challenger-bilinear-listwise-uniform-A-v2.1-locked"
LISTWISE_OBJECTIVE = "signal_only_uniform_top_outcome_cross_entropy"
LISTWISE_SEMANTICS = "top-outcome-set classification; not full ordinal"
SOLVER = "L-BFGS-B-1000"
SOLVER_OPTIONS = dict(locked.SOLVERS[SOLVER]["options"])
GRADIENT_INF_TOLERANCE = locked.GRADIENT_INF_TOLERANCE
OBJECTIVE_TOLERANCE = locked.OBJECTIVE_TOLERANCE
LISTWISE_L2_PER_CELL = locked.LISTWISE_L2_PER_CELL
RISK_L2_PER_CELL = locked.RISK_L2_PER_CELL

DEFAULT_CONFIG: dict[str, Any] = {
    **frozen.DEFAULT_CONFIG,
    "optimizer": "scipy.optimize.L-BFGS-B",
    "optimizer_initialization": "all-zero",
    "optimizer_maxiter": 1000,
    "optimizer_ftol": SOLVER_OPTIONS["ftol"],
    "optimizer_gtol": SOLVER_OPTIONS["gtol"],
    "optimizer_maxls": SOLVER_OPTIONS["maxls"],
    "optimizer_fail_closed_gates": (
        "scipy_success", "finite", "gradient_inf_at_most_1e-6",
        "objective_nonincreasing", "objective_decreased_or_initially_stationary",
    ),
    "ordinal_objective": LISTWISE_OBJECTIVE,
    "ordinal_semantics": LISTWISE_SEMANTICS,
    "listwise_l2": LISTWISE_L2_PER_CELL,
    "risk_l2": RISK_L2_PER_CELL,
    "regularization_semantics": "fixed per paired-cell after division by all paired-cell mass",
    "listwise_no_signal_group_weight": 0.0,
}

EXPECTED_ACCEPTANCE_KEYS = {
    "each_fold_each_opponent_zero_outcome_regressions_and_wins_lost",
    "at_least_two_headroom_folds_gain_at_least_one_win",
    "headroom_folds_with_gain", "total_raw_win_gain_at_least_two",
    "gain_covers_at_least_two_seeds", "gain_covers_at_least_two_opponents",
    "same_outcome_margin_nonnegative",
    "primary_rank_Pareto_over_SHA_prior_and_all_state_shuffles",
    "full_outer_decision_coverage", "outer_labels_evaluation_acceptance_only",
    "passed",
}
EXPECTED_CONTROL_NAMES = {
    "primary", "bilinear_no_SHA", "SHA_prior_only", "KEEP",
    *(f"state_shuffle_{seed}" for seed in frozen.SHUFFLE_SEEDS),
}
_FROZEN_FIT_PREDICT = frozen.fit_predict
_FROZEN_CALIBRATE_THRESHOLD = frozen.calibrate_threshold


def _gradient_inf(gradient: np.ndarray) -> float:
    return float(np.max(np.abs(gradient))) if len(gradient) else 0.0


def _strict_optimize(problem: Mapping[str, Any], head: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the pre-registered solver and raise unless every gate passes."""

    objective = problem["fun_jac"]
    initial = np.zeros(int(problem["width"]), np.float64)
    initial_objective, initial_gradient = objective(initial)
    if not np.isfinite(initial_objective) or not np.isfinite(initial_gradient).all():
        raise RuntimeError(f"{head} initial objective/gradient is non-finite")
    fit = minimize(
        objective, initial, method="L-BFGS-B", jac=True,
        options=dict(SOLVER_OPTIONS),
    )
    coefficient = np.asarray(fit.x, np.float64)
    final_objective, final_gradient = objective(coefficient)
    finite = bool(
        np.isfinite(final_objective) and np.isfinite(final_gradient).all()
        and np.isfinite(coefficient).all()
    )
    initial_gradient_inf = _gradient_inf(initial_gradient)
    final_gradient_inf = _gradient_inf(final_gradient) if finite else math.inf
    decrease = float(initial_objective - final_objective) if finite else -math.inf
    initially_stationary = initial_gradient_inf <= GRADIENT_INF_TOLERANCE
    gates = {
        "scipy_success": bool(fit.success),
        "finite": finite,
        "gradient_inf_at_most_tolerance": finite and final_gradient_inf <= GRADIENT_INF_TOLERANCE,
        "objective_nonincreasing": finite and decrease >= -OBJECTIVE_TOLERANCE,
        "objective_decreased_or_initially_stationary": finite and (
            decrease > OBJECTIVE_TOLERANCE or initially_stationary
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"{head} {SOLVER} failed closed: status={getattr(fit, 'status', -1)}; "
            f"message={getattr(fit, 'message', '')}; gates={gates}; "
            f"gradient_inf={final_gradient_inf}; decrease={decrease}"
        )
    return coefficient, {
        "kind": f"{head}-{SOLVER}", "optimizer_called": True,
        "success": True, "gate_passed": True, "gates": gates,
        "status": int(fit.status), "message": str(fit.message),
        "nit": int(getattr(fit, "nit", -1)),
        "nfev": int(getattr(fit, "nfev", -1)),
        "njev": int(getattr(fit, "njev", -1)),
        "objective_initial": float(initial_objective),
        "objective_final": float(final_objective),
        "objective_decrease": decrease,
        "gradient_inf_initial": initial_gradient_inf,
        "gradient_inf_final": final_gradient_inf,
        "total_weight": float(problem["total_weight"]),
        "signal_weight": float(problem["signal_weight"]),
        "signal_weight_rate": float(problem["signal_weight_rate"]),
        "l2_per_cell": float(problem["l2_per_cell"]),
    }


def _fit_listwise(
    x: np.ndarray, arrays: Mapping[str, np.ndarray], rows: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    problem = locked.listwise_problem(
        x, arrays, rows, objective=LISTWISE_OBJECTIVE,
        l2_per_cell=LISTWISE_L2_PER_CELL,
    )
    if int(problem["signal_groups"]) == 0:
        coefficient = np.zeros(x.shape[1], np.float64)
        return coefficient, {
            "kind": "constant-all-equal-listwise-safe",
            "optimizer_called": False, "success": True, "gate_passed": True,
            "coefficient_exact_zero": bool(np.array_equal(coefficient, np.zeros_like(coefficient))),
            "signal_groups": 0,
            "total_weight": float(problem["total_weight"]),
            "signal_weight": 0.0, "signal_weight_rate": 0.0,
            "l2_per_cell": LISTWISE_L2_PER_CELL,
        }
    coefficient, audit = _strict_optimize(problem, "uniform-top-outcome-listwise")
    audit["signal_groups"] = int(problem["signal_groups"])
    return coefficient, audit


def _fit_risk(
    x: np.ndarray, arrays: Mapping[str, np.ndarray], rows: np.ndarray,
) -> tuple[np.ndarray | None, float, dict[str, Any]]:
    problem = locked.risk_problem(x, arrays, rows, l2_per_cell=RISK_L2_PER_CELL)
    if problem["constant"] is not None:
        constant = float(problem["constant"])
        if constant not in (0.0, 1.0):
            raise RuntimeError("single-class risk shortcut lost its observed class")
        return None, constant, {
            "kind": "constant-single-class-risk-safe",
            "optimizer_called": False, "success": True, "gate_passed": True,
            "constant_probability": constant,
            "total_weight": float(problem["total_weight"]),
            "signal_weight": float(problem["signal_weight"]),
            "l2_per_cell": RISK_L2_PER_CELL,
        }
    coefficient, audit = _strict_optimize(problem, "independent-risk-logistic")
    return coefficient, math.nan, audit


def fit_predict(
    arrays: Mapping[str, np.ndarray], train: np.ndarray, predict: np.ndarray,
    *, include_sha: bool, shuffle_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit locked heads and retain the frozen v2 prediction semantics."""

    alpha = float(frozen.DEFAULT_CONFIG["sha_prior_alpha"])
    if include_sha:
        train_prior, _ = frozen._crossfit_prior_features(arrays, train, alpha)
        predict_prior, _ = frozen._predict_prior_features(arrays, train, predict, alpha)
    else:
        train_prior = predict_prior = None
    train_x, predict_x = frozen._projected_features(
        arrays, train, predict, train_prior=train_prior, predict_prior=predict_prior,
        shuffle_seed=shuffle_seed,
    )
    coefficient, listwise_audit = _fit_listwise(train_x, arrays, train)
    risk_coefficient, risk_constant, risk_audit = _fit_risk(train_x, arrays, train)

    raw_score = predict_x @ coefficient
    score = np.zeros(len(predict), np.float64)
    position = {int(row): at for at, row in enumerate(predict)}
    edit_all = np.asarray(arrays["edit"])
    for group in frozen.decision_slices(np.asarray(arrays["decision"]), predict):
        local = np.asarray([position[int(row)] for row in group], np.int64)
        logits = raw_score[local]
        probability = np.exp(logits - np.max(logits))
        probability /= probability.sum()
        keep_local = local[edit_all[group] == frozen.KEEP_EDIT]
        if len(keep_local) != 1:
            raise ValueError("prediction group lost KEEP")
        keep_position = int(np.flatnonzero(local == keep_local[0])[0])
        score[local] = probability - probability[keep_position]
        score[int(keep_local[0])] = 0.0
    score = np.clip(
        score, -float(frozen.DEFAULT_CONFIG["selection_score_symmetric_clip"]),
        float(frozen.DEFAULT_CONFIG["selection_score_symmetric_clip"]),
    )
    if risk_coefficient is None:
        risk_probability = np.full(len(predict), risk_constant, np.float64)
    else:
        design = np.concatenate((predict_x, np.ones((len(predict), 1))), axis=1)
        risk_probability = expit(design @ risk_coefficient)
    risk_probability[edit_all[predict] == frozen.KEEP_EDIT] = 0.0
    if not np.isfinite(score).all() or not np.isfinite(risk_probability).all():
        raise RuntimeError("locked prediction produced non-finite values")
    return score, risk_probability, {
        "feature_width": int(train_x.shape[1]),
        "include_sha_soft_prior": bool(include_sha), "shuffle_seed": shuffle_seed,
        "listwise": listwise_audit, "risk": risk_audit,
    }


def calibrate_threshold(
    arrays: Mapping[str, np.ndarray], rows: np.ndarray,
    score: np.ndarray, risk_probability: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Reuse frozen calibration and explicitly force KEEP when gain is absent."""

    threshold, audit = _FROZEN_CALIBRATE_THRESHOLD(arrays, rows, score, risk_probability)
    audit = dict(audit)
    metrics = audit["inner_OOF_metrics"]
    no_credible_gain = int(metrics["wins_gained"]) == 0 and int(metrics["outcome_upgrades"]) == 0
    if no_credible_gain:
        threshold = math.inf
        choices = frozen.select_choices(arrays, rows, score, risk_probability, threshold)
        if not np.all(np.asarray(arrays["edit"])[choices] == frozen.KEEP_EDIT):
            raise RuntimeError("positive-infinity threshold did not force every choice to KEEP")
    audit.update({
        "score_floor": float(threshold),
        "no_credible_ordinal_gain_forces_KEEP": bool(no_credible_gain),
        "no_credible_gain_shortcut": bool(no_credible_gain),
        "all_choices_KEEP_when_shortcut": bool(no_credible_gain),
    })
    return float(threshold), audit


@contextmanager
def scoped_frozen_flow_adapters() -> Iterator[None]:
    """Temporarily adapt frozen globals and restore their exact prior objects."""

    previous_fit = frozen.fit_predict
    previous_calibration = frozen.calibrate_threshold
    frozen.fit_predict = fit_predict
    frozen.calibrate_threshold = calibrate_threshold
    try:
        yield
    finally:
        frozen.calibrate_threshold = previous_calibration
        frozen.fit_predict = previous_fit


def _assert_frozen_flow_contract(result: Mapping[str, Any]) -> None:
    if tuple(tuple(values) for values in result.get("seed_folds", ())) != frozen.SEED_FOLDS:
        raise RuntimeError("frozen outer folds changed")
    if set(result.get("controls", {}).get("rank_metrics", {})) != EXPECTED_CONTROL_NAMES:
        raise RuntimeError("frozen control set changed")
    acceptance = result.get("acceptance_gate", {})
    if set(acceptance) != EXPECTED_ACCEPTANCE_KEYS:
        raise RuntimeError("frozen acceptance conditions changed")
    folds = list(result.get("folds", ()))
    if len(folds) != 4 or [int(fold.get("outer_fold", -1)) for fold in folds] != list(range(4)):
        raise RuntimeError("frozen four-fold report changed")
    if any(
        fold.get("threshold_source") != "outer-train inner seed-fold OOF only"
        or fold.get("outer_labels_used_for_model_or_threshold") is not False
        or fold.get("train_valid_seed_overlap") != []
        for fold in folds
    ):
        raise RuntimeError("frozen threshold or outer-label boundary changed")
    if result.get("outer_labels_used_for_model_or_threshold") is not False:
        raise RuntimeError("outer labels entered model fitting or threshold calibration")
    if "A_choices" not in result:
        raise RuntimeError("frozen evaluator did not return complete choices")


def evaluate(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Execute the unchanged frozen screen under the locked model adapters."""

    with scoped_frozen_flow_adapters():
        result = frozen.evaluate(arrays)
    _assert_frozen_flow_contract(result)
    result = dict(result)
    result["schema"] = SCHEMA
    result["config"] = dict(DEFAULT_CONFIG)
    result["optimizer_contract"] = {
        "listwise_objective": LISTWISE_OBJECTIVE,
        "listwise_semantics": LISTWISE_SEMANTICS,
        "listwise_no_signal_group_weight": 0.0,
        "loss_denominator": "all paired-cell mass",
        "listwise_l2_per_cell": LISTWISE_L2_PER_CELL,
        "risk_l2_per_cell": RISK_L2_PER_CELL,
        "solver": SOLVER, "solver_options": dict(SOLVER_OPTIONS),
        "gradient_inf_tolerance": GRADIENT_INF_TOLERANCE,
        "objective_tolerance": OBJECTIVE_TOLERANCE,
        "all_equal_listwise_skips_optimizer": True,
        "single_class_risk_skips_optimizer": True,
        "no_credible_inner_gain_forces_KEEP": True,
    }
    result["frozen_flow_contract"] = {
        "folds_unchanged": True, "controls_unchanged": True,
        "threshold_source_unchanged": True, "acceptance_unchanged": True,
        "scoped_adapter_restored": True,
    }
    ordinal = dict(result["ordinal_contract"])
    ordinal.update({
        "objective": LISTWISE_OBJECTIVE,
        "semantics": LISTWISE_SEMANTICS,
        "full_ordinal": False,
        "win_tie_loss_full_order_modeled": False,
    })
    result["ordinal_contract"] = ordinal
    return result


__all__ = [
    "DEFAULT_CONFIG", "EXPECTED_ACCEPTANCE_KEYS", "EXPECTED_CONTROL_NAMES",
    "GRADIENT_INF_TOLERANCE", "LISTWISE_L2_PER_CELL", "LISTWISE_OBJECTIVE",
    "RISK_L2_PER_CELL", "SCHEMA", "SOLVER", "SOLVER_OPTIONS",
    "calibrate_threshold", "evaluate", "fit_predict",
    "scoped_frozen_flow_adapters",
]
