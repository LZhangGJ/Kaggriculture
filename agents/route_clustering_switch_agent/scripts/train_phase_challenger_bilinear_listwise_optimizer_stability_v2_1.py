#!/usr/bin/env python3
"""Deterministic optimization-only study for bilinear listwise v2.1.

This module never predicts an outer fold and never computes selector outcome
metrics.  It only fits the frozen outer-train objectives and records numerical
convergence.  The original tied-best-mass loss is retained as a non-convex
baseline; uniform-over-best cross entropy is the convex candidate.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Callable, Mapping

import numpy as np
from scipy.optimize import minimize

import train_phase_challenger_bilinear_listwise_A_v2 as frozen


SCHEMA = "bilinear-listwise-optimizer-stability-v2.1"
GRADIENT_INF_TOLERANCE = 1e-6
OBJECTIVE_TOLERANCE = 1e-10
COEFFICIENT_LINF_TOLERANCE = 2e-4
MULTISTART_OBJECTIVE_TOLERANCE = 1e-8
SOLVERS: dict[str, dict[str, Any]] = {
    "L-BFGS-B-1000": {
        "method": "L-BFGS-B",
        "maxiter": 1000,
        "options": {"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-8, "maxls": 50},
        "uses_hessp": False,
    },
    "trust-krylov-500": {
        "method": "trust-krylov",
        "maxiter": 500,
        "options": {"maxiter": 500, "gtol": 1e-8},
        "uses_hessp": True,
    },
}
INITIALIZATIONS = ("zero", "sha256-sign-0.01")


def convexity_audit() -> dict[str, Any]:
    return {
        "tied_best_probability_mass": {
            "convex": False,
            "reason": "logsumexp(all)-logsumexp(tied-best) is a difference of convex functions",
            "counterexample": {
                "logits": [0.0, 0.0, 0.0],
                "best_mask": [1, 1, 0],
                "direction": [1.0, -1.0, 0.0],
                "directional_second_derivative": -1.0 / 3.0,
            },
            "semantics": "maximize total probability mass assigned to any tied-best candidate",
            "eligible_for_solver_lock": False,
        },
        "uniform_over_best_cross_entropy": {
            "convex": True,
            "strictly_convex_after_positive_L2_on_coefficients": True,
            "semantics": "cross entropy to a uniform distribution over tied-best candidates",
            "difference": "also penalizes unequal probability within the tied-best set",
            "eligible_for_solver_lock": True,
        },
        "independent_risk_logistic": {
            "convex": True,
            "semantics": "weighted binary logistic loss on non-KEEP candidates",
            "eligible_for_solver_lock": True,
        },
    }


def deterministic_initializations(width: int) -> dict[str, np.ndarray]:
    signs = np.empty(width, np.float64)
    for index in range(width):
        signs[index] = 1.0 if hashlib.sha256(f"v2.1-init|{width}|{index}".encode()).digest()[0] & 1 else -1.0
    return {
        "zero": np.zeros(width, np.float64),
        "sha256-sign-0.01": signs * (0.01 / math.sqrt(max(1, width))),
    }


def _group_layout(
    arrays: Mapping[str, np.ndarray], rows: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    groups = frozen.decision_slices(np.asarray(arrays["decision"]), rows)
    position = {int(row): at for at, row in enumerate(rows)}
    local = [np.asarray([position[int(row)] for row in group], np.int64) for group in groups]
    lengths = np.asarray([len(group) for group in local], np.int64)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    decision_weight = frozen._decision_weights(arrays, rows)
    row_weight = np.zeros(len(rows), np.float64)
    for original, loc in zip(groups, local, strict=True):
        row_weight[loc] = decision_weight[int(np.asarray(arrays["decision"])[original[0]])]
    return groups, starts, lengths, row_weight


def listwise_problem(
    x: np.ndarray, arrays: Mapping[str, np.ndarray], rows: np.ndarray,
    *, objective: str, l2: float,
) -> dict[str, Any]:
    if objective not in {"tied_best_probability_mass", "uniform_over_best_cross_entropy"}:
        raise ValueError(f"unknown listwise objective: {objective}")
    groups, starts, lengths, row_weight = _group_layout(arrays, rows)
    outcome = np.asarray(arrays["outcome"], np.int8)
    best_mask = np.zeros(len(rows), np.float64)
    uniform_target = np.zeros(len(rows), np.float64)
    position = {int(row): at for at, row in enumerate(rows)}
    signal_groups = 0
    for group in groups:
        local = np.asarray([position[int(row)] for row in group], np.int64)
        winners = local[outcome[group] == np.max(outcome[group])]
        best_mask[winners] = 1.0
        uniform_target[winners] = 1.0 / len(winners)
        signal_groups += len(set(map(int, outcome[group]))) > 1

    def probabilities(score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        maxima = np.maximum.reduceat(score, starts)
        exponent = np.exp(score - np.repeat(maxima, lengths))
        denominator = np.add.reduceat(exponent, starts)
        probability = exponent / np.repeat(denominator, lengths)
        best_score = np.where(best_mask > 0.0, score, -np.inf)
        best_maxima = np.maximum.reduceat(best_score, starts)
        best_exponent = np.exp(best_score - np.repeat(best_maxima, lengths)) * best_mask
        best_denominator = np.add.reduceat(best_exponent, starts)
        best_probability = best_exponent / np.repeat(best_denominator, lengths)
        log_all = maxima + np.log(denominator)
        log_best = best_maxima + np.log(best_denominator)
        return probability, best_probability, log_all, log_best

    def fun_jac(coef: np.ndarray) -> tuple[float, np.ndarray]:
        score = x @ coef
        probability, best_probability, log_all, log_best = probabilities(score)
        if objective == "tied_best_probability_mass":
            per_group = log_all - log_best
            residual = probability - best_probability
        else:
            per_group = log_all - np.add.reduceat(uniform_target * score, starts)
            residual = probability - uniform_target
        mass = row_weight[starts]
        value = float(np.dot(mass, per_group) + 0.5 * l2 * np.dot(coef, coef))
        gradient = x.T @ (row_weight * residual) + l2 * coef
        return value, gradient

    def hessp(coef: np.ndarray, vector: np.ndarray) -> np.ndarray:
        score = x @ coef
        probability, best_probability, _, _ = probabilities(score)
        direction = x @ vector
        all_mean = np.add.reduceat(probability * direction, starts)
        curvature = probability * (direction - np.repeat(all_mean, lengths))
        if objective == "tied_best_probability_mass":
            best_mean = np.add.reduceat(best_probability * direction, starts)
            curvature -= best_probability * (direction - np.repeat(best_mean, lengths))
        return x.T @ (row_weight * curvature) + l2 * vector

    return {
        "name": objective,
        "width": int(x.shape[1]),
        "fun_jac": fun_jac,
        "hessp": hessp,
        "signal_groups": int(signal_groups),
        "rows": int(len(rows)),
        "groups": int(len(groups)),
        "convex": objective == "uniform_over_best_cross_entropy",
    }


def risk_problem(
    x: np.ndarray, arrays: Mapping[str, np.ndarray], rows: np.ndarray, *, l2: float,
) -> dict[str, Any]:
    _, risk = frozen._relative_labels(arrays, rows)
    edit = np.asarray(arrays["edit"])
    candidate = np.flatnonzero(edit[rows] != frozen.KEEP_EDIT)
    y = risk[rows][candidate]
    values = x[candidate]
    weights = frozen._candidate_row_weights(arrays, rows)[rows][candidate]
    design = np.concatenate((values, np.ones((len(values), 1))), axis=1)
    constant = float(y[0]) if len(np.unique(y)) == 1 else None

    def fun_jac(coef: np.ndarray) -> tuple[float, np.ndarray]:
        logit = design @ coef
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -700.0, 700.0)))
        value = float(np.dot(weights, np.logaddexp(0.0, logit) - y * logit))
        value += 0.5 * l2 * float(np.dot(coef[:-1], coef[:-1]))
        gradient = design.T @ (weights * (probability - y))
        gradient[:-1] += l2 * coef[:-1]
        return value, gradient

    def hessp(coef: np.ndarray, vector: np.ndarray) -> np.ndarray:
        logit = design @ coef
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -700.0, 700.0)))
        curvature = weights * probability * (1.0 - probability)
        result = design.T @ (curvature * (design @ vector))
        result[:-1] += l2 * vector[:-1]
        return result

    return {
        "name": "independent_risk_logistic",
        "width": int(design.shape[1]),
        "fun_jac": fun_jac,
        "hessp": hessp,
        "signal_groups": int(len(np.unique(y)) > 1),
        "rows": int(len(y)),
        "groups": None,
        "convex": True,
        "constant": constant,
    }


def run_solver(problem: Mapping[str, Any], solver_name: str, initialization: np.ndarray) -> dict[str, Any]:
    config = SOLVERS[solver_name]
    fun_jac: Callable[[np.ndarray], tuple[float, np.ndarray]] = problem["fun_jac"]
    initial_objective, initial_gradient = fun_jac(initialization)
    started = time.perf_counter()
    try:
        kwargs: dict[str, Any] = {
            "method": config["method"], "jac": True, "options": dict(config["options"]),
        }
        if config["uses_hessp"]:
            kwargs["hessp"] = problem["hessp"]
        fit = minimize(fun_jac, initialization.copy(), **kwargs)
        final_objective, final_gradient = fun_jac(np.asarray(fit.x, np.float64))
        coefficient = np.asarray(fit.x, np.float64)
        finite = bool(
            np.isfinite(final_objective) and np.isfinite(final_gradient).all()
            and np.isfinite(coefficient).all()
        )
        gradient_inf = float(np.max(np.abs(final_gradient))) if len(final_gradient) else 0.0
        decrease = float(initial_objective - final_objective)
        initially_stationary = float(np.max(np.abs(initial_gradient))) <= GRADIENT_INF_TOLERANCE
        gates = {
            "scipy_success": bool(fit.success),
            "finite": finite,
            "gradient_inf_at_most_tolerance": finite and gradient_inf <= GRADIENT_INF_TOLERANCE,
            "objective_nonincreasing": finite and decrease >= -OBJECTIVE_TOLERANCE,
            "objective_decreased_or_initially_stationary": finite and (
                decrease > OBJECTIVE_TOLERANCE or initially_stationary
            ),
        }
        return {
            "solver": solver_name,
            "success": bool(fit.success),
            "gate_passed": all(gates.values()),
            "gates": gates,
            "status": int(fit.status), "message": str(fit.message),
            "nit": int(getattr(fit, "nit", -1)), "nfev": int(getattr(fit, "nfev", -1)),
            "objective_initial": float(initial_objective),
            "objective_final": float(final_objective),
            "objective_decrease": decrease,
            "gradient_inf_initial": float(np.max(np.abs(initial_gradient))),
            "gradient_inf_final": gradient_inf,
            "wall_seconds": time.perf_counter() - started,
            "coefficient_sha256": hashlib.sha256(coefficient.tobytes()).hexdigest(),
            "_coefficient": coefficient,
        }
    except Exception as error:  # A study records a failed candidate; it never promotes it.
        return {
            "solver": solver_name, "success": False, "gate_passed": False,
            "gates": {"exception_free": False}, "status": -1,
            "message": f"{type(error).__name__}: {error}", "nit": -1, "nfev": -1,
            "objective_initial": float(initial_objective), "objective_final": math.nan,
            "objective_decrease": math.nan,
            "gradient_inf_initial": float(np.max(np.abs(initial_gradient))),
            "gradient_inf_final": math.inf,
            "wall_seconds": time.perf_counter() - started,
            "coefficient_sha256": None, "_coefficient": None,
        }


def study_problem(problem: Mapping[str, Any], solver_names: tuple[str, ...]) -> dict[str, Any]:
    if problem.get("constant") is not None:
        return {
            "constant_safe": True, "constant_probability": problem["constant"],
            "solvers": {}, "stable": True,
        }
    initializations = deterministic_initializations(int(problem["width"]))
    solver_reports: dict[str, Any] = {}
    for solver_name in solver_names:
        runs = []
        for initialization_name in INITIALIZATIONS:
            report = run_solver(problem, solver_name, initializations[initialization_name])
            report["initialization"] = initialization_name
            runs.append(report)
        coefficients = [run["_coefficient"] for run in runs]
        if all(value is not None for value in coefficients):
            coefficient_linf = float(np.max(np.abs(coefficients[0] - coefficients[1])))
            objective_range = float(max(run["objective_final"] for run in runs) - min(run["objective_final"] for run in runs))
        else:
            coefficient_linf = objective_range = math.inf
        consistency = {
            "coefficient_linf": coefficient_linf,
            "objective_range": objective_range,
            "coefficient_linf_at_most_tolerance": coefficient_linf <= COEFFICIENT_LINF_TOLERANCE,
            "objective_range_at_most_tolerance": objective_range <= MULTISTART_OBJECTIVE_TOLERANCE,
        }
        for run in runs:
            run.pop("_coefficient", None)
        solver_reports[solver_name] = {
            "runs": runs, "multistart_consistency": consistency,
            "stable": all(run["gate_passed"] for run in runs) and all(
                value for key, value in consistency.items() if key.endswith("_at_most_tolerance")
            ),
        }
    return {
        "constant_safe": False, "solvers": solver_reports,
        "stable": any(report["stable"] for report in solver_reports.values()),
    }


def _preferred_solver(folds: list[dict[str, Any]], head: str) -> str | None:
    # Pre-registered simplicity preference; no timing or outcome label chooses the solver.
    for solver in ("L-BFGS-B-1000", "trust-krylov-500"):
        if all(fold[head]["solvers"].get(solver, {}).get("stable", False) for fold in folds):
            return solver
    return None


def optimization_only_study(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    input_audit = frozen.validate_arrays(arrays)
    row_fold = frozen._row_folds(np.asarray(arrays["seed"]))
    all_rows = np.arange(len(row_fold), dtype=np.int64)
    fold_reports: list[dict[str, Any]] = []
    for outer_fold in range(4):
        train = all_rows[row_fold != outer_fold]
        prior, _ = frozen._crossfit_prior_features(
            arrays, train, float(frozen.DEFAULT_CONFIG["sha_prior_alpha"]),
        )
        train_x, _ = frozen._projected_features(
            arrays, train, train, train_prior=prior, predict_prior=prior,
            shuffle_seed=None,
        )
        tied = listwise_problem(
            train_x, arrays, train, objective="tied_best_probability_mass",
            l2=float(frozen.DEFAULT_CONFIG["listwise_l2"]),
        )
        uniform = listwise_problem(
            train_x, arrays, train, objective="uniform_over_best_cross_entropy",
            l2=float(frozen.DEFAULT_CONFIG["listwise_l2"]),
        )
        risk = risk_problem(
            train_x, arrays, train, l2=float(frozen.DEFAULT_CONFIG["risk_l2"]),
        )
        fold_reports.append({
            "outer_fold": outer_fold,
            "training_rows": int(len(train)),
            "training_decisions": len(frozen.decision_slices(np.asarray(arrays["decision"]), train)),
            "training_feature_sha256": hashlib.sha256(train_x.tobytes()).hexdigest(),
            "tied_best_probability_mass": study_problem(tied, ("L-BFGS-B-1000",)),
            "uniform_over_best_cross_entropy": study_problem(
                uniform, ("L-BFGS-B-1000", "trust-krylov-500"),
            ),
            "independent_risk_logistic": study_problem(
                risk, ("L-BFGS-B-1000", "trust-krylov-500"),
            ),
            "outer_valid_prediction_executed": False,
            "outer_valid_outcome_metric_computed": False,
        })

    uniform_solver = _preferred_solver(fold_reports, "uniform_over_best_cross_entropy")
    risk_solver = _preferred_solver(fold_reports, "independent_risk_logistic")
    return {
        "schema": SCHEMA,
        "status": "optimization_only_stable" if uniform_solver and risk_solver else "optimization_only_unstable",
        "convexity_audit": convexity_audit(),
        "config": {
            "solvers": SOLVERS,
            "initializations": INITIALIZATIONS,
            "gradient_inf_tolerance": GRADIENT_INF_TOLERANCE,
            "objective_tolerance": OBJECTIVE_TOLERANCE,
            "coefficient_linf_tolerance": COEFFICIENT_LINF_TOLERANCE,
            "multistart_objective_tolerance": MULTISTART_OBJECTIVE_TOLERANCE,
            "solver_preference": ["L-BFGS-B-1000", "trust-krylov-500"],
        },
        "boundary": {
            "train_proxy_only": True,
            "outer_valid_predictions": False,
            "outer_valid_outcome_acceptance": False,
            "validation_opened": False, "sealed_opened": False,
            "choices_generated": False, "candidate_committed": False,
        },
        "input_audit": input_audit,
        "folds": fold_reports,
        "recommendation": {
            "listwise_objective": "uniform_over_best_cross_entropy" if uniform_solver else None,
            "listwise_solver": uniform_solver,
            "risk_solver": risk_solver,
            "tied_best_probability_mass_eligible": False,
            "selection_used_outer_outcome_metrics": False,
        },
    }


__all__ = [
    "INITIALIZATIONS", "SCHEMA", "SOLVERS", "convexity_audit",
    "deterministic_initializations", "listwise_problem", "optimization_only_study",
    "risk_problem", "run_solver", "study_problem",
]
