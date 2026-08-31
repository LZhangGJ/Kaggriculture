#!/usr/bin/env python3
"""Locked optimization-only study for bilinear listwise selector v2.1.

No held-out prediction, selector choice, or outcome acceptance metric is
computed here.  The frozen train-proxy rows are used only to construct the ten
unique training objectives exercised by nested four-fold fitting.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import time
from typing import Any, Callable, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import train_phase_challenger_bilinear_listwise_A_v2 as frozen


SCHEMA = "bilinear-listwise-optimizer-stability-v2.1-locked"
REFERENCE_OUTER_TOTAL_WEIGHT = 576.0
LISTWISE_L2_PER_CELL = 0.08 / REFERENCE_OUTER_TOTAL_WEIGHT
RISK_L2_PER_CELL = 0.08 / REFERENCE_OUTER_TOTAL_WEIGHT
GRADIENT_INF_TOLERANCE = 1e-6
OBJECTIVE_TOLERANCE = 1e-10
COEFFICIENT_LINF_TOLERANCE = 2e-4
MULTISTART_OBJECTIVE_TOLERANCE = 1e-8
INITIALIZATIONS = ("zero", "sha256-sign-0.01")
SOLVERS: dict[str, dict[str, Any]] = {
    "L-BFGS-B-1000": {
        "method": "L-BFGS-B",
        "options": {"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-8, "maxls": 50},
        "uses_hessp": False,
    },
    "trust-krylov-500": {
        "method": "trust-krylov",
        "options": {"maxiter": 500, "gtol": 1e-8},
        "uses_hessp": True,
    },
}


def convexity_audit() -> dict[str, Any]:
    tied_curvature = -1.0 / 3.0
    return {
        "tied_top_outcome_set_probability_mass": {
            "convex": False,
            "reason": "logsumexp(all)-logsumexp(top-set) is a difference of convex functions",
            "counterexample": {
                "logits": [0.0, 0.0, 0.0], "top_mask": [1, 1, 0],
                "direction": [1.0, -1.0, 0.0],
                "data_directional_second_derivative": tied_curvature,
                "with_locked_L2_directional_second_derivative": (
                    tied_curvature + 2.0 * LISTWISE_L2_PER_CELL
                ),
            },
            "eligible_for_solver_lock": False,
        },
        "signal_only_uniform_top_outcome_cross_entropy": {
            "convex": True,
            "strictly_convex_with_locked_positive_L2": True,
            "semantics": "top-outcome-set classification with a uniform soft target",
            "not_full_ordinal": True,
            "non_top_outcomes_are_not_ordered": True,
            "no_within_decision_signal_group_weight": 0.0,
            "eligible_for_solver_lock": True,
        },
        "independent_risk_logistic": {
            "convex": True,
            "signal_group_filter_applied": False,
            "eligible_for_solver_lock": True,
        },
        "deferred_v2_2_candidate": {
            "name": "strict-pair ordinal Bradley-Terry",
            "objective": "mean softplus(score_low-score_high) over strict outcome pairs per decision",
            "benefit": "convex, all-equal groups are zero, and win>tie>loss order is retained",
            "included_in_v2_1": False,
        },
    }


def regularization_audit() -> dict[str, Any]:
    return {
        "loss_reduction": "weighted sum divided by all paired-cell mass",
        "listwise_no_signal_contribution": "exactly zero before L2",
        "signal_rarity_semantics": "signal weight remains relative to all paired cells",
        "reference_outer_total_weight": REFERENCE_OUTER_TOTAL_WEIGHT,
        "legacy_lambda": 0.08,
        "legacy_lambda_semantics": "ambiguous fixed-prior versus per-cell; not asserted to be a mathematical bug",
        "locked_listwise_lambda_per_cell": LISTWISE_L2_PER_CELL,
        "locked_risk_lambda_per_cell": RISK_L2_PER_CELL,
        "outer_equivalence": "for mass 576, the legacy sum objective differs only by multiplication by 576",
        "inner_contract": "fixed per-cell regularization replaces legacy sample-size-dependent effective strength",
    }


def training_target_specs() -> tuple[dict[str, Any], ...]:
    inner = tuple({
        "target_id": f"inner_pair_{left}_{right}",
        "target_kind": "two_fold_inner_train",
        "fold_ids": (left, right),
    } for left, right in itertools.combinations(range(4), 2))
    outer = tuple({
        "target_id": f"outer_without_{heldout}",
        "target_kind": "three_fold_outer_train",
        "fold_ids": tuple(fold for fold in range(4) if fold != heldout),
    } for heldout in range(4))
    result = inner + outer
    identities = {tuple(item["fold_ids"]) for item in result}
    if len(result) != 10 or len(identities) != 10:
        raise AssertionError("the ten unique nested-CV training targets changed")
    return result


def deterministic_initializations(width: int) -> dict[str, np.ndarray]:
    signs = np.empty(width, np.float64)
    for index in range(width):
        byte = hashlib.sha256(f"v2.1-locked-init|{width}|{index}".encode("ascii")).digest()[0]
        signs[index] = 1.0 if byte & 1 else -1.0
    return {
        "zero": np.zeros(width, np.float64),
        "sha256-sign-0.01": signs * (0.01 / math.sqrt(max(1, width))),
    }


def _group_layout(
    arrays: Mapping[str, np.ndarray], rows: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    groups = frozen.decision_slices(np.asarray(arrays["decision"]), rows)
    position = {int(row): at for at, row in enumerate(rows)}
    local = [np.asarray([position[int(row)] for row in group], np.int64) for group in groups]
    lengths = np.asarray([len(group) for group in local], np.int64)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    decision_weight = frozen._decision_weights(arrays, rows)
    row_weight = np.zeros(len(rows), np.float64)
    for original, loc in zip(groups, local, strict=True):
        row_weight[loc] = decision_weight[int(np.asarray(arrays["decision"])[original[0]])]
    return groups, local, starts, lengths, row_weight


def listwise_problem(
    x: np.ndarray, arrays: Mapping[str, np.ndarray], rows: np.ndarray,
    *, objective: str, l2_per_cell: float = LISTWISE_L2_PER_CELL,
) -> dict[str, Any]:
    allowed = {
        "tied_top_outcome_set_probability_mass",
        "signal_only_uniform_top_outcome_cross_entropy",
    }
    if objective not in allowed:
        raise ValueError(f"unknown locked listwise objective: {objective}")
    groups, local, starts, lengths, base_row_weight = _group_layout(arrays, rows)
    outcome = np.asarray(arrays["outcome"], np.int8)
    top_mask = np.zeros(len(rows), np.float64)
    uniform_target = np.zeros(len(rows), np.float64)
    active_row_weight = np.zeros(len(rows), np.float64)
    signal_groups = 0
    for original, loc in zip(groups, local, strict=True):
        winners = loc[outcome[original] == np.max(outcome[original])]
        top_mask[winners] = 1.0
        uniform_target[winners] = 1.0 / len(winners)
        if len(set(map(int, outcome[original]))) > 1:
            active_row_weight[loc] = base_row_weight[loc]
            signal_groups += 1
    total_weight = float(np.sum(base_row_weight[starts]))
    signal_weight = float(np.sum(active_row_weight[starts]))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("paired-cell mass must be finite and positive")

    def probabilities(score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        maxima = np.maximum.reduceat(score, starts)
        exponent = np.exp(score - np.repeat(maxima, lengths))
        denominator = np.add.reduceat(exponent, starts)
        probability = exponent / np.repeat(denominator, lengths)
        top_score = np.where(top_mask > 0.0, score, -np.inf)
        top_maxima = np.maximum.reduceat(top_score, starts)
        top_exponent = np.exp(top_score - np.repeat(top_maxima, lengths)) * top_mask
        top_denominator = np.add.reduceat(top_exponent, starts)
        top_probability = top_exponent / np.repeat(top_denominator, lengths)
        return (
            probability, top_probability,
            maxima + np.log(denominator), top_maxima + np.log(top_denominator),
        )

    def fun_jac(coef: np.ndarray) -> tuple[float, np.ndarray]:
        score = x @ coef
        probability, top_probability, log_all, log_top = probabilities(score)
        if objective == "tied_top_outcome_set_probability_mass":
            per_group = log_all - log_top
            residual = probability - top_probability
        else:
            per_group = log_all - np.add.reduceat(uniform_target * score, starts)
            residual = probability - uniform_target
        value = float(
            np.dot(active_row_weight[starts], per_group) / total_weight
            + 0.5 * l2_per_cell * np.dot(coef, coef)
        )
        gradient = x.T @ (active_row_weight * residual) / total_weight + l2_per_cell * coef
        return value, gradient

    def hessp(coef: np.ndarray, vector: np.ndarray) -> np.ndarray:
        score = x @ coef
        probability, top_probability, _, _ = probabilities(score)
        direction = x @ vector
        mean = np.add.reduceat(probability * direction, starts)
        curvature = probability * (direction - np.repeat(mean, lengths))
        if objective == "tied_top_outcome_set_probability_mass":
            top_mean = np.add.reduceat(top_probability * direction, starts)
            curvature -= top_probability * (direction - np.repeat(top_mean, lengths))
        return x.T @ (active_row_weight * curvature) / total_weight + l2_per_cell * vector

    return {
        "name": objective, "width": int(x.shape[1]),
        "fun_jac": fun_jac, "hessp": hessp,
        "rows": int(len(rows)), "groups": int(len(groups)),
        "signal_groups": int(signal_groups),
        "total_weight": total_weight, "signal_weight": signal_weight,
        "signal_weight_rate": signal_weight / total_weight,
        "l2_per_cell": float(l2_per_cell),
        "convex": objective == "signal_only_uniform_top_outcome_cross_entropy",
        "constant": None,
    }


def risk_problem(
    x: np.ndarray, arrays: Mapping[str, np.ndarray], rows: np.ndarray,
    *, l2_per_cell: float = RISK_L2_PER_CELL,
) -> dict[str, Any]:
    _, risk = frozen._relative_labels(arrays, rows)
    edit = np.asarray(arrays["edit"])
    candidate = np.flatnonzero(edit[rows] != frozen.KEEP_EDIT)
    y = risk[rows][candidate]
    values = x[candidate]
    weights = frozen._candidate_row_weights(arrays, rows)[rows][candidate]
    total_weight = float(np.sum(weights))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("risk paired-cell mass must be finite and positive")
    design = np.concatenate((values, np.ones((len(values), 1))), axis=1)
    constant = float(y[0]) if len(np.unique(y)) == 1 else None

    def fun_jac(coef: np.ndarray) -> tuple[float, np.ndarray]:
        logit = design @ coef
        probability = expit(logit)
        value = float(
            np.dot(weights, np.logaddexp(0.0, logit) - y * logit) / total_weight
            + 0.5 * l2_per_cell * np.dot(coef[:-1], coef[:-1])
        )
        gradient = design.T @ (weights * (probability - y)) / total_weight
        gradient[:-1] += l2_per_cell * coef[:-1]
        return value, gradient

    def hessp(coef: np.ndarray, vector: np.ndarray) -> np.ndarray:
        probability = expit(design @ coef)
        curvature = weights * probability * (1.0 - probability)
        result = design.T @ (curvature * (design @ vector)) / total_weight
        result[:-1] += l2_per_cell * vector[:-1]
        return result

    return {
        "name": "independent_risk_logistic", "width": int(design.shape[1]),
        "fun_jac": fun_jac, "hessp": hessp,
        "rows": int(len(y)), "groups": None,
        "signal_groups": int(len(np.unique(y)) > 1),
        "total_weight": total_weight, "signal_weight": total_weight,
        "signal_weight_rate": 1.0, "l2_per_cell": float(l2_per_cell),
        "convex": True, "constant": constant,
    }


def run_solver(
    problem: Mapping[str, Any], solver_name: str, initialization: np.ndarray,
) -> dict[str, Any]:
    config = SOLVERS[solver_name]
    fun_jac: Callable[[np.ndarray], tuple[float, np.ndarray]] = problem["fun_jac"]
    initial_objective, initial_gradient = fun_jac(initialization)
    started = time.perf_counter()
    try:
        kwargs: dict[str, Any] = {
            "method": config["method"], "jac": True,
            "options": dict(config["options"]),
        }
        if config["uses_hessp"]:
            kwargs["hessp"] = problem["hessp"]
        fit = minimize(fun_jac, initialization.copy(), **kwargs)
        coefficient = np.asarray(fit.x, np.float64)
        final_objective, final_gradient = fun_jac(coefficient)
        finite = bool(
            np.isfinite(final_objective) and np.isfinite(final_gradient).all()
            and np.isfinite(coefficient).all()
        )
        initial_gradient_inf = float(np.max(np.abs(initial_gradient))) if len(initial_gradient) else 0.0
        final_gradient_inf = float(np.max(np.abs(final_gradient))) if len(final_gradient) else 0.0
        decrease = float(initial_objective - final_objective)
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
        return {
            "solver": solver_name, "success": bool(fit.success),
            "gate_passed": all(gates.values()), "gates": gates,
            "status": int(fit.status), "message": str(fit.message),
            "nit": int(getattr(fit, "nit", -1)),
            "nfev": int(getattr(fit, "nfev", -1)),
            "njev": int(getattr(fit, "njev", -1)),
            "nhev": int(getattr(fit, "nhev", -1)),
            "objective_initial": float(initial_objective),
            "objective_final": float(final_objective),
            "objective_decrease": decrease,
            "gradient_inf_initial": initial_gradient_inf,
            "gradient_inf_final": final_gradient_inf,
            "wall_seconds": time.perf_counter() - started,
            "coefficient_sha256": hashlib.sha256(coefficient.tobytes()).hexdigest(),
            "_coefficient": coefficient,
        }
    except Exception as error:
        return {
            "solver": solver_name, "success": False, "gate_passed": False,
            "gates": {"exception_free": False}, "status": -1,
            "message": f"{type(error).__name__}: {error}",
            "nit": -1, "nfev": -1, "njev": -1, "nhev": -1,
            "objective_initial": float(initial_objective), "objective_final": None,
            "objective_decrease": None,
            "gradient_inf_initial": float(np.max(np.abs(initial_gradient))),
            "gradient_inf_final": None,
            "wall_seconds": time.perf_counter() - started,
            "coefficient_sha256": None, "_coefficient": None,
        }


def study_problem(problem: Mapping[str, Any], solver_names: tuple[str, ...]) -> dict[str, Any]:
    if problem.get("constant") is not None:
        return {
            "constant_safe": True, "constant_probability": problem["constant"],
            "solvers": {}, "stable": True,
            "total_weight": problem["total_weight"],
            "signal_weight": problem["signal_weight"],
        }
    initializations = deterministic_initializations(int(problem["width"]))
    solver_reports: dict[str, Any] = {}
    for solver_name in solver_names:
        runs = []
        coefficients = []
        for initialization_name in INITIALIZATIONS:
            run = run_solver(problem, solver_name, initializations[initialization_name])
            run["initialization"] = initialization_name
            coefficients.append(run.pop("_coefficient"))
            runs.append(run)
        if all(value is not None for value in coefficients):
            coefficient_linf = float(np.max(np.abs(coefficients[0] - coefficients[1])))
            objectives = [float(run["objective_final"]) for run in runs]
            objective_range = max(objectives) - min(objectives)
        else:
            coefficient_linf = objective_range = None
        consistency = {
            "coefficient_linf": coefficient_linf,
            "objective_range": objective_range,
            "coefficient_linf_at_most_tolerance": (
                coefficient_linf is not None and coefficient_linf <= COEFFICIENT_LINF_TOLERANCE
            ),
            "objective_range_at_most_tolerance": (
                objective_range is not None and objective_range <= MULTISTART_OBJECTIVE_TOLERANCE
            ),
        }
        solver_reports[solver_name] = {
            "runs": runs, "multistart_consistency": consistency,
            "stable": all(run["gate_passed"] for run in runs)
            and consistency["coefficient_linf_at_most_tolerance"]
            and consistency["objective_range_at_most_tolerance"],
        }
    return {
        "constant_safe": False, "solvers": solver_reports,
        "stable": any(report["stable"] for report in solver_reports.values()),
        "total_weight": problem["total_weight"],
        "signal_weight": problem["signal_weight"],
        "signal_weight_rate": problem["signal_weight_rate"],
        "signal_groups": problem["signal_groups"],
        "groups": problem["groups"],
    }


def _solver_stable(targets: list[dict[str, Any]], head: str, solver_name: str) -> bool:
    for target in targets:
        report = target[head]
        if report["constant_safe"]:
            continue
        if not report["solvers"].get(solver_name, {}).get("stable", False):
            return False
    return True


def _locked_listwise_solver(targets: list[dict[str, Any]]) -> str | None:
    for solver_name in ("L-BFGS-B-1000", "trust-krylov-500"):
        if _solver_stable(
            targets, "signal_only_uniform_top_outcome_cross_entropy", solver_name,
        ):
            return solver_name
    return None


def optimization_only_study(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    input_audit = frozen.validate_arrays(arrays)
    row_fold = frozen._row_folds(np.asarray(arrays["seed"]))
    all_rows = np.arange(len(row_fold), dtype=np.int64)
    target_reports: list[dict[str, Any]] = []
    for spec in training_target_specs():
        fold_ids = tuple(map(int, spec["fold_ids"]))
        train = all_rows[np.isin(row_fold, fold_ids)]
        prior, _ = frozen._crossfit_prior_features(
            arrays, train, float(frozen.DEFAULT_CONFIG["sha_prior_alpha"]),
        )
        train_x, _ = frozen._projected_features(
            arrays, train, train, train_prior=prior, predict_prior=prior,
            shuffle_seed=None,
        )
        tied_problem = listwise_problem(
            train_x, arrays, train,
            objective="tied_top_outcome_set_probability_mass",
        )
        uniform_problem = listwise_problem(
            train_x, arrays, train,
            objective="signal_only_uniform_top_outcome_cross_entropy",
        )
        risk = risk_problem(train_x, arrays, train)
        target_reports.append({
            **spec,
            "fold_ids": list(fold_ids),
            "training_rows": int(len(train)),
            "training_decisions": len(frozen.decision_slices(np.asarray(arrays["decision"]), train)),
            "training_feature_sha256": hashlib.sha256(train_x.tobytes()).hexdigest(),
            "total_weight": uniform_problem["total_weight"],
            "signal_weight": uniform_problem["signal_weight"],
            "signal_weight_rate": uniform_problem["signal_weight_rate"],
            "tied_top_outcome_set_probability_mass": study_problem(
                tied_problem, ("L-BFGS-B-1000",),
            ),
            "signal_only_uniform_top_outcome_cross_entropy": study_problem(
                uniform_problem, ("L-BFGS-B-1000", "trust-krylov-500"),
            ),
            "independent_risk_logistic": study_problem(
                risk, ("L-BFGS-B-1000",),
            ),
            "heldout_prediction_executed": False,
            "heldout_outcome_metric_computed": False,
        })

    listwise_solver = _locked_listwise_solver(target_reports)
    risk_solver = (
        "L-BFGS-B-1000"
        if _solver_stable(target_reports, "independent_risk_logistic", "L-BFGS-B-1000")
        else None
    )
    return {
        "schema": SCHEMA,
        "status": "optimization_only_stable" if listwise_solver and risk_solver else "optimization_only_unstable",
        "convexity_audit": convexity_audit(),
        "regularization_audit": regularization_audit(),
        "config": {
            "solvers": SOLVERS, "initializations": INITIALIZATIONS,
            "gradient_inf_tolerance": GRADIENT_INF_TOLERANCE,
            "objective_tolerance": OBJECTIVE_TOLERANCE,
            "coefficient_linf_tolerance": COEFFICIENT_LINF_TOLERANCE,
            "multistart_objective_tolerance": MULTISTART_OBJECTIVE_TOLERANCE,
            "listwise_solver_preference": ["L-BFGS-B-1000", "trust-krylov-500"],
            "risk_solver_locked": "L-BFGS-B-1000",
            "unique_training_targets": 10,
            "training_target_deduplication": "all 6 unordered two-fold inner sets plus 4 three-fold outer sets",
        },
        "boundary": {
            "train_proxy_only": True,
            "heldout_predictions": False, "heldout_outcome_acceptance": False,
            "validation_opened": False, "sealed_opened": False,
            "choices_generated": False, "candidate_committed": False,
        },
        "input_audit": input_audit,
        "targets": target_reports,
        "recommendation": {
            "listwise_objective": (
                "signal_only_uniform_top_outcome_cross_entropy" if listwise_solver else None
            ),
            "listwise_semantics": "top-outcome-set classification; not full ordinal",
            "listwise_solver": listwise_solver,
            "risk_solver": risk_solver,
            "tied_probability_mass_eligible": False,
            "selection_used_heldout_outcome_metrics": False,
        },
    }


__all__ = [
    "INITIALIZATIONS", "LISTWISE_L2_PER_CELL", "RISK_L2_PER_CELL",
    "SCHEMA", "SOLVERS", "convexity_audit", "deterministic_initializations",
    "listwise_problem", "optimization_only_study", "regularization_audit",
    "risk_problem", "run_solver", "study_problem", "training_target_specs",
]
