"""Nested residual evaluator with one threshold safe on opponent and seed OOF."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

import run_farmer_augment_union_mpc_v1 as gate
import train_phase_challenger_residual_gate_v1 as base
import train_phase_challenger_residual_nested_v1 as nested_v1
import train_phase_challenger_residual_nested_v2 as nested_v2
import train_phase_challenger_residual_seed_oof_nested_v1 as seed_nested


SCHEMA = "phase-challenger-residual-dual-oof-nested-v1"


def dual_view_calibration(
    arrays: Mapping[str, np.ndarray],
    opponent_moments: tuple[np.ndarray, np.ndarray, np.ndarray],
    seed_moments: tuple[np.ndarray, np.ndarray, np.ndarray],
    *, farmer_only: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Require pointwise zero harm in both OOF views, then maximize worst view."""

    om, ost, op = opponent_moments
    sm, sst, sp = seed_moments
    length = len(np.asarray(arrays["decision"]))
    if any(len(value) != length for value in (om, ost, op, sm, sst, sp)):
        raise ValueError("dual OOF moment arrays do not align")
    trials = []
    for beta in (0.0, 0.5, 1.0):
        thresholds = sorted(set(
            gate._thresholds(om - beta * ost)
            + gate._thresholds(sm - beta * sst)
        ))
        minimums = (0.6, 0.7, 0.8, 0.9) if farmer_only else (0.5, 0.6, 0.7)
        for threshold in thresholds:
            for minimum in minimums:
                params = {
                    "beta": beta,
                    "threshold": threshold,
                    "min_positive_fraction": minimum,
                }
                opponent_metrics, _ = gate._selection_metrics(
                    arrays, om, ost, op, params, farmer_only=farmer_only,
                )
                seed_metrics, _ = gate._selection_metrics(
                    arrays, sm, sst, sp, params, farmer_only=farmer_only,
                )
                safe = all(
                    metrics["harmful"] == 0
                    and not any(metrics["opponent_harm"].values())
                    for metrics in (opponent_metrics, seed_metrics)
                )
                trials.append({
                    **params,
                    "safe_both_views": safe,
                    "worst_view_sum": min(
                        opponent_metrics["sum_realized_delta"],
                        seed_metrics["sum_realized_delta"],
                    ),
                    "worst_view_beneficial": min(
                        opponent_metrics["beneficial"],
                        seed_metrics["beneficial"],
                    ),
                    "total_view_sum": (
                        opponent_metrics["sum_realized_delta"]
                        + seed_metrics["sum_realized_delta"]
                    ),
                    "max_view_selected": max(
                        opponent_metrics["selected"], seed_metrics["selected"],
                    ),
                    "view_metrics": {
                        "opponent_OOF": opponent_metrics,
                        "seed_OOF": seed_metrics,
                    },
                })
    allowed = [row for row in trials if row["safe_both_views"]]
    if not allowed:
        raise RuntimeError("dual-view calibration lost the KEEP/A fallback")
    chosen = max(allowed, key=lambda row: (
        row["worst_view_sum"],
        row["worst_view_beneficial"],
        row["total_view_sum"],
        -row["max_view_selected"],
        row["min_positive_fraction"],
        row["threshold"],
        row["beta"],
    ))
    params = {
        key: chosen[key]
        for key in ("beta", "threshold", "min_positive_fraction")
    }
    opponent_metrics, opponent_choices = gate._selection_metrics(
        arrays, om, ost, op, params, farmer_only=farmer_only,
    )
    seed_metrics, seed_choices = gate._selection_metrics(
        arrays, sm, sst, sp, params, farmer_only=farmer_only,
    )
    result = {
        **params,
        "safe_both_views": True,
        "selection_objective": "maximin_view_sum_then_benefit_then_total",
        "worst_view_sum": chosen["worst_view_sum"],
        "view_metrics": {
            "opponent_OOF": opponent_metrics,
            "seed_OOF": seed_metrics,
        },
    }
    return result, {
        "opponent_OOF": opponent_choices,
        "seed_OOF": seed_choices,
    }, trials


def evaluate(
    arrays: Mapping[str, np.ndarray], *, trees: int = 24,
    random_seed: int = 20260829,
) -> dict[str, Any]:
    audit = base.validate_arrays(arrays)
    if trees < 1:
        raise ValueError("trees must be positive")
    all_seeds = tuple(sorted(set(map(int, np.asarray(arrays["seed"])))))
    fold_by_seed = gate.seed_fold_map(all_seeds)
    if len(set(fold_by_seed.values())) != 4:
        raise ValueError("nested evaluation requires four non-empty seed folds")
    row_folds = np.asarray(
        [fold_by_seed[int(value)] for value in arrays["seed"]], np.int8,
    )
    a_by_decision: dict[int, int] = {}
    final_by_decision: dict[int, int] = {}
    folds = []
    inner_models_fit = 0
    outer_models_fit = 0
    for fold in range(4):
        train_rows = np.flatnonzero(row_folds != fold)
        valid_rows = np.flatnonzero(row_folds == fold)
        train = {key: np.asarray(value)[train_rows] for key, value in arrays.items()}
        valid = {key: np.asarray(value)[valid_rows] for key, value in arrays.items()}
        train_seeds = sorted(set(map(int, train["seed"])))
        valid_seeds = sorted(set(map(int, valid["seed"])))
        overlap = sorted(set(train_seeds) & set(valid_seeds))
        if overlap:
            raise AssertionError("outer seed labels leaked into calibration")
        train_opponents = set(map(int, train["opponent"]))
        phase_mask = train["membership"] == base.PHASE_ONLY
        missing_opponents = sorted(
            train_opponents - set(map(int, train["opponent"][phase_mask]))
        )
        if missing_opponents:
            return {
                "schema": SCHEMA,
                "status": "outer_train_phase_opponent_coverage_insufficient",
                "failed_fold": fold,
                "missing_phase_train_opponents": missing_opponents,
                "input_audit": audit,
                "outer_labels_used_for_calibration": False,
                "models_fit": 0,
            }

        a_train, a_train_source = base.r0_arrays(train)
        am, ast, ap, a_lopo = gate._lopo_predictions(
            a_train, trees, random_seed + fold * 100_003 + 101,
        )
        sm, sst, sp, a_seed_oof = seed_nested.seed_oof_predictions(
            a_train, trees, random_seed + fold * 100_003 + 1_101,
        )
        a_cal, a_view_choices, _ = dual_view_calibration(
            a_train, (am, ast, ap), (sm, sst, sp), farmer_only=False,
        )
        a_train_choices = a_view_choices["seed_OOF"]
        phase_train_pair = seed_nested.phase_pair_feature_arrays(
            train, a_train, a_train_source, a_train_choices,
        )
        phase_train = seed_nested.attach_phase_outcomes(phase_train_pair, train)
        phase_signal = base.deployment_signal(phase_train)
        if not phase_signal["passed"]:
            return {
                "schema": SCHEMA,
                "status": "outer_train_phase_signal_insufficient",
                "failed_fold": fold,
                "phase_signal": phase_signal,
                "input_audit": audit,
                "outer_labels_used_for_calibration": False,
                "models_fit": len(a_lopo) + len(a_seed_oof),
            }
        pm, pst, pp, p_lopo = gate._lopo_predictions(
            phase_train, trees, random_seed + fold * 100_003 + 10_101,
        )
        qm, qst, qp, p_seed_oof = seed_nested.seed_oof_predictions(
            phase_train, trees, random_seed + fold * 100_003 + 11_101,
        )
        p_cal, _, _ = dual_view_calibration(
            phase_train, (pm, pst, pp), (qm, qst, qp), farmer_only=True,
        )
        a_model = nested_v1._fit(
            a_train, trees, random_seed + fold * 100_003 + 20_101,
        )
        p_model = nested_v1._fit(
            phase_train, trees, random_seed + fold * 100_003 + 30_101,
        )
        inner_models_fit += (
            len(a_lopo) + len(a_seed_oof) + len(p_lopo) + len(p_seed_oof)
        )
        outer_models_fit += 2

        a_valid, a_valid_source = base.r0_arrays(valid)
        vm, vst, vp = nested_v1._moments(a_model, a_valid["features"])
        a_metrics, a_choices = gate._selection_metrics(
            a_valid, vm, vst, vp, a_cal, farmer_only=False,
        )
        local_a = {}
        for rows, choice in zip(
            base.decision_slices(a_valid["decision"]), a_choices, strict=True,
        ):
            local_a[int(a_valid["decision"][rows[0]])] = int(
                a_valid_source[int(choice)]
            )
        local_final = dict(local_a)

        phase_valid_pair = seed_nested.phase_pair_feature_arrays(
            valid, a_valid, a_valid_source, a_choices,
        )
        if len(phase_valid_pair["features"]):
            rm, rst, rp = nested_v1._moments(
                p_model, phase_valid_pair["features"],
            )
            phase_valid = seed_nested.attach_phase_outcomes(
                phase_valid_pair, valid,
            )
            p_metrics, p_choices = gate._selection_metrics(
                phase_valid, rm, rst, rp, p_cal, farmer_only=True,
            )
            for rows, choice in zip(
                base.decision_slices(phase_valid["decision"]),
                p_choices, strict=True,
            ):
                if int(choice) >= 0:
                    local_final[int(phase_valid["decision"][rows[0]])] = int(
                        phase_valid["source_row_index"][int(choice)]
                    )
            phase_valid_decisions = len(base.decision_slices(phase_valid["decision"]))
        else:
            p_metrics = nested_v2._empty_phase_metrics(p_cal)
            phase_valid_decisions = 0
        for decision, source in local_a.items():
            a_by_decision[decision] = int(valid_rows[source])
            final_by_decision[decision] = int(valid_rows[local_final[decision]])
        folds.append({
            "left_out_seed_fold": fold,
            "train_seeds": train_seeds,
            "valid_seeds": valid_seeds,
            "calibration_valid_seed_overlap": overlap,
            "inner_strategy": "shared_threshold_opponent_OOF_plus_seed_OOF",
            "inner_A_LOPO_complete": len(a_lopo) == len(train_opponents),
            "inner_A_seed_OOF_complete": len(a_seed_oof) == len(train_seeds),
            "inner_phase_LOPO_complete": len(p_lopo) == len(train_opponents),
            "inner_phase_seed_OOF_complete": len(p_seed_oof) == len(
                set(map(int, phase_train["seed"]))
            ),
            "phase_valid_candidate_decisions": phase_valid_decisions,
            "phase_valid_exact_A_fallback_decisions": (
                len(local_a) - phase_valid_decisions
            ),
            "A_inner_chosen": a_cal,
            "phase_inner_chosen": p_cal,
            "A_outer_metrics": a_metrics,
            "phase_outer_uplift_metrics": p_metrics,
            "phase_deployment_signal": phase_signal,
        })

    slices = base.decision_slices(np.asarray(arrays["decision"], np.int64))
    decision_order = [int(arrays["decision"][rows[0]]) for rows in slices]
    if set(decision_order) != set(a_by_decision) or set(decision_order) != set(final_by_decision):
        raise RuntimeError("nested choices do not cover every decision")
    a_choices = np.asarray([a_by_decision[key] for key in decision_order], np.int64)
    final_choices = np.asarray(
        [final_by_decision[key] for key in decision_order], np.int64,
    )
    return {
        "schema": SCHEMA,
        "status": "nested_seed_fold_evaluation_complete",
        "scheme": "outer_seed_fold_with_shared_opponent_and_seed_OOF_threshold",
        "outer_labels_used_for_calibration": False,
        "full_unseen_opponent_stacked_evidence": False,
        "folds": folds,
        "A_choices": a_choices,
        "final_choices": final_choices,
        "A_diagnostics": nested_v1._diagnostics(arrays, a_choices, all_seeds),
        "A_plus_phase_diagnostics": nested_v1._diagnostics(
            arrays, final_choices, all_seeds,
        ),
        "input_audit": audit,
        "models_fit": inner_models_fit + outer_models_fit,
        "model_fit_audit": {
            "inner_dual_OOF_models": inner_models_fit,
            "outer_final_models": outer_models_fit,
        },
        "inference_feature_boundary": {
            "pair_features_built_without_delta_margin": True,
            "outer_outcomes_attached_after_prediction": True,
        },
    }
