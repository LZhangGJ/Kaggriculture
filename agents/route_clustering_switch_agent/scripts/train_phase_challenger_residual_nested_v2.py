"""Hardened nested evaluation for the R0 A -> phase residual pipeline."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

import run_farmer_augment_union_mpc_v1 as gate
import train_phase_challenger_residual_gate_v1 as base
import train_phase_challenger_residual_nested_v1 as v1


SCHEMA = "phase-challenger-residual-nested-v2"


def _empty_phase_metrics(calibration: Mapping[str, float]) -> dict[str, Any]:
    return {
        "decisions": 0,
        "selected": 0,
        "sum_realized_delta": 0.0,
        "mean_realized_delta": 0.0,
        "harmful": 0,
        "beneficial": 0,
        "opponent_harm": {},
        "selected_opponents": [],
        "selected_seeds": [],
        **{
            key: float(calibration[key])
            for key in ("beta", "threshold", "min_positive_fraction")
        },
    }


def evaluate(
    arrays: Mapping[str, np.ndarray], *, trees: int = 24,
    random_seed: int = 20260829,
) -> dict[str, Any]:
    """Evaluate outer seed folds with all calibration confined to outer train."""

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
        phase_train_opponents = set(map(
            int, train["opponent"][train["membership"] == base.PHASE_ONLY],
        ))
        missing_phase_train_opponents = sorted(
            train_opponents - phase_train_opponents
        )
        if missing_phase_train_opponents:
            return {
                "schema": SCHEMA,
                "status": "outer_train_phase_opponent_coverage_insufficient",
                "failed_fold": fold,
                "missing_phase_train_opponents": missing_phase_train_opponents,
                "input_audit": audit,
                "outer_labels_used_for_calibration": False,
                "models_fit": 0,
            }

        a_train, a_train_source = base.r0_arrays(train)
        am, ast, ap, a_inner = gate._lopo_predictions(
            a_train, trees, random_seed + fold * 100_003 + 101,
        )
        a_cal, _, a_train_choices = gate.zero_harm_calibration(
            a_train, am, ast, ap, farmer_only=False,
        )
        phase_train = base.phase_training_arrays(
            train, a_train, a_train_source, a_train_choices,
        )
        phase_signal = base.deployment_signal(phase_train)
        if not phase_signal["passed"]:
            return {
                "schema": SCHEMA,
                "status": "outer_train_phase_signal_insufficient",
                "failed_fold": fold,
                "phase_signal": phase_signal,
                "input_audit": audit,
                "outer_labels_used_for_calibration": False,
                "models_fit": 0,
            }
        pm, pst, pp, p_inner = gate._lopo_predictions(
            phase_train, trees, random_seed + fold * 100_003 + 10_101,
        )
        if len(p_inner) != len(train_opponents):
            raise AssertionError("phase inner LOPO lost a train opponent")
        p_cal, _, _ = gate.zero_harm_calibration(
            phase_train, pm, pst, pp, farmer_only=True,
        )
        a_model = v1._fit(a_train, trees, random_seed + fold * 100_003 + 20_101)
        p_model = v1._fit(phase_train, trees, random_seed + fold * 100_003 + 30_101)
        inner_models_fit += len(a_inner) + len(p_inner)
        outer_models_fit += 2

        a_valid, a_valid_source = base.r0_arrays(valid)
        vm, vst, vp = v1._moments(a_model, a_valid["features"])
        a_metrics, a_choices = gate._selection_metrics(
            a_valid, vm, vst, vp, a_cal, farmer_only=False,
        )
        local_a = {}
        for indices, choice in zip(
            base.decision_slices(a_valid["decision"]), a_choices, strict=True,
        ):
            local_a[int(a_valid["decision"][indices[0]])] = int(
                a_valid_source[int(choice)]
            )
        local_final = dict(local_a)

        valid_phase_mask = valid["membership"] == base.PHASE_ONLY
        phase_valid_opponents = set(map(int, valid["opponent"][valid_phase_mask]))
        if np.any(valid_phase_mask):
            phase_valid = base.phase_training_arrays(
                valid, a_valid, a_valid_source, a_choices,
            )
            qm, qst, qp = v1._moments(p_model, phase_valid["features"])
            p_metrics, p_choices = gate._selection_metrics(
                phase_valid, qm, qst, qp, p_cal, farmer_only=True,
            )
            for indices, choice in zip(
                base.decision_slices(phase_valid["decision"]),
                p_choices, strict=True,
            ):
                if int(choice) >= 0:
                    local_final[int(phase_valid["decision"][indices[0]])] = int(
                        phase_valid["source_row_index"][int(choice)]
                    )
            phase_valid_candidate_decisions = len(
                base.decision_slices(phase_valid["decision"])
            )
        else:
            p_metrics = _empty_phase_metrics(p_cal)
            phase_valid_candidate_decisions = 0

        for decision, source in local_a.items():
            a_by_decision[decision] = int(valid_rows[source])
            final_by_decision[decision] = int(valid_rows[local_final[decision]])
        folds.append({
            "left_out_seed_fold": fold,
            "train_seeds": train_seeds,
            "valid_seeds": valid_seeds,
            "calibration_valid_seed_overlap": overlap,
            "phase_train_opponent_coverage_complete": True,
            "phase_valid_missing_opponents": sorted(
                set(map(int, valid["opponent"])) - phase_valid_opponents
            ),
            "phase_valid_candidate_decisions": phase_valid_candidate_decisions,
            "phase_valid_exact_A_fallback_decisions": (
                len(local_a) - phase_valid_candidate_decisions
            ),
            "inner_A_LOPO_complete": len(a_inner) == len(train_opponents),
            "inner_phase_LOPO_complete": len(p_inner) == len(train_opponents),
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
        "scheme": "outer_seed_fold_with_inner_LOPO_A_and_phase_calibration",
        "outer_labels_used_for_calibration": False,
        "folds": folds,
        "A_choices": a_choices,
        "final_choices": final_choices,
        "A_diagnostics": v1._diagnostics(arrays, a_choices, all_seeds),
        "A_plus_phase_diagnostics": v1._diagnostics(
            arrays, final_choices, all_seeds,
        ),
        "input_audit": audit,
        "models_fit": inner_models_fit + outer_models_fit,
        "model_fit_audit": {
            "inner_LOPO_models": inner_models_fit,
            "outer_final_models": outer_models_fit,
        },
    }
