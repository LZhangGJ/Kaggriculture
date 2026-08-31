"""Outer-seed evaluation with seed-aligned inner OOF calibration."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

import run_farmer_augment_union_mpc_v1 as gate
import train_phase_challenger_residual_gate_v1 as base
import train_phase_challenger_residual_nested_v1 as nested_v1
import train_phase_challenger_residual_nested_v2 as nested_v2


SCHEMA = "phase-challenger-residual-seed-oof-nested-v1"


def seed_oof_predictions(
    arrays: Mapping[str, np.ndarray], trees: int, random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    seeds = np.asarray(arrays["seed"], np.int64)
    unique = tuple(sorted(set(map(int, seeds))))
    if len(unique) < 2:
        raise ValueError("seed OOF requires at least two seeds")
    mean = np.full(len(seeds), np.nan, np.float64)
    std = np.full(len(seeds), np.nan, np.float64)
    positive = np.full(len(seeds), np.nan, np.float64)
    folds = []
    for fold, seed in enumerate(unique):
        valid = np.flatnonzero(seeds == seed)
        train = np.flatnonzero(seeds != seed)
        model = nested_v1._fit(
            {key: np.asarray(value)[train] for key, value in arrays.items()},
            trees, random_seed + fold * 1009,
        )
        predictions = nested_v1.residual._tree_predictions(
            model, np.asarray(arrays["features"], np.float32)[valid],
        )
        mean[valid] = predictions.mean(axis=1)
        std[valid] = predictions.std(axis=1)
        positive[valid] = (predictions > 0.0).mean(axis=1)
        folds.append({
            "left_out_seed": seed,
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
        })
    if not all(np.isfinite(value).all() for value in (mean, std, positive)):
        raise RuntimeError("seed OOF prediction panel is incomplete")
    return mean, std, positive, folds


def phase_pair_feature_arrays(
    arrays: Mapping[str, np.ndarray], a: Mapping[str, np.ndarray],
    a_source_indices: np.ndarray, a_choices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build phase/A pair features without reading margins or outcome labels."""

    features = np.asarray(arrays["features"], np.float32)
    membership = np.asarray(arrays["membership"], np.int8)
    decisions = np.asarray(arrays["decision"], np.int64)
    if features.ndim != 2 or len(features) != len(membership):
        raise ValueError("label-free feature panel has invalid shape")
    a_slices = base.decision_slices(np.asarray(a["decision"], np.int64))
    choices = np.asarray(a_choices, np.int64)
    if len(a_slices) != len(choices):
        raise ValueError("A choices do not align with decisions")
    choice_by_decision = {}
    for rows, choice in zip(a_slices, choices, strict=True):
        if not np.any(rows == choice):
            raise ValueError("A choice escaped its decision")
        choice_by_decision[int(a["decision"][rows[0]])] = int(choice)
    phase_source = np.flatnonzero(membership == base.PHASE_ONLY)
    width = features.shape[1]
    pair = np.empty((len(phase_source), 2 * width), np.float32)
    a_choice_source = np.empty(len(phase_source), np.int64)
    for local, source in enumerate(phase_source):
        decision = int(decisions[source])
        a_source = int(a_source_indices[choice_by_decision[decision]])
        pair[local, :width] = features[source]
        pair[local, width:] = features[source] - features[a_source]
        a_choice_source[local] = a_source
    return {
        "features": pair,
        "decision": decisions[phase_source],
        "opponent": np.asarray(arrays["opponent"], np.int64)[phase_source],
        "seed": np.asarray(arrays["seed"], np.int64)[phase_source],
        "source_row_index": phase_source,
        "a_choice_source_row_index": a_choice_source,
    }


def attach_phase_outcomes(
    pair: Mapping[str, np.ndarray], arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Attach labels after pair features have been built and, at eval, scored."""

    source = np.asarray(pair["source_row_index"], np.int64)
    a_source = np.asarray(pair["a_choice_source_row_index"], np.int64)
    delta = np.asarray(arrays["delta_margin"], np.float64)
    deployment = delta[source] - delta[a_source]
    if np.any(
        np.asarray(arrays["opponent"])[source]
        != np.asarray(arrays["opponent"])[a_source]
    ) or np.any(
        np.asarray(arrays["seed"])[source]
        != np.asarray(arrays["seed"])[a_source]
    ):
        raise AssertionError("phase target crossed a decision boundary")
    return {
        **{key: np.asarray(value) for key, value in pair.items()},
        "target_signed_log_margin": base._signed_log(deployment).astype(np.float32),
        "deployment_uplift": deployment,
    }


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
        missing_seeds = sorted(
            set(train_seeds) - set(map(int, train["seed"][phase_mask]))
        )
        if missing_opponents:
            return {
                "schema": SCHEMA,
                "status": "outer_train_phase_coverage_insufficient",
                "failed_fold": fold,
                "missing_phase_train_opponents": missing_opponents,
                "missing_phase_train_seeds": missing_seeds,
                "input_audit": audit,
                "outer_labels_used_for_calibration": False,
                "models_fit": 0,
            }

        a_train, a_train_source = base.r0_arrays(train)
        am, ast, ap, a_inner = seed_oof_predictions(
            a_train, trees, random_seed + fold * 100_003 + 101,
        )
        a_cal, _, a_train_choices = gate.zero_harm_calibration(
            a_train, am, ast, ap, farmer_only=False,
        )
        phase_train_pair = phase_pair_feature_arrays(
            train, a_train, a_train_source, a_train_choices,
        )
        phase_train = attach_phase_outcomes(phase_train_pair, train)
        phase_signal = base.deployment_signal(phase_train)
        if not phase_signal["passed"]:
            return {
                "schema": SCHEMA,
                "status": "outer_train_phase_signal_insufficient",
                "failed_fold": fold,
                "phase_signal": phase_signal,
                "input_audit": audit,
                "outer_labels_used_for_calibration": False,
                "models_fit": len(a_inner),
            }
        pm, pst, pp, p_inner = seed_oof_predictions(
            phase_train, trees, random_seed + fold * 100_003 + 10_101,
        )
        p_cal, _, _ = gate.zero_harm_calibration(
            phase_train, pm, pst, pp, farmer_only=True,
        )
        a_model = nested_v1._fit(
            a_train, trees, random_seed + fold * 100_003 + 20_101,
        )
        p_model = nested_v1._fit(
            phase_train, trees, random_seed + fold * 100_003 + 30_101,
        )
        inner_models_fit += len(a_inner) + len(p_inner)
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

        phase_valid_pair = phase_pair_feature_arrays(
            valid, a_valid, a_valid_source, a_choices,
        )
        if len(phase_valid_pair["features"]):
            qm, qst, qp = nested_v1._moments(
                p_model, phase_valid_pair["features"],
            )
            phase_valid = attach_phase_outcomes(phase_valid_pair, valid)
            p_metrics, p_choices = gate._selection_metrics(
                phase_valid, qm, qst, qp, p_cal, farmer_only=True,
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
            "inner_strategy": "leave_one_market_seed_out",
            "inner_A_seed_OOF_complete": len(a_inner) == len(train_seeds),
            "inner_phase_seed_OOF_complete": len(p_inner) == len(
                set(map(int, phase_train["seed"]))
            ),
            "phase_train_seeds_without_candidates": missing_seeds,
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
        "scheme": "outer_seed_fold_with_inner_leave_one_seed_out_A_and_phase_calibration",
        "outer_labels_used_for_calibration": False,
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
            "inner_seed_OOF_models": inner_models_fit,
            "outer_final_models": outer_models_fit,
        },
        "inference_feature_boundary": {
            "pair_features_built_without_delta_margin": True,
            "outer_outcomes_attached_after_prediction": True,
        },
    }
