"""Nested A->phase residual evaluation with group-OOF lower-quantile risk scores."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

import run_farmer_augment_union_mpc_v1 as gate
import train_phase_challenger_residual_gate_v1 as base
import train_phase_challenger_residual_nested_v1 as nested_v1
import train_phase_challenger_residual_nested_v2 as nested_v2
import train_phase_challenger_residual_seed_oof_nested_v1 as seed_nested


SCHEMA = "phase-challenger-residual-group-quantile-nested-v1"
DEFAULT_QUANTILE = 0.10


def group_oof_tree_predictions(
    arrays: Mapping[str, np.ndarray], group_key: str, trees: int,
    random_seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    groups = np.asarray(arrays[group_key])
    unique = tuple(sorted(set(map(int, groups))))
    if len(unique) < 2:
        raise ValueError(f"{group_key} OOF requires at least two groups")
    result = np.full((len(groups), trees), np.nan, np.float32)
    provenance = []
    for fold, group in enumerate(unique):
        valid = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        model = nested_v1._fit(
            {key: np.asarray(value)[train] for key, value in arrays.items()},
            trees, random_seed + fold * 1009,
        )
        result[valid] = nested_v1.residual._tree_predictions(
            model, np.asarray(arrays["features"], np.float32)[valid],
        )
        provenance.append({
            "left_out_group": group,
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "train_valid_group_overlap": sorted(
                set(map(int, groups[train])) & set(map(int, groups[valid]))
            ),
            "train_valid_decision_overlap": sorted(
                set(map(int, np.asarray(arrays["decision"])[train]))
                & set(map(int, np.asarray(arrays["decision"])[valid]))
            ),
        })
    if not np.isfinite(result).all():
        raise RuntimeError(f"{group_key} OOF tree predictions are incomplete")
    return result, provenance


def lower_quantile(values: np.ndarray, quantile: float) -> np.ndarray:
    if not 0.0 <= quantile <= 0.5:
        raise ValueError("risk quantile must be between zero and one half")
    return np.quantile(
        np.asarray(values, np.float64), quantile, axis=1, method="lower",
    )


def risk_moments(score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(score, np.float64)
    return values, np.zeros(len(values), np.float64), (values > 0.0).astype(np.float64)


def risk_calibration(
    arrays: Mapping[str, np.ndarray], score: np.ndarray, *, farmer_only: bool,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    mean, std, positive = risk_moments(score)
    trials = []
    for threshold in gate._thresholds(mean):
        params = {
            "beta": 0.0,
            "threshold": threshold,
            "min_positive_fraction": 0.5,
        }
        metrics, _ = gate._selection_metrics(
            arrays, mean, std, positive, params, farmer_only=farmer_only,
        )
        trials.append(metrics)
    allowed = [
        row for row in trials
        if row["harmful"] == 0 and not any(row["opponent_harm"].values())
    ]
    if not allowed:
        raise RuntimeError("quantile risk calibration lost its fallback")
    chosen = max(allowed, key=lambda row: (
        row["sum_realized_delta"], row["beneficial"], -row["selected"],
        row["threshold"],
    ))
    metrics, choices = gate._selection_metrics(
        arrays, mean, std, positive, chosen, farmer_only=farmer_only,
    )
    return {
        **{key: chosen[key] for key in (
            "beta", "threshold", "min_positive_fraction",
        )},
        "OOF_metrics": metrics,
        "risk_score": "discrete_lower_quantile_of_equal_LOPO_and_LOSO_trees",
    }, choices, trials


def cross_calibrated_choices(
    arrays: Mapping[str, np.ndarray], score: np.ndarray,
    global_fold_by_seed: Mapping[int, int], *, farmer_only: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    row_folds = np.asarray([
        global_fold_by_seed[int(seed)] for seed in arrays["seed"]
    ], np.int8)
    choice_by_decision: dict[int, int] = {}
    audits = []
    for fold in sorted(set(map(int, row_folds))):
        calibration_rows = np.flatnonzero(row_folds != fold)
        held_rows = np.flatnonzero(row_folds == fold)
        calibration = {
            key: np.asarray(value)[calibration_rows]
            for key, value in arrays.items()
        }
        held = {
            key: np.asarray(value)[held_rows]
            for key, value in arrays.items()
        }
        chosen, _, _ = risk_calibration(
            calibration, np.asarray(score)[calibration_rows],
            farmer_only=farmer_only,
        )
        moments = risk_moments(np.asarray(score)[held_rows])
        _, local_choices = gate._selection_metrics(
            held, *moments, chosen, farmer_only=farmer_only,
        )
        for rows, local_choice in zip(
            base.decision_slices(np.asarray(held["decision"], np.int64)),
            local_choices, strict=True,
        ):
            decision = int(held["decision"][rows[0]])
            choice_by_decision[decision] = (
                -1 if int(local_choice) < 0
                else int(held_rows[int(local_choice)])
            )
        calibration_seeds = set(map(int, calibration["seed"]))
        held_seeds = set(map(int, held["seed"]))
        audits.append({
            "left_out_seed_fold": fold,
            "calibration_seeds": sorted(calibration_seeds),
            "held_seeds": sorted(held_seeds),
            "calibration_held_seed_overlap": sorted(
                calibration_seeds & held_seeds
            ),
            "chosen": chosen,
        })
    decisions = [
        int(arrays["decision"][rows[0]])
        for rows in base.decision_slices(np.asarray(arrays["decision"], np.int64))
    ]
    if set(decisions) != set(choice_by_decision):
        raise RuntimeError("cross-calibrated choices missed decisions")
    return np.asarray([choice_by_decision[value] for value in decisions]), audits


def full_model_risk(
    arrays: Mapping[str, np.ndarray], train_arrays: Mapping[str, np.ndarray],
    trees: int, quantile: float, random_seed: int,
) -> tuple[Any, np.ndarray]:
    model = nested_v1._fit(train_arrays, 2 * trees, random_seed)
    predictions = nested_v1.residual._tree_predictions(
        model, np.asarray(arrays["features"], np.float32),
    )
    if predictions.shape[1] != 2 * trees:
        raise AssertionError("full model tree count changed")
    return model, lower_quantile(predictions, quantile)


def evaluate(
    arrays: Mapping[str, np.ndarray], *, trees: int = 24,
    random_seed: int = 20260829, quantile: float = DEFAULT_QUANTILE,
) -> dict[str, Any]:
    audit = base.validate_arrays(arrays)
    if trees < 2:
        raise ValueError("group quantile evaluation requires at least two trees per OOF model")
    all_seeds = tuple(sorted(set(map(int, np.asarray(arrays["seed"])))))
    global_fold_by_seed = gate.seed_fold_map(all_seeds)
    if len(set(global_fold_by_seed.values())) != 4:
        raise ValueError("nested evaluation requires four non-empty seed folds")
    row_folds = np.asarray([
        global_fold_by_seed[int(seed)] for seed in arrays["seed"]
    ], np.int8)
    a_by_decision: dict[int, int] = {}
    final_by_decision: dict[int, int] = {}
    folds = []
    model_count = 0
    tree_count = 0
    for outer_fold in range(4):
        train_rows = np.flatnonzero(row_folds != outer_fold)
        valid_rows = np.flatnonzero(row_folds == outer_fold)
        train = {key: np.asarray(value)[train_rows] for key, value in arrays.items()}
        valid = {key: np.asarray(value)[valid_rows] for key, value in arrays.items()}
        train_seeds = sorted(set(map(int, train["seed"])))
        valid_seeds = sorted(set(map(int, valid["seed"])))
        if set(train_seeds) & set(valid_seeds):
            raise AssertionError("outer seed leaked into training")
        phase_mask = train["membership"] == base.PHASE_ONLY
        train_opponents = set(map(int, train["opponent"]))
        if train_opponents - set(map(int, train["opponent"][phase_mask])):
            return {
                "schema": SCHEMA,
                "status": "outer_train_phase_opponent_coverage_insufficient",
                "failed_fold": outer_fold,
                "input_audit": audit,
                "models_fit": 0,
            }

        a_train, a_source = base.r0_arrays(train)
        a_lopo, a_lopo_audit = group_oof_tree_predictions(
            a_train, "opponent", trees,
            random_seed + outer_fold * 100_003 + 101,
        )
        a_loso, a_loso_audit = group_oof_tree_predictions(
            a_train, "seed", trees,
            random_seed + outer_fold * 100_003 + 1_101,
        )
        a_oof_score = lower_quantile(
            np.concatenate([a_lopo, a_loso], axis=1), quantile,
        )
        a_calibration, _, _ = risk_calibration(
            a_train, a_oof_score, farmer_only=False,
        )
        a_cross_choices, a_cross_audit = cross_calibrated_choices(
            a_train, a_oof_score, global_fold_by_seed, farmer_only=False,
        )
        phase_pair = seed_nested.phase_pair_feature_arrays(
            train, a_train, a_source, a_cross_choices,
        )
        phase_train = seed_nested.attach_phase_outcomes(phase_pair, train)
        phase_signal = base.deployment_signal(phase_train)
        if not phase_signal["passed"]:
            return {
                "schema": SCHEMA,
                "status": "outer_train_phase_signal_insufficient",
                "failed_fold": outer_fold,
                "phase_signal": phase_signal,
                "input_audit": audit,
                "models_fit": 0,
            }
        p_lopo, p_lopo_audit = group_oof_tree_predictions(
            phase_train, "opponent", trees,
            random_seed + outer_fold * 100_003 + 10_101,
        )
        p_loso, p_loso_audit = group_oof_tree_predictions(
            phase_train, "seed", trees,
            random_seed + outer_fold * 100_003 + 11_101,
        )
        p_oof_score = lower_quantile(
            np.concatenate([p_lopo, p_loso], axis=1), quantile,
        )
        p_calibration, _, _ = risk_calibration(
            phase_train, p_oof_score, farmer_only=True,
        )
        group_models = (
            len(a_lopo_audit) + len(a_loso_audit)
            + len(p_lopo_audit) + len(p_loso_audit)
        )
        model_count += group_models
        tree_count += group_models * trees

        a_valid, a_valid_source = base.r0_arrays(valid)
        _, a_valid_score = full_model_risk(
            a_valid, a_train, trees, quantile,
            random_seed + outer_fold * 100_003 + 20_101,
        )
        model_count += 1
        tree_count += 2 * trees
        a_metrics, a_choices = gate._selection_metrics(
            a_valid, *risk_moments(a_valid_score), a_calibration,
            farmer_only=False,
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
            _, p_valid_score = full_model_risk(
                phase_valid_pair, phase_train, trees, quantile,
                random_seed + outer_fold * 100_003 + 30_101,
            )
            model_count += 1
            tree_count += 2 * trees
            phase_valid = seed_nested.attach_phase_outcomes(
                phase_valid_pair, valid,
            )
            p_metrics, p_choices = gate._selection_metrics(
                phase_valid, *risk_moments(p_valid_score), p_calibration,
                farmer_only=True,
            )
            for rows, choice in zip(
                base.decision_slices(phase_valid["decision"]), p_choices,
                strict=True,
            ):
                if int(choice) >= 0:
                    local_final[int(phase_valid["decision"][rows[0]])] = int(
                        phase_valid["source_row_index"][int(choice)]
                    )
            phase_valid_decisions = len(
                base.decision_slices(phase_valid["decision"])
            )
        else:
            p_metrics = nested_v2._empty_phase_metrics(p_calibration)
            phase_valid_decisions = 0
        for decision, source in local_a.items():
            a_by_decision[decision] = int(valid_rows[source])
            final_by_decision[decision] = int(valid_rows[local_final[decision]])
        folds.append({
            "left_out_seed_fold": outer_fold,
            "train_seeds": train_seeds,
            "valid_seeds": valid_seeds,
            "calibration_valid_seed_overlap": [],
            "risk_quantile": quantile,
            "OOF_trees_per_group_model": trees,
            "OOF_combined_tree_predictions_per_row": 2 * trees,
            "outer_full_model_trees": 2 * trees,
            "A_cross_calibration": a_cross_audit,
            "A_group_OOF_audit": {
                "opponent": a_lopo_audit, "seed": a_loso_audit,
            },
            "phase_group_OOF_audit": {
                "opponent": p_lopo_audit, "seed": p_loso_audit,
            },
            "A_inner_chosen": a_calibration,
            "phase_inner_chosen": p_calibration,
            "A_outer_metrics": a_metrics,
            "phase_outer_uplift_metrics": p_metrics,
            "phase_deployment_signal": phase_signal,
            "phase_valid_candidate_decisions": phase_valid_decisions,
            "phase_valid_exact_A_fallback_decisions": (
                len(local_a) - phase_valid_decisions
            ),
        })

    slices = base.decision_slices(np.asarray(arrays["decision"], np.int64))
    decision_order = [int(arrays["decision"][rows[0]]) for rows in slices]
    if set(decision_order) != set(a_by_decision) or set(decision_order) != set(final_by_decision):
        raise RuntimeError("nested choices do not cover all decisions")
    a_choices = np.asarray([a_by_decision[value] for value in decision_order])
    final_choices = np.asarray([final_by_decision[value] for value in decision_order])
    return {
        "schema": SCHEMA,
        "status": "nested_seed_fold_evaluation_complete",
        "scheme": "outer_seed_fold_group_OOF_to_equal_tree_full_model_lower_quantile",
        "outer_labels_used_for_calibration": False,
        "phase_A_reference_cross_calibrated_by_seed_fold": True,
        "full_unseen_opponent_stacked_evidence": False,
        "folds": folds,
        "A_choices": a_choices,
        "final_choices": final_choices,
        "A_diagnostics": nested_v1._diagnostics(arrays, a_choices, all_seeds),
        "A_plus_phase_diagnostics": nested_v1._diagnostics(
            arrays, final_choices, all_seeds,
        ),
        "input_audit": audit,
        "models_fit": model_count,
        "trees_fit": tree_count,
        "risk_contract": {
            "quantile": quantile,
            "quantile_method": "lower",
            "OOF_tree_values_per_row": 2 * trees,
            "outer_tree_values_per_row": 2 * trees,
            "beta_grid_removed": True,
            "positive_fraction_grid_removed": True,
        },
        "inference_feature_boundary": {
            "pair_features_built_without_delta_margin": True,
            "outer_outcomes_attached_after_prediction": True,
        },
    }
