"""Nested paired-seat DRO selector with separate utility and harm-risk heads."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

import run_farmer_augment_union_mpc_v1 as gate
import run_route_residual_adapter_v0 as residual
import train_phase_challenger_residual_gate_v1 as base
import train_phase_challenger_residual_group_quantile_nested_v1 as quantile_v1
import train_phase_challenger_residual_nested_v1 as nested_v1
import train_phase_challenger_residual_nested_v2 as nested_v2
import train_phase_challenger_residual_seed_oof_nested_v1 as seed_nested


SCHEMA = "phase-challenger-residual-paired-seat-two-head-v0"
UTILITY_QUANTILE = 0.10
RISK_QUANTILE = 0.90
GROUP_KEY = "observable_state_action_group"


def robustify(
    arrays: Mapping[str, np.ndarray], value_key: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if GROUP_KEY not in arrays:
        raise ValueError(f"paired-seat robustification requires {GROUP_KEY}")
    values = np.asarray(arrays[value_key], np.float64)
    groups = np.asarray(arrays[GROUP_KEY])
    if groups.ndim != 1 or len(groups) != len(values):
        raise ValueError("paired-seat group IDs do not align with rows")
    members: dict[bytes, list[int]] = defaultdict(list)
    for row, group in enumerate(groups):
        members[bytes(group)].append(row)
    robust = values.copy()
    paired = 0
    conflicting = 0
    for rows in members.values():
        group_values = values[rows]
        worst = float(np.min(group_values))
        robust[rows] = worst
        paired += int(len(rows) > 1)
        conflicting += int(
            np.any(group_values > 0.0) and np.any(group_values <= 0.0)
        )
    result = {key: np.asarray(value) for key, value in arrays.items()}
    result["target_signed_log_margin"] = base._signed_log(robust).astype(
        np.float32
    )
    result["target_nonpositive"] = (robust <= 0.0).astype(np.int8)
    result["paired_worst_value"] = robust
    return result, {
        "rows": len(values),
        "unique_groups": len(members),
        "multirow_groups": paired,
        "positive_nonpositive_conflict_groups": conflicting,
        "actual_negative_rows": int(np.count_nonzero(values < 0.0)),
        "actual_zero_rows": int(np.count_nonzero(values == 0.0)),
        "actual_positive_rows": int(np.count_nonzero(values > 0.0)),
        "robust_nonpositive_rows": int(np.count_nonzero(robust <= 0.0)),
        "realized_values_preserved": np.array_equal(
            np.asarray(result[value_key], np.float64), values,
        ),
    }


def _risk_params(trees: int, random_seed: int) -> dict[str, Any]:
    return {
        **residual._model_params(trees, random_seed),
        "class_weight": "balanced",
    }


def fit_risk(arrays: Mapping[str, np.ndarray], trees: int, seed: int) -> Any:
    target = np.asarray(arrays["target_nonpositive"], np.int8)
    classes = np.unique(target)
    if len(classes) == 1:
        return {
            "constant": float(classes[0]),
            "trees": int(trees),
        }
    model = ExtraTreesClassifier(**_risk_params(trees, seed))
    model.fit(
        np.asarray(arrays["features"], np.float32), target,
        sample_weight=gate._decision_weights(
            np.asarray(arrays["decision"], np.int64)
        ),
    )
    return model


def risk_tree_predictions(model: Any, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, np.float32)
    if isinstance(model, dict):
        return np.full(
            (len(values), int(model["trees"])),
            float(model["constant"]), np.float32,
        )
    predictions = []
    for tree in model.estimators_:
        classes = np.asarray(tree.classes_)
        column = np.flatnonzero(classes == 1)
        if len(column) != 1:
            raise RuntimeError("risk tree lost its nonpositive class")
        predictions.append(tree.predict_proba(values)[:, int(column[0])])
    return np.stack(predictions, axis=1).astype(np.float32)


def group_oof_two_head(
    arrays: Mapping[str, np.ndarray], group_key: str, trees: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    groups = np.asarray(arrays[group_key])
    unique = tuple(sorted(set(map(int, groups))))
    if len(unique) < 2:
        raise ValueError(f"{group_key} OOF requires at least two groups")
    utility = np.full((len(groups), trees), np.nan, np.float32)
    risk = np.full((len(groups), trees), np.nan, np.float32)
    provenance = []
    for fold, group in enumerate(unique):
        valid = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        train_arrays = {
            key: np.asarray(value)[train] for key, value in arrays.items()
        }
        utility_model = nested_v1._fit(
            train_arrays, trees, random_seed + fold * 2017,
        )
        risk_model = fit_risk(
            train_arrays, trees, random_seed + fold * 2017 + 1009,
        )
        valid_features = np.asarray(arrays["features"], np.float32)[valid]
        utility[valid] = residual._tree_predictions(
            utility_model, valid_features,
        )
        risk[valid] = risk_tree_predictions(risk_model, valid_features)
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
    if not np.isfinite(utility).all() or not np.isfinite(risk).all():
        raise RuntimeError(f"{group_key} two-head OOF panel is incomplete")
    return utility, risk, provenance


def two_head_scores(
    utility_trees: np.ndarray, risk_trees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    utility = np.quantile(
        np.asarray(utility_trees, np.float64), UTILITY_QUANTILE,
        axis=1, method="lower",
    )
    risk = np.quantile(
        np.asarray(risk_trees, np.float64), RISK_QUANTILE,
        axis=1, method="higher",
    )
    return utility, risk


def _risk_thresholds(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    result = [0.0, 1.0]
    if len(finite):
        result.extend(map(float, np.quantile(
            finite, [0.05, 0.10, 0.25, 0.50, 0.75], method="linear",
        )))
    return sorted(set(result))


def harmful_cache(
    arrays: Mapping[str, np.ndarray], value_key: str,
) -> set[bytes]:
    values = np.asarray(arrays[value_key], np.float64)
    groups = np.asarray(arrays[GROUP_KEY])
    minimum: dict[bytes, float] = {}
    for group, value in zip(groups, values, strict=True):
        key = bytes(group)
        minimum[key] = min(minimum.get(key, float("inf")), float(value))
    return {key for key, value in minimum.items() if value <= 0.0}


def cache_mask(arrays: Mapping[str, np.ndarray], cache: set[bytes]) -> np.ndarray:
    return np.asarray(
        [bytes(group) in cache for group in arrays[GROUP_KEY]], np.bool_,
    )


def selection_metrics(
    arrays: Mapping[str, np.ndarray], utility: np.ndarray, risk: np.ndarray,
    params: Mapping[str, float], *, farmer_only: bool,
    known_harm: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    utility = np.asarray(utility, np.float64)
    risk = np.asarray(risk, np.float64)
    if len(utility) != len(risk) or len(utility) != len(arrays["decision"]):
        raise ValueError("two-head scores do not align with rows")
    blocked = (
        np.zeros(len(utility), np.bool_)
        if known_harm is None else np.asarray(known_harm, np.bool_)
    )
    utility_threshold = float(params["utility_threshold"])
    risk_threshold = float(params["max_nonpositive_risk"])
    chosen_rows = []
    realized = []
    selected_opponents = []
    selected_seeds = []
    cache_rejections = 0
    for rows in base.decision_slices(np.asarray(arrays["decision"], np.int64)):
        if farmer_only:
            chosen = -1
            candidates = rows
            value_key = "deployment_uplift"
        else:
            keep = rows[np.asarray(arrays["edit"])[rows] == base.KEEP_EDIT]
            if len(keep) != 1:
                raise RuntimeError("A decision lost its unique KEEP")
            chosen = int(keep[0])
            candidates = rows[np.asarray(arrays["edit"])[rows] != base.KEEP_EDIT]
            value_key = "delta_margin"
        if len(candidates):
            passes_score = (
                (utility[candidates] > utility_threshold)
                & (risk[candidates] <= risk_threshold)
            )
            cache_rejections += int(np.count_nonzero(
                passes_score & blocked[candidates]
            ))
            eligible = candidates[passes_score & ~blocked[candidates]]
            if len(eligible):
                chosen = int(eligible[int(np.argmax(utility[eligible]))])
        value = 0.0 if chosen < 0 else float(arrays[value_key][chosen])
        chosen_rows.append(chosen)
        realized.append(value)
        if chosen >= 0 and (
            farmer_only or int(arrays["edit"][chosen]) != int(base.KEEP_EDIT)
        ):
            selected_opponents.append(int(arrays["opponent"][chosen]))
            selected_seeds.append(int(arrays["seed"][chosen]))
    realized_values = np.asarray(realized, np.float64)
    opponent_harm = {
        str(opponent): int(sum(
            value < 0.0
            for value, row in zip(realized, chosen_rows, strict=True)
            if row >= 0 and int(arrays["opponent"][row]) == opponent
        ))
        for opponent in sorted(set(map(int, arrays["opponent"])))
    }
    return {
        "decisions": len(realized),
        "selected": int(sum(
            row >= 0 and (
                farmer_only
                or int(arrays["edit"][row]) != int(base.KEEP_EDIT)
            )
            for row in chosen_rows
        )),
        "sum_realized_delta": float(realized_values.sum()),
        "mean_realized_delta": (
            float(realized_values.mean()) if len(realized_values) else 0.0
        ),
        "harmful": int(np.count_nonzero(realized_values < 0.0)),
        "zero": int(np.count_nonzero(realized_values == 0.0)),
        "beneficial": int(np.count_nonzero(realized_values > 0.0)),
        "opponent_harm": opponent_harm,
        "selected_opponents": sorted(set(selected_opponents)),
        "selected_seeds": sorted(set(selected_seeds)),
        "cache_candidate_rejections": cache_rejections,
        "utility_threshold": utility_threshold,
        "max_nonpositive_risk": risk_threshold,
    }, np.asarray(chosen_rows, np.int64)


def calibrate(
    arrays: Mapping[str, np.ndarray], utility: np.ndarray, risk: np.ndarray,
    *, farmer_only: bool,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    trials = []
    for risk_threshold in _risk_thresholds(risk):
        for utility_threshold in gate._thresholds(utility):
            params = {
                "utility_threshold": utility_threshold,
                "max_nonpositive_risk": risk_threshold,
            }
            metrics, _ = selection_metrics(
                arrays, utility, risk, params, farmer_only=farmer_only,
            )
            trials.append(metrics)
    allowed = [
        row for row in trials
        if row["harmful"] == 0 and not any(row["opponent_harm"].values())
    ]
    if not allowed:
        raise RuntimeError("two-head calibration lost its KEEP/A fallback")
    chosen = max(allowed, key=lambda row: (
        row["sum_realized_delta"], row["beneficial"], -row["selected"],
        -row["max_nonpositive_risk"], row["utility_threshold"],
    ))
    params = {
        "utility_threshold": chosen["utility_threshold"],
        "max_nonpositive_risk": chosen["max_nonpositive_risk"],
    }
    metrics, choices = selection_metrics(
        arrays, utility, risk, params, farmer_only=farmer_only,
    )
    return {
        **params,
        "OOF_metrics": metrics,
        "utility_score": "lower_q10_of_equal_LOPO_and_LOSO_regression_trees",
        "risk_score": "upper_q90_probability_of_paired_worst_nonpositive",
    }, choices, trials


def cross_calibrated_A_choices(
    arrays: Mapping[str, np.ndarray], utility: np.ndarray, risk: np.ndarray,
    global_fold_by_seed: Mapping[int, int],
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
            key: np.asarray(value)[held_rows] for key, value in arrays.items()
        }
        chosen, _, _ = calibrate(
            calibration, utility[calibration_rows], risk[calibration_rows],
            farmer_only=False,
        )
        known = cache_mask(
            held, harmful_cache(calibration, "delta_margin")
        )
        _, local_choices = selection_metrics(
            held, utility[held_rows], risk[held_rows], chosen,
            farmer_only=False, known_harm=known,
        )
        for rows, local_choice in zip(
            base.decision_slices(held["decision"]), local_choices, strict=True,
        ):
            decision = int(held["decision"][rows[0]])
            choice_by_decision[decision] = int(held_rows[int(local_choice)])
        calibration_seeds = set(map(int, calibration["seed"]))
        held_seeds = set(map(int, held["seed"]))
        audits.append({
            "left_out_seed_fold": fold,
            "calibration_seeds": sorted(calibration_seeds),
            "held_seeds": sorted(held_seeds),
            "calibration_held_seed_overlap": sorted(
                calibration_seeds & held_seeds
            ),
            "known_harm_cache_hits": int(np.count_nonzero(known)),
            "chosen": chosen,
        })
    decisions = [
        int(arrays["decision"][rows[0]])
        for rows in base.decision_slices(arrays["decision"])
    ]
    if set(decisions) != set(choice_by_decision):
        raise RuntimeError("cross-calibrated A choices missed decisions")
    return np.asarray([choice_by_decision[value] for value in decisions]), audits


def full_two_head_scores(
    arrays: Mapping[str, np.ndarray], train_arrays: Mapping[str, np.ndarray],
    trees: int, random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    utility_model = nested_v1._fit(train_arrays, 2 * trees, random_seed)
    risk_model = fit_risk(train_arrays, 2 * trees, random_seed + 1009)
    features = np.asarray(arrays["features"], np.float32)
    return two_head_scores(
        residual._tree_predictions(utility_model, features),
        risk_tree_predictions(risk_model, features),
    )


def _attach_group(
    target: dict[str, np.ndarray], source_arrays: Mapping[str, np.ndarray],
    source_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        **target,
        GROUP_KEY: np.asarray(source_arrays[GROUP_KEY])[source_rows],
    }


def evaluate(
    arrays: Mapping[str, np.ndarray], *, trees: int = 24,
    random_seed: int = 20260829,
) -> dict[str, Any]:
    audit = base.validate_arrays(arrays)
    if GROUP_KEY not in arrays or len(arrays[GROUP_KEY]) != len(arrays["decision"]):
        raise ValueError("paired-seat fingerprints are absent or misaligned")
    if trees < 2:
        raise ValueError("two-head evaluation requires at least two trees")
    all_seeds = tuple(sorted(set(map(int, arrays["seed"]))))
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

        a_train, a_source = base.r0_arrays(train)
        a_train = _attach_group(a_train, train, a_source)
        a_robust, a_robust_audit = robustify(a_train, "delta_margin")
        au_lopo, ar_lopo, a_lopo_audit = group_oof_two_head(
            a_robust, "opponent", trees,
            random_seed + outer_fold * 100_003 + 101,
        )
        au_loso, ar_loso, a_loso_audit = group_oof_two_head(
            a_robust, "seed", trees,
            random_seed + outer_fold * 100_003 + 1_101,
        )
        a_utility, a_risk = two_head_scores(
            np.concatenate([au_lopo, au_loso], axis=1),
            np.concatenate([ar_lopo, ar_loso], axis=1),
        )
        a_calibration, _, _ = calibrate(
            a_robust, a_utility, a_risk, farmer_only=False,
        )
        a_cross_choices, a_cross_audit = cross_calibrated_A_choices(
            a_robust, a_utility, a_risk, global_fold_by_seed,
        )
        phase_pair = seed_nested.phase_pair_feature_arrays(
            train, a_train, a_source, a_cross_choices,
        )
        phase_pair = _attach_group(
            phase_pair, train, phase_pair["source_row_index"],
        )
        phase_train_actual = seed_nested.attach_phase_outcomes(phase_pair, train)
        phase_signal = base.deployment_signal(phase_train_actual)
        if not phase_signal["passed"]:
            return {
                "schema": SCHEMA,
                "status": "outer_train_phase_signal_insufficient",
                "failed_fold": outer_fold,
                "phase_signal": phase_signal,
                "input_audit": audit,
                "models_fit": model_count,
            }
        phase_robust, phase_robust_audit = robustify(
            phase_train_actual, "deployment_uplift",
        )
        pu_lopo, pr_lopo, p_lopo_audit = group_oof_two_head(
            phase_robust, "opponent", trees,
            random_seed + outer_fold * 100_003 + 10_101,
        )
        pu_loso, pr_loso, p_loso_audit = group_oof_two_head(
            phase_robust, "seed", trees,
            random_seed + outer_fold * 100_003 + 11_101,
        )
        p_utility, p_risk = two_head_scores(
            np.concatenate([pu_lopo, pu_loso], axis=1),
            np.concatenate([pr_lopo, pr_loso], axis=1),
        )
        p_calibration, _, _ = calibrate(
            phase_robust, p_utility, p_risk, farmer_only=True,
        )
        group_models = 2 * (
            len(a_lopo_audit) + len(a_loso_audit)
            + len(p_lopo_audit) + len(p_loso_audit)
        )
        model_count += group_models
        tree_count += group_models * trees

        a_valid, a_valid_source = base.r0_arrays(valid)
        a_valid = _attach_group(a_valid, valid, a_valid_source)
        a_valid_utility, a_valid_risk = full_two_head_scores(
            a_valid, a_robust, trees,
            random_seed + outer_fold * 100_003 + 20_101,
        )
        model_count += 2
        tree_count += 4 * trees
        a_known = cache_mask(
            a_valid, harmful_cache(a_train, "delta_margin")
        )
        a_metrics, a_choices = selection_metrics(
            a_valid, a_valid_utility, a_valid_risk, a_calibration,
            farmer_only=False, known_harm=a_known,
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
        phase_valid_pair = _attach_group(
            phase_valid_pair, valid, phase_valid_pair["source_row_index"],
        )
        if len(phase_valid_pair["features"]):
            p_valid_utility, p_valid_risk = full_two_head_scores(
                phase_valid_pair, phase_robust, trees,
                random_seed + outer_fold * 100_003 + 30_101,
            )
            model_count += 2
            tree_count += 4 * trees
            phase_valid = seed_nested.attach_phase_outcomes(
                phase_valid_pair, valid,
            )
            p_known = cache_mask(
                phase_valid,
                harmful_cache(phase_train_actual, "deployment_uplift"),
            )
            p_metrics, p_choices = selection_metrics(
                phase_valid, p_valid_utility, p_valid_risk, p_calibration,
                farmer_only=True, known_harm=p_known,
            )
            for rows, choice in zip(
                base.decision_slices(phase_valid["decision"]),
                p_choices, strict=True,
            ):
                if int(choice) >= 0:
                    local_final[int(phase_valid["decision"][rows[0]])] = int(
                        phase_valid["source_row_index"][int(choice)]
                    )
            phase_valid_decisions = len(
                base.decision_slices(phase_valid["decision"])
            )
            phase_cache_hits = int(np.count_nonzero(p_known))
        else:
            p_metrics = nested_v2._empty_phase_metrics({
                "beta": 0.0, "threshold": 0.0,
                "min_positive_fraction": 1.0,
            })
            phase_valid_decisions = 0
            phase_cache_hits = 0
        for decision, source in local_a.items():
            a_by_decision[decision] = int(valid_rows[source])
            final_by_decision[decision] = int(valid_rows[local_final[decision]])
        folds.append({
            "left_out_seed_fold": outer_fold,
            "train_seeds": train_seeds,
            "valid_seeds": valid_seeds,
            "calibration_valid_seed_overlap": [],
            "A_cross_calibration": a_cross_audit,
            "A_group_OOF_audit": {
                "opponent": a_lopo_audit, "seed": a_loso_audit,
            },
            "phase_group_OOF_audit": {
                "opponent": p_lopo_audit, "seed": p_loso_audit,
            },
            "A_paired_seat_robustification": a_robust_audit,
            "phase_paired_seat_robustification": phase_robust_audit,
            "A_inner_chosen": a_calibration,
            "phase_inner_chosen": p_calibration,
            "A_outer_metrics": a_metrics,
            "phase_outer_uplift_metrics": p_metrics,
            "phase_deployment_signal": phase_signal,
            "A_outer_known_harm_cache_hits": int(np.count_nonzero(a_known)),
            "phase_outer_known_harm_cache_hits": phase_cache_hits,
            "phase_valid_candidate_decisions": phase_valid_decisions,
            "phase_valid_exact_A_fallback_decisions": (
                len(local_a) - phase_valid_decisions
            ),
        })

    slices = base.decision_slices(np.asarray(arrays["decision"], np.int64))
    decision_order = [int(arrays["decision"][rows[0]]) for rows in slices]
    if set(decision_order) != set(a_by_decision) or set(decision_order) != set(final_by_decision):
        raise RuntimeError("nested paired-seat choices do not cover all decisions")
    a_choices = np.asarray([a_by_decision[value] for value in decision_order])
    final_choices = np.asarray([
        final_by_decision[value] for value in decision_order
    ])
    return {
        "schema": SCHEMA,
        "status": "nested_seed_fold_evaluation_complete",
        "scheme": "paired-seat worst-case utility plus nonpositive-risk two-head 2T-to-2T",
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
            "harm_target": "paired-seat group minimum <= 0",
            "utility_target": "signed-log paired-seat group minimum",
            "utility_quantile": UTILITY_QUANTILE,
            "risk_probability_quantile": RISK_QUANTILE,
            "calibration_realized_values": "actual seat-specific deltas",
            "exact_cache_source": "outer train only",
        },
        "inference_feature_boundary": {
            "pair_features_built_without_delta_margin": True,
            "outer_outcomes_attached_after_prediction": True,
            "observable_state_action_hash_used_as_model_feature": False,
            "seat_remains_an_explicit_numeric_feature": True,
        },
    }
