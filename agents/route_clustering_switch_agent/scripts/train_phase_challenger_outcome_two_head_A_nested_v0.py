"""Nested A-stage selector with outcome-upgrade and outcome-regression heads."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

import run_farmer_augment_union_mpc_v1 as gate
import run_route_residual_adapter_v0 as residual
import train_phase_challenger_residual_gate_v1 as base
import train_phase_challenger_residual_nested_v1 as nested_v1
import train_phase_challenger_residual_paired_seat_two_head_v0 as paired


SCHEMA = "phase-challenger-outcome-two-head-A-nested-v0"
UPGRADE_QUANTILE = 0.10
REGRESSION_QUANTILE = 0.90


def outcome_arrays(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if "outcome" not in arrays or "seat" not in arrays:
        raise ValueError("outcome selector requires outcome and seat arrays")
    a, source = base.r0_arrays(arrays)
    outcome = np.asarray(arrays["outcome"], np.int8)[source]
    seat = np.asarray(arrays["seat"], np.int8)[source]
    if not set(map(int, outcome)) <= {0, 1, 2}:
        raise ValueError("outcome must use the ordinal loss/tie/win encoding 0/1/2")
    fallback = np.empty(len(outcome), np.int8)
    for rows in base.decision_slices(a["decision"]):
        keep = rows[np.asarray(a["edit"])[rows] == base.KEEP_EDIT]
        if len(keep) != 1:
            raise ValueError("outcome A panel lost its unique KEEP")
        fallback[rows] = outcome[int(keep[0])]
    delta = outcome.astype(np.int16) - fallback.astype(np.int16)
    return {
        **a,
        "outcome": outcome,
        "fallback_outcome": fallback,
        "outcome_delta": delta,
        "target_outcome_regression": (delta < 0).astype(np.int8),
        "target_outcome_upgrade": (delta > 0).astype(np.int8),
        "seat": seat,
    }, source


def fit_classifier(
    arrays: Mapping[str, np.ndarray], target_key: str,
    trees: int, random_seed: int,
) -> Any:
    target = np.asarray(arrays[target_key], np.int8)
    classes = np.unique(target)
    if len(classes) == 1:
        return {"constant": float(classes[0]), "trees": int(trees)}
    model = ExtraTreesClassifier(**paired._risk_params(trees, random_seed))
    model.fit(
        np.asarray(arrays["features"], np.float32), target,
        sample_weight=gate._decision_weights(
            np.asarray(arrays["decision"], np.int64)
        ),
    )
    return model


def group_oof_probabilities(
    arrays: Mapping[str, np.ndarray], group_key: str, trees: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    groups = np.asarray(arrays[group_key])
    unique = tuple(sorted(set(map(int, groups))))
    if len(unique) < 2:
        raise ValueError(f"{group_key} OOF requires at least two groups")
    regression = np.full((len(groups), trees), np.nan, np.float32)
    upgrade = np.full((len(groups), trees), np.nan, np.float32)
    audit = []
    for fold, group in enumerate(unique):
        valid = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        subset = {
            key: np.asarray(value)[train] for key, value in arrays.items()
        }
        regression_model = fit_classifier(
            subset, "target_outcome_regression", trees,
            random_seed + fold * 2017,
        )
        upgrade_model = fit_classifier(
            subset, "target_outcome_upgrade", trees,
            random_seed + fold * 2017 + 1009,
        )
        features = np.asarray(arrays["features"], np.float32)[valid]
        regression[valid] = paired.risk_tree_predictions(
            regression_model, features,
        )
        upgrade[valid] = paired.risk_tree_predictions(upgrade_model, features)
        audit.append({
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
    if not np.isfinite(regression).all() or not np.isfinite(upgrade).all():
        raise RuntimeError(f"{group_key} outcome OOF panel is incomplete")
    return regression, upgrade, audit


def outcome_scores(
    regression_trees: np.ndarray, upgrade_trees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    regression = np.quantile(
        np.asarray(regression_trees, np.float64), REGRESSION_QUANTILE,
        axis=1, method="higher",
    )
    upgrade = np.quantile(
        np.asarray(upgrade_trees, np.float64), UPGRADE_QUANTILE,
        axis=1, method="lower",
    )
    return regression, upgrade


def _risk_thresholds(values: np.ndarray) -> list[float]:
    data = np.asarray(values, np.float64)
    result = [0.0, 1.0]
    if len(data):
        result.extend(map(float, np.quantile(
            data, [0.05, 0.10, 0.25, 0.50, 0.75], method="linear",
        )))
    return sorted(set(result))


def selection_metrics(
    arrays: Mapping[str, np.ndarray], regression: np.ndarray,
    upgrade: np.ndarray, params: Mapping[str, float],
) -> tuple[dict[str, Any], np.ndarray]:
    regression = np.asarray(regression, np.float64)
    upgrade = np.asarray(upgrade, np.float64)
    if len(regression) != len(arrays["decision"]) or len(upgrade) != len(regression):
        raise ValueError("outcome scores do not align with rows")
    risk_cap = float(params["max_outcome_regression_risk"])
    upgrade_threshold = float(params["outcome_upgrade_threshold"])
    choices = []
    outcome_deltas = []
    win_deltas = []
    margins = []
    selected_opponents = []
    selected_seeds = []
    selected_seats = []
    for rows in base.decision_slices(np.asarray(arrays["decision"], np.int64)):
        keep = rows[np.asarray(arrays["edit"])[rows] == base.KEEP_EDIT]
        if len(keep) != 1:
            raise ValueError("outcome selection lost its unique KEEP")
        chosen = int(keep[0])
        candidates = rows[np.asarray(arrays["edit"])[rows] != base.KEEP_EDIT]
        eligible = candidates[
            (regression[candidates] <= risk_cap)
            & (upgrade[candidates] > upgrade_threshold)
        ]
        if len(eligible):
            chosen = max(map(int, eligible), key=lambda row: (
                float(upgrade[row]), -float(regression[row]),
            ))
        keep_outcome = int(arrays["outcome"][int(keep[0])])
        chosen_outcome = int(arrays["outcome"][chosen])
        fired = int(arrays["edit"][chosen]) != int(base.KEEP_EDIT)
        choices.append(chosen)
        outcome_deltas.append(chosen_outcome - keep_outcome)
        win_deltas.append(
            int(chosen_outcome == 2) - int(keep_outcome == 2)
        )
        margins.append(float(arrays["delta_margin"][chosen]))
        if fired:
            selected_opponents.append(int(arrays["opponent"][chosen]))
            selected_seeds.append(int(arrays["seed"][chosen]))
            selected_seats.append(int(arrays["seat"][chosen]))
    outcome_delta = np.asarray(outcome_deltas, np.int16)
    win_delta = np.asarray(win_deltas, np.int16)
    margin = np.asarray(margins, np.float64)
    chosen_array = np.asarray(choices, np.int64)
    fired = np.asarray(arrays["edit"])[chosen_array] != base.KEEP_EDIT
    opponents = sorted(set(map(int, arrays["opponent"])))
    opponent_outcome_regressions = {
        str(opponent): int(sum(
            value < 0 for value, row in zip(
                outcome_delta, chosen_array, strict=True,
            ) if int(arrays["opponent"][row]) == opponent
        )) for opponent in opponents
    }
    opponent_win_regressions = {
        str(opponent): int(sum(
            value < 0 for value, row in zip(
                win_delta, chosen_array, strict=True,
            ) if int(arrays["opponent"][row]) == opponent
        )) for opponent in opponents
    }
    return {
        "decisions": len(choices),
        "fires": int(np.count_nonzero(fired)),
        "outcome_upgrades": int(np.count_nonzero(outcome_delta > 0)),
        "outcome_regressions": int(np.count_nonzero(outcome_delta < 0)),
        "wins_gained": int(np.count_nonzero(win_delta > 0)),
        "wins_lost": int(np.count_nonzero(win_delta < 0)),
        "raw_win_delta": int(win_delta.sum()),
        "outcome_delta_sum": int(outcome_delta.sum()),
        "same_outcome_margin_positive": int(np.count_nonzero(
            fired & (outcome_delta == 0) & (margin > 0.0)
        )),
        "same_outcome_margin_negative": int(np.count_nonzero(
            fired & (outcome_delta == 0) & (margin < 0.0)
        )),
        "selected_margin_delta_sum": float(margin.sum()),
        "selected_opponents": sorted(set(selected_opponents)),
        "selected_seeds": sorted(set(selected_seeds)),
        "selected_seats": sorted(set(selected_seats)),
        "opponent_outcome_regressions": opponent_outcome_regressions,
        "opponent_win_regressions": opponent_win_regressions,
        "max_outcome_regression_risk": risk_cap,
        "outcome_upgrade_threshold": upgrade_threshold,
    }, chosen_array


def calibrate(
    arrays: Mapping[str, np.ndarray], regression: np.ndarray,
    upgrade: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    trials = []
    for risk_cap in _risk_thresholds(regression):
        for upgrade_threshold in gate._thresholds(upgrade):
            metrics, _ = selection_metrics(arrays, regression, upgrade, {
                "max_outcome_regression_risk": risk_cap,
                "outcome_upgrade_threshold": upgrade_threshold,
            })
            trials.append(metrics)
    allowed = [row for row in trials if (
        row["outcome_regressions"] == 0
        and row["wins_lost"] == 0
        and not any(row["opponent_outcome_regressions"].values())
        and not any(row["opponent_win_regressions"].values())
    )]
    if not allowed:
        raise RuntimeError("outcome calibration lost its KEEP fallback")
    chosen = max(allowed, key=lambda row: (
        row["raw_win_delta"], row["wins_gained"],
        row["outcome_delta_sum"], row["selected_margin_delta_sum"],
        -row["fires"], -row["max_outcome_regression_risk"],
        row["outcome_upgrade_threshold"],
    ))
    params = {
        "max_outcome_regression_risk": chosen[
            "max_outcome_regression_risk"
        ],
        "outcome_upgrade_threshold": chosen["outcome_upgrade_threshold"],
    }
    metrics, choices = selection_metrics(
        arrays, regression, upgrade, params,
    )
    return {
        **params,
        "OOF_metrics": metrics,
        "upgrade_score": "lower_q10_of_equal_LOPO_and_LOSO_tree_probabilities",
        "regression_score": "upper_q90_of_equal_LOPO_and_LOSO_tree_probabilities",
    }, choices, trials


def full_scores(
    valid: Mapping[str, np.ndarray], train: Mapping[str, np.ndarray],
    trees: int, random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    regression_model = fit_classifier(
        train, "target_outcome_regression", 2 * trees, random_seed,
    )
    upgrade_model = fit_classifier(
        train, "target_outcome_upgrade", 2 * trees, random_seed + 1009,
    )
    features = np.asarray(valid["features"], np.float32)
    return outcome_scores(
        paired.risk_tree_predictions(regression_model, features),
        paired.risk_tree_predictions(upgrade_model, features),
    )


def oracle_headroom(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    gains = 0
    upgrades = 0
    available_folds = set()
    seeds = sorted(set(map(int, arrays["seed"])))
    fold_by_seed = gate.seed_fold_map(seeds)
    for rows in base.decision_slices(np.asarray(arrays["decision"], np.int64)):
        keep = rows[np.asarray(arrays["edit"])[rows] == base.KEEP_EDIT]
        keep_outcome = int(arrays["outcome"][int(keep[0])])
        best = max(int(arrays["outcome"][row]) for row in rows)
        upgrades += int(best > keep_outcome)
        gains += int(best == 2) - int(keep_outcome == 2)
        if best > keep_outcome:
            available_folds.add(fold_by_seed[int(arrays["seed"][rows[0]])])
    return {
        "decisions_with_outcome_upgrade": upgrades,
        "oracle_raw_win_gain": gains,
        "seed_folds_with_outcome_upgrade": sorted(available_folds),
    }


def diagnostics(
    arrays: Mapping[str, np.ndarray], choices: np.ndarray,
    seeds: Sequence[int],
) -> dict[str, Any]:
    slices = base.decision_slices(np.asarray(arrays["decision"], np.int64))
    if len(slices) != len(choices):
        raise ValueError("outcome diagnostics choices do not align")
    fold_by_seed = gate.seed_fold_map(seeds)
    cells: dict[str, dict[str, dict[str, float]]] = {
        "opponent": defaultdict(lambda: defaultdict(float)),
        "seed_fold": defaultdict(lambda: defaultdict(float)),
        "seat": defaultdict(lambda: defaultdict(float)),
    }
    overall: dict[str, float] = defaultdict(float)
    for rows, choice in zip(slices, choices, strict=True):
        keep = rows[np.asarray(arrays["edit"])[rows] == base.KEEP_EDIT]
        keep_outcome = int(arrays["outcome"][int(keep[0])])
        chosen_outcome = int(arrays["outcome"][choice])
        outcome_delta = chosen_outcome - keep_outcome
        win_delta = int(chosen_outcome == 2) - int(keep_outcome == 2)
        fired = int(arrays["edit"][choice]) != int(base.KEEP_EDIT)
        targets = (
            overall,
            cells["opponent"][str(int(arrays["opponent"][choice]))],
            cells["seed_fold"][str(fold_by_seed[int(arrays["seed"][choice])])],
            cells["seat"][str(int(arrays["seat"][choice]))],
        )
        for target in targets:
            target["decisions"] += 1
            target["fires"] += int(fired)
            target["outcome_upgrades"] += int(outcome_delta > 0)
            target["outcome_regressions"] += int(outcome_delta < 0)
            target["raw_win_delta"] += win_delta
            target["selected_wins"] += int(chosen_outcome == 2)
            target["keep_wins"] += int(keep_outcome == 2)
            target["margin_delta_sum"] += float(arrays["delta_margin"][choice])
    return {
        "overall": dict(overall),
        **{
            name: {key: dict(value) for key, value in sorted(group.items())}
            for name, group in cells.items()
        },
    }


def evaluate(
    arrays: Mapping[str, np.ndarray], *, trees: int = 24,
    random_seed: int = 20260829,
) -> dict[str, Any]:
    audit = base.validate_arrays(arrays)
    if trees < 2:
        raise ValueError("outcome two-head evaluation requires at least two trees")
    all_seeds = tuple(sorted(set(map(int, arrays["seed"]))))
    fold_by_seed = gate.seed_fold_map(all_seeds)
    if len(set(fold_by_seed.values())) != 4:
        raise ValueError("outcome nested evaluation requires four seed folds")
    row_folds = np.asarray([
        fold_by_seed[int(seed)] for seed in arrays["seed"]
    ], np.int8)
    chosen_by_decision: dict[int, int] = {}
    folds = []
    models_fit = 0
    trees_fit = 0
    for outer_fold in range(4):
        train_rows = np.flatnonzero(row_folds != outer_fold)
        valid_rows = np.flatnonzero(row_folds == outer_fold)
        train_raw = {
            key: np.asarray(value)[train_rows] for key, value in arrays.items()
        }
        valid_raw = {
            key: np.asarray(value)[valid_rows] for key, value in arrays.items()
        }
        train, train_source = outcome_arrays(train_raw)
        valid, valid_source = outcome_arrays(valid_raw)
        if not np.any(train["target_outcome_upgrade"]):
            return {
                "schema": SCHEMA,
                "status": "outer_train_has_no_outcome_upgrade_signal",
                "failed_fold": outer_fold,
                "input_audit": audit,
                "models_fit": models_fit,
            }
        rr_lopo, ur_lopo, lopo_audit = group_oof_probabilities(
            train, "opponent", trees,
            random_seed + outer_fold * 100_003 + 101,
        )
        rr_loso, ur_loso, loso_audit = group_oof_probabilities(
            train, "seed", trees,
            random_seed + outer_fold * 100_003 + 1101,
        )
        regression, upgrade = outcome_scores(
            np.concatenate([rr_lopo, rr_loso], axis=1),
            np.concatenate([ur_lopo, ur_loso], axis=1),
        )
        calibration, _, _ = calibrate(train, regression, upgrade)
        valid_regression, valid_upgrade = full_scores(
            valid, train, trees,
            random_seed + outer_fold * 100_003 + 20_101,
        )
        metrics, choices = selection_metrics(
            valid, valid_regression, valid_upgrade, calibration,
        )
        group_models = 2 * (len(lopo_audit) + len(loso_audit))
        models_fit += group_models + 2
        trees_fit += group_models * trees + 4 * trees
        for rows, choice in zip(
            base.decision_slices(valid["decision"]), choices, strict=True,
        ):
            decision = int(valid["decision"][rows[0]])
            chosen_by_decision[decision] = int(
                valid_rows[int(valid_source[int(choice)])]
            )
        folds.append({
            "left_out_seed_fold": outer_fold,
            "train_seeds": sorted(set(map(int, train["seed"]))),
            "valid_seeds": sorted(set(map(int, valid["seed"]))),
            "calibration_valid_seed_overlap": [],
            "train_outcome_upgrade_rows": int(np.count_nonzero(
                train["target_outcome_upgrade"]
            )),
            "train_outcome_regression_rows": int(np.count_nonzero(
                train["target_outcome_regression"]
            )),
            "inner_chosen": calibration,
            "outer_metrics": metrics,
            "outer_oracle_headroom": oracle_headroom(valid),
            "group_OOF_audit": {
                "opponent": lopo_audit, "seed": loso_audit,
            },
        })
    slices = base.decision_slices(np.asarray(arrays["decision"], np.int64))
    decision_order = [int(arrays["decision"][rows[0]]) for rows in slices]
    if set(decision_order) != set(chosen_by_decision):
        raise RuntimeError("outcome nested choices missed decisions")
    choices = np.asarray([
        chosen_by_decision[decision] for decision in decision_order
    ], np.int64)
    return {
        "schema": SCHEMA,
        "status": "nested_seed_fold_evaluation_complete",
        "scheme": "A-only outcome-upgrade lower-q10 and outcome-regression upper-q90",
        "outer_labels_used_for_calibration": False,
        "folds": folds,
        "A_choices": choices,
        "final_choices": choices.copy(),
        "outcome_diagnostics": diagnostics(arrays, choices, all_seeds),
        "oracle_headroom": oracle_headroom(outcome_arrays(arrays)[0]),
        "input_audit": audit,
        "models_fit": models_fit,
        "trees_fit": trees_fit,
        "objective_contract": {
            "primary": "raw win gain with zero outcome/win regression",
            "secondary": "outcome delta then margin delta",
            "phase_enabled": False,
            "margin_negative_same_outcome_is_not_a_win_rate_regression": True,
        },
    }
