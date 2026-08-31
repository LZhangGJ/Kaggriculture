#!/usr/bin/env python3
"""Pure direct U_phase ranker used only for full/compact representation A/B.

The core performs no file I/O.  It deliberately calibrates safety against the
same-decision KEEP row, not against a previously selected R0 action; therefore
it is a diagnostic baseline and not a replacement for the residual gate.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for import_path in (
    Path(__file__).resolve().parent,
    CODE_ROOT / "src",
    CODE_ROOT / "fast_kaggriculture" / "python",
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_farmer_augment_union_mpc_v1 as gate_base
import run_route_residual_adapter_v0 as residual
import train_phase_challenger_residual_gate_v1 as residual_contract


SCHEMA = "phase-challenger-direct-ranker-v1"
KEEP_EDIT = residual_contract.KEEP_EDIT
MIN_POSITIVE_DECISIONS = 4
MIN_POSITIVE_OPPONENTS = 2
MIN_POSITIVE_SEEDS = 2
SEED_FOLDS = 4


def validate(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Validate the residual-gate Mapping contract and direct-ranker boundary."""

    audit = residual_contract.validate_arrays(arrays)
    return {
        **audit,
        "target": "signed_log(candidate margin minus same-decision KEEP margin)",
        "zero_harm_reference": "KEEP",
        "architecture_role": "direct_union_representation_ablation_only",
        "candidate_sha_order": "caller_prevalidated",
    }


def _signal(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    delta = np.asarray(arrays["delta_margin"], np.float64)
    edit = np.asarray(arrays["edit"], np.int16)
    positive = np.flatnonzero((edit != KEEP_EDIT) & (delta > 0.0))
    decisions = sorted(set(map(int, np.asarray(arrays["decision"])[positive])))
    opponents = sorted(set(map(int, np.asarray(arrays["opponent"])[positive])))
    seeds = sorted(set(map(int, np.asarray(arrays["seed"])[positive])))
    reasons = []
    if len(decisions) < MIN_POSITIVE_DECISIONS:
        reasons.append("positive_non_KEEP_decisions_below_minimum")
    if len(opponents) < MIN_POSITIVE_OPPONENTS:
        reasons.append("positive_non_KEEP_opponent_coverage_below_minimum")
    if len(seeds) < MIN_POSITIVE_SEEDS:
        reasons.append("positive_non_KEEP_seed_coverage_below_minimum")
    all_seeds = sorted(set(map(int, np.asarray(arrays["seed"]))))
    if len(all_seeds) < SEED_FOLDS:
        reasons.append("four_seed_fold_evaluation_requires_at_least_four_seeds")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "positive_rows": int(len(positive)),
        "positive_decisions": int(len(decisions)),
        "positive_opponents": opponents,
        "positive_seeds": seeds,
        "all_seeds": all_seeds,
        "minimum_decisions": MIN_POSITIVE_DECISIONS,
        "minimum_opponents": MIN_POSITIVE_OPPONENTS,
        "minimum_seeds": MIN_POSITIVE_SEEDS,
    }


def _fixed_seed_fold_evaluation(
    arrays: Mapping[str, np.ndarray],
    *,
    trees: int,
    random_seed: int,
) -> dict[str, Any]:
    """Nested evaluation: inner LOPO calibration, outer unseen-seed scoring."""

    seeds = np.asarray(arrays["seed"], np.int64)
    opponents = np.asarray(arrays["opponent"], np.int64)
    fold_by_seed = gate_base.seed_fold_map(sorted(set(map(int, seeds))), SEED_FOLDS)
    row_folds = np.asarray([fold_by_seed[int(seed)] for seed in seeds], np.int8)
    mean = np.full(len(seeds), np.nan, np.float64)
    std = np.full(len(seeds), np.nan, np.float64)
    positive = np.full(len(seeds), np.nan, np.float64)
    folds = []
    choice_by_decision: dict[int, int] = {}
    for fold in range(SEED_FOLDS):
        valid = np.flatnonzero(row_folds == fold)
        train = np.flatnonzero(row_folds != fold)
        if not len(valid) or not len(train):
            raise ValueError("four-fold seed evaluation requires a non-empty train and valid split")
        train_arrays = {key: np.asarray(value)[train] for key, value in arrays.items()}
        valid_arrays = {key: np.asarray(value)[valid] for key, value in arrays.items()}
        inner_mean, inner_std, inner_positive, inner_folds = (
            gate_base._lopo_predictions(
                train_arrays, trees, random_seed + 10_007 + fold * 4099,
            )
        )
        inner_opponents = set(map(int, train_arrays["opponent"]))
        if {
            int(row["left_out_opponent_index"]) for row in inner_folds
        } != inner_opponents:
            raise AssertionError("inner LOPO folds do not cover every outer-train opponent")
        audited_inner_folds = [{
            **row,
            "train_opponent_indices": sorted(
                inner_opponents - {int(row["left_out_opponent_index"])}
            ),
            "valid_opponent_indices": [int(row["left_out_opponent_index"])],
            "group_leakage": False,
        } for row in inner_folds]
        inner_chosen, inner_trials, _ = gate_base.zero_harm_calibration(
            train_arrays, inner_mean, inner_std, inner_positive,
            farmer_only=False,
        )
        if not any(
            int(row["selected"]) == 0
            and int(row["harmful"]) == 0
            and float(row["sum_realized_delta"]) == 0.0
            for row in inner_trials
        ):
            raise AssertionError("nested calibration lost the KEEP-only fallback")
        model = residual.ExtraTreesRegressor(**residual._model_params(
            trees, random_seed + 50_021 + fold * 1009,
        ))
        model.fit(
            train_arrays["features"], train_arrays["target_signed_log_margin"],
            sample_weight=gate_base._decision_weights(
                np.asarray(train_arrays["decision"]),
            ),
        )
        predictions = residual._tree_predictions(model, valid_arrays["features"])
        mean[valid] = predictions.mean(axis=1)
        std[valid] = predictions.std(axis=1)
        positive[valid] = (predictions > 0.0).mean(axis=1)
        _, local_choices = gate_base._selection_metrics(
            valid_arrays, mean[valid], std[valid], positive[valid],
            inner_chosen, farmer_only=False,
        )
        for indices, choice in zip(
            residual_contract.decision_slices(
                np.asarray(valid_arrays["decision"], np.int64),
            ),
            local_choices,
            strict=True,
        ):
            decision = int(valid_arrays["decision"][indices[0]])
            choice_by_decision[decision] = int(valid[int(choice)])
        heldout = _oof_diagnostics(valid_arrays, local_choices)
        train_seeds = sorted(set(map(int, seeds[train])))
        valid_seeds = sorted(set(map(int, seeds[valid])))
        if set(train_seeds) & set(valid_seeds):
            raise AssertionError("outer calibration seeds leaked into held seed fold")
        folds.append({
            "left_out_seed_fold": fold,
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "train_seeds": train_seeds,
            "valid_seeds": valid_seeds,
            "train_opponents": sorted(set(map(int, opponents[train]))),
            "valid_opponents": sorted(set(map(int, opponents[valid]))),
            "seed_group_leakage": bool(set(train_seeds) & set(valid_seeds)),
            "threshold_selection_rows": int(len(train)),
            "threshold_selection_seeds": train_seeds,
            "calibration_seeds": train_seeds,
            "threshold_selection_valid_seed_overlap": sorted(
                set(train_seeds) & set(valid_seeds)
            ),
            "threshold_selection_rows_exclude_valid_seeds": not bool(
                set(train_seeds) & set(valid_seeds)
            ),
            "inner_LOPO_folds": audited_inner_folds,
            "inner_LOPO_complete": len(audited_inner_folds) == len(inner_opponents),
            "inner_chosen": inner_chosen,
            "outer_heldout": {
                "overall": heldout["overall"],
                "by_opponent": heldout["left_out_opponent"],
            },
        })
    if not all(np.isfinite(value).all() for value in (mean, std, positive)):
        raise RuntimeError("leave-seed-fold-out prediction panel is incomplete")
    slices = residual_contract.decision_slices(
        np.asarray(arrays["decision"], np.int64),
    )
    choices = np.asarray([
        choice_by_decision[int(arrays["decision"][indices[0]])]
        for indices in slices
    ], np.int64)
    diagnostics = _oof_diagnostics(arrays, choices)
    zero_fold_harm = all(
        int(cell["harmful"]) == 0 for cell in diagnostics["seed_fold"].values()
    )
    zero_opponent_harm = all(
        int(cell["harmful"]) == 0
        for cell in diagnostics["left_out_opponent"].values()
    )
    return {
        "scheme": "outer_leave_seed_fold_out_with_inner_LOPO_calibration",
        "calibration_source": "independent inner LOPO on each outer-train partition",
        "global_calibration_used_for_outer": False,
        "outer_labels_used_to_select_threshold": False,
        "threshold_reselected_on_outer": False,
        "complete_four_folds": len(folds) == SEED_FOLDS,
        "folds": folds,
        "overall": diagnostics["overall"],
        "by_seed_fold": diagnostics["seed_fold"],
        "by_opponent": diagnostics["left_out_opponent"],
        "opponents_with_fire": diagnostics["opponents_with_fire"],
        "seeds_with_fire": diagnostics["seeds_with_fire"],
        "zero_harm_each_seed_fold": zero_fold_harm,
        "zero_harm_each_opponent": zero_opponent_harm,
        "zero_harm_fixed_threshold": zero_fold_harm and zero_opponent_harm,
        "oof": {
            "mean": mean,
            "std": std,
            "positive_fraction": positive,
            "left_out_seed_fold": row_folds,
        },
        "choices": choices,
    }


def _cell() -> dict[str, float]:
    return defaultdict(float)


def _finish_cell(cell: Mapping[str, float]) -> dict[str, float | int]:
    decisions = int(cell.get("decisions", 0.0))
    realized = float(cell.get("realized_sum", 0.0))
    return {
        "decisions": decisions,
        "fires": int(cell.get("fires", 0.0)),
        "harmful": int(cell.get("harmful", 0.0)),
        "beneficial": int(cell.get("beneficial", 0.0)),
        "realized_sum": realized,
        "mean_realized": realized / decisions if decisions else 0.0,
    }


def _oof_diagnostics(
    arrays: Mapping[str, np.ndarray], choices: np.ndarray,
) -> dict[str, Any]:
    slices = residual_contract.decision_slices(
        np.asarray(arrays["decision"], np.int64),
    )
    choices = np.asarray(choices, np.int64)
    if len(slices) != len(choices):
        raise ValueError("OOF choices do not align with decisions")
    fold_by_seed = gate_base.seed_fold_map(
        sorted(set(map(int, np.asarray(arrays["seed"])))), SEED_FOLDS,
    )
    overall = _cell()
    by_opponent: dict[str, dict[str, float]] = defaultdict(_cell)
    by_seed_fold = {str(fold): _cell() for fold in range(SEED_FOLDS)}
    fired_opponents: set[int] = set()
    fired_seeds: set[int] = set()
    for indices, choice in zip(slices, choices, strict=True):
        if int(choice) not in set(map(int, indices)):
            raise ValueError("OOF choice escaped its decision")
        keep = int(indices[0])
        if int(arrays["edit"][keep]) != KEEP_EDIT:
            raise AssertionError("validated leading KEEP changed")
        opponent = int(arrays["opponent"][keep])
        seed = int(arrays["seed"][keep])
        fired = int(choice != keep)
        value = float(arrays["delta_margin"][choice])
        if fired:
            fired_opponents.add(opponent)
            fired_seeds.add(seed)
        for cell in (overall, by_opponent[str(opponent)], by_seed_fold[str(fold_by_seed[seed])]):
            cell["decisions"] += 1
            cell["fires"] += fired
            cell["harmful"] += int(value < 0.0)
            cell["beneficial"] += int(value > 0.0)
            cell["realized_sum"] += value
    return {
        "overall": _finish_cell(overall),
        "left_out_opponent": {
            key: _finish_cell(value) for key, value in sorted(by_opponent.items())
        },
        "seed_fold": {
            key: _finish_cell(value) for key, value in sorted(by_seed_fold.items())
        },
        "opponents_with_fire": sorted(fired_opponents),
        "seeds_with_fire": sorted(fired_seeds),
        "seed_fold_count": SEED_FOLDS,
    }


def fit_direct_ranker(
    arrays: Mapping[str, np.ndarray],
    *,
    trees: int = 96,
    cv_trees: int = 24,
    random_seed: int = 20260829,
) -> dict[str, Any]:
    """Fit one candidate-value model with complete leave-one-opponent-out OOF."""

    audit = validate(arrays)
    signal = _signal(arrays)
    boundary = {
        "file_IO": False,
        "validation_or_sealed_access": False,
        "zero_harm_reference": "KEEP",
        "not_a_residual_gate_replacement": True,
        "outcome_head": False,
    }
    if not signal["passed"]:
        return {
            "schema": SCHEMA,
            "status": "data_insufficient_before_fit",
            "input_audit": audit,
            "signal": signal,
            "model": None,
            "models_fit": 0,
            "data_boundary": boundary,
        }
    if trees < 1 or cv_trees < 1:
        raise ValueError("trees and cv_trees must be positive")

    direct = {key: np.asarray(value) for key, value in arrays.items()}
    delta = np.asarray(arrays["delta_margin"], np.float64)
    direct["target_signed_log_margin"] = (
        np.sign(delta) * np.log1p(np.abs(delta))
    ).astype(np.float32)
    mean, std, positive, folds = gate_base._lopo_predictions(
        direct, cv_trees, random_seed,
    )
    if not all(np.isfinite(value).all() for value in (mean, std, positive)):
        raise RuntimeError("LOPO OOF prediction panel is incomplete")
    chosen, trials, choices = gate_base.zero_harm_calibration(
        direct, mean, std, positive, farmer_only=False,
    )
    fallback_present = any(
        int(row["selected"]) == 0
        and int(row["harmful"]) == 0
        and float(row["sum_realized_delta"]) == 0.0
        for row in trials
    )
    if not fallback_present:
        raise AssertionError("calibration grid lost the KEEP-only fallback")
    seed_fold_evaluation = _fixed_seed_fold_evaluation(
        direct, trees=cv_trees, random_seed=random_seed,
    )

    weights = gate_base._decision_weights(np.asarray(direct["decision"], np.int64))
    weight_sums = [float(weights[indices].sum()) for indices in residual_contract.decision_slices(
        np.asarray(direct["decision"], np.int64),
    )]
    if not np.allclose(weight_sums, 1.0):
        raise AssertionError("decision-wise training weights are not equal")
    model = residual.ExtraTreesRegressor(**residual._model_params(trees, random_seed))
    model.fit(
        direct["features"], direct["target_signed_log_margin"],
        sample_weight=weights,
    )
    diagnostics = _oof_diagnostics(direct, choices)
    zero_harm = not any(
        int(cell["harmful"]) for cell in diagnostics["left_out_opponent"].values()
    )
    if not zero_harm:
        raise AssertionError("calibrated direct ranker lost per-LOPO zero harm")
    fold_audit = []
    all_opponents = set(map(int, np.asarray(direct["opponent"])))
    for fold in folds:
        left_out = int(fold["left_out_opponent_index"])
        fold_audit.append({
            **fold,
            "train_opponent_indices": sorted(all_opponents - {left_out}),
            "valid_opponent_indices": [left_out],
            "group_leakage": False,
        })
    return {
        "schema": SCHEMA,
        "status": "fit_complete_train_only_core",
        "input_audit": audit,
        "signal": signal,
        "model": model,
        "models_fit": 1,
        "calibration": {
            "primary_grouping": "leave-one-train-opponent-out",
            "scope": "all training seeds; deployment calibration only",
            "used_for_nested_outer_evaluation": False,
            "folds": fold_audit,
            "chosen": chosen,
            "trials": trials,
            "zero_harm_each_lopo_fold": zero_harm,
            "zero_harm_interpretation": (
                "calibration constraint only; fixed-threshold seed-fold result "
                "is the safety estimate"
            ),
            "keep_only_fallback_present": fallback_present,
            "decision_weight_sum_min": min(weight_sums),
            "decision_weight_sum_max": max(weight_sums),
            "model_params": residual._model_params(trees, random_seed),
        },
        "oof": {
            "mean": mean,
            "std": std,
            "positive_fraction": positive,
            "left_out_opponent": np.asarray(direct["opponent"], np.int64).copy(),
        },
        "oof_choices": choices,
        "OOF_diagnostics": diagnostics,
        "seed_fold_fixed_threshold": seed_fold_evaluation,
        "selection_contract": {
            "fallback": "unique leading KEEP",
            "fire": "best non-KEEP LCB strictly exceeds threshold and positive-tree fraction gate",
            "ties": "first canonical non-KEEP wins; R0 prefix precedes phase-only suffix",
            "zero_harm_reference": "KEEP",
        },
        "data_boundary": boundary,
    }


def choose_direct(
    arrays: Mapping[str, np.ndarray],
    model: Any,
    calibration: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Choose one canonical union row without consulting its labels."""

    audit = validate(arrays)
    if audit["decisions"] != 1:
        raise ValueError("deployment choice requires exactly one decision panel")
    features = np.asarray(arrays["features"], np.float32)
    predictions = residual._tree_predictions(model, features)
    mean = predictions.mean(axis=1)
    std = predictions.std(axis=1)
    positive = (predictions > 0.0).mean(axis=1)
    chosen_calibration = calibration.get("chosen", calibration)
    keep = 0
    selected = keep
    candidates = np.flatnonzero(np.asarray(arrays["edit"], np.int16) != KEEP_EDIT)
    score = np.empty(0, np.float64)
    candidate = -1
    if len(candidates):
        score = mean[candidates] - float(chosen_calibration["beta"]) * std[candidates]
        local = int(np.argmax(score))
        candidate = int(candidates[local])
        if (
            float(score[local]) > float(chosen_calibration["threshold"])
            and float(positive[candidate])
            >= float(chosen_calibration["min_positive_fraction"])
        ):
            selected = candidate
    return selected, {
        "fired": selected != keep,
        "fallback_exact_leading_KEEP": selected == keep,
        "candidate_considered": candidate,
        "source": "R0" if int(arrays["membership"][selected]) == 0 else "phase_only",
        "mean": mean,
        "std": std,
        "positive_fraction": positive,
        "non_KEEP_lcb": score,
        "strict_threshold": True,
        "canonical_tie_order_preserved": True,
        "zero_harm_reference": "KEEP",
    }


def compare_representations(
    full_result: Mapping[str, Any], compact_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the pre-registered pure-metric compact-vs-full acceptance gate."""

    def diagnostics(result: Mapping[str, Any]) -> Mapping[str, Any]:
        if result.get("status") != "fit_complete_train_only_core":
            raise ValueError("representation result is not a completed direct-ranker fit")
        return result["OOF_diagnostics"]

    full = diagnostics(full_result)
    compact = diagnostics(compact_result)
    full_opponents = full["left_out_opponent"]
    compact_opponents = compact["left_out_opponent"]
    cells_match = set(full_opponents) == set(compact_opponents)
    compact_zero_harm = cells_match and all(
        int(cell["harmful"]) == 0 for cell in compact_opponents.values()
    )
    try:
        full_fixed = full_result["seed_fold_fixed_threshold"]
        compact_fixed = compact_result["seed_fold_fixed_threshold"]
    except KeyError as error:
        raise ValueError(
            "representation result lacks fixed-threshold seed-fold evaluation"
        ) from error
    nested_protocol = all(
        bool(result.get("complete_four_folds"))
        and result.get("global_calibration_used_for_outer") is False
        and result.get("outer_labels_used_to_select_threshold") is False
        for result in (full_fixed, compact_fixed)
    )
    compact_fixed_zero_harm = (
        bool(compact_fixed["zero_harm_fixed_threshold"])
        and all(
            int(cell["harmful"]) == 0
            for cell in compact_fixed["by_seed_fold"].values()
        )
        and all(
            int(cell["harmful"]) == 0
            for cell in compact_fixed["by_opponent"].values()
        )
    )
    fires = int(compact_fixed["overall"]["fires"])
    fire_opponents = len(compact_fixed["opponents_with_fire"])
    fire_seeds = len(compact_fixed["seeds_with_fire"])
    full_sum = float(full_fixed["overall"]["realized_sum"])
    compact_sum = float(compact_fixed["overall"]["realized_sum"])
    retention = compact_sum >= 0.98 * full_sum
    full_fixed_opponents = full_fixed["by_opponent"]
    compact_fixed_opponents = compact_fixed["by_opponent"]
    fixed_cells_match = set(full_fixed_opponents) == set(compact_fixed_opponents)
    per_opponent_delta = {
        key: float(compact_fixed_opponents[key]["realized_sum"])
        - float(full_fixed_opponents[key]["realized_sum"])
        for key in sorted(set(full_fixed_opponents) & set(compact_fixed_opponents))
    }
    per_opponent_nonnegative = fixed_cells_match and all(
        value >= 0.0 for value in per_opponent_delta.values()
    )
    checks = {
        "nested_outer_protocol_verified": nested_protocol,
        "same_nested_opponent_cells": fixed_cells_match,
        "compact_fixed_threshold_zero_harm_each_seed_fold_and_opponent": (
            compact_fixed_zero_harm
        ),
        "compact_fires_at_least_4": fires >= 4,
        "compact_fires_cover_at_least_2_opponents": fire_opponents >= 2,
        "compact_fires_cover_at_least_2_seeds": fire_seeds >= 2,
        "compact_realized_sum_at_least_98pct_full": retention,
        "compact_minus_full_nonnegative_each_opponent": per_opponent_nonnegative,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "calibration_diagnostics_not_used_by_gate": {
            "same_LOPO_opponent_cells": cells_match,
            "compact_zero_harm_each_LOPO": compact_zero_harm,
        },
        "full_realized_sum": full_sum,
        "compact_realized_sum": compact_sum,
        "retention_ratio": (
            compact_sum / full_sum if full_sum > 0.0 else (1.0 if compact_sum >= 0.0 else float("-inf"))
        ),
        "compact_fires": fires,
        "compact_fire_opponents": fire_opponents,
        "compact_fire_seeds": fire_seeds,
        "per_opponent_compact_minus_full": per_opponent_delta,
        "candidate_sha_order": "must_be_prevalidated_by_caller; not inspected by this metric gate",
        "architecture_boundary": "direct union, zero-harm reference KEEP; representation A/B only",
        "evaluation_boundary": (
            "LOPO selects calibration; all fire/coverage/retention/replacement metrics "
            "use the frozen rule on leave-seed-fold-out predictions"
        ),
    }
