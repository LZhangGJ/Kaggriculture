#!/usr/bin/env python3
"""Locked optimization-only stability study for the 16 formal control views."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

import train_phase_challenger_bilinear_listwise_A_v2 as frozen
import train_phase_challenger_bilinear_listwise_optimizer_stability_v2_1_locked as locked


SCHEMA = "control-view-optimizer-stability-v2.1-locked"
LOCKED_SOLVER = "L-BFGS-B-1000"
INTEGRATION_SHORTCUT_TEST_CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        "id": "all_equal_listwise_skips_optimizer",
        "precondition": "every candidate in a training objective has the same outcome as its decision peers",
        "required_behavior": "do not call minimize; return exact zero coefficients and a constant-safe audit",
        "required_test": "monkeypatch minimize to raise and assert exact zero coefficients",
    },
    {
        "id": "single_class_risk_skips_optimizer",
        "precondition": "the non-KEEP regression-risk target contains exactly one class",
        "required_behavior": "do not call minimize; emit the observed class as the constant probability",
        "required_test": "monkeypatch minimize to raise and assert the exact constant probability",
    },
    {
        "id": "no_credible_inner_gain_forces_KEEP",
        "precondition": "inner-OOF calibration has no safe credible ordinal or win gain",
        "required_behavior": "set threshold to positive infinity and select KEEP for every outer decision",
        "required_test": "mutate only inner labels to a null panel and assert threshold=inf and all KEEP",
    },
)


def control_view_specs() -> tuple[dict[str, Any], ...]:
    views: list[dict[str, Any]] = []
    for outer_fold in range(4):
        train_folds = tuple(fold for fold in range(4) if fold != outer_fold)
        views.append({
            "view_id": f"outer_{outer_fold}_bilinear_no_SHA",
            "view_kind": "bilinear_no_SHA",
            "outer_fold": outer_fold,
            "train_fold_ids": train_folds,
            "include_SHA_soft_prior": False,
            "shuffle_seed": None,
            "expected_feature_width": locked.frozen.BASE_WIDTH,
        })
        for shuffle_seed in frozen.SHUFFLE_SEEDS:
            views.append({
                "view_id": f"outer_{outer_fold}_state_shuffle_{shuffle_seed}",
                "view_kind": "state_shuffle",
                "outer_fold": outer_fold,
                "train_fold_ids": train_folds,
                "include_SHA_soft_prior": True,
                "shuffle_seed": int(shuffle_seed),
                "expected_feature_width": locked.frozen.PRIMARY_WIDTH,
            })
    identities = {view["view_id"] for view in views}
    if len(views) != 16 or len(identities) != 16:
        raise AssertionError("the sixteen frozen formal control views changed")
    return tuple(views)


def build_train_features(
    arrays: Mapping[str, np.ndarray], train: np.ndarray, view: Mapping[str, Any],
) -> np.ndarray:
    if bool(view["include_SHA_soft_prior"]):
        prior, _ = frozen._crossfit_prior_features(
            arrays, train, float(frozen.DEFAULT_CONFIG["sha_prior_alpha"]),
        )
    else:
        prior = None
    train_x, _ = frozen._projected_features(
        arrays, train, train,
        train_prior=prior, predict_prior=prior,
        shuffle_seed=view["shuffle_seed"],
    )
    expected_width = int(view["expected_feature_width"])
    if train_x.shape != (len(train), expected_width) or not np.isfinite(train_x).all():
        raise RuntimeError(f"control-view feature contract changed: {view['view_id']}")
    return train_x


def _solver_stable(report: Mapping[str, Any]) -> bool:
    if bool(report["constant_safe"]):
        return True
    return bool(report["solvers"].get(LOCKED_SOLVER, {}).get("stable", False))


def optimization_only_control_study(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    input_audit = frozen.validate_arrays(arrays)
    row_fold = frozen._row_folds(np.asarray(arrays["seed"]))
    all_rows = np.arange(len(row_fold), dtype=np.int64)
    reports: list[dict[str, Any]] = []
    optimizer_runs = 0
    for view in control_view_specs():
        fold_ids = tuple(map(int, view["train_fold_ids"]))
        train = all_rows[np.isin(row_fold, fold_ids)]
        train_x = build_train_features(arrays, train, view)
        listwise_problem = locked.listwise_problem(
            train_x, arrays, train,
            objective="signal_only_uniform_top_outcome_cross_entropy",
        )
        risk_problem = locked.risk_problem(train_x, arrays, train)
        listwise = locked.study_problem(listwise_problem, (LOCKED_SOLVER,))
        risk = locked.study_problem(risk_problem, (LOCKED_SOLVER,))
        optimizer_runs += sum(
            len(head["solvers"][LOCKED_SOLVER]["runs"])
            for head in (listwise, risk) if not head["constant_safe"]
        )
        reports.append({
            **view,
            "train_fold_ids": list(fold_ids),
            "training_rows": int(len(train)),
            "training_decisions": len(frozen.decision_slices(np.asarray(arrays["decision"]), train)),
            "feature_width": int(train_x.shape[1]),
            "training_feature_sha256": hashlib.sha256(train_x.tobytes()).hexdigest(),
            "total_weight": listwise_problem["total_weight"],
            "signal_weight": listwise_problem["signal_weight"],
            "signal_weight_rate": listwise_problem["signal_weight_rate"],
            "listwise": listwise, "risk": risk,
            "listwise_stable": _solver_stable(listwise),
            "risk_stable": _solver_stable(risk),
            "heldout_feature_constructed": False,
            "heldout_prediction_executed": False,
            "heldout_outcome_metric_computed": False,
        })

    all_stable = all(view["listwise_stable"] and view["risk_stable"] for view in reports)
    expected_runs = 16 * 2 * len(locked.INITIALIZATIONS)
    if optimizer_runs != expected_runs:
        raise RuntimeError(
            f"control-view run coverage changed: {optimizer_runs} != {expected_runs}"
        )
    return {
        "schema": SCHEMA,
        "status": "control_views_optimization_stable" if all_stable else "control_views_optimization_unstable",
        "config": {
            "listwise_objective": "signal_only_uniform_top_outcome_cross_entropy",
            "listwise_semantics": "top-outcome-set classification; not full ordinal",
            "solver": LOCKED_SOLVER,
            "initializations": locked.INITIALIZATIONS,
            "listwise_l2_per_cell": locked.LISTWISE_L2_PER_CELL,
            "risk_l2_per_cell": locked.RISK_L2_PER_CELL,
            "loss_denominator": "all paired-cell mass",
            "control_views": 16, "heads_per_view": 2,
            "expected_optimizer_runs": expected_runs,
        },
        "boundary": {
            "train_proxy_only": True,
            "heldout_features": False, "heldout_predictions": False,
            "heldout_outcome_acceptance": False,
            "validation_opened": False, "sealed_opened": False,
            "choices_generated": False, "candidate_committed": False,
        },
        "input_audit": input_audit,
        "views": reports,
        "optimizer_runs": optimizer_runs,
        "integration_shortcut_test_contracts": INTEGRATION_SHORTCUT_TEST_CONTRACTS,
        "formal_integration_gate": {
            "all_16_views_both_heads_stable": all_stable,
            "all_64_optimizer_runs_accounted": optimizer_runs == expected_runs,
            "heldout_outcome_metrics_used": False,
            "ready": all_stable and optimizer_runs == expected_runs,
        },
    }


__all__ = [
    "INTEGRATION_SHORTCUT_TEST_CONTRACTS", "LOCKED_SOLVER", "SCHEMA",
    "build_train_features", "control_view_specs", "optimization_only_control_study",
]
