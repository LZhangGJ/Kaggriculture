"""Exact decision-event threshold ablation for dual-OOF residual calibration."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

import run_farmer_augment_union_mpc_v1 as gate
import train_phase_challenger_residual_dual_oof_nested_v1 as dual


SCHEMA = "phase-challenger-residual-dual-oof-exact-nested-v1"


def event_thresholds(
    arrays: Mapping[str, np.ndarray], scores: np.ndarray, *, farmer_only: bool,
) -> list[float]:
    values = []
    for rows in gate._decision_slices(np.asarray(arrays["decision"], np.int64)):
        candidates = rows
        if not farmer_only:
            candidates = rows[np.asarray(arrays["edit"])[rows] != 0]
        if len(candidates):
            value = float(np.max(np.asarray(scores)[candidates]))
            if value > 0.0:
                values.append(value)
    result = [0.0, *sorted(set(values))]
    if values:
        result.append(float(max(values) + 1.0))
    return sorted(set(result))


def exact_dual_view_calibration(
    arrays: Mapping[str, np.ndarray],
    opponent_moments: tuple[np.ndarray, np.ndarray, np.ndarray],
    seed_moments: tuple[np.ndarray, np.ndarray, np.ndarray],
    *, farmer_only: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    om, ost, op = opponent_moments
    sm, sst, sp = seed_moments
    trials = []
    for beta in (0.0, 0.5, 1.0):
        thresholds = sorted(set(
            event_thresholds(arrays, om - beta * ost, farmer_only=farmer_only)
            + event_thresholds(arrays, sm - beta * sst, farmer_only=farmer_only)
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
        raise RuntimeError("exact dual-view scan lost the KEEP/A fallback")
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
        "threshold_grid": "exact_unique_decision_event_scores",
        "selection_objective": "maximin_view_sum_then_benefit_then_total",
        "worst_view_sum": chosen["worst_view_sum"],
        "positive_utility_both_views": (
            chosen["worst_view_sum"] > 0.0
            and chosen["worst_view_beneficial"] > 0
        ),
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
    original = dual.dual_view_calibration
    dual.dual_view_calibration = exact_dual_view_calibration
    try:
        result = dual.evaluate(
            arrays, trees=trees, random_seed=random_seed,
        )
    finally:
        dual.dual_view_calibration = original
    result["schema"] = SCHEMA
    result["calibration_ablation"] = "exact_decision_event_threshold_grid"
    return result
