#!/usr/bin/env python3
"""Pure two-level LOPO training core for R0 plus phase challengers.

The module accepts already encoded, contiguous decision arrays.  It performs
no file I/O and exposes no experiment CLI, so validation/sealed data cannot be
opened here.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

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


SCHEMA = "phase-challenger-residual-gate-v1"
R0 = np.int8(0)
PHASE_ONLY = np.int8(1)
KEEP_EDIT = np.int8(0)
MIN_POSITIVE_DECISIONS = 4
MIN_POSITIVE_OPPONENTS = 2
MIN_POSITIVE_SEEDS = 2
REQUIRED = (
    "features", "delta_margin", "membership", "decision", "opponent",
    "seed", "edit",
)
ARRAY_CONTRACT = {
    "features": "float32 [N,D]",
    "delta_margin": "float64 [N], candidate margin minus same-decision KEEP margin",
    "membership": "int8 [N], 0=R0 prefix, 1=phase-only suffix",
    "decision": "int64 [N], nondecreasing contiguous decision groups",
    "opponent": "integer [N], constant within decision; LOPO group",
    "seed": "int64 [N], constant within decision; seed-fold diagnostic group",
    "edit": "integer [N], 0=KEEP; exactly one leading KEEP per decision",
}


def decision_slices(decision: np.ndarray) -> list[np.ndarray]:
    return gate_base._decision_slices(np.asarray(decision, np.int64))


def validate_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    missing = [key for key in REQUIRED if key not in arrays]
    if missing:
        raise ValueError(f"missing continuous decision arrays: {missing}")
    n = len(np.asarray(arrays["decision"]))
    features = np.asarray(arrays["features"])
    if features.ndim != 2 or len(features) != n or n == 0:
        raise ValueError("features must be a non-empty [rows, width] array")
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    for key in REQUIRED[1:]:
        if len(np.asarray(arrays[key])) != n:
            raise ValueError(f"array length mismatch: {key}")
    delta = np.asarray(arrays["delta_margin"], np.float64)
    membership = np.asarray(arrays["membership"], np.int8)
    edit = np.asarray(arrays["edit"], np.int16)
    opponent = np.asarray(arrays["opponent"], np.int64)
    seed = np.asarray(arrays["seed"], np.int64)
    if not np.isfinite(delta).all() or not set(map(int, membership)) <= {0, 1}:
        raise ValueError("invalid delta_margin or membership")
    slices = decision_slices(np.asarray(arrays["decision"], np.int64))
    phase_rows = 0
    for indices in slices:
        values = membership[indices]
        phase_start = int(np.argmax(values == PHASE_ONLY)) if np.any(values == PHASE_ONLY) else len(values)
        if (
            np.any(values[:phase_start] != R0)
            or np.any(values[phase_start:] != PHASE_ONLY)
            or phase_start == 0
        ):
            raise ValueError("each decision must be an R0 prefix then phase-only suffix")
        if edit[indices[0]] != KEEP_EDIT or np.count_nonzero(edit[indices] == KEEP_EDIT) != 1:
            raise ValueError("each decision requires exactly one leading R0 KEEP")
        if delta[indices[0]] != 0.0:
            raise ValueError("KEEP delta_margin must be exactly zero")
        if len(set(map(int, opponent[indices]))) != 1 or len(set(map(int, seed[indices]))) != 1:
            raise ValueError("opponent and seed must be constant within a decision")
        phase_rows += len(indices) - phase_start
    return {
        "rows": n,
        "feature_width": int(features.shape[1]),
        "decisions": len(slices),
        "R0_rows": int(np.count_nonzero(membership == R0)),
        "phase_only_rows": int(phase_rows),
        "opponents": sorted(set(map(int, opponent))),
        "seeds": sorted(set(map(int, seed))),
        "R0_exact_prefix_and_unique_KEEP": True,
    }


def _signed_log(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def r0_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    validate_arrays(arrays)
    indices = np.flatnonzero(np.asarray(arrays["membership"], np.int8) == R0)
    delta = np.asarray(arrays["delta_margin"], np.float64)[indices]
    result = {
        "features": np.asarray(arrays["features"], np.float32)[indices],
        "target_signed_log_margin": _signed_log(delta).astype(np.float32),
        "delta_margin": delta,
        "decision": np.asarray(arrays["decision"], np.int64)[indices],
        "opponent": np.asarray(arrays["opponent"], np.int64)[indices],
        "seed": np.asarray(arrays["seed"], np.int64)[indices],
        "edit": np.asarray(arrays["edit"], np.int16)[indices],
    }
    return result, indices


def oracle_novelty_signal(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Label-only stop gate; oracle R0 is diagnostic and never a train target."""

    validate_arrays(arrays)
    membership = np.asarray(arrays["membership"], np.int8)
    delta = np.asarray(arrays["delta_margin"], np.float64)
    opponent = np.asarray(arrays["opponent"], np.int64)
    seed = np.asarray(arrays["seed"], np.int64)
    positive_rows = []
    positive_decisions = []
    for indices in decision_slices(np.asarray(arrays["decision"], np.int64)):
        r0 = indices[membership[indices] == R0]
        phase = indices[membership[indices] == PHASE_ONLY]
        best_r0 = float(np.max(delta[r0]))
        positive = phase[delta[phase] > best_r0]
        positive_rows.extend(map(int, positive))
        if len(positive):
            positive_decisions.append(int(indices[0]))
    positive_rows_array = np.asarray(positive_rows, np.int64)
    opponents = sorted(set(map(int, opponent[positive_rows_array]))) if len(positive_rows_array) else []
    seeds = sorted(set(map(int, seed[positive_rows_array]))) if len(positive_rows_array) else []
    reasons = []
    if len(positive_decisions) < MIN_POSITIVE_DECISIONS:
        reasons.append("positive_novelty_decisions_below_minimum")
    if len(opponents) < MIN_POSITIVE_OPPONENTS:
        reasons.append("positive_novelty_opponent_coverage_below_minimum")
    if len(seeds) < MIN_POSITIVE_SEEDS:
        reasons.append("positive_novelty_seed_coverage_below_minimum")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "positive_rows": len(positive_rows),
        "positive_decisions": len(positive_decisions),
        "positive_opponents": opponents,
        "positive_seeds": seeds,
        "minimum_decisions": MIN_POSITIVE_DECISIONS,
        "minimum_opponents": MIN_POSITIVE_OPPONENTS,
        "minimum_seeds": MIN_POSITIVE_SEEDS,
        "oracle_R0_used_as_model_target": False,
    }


def phase_training_arrays(
    arrays: Mapping[str, np.ndarray],
    a: Mapping[str, np.ndarray],
    a_source_indices: np.ndarray,
    a_lopo_oof_choices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pair every phase row with the deployable LOPO-OOF A choice."""

    validate_arrays(arrays)
    a_slices = decision_slices(np.asarray(a["decision"], np.int64))
    choices = np.asarray(a_lopo_oof_choices, np.int64)
    if len(a_slices) != len(choices):
        raise ValueError("A LOPO choices do not align with all decisions")
    choice_by_decision: dict[int, int] = {}
    for indices, choice in zip(a_slices, choices, strict=True):
        if int(choice) not in set(map(int, indices)):
            raise ValueError("A LOPO choice escaped its own decision")
        decision = int(a["decision"][indices[0]])
        choice_by_decision[decision] = int(choice)
    membership = np.asarray(arrays["membership"], np.int8)
    phase_source = np.flatnonzero(membership == PHASE_ONLY)
    if not len(phase_source):
        raise ValueError("phase-only panel is empty")
    full_features = np.asarray(arrays["features"], np.float32)
    full_delta = np.asarray(arrays["delta_margin"], np.float64)
    decisions = np.asarray(arrays["decision"], np.int64)
    width = full_features.shape[1]
    pair = np.empty((len(phase_source), 2 * width), np.float32)
    deployment = np.empty(len(phase_source), np.float64)
    a_choice_source = np.empty(len(phase_source), np.int64)
    oracle_novelty = np.empty(len(phase_source), np.float64)
    a_by_decision = {
        int(a["decision"][indices[0]]): indices
        for indices in a_slices
    }
    for local, source in enumerate(phase_source):
        decision = int(decisions[source])
        a_choice = choice_by_decision[decision]
        a_source = int(a_source_indices[a_choice])
        phase_values = full_features[source]
        a_values = full_features[a_source]
        pair[local, :width] = phase_values
        pair[local, width:] = phase_values - a_values
        deployment[local] = full_delta[source] - full_delta[a_source]
        a_choice_source[local] = a_source
        oracle_novelty[local] = full_delta[source] - float(np.max(
            a["delta_margin"][a_by_decision[decision]]
        ))
    result = {
        "features": pair,
        "target_signed_log_margin": _signed_log(deployment).astype(np.float32),
        "deployment_uplift": deployment,
        "novelty_uplift": oracle_novelty,
        "decision": decisions[phase_source],
        "opponent": np.asarray(arrays["opponent"], np.int64)[phase_source],
        "seed": np.asarray(arrays["seed"], np.int64)[phase_source],
        "source_row_index": phase_source,
        "a_lopo_oof_choice_source_row_index": a_choice_source,
    }
    if np.any(
        np.asarray(arrays["opponent"])[phase_source]
        != np.asarray(arrays["opponent"])[a_choice_source]
    ):
        raise AssertionError("phase target crossed a left-out opponent boundary")
    return result


def deployment_signal(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    uplift = np.asarray(arrays["deployment_uplift"], np.float64)
    positive = np.flatnonzero(uplift > 0)
    decisions = sorted(set(map(int, np.asarray(arrays["decision"])[positive])))
    opponents = sorted(set(map(int, np.asarray(arrays["opponent"])[positive])))
    seeds = sorted(set(map(int, np.asarray(arrays["seed"])[positive])))
    reasons = []
    if len(decisions) < MIN_POSITIVE_DECISIONS:
        reasons.append("positive_deployment_decisions_below_minimum")
    if len(opponents) < MIN_POSITIVE_OPPONENTS:
        reasons.append("positive_deployment_opponent_coverage_below_minimum")
    if len(seeds) < MIN_POSITIVE_SEEDS:
        reasons.append("positive_deployment_seed_coverage_below_minimum")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "positive_rows": len(positive),
        "positive_decisions": len(decisions),
        "positive_opponents": opponents,
        "positive_seeds": seeds,
        "target": "phase margin minus same-decision LOPO-OOF A choice margin",
        "oracle_A_target_leakage": False,
    }


def phase_override_from_moments(
    mean: np.ndarray,
    std: np.ndarray,
    positive_fraction: np.ndarray,
    calibration: Mapping[str, float],
) -> int:
    if not len(mean):
        return -1
    score = np.asarray(mean) - float(calibration["beta"]) * np.asarray(std)
    local = int(np.argmax(score))
    return local if (
        float(score[local]) > float(calibration["threshold"])
        and float(positive_fraction[local])
        >= float(calibration["min_positive_fraction"])
    ) else -1


def calibrate_phase_oof(
    arrays: Mapping[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    positive_fraction: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    return gate_base.zero_harm_calibration(
        arrays, mean, std, positive_fraction, farmer_only=True,
    )


def _choice_diagnostics(
    arrays: Mapping[str, np.ndarray],
    choices: np.ndarray,
    *,
    phase_only: bool,
    train_seeds: Sequence[int],
) -> dict[str, Any]:
    slices = decision_slices(np.asarray(arrays["decision"], np.int64))
    choices = np.asarray(choices, np.int64)
    if len(slices) != len(choices):
        raise ValueError("OOF choices do not align with decisions")
    fold_by_seed = gate_base.seed_fold_map(train_seeds)
    groups: dict[str, dict[str, dict[str, float]]] = {
        "left_out_opponent": defaultdict(lambda: defaultdict(float)),
        "seed_fold": defaultdict(lambda: defaultdict(float)),
    }
    overall: dict[str, float] = defaultdict(float)
    for indices, choice in zip(slices, choices, strict=True):
        if not phase_only and choice < 0:
            raise ValueError("A OOF choice cannot use the phase fallback sentinel")
        opponent = int(arrays["opponent"][indices[0]])
        seed = int(arrays["seed"][indices[0]])
        fired = int(choice >= 0) if phase_only else int(
            arrays["edit"][choice] != KEEP_EDIT
        )
        value = (
            0.0 if choice < 0 else float(
                arrays["deployment_uplift"][choice]
                if phase_only else arrays["delta_margin"][choice]
            )
        )
        for target in (
            overall,
            groups["left_out_opponent"][str(opponent)],
            groups["seed_fold"][str(fold_by_seed[seed])],
        ):
            target["decisions"] += 1
            target["fires"] += fired
            target["harmful"] += int(value < 0)
            target["beneficial"] += int(value > 0)
            target["realized_uplift_sum"] += value
    return {
        "overall": dict(overall),
        **{
            name: {key: dict(value) for key, value in sorted(cells.items())}
            for name, cells in groups.items()
        },
    }


def fit_two_level(
    arrays: Mapping[str, np.ndarray],
    *,
    trees: int = 96,
    cv_trees: int = 24,
    random_seed: int = 20260829,
) -> dict[str, Any]:
    """Fit A then phase using only complete leave-one-opponent-out predictions."""

    audit = validate_arrays(arrays)
    label_signal = oracle_novelty_signal(arrays)
    membership = np.asarray(arrays["membership"], np.int8)
    phase_opponents = set(map(
        int, np.asarray(arrays["opponent"])[membership == PHASE_ONLY],
    ))
    if phase_opponents != set(audit["opponents"]):
        label_signal = dict(label_signal)
        label_signal["passed"] = False
        label_signal["reasons"] = [
            *label_signal["reasons"], "phase_rows_missing_left_out_opponent",
        ]
    if not label_signal["passed"]:
        return {
            "schema": SCHEMA,
            "status": "data_insufficient_before_fit",
            "input_audit": audit,
            "array_contract": ARRAY_CONTRACT,
            "label_signal": label_signal,
            "a_model": None,
            "phase_model": None,
            "models_fit": 0,
        }

    a, a_source = r0_arrays(arrays)
    a_model, a_calibration, a_oof, a_choices = gate_base.fit_a_backbone(
        a, trees, cv_trees, random_seed,
    )
    if not bool(a_calibration["zero_harm_each_lopo_fold"]):
        raise AssertionError("R0 backbone lost the zero-harm LOPO invariant")
    phase = phase_training_arrays(arrays, a, a_source, a_choices)
    deploy_signal = deployment_signal(phase)
    if not deploy_signal["passed"]:
        return {
            "schema": SCHEMA,
            "status": "data_insufficient_before_phase_fit",
            "input_audit": audit,
            "array_contract": ARRAY_CONTRACT,
            "label_signal": label_signal,
            "deployment_signal": deploy_signal,
            "a_model": a_model,
            "a_calibration": a_calibration,
            "a_oof": a_oof,
            "a_lopo_oof_choices": a_choices,
            "phase_model": None,
            "models_fit": 1,
        }
    train_seeds = tuple(sorted(set(map(int, np.asarray(arrays["seed"])))))
    phase_model, phase_calibration, phase_oof, phase_choices = (
        gate_base.fit_farmer_uplift_gate(
            phase, trees, cv_trees, random_seed + 100_003, train_seeds,
        )
    )
    if not bool(phase_calibration["zero_harm_each_lopo_fold"]):
        raise AssertionError("phase gate lost the zero-harm LOPO invariant")
    a_diagnostics = _choice_diagnostics(
        a, a_choices, phase_only=False, train_seeds=train_seeds,
    )
    phase_diagnostics = _choice_diagnostics(
        phase, phase_choices, phase_only=True, train_seeds=train_seeds,
    )
    if (
        a_diagnostics["overall"]["harmful"] != 0
        or phase_diagnostics["overall"]["harmful"] != 0
    ):
        raise AssertionError("calibrated OOF selector is not zero harm")
    return {
        "schema": SCHEMA,
        "status": "fit_complete_train_only_core",
        "input_audit": audit,
        "array_contract": ARRAY_CONTRACT,
        "label_signal": label_signal,
        "deployment_signal": deploy_signal,
        "models_fit": 2,
        "a_model": a_model,
        "a_calibration": a_calibration,
        "a_oof": a_oof,
        "a_lopo_oof_choices": a_choices,
        "a_source_row_indices": a_source,
        "phase_model": phase_model,
        "phase_calibration": phase_calibration,
        "phase_oof": phase_oof,
        "phase_lopo_oof_choices": phase_choices,
        "phase_training_arrays": phase,
        "OOF_diagnostics": {
            "A_backbone": a_diagnostics,
            "phase_gate": phase_diagnostics,
        },
        "calibration_contract": {
            "A": "LOPO lower-confidence threshold plus positive-tree fraction; KEEP fallback",
            "phase": (
                "LOPO lower-confidence threshold plus positive-tree fraction; "
                "exact A-choice fallback"
            ),
            "threshold_comparison": "strict greater-than; score ties never fire",
            "phase_target": (
                "candidate delta_margin minus same-decision A LOPO-OOF choice delta_margin"
            ),
            "oracle_A_used_for_phase_target": False,
        },
        "data_boundary": {
            "file_IO": False,
            "validation_or_sealed_access": False,
        },
    }


def choose_two_level(
    arrays: Mapping[str, np.ndarray],
    a_model: Any,
    a_calibration: Mapping[str, Any],
    phase_model: Any,
    phase_calibration: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Choose without labels; phase rejection returns the exact A row."""

    audit = validate_arrays(arrays)
    if audit["decisions"] != 1:
        raise ValueError("deployment choice requires exactly one decision panel")
    membership = np.asarray(arrays["membership"], np.int8)
    r0 = np.flatnonzero(membership == R0)
    phase = np.flatnonzero(membership == PHASE_ONLY)
    a_local, a_mean, a_std, a_positive = residual.model_choice(
        a_model,
        np.asarray(arrays["features"], np.float32)[r0],
        a_calibration["chosen"] if "chosen" in a_calibration else a_calibration,
    )
    a_source = int(r0[a_local])
    if not len(phase):
        return a_source, {
            "source": "A", "phase_fired": False,
            "fallback_exact_A": True, "A_local_choice": int(a_local),
        }
    phase_values = np.asarray(arrays["features"], np.float32)[phase]
    a_values = np.asarray(arrays["features"], np.float32)[a_source]
    pair = np.concatenate(
        [phase_values, phase_values - a_values[None, :]], axis=1,
        dtype=np.float32,
    )
    phase_local, p_mean, p_std, p_positive = gate_base.choose_farmer_override(
        phase_model, phase_calibration, pair,
    )
    selected = a_source if phase_local < 0 else int(phase[phase_local])
    return selected, {
        "source": "A" if phase_local < 0 else "phase_only",
        "phase_fired": phase_local >= 0,
        "fallback_exact_A": phase_local < 0 and selected == a_source,
        "A_local_choice": int(a_local),
        "A_mean": a_mean,
        "A_std": a_std,
        "A_positive_fraction": a_positive,
        "phase_mean": p_mean,
        "phase_std": p_std,
        "phase_positive_fraction": p_positive,
    }
