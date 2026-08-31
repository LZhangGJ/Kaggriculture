"""Nested outer-seed evaluation for the R0 A -> phase residual pipeline."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for import_path in (Path(__file__).resolve().parent, CODE_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_farmer_augment_union_mpc_v1 as gate
import run_route_residual_adapter_v0 as residual
import train_phase_challenger_residual_gate_v1 as base


SCHEMA = "phase-challenger-residual-nested-v1"


def _fit(arrays: Mapping[str, np.ndarray], trees: int, seed: int) -> Any:
    model = residual.ExtraTreesRegressor(**residual._model_params(trees, seed))
    model.fit(
        arrays["features"], arrays["target_signed_log_margin"],
        sample_weight=gate._decision_weights(np.asarray(arrays["decision"])),
    )
    return model


def _moments(model: Any, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = residual._tree_predictions(model, np.asarray(features, np.float32))
    return values.mean(1), values.std(1), (values > 0.0).mean(1)


def _diagnostics(
    arrays: Mapping[str, np.ndarray], choices: np.ndarray,
    seeds: Sequence[int],
) -> dict[str, Any]:
    slices = base.decision_slices(np.asarray(arrays["decision"], np.int64))
    if len(slices) != len(choices):
        raise ValueError("pipeline choices do not align with decisions")
    fold_by_seed = gate.seed_fold_map(seeds)
    cells: dict[str, dict[str, dict[str, float]]] = {
        "opponent": defaultdict(lambda: defaultdict(float)),
        "seed_fold": defaultdict(lambda: defaultdict(float)),
    }
    overall: dict[str, float] = defaultdict(float)
    for indices, choice in zip(slices, choices, strict=True):
        if not np.any(indices == choice):
            raise ValueError("pipeline choice escaped its decision")
        opponent = int(arrays["opponent"][indices[0]])
        seed = int(arrays["seed"][indices[0]])
        value = float(arrays["delta_margin"][choice])
        for target in (
            overall, cells["opponent"][str(opponent)],
            cells["seed_fold"][str(fold_by_seed[seed])],
        ):
            target["decisions"] += 1
            target["fires"] += int(arrays["edit"][choice] != base.KEEP_EDIT)
            target["phase_fires"] += int(
                arrays["membership"][choice] == base.PHASE_ONLY
            )
            target["harmful"] += int(value < 0)
            target["beneficial"] += int(value > 0)
            target["realized_margin_sum"] += value
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
    """Calibrate A and phase only inside each outer train seed partition."""

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
                "failed_fold": fold, "phase_signal": phase_signal,
                "input_audit": audit, "models_fit": 0,
            }
        pm, pst, pp, p_inner = gate._lopo_predictions(
            phase_train, trees, random_seed + fold * 100_003 + 10_101,
        )
        p_cal, _, _ = gate.zero_harm_calibration(
            phase_train, pm, pst, pp, farmer_only=True,
        )
        a_model = _fit(a_train, trees, random_seed + fold * 100_003 + 20_101)
        p_model = _fit(phase_train, trees, random_seed + fold * 100_003 + 30_101)

        a_valid, a_valid_source = base.r0_arrays(valid)
        vm, vst, vp = _moments(a_model, a_valid["features"])
        a_metrics, a_choices = gate._selection_metrics(
            a_valid, vm, vst, vp, a_cal, farmer_only=False,
        )
        phase_valid = base.phase_training_arrays(
            valid, a_valid, a_valid_source, a_choices,
        )
        qm, qst, qp = _moments(p_model, phase_valid["features"])
        p_metrics, p_choices = gate._selection_metrics(
            phase_valid, qm, qst, qp, p_cal, farmer_only=True,
        )

        local_a = {}
        for indices, choice in zip(
            base.decision_slices(a_valid["decision"]), a_choices, strict=True,
        ):
            local_a[int(a_valid["decision"][indices[0]])] = int(
                a_valid_source[int(choice)]
            )
        local_final = dict(local_a)
        for indices, choice in zip(
            base.decision_slices(phase_valid["decision"]), p_choices, strict=True,
        ):
            if int(choice) >= 0:
                local_final[int(phase_valid["decision"][indices[0]])] = int(
                    phase_valid["source_row_index"][int(choice)]
                )
        for decision, source in local_a.items():
            a_by_decision[decision] = int(valid_rows[source])
            final_by_decision[decision] = int(valid_rows[local_final[decision]])
        folds.append({
            "left_out_seed_fold": fold,
            "train_seeds": train_seeds, "valid_seeds": valid_seeds,
            "calibration_valid_seed_overlap": overlap,
            "inner_A_LOPO_complete": len(a_inner) == len(set(map(int, a_train["opponent"]))),
            "inner_phase_LOPO_complete": len(p_inner) == len(set(map(int, phase_train["opponent"]))),
            "A_inner_chosen": a_cal, "phase_inner_chosen": p_cal,
            "A_outer_metrics": a_metrics,
            "phase_outer_uplift_metrics": p_metrics,
            "phase_deployment_signal": phase_signal,
        })

    slices = base.decision_slices(np.asarray(arrays["decision"], np.int64))
    decision_order = [int(arrays["decision"][row[0]]) for row in slices]
    if set(decision_order) != set(a_by_decision) or set(decision_order) != set(final_by_decision):
        raise RuntimeError("nested choices do not cover every decision")
    a_choices = np.asarray([a_by_decision[key] for key in decision_order], np.int64)
    final_choices = np.asarray([final_by_decision[key] for key in decision_order], np.int64)
    return {
        "schema": SCHEMA,
        "status": "nested_seed_fold_evaluation_complete",
        "scheme": "outer_seed_fold_with_inner_LOPO_A_and_phase_calibration",
        "outer_labels_used_for_calibration": False,
        "folds": folds,
        "A_choices": a_choices, "final_choices": final_choices,
        "A_diagnostics": _diagnostics(arrays, a_choices, all_seeds),
        "A_plus_phase_diagnostics": _diagnostics(arrays, final_choices, all_seeds),
        "input_audit": audit, "models_fit": 8,
    }
