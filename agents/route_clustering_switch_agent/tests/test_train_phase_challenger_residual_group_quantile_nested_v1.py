from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
for import_path in (TESTS, SCRIPTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import train_phase_challenger_residual_group_quantile_nested_v1 as quantile
from test_train_phase_challenger_residual_nested_v2 import _panel


def test_lower_quantile_uses_discrete_lower_method() -> None:
    values = np.asarray([[1.0, 2.0, 3.0, 100.0], [-5.0, 0.0, 4.0, 8.0]])
    np.testing.assert_array_equal(
        quantile.lower_quantile(values, 0.25), [1.0, -5.0],
    )


def test_group_oof_models_exclude_group_and_decisions() -> None:
    panel = _panel()
    a, _ = quantile.base.r0_arrays(panel)
    predictions, audit = quantile.group_oof_tree_predictions(
        a, "seed", trees=2, random_seed=59,
    )
    assert predictions.shape == (96, 2)
    assert all(row["train_valid_group_overlap"] == [] for row in audit)
    assert all(row["train_valid_decision_overlap"] == [] for row in audit)


def test_group_quantile_nested_pipeline_completes() -> None:
    result = quantile.evaluate(_panel(), trees=2, random_seed=61)
    assert result["status"] == "nested_seed_fold_evaluation_complete"
    assert result["phase_A_reference_cross_calibrated_by_seed_fold"] is True
    assert result["risk_contract"]["OOF_tree_values_per_row"] == 4
    assert result["risk_contract"]["outer_tree_values_per_row"] == 4
    assert result["models_fit"] == 104
    assert all(
        audit["calibration_held_seed_overlap"] == []
        for fold in result["folds"]
        for audit in fold["A_cross_calibration"]
    )
