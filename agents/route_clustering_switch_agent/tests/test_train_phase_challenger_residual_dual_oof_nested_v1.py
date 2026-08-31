from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
for import_path in (TESTS, SCRIPTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import train_phase_challenger_residual_dual_oof_nested_v1 as dual
from test_train_phase_challenger_residual_nested_v2 import _panel


def test_dual_view_calibration_rejects_conflicting_unsafe_edit() -> None:
    arrays = {
        "decision": np.asarray([0, 0, 1, 1]),
        "edit": np.asarray([0, 1, 0, 1]),
        "delta_margin": np.asarray([0.0, 5.0, 0.0, -5.0]),
        "deployment_uplift": np.asarray([0.0, 5.0, 0.0, -5.0]),
        "opponent": np.asarray([0, 0, 1, 1]),
        "seed": np.asarray([10, 10, 11, 11]),
    }
    opponent = (
        np.asarray([0.0, 1.0, 0.0, 0.0]),
        np.zeros(4), np.ones(4),
    )
    seed = (
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        np.zeros(4), np.ones(4),
    )
    chosen, choices, _ = dual.dual_view_calibration(
        arrays, opponent, seed, farmer_only=False,
    )
    assert chosen["safe_both_views"] is True
    assert chosen["view_metrics"]["opponent_OOF"]["harmful"] == 0
    assert chosen["view_metrics"]["seed_OOF"]["harmful"] == 0
    np.testing.assert_array_equal(choices["opponent_OOF"], [0, 2])
    np.testing.assert_array_equal(choices["seed_OOF"], [0, 2])


def test_dual_oof_nested_pipeline_completes() -> None:
    result = dual.evaluate(_panel(), trees=2, random_seed=47)
    assert result["status"] == "nested_seed_fold_evaluation_complete"
    assert result["models_fit"] == 104
    assert result["full_unseen_opponent_stacked_evidence"] is False
    assert all(row["A_inner_chosen"]["safe_both_views"] for row in result["folds"])
    assert all(row["phase_inner_chosen"]["safe_both_views"] for row in result["folds"])
