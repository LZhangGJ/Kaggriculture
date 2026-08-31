from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
for import_path in (TESTS, SCRIPTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import train_phase_challenger_residual_dual_oof_exact_v1 as exact
from test_train_phase_challenger_residual_nested_v2 import _panel


def test_event_thresholds_use_decision_maxima_and_fallback() -> None:
    arrays = {
        "decision": np.asarray([0, 0, 0, 1, 1]),
        "edit": np.asarray([0, 1, 1, 0, 1]),
    }
    scores = np.asarray([0.0, 0.2, 0.7, 0.0, 0.4])
    assert exact.event_thresholds(arrays, scores, farmer_only=False) == [
        0.0, 0.4, 0.7, 1.7,
    ]


def test_exact_dual_nested_pipeline_completes() -> None:
    result = exact.evaluate(_panel(), trees=2, random_seed=53)
    assert result["status"] == "nested_seed_fold_evaluation_complete"
    assert result["calibration_ablation"] == "exact_decision_event_threshold_grid"
    assert all(
        row["A_inner_chosen"]["threshold_grid"]
        == "exact_unique_decision_event_scores"
        for row in result["folds"]
    )
