from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
for import_path in (TESTS, SCRIPTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import train_phase_challenger_residual_seed_oof_nested_v1 as seed_nested
from test_train_phase_challenger_residual_nested_v2 import _panel


def test_pair_feature_builder_does_not_require_outcome_labels() -> None:
    panel = _panel()
    a, source = seed_nested.base.r0_arrays(panel)
    choices = np.asarray([
        rows[0] for rows in seed_nested.base.decision_slices(a["decision"])
    ])
    label_free = {
        key: panel[key]
        for key in ("features", "membership", "decision", "opponent", "seed")
    }
    pair = seed_nested.phase_pair_feature_arrays(label_free, a, source, choices)
    assert pair["features"].shape == (48, 8)
    assert "delta_margin" not in pair
    assert "target_signed_log_margin" not in pair


def test_seed_oof_nested_pipeline_completes() -> None:
    result = seed_nested.evaluate(_panel(), trees=2, random_seed=41)
    assert result["status"] == "nested_seed_fold_evaluation_complete"
    assert result["outer_labels_used_for_calibration"] is False
    assert all(row["inner_A_seed_OOF_complete"] for row in result["folds"])
    assert all(row["inner_phase_seed_OOF_complete"] for row in result["folds"])
    assert result["inference_feature_boundary"] == {
        "pair_features_built_without_delta_margin": True,
        "outer_outcomes_attached_after_prediction": True,
    }


def test_seed_oof_outer_fold_without_phase_returns_exact_A() -> None:
    result = seed_nested.evaluate(
        _panel(no_phase_fold=0), trees=2, random_seed=43,
    )
    assert result["status"] == "nested_seed_fold_evaluation_complete"
    fold = result["folds"][0]
    assert fold["phase_valid_candidate_decisions"] == 0
    assert fold["phase_valid_exact_A_fallback_decisions"] == 12
