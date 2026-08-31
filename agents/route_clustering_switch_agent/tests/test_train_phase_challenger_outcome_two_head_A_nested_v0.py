from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_phase_challenger_outcome_two_head_A_nested_v0 as outcome_v0


def test_outcome_arrays_are_relative_to_same_decision_keep() -> None:
    arrays = {
        "features": np.zeros((4, 2), np.float32),
        "delta_margin": np.asarray([0.0, 5.0, 0.0, -2.0]),
        "membership": np.zeros(4, np.int8),
        "decision": np.asarray([0, 0, 1, 1]),
        "opponent": np.asarray([0, 0, 1, 1]),
        "seed": np.asarray([1, 1, 2, 2]),
        "edit": np.asarray([0, 1, 0, 1]),
        "outcome": np.asarray([0, 2, 2, 1]),
        "seat": np.asarray([0, 0, 1, 1]),
    }
    result, source = outcome_v0.outcome_arrays(arrays)
    assert source.tolist() == [0, 1, 2, 3]
    assert result["outcome_delta"].tolist() == [0, 2, 0, -1]
    assert result["target_outcome_upgrade"].tolist() == [0, 1, 0, 0]
    assert result["target_outcome_regression"].tolist() == [0, 0, 0, 1]


def test_selection_prefers_safe_upgrade_and_reports_raw_win_gain() -> None:
    arrays = {
        "decision": np.asarray([0, 0, 0]),
        "edit": np.asarray([0, 1, 1]),
        "outcome": np.asarray([0, 0, 2]),
        "delta_margin": np.asarray([0.0, 20.0, 10.0]),
        "opponent": np.asarray([0, 0, 0]),
        "seed": np.asarray([1, 1, 1]),
        "seat": np.asarray([0, 0, 0]),
    }
    metrics, choices = outcome_v0.selection_metrics(
        arrays,
        regression=np.asarray([0.0, 0.0, 0.1]),
        upgrade=np.asarray([0.0, 0.9, 0.8]),
        params={
            "max_outcome_regression_risk": 0.2,
            "outcome_upgrade_threshold": 0.5,
        },
    )
    assert choices.tolist() == [1]
    assert metrics["raw_win_delta"] == 0
    metrics, choices = outcome_v0.selection_metrics(
        arrays,
        regression=np.asarray([0.0, 0.3, 0.1]),
        upgrade=np.asarray([0.0, 0.9, 0.8]),
        params={
            "max_outcome_regression_risk": 0.2,
            "outcome_upgrade_threshold": 0.5,
        },
    )
    assert choices.tolist() == [2]
    assert metrics["raw_win_delta"] == 1


def test_calibration_keeps_zero_regression_fallback() -> None:
    arrays = {
        "decision": np.asarray([0, 0]),
        "edit": np.asarray([0, 1]),
        "outcome": np.asarray([2, 0]),
        "delta_margin": np.asarray([0.0, -5.0]),
        "opponent": np.asarray([0, 0]),
        "seed": np.asarray([1, 1]),
        "seat": np.asarray([0, 0]),
    }
    chosen, choices, _ = outcome_v0.calibrate(
        arrays,
        regression=np.asarray([0.0, 0.0]),
        upgrade=np.asarray([0.0, 1.0]),
    )
    assert choices.tolist() == [0]
    assert chosen["OOF_metrics"]["outcome_regressions"] == 0
