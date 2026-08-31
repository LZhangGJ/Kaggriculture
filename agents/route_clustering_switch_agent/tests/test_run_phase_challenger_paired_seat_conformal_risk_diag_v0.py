from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase_challenger_paired_seat_conformal_risk_diag_v0 as diagnostic


def test_cap_is_strictly_below_all_calibration_nonpositive_scores() -> None:
    risk = np.asarray([0.2, 0.4, 0.1, 0.9])
    nonpositive = np.asarray([True, True, False, False])
    cap = diagnostic.zero_false_negative_cap(risk, nonpositive)
    assert cap < 0.2
    assert not np.any(nonpositive & (risk <= cap))


def test_cap_requires_a_nonpositive_calibration_example() -> None:
    with pytest.raises(ValueError, match="nonpositive"):
        diagnostic.zero_false_negative_cap(
            np.asarray([0.1]), np.asarray([False]),
        )


def test_eligibility_reports_candidate_level_failures() -> None:
    arrays = {
        "edit": np.asarray([0, 1, 1]),
        "target_nonpositive": np.asarray([1, 1, 0]),
        "delta_margin": np.asarray([0.0, -2.0, 3.0]),
        "decision": np.asarray([0, 0, 0]),
    }
    result = diagnostic.candidate_eligibility(
        arrays,
        utility=np.asarray([0.0, 2.0, 1.0]),
        risk=np.asarray([1.0, 0.1, 0.05]),
        cap=0.08,
    )
    assert result["eligible_rows"] == 1
    assert result["eligible_robust_nonpositive_rows"] == 0
    assert result["eligible_actual_positive_rows"] == 1
