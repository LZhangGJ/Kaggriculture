from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase_challenger_residual_nested_smoke_v1 as smoke


def test_incremental_diagnostics_uses_final_minus_A_and_exact_fallback() -> None:
    arrays = {
        "delta_margin": np.asarray([0.0, 2.0, 5.0, 0.0, 3.0, 1.0]),
        "membership": np.asarray([0, 0, 1, 0, 0, 1], np.int8),
        "decision": np.asarray([0, 0, 0, 1, 1, 1]),
        "opponent": np.asarray([0, 0, 0, 1, 1, 1]),
        "seed": np.asarray([11, 11, 11, 12, 12, 12]),
    }
    result = smoke.incremental_diagnostics(
        arrays, np.asarray([1, 4]), np.asarray([2, 4]),
    )["overall"]
    assert result == {
        "decisions": 2.0,
        "fires": 1.0,
        "fallback_exact_A": 1.0,
        "harmful": 0.0,
        "beneficial": 1.0,
        "realized_uplift_sum": 3.0,
    }


def test_reportable_excludes_large_choice_arrays() -> None:
    result = smoke.reportable({
        "status": "ok", "A_choices": np.arange(3), "final_choices": np.arange(3),
    })
    assert result == {"status": "ok"}
