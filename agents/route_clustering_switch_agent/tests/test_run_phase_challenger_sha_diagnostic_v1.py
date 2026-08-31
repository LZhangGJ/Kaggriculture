from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase_challenger_sha_diagnostic_v1 as diagnostic


def test_sign_group_summary_counts_mixed_groups() -> None:
    result = diagnostic.sign_group_summary(
        [b"a", b"a", b"b", b"b", b"c"],
        np.asarray([2.0, -1.0, 3.0, 0.0, -4.0]),
    )
    assert result == {
        "unique_groups": 3,
        "positive_groups": 2,
        "negative_groups": 2,
        "mixed_positive_negative_groups": 1,
        "rows_in_mixed_groups": 2,
        "maximum_group_rows": 2,
    }


def test_sha_fold_is_stable() -> None:
    value = "01234567" + "0" * 56
    assert diagnostic._fold(value) == int("01234567", 16) % 4
