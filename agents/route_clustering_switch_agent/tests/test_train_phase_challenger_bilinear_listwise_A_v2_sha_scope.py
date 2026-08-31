from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TESTS = Path(__file__).resolve().parent
for import_path in (SCRIPTS, TESTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import train_phase_challenger_bilinear_listwise_A_v2 as model
import test_train_phase_challenger_bilinear_listwise_A_v2 as fixture


def test_absolute_sha_may_have_different_relative_action_across_states() -> None:
    arrays = fixture._arrays()
    groups = model.decision_slices(arrays["decision"])
    # The absolute four-step SHA stays the same.  The relative feature may
    # differ because each state has a different KEEP plan.
    arrays["action"][groups[1][1], 0] += 0.25
    audit = model.validate_arrays(arrays)
    assert audit["candidate_sha_unique"] == 19

