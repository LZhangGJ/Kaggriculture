from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TESTS = Path(__file__).resolve().parent
for import_path in (SCRIPTS, TESTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import train_phase_challenger_bilinear_listwise_A_v2 as model
import test_train_phase_challenger_bilinear_listwise_A_v2 as fixture


def test_outer_valid_labels_cannot_change_that_folds_threshold_or_choices() -> None:
    original = fixture._arrays()
    changed = {name: value.copy() for name, value in original.items()}
    heldout = np.isin(changed["seed"], model.SEED_FOLDS[2])
    candidate_one = heldout & (changed["union_index"] == 1)
    changed["outcome"][candidate_one] = np.where(
        changed["outcome"][candidate_one] == 2, 0, 2,
    )
    changed["margin"][candidate_one] += 777.0
    before = model.evaluate(original)
    after = model.evaluate(changed)
    decision_groups = model.decision_slices(original["decision"])
    heldout_positions = [
        index for index, group in enumerate(decision_groups)
        if int(original["seed"][group[0]]) in model.SEED_FOLDS[2]
    ]
    assert before["folds"][2]["calibration"] == after["folds"][2]["calibration"]
    assert before["folds"][2]["primary_model"] == after["folds"][2]["primary_model"]
    assert np.array_equal(
        before["A_choices"][heldout_positions], after["A_choices"][heldout_positions],
    )
    assert before["folds"][2]["outer_metrics"] != after["folds"][2]["outer_metrics"]
    assert before["folds"][2]["outer_labels_used_for_model_or_threshold"] is False
