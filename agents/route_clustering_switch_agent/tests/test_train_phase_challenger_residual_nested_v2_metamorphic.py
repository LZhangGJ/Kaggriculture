from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
for import_path in (TESTS, SCRIPTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import train_phase_challenger_residual_nested_v2 as nested
from test_train_phase_challenger_residual_nested_v2 import _panel


def _decision_seed(panel):
    slices = nested.base.decision_slices(panel["decision"])
    return np.asarray([panel["seed"][rows[0]] for rows in slices])


def test_outer_fold_choices_ignore_heldout_label_mutation() -> None:
    original = _panel()
    mutated = {key: value.copy() for key, value in original.items()}
    heldout = np.isin(mutated["seed"], [2026086300, 2026086304])
    r0_edit = heldout & (mutated["membership"] == 0) & (mutated["edit"] != 0)
    phase = heldout & (mutated["membership"] == 1)
    mutated["delta_margin"][r0_edit] -= 0.25
    mutated["delta_margin"][phase] += 0.25

    before = nested.evaluate(original, trees=2, random_seed=31)
    after = nested.evaluate(mutated, trees=2, random_seed=31)
    assert before["folds"][0]["A_inner_chosen"] == after["folds"][0]["A_inner_chosen"]
    assert before["folds"][0]["phase_inner_chosen"] == after["folds"][0]["phase_inner_chosen"]
    fold0_decisions = np.isin(_decision_seed(original), [2026086300, 2026086304])
    np.testing.assert_array_equal(
        before["A_choices"][fold0_decisions], after["A_choices"][fold0_decisions],
    )
    np.testing.assert_array_equal(
        before["final_choices"][fold0_decisions],
        after["final_choices"][fold0_decisions],
    )


def test_noncontiguous_decisions_keep_exact_A_when_phase_absent() -> None:
    panel = _panel(no_phase_fold=0)
    panel["decision"] = panel["decision"] * 7 + 101
    result = nested.evaluate(panel, trees=2, random_seed=37)
    assert result["status"] == "nested_seed_fold_evaluation_complete"
    for a_choice, final_choice in zip(
        result["A_choices"], result["final_choices"], strict=True,
    ):
        if panel["membership"][final_choice] == 0:
            assert final_choice == a_choice
