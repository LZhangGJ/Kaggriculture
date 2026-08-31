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


def test_rare_signal_row_sparsity_and_fold_shape_match_formal_order() -> None:
    arrays = fixture._arrays()
    gain, _ = model._relative_labels(arrays, np.arange(len(arrays["decision"])))
    candidate = arrays["edit"] != model.KEEP_EDIT
    # 8/2304 = 0.347%; frozen formal is 93/27552 = 0.338%.
    assert int(gain[candidate].sum()) == 8
    assert int(candidate.sum()) == 2304
    assert abs(float(gain[candidate].mean()) - 93.0 / 27_552.0) < 0.0001
    row_fold = np.asarray([model.seed_fold_map()[int(seed)] for seed in arrays["seed"]])
    assert int(gain[row_fold == 0].sum()) == 0
    assert all(int(gain[row_fold == fold].sum()) > 0 for fold in (1, 2, 3))


def test_one_upgrade_one_regression_positive_net_is_forced_to_keep() -> None:
    # Both candidates have the same observable score: a threshold either fires
    # both (+1 raw win net, but one regression) or fires neither.
    arrays = {
        "decision": np.asarray([0, 0, 1, 1]),
        "edit": np.asarray([0, 1, 0, 1]),
        "union_index": np.asarray([0, 1, 0, 1]),
        "outcome": np.asarray([0, 2, 1, 0]),
        "margin": np.zeros(4),
        "opponent": np.asarray([0, 0, 1, 1]),
        "seed": np.asarray([2026086301, 2026086301, 2026086302, 2026086302]),
    }
    rows = np.arange(4)
    score = np.asarray([0.0, 0.8, 0.0, 0.8])
    risk = np.zeros(4)
    unsafe = model.choice_metrics(
        arrays, rows, model.select_choices(arrays, rows, score, risk, 0.7),
    )
    assert unsafe["raw_win_delta"] == 1
    assert unsafe["outcome_delta_sum"] == 1
    assert unsafe["outcome_regressions"] == 1
    floor, calibration = model.calibrate_threshold(arrays, rows, score, risk)
    choices = model.select_choices(arrays, rows, score, risk, floor)
    safe = model.choice_metrics(arrays, rows, choices)
    assert np.isinf(floor)
    assert calibration["no_credible_ordinal_gain_forces_KEEP"] is True
    assert choices.tolist() == [0, 2]
    assert safe["fires"] == 0
    assert safe["raw_win_delta"] == 0
    assert safe["outcome_regressions"] == 0
