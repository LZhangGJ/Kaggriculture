from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase_challenger_direct_nested_smoke_v1 as smoke


def test_result_summary_drops_models_and_oof_arrays() -> None:
    result = {
        "status": "fit_complete_train_only_core",
        "input_audit": {"rows": 4},
        "signal": {"passed": True},
        "calibration": {"chosen": {"threshold": 1.0}, "used_for_nested_outer_evaluation": False},
        "model": object(),
        "oof": {"mean": np.ones(4)},
        "seed_fold_fixed_threshold": {
            "scheme": "outer_leave_seed_fold_out_with_inner_LOPO_calibration",
            "global_calibration_used_for_outer": False,
            "outer_labels_used_to_select_threshold": False,
            "complete_four_folds": True,
            "overall": {}, "by_seed_fold": {}, "by_opponent": {},
            "opponents_with_fire": [], "seeds_with_fire": [],
            "zero_harm_each_seed_fold": True, "zero_harm_each_opponent": True,
            "zero_harm_fixed_threshold": True,
            "folds": [{
                "left_out_seed_fold": 0, "train_seeds": [1, 2, 3],
                "valid_seeds": [0], "threshold_selection_valid_seed_overlap": [],
                "inner_LOPO_complete": True, "inner_chosen": {},
                "outer_heldout": {},
            }],
        },
    }
    summary = smoke.result_summary(result)
    assert "model" not in summary
    assert "oof" not in summary
    assert summary["nested_fold_audit"][0]["threshold_selection_valid_seed_overlap"] == []


def test_representation_names_and_default_step_are_frozen() -> None:
    assert smoke.REPRESENTATIONS == {
        "full2844": "full_2844.npy", "compact1078": "compact_1078.npy",
    }
    args = smoke.parser().parse_args([])
    assert args.decision_step == 217
    assert (args.trees, args.cv_trees) == (8, 2)

