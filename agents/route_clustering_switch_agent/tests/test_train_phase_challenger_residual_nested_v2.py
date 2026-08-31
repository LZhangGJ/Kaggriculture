from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_phase_challenger_residual_nested_v2 as nested


def _panel(*, no_phase_fold: int | None = None, no_phase_opponent: int | None = None):
    values = {key: [] for key in (
        "features", "delta_margin", "membership", "decision", "opponent",
        "seed", "edit",
    )}
    decision = 0
    for opponent in range(6):
        for seed_index in range(8):
            kinds = [0, 1]
            if no_phase_fold != seed_index % 4 and no_phase_opponent != opponent:
                kinds.append(2)
            for kind in kinds:
                values["features"].append([
                    opponent / 10.0, seed_index / 10.0, kind, kind * kind,
                ])
                values["delta_margin"].append((0.0, 2.0, 4.0)[kind])
                values["membership"].append(int(kind == 2))
                values["decision"].append(decision)
                values["opponent"].append(opponent)
                values["seed"].append(2026086300 + seed_index)
                values["edit"].append(0 if kind == 0 else 1)
            decision += 1
    return {
        "features": np.asarray(values["features"], np.float32),
        "delta_margin": np.asarray(values["delta_margin"], np.float64),
        "membership": np.asarray(values["membership"], np.int8),
        "decision": np.asarray(values["decision"], np.int64),
        "opponent": np.asarray(values["opponent"], np.int16),
        "seed": np.asarray(values["seed"], np.int64),
        "edit": np.asarray(values["edit"], np.int8),
    }


def test_v2_nested_complete_without_label_as_feature() -> None:
    result = nested.evaluate(_panel(), trees=2, random_seed=19)
    assert result["status"] == "nested_seed_fold_evaluation_complete"
    assert result["outer_labels_used_for_calibration"] is False
    assert all(row["phase_train_opponent_coverage_complete"] for row in result["folds"])


def test_v2_no_phase_outer_valid_fold_falls_back_to_exact_A() -> None:
    result = nested.evaluate(_panel(no_phase_fold=0), trees=2, random_seed=23)
    assert result["status"] == "nested_seed_fold_evaluation_complete"
    fold = result["folds"][0]
    assert fold["phase_valid_candidate_decisions"] == 0
    assert fold["phase_valid_exact_A_fallback_decisions"] == 12
    assert fold["phase_outer_uplift_metrics"]["selected"] == 0


def test_v2_stops_when_phase_train_lacks_an_opponent() -> None:
    result = nested.evaluate(
        _panel(no_phase_opponent=5), trees=2, random_seed=29,
    )
    assert result["status"] == "outer_train_phase_opponent_coverage_insufficient"
    assert result["missing_phase_train_opponents"] == [5]
    assert result["outer_labels_used_for_calibration"] is False
