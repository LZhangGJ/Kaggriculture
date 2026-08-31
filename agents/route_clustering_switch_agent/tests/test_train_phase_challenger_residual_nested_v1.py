from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_phase_challenger_residual_nested_v1 as nested


def _panel() -> dict[str, np.ndarray]:
    values = {key: [] for key in (
        "features", "delta_margin", "membership", "decision", "opponent",
        "seed", "edit",
    )}
    decision = 0
    for opponent in range(6):
        for seed_index in range(8):
            for kind, delta in enumerate((0.0, 2.0, 4.0)):
                values["features"].append([opponent, seed_index, kind, delta])
                values["delta_margin"].append(delta)
                values["membership"].append(0 if kind < 2 else 1)
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


def test_nested_two_level_excludes_outer_seed_labels() -> None:
    result = nested.evaluate(_panel(), trees=2, random_seed=19)
    assert result["status"] == "nested_seed_fold_evaluation_complete"
    assert result["outer_labels_used_for_calibration"] is False
    assert len(result["folds"]) == 4
    assert all(row["calibration_valid_seed_overlap"] == [] for row in result["folds"])
    assert all(row["inner_A_LOPO_complete"] for row in result["folds"])
    assert all(row["inner_phase_LOPO_complete"] for row in result["folds"])
    assert len(result["final_choices"]) == 48

