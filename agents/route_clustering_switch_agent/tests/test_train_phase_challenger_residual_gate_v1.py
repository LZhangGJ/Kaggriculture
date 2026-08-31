from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_phase_challenger_residual_gate_v1 as core


def _panel(phase_delta: float = 3.0) -> dict[str, np.ndarray]:
    decisions = []
    opponent = []
    seed = []
    membership = []
    edit = []
    delta = []
    features = []
    for index in range(4):
        for kind, value in enumerate((0.0, 2.0, phase_delta)):
            decisions.append(100 + index)
            opponent.append(index // 2)
            seed.append(10 + index % 2)
            membership.append(0 if kind < 2 else 1)
            edit.append(0 if kind == 0 else 1)
            delta.append(value)
            features.append([float(index), float(kind), value])
    return {
        "features": np.asarray(features, np.float32),
        "delta_margin": np.asarray(delta, np.float64),
        "membership": np.asarray(membership, np.int8),
        "decision": np.asarray(decisions, np.int64),
        "opponent": np.asarray(opponent, np.int16),
        "seed": np.asarray(seed, np.int64),
        "edit": np.asarray(edit, np.int8),
    }


def test_r0_must_be_prefix_with_one_leading_keep() -> None:
    panel = _panel()
    assert core.validate_arrays(panel)["R0_exact_prefix_and_unique_KEEP"] is True
    duplicate = {key: value.copy() for key, value in panel.items()}
    duplicate["edit"][1] = 0
    with pytest.raises(ValueError, match="exactly one leading R0 KEEP"):
        core.validate_arrays(duplicate)
    reordered = {key: value.copy() for key, value in panel.items()}
    reordered["membership"][:3] = [1, 0, 0]
    with pytest.raises(ValueError, match="R0 prefix"):
        core.validate_arrays(reordered)


def test_phase_target_uses_supplied_lopo_a_choice_not_oracle_a() -> None:
    panel = _panel()
    a, source = core.r0_arrays(panel)
    # Deliberately choose KEEP (delta 0) although each oracle R0 edit has delta 2.
    oof_keep_choices = np.asarray([0, 2, 4, 6], np.int64)
    phase = core.phase_training_arrays(panel, a, source, oof_keep_choices)
    assert np.all(phase["deployment_uplift"] == 3.0)
    assert np.all(phase["novelty_uplift"] == 1.0)
    assert np.all(
        panel["delta_margin"][phase["a_lopo_oof_choice_source_row_index"]] == 0.0
    )


def test_negative_phase_panel_calibrates_to_exact_a_fallback_zero_harm() -> None:
    arrays = {
        "decision": np.arange(4, dtype=np.int64),
        "opponent": np.asarray([0, 0, 1, 1], np.int16),
        "seed": np.asarray([10, 11, 10, 11], np.int64),
        "deployment_uplift": -np.ones(4, np.float64),
    }
    chosen, _, rows = core.calibrate_phase_oof(
        arrays,
        mean=np.ones(4, np.float64),
        std=np.zeros(4, np.float64),
        positive_fraction=np.ones(4, np.float64),
    )
    assert chosen["harmful"] == 0
    assert chosen["selected"] == 0
    assert np.all(rows == -1)
    assert not any(chosen["opponent_harm"].values())


def test_phase_score_equal_to_threshold_does_not_fire() -> None:
    calibration = {
        "beta": 0.0, "threshold": 0.5, "min_positive_fraction": 0.6,
    }
    assert core.phase_override_from_moments(
        np.asarray([0.5]), np.asarray([0.0]), np.asarray([1.0]), calibration,
    ) == -1
    assert core.phase_override_from_moments(
        np.asarray([0.5001]), np.asarray([0.0]), np.asarray([1.0]), calibration,
    ) == 0


def test_no_positive_signal_stops_before_any_model_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _panel(phase_delta=1.0)
    monkeypatch.setattr(
        core.gate_base,
        "fit_a_backbone",
        lambda *args, **kwargs: pytest.fail("A fit must not run"),
    )
    result = core.fit_two_level(panel, trees=1, cv_trees=1)
    assert result["status"] == "data_insufficient_before_fit"
    assert result["models_fit"] == 0


def test_fit_core_slices_r0_and_passes_oof_a_deployment_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()

    def fake_a(arrays, trees, cv_trees, random_seed):
        assert len(arrays["features"]) == 8
        assert np.all(arrays["edit"][::2] == 0)
        choices = np.asarray([0, 2, 4, 6], np.int64)
        return object(), {
            "zero_harm_each_lopo_fold": True,
            "chosen": {"beta": 0.0, "threshold": 1.0, "min_positive_fraction": 1.0},
        }, {"mean": np.zeros(8)}, choices

    def fake_phase(arrays, trees, cv_trees, random_seed, train_seeds):
        assert np.all(arrays["deployment_uplift"] == 3.0)
        assert np.all(arrays["novelty_uplift"] == 1.0)
        assert sorted(set(map(int, arrays["opponent"]))) == [0, 1]
        choices = -np.ones(4, np.int64)
        return object(), {
            "zero_harm_each_lopo_fold": True,
            "chosen": {"beta": 0.0, "threshold": 99.0, "min_positive_fraction": 1.0},
        }, {"mean": np.zeros(4)}, choices

    monkeypatch.setattr(core.gate_base, "fit_a_backbone", fake_a)
    monkeypatch.setattr(core.gate_base, "fit_farmer_uplift_gate", fake_phase)
    result = core.fit_two_level(panel, trees=1, cv_trees=1)
    assert result["status"] == "fit_complete_train_only_core"
    assert result["models_fit"] == 2
    assert result["calibration_contract"]["oracle_A_used_for_phase_target"] is False
    assert result["OOF_diagnostics"]["phase_gate"]["overall"]["harmful"] == 0
    assert set(result["OOF_diagnostics"]["phase_gate"]["left_out_opponent"]) == {"0", "1"}
    assert set(result["OOF_diagnostics"]["phase_gate"]["seed_fold"]) == {"0", "1"}


class _ZeroTree:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.zeros(len(values), np.float64)


class _ZeroModel:
    estimators_ = [_ZeroTree(), _ZeroTree()]


def test_label_free_deployment_returns_exact_keep_a_on_phase_tie() -> None:
    panel = {key: value[:3].copy() for key, value in _panel().items()}
    calibration = {
        "chosen": {
            "beta": 0.0, "threshold": 0.0, "min_positive_fraction": 0.5,
        }
    }
    selected, audit = core.choose_two_level(
        panel, _ZeroModel(), calibration, _ZeroModel(), calibration,
    )
    assert selected == 0
    assert audit["source"] == "A"
    assert audit["phase_fired"] is False
    assert audit["fallback_exact_A"] is True


def test_real_reused_lopo_helpers_fit_tiny_synthetic_panel() -> None:
    result = core.fit_two_level(_panel(), trees=2, cv_trees=2, random_seed=7)
    assert result["status"] == "fit_complete_train_only_core"
    assert result["a_calibration"]["zero_harm_each_lopo_fold"] is True
    assert result["phase_calibration"]["zero_harm_each_lopo_fold"] is True
    assert len(result["a_calibration"]["folds"]) == 2
    assert len(result["phase_calibration"]["folds"]) == 2
