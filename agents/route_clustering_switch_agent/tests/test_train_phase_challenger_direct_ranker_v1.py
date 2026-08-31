from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_phase_challenger_direct_ranker_v1 as core


def _panel(
    *, decisions: int = 8, phase_delta: float = 3.0,
) -> dict[str, np.ndarray]:
    rows = {
        "features": [], "delta_margin": [], "membership": [], "decision": [],
        "opponent": [], "seed": [], "edit": [],
    }
    for decision in range(decisions):
        values = (0.0, 2.0, phase_delta)
        for local, delta in enumerate(values):
            rows["features"].append([float(decision), float(local), float(delta)])
            rows["delta_margin"].append(delta)
            rows["membership"].append(0 if local < 2 else 1)
            rows["decision"].append(100 + decision)
            rows["opponent"].append(decision % 2)
            rows["seed"].append(10 + decision % 4)
            rows["edit"].append(0 if local == 0 else 1)
    return {
        "features": np.asarray(rows["features"], np.float32),
        "delta_margin": np.asarray(rows["delta_margin"], np.float64),
        "membership": np.asarray(rows["membership"], np.int8),
        "decision": np.asarray(rows["decision"], np.int64),
        "opponent": np.asarray(rows["opponent"], np.int16),
        "seed": np.asarray(rows["seed"], np.int64),
        "edit": np.asarray(rows["edit"], np.int8),
    }


class _Tree:
    def __init__(self, values: list[float]) -> None:
        self.values = np.asarray(values, np.float64)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.values[:len(features)]


class _Model:
    def __init__(self, *trees: list[float]) -> None:
        self.estimators_ = [_Tree(values) for values in trees]


def _single_panel() -> dict[str, np.ndarray]:
    return {key: value[:3].copy() for key, value in _panel(decisions=1).items()}


def test_validate_reuses_canonical_r0_prefix_and_marks_keep_boundary() -> None:
    audit = core.validate(_panel())
    assert audit["R0_exact_prefix_and_unique_KEEP"] is True
    assert audit["zero_harm_reference"] == "KEEP"
    assert audit["architecture_role"] == "direct_union_representation_ablation_only"
    broken = {key: value.copy() for key, value in _panel().items()}
    broken["membership"][:3] = [0, 1, 0]
    with pytest.raises(ValueError, match="R0 prefix"):
        core.validate(broken)


def test_canonical_tie_prefers_earlier_r0_non_keep() -> None:
    panel = _single_panel()
    model = _Model([0.0, 1.0, 1.0], [0.0, 1.0, 1.0])
    selected, audit = core.choose_direct(panel, model, {
        "beta": 0.0, "threshold": 0.0, "min_positive_fraction": 1.0,
    })
    assert selected == 1
    assert audit["source"] == "R0"
    assert audit["canonical_tie_order_preserved"] is True


def test_threshold_is_strict_and_positive_fraction_gate_is_enforced() -> None:
    panel = _single_panel()
    tied, _ = core.choose_direct(
        panel, _Model([0.0, 0.5, 0.4], [0.0, 0.5, 0.4]),
        {"beta": 0.0, "threshold": 0.5, "min_positive_fraction": 1.0},
    )
    gated, _ = core.choose_direct(
        panel, _Model([0.0, 1.0, 0.4], [0.0, -0.2, 0.4]),
        {"beta": 0.0, "threshold": 0.0, "min_positive_fraction": 0.75},
    )
    assert tied == 0
    assert gated == 0


def test_no_signal_stops_before_lopo_or_final_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _panel(phase_delta=-1.0)
    panel["delta_margin"][1::3] = -1.0
    monkeypatch.setattr(
        core.gate_base, "_lopo_predictions",
        lambda *args, **kwargs: pytest.fail("LOPO fit must not run"),
    )
    monkeypatch.setattr(
        core.residual, "ExtraTreesRegressor",
        lambda *args, **kwargs: pytest.fail("final fit must not run"),
    )
    result = core.fit_direct_ranker(panel, trees=1, cv_trees=1)
    assert result["status"] == "data_insufficient_before_fit"
    assert result["models_fit"] == 0
    assert result["model"] is None


def test_real_lopo_has_no_opponent_group_leak_and_four_seed_cells() -> None:
    result = core.fit_direct_ranker(_panel(), trees=2, cv_trees=2, random_seed=7)
    assert result["status"] == "fit_complete_train_only_core"
    assert result["calibration"]["keep_only_fallback_present"] is True
    assert result["calibration"]["zero_harm_each_lopo_fold"] is True
    assert result["calibration"]["decision_weight_sum_min"] == 1.0
    assert result["calibration"]["decision_weight_sum_max"] == 1.0
    for fold in result["calibration"]["folds"]:
        assert fold["left_out_opponent_index"] not in fold["train_opponent_indices"]
        assert fold["valid_opponent_indices"] == [fold["left_out_opponent_index"]]
        assert fold["group_leakage"] is False
    assert np.array_equal(
        result["oof"]["left_out_opponent"], _panel()["opponent"],
    )
    assert set(result["OOF_diagnostics"]["seed_fold"]) == {"0", "1", "2", "3"}
    fixed = result["seed_fold_fixed_threshold"]
    assert fixed["threshold_reselected_on_outer"] is False
    assert fixed["global_calibration_used_for_outer"] is False
    assert fixed["outer_labels_used_to_select_threshold"] is False
    assert fixed["complete_four_folds"] is True
    assert len(fixed["folds"]) == 4
    assert all(
        not (set(fold["train_seeds"]) & set(fold["valid_seeds"]))
        and not (set(fold["calibration_seeds"]) & set(fold["valid_seeds"]))
        and fold["seed_group_leakage"] is False
        and fold["threshold_selection_rows_exclude_valid_seeds"] is True
        and fold["threshold_selection_valid_seed_overlap"] == []
        and fold["inner_LOPO_complete"] is True
        and {
            inner["left_out_opponent_index"]
            for inner in fold["inner_LOPO_folds"]
        } == set(fold["train_opponents"])
        and all(
            inner["left_out_opponent_index"] not in inner["train_opponent_indices"]
            and inner["valid_opponent_indices"] == [
                inner["left_out_opponent_index"]
            ]
            and inner["group_leakage"] is False
            for inner in fold["inner_LOPO_folds"]
        )
        for fold in fixed["folds"]
    )
    assert result["data_boundary"]["not_a_residual_gate_replacement"] is True


def test_nested_outer_seed_fold_catches_harm_hidden_by_global_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    harmful = (
        (panel["seed"] == 10)
        & (panel["membership"] == 0)
        & (panel["edit"] != 0)
    )
    panel["delta_margin"][harmful] = -5.0
    panel["features"][:, 2] = panel["delta_margin"]
    calls = 0

    def fake_lopo(arrays, trees, random_seed):
        size = len(arrays["decision"])
        folds = [{
            "left_out_opponent_index": opponent,
            "train_rows": int(np.count_nonzero(arrays["opponent"] != opponent)),
            "valid_rows": int(np.count_nonzero(arrays["opponent"] == opponent)),
        } for opponent in sorted(set(map(int, arrays["opponent"])))]
        return (
            np.zeros(size), np.zeros(size), np.ones(size), folds,
        )

    def fake_calibration(arrays, mean, std, positive, *, farmer_only):
        nonlocal calls
        calls += 1
        fallback = {
            "beta": 0.0, "threshold": 99.0, "min_positive_fraction": 1.0,
            "selected": 0, "harmful": 0, "sum_realized_delta": 0.0,
        }
        chosen = fallback if calls == 1 else {
            **fallback, "threshold": 0.1, "selected": 1,
        }
        keep = np.asarray([
            int(indices[0]) for indices in core.residual_contract.decision_slices(
                np.asarray(arrays["decision"], np.int64),
            )
        ], np.int64)
        return chosen, [fallback], keep

    class FakeExtraTrees:
        def __init__(self, **kwargs) -> None:
            self.estimators_ = []

        def fit(self, features, target, sample_weight):
            return self

    def fake_tree_predictions(model, features):
        local = np.asarray(features)[:, 1]
        prediction = np.where(local == 1.0, 1.0, np.where(local == 2.0, 0.5, 0.0))
        return np.repeat(prediction[:, None], 2, axis=1)

    monkeypatch.setattr(core.gate_base, "_lopo_predictions", fake_lopo)
    monkeypatch.setattr(core.gate_base, "zero_harm_calibration", fake_calibration)
    monkeypatch.setattr(core.residual, "ExtraTreesRegressor", FakeExtraTrees)
    monkeypatch.setattr(core.residual, "_tree_predictions", fake_tree_predictions)
    result = core.fit_direct_ranker(panel, trees=2, cv_trees=2)
    assert result["calibration"]["zero_harm_each_lopo_fold"] is True
    assert result["OOF_diagnostics"]["overall"]["harmful"] == 0
    nested = result["seed_fold_fixed_threshold"]
    assert nested["global_calibration_used_for_outer"] is False
    assert nested["zero_harm_fixed_threshold"] is False
    assert nested["by_seed_fold"]["0"]["harmful"] == 2
    assert all(
        fold["threshold_selection_valid_seed_overlap"] == []
        for fold in nested["folds"]
    )


def _result(
    total: float, opponents: dict[str, tuple[float, int]],
    *, fires: int = 4, fire_opponents: tuple[int, ...] = (0, 1),
    fire_seeds: tuple[int, ...] = (10, 11), fixed_harm: bool = False,
) -> dict:
    cells = {
        key: {"realized_sum": value, "harmful": harmful}
        for key, (value, harmful) in opponents.items()
    }
    fixed_cells = {
        key: {
            "realized_sum": value,
            "harmful": int(fixed_harm and index == 0),
        }
        for index, (key, (value, _)) in enumerate(opponents.items())
    }
    fixed_folds = {
        str(fold): {
            "realized_sum": total / 4.0,
            "harmful": int(fixed_harm and fold == 0),
        }
        for fold in range(4)
    }
    return {
        "status": "fit_complete_train_only_core",
        "OOF_diagnostics": {
            "overall": {"fires": fires, "realized_sum": total},
            "left_out_opponent": cells,
            "opponents_with_fire": list(fire_opponents),
            "seeds_with_fire": list(fire_seeds),
        },
        "seed_fold_fixed_threshold": {
            "complete_four_folds": True,
            "global_calibration_used_for_outer": False,
            "outer_labels_used_to_select_threshold": False,
            "overall": {"fires": fires, "realized_sum": total},
            "by_seed_fold": fixed_folds,
            "by_opponent": fixed_cells,
            "opponents_with_fire": list(fire_opponents),
            "seeds_with_fire": list(fire_seeds),
            "zero_harm_fixed_threshold": not fixed_harm,
        },
    }


def test_compare_representations_applies_all_pure_metric_gates() -> None:
    full = _result(10.0, {"0": (4.0, 0), "1": (6.0, 0)})
    compact = _result(10.2, {"0": (4.1, 0), "1": (6.1, 0)})
    passed = core.compare_representations(full, compact)
    assert passed["passed"] is True
    assert "prevalidated_by_caller" in passed["candidate_sha_order"]

    one_regression = _result(10.2, {"0": (3.9, 0), "1": (6.3, 0)})
    failed = core.compare_representations(full, one_regression)
    assert failed["passed"] is False
    assert failed["checks"]["compact_realized_sum_at_least_98pct_full"] is True
    assert failed["checks"]["compact_minus_full_nonnegative_each_opponent"] is False


def test_compare_representations_rejects_harm_or_weak_fire_coverage() -> None:
    full = _result(10.0, {"0": (4.0, 0), "1": (6.0, 0)})
    compact = _result(
        10.0, {"0": (4.0, 1), "1": (6.0, 0)},
        fires=3, fire_opponents=(0,), fire_seeds=(10,), fixed_harm=True,
    )
    checks = core.compare_representations(full, compact)["checks"]
    assert checks[
        "compact_fixed_threshold_zero_harm_each_seed_fold_and_opponent"
    ] is False
    assert checks["compact_fires_at_least_4"] is False
    assert checks["compact_fires_cover_at_least_2_opponents"] is False
    assert checks["compact_fires_cover_at_least_2_seeds"] is False
    diagnostic = core.compare_representations(
        full, compact,
    )["calibration_diagnostics_not_used_by_gate"]
    assert diagnostic["compact_zero_harm_each_LOPO"] is False


def test_compare_pass_uses_nested_outer_not_optimistic_lopo_diagnostic() -> None:
    full = _result(10.0, {"0": (4.0, 0), "1": (6.0, 0)})
    compact = _result(10.2, {"0": (4.1, 1), "1": (6.1, 0)})
    comparison = core.compare_representations(full, compact)
    assert comparison["passed"] is True
    assert comparison[
        "calibration_diagnostics_not_used_by_gate"
    ]["compact_zero_harm_each_LOPO"] is False
