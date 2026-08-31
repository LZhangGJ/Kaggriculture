from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase_challenger_outcome_multiclass_A_v1 as runner
import train_phase_challenger_outcome_multiclass_A_nested_v1 as model


POSITIVE_SEEDS = {2026086301, 2026086302, 2026086303}


def _formal_arrays() -> dict[str, np.ndarray]:
    values: dict[str, list] = {
        key: [] for key in (
            "features", "delta_margin", "membership", "decision", "opponent",
            "seed", "edit", "outcome", "seat", "step",
        )
    }
    decision = 0
    for seed in model.FORMAL_SEEDS:
        for opponent in (0, 1):
            if opponent == 0:
                outcomes = (0, 2 if seed in POSITIVE_SEEDS else 0, 1)
                margins = (0.0, 10.0 if seed in POSITIVE_SEEDS else 1.0, 2.0)
            else:
                outcomes = (2, 2, 0)
                margins = (0.0, 1.0, -10.0)
            for candidate, (outcome, margin) in enumerate(zip(outcomes, margins)):
                values["features"].append([
                    float(candidate), float(opponent), float(seed % 10), 1.0,
                ])
                values["delta_margin"].append(margin)
                values["membership"].append(0)
                values["decision"].append(decision)
                values["opponent"].append(opponent)
                values["seed"].append(seed)
                values["edit"].append(0 if candidate == 0 else 1)
                values["outcome"].append(outcome)
                values["seat"].append(decision % 2)
                values["step"].append(217)
            # A tempting phase-only row must never enter the model or choices.
            values["features"].append([99.0, float(opponent), float(seed % 10), 1.0])
            values["delta_margin"].append(1000.0)
            values["membership"].append(1)
            values["decision"].append(decision)
            values["opponent"].append(opponent)
            values["seed"].append(seed)
            values["edit"].append(1)
            values["outcome"].append(2)
            values["seat"].append(decision % 2)
            values["step"].append(217)
            decision += 1
    dtypes = {
        "features": np.float32, "delta_margin": np.float64,
        "membership": np.int8, "decision": np.int64,
        "opponent": np.int16, "seed": np.int64, "edit": np.int8,
        "outcome": np.int8, "seat": np.int8, "step": np.int16,
    }
    return {key: np.asarray(value, dtypes[key]) for key, value in values.items()}


def _cartesian_keep_panel() -> dict[str, np.ndarray]:
    cells = [
        (step, seed, opponent, seat)
        for step in runner.DEFAULT_STEPS
        for seed in model.FORMAL_SEEDS
        for opponent in runner.FORMAL_OPPONENTS
        for seat in runner.FORMAL_SEATS
    ]
    rows = len(cells)
    return {
        "features": np.zeros((rows, 2), np.float32),
        "delta_margin": np.zeros(rows, np.float64),
        "membership": np.zeros(rows, np.int8),
        "decision": np.arange(rows, dtype=np.int64),
        "opponent": np.asarray([cell[2] for cell in cells], np.int16),
        "seed": np.asarray([cell[1] for cell in cells], np.int64),
        "edit": np.zeros(rows, np.int8),
        "outcome": np.zeros(rows, np.int8),
        "seat": np.asarray([cell[3] for cell in cells], np.int8),
        "step": np.asarray([cell[0] for cell in cells], np.int16),
    }


def test_seed_folds_are_the_pre_registered_four_pairs() -> None:
    assert model.seed_fold_map() == {
        2026086300: 0, 2026086304: 0,
        2026086301: 1, 2026086305: 1,
        2026086302: 2, 2026086306: 2,
        2026086303: 3, 2026086307: 3,
    }


def test_labels_make_only_actual_win_gain_the_primary_positive() -> None:
    arrays = _formal_arrays()
    a, source = model.outcome_arrays(arrays)
    assert np.all(np.asarray(arrays["membership"])[source] == 0)
    rows = model.base.decision_slices(a["decision"])
    positive = rows[2]  # seed 6301, opponent 0: loss -> win and loss -> tie.
    assert a["target_class"][positive].tolist() == [
        model.NEUTRAL, model.WIN_GAIN, model.NEUTRAL,
    ]
    assert a["relative_outcome"][positive].tolist() == [0, 1, 1]
    regression = rows[3]  # seed 6301, opponent 1: win -> loss on row 2.
    assert a["target_class"][regression].tolist() == [
        model.NEUTRAL, model.NEUTRAL, model.REGRESSION,
    ]


def test_small_leaf_single_model_and_missing_class_alignment() -> None:
    params = model.model_params(2, 7)
    assert params["min_samples_leaf"] == 2
    assert params["min_samples_leaf"] != 16
    arrays = model.outcome_arrays(_formal_arrays())[0]
    neutral = np.flatnonzero(arrays["target_class"] == model.NEUTRAL)
    fitted, audit = model.fit_model(
        {key: np.asarray(value)[neutral] for key, value in arrays.items()}, 2, 7,
    )
    probability = model.predict_proba3(fitted, arrays["features"][:3])
    assert probability.shape == (3, 3)
    assert np.all(probability[:, model.WIN_GAIN] == 0.0)
    assert audit["class_counts"]["win_gain"] == 0


def test_selection_needs_no_outer_outcome_or_margin_labels() -> None:
    arrays = {
        "decision": np.asarray([0, 0, 0]),
        "edit": np.asarray([0, 1, 1]),
    }
    probability = np.asarray([
        [0.0, 1.0, 0.0], [0.1, 0.1, 0.8], [0.3, 0.0, 0.7],
    ])
    choice = model.select_choices(arrays, probability, {
        "max_regression_probability": 0.2,
        "min_win_gain_probability": 0.5,
    })
    assert choice.tolist() == [1]
    fallback = model.select_choices(arrays, probability, {
        "max_regression_probability": 0.0,
        "min_win_gain_probability": 1.0,
    })
    assert fallback.tolist() == [0]


def test_four_outer_folds_cover_every_decision_and_keep() -> None:
    arrays = _formal_arrays()
    result = model.evaluate(arrays, trees=2, random_seed=19)
    assert result["status"] == "strict_four_seed_fold_outcome_multiclass_A_complete"
    assert result["seed_folds"] == [list(values) for values in model.SEED_FOLDS]
    assert result["A_decisions"] == 16
    assert result["all_decision_metrics"]["decisions"] == 16
    assert result["all_decision_metrics"]["KEEP_candidates_evaluated"] == 16
    assert result["phase_enabled"] is False
    assert result["model_params"]["min_samples_leaf"] == 2
    assert len(result["folds"]) == 4
    for fold in result["folds"]:
        assert not fold["train_valid_seed_overlap"]
        assert fold["outer_labels_used_for_threshold_calibration"] is False
        assert fold["outer_metrics_all_decisions"]["decisions"] == 4
        assert fold["outer_oracle_headroom_all_decisions"]["decisions"] == 4
        assert fold["outer_prediction_view_models"]["mode"] == "known_pool_loso_primary"
        assert fold["inner_OOF_audit"][
            "individual_tree_probabilities_treated_as_independent"
        ] is False
        assert fold["unknown_opponent_lopo"]["status"] == "unsupported_abstain"
        assert fold["unknown_opponent_lopo"][
            "used_for_known_pool_choice_or_threshold"
        ] is False
    assert np.all(np.asarray(arrays["membership"])[
        result["A_source_indices"][result["A_choices"]]
    ] == 0)


def test_outer_valid_label_mutation_cannot_change_that_folds_threshold_or_choice() -> None:
    original = _formal_arrays()
    changed = {key: value.copy() for key, value in original.items()}
    heldout = np.isin(changed["seed"], model.SEED_FOLDS[0])
    newly_winning = (
        heldout
        & (changed["opponent"] == 0)
        & (changed["features"][:, 0] == 1.0)
    )
    changed["outcome"][newly_winning] = 2
    changed["delta_margin"][newly_winning] = 999.0
    before = model.evaluate(original, trees=2, random_seed=31)
    after = model.evaluate(changed, trees=2, random_seed=31)
    before_a = model.outcome_arrays(original)[0]
    heldout_decisions = {
        int(before_a["decision"][rows[0]])
        for rows in model.base.decision_slices(before_a["decision"])
        if int(before_a["seed"][rows[0]]) in model.SEED_FOLDS[0]
    }
    decision_order = [
        int(before_a["decision"][rows[0]])
        for rows in model.base.decision_slices(before_a["decision"])
    ]
    positions = [
        index for index, decision in enumerate(decision_order)
        if decision in heldout_decisions
    ]
    assert before["folds"][0]["chosen_thresholds"] == after["folds"][0][
        "chosen_thresholds"
    ]
    assert np.array_equal(
        before["A_choices"][positions], after["A_choices"][positions],
    )
    assert before["folds"][0][
        "outer_oracle_headroom_all_decisions"
    ] != after["folds"][0]["outer_oracle_headroom_all_decisions"]


def test_panel_audit_and_choice_mapping_preserve_source_indices() -> None:
    arrays = _formal_arrays()
    source = np.arange(len(arrays["decision"]), dtype=np.int64) + 100
    panel = {
        "decision_steps": list(runner.DEFAULT_STEPS),
        "outcome_metadata_sha256": "f" * 64,
    }
    audit = runner.audit_full_panel(
        arrays, panel, source, expected_decisions=16,
    )
    assert audit["decisions"] == 16
    assert audit["A_decisions"] == 16
    assert audit["phase_rows_used_by_model"] == 0
    assert len(audit["source_indices_sha256"]) == 64

    result = model.evaluate(arrays, trees=2, random_seed=23)
    mapped = runner.map_choice_rows(
        arrays, source, result["A_source_indices"], result["A_choices"],
    )
    assert len(mapped["decision"]) == 16
    assert np.array_equal(
        mapped["materialized_row_index"],
        source[mapped["input_panel_row_index"]],
    )
    broken = np.asarray(result["A_choices"]).copy()
    broken[0] = broken[1]
    with pytest.raises(ValueError, match="escaped"):
        runner.map_choice_rows(arrays, source, result["A_source_indices"], broken)


def test_panel_audit_rejects_nonunique_source_mapping() -> None:
    arrays = _formal_arrays()
    source = np.arange(len(arrays["decision"]), dtype=np.int64)
    source[1] = source[0]
    with pytest.raises(ValueError, match="source_indices"):
        runner.audit_full_panel(
            arrays,
            {"decision_steps": list(runner.DEFAULT_STEPS)},
            source,
            expected_decisions=16,
        )


def test_choices_provenance_hashes_the_actual_v1_trainer(tmp_path: Path) -> None:
    choices = tmp_path / "choices.npz"
    report = tmp_path / "report.json"
    choices.write_bytes(b"choices")
    report.write_bytes(b"report")
    payload = runner.choices_provenance(choices, report, {"train_only": True})
    provenance = payload["implementation"]
    trainer_path = Path(model.__file__).resolve()
    assert Path(provenance["trainer"]["path"]) == trainer_path
    assert provenance["trainer"]["sha256"] == runner.residual._sha256_file(
        trainer_path,
    )


def test_panel_audit_rejects_duplicate_and_missing_cartesian_cell_at_count_96() -> None:
    arrays = _cartesian_keep_panel()
    source = np.arange(len(arrays["decision"]), dtype=np.int64)
    panel = {
        "decision_steps": list(runner.DEFAULT_STEPS),
        "outcome_metadata_sha256": "f" * 64,
    }
    audit = runner.audit_full_panel(arrays, panel, source)
    assert audit["per_step_seed_opponent_seat_Cartesian_complete"] is True
    assert set(audit["decisions_by_step"].values()) == {
        runner.EXPECTED_DECISIONS_PER_STEP,
    }

    broken = {key: value.copy() for key, value in arrays.items()}
    first_step = broken["step"] == runner.DEFAULT_STEPS[0]
    missing = (
        first_step
        & (broken["seed"] == model.FORMAL_SEEDS[-1])
        & (broken["opponent"] == runner.FORMAL_OPPONENTS[-1])
        & (broken["seat"] == runner.FORMAL_SEATS[-1])
    )
    duplicate = (
        first_step
        & (broken["seed"] == model.FORMAL_SEEDS[0])
        & (broken["opponent"] == runner.FORMAL_OPPONENTS[0])
        & (broken["seat"] == runner.FORMAL_SEATS[0])
    )
    assert int(np.count_nonzero(missing)) == 1
    assert int(np.count_nonzero(duplicate)) == 1
    broken["seed"][missing] = model.FORMAL_SEEDS[0]
    broken["opponent"][missing] = runner.FORMAL_OPPONENTS[0]
    broken["seat"][missing] = runner.FORMAL_SEATS[0]
    assert int(np.count_nonzero(first_step)) == runner.EXPECTED_DECISIONS_PER_STEP
    with pytest.raises(ValueError, match="Cartesian coverage"):
        runner.audit_full_panel(broken, panel, source)


def test_runner_hard_codes_all_sixteen_repair_on_dev_steps() -> None:
    assert len(runner.DEFAULT_STEPS) == 16
    assert runner.DEFAULT_STEPS == (
        217, 218, 219, 266, 267, 268, 435, 436,
        457, 482, 483, 484, 529, 530, 531, 553,
    )
    destinations = {action.dest for action in runner.parser()._actions}
    assert "decision_steps" not in destinations
    assert "phase" not in destinations
    assert runner.parser().parse_args([]).trees == 48
