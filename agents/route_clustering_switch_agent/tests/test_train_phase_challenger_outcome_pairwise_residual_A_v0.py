from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_phase_challenger_outcome_pairwise_residual_A_v0 as model
import run_phase_challenger_outcome_pairwise_residual_A_v0 as runner


POSITIVE_SEEDS = {2026086301, 2026086302, 2026086303}


def _group(seed: int, opponent: int, candidate: int) -> bytes:
    return hashlib.blake2b(
        f"{seed}:{opponent}:{candidate}".encode(), digest_size=16,
    ).digest()


def _arrays() -> dict[str, np.ndarray]:
    values: dict[str, list] = {key: [] for key in (
        "features", "delta_margin", "membership", "decision", "opponent",
        "seed", "edit", "outcome", "seat", "step", model.GROUP_KEY,
    )}
    decision = 0
    donor = np.asarray(model.FEATURE_SCHEMA["excluded_indices"], np.int64)
    relative = model.RELATIVE_INDICES
    for seed in model.FORMAL_SEEDS:
        for opponent in (0, 1):
            for seat in (0, 1):
                if opponent == 0:
                    outcomes = (0, 2 if seed in POSITIVE_SEEDS else 0, 1)
                    margins = (0.0, 10.0 if seed in POSITIVE_SEEDS else 1.0, 2.0)
                else:
                    outcomes, margins = (2, 2, 0), (0.0, 1.0, -10.0)
                for candidate, (outcome, margin) in enumerate(zip(outcomes, margins)):
                    row = np.zeros(1078, np.float32)
                    row[0] = float(seed % 10)
                    row[1] = float(opponent)
                    row[2] = float(seat)
                    row[model.SEAT_FEATURE_INDEX] = float(seat)
                    if candidate:
                        row[relative[candidate - 1]] = float(candidate)
                    row[donor] = candidate * 1000.0 + np.arange(len(donor))
                    values["features"].append(row)
                    values["delta_margin"].append(margin)
                    values["membership"].append(0)
                    values["decision"].append(decision)
                    values["opponent"].append(opponent)
                    values["seed"].append(seed)
                    values["edit"].append(0 if candidate == 0 else 1)
                    values["outcome"].append(outcome)
                    values["seat"].append(seat)
                    values["step"].append(217)
                    values[model.GROUP_KEY].append(_group(seed, opponent, candidate))
                phase = np.asarray(values["features"][-1]).copy()
                phase[relative[3]] = 99.0
                values["features"].append(phase)
                values["delta_margin"].append(1000.0)
                values["membership"].append(1)
                values["decision"].append(decision)
                values["opponent"].append(opponent)
                values["seed"].append(seed)
                values["edit"].append(1)
                values["outcome"].append(2)
                values["seat"].append(seat)
                values["step"].append(217)
                values[model.GROUP_KEY].append(_group(seed, opponent, 3))
                decision += 1
    dtypes = {
        "features": np.float32, "delta_margin": np.float64,
        "membership": np.int8, "decision": np.int64,
        "opponent": np.int16, "seed": np.int64, "edit": np.int8,
        "outcome": np.int8, "seat": np.int8, "step": np.int16,
        model.GROUP_KEY: "S16",
    }
    return {key: np.asarray(value, dtypes[key]) for key, value in values.items()}


def test_schema_is_exact_908_plus_154_without_provenance_or_labels() -> None:
    assert model.PAIR_WIDTH == 1062
    assert model.FEATURE_SCHEMA["shared_width"] == 908
    assert model.FEATURE_SCHEMA["relative_action_width"] == 154
    assert len(model.FEATURE_SCHEMA["excluded_donor_provenance_names"]) == 12
    assert set(model.FEATURE_SCHEMA["excluded_derivable_context_names"]) == {
        "runtime_actor_count", "source_actor_count", "segment", "segment_offset",
    }
    assert not any(
        token in name.lower()
        for name in model.PAIR_FEATURE_NAMES
        for token in ("outcome", "margin", "reward", "target", "label", "donor_")
    )


def test_pair_X_is_candidate_vs_KEEP_only_and_label_provenance_invariant() -> None:
    arrays = _arrays()
    a, _ = model.outcome_arrays(arrays)
    before = model.build_pair_features(a)
    assert before["features"].shape == (64, 1062)
    assert len(before["features"]) == sum(
        len(rows) - 1 for rows in model.outcome_v1.base.decision_slices(a["decision"])
    )
    changed = {key: value.copy() for key, value in a.items()}
    changed["outcome"][:] = changed["outcome"][::-1]
    changed["delta_margin"][:] += 999.0
    changed["features"][:, model.FEATURE_SCHEMA["excluded_indices"]] += 12345.0
    after = model.build_pair_features(changed)
    for key in (
        "features", "candidate_row", "keep_row", "pair_fingerprint",
        "action_delta_fingerprint",
    ):
        assert np.array_equal(before[key], after[key])


def test_paired_cell_weights_match_36_and_step219_34_contract() -> None:
    rows = 70
    pairs = {
        "decision": np.arange(rows),
        "seed": np.full(rows, 2026086301),
        "opponent": np.zeros(rows),
        "step": np.r_[np.full(36, 217), np.full(34, 219)],
        "seat": np.r_[np.zeros(18), np.ones(18), np.zeros(17), np.ones(17)],
    }
    weights, audit = model.pair_weights(pairs)
    assert np.all(weights[:36] == 1.0 / 36.0)
    assert np.all(weights[36:] == 1.0 / 34.0)
    assert np.isclose(weights[:36].sum(), 1.0)
    assert np.isclose(weights[36:].sum(), 1.0)
    assert audit["pair_rows_are_not_reported_as_independent_samples"] is True


def test_neutral_margin_head_is_neutral_only_and_shallow_nonlinear() -> None:
    a, _ = model.outcome_arrays(_arrays())
    pairs = model.attach_pair_targets(a, model.build_pair_features(a))
    fitted, audit = model.fit_models(pairs, trees=2, random_seed=7)
    probability, margin = model.predict_models(fitted, pairs["features"][:4])
    assert probability.shape == (4, 3)
    assert margin.shape == (4,)
    assert audit["margin_training_rows"] == int(np.count_nonzero(
        pairs["target_class"] == model.NEUTRAL
    ))
    assert audit["margin_training_target_class"] == "neutral_only"
    assert model.model_params(2, 7)["max_depth"] == 6
    assert model.model_params(2, 7)["min_samples_leaf"] == 2


def test_margin_never_creates_eligibility() -> None:
    arrays = {"decision": np.asarray([0, 0, 0]), "edit": np.asarray([0, 1, 1])}
    scores = {
        "regression_probability": np.zeros(3),
        "win_gain_probability": np.zeros(3),
        "neutral_margin_prediction": np.asarray([0.0, 1e9, 2e9]),
    }
    keep = model.select_choices(arrays, scores, {
        "max_regression_probability": 1.0, "min_win_gain_probability": 0.0,
    })
    assert keep.tolist() == [0]
    scores["win_gain_probability"][1:] = 0.5
    assert model.select_choices(arrays, scores, {
        "max_regression_probability": 1.0, "min_win_gain_probability": 0.1,
    }).tolist() == [2]


def test_strict_four_fold_screen_reports_rank_baselines_and_negative_control() -> None:
    result = model.evaluate(_arrays(), trees=2, random_seed=19)
    assert result["A_decisions"] == 32
    assert result["A_candidate_KEEP_pairs"] == 64
    assert len(result["folds"]) == 4
    assert result["folds"][0]["fold0_pure_negative_control"] == {
        "pre_registered": True, "candidate_win_gain_decisions": 0, "passed": True,
    }
    for fold in result["folds"]:
        assert fold["outer_labels_used_for_model_or_threshold"] is False
        assert fold["inner_OOF_audit"]["only_seed_fold_OOF_used_for_calibration"] is True
        assert fold["unknown_opponent_lopo"]["used_for_choice_or_threshold"] is False
        rank = fold["no_threshold_rank_diagnostics"]
        assert "top1_win_recall" in rank["no_threshold_model_primary_rank"]
        assert "mean_reciprocal_rank_first_win" in rank[
            "outer_train_exact_action_delta_win_frequency_baseline"
        ]
        assert rank["margin_prediction_used_in_primary_rank_diagnostic"] is False
        assert "fire_attribution" in fold["outer_metrics_all_decisions"]
    assert set(result["acceptance_gate"]) >= {
        "at_least_two_headroom_outer_folds_add_wins", "wins_lost_zero",
        "outcome_regressions_zero", "passed",
    }


def test_outer_valid_labels_cannot_change_that_folds_threshold_or_choices() -> None:
    original = _arrays()
    changed = {key: value.copy() for key, value in original.items()}
    heldout = np.isin(changed["seed"], model.SEED_FOLDS[1])
    candidate = heldout & (changed["edit"] != 0) & (changed["membership"] == 0)
    changed["outcome"][candidate] = np.where(changed["outcome"][candidate] == 2, 0, 2)
    changed["delta_margin"][candidate] += 777.0
    before = model.evaluate(original, trees=2, random_seed=31)
    after = model.evaluate(changed, trees=2, random_seed=31)
    a, _ = model.outcome_arrays(original)
    positions = [
        index for index, rows in enumerate(model.outcome_v1.base.decision_slices(a["decision"]))
        if int(a["seed"][rows[0]]) in model.SEED_FOLDS[1]
    ]
    assert before["folds"][1]["chosen_thresholds"] == after["folds"][1][
        "chosen_thresholds"
    ]
    assert np.array_equal(before["A_choices"][positions], after["A_choices"][positions])
    assert before["folds"][1]["outer_oracle_headroom_all_decisions"] != after[
        "folds"
    ][1]["outer_oracle_headroom_all_decisions"]


def test_runner_reuses_v1_strong_gates_and_hashes_actual_v0_trainer(
    tmp_path: Path,
) -> None:
    assert runner.audit_full_panel is runner.v1_runner.audit_full_panel
    assert runner.map_choice_rows is runner.v1_runner.map_choice_rows
    choices, report = tmp_path / "choices.npz", tmp_path / "report.json"
    choices.write_bytes(b"choices")
    report.write_bytes(b"report")
    payload = runner.choices_provenance(choices, report, {"train_only": True})
    trainer = payload["implementation"]["trainer"]
    trainer_path = Path(model.__file__).resolve()
    assert Path(trainer["path"]) == trainer_path
    assert trainer["sha256"] == runner.residual._sha256_file(trainer_path)


def test_runner_locks_27552_pair_and_per_step_contract() -> None:
    result = {
        "A_decisions": runner.EXPECTED_DECISIONS,
        "A_candidate_KEEP_pairs": runner.EXPECTED_PAIRS,
        "A_candidate_KEEP_pairs_by_step": dict(runner.EXPECTED_PAIRS_BY_STEP),
        "folds": [{
            "outer_pair_weight_audit": {
                "each_two_seat_cell_total_weight": 1.0,
                "all_cells_contain_both_seats": True,
                "equal_candidate_count_by_seat": True,
            }
        }] * 4,
    }
    assert runner.audit_pair_result(result)["candidate_KEEP_pairs"] == 27_552
    broken = {**result, "A_candidate_KEEP_pairs_by_step": dict(
        result["A_candidate_KEEP_pairs_by_step"]
    )}
    broken["A_candidate_KEEP_pairs_by_step"]["219"] += 1
    broken["A_candidate_KEEP_pairs_by_step"]["217"] -= 1
    try:
        runner.audit_pair_result(broken)
    except ValueError as error:
        assert "by step" in str(error)
    else:
        raise AssertionError("same-total wrong per-step pair panel was accepted")
