from __future__ import annotations

import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase_challenger_action_prior_residual_A_v1 as runner
import train_phase_challenger_action_prior_residual_A_v1 as model


def _group(seed: int, opponent: int, seat: int, candidate: int) -> bytes:
    return hashlib.blake2b(
        f"{seed}:{opponent}:{seat}:{candidate}".encode(), digest_size=16,
    ).digest()


def _arrays() -> dict[str, np.ndarray]:
    keys = (
        "features", "delta_margin", "membership", "decision", "opponent",
        "seed", "edit", "outcome", "seat", "step", model.pair_v0.GROUP_KEY,
    )
    values: dict[str, list] = {key: [] for key in keys}
    relative = model.pair_v0.RELATIVE_INDICES
    decision = 0
    for seed in model.FORMAL_SEEDS:
        for opponent in (0, 1):
            context = opponent
            for seat in (0, 1):
                # KEEP=tie. A wins in context0, B wins in context1; the other
                # action regresses. C is a same-outcome margin-only candidate.
                outcomes = (
                    1, 2 if context == 0 else 0,
                    2 if context == 1 else 0, 1,
                )
                margins = (0.0, 0.0, 0.0, 1.0)
                for candidate, (outcome, margin) in enumerate(zip(outcomes, margins)):
                    row = np.zeros(1078, np.float32)
                    row[:32] = float(context)
                    row[model.pair_v0.SEAT_FEATURE_INDEX] = float(seat)
                    if candidate:
                        row[relative[candidate - 1]] = 1.0
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
                    values[model.pair_v0.GROUP_KEY].append(
                        _group(seed, opponent, seat, candidate)
                    )
                decision += 1
    dtypes = {
        "features": np.float32, "delta_margin": np.float64,
        "membership": np.int8, "decision": np.int64,
        "opponent": np.int16, "seed": np.int64, "edit": np.int8,
        "outcome": np.int8, "seat": np.int8, "step": np.int16,
        model.pair_v0.GROUP_KEY: "S16",
    }
    return {key: np.asarray(value, dtypes[key]) for key, value in values.items()}


@lru_cache(maxsize=1)
def _positive_result() -> dict:
    return model.evaluate(_arrays(), trees=16, random_seed=20260829)


def test_pair1062_schema_is_reused_without_label_or_donor_features() -> None:
    assert model.PAIR_WIDTH == 1062
    assert model.pair_v0.FEATURE_SCHEMA["shared_width"] == 908
    assert model.pair_v0.FEATURE_SCHEMA["relative_action_width"] == 154
    assert not any(
        token in name.lower()
        for name in model.PAIR_FEATURE_NAMES
        for token in ("outcome", "margin", "reward", "target", "label", "donor_")
    )


def test_heads_are_class_balanced_and_margin_is_exact_same_outcome_only() -> None:
    a, _ = model.outcome_arrays(_arrays())
    pairs = model.build_pairs(a)
    _, audit = model.fit_hierarchy(pairs, trees=2, random_seed=7)
    assert model.classifier_params(2, 7)["class_weight"] == "balanced"
    assert audit["gain_head"]["positive_rows"] == 32
    assert audit["risk_head"]["positive_rows"] == 32
    assert audit["neutral_margin_head"] == {
        "kind": "constant-same-outcome-margin",
        "training_rows": 32,
        "training_filter": "relative_outcome == 0",
        "loss_to_tie_rows_admitted": 0,
    }


def test_action_prior_is_train_only_exact_and_unknown_action_falls_back_KEEP() -> None:
    a, _ = model.outcome_arrays(_arrays())
    pairs = model.build_pairs(a)
    hierarchy, audit = model.fit_hierarchy(pairs, trees=2, random_seed=11)
    assert audit["action_prior"]["exact_action_deltas"] == 3
    unknown = {key: np.asarray(value).copy() for key, value in pairs.items()}
    unknown["action_delta_fingerprint"][:] = b"unknown-action!!"
    predicted = model.predict_hierarchy(hierarchy, unknown)["primary"]
    assert not predicted["known_action"].any()
    scores = model.candidate_scores(a, unknown, predicted)
    choices = model.select_choices(a, scores, -np.inf)
    assert np.all(np.asarray(a["edit"])[choices] == 0)


def test_state_shuffle_preserves_action_and_seat_but_breaks_state_assignment() -> None:
    a, _ = model.outcome_arrays(_arrays())
    pairs = model.build_pairs(a)
    shuffled, audit = model.shuffled_state_features(pairs, 19)
    assert np.array_equal(shuffled[:, 907:], pairs["features"][:, 907:])
    assert audit["moved_decision_groups"] > 0
    assert audit["labels_used_to_construct_permutation"] is False


def test_thresholds_are_unique_decision_top_breakpoints_with_KEEP() -> None:
    arrays = {
        "decision": np.asarray([0, 0, 1, 1]),
        "edit": np.asarray([0, 1, 0, 1]),
    }
    scores = {
        "known_action": np.asarray([False, True, False, True]),
        "action_support": np.asarray([0, 3, 0, 4]),
        "gain_probability": np.asarray([0.0, 0.3, 0.0, 0.8]),
        "risk_probability": np.asarray([1.0, 0.1, 1.0, 0.1]),
        "safe_score": np.asarray([-np.inf, 0.2, -np.inf, 0.7]),
        "neutral_margin_prediction": np.zeros(4),
    }
    assert model.threshold_breakpoints(arrays, scores) == (
        model.ALL_KNOWN_SAFE_SCORE, 0.2, 0.7, model.KEEP_ONLY_SAFE_SCORE,
    )
    assert model.select_choices(arrays, scores, 0.7).tolist() == [0, 3]
    assert model.select_choices(
        arrays, scores, model.KEEP_ONLY_SAFE_SCORE,
    ).tolist() == [0, 2]


def test_positive_control_passes_and_primary_beats_both_ablations() -> None:
    result = _positive_result()
    assert result["acceptance_gate"]["passed"] is True
    assert result["acceptance_gate"][
        "at_least_two_headroom_outer_folds_add_wins"
    ] is True
    assert result["acceptance_gate"][
        "primary_rank_strictly_beats_support_only"
    ] is True
    assert result["acceptance_gate"][
        "primary_rank_strictly_beats_state_shuffle"
    ] is True
    for fold in result["folds"]:
        assert fold["outer_full_train_refit"] is True
        assert fold["outer_refit_train_seed_count"] == 6
        assert fold["inner_seed_fold_OOF_used_for_calibration_only"] is True
        assert fold["LOPO"]["models_fit"] == 0
        assert fold["LOPO"]["predictions_discarded"] == 0
        for mode in ("primary", "support_only", "state_shuffle"):
            calibration = fold["calibration"][mode]
            assert calibration["candidate_row_quantiles_used"] is False
            assert calibration["KEEP_only_included"] is True


def test_outer_valid_labels_cannot_change_its_threshold_or_choices() -> None:
    original = _arrays()
    changed = {key: value.copy() for key, value in original.items()}
    heldout = np.isin(changed["seed"], model.SEED_FOLDS[1])
    candidates = heldout & (changed["edit"] != 0)
    changed["outcome"][candidates] = np.where(
        changed["outcome"][candidates] == 2, 0, 2,
    )
    changed["delta_margin"][candidates] += 777.0
    before = _positive_result()
    after = model.evaluate(changed, trees=16, random_seed=20260829)
    assert before["folds"][1]["calibration"] == after["folds"][1]["calibration"]
    a, _ = model.outcome_arrays(original)
    positions = [
        index for index, rows in enumerate(
            model.pair_v0.outcome_v1.base.decision_slices(a["decision"])
        )
        if int(a["seed"][rows[0]]) in model.SEED_FOLDS[1]
    ]
    assert np.array_equal(
        before["A_choices"][positions], after["A_choices"][positions],
    )
    assert before["folds"][1]["outer_metrics"]["primary"] != after[
        "folds"
    ][1]["outer_metrics"]["primary"]


def _formal_payloads() -> tuple[dict, dict, dict]:
    expected = runner.EXPECTED_INPUT
    materialized = {
        "storage": {
            "compact1078": {"sha256": expected["compact1078_sha256"]},
            "metadata": {"sha256": expected["metadata_sha256"]},
        },
        "panel": {"candidate_order_sha256": expected["candidate_order_sha256"]},
    }
    diagnostic = {
        "artifacts": {"derived_fingerprints": {
            "sha256": expected["derived_fingerprints_sha256"],
        }},
    }
    panel = {"outcome_metadata_sha256": expected["outcome_metadata_sha256"]}
    return materialized, diagnostic, panel


def test_formal_identity_locks_inputs_parameters_and_schema() -> None:
    materialized, diagnostic, panel = _formal_payloads()
    frozen = runner.frozen_identity(
        materialized_report=materialized, diagnostic_report=diagnostic,
        panel=panel,
        materialized_report_sha256=runner.EXPECTED_INPUT[
            "materialized_report_sha256"
        ],
        diagnostic_report_sha256=runner.EXPECTED_INPUT[
            "diagnostic_report_sha256"
        ],
        trees=runner.FORMAL_TREES, random_seed=runner.FORMAL_RANDOM_SEED,
    )
    assert frozen["all_formal_inputs_and_parameters_locked"] is True
    with pytest.raises(ValueError, match="trees/random seed"):
        runner.frozen_identity(
            materialized_report=materialized, diagnostic_report=diagnostic,
            panel=panel,
            materialized_report_sha256=runner.EXPECTED_INPUT[
                "materialized_report_sha256"
            ],
            diagnostic_report_sha256=runner.EXPECTED_INPUT[
                "diagnostic_report_sha256"
            ],
            trees=runner.FORMAL_TREES + 1,
            random_seed=runner.FORMAL_RANDOM_SEED,
        )
    broken = {**panel, "outcome_metadata_sha256": "0" * 64}
    with pytest.raises(ValueError, match="frozen input identity"):
        runner.frozen_identity(
            materialized_report=materialized, diagnostic_report=diagnostic,
            panel=broken,
            materialized_report_sha256=runner.EXPECTED_INPUT[
                "materialized_report_sha256"
            ],
            diagnostic_report_sha256=runner.EXPECTED_INPUT[
                "diagnostic_report_sha256"
            ],
            trees=runner.FORMAL_TREES,
            random_seed=runner.FORMAL_RANDOM_SEED,
        )


def test_runner_hashes_new_trainer_tests_and_pair1062_dependency(tmp_path: Path) -> None:
    choices, report = tmp_path / "choices.npz", tmp_path / "report.json"
    choices.write_bytes(b"choices")
    report.write_bytes(b"report")
    payload = runner.choices_provenance(
        choices, report, {"train_only": True}, {"locked": True},
    )
    implementation = payload["implementation"]
    assert Path(implementation["trainer"]["path"]) == Path(model.__file__).resolve()
    assert Path(implementation["focused_tests"]["path"]) == Path(__file__).resolve()
    assert Path(implementation["pair1062_dependency"]["path"]) == Path(
        model.pair_v0.__file__
    ).resolve()
