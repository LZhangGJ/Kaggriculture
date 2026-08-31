from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_phase_challenger_bilinear_listwise_A_v2 as model


STEPS = (217, 218, 219, 220)
POSITIVE = {
    (2026086301, 0, 0, 217), (2026086301, 1, 1, 218),
    (2026086302, 0, 1, 219), (2026086302, 1, 0, 220),
    (2026086303, 0, 0, 217), (2026086303, 1, 1, 218),
    (2026086305, 1, 0, 219), (2026086306, 0, 1, 220),
}


def _sha(index: int) -> str:
    return hashlib.sha256(f"synthetic-four-step-block-{index}".encode()).hexdigest()


def _arrays(*, null: bool = False) -> dict[str, np.ndarray]:
    values: dict[str, list] = {name: [] for name in (
        "state", "action", "candidate_sha256", "decision", "union_index",
        "seed", "seat", "opponent", "step", "edit", "outcome", "margin",
    )}
    decision = 0
    for seed in model.FORMAL_SEEDS:
        for opponent in (0, 1):
            for step in STEPS:
                for seat in (0, 1):
                    positive = (seed, opponent, seat, step) in POSITIVE
                    keep_outcome = 0 if positive else 2
                    state = np.full(model.STATE_WIDTH, 4.0 if positive else -4.0)
                    for candidate in range(19):
                        action = np.zeros(model.ACTION_WIDTH)
                        if candidate == 1:
                            action[:] = 1.0
                        elif candidate > 1:
                            action[(candidate - 2) % model.ACTION_WIDTH] = 0.1 + candidate / 1000.0
                        outcome = keep_outcome
                        margin = 0.0
                        if candidate == 1:
                            outcome = 2 if positive else 0
                            margin = 5.0 if positive else -5.0
                        if null:
                            outcome, margin = keep_outcome, 0.0
                        values["state"].append(state.copy())
                        values["action"].append(action)
                        values["candidate_sha256"].append(_sha(candidate))
                        values["decision"].append(decision)
                        values["union_index"].append(candidate)
                        values["seed"].append(seed)
                        values["seat"].append(seat)
                        values["opponent"].append(opponent)
                        values["step"].append(step)
                        values["edit"].append(0 if candidate == 0 else 1)
                        values["outcome"].append(outcome)
                        values["margin"].append(margin)
                    decision += 1
    dtype = {
        "state": np.float64, "action": np.float64, "candidate_sha256": "S64",
        "decision": np.int64, "union_index": np.int16, "seed": np.int64,
        "seat": np.int8, "opponent": np.int16, "step": np.int16,
        "edit": np.int8, "outcome": np.int8, "margin": np.float64,
    }
    return {name: np.asarray(value, dtype[name]) for name, value in values.items()}


def _copy(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in arrays.items()}


def test_bilinear72_is_fixed_label_free_and_keep_zero() -> None:
    arrays = _arrays()
    model.validate_arrays(arrays)
    folds = np.asarray([model.seed_fold_map()[int(seed)] for seed in arrays["seed"]])
    train = np.flatnonzero(folds != 0)
    valid = np.flatnonzero(folds == 0)
    before_train, before_valid = model._projected_features(
        arrays, train, valid, train_prior=None, predict_prior=None, shuffle_seed=None,
    )
    changed = _copy(arrays)
    changed["outcome"] = changed["outcome"][::-1]
    changed["margin"] += 999.0
    after_train, after_valid = model._projected_features(
        changed, train, valid, train_prior=None, predict_prior=None, shuffle_seed=None,
    )
    assert before_train.shape[1] == model.BASE_WIDTH == 72
    assert np.array_equal(before_train, after_train)
    assert np.array_equal(before_valid, after_valid)
    assert model.projection_hash() == model.projection_hash()
    assert np.all(before_train[changed["edit"][train] == 0] == 0.0)


def test_rare_signal_fourfold_passes_and_beats_prior_and_shuffles() -> None:
    result = model.evaluate(_arrays())
    assert result["status"] == "repair_on_dev_screen_passed"
    assert result["acceptance_gate"]["passed"] is True
    assert result["all_decision_metrics"]["wins_lost"] == 0
    assert result["all_decision_metrics"]["outcome_regressions"] == 0
    assert result["all_decision_metrics"]["raw_win_delta"] >= 2
    assert len(result["all_decision_metrics"]["gain_seeds"]) >= 2
    assert len(result["all_decision_metrics"]["gain_opponents"]) >= 2
    assert all(result["controls"]["primary_Pareto"].values())
    assert len(result["A_choices"]) == result["input_audit"]["decisions"]
    assert all(fold["outer_labels_used_for_model_or_threshold"] is False for fold in result["folds"])
    assert all(not fold["train_valid_seed_overlap"] for fold in result["folds"])


def test_null_panel_is_constant_safe_abstention_and_fails_screen() -> None:
    result = model.evaluate(_arrays(null=True))
    assert result["status"] == "repair_on_dev_screen_failed"
    assert result["acceptance_gate"]["passed"] is False
    assert result["all_decision_metrics"]["fires"] == 0
    assert result["all_decision_metrics"]["raw_win_delta"] == 0
    assert result["all_decision_metrics"]["outcome_regressions"] == 0
    assert all(fold["primary_model"]["listwise"]["kind"].startswith("constant") for fold in result["folds"])
    assert all(fold["primary_model"]["risk"]["kind"].startswith("constant") for fold in result["folds"])


def test_positive_net_one_up_one_down_is_not_pareto_safe() -> None:
    assert model.pareto_better(
        {"upgrade_decisions": 2, "top1": 0.75, "MRR": 0.60},
        {"upgrade_decisions": 2, "top1": 0.70, "MRR": 0.55},
    )
    assert not model.pareto_better(
        {"upgrade_decisions": 2, "top1": 0.80, "MRR": 0.50},
        {"upgrade_decisions": 2, "top1": 0.70, "MRR": 0.60},
    )


@pytest.mark.parametrize("corruption", ("collision", "wrong_order", "bad_hex"))
def test_sha_collision_order_and_tamper_fail_closed(corruption: str) -> None:
    arrays = _arrays()
    first = model.decision_slices(arrays["decision"])[0]
    if corruption == "collision":
        arrays["candidate_sha256"][first[2]] = arrays["candidate_sha256"][first[1]]
    elif corruption == "wrong_order":
        arrays["union_index"][first[1]], arrays["union_index"][first[2]] = 2, 1
    else:
        arrays["candidate_sha256"][first[1]] = b"G" * 64
    with pytest.raises(ValueError):
        model.validate_arrays(arrays)


def test_same_sha_across_decisions_is_allowed_when_action_is_identical() -> None:
    audit = model.validate_arrays(_arrays())
    assert audit["candidate_sha_unique"] == 19
    assert audit["candidate_sha_is_soft_prior_not_eligibility"] is True


