from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_phase_challenger_bilinear_listwise_uniform_A_v2_1_locked as runner
import train_phase_challenger_bilinear_listwise_uniform_A_v2_1_locked as trainer


def _synthetic_arrays(group_outcomes: tuple[int, int, int] = (1, 2, 0)) -> dict[str, np.ndarray]:
    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    sha: list[bytes] = []
    decision: list[int] = []
    union: list[int] = []
    seeds: list[int] = []
    seats: list[int] = []
    edits: list[int] = []
    outcomes: list[int] = []
    next_decision = 0
    for seed_index, seed in enumerate(trainer.frozen.FORMAL_SEEDS):
        for seat in (0, 1):
            state = np.zeros(trainer.frozen.STATE_WIDTH, np.float64)
            state[0] = seed_index / 7.0
            state[1] = seat
            for candidate, outcome in enumerate(group_outcomes):
                action = np.zeros(trainer.frozen.ACTION_WIDTH, np.float64)
                action[0] = (0.0, 1.0, -1.0)[candidate]
                state_rows.append(state.copy())
                action_rows.append(action)
                sha.append(hashlib.sha256(
                    f"uniform-v2.1-test|{next_decision}|{candidate}".encode("ascii")
                ).hexdigest().encode("ascii"))
                decision.append(next_decision)
                union.append(candidate)
                seeds.append(seed)
                seats.append(seat)
                edits.append(candidate)
                outcomes.append(outcome)
            next_decision += 1
    rows = len(decision)
    return {
        "state": np.asarray(state_rows), "action": np.asarray(action_rows),
        "candidate_sha256": np.asarray(sha, "S64"),
        "decision": np.asarray(decision, np.int64),
        "union_index": np.asarray(union, np.int16),
        "seed": np.asarray(seeds, np.int64), "seat": np.asarray(seats, np.int8),
        "opponent": np.zeros(rows, np.int16), "step": np.full(rows, 217, np.int16),
        "edit": np.asarray(edits, np.int16),
        "outcome": np.asarray(outcomes, np.int8),
        "margin": np.asarray(outcomes, np.float64),
    }


def _small_design(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(len(arrays["decision"]), dtype=np.int64)
    action = np.asarray(arrays["action"])
    return np.column_stack((action[:, 0], np.asarray(arrays["seat"]))), rows


def test_all_equal_listwise_skips_minimize_and_returns_exact_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = _synthetic_arrays((1, 1, 1))
    x, rows = _small_design(arrays)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("minimize must not be called for all-equal listwise labels")

    monkeypatch.setattr(trainer, "minimize", forbidden)
    coefficient, audit = trainer._fit_listwise(x, arrays, rows)
    assert np.array_equal(coefficient, np.zeros(x.shape[1], np.float64))
    assert audit["optimizer_called"] is False
    assert audit["coefficient_exact_zero"] is True
    assert audit["signal_groups"] == 0
    assert audit["signal_weight"] == 0.0


def test_single_class_risk_skips_minimize_and_returns_observed_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = _synthetic_arrays((2, 1, 0))
    x, rows = _small_design(arrays)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("minimize must not be called for a single risk class")

    monkeypatch.setattr(trainer, "minimize", forbidden)
    coefficient, constant, audit = trainer._fit_risk(x, arrays, rows)
    assert coefficient is None
    assert constant == 1.0
    assert audit["constant_probability"] == 1.0
    assert audit["optimizer_called"] is False


def test_no_credible_inner_gain_forces_positive_infinity_and_all_KEEP() -> None:
    arrays = _synthetic_arrays((1, 1, 1))
    rows = np.arange(len(arrays["decision"]), dtype=np.int64)
    score = np.zeros(len(rows), np.float64)
    risk = np.zeros(len(rows), np.float64)
    for group in trainer.frozen.decision_slices(arrays["decision"], rows):
        score[group[1:]] = (0.8, 0.7)
    threshold, audit = trainer.calibrate_threshold(arrays, rows, score, risk)
    choices = trainer.frozen.select_choices(arrays, rows, score, risk, threshold)
    assert math.isinf(threshold) and threshold > 0
    assert audit["no_credible_gain_shortcut"] is True
    assert audit["all_choices_KEEP_when_shortcut"] is True
    assert np.all(arrays["edit"][choices] == trainer.frozen.KEEP_EDIT)


def test_real_locked_heads_pass_all_optimizer_gates() -> None:
    arrays = _synthetic_arrays()
    x, rows = _small_design(arrays)
    coefficient, listwise = trainer._fit_listwise(x, arrays, rows)
    risk_coefficient, risk_constant, risk = trainer._fit_risk(x, arrays, rows)
    assert coefficient.shape == (x.shape[1],)
    assert risk_coefficient is not None and math.isnan(risk_constant)
    for audit in (listwise, risk):
        assert audit["optimizer_called"] is True
        assert audit["gate_passed"] is True
        assert all(audit["gates"].values())
        assert audit["gradient_inf_final"] <= trainer.GRADIENT_INF_TOLERANCE
        assert audit["objective_decrease"] > 0.0


@pytest.mark.parametrize("success,coefficient", [(False, 1.0), (True, 0.0)])
def test_optimizer_nonconvergence_or_large_gradient_fails_closed(
    monkeypatch: pytest.MonkeyPatch, success: bool, coefficient: float,
) -> None:
    problem = {
        "width": 1,
        "fun_jac": lambda value: (
            float((value[0] - 1.0) ** 2), np.asarray([2.0 * (value[0] - 1.0)]),
        ),
        "total_weight": 1.0, "signal_weight": 1.0,
        "signal_weight_rate": 1.0, "l2_per_cell": trainer.LISTWISE_L2_PER_CELL,
    }

    def fake_minimize(*_args, **_kwargs):
        return SimpleNamespace(
            x=np.asarray([coefficient]), success=success, status=1,
            message="synthetic failure", nit=1, nfev=2, njev=2,
        )

    monkeypatch.setattr(trainer, "minimize", fake_minimize)
    with pytest.raises(RuntimeError, match="failed closed"):
        trainer._strict_optimize(problem, "synthetic")


def test_scoped_adapter_restores_exact_objects_normally_and_after_exception() -> None:
    original_fit = trainer.frozen.fit_predict
    original_calibration = trainer.frozen.calibrate_threshold
    with trainer.scoped_frozen_flow_adapters():
        assert trainer.frozen.fit_predict is trainer.fit_predict
        assert trainer.frozen.calibrate_threshold is trainer.calibrate_threshold
    assert trainer.frozen.fit_predict is original_fit
    assert trainer.frozen.calibrate_threshold is original_calibration
    with pytest.raises(RuntimeError, match="scope probe"):
        with trainer.scoped_frozen_flow_adapters():
            raise RuntimeError("scope probe")
    assert trainer.frozen.fit_predict is original_fit
    assert trainer.frozen.calibrate_threshold is original_calibration


def _fake_frozen_result() -> dict[str, object]:
    acceptance = {key: True for key in trainer.EXPECTED_ACCEPTANCE_KEYS}
    acceptance["headroom_folds_with_gain"] = 2
    folds = [{
        "outer_fold": fold,
        "threshold_source": "outer-train inner seed-fold OOF only",
        "outer_labels_used_for_model_or_threshold": False,
        "train_valid_seed_overlap": [],
    } for fold in range(4)]
    return {
        "schema": "frozen", "status": "repair_on_dev_screen_failed",
        "config": {}, "seed_folds": [list(values) for values in trainer.frozen.SEED_FOLDS],
        "controls": {"rank_metrics": {name: {} for name in trainer.EXPECTED_CONTROL_NAMES}},
        "folds": folds, "acceptance_gate": acceptance,
        "outer_labels_used_for_model_or_threshold": False,
        "ordinal_contract": {}, "A_choices": np.asarray([0]),
    }


def test_evaluate_uses_scoped_adapter_and_preserves_frozen_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = trainer.frozen.fit_predict
    original_calibration = trainer.frozen.calibrate_threshold

    def fake_evaluate(_arrays):
        assert trainer.frozen.fit_predict is trainer.fit_predict
        assert trainer.frozen.calibrate_threshold is trainer.calibrate_threshold
        return _fake_frozen_result()

    monkeypatch.setattr(trainer.frozen, "evaluate", fake_evaluate)
    result = trainer.evaluate({})
    assert trainer.frozen.fit_predict is original_fit
    assert trainer.frozen.calibrate_threshold is original_calibration
    assert result["schema"] == trainer.SCHEMA
    assert result["optimizer_contract"]["no_credible_inner_gain_forces_KEEP"] is True
    assert set(result["acceptance_gate"]) == trainer.EXPECTED_ACCEPTANCE_KEYS


def test_evaluate_exception_still_restores_frozen_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = trainer.frozen.fit_predict
    original_calibration = trainer.frozen.calibrate_threshold

    def fail(_arrays):
        raise RuntimeError("frozen failure")

    monkeypatch.setattr(trainer.frozen, "evaluate", fail)
    with pytest.raises(RuntimeError, match="frozen failure"):
        trainer.evaluate({})
    assert trainer.frozen.fit_predict is original_fit
    assert trainer.frozen.calibrate_threshold is original_calibration


def test_runner_source_dependency_locks_and_normalized_self_hash(tmp_path: Path) -> None:
    runner_path = Path(runner.__file__)
    assert runner._runner_normalized_sha256(runner_path) == runner.EXPECTED_RUNNER_NORMALIZED_SHA256
    assert runner._sha256(Path(runner.trainer.__file__)) == runner.EXPECTED_NEW_TRAINER_SHA256
    assert runner._sha256(Path(runner.frozen_trainer.__file__)) == runner.EXPECTED_FROZEN_TRAINER_SHA256
    assert runner._sha256(Path(runner.optimizer_locked.__file__)) == runner.EXPECTED_OPTIMIZER_LOCKED_SHA256
    assert runner._sha256(Path(runner.frozen_runner.__file__)) == runner.EXPECTED_FROZEN_RUNNER_SHA256
    tampered = tmp_path / "tampered_runner.py"
    tampered.write_text(
        runner_path.read_text(encoding="utf-8").replace(
            "Formal train-only runner", "Tampered train-only runner", 1,
        ),
        encoding="utf-8",
    )
    assert runner._runner_normalized_sha256(tampered) != runner.EXPECTED_RUNNER_NORMALIZED_SHA256
    with pytest.raises(ValueError, match="SHA mismatch"):
        runner._artifact(tampered, "0" * 64)


def test_environment_preflight_names_all_frozen_inputs_before_array_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, str | None]] = []

    def artifact(path: Path, expected: str | None = None):
        seen.append((Path(path).name, expected))
        return {"path": str(path), "sha256": expected, "bytes": 0}

    monkeypatch.setattr(runner, "_artifact", artifact)
    monkeypatch.setattr(
        runner, "_runner_normalized_sha256",
        lambda _path: runner.EXPECTED_RUNNER_NORMALIZED_SHA256,
    )
    monkeypatch.setattr(runner.frozen_runner, "verify_pre_registered_environment", lambda: {})
    monkeypatch.setattr(runner.frozen_runner, "verify_frozen_reports", lambda *_args: {})
    monkeypatch.setattr(runner, "_verify_control_report", lambda _path: {"ready": True})
    args = SimpleNamespace(
        label_root=tmp_path / "labels", materialized_root=tmp_path / "materialized",
        output_root=tmp_path / "output",
    )
    audit = runner.verify_locked_environment(args)
    names = {name for name, _ in seen}
    assert {"canonical_union_labels.jsonl", "metadata.npz", "compact_1078.npy"} <= names
    assert all(expected is not None for name, expected in seen if name in {
        "canonical_union_labels.jsonl", "metadata.npz", "compact_1078.npy",
    })
    assert audit["all_locks_verified_before_array_open"] is True


def test_control_study_report_is_content_locked(tmp_path: Path) -> None:
    tampered = tmp_path / "FINAL_REPORT.json"
    tampered.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        runner._verify_control_report(tampered)


def test_preflight_failure_prevents_array_open_evaluation_and_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_lock(_args):
        calls.append("lock")
        raise ValueError("expected lock failure")

    monkeypatch.setattr(runner, "verify_locked_environment", fail_lock)
    monkeypatch.setattr(
        runner.frozen_runner, "load_formal_arrays",
        lambda _args: (_ for _ in ()).throw(AssertionError("array opened")),
    )
    monkeypatch.setattr(
        runner.trainer, "evaluate",
        lambda _arrays: (_ for _ in ()).throw(AssertionError("evaluation ran")),
    )
    output = tmp_path / "must_not_exist"
    args = SimpleNamespace(
        label_root=tmp_path / "labels", materialized_root=tmp_path / "materialized",
        output_root=output,
    )
    with pytest.raises(ValueError, match="lock failure"):
        runner.run(args)
    assert calls == ["lock"]
    assert not output.exists()


def _stub_formal_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, Path]:
    arrays = {
        "decision": np.asarray([0, 0], np.int64),
        "edit": np.asarray([0, 1], np.int16),
        "state_id": np.asarray(["state", "state"]),
        "union_index": np.asarray([0, 1], np.int16),
        "candidate_sha256": np.asarray([b"a" * 64, b"b" * 64], "S64"),
    }
    source_rows = np.asarray([10, 11], np.int64)
    result = _fake_frozen_result()
    result.update({
        "schema": trainer.SCHEMA,
        "frozen_flow_contract": {
            "folds_unchanged": True, "controls_unchanged": True,
            "threshold_source_unchanged": True, "acceptance_unchanged": True,
            "scoped_adapter_restored": True,
        },
        "A_choices": np.asarray([0]),
    })
    monkeypatch.setattr(
        runner, "verify_locked_environment",
        lambda _args: {"EXPECTED_CONFIG_sha256": "synthetic", "all_locks_verified_before_array_open": True},
    )
    monkeypatch.setattr(
        runner.frozen_runner, "load_formal_arrays",
        lambda _args: (arrays, source_rows, {"synthetic": True}),
    )
    monkeypatch.setattr(runner.trainer, "evaluate", lambda _arrays: dict(result))
    monkeypatch.setattr(runner.frozen_runner, "EXPECTED_DECISIONS", 1)
    output = tmp_path / "formal_output"
    args = SimpleNamespace(
        label_root=tmp_path / "labels", materialized_root=tmp_path / "materialized",
        output_root=output,
    )
    return args, output


def test_runner_publishes_complete_output_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, output = _stub_formal_run(tmp_path, monkeypatch)
    report = runner.run(args)
    assert report["status"] == "formal_train_only_uniform_v2_1_complete"
    assert report["evidence_boundary"]["validation_opened"] is False
    assert report["evidence_boundary"]["sealed_opened"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "FINAL_REPORT.json", "FINAL_REPORT.md", "PROVENANCE.json",
        "choices_bilinear_listwise_uniform_A_v2_1_locked.npz",
    ]
    assert not list(tmp_path.glob(".formal_output.*.tmp"))


def test_runner_write_failure_removes_stage_and_leaves_no_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, output = _stub_formal_run(tmp_path, monkeypatch)

    def fail_write(_path: Path, _text: str) -> None:
        raise OSError("synthetic write failure")

    monkeypatch.setattr(runner, "_write_text", fail_write)
    with pytest.raises(OSError, match="write failure"):
        runner.run(args)
    assert not output.exists()
    assert not list(tmp_path.glob(".formal_output.*.tmp"))


def test_runner_has_no_model_or_acceptance_relaxation_flags() -> None:
    destinations = {action.dest for action in runner.parser()._actions}
    assert destinations == {"help", "label_root", "materialized_root", "output_root"}
    assert trainer.DEFAULT_CONFIG["optimizer_maxiter"] == 1000
    assert trainer.DEFAULT_CONFIG["listwise_l2"] == pytest.approx(0.08 / 576.0)
    assert trainer.DEFAULT_CONFIG["risk_l2"] == pytest.approx(0.08 / 576.0)
    assert trainer.LISTWISE_SEMANTICS == "top-outcome-set classification; not full ordinal"
