from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_phase_challenger_control_view_optimizer_stability_v2_1_locked as runner
import train_phase_challenger_control_view_optimizer_stability_v2_1_locked as control


def _synthetic_arrays() -> dict[str, np.ndarray]:
    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    sha: list[bytes] = []
    decision: list[int] = []
    union: list[int] = []
    seeds: list[int] = []
    seats: list[int] = []
    edits: list[int] = []
    outcomes: list[int] = []
    margins: list[float] = []
    next_decision = 0
    for seed_index, seed in enumerate(control.frozen.FORMAL_SEEDS):
        for seat in (0, 1):
            observable = np.zeros(control.frozen.STATE_WIDTH, np.float64)
            observable[0] = seed_index / 7.0
            observable[1] = float(seat)
            for candidate, outcome in enumerate((1, 2, 0)):
                relative = np.zeros(control.frozen.ACTION_WIDTH, np.float64)
                if candidate == 1:
                    relative[0] = 1.0
                elif candidate == 2:
                    relative[0] = -1.0
                state_rows.append(observable.copy())
                action_rows.append(relative)
                sha.append(hashlib.sha256(
                    f"control-synthetic|{next_decision}|{candidate}".encode("ascii")
                ).hexdigest().encode("ascii"))
                decision.append(next_decision)
                union.append(candidate)
                seeds.append(seed)
                seats.append(seat)
                edits.append(candidate)
                outcomes.append(outcome)
                margins.append(float(outcome))
            next_decision += 1
    rows = len(decision)
    return {
        "state": np.asarray(state_rows), "action": np.asarray(action_rows),
        "candidate_sha256": np.asarray(sha, "S64"),
        "decision": np.asarray(decision, np.int64),
        "union_index": np.asarray(union, np.int16),
        "seed": np.asarray(seeds, np.int64),
        "seat": np.asarray(seats, np.int8),
        "opponent": np.zeros(rows, np.int16),
        "step": np.full(rows, 217, np.int16),
        "edit": np.asarray(edits, np.int16),
        "outcome": np.asarray(outcomes, np.int8),
        "margin": np.asarray(margins, np.float64),
    }


def test_control_view_specs_are_exactly_four_no_sha_and_twelve_shuffles() -> None:
    specs = control.control_view_specs()
    assert len(specs) == 16
    assert len({spec["view_id"] for spec in specs}) == 16
    assert sum(spec["view_kind"] == "bilinear_no_SHA" for spec in specs) == 4
    assert sum(spec["view_kind"] == "state_shuffle" for spec in specs) == 12
    for outer_fold in range(4):
        selected = [spec for spec in specs if spec["outer_fold"] == outer_fold]
        assert len(selected) == 4
        assert {tuple(spec["train_fold_ids"]) for spec in selected} == {
            tuple(fold for fold in range(4) if fold != outer_fold)
        }
        assert {spec["shuffle_seed"] for spec in selected} == {
            None, *control.frozen.SHUFFLE_SEEDS,
        }


def test_feature_builder_passes_exact_no_sha_and_shuffle_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = {"decision": np.arange(5)}
    train = np.arange(5)
    calls: list[dict[str, object]] = []

    def fake_prior(_arrays, rows, _alpha):
        calls.append({"prior_rows": tuple(map(int, rows))})
        return np.zeros((len(_arrays["decision"]), 2)), np.zeros((len(_arrays["decision"]), 2))

    def fake_project(_arrays, rows, _predict, *, train_prior, predict_prior, shuffle_seed):
        width = control.locked.frozen.BASE_WIDTH + (0 if train_prior is None else 2)
        calls.append({
            "shuffle_seed": shuffle_seed,
            "has_prior": train_prior is not None,
            "same_prior": train_prior is predict_prior,
        })
        values = np.zeros((len(rows), width), np.float64)
        return values, values.copy()

    monkeypatch.setattr(control.frozen, "_crossfit_prior_features", fake_prior)
    monkeypatch.setattr(control.frozen, "_projected_features", fake_project)
    no_sha = next(spec for spec in control.control_view_specs() if spec["view_kind"] == "bilinear_no_SHA")
    shuffled = next(spec for spec in control.control_view_specs() if spec["view_kind"] == "state_shuffle")
    assert control.build_train_features(arrays, train, no_sha).shape[1] == control.locked.frozen.BASE_WIDTH
    assert calls[-1] == {"shuffle_seed": None, "has_prior": False, "same_prior": True}
    assert control.build_train_features(arrays, train, shuffled).shape[1] == control.locked.frozen.PRIMARY_WIDTH
    assert calls[-1] == {
        "shuffle_seed": shuffled["shuffle_seed"], "has_prior": True, "same_prior": True,
    }
    assert calls[-2]["prior_rows"] == tuple(map(int, train))


@pytest.fixture(scope="module")
def synthetic_result() -> dict[str, object]:
    return control.optimization_only_control_study(_synthetic_arrays())


def test_synthetic_control_study_accounts_for_all_64_fail_closed_runs(
    synthetic_result: dict[str, object],
) -> None:
    assert synthetic_result["status"] == "control_views_optimization_stable"
    assert synthetic_result["optimizer_runs"] == 64
    assert synthetic_result["formal_integration_gate"] == {
        "all_16_views_both_heads_stable": True,
        "all_64_optimizer_runs_accounted": True,
        "heldout_outcome_metrics_used": False,
        "ready": True,
    }
    views = synthetic_result["views"]
    assert len(views) == 16
    assert all(view["listwise_stable"] and view["risk_stable"] for view in views)
    assert all(not view["heldout_feature_constructed"] for view in views)
    assert all(not view["heldout_prediction_executed"] for view in views)
    assert all(not view["heldout_outcome_metric_computed"] for view in views)
    assert all(view["feature_width"] in {72, 74} for view in views)


def test_future_formal_shortcut_test_contracts_are_registered() -> None:
    contracts = control.INTEGRATION_SHORTCUT_TEST_CONTRACTS
    assert [contract["id"] for contract in contracts] == [
        "all_equal_listwise_skips_optimizer",
        "single_class_risk_skips_optimizer",
        "no_credible_inner_gain_forces_KEEP",
    ]
    assert "do not call minimize" in contracts[0]["required_behavior"]
    assert "do not call minimize" in contracts[1]["required_behavior"]
    assert "positive infinity" in contracts[2]["required_behavior"]
    assert "all KEEP" in contracts[2]["required_test"]


def _fake_result(*, cross_boundary: bool = False) -> dict[str, object]:
    views = [{
        "view_id": f"view_{index}",
        "heldout_feature_constructed": cross_boundary and index == 0,
        "heldout_prediction_executed": False,
        "heldout_outcome_metric_computed": False,
    } for index in range(16)]
    return {
        "status": "control_views_optimization_stable",
        "views": views, "optimizer_runs": 64,
        "formal_integration_gate": {
            "all_16_views_both_heads_stable": True,
            "all_64_optimizer_runs_accounted": True,
            "heldout_outcome_metrics_used": False,
            "ready": True,
        },
    }


def test_runner_source_and_trainer_locks_match_and_tamper_fails(tmp_path: Path) -> None:
    runner_path = Path(runner.__file__)
    assert runner._runner_normalized_sha256(runner_path) == runner.EXPECTED_RUNNER_NORMALIZED_SHA256
    assert runner._sha256(Path(runner.control.__file__)) == runner.EXPECTED_CONTROL_TRAINER_SHA256
    assert runner._sha256(Path(runner.base_locked.__file__)) == runner.EXPECTED_BASE_LOCKED_SHA256
    tampered = tmp_path / "tampered_runner.py"
    tampered.write_text(
        runner_path.read_text(encoding="utf-8").replace(
            "control-view optimization-only", "tampered control-view optimization-only", 1,
        ),
        encoding="utf-8",
    )
    assert runner._runner_normalized_sha256(tampered) != runner.EXPECTED_RUNNER_NORMALIZED_SHA256
    with pytest.raises(ValueError, match="SHA mismatch"):
        runner._artifact(tampered, "0" * 64)


def test_preflight_failure_happens_before_array_open_or_optimization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_lock(_args):
        calls.append("lock")
        raise ValueError("expected trainer tamper")

    def forbidden_load(_args):
        calls.append("array-open")
        raise AssertionError("array opened after failed lock")

    def forbidden_study(_arrays):
        calls.append("optimize")
        raise AssertionError("optimization ran after failed lock")

    monkeypatch.setattr(runner, "verify_locked_environment", fail_lock)
    monkeypatch.setattr(runner.frozen_runner, "load_formal_arrays", forbidden_load)
    monkeypatch.setattr(runner.control, "optimization_only_control_study", forbidden_study)
    output = tmp_path / "must_not_exist"
    args = SimpleNamespace(
        label_root=tmp_path / "labels", materialized_root=tmp_path / "materialized",
        output_root=output,
    )
    with pytest.raises(ValueError, match="tamper"):
        runner.run(args)
    assert calls == ["lock"]
    assert not output.exists()


def test_runner_preflight_order_and_optimization_only_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def locks(_args):
        calls.append("locks")
        return {"all_locks_verified_before_array_open": True}

    def load(_args):
        calls.append("array-open")
        return {}, np.empty(0, np.int64), {"train_proxy_only": True}

    def optimize(_arrays):
        calls.append("optimize")
        return _fake_result()

    monkeypatch.setattr(runner, "verify_locked_environment", locks)
    monkeypatch.setattr(runner.frozen_runner, "load_formal_arrays", load)
    monkeypatch.setattr(runner.control, "optimization_only_control_study", optimize)
    output = tmp_path / "control_output"
    args = SimpleNamespace(
        label_root=tmp_path / "labels", materialized_root=tmp_path / "materialized",
        output_root=output,
    )
    report = runner.run(args)
    assert calls == ["locks", "array-open", "optimize"]
    assert report["boundary"]["heldout_features"] is False
    assert report["boundary"]["heldout_outcome_acceptance"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "FINAL_REPORT.json", "FINAL_REPORT.md", "PROVENANCE.json",
    ]
    assert not any("choice" in path.name.lower() for path in output.iterdir())
    loaded = json.loads((output / "FINAL_REPORT.json").read_text(encoding="utf-8"))
    assert loaded["locks"]["all_locks_verified_before_array_open"] is True


def test_result_boundary_and_runner_flags_are_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="held-out boundary"):
        runner.verify_control_result(_fake_result(cross_boundary=True))
    destinations = {action.dest for action in runner.parser()._actions}
    assert destinations == {"help", "label_root", "materialized_root", "output_root"}
