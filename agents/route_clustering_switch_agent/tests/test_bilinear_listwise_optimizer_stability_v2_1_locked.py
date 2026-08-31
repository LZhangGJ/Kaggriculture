from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_phase_challenger_bilinear_listwise_optimizer_stability_v2_1_locked as runner
import train_phase_challenger_bilinear_listwise_optimizer_stability_v2_1_locked as study


def _paired_arrays(outcomes: tuple[int, int, int]) -> dict[str, np.ndarray]:
    return {
        "decision": np.repeat(np.arange(2), 3),
        "seed": np.full(6, 2026086300, np.int64),
        "opponent": np.zeros(6, np.int16),
        "step": np.full(6, 217, np.int16),
        "seat": np.repeat(np.arange(2), 3),
        "edit": np.tile(np.arange(3), 2),
        "outcome": np.tile(np.asarray(outcomes, np.int8), 2),
    }


def _finite_gradient(fun_jac, coef: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    result = np.empty_like(coef)
    for index in range(len(coef)):
        delta = np.zeros_like(coef)
        delta[index] = epsilon
        result[index] = (fun_jac(coef + delta)[0] - fun_jac(coef - delta)[0]) / (2.0 * epsilon)
    return result


def test_tied_top_set_has_locked_negative_curvature_counterexample() -> None:
    arrays = _paired_arrays((2, 2, 0))
    x = np.vstack((np.eye(3), np.eye(3)))
    problem = study.listwise_problem(
        x, arrays, np.arange(6),
        objective="tied_top_outcome_set_probability_mass", l2_per_cell=0.0,
    )
    direction = np.asarray([1.0, -1.0, 0.0])
    curvature = float(direction @ problem["hessp"](np.zeros(3), direction))
    assert curvature == pytest.approx(-1.0 / 3.0, abs=1e-12)
    audit = study.convexity_audit()
    assert audit["tied_top_outcome_set_probability_mass"]["convex"] is False
    assert audit["signal_only_uniform_top_outcome_cross_entropy"]["not_full_ordinal"] is True


@pytest.mark.parametrize("objective", [
    "tied_top_outcome_set_probability_mass",
    "signal_only_uniform_top_outcome_cross_entropy",
])
def test_listwise_gradient_and_exact_hessp(objective: str) -> None:
    arrays = _paired_arrays((2, 2, 0))
    rng = np.random.default_rng(20260829)
    x = rng.normal(size=(6, 4))
    problem = study.listwise_problem(
        x, arrays, np.arange(6), objective=objective, l2_per_cell=0.03,
    )
    coef = rng.normal(scale=0.2, size=4)
    vector = rng.normal(size=4)
    value, gradient = problem["fun_jac"](coef)
    assert np.isfinite(value)
    assert np.allclose(gradient, _finite_gradient(problem["fun_jac"], coef), rtol=2e-5, atol=2e-7)
    epsilon = 1e-6
    finite_hvp = (
        problem["fun_jac"](coef + epsilon * vector)[1]
        - problem["fun_jac"](coef - epsilon * vector)[1]
    ) / (2.0 * epsilon)
    exact_hvp = problem["hessp"](coef, vector)
    assert np.allclose(exact_hvp, finite_hvp, rtol=3e-5, atol=3e-7)
    if objective == "signal_only_uniform_top_outcome_cross_entropy":
        assert float(vector @ exact_hvp) >= 0.03 * float(vector @ vector) - 1e-10


def test_all_equal_outcome_group_has_exactly_zero_data_loss_gradient_and_hvp() -> None:
    arrays = _paired_arrays((1, 1, 1))
    rng = np.random.default_rng(17)
    x = rng.normal(size=(6, 4))
    problem = study.listwise_problem(
        x, arrays, np.arange(6),
        objective="signal_only_uniform_top_outcome_cross_entropy", l2_per_cell=0.0,
    )
    coef = rng.normal(size=4)
    vector = rng.normal(size=4)
    value, gradient = problem["fun_jac"](coef)
    assert value == pytest.approx(0.0, abs=1e-14)
    assert np.allclose(gradient, 0.0, atol=1e-14)
    assert np.allclose(problem["hessp"](coef, vector), 0.0, atol=1e-14)
    assert problem["total_weight"] == pytest.approx(1.0)
    assert problem["signal_weight"] == pytest.approx(0.0)
    assert problem["signal_weight_rate"] == pytest.approx(0.0)


def test_risk_gradient_hessp_and_cell_mass_normalization() -> None:
    arrays = _paired_arrays((1, 2, 0))
    rng = np.random.default_rng(23)
    x = rng.normal(size=(6, 4))
    problem = study.risk_problem(x, arrays, np.arange(6), l2_per_cell=0.02)
    coef = rng.normal(scale=0.2, size=5)
    vector = rng.normal(size=5)
    _, gradient = problem["fun_jac"](coef)
    assert np.allclose(gradient, _finite_gradient(problem["fun_jac"], coef), rtol=2e-5, atol=2e-7)
    epsilon = 1e-6
    finite_hvp = (
        problem["fun_jac"](coef + epsilon * vector)[1]
        - problem["fun_jac"](coef - epsilon * vector)[1]
    ) / (2.0 * epsilon)
    assert np.allclose(problem["hessp"](coef, vector), finite_hvp, rtol=3e-5, atol=3e-7)
    assert problem["total_weight"] == pytest.approx(1.0)
    assert problem["signal_weight_rate"] == pytest.approx(1.0)


def _quadratic_problem() -> dict[str, object]:
    optimum = np.asarray([0.5, -0.25])

    def fun_jac(coef: np.ndarray):
        delta = coef - optimum
        return 0.5 * float(delta @ delta), delta

    return {
        "fun_jac": fun_jac,
        "hessp": lambda _coef, vector: vector,
        "width": 2, "constant": None,
    }


def test_nonconverged_optimizer_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_minimize(_fun, x0, **_kwargs):
        return SimpleNamespace(
            x=np.asarray(x0), success=False, status=1,
            message="ITERATION LIMIT", nit=1000, nfev=1001, njev=1001, nhev=0,
        )

    monkeypatch.setattr(study, "minimize", fake_minimize)
    report = study.run_solver(_quadratic_problem(), "L-BFGS-B-1000", np.zeros(2))
    assert report["success"] is False
    assert report["gate_passed"] is False
    assert report["gates"]["scipy_success"] is False
    assert report["nit"] == 1000


def test_solver_is_deterministic_for_fixed_problem_and_initialization() -> None:
    first = study.run_solver(_quadratic_problem(), "L-BFGS-B-1000", np.zeros(2))
    second = study.run_solver(_quadratic_problem(), "L-BFGS-B-1000", np.zeros(2))
    assert first["gate_passed"] is True
    assert second["gate_passed"] is True
    assert first["coefficient_sha256"] == second["coefficient_sha256"]
    assert first["objective_final"] == pytest.approx(second["objective_final"], abs=0.0)


def test_initializations_and_ten_training_targets_are_locked() -> None:
    first = study.deterministic_initializations(74)
    second = study.deterministic_initializations(74)
    assert tuple(first) == study.INITIALIZATIONS
    assert all(np.array_equal(first[name], second[name]) for name in first)
    specs = study.training_target_specs()
    assert len(specs) == 10
    assert len({tuple(spec["fold_ids"]) for spec in specs}) == 10
    assert sum(spec["target_kind"] == "two_fold_inner_train" for spec in specs) == 6
    assert sum(spec["target_kind"] == "three_fold_outer_train" for spec in specs) == 4
    assert study.LISTWISE_L2_PER_CELL == pytest.approx(0.08 / 576.0)
    assert study.RISK_L2_PER_CELL == pytest.approx(0.08 / 576.0)


def _fake_study_result(*, cross_boundary: bool = False) -> dict[str, object]:
    targets = [{
        "target_id": f"target_{index}",
        "heldout_prediction_executed": cross_boundary and index == 0,
        "heldout_outcome_metric_computed": False,
    } for index in range(10)]
    return {
        "status": "optimization_only_stable",
        "targets": targets,
        "recommendation": {
            "listwise_objective": "signal_only_uniform_top_outcome_cross_entropy",
            "listwise_solver": "L-BFGS-B-1000",
            "risk_solver": "L-BFGS-B-1000",
            "selection_used_heldout_outcome_metrics": False,
        },
    }


def test_runner_writes_only_optimization_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.frozen_runner, "verify_pre_registered_environment", lambda: {"locked": True})
    monkeypatch.setattr(
        runner.frozen_runner, "load_formal_arrays",
        lambda _args: ({}, np.empty(0, np.int64), {"train_proxy_only": True}),
    )
    monkeypatch.setattr(runner.study, "optimization_only_study", lambda _arrays: _fake_study_result())
    output = tmp_path / "locked_optimizer_study"
    args = SimpleNamespace(
        label_root=tmp_path / "labels", materialized_root=tmp_path / "materialized",
        output_root=output,
    )
    report = runner.run(args)
    assert report["boundary"]["heldout_predictions"] is False
    assert report["boundary"]["heldout_outcome_acceptance"] is False
    assert report["boundary"]["validation_opened"] is False
    assert report["boundary"]["sealed_opened"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "FINAL_REPORT.json", "FINAL_REPORT.md", "PROVENANCE.json",
    ]
    assert not any("choice" in path.name.lower() for path in output.iterdir())
    loaded = json.loads((output / "FINAL_REPORT.json").read_text(encoding="utf-8"))
    assert loaded["evidence_class"].startswith("train-proxy optimization-only")


def test_runner_rejects_any_heldout_boundary_crossing() -> None:
    with pytest.raises(RuntimeError, match="held-out boundary"):
        runner.verify_optimization_only_result(_fake_study_result(cross_boundary=True))


def test_runner_exposes_no_model_or_optimizer_tuning_flags() -> None:
    destinations = {action.dest for action in runner.parser()._actions}
    assert destinations == {"help", "label_root", "materialized_root", "output_root"}
