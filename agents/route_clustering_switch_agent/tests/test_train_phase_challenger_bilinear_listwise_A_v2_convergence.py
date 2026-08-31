from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TESTS = Path(__file__).resolve().parent
for import_path in (SCRIPTS, TESTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import train_phase_challenger_bilinear_listwise_A_v2 as model
import test_train_phase_challenger_bilinear_listwise_A_v2 as fixture


def _split(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    folds = np.asarray([
        model.seed_fold_map()[int(seed)] for seed in arrays["seed"]
    ])
    rows = np.arange(len(folds))
    return rows[folds != 0], rows[folds == 0]


def _result(x0: np.ndarray, *, success: bool) -> SimpleNamespace:
    return SimpleNamespace(
        x=np.zeros_like(x0), fun=1.25, success=success,
        status=0 if success else 1,
        message="CONVERGED" if success else "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
        nit=7 if success else 120,
    )


@pytest.mark.parametrize(
    ("include_sha", "shuffle_seed", "mode"),
    (
        (True, None, "primary"),
        (False, None, "bilinear-no-SHA"),
        (True, model.SHUFFLE_SEEDS[0], "state-shuffle-0"),
        (True, model.SHUFFLE_SEEDS[1], "state-shuffle-1"),
        (True, model.SHUFFLE_SEEDS[2], "state-shuffle-2"),
    ),
)
def test_finite_nonconverged_listwise_is_rejected_for_every_model_view(
    monkeypatch: pytest.MonkeyPatch, include_sha: bool,
    shuffle_seed: int | None, mode: str,
) -> None:
    arrays = fixture._arrays()
    train, valid = _split(arrays)

    def fake_minimize(_objective, x0, **_kwargs):
        return _result(np.asarray(x0), success=False)

    monkeypatch.setattr(model, "minimize", fake_minimize)
    with pytest.raises(RuntimeError) as error:
        model.fit_predict(
            arrays, train, valid, include_sha=include_sha,
            shuffle_seed=shuffle_seed,
        )
    message = str(error.value)
    assert mode
    assert "listwise L-BFGS-B did not converge" in message
    assert "status=1" in message
    assert "nit=120" in message
    assert "fun=1.25" in message
    assert "ITERATIONS REACHED LIMIT" in message


def test_finite_nonconverged_independent_risk_head_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = fixture._arrays()
    train, valid = _split(arrays)
    calls = 0

    def fake_minimize(_objective, x0, **_kwargs):
        nonlocal calls
        calls += 1
        return _result(np.asarray(x0), success=calls == 1)

    monkeypatch.setattr(model, "minimize", fake_minimize)
    with pytest.raises(RuntimeError) as error:
        model.fit_predict(arrays, train, valid, include_sha=True)
    message = str(error.value)
    assert calls == 2
    assert "risk L-BFGS-B did not converge" in message
    assert "status=1" in message
    assert "nit=120" in message
    assert "fun=1.25" in message


def test_constant_safe_paths_do_not_require_optimizer_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_minimize(*_args, **_kwargs):
        raise AssertionError("constant-safe path called scipy.optimize.minimize")

    monkeypatch.setattr(model, "minimize", forbidden_minimize)
    result = model.evaluate(fixture._arrays(null=True))
    assert result["status"] == "repair_on_dev_screen_failed"
    assert result["all_decision_metrics"]["fires"] == 0

