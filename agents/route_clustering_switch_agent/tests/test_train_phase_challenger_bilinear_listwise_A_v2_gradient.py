from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_phase_challenger_bilinear_listwise_A_v2 as model


def test_tied_best_set_listwise_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(20260829)
    x = rng.normal(size=(7, 5))
    coef = rng.normal(scale=0.2, size=5)
    starts = np.asarray([0, 3], np.int64)
    lengths = np.asarray([3, 4], np.int64)
    best = np.asarray([0, 1, 1, 1, 0, 1, 0], np.float64)
    weight = np.asarray([0.5] * 3 + [0.75] * 4, np.float64)
    l2 = 0.13
    loss, analytic = model._listwise_loss_gradient(
        coef, x, starts, lengths, best, weight, l2,
    )
    epsilon = 1e-6
    numeric = np.empty_like(coef)
    for index in range(len(coef)):
        plus, minus = coef.copy(), coef.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        high = model._listwise_loss_gradient(
            plus, x, starts, lengths, best, weight, l2,
        )[0]
        low = model._listwise_loss_gradient(
            minus, x, starts, lengths, best, weight, l2,
        )[0]
        numeric[index] = (high - low) / (2.0 * epsilon)
    assert np.isfinite(loss)
    assert np.allclose(analytic, numeric, rtol=2e-6, atol=2e-7)


def test_tied_best_gradient_is_probability_mass_not_uniform_target() -> None:
    x = np.eye(3)
    coef = np.asarray([0.0, 2.0, -1.0])
    starts = np.asarray([0], np.int64)
    lengths = np.asarray([3], np.int64)
    best = np.asarray([1.0, 1.0, 0.0])
    weight = np.ones(3)
    _, gradient = model._listwise_loss_gradient(
        coef, x, starts, lengths, best, weight, 0.0,
    )
    # The two best rows have unequal conditional probability, hence their
    # gradients cannot equal p - uniform(best).
    probability = np.exp(coef) / np.exp(coef).sum()
    wrong = probability - np.asarray([0.5, 0.5, 0.0])
    assert not np.allclose(gradient, wrong)
