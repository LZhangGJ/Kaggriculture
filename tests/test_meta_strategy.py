from __future__ import annotations

import numpy as np

from kaggriculture_lab.meta_strategy import (
    MatchOutcome,
    build_payoff_matrix,
    regret_matching_mixture,
    robust_opponent_weights,
)


def test_payoff_matrix_and_psro_capture_cyclic_meta_game() -> None:
    records = [
        MatchOutcome("a", "b", 0, 1, 1.0),
        MatchOutcome("b", "a", 1, 0, -1.0),
        MatchOutcome("b", "c", 1, 2, 1.0),
        MatchOutcome("c", "b", 2, 1, -1.0),
        MatchOutcome("c", "a", 2, 0, 1.0),
        MatchOutcome("a", "c", 0, 2, -1.0),
    ]
    matrix = build_payoff_matrix(records)
    mixture = regret_matching_mixture(matrix.payoff, iterations=2_000)
    robust = robust_opponent_weights(matrix.payoff, mixture)

    np.testing.assert_allclose(matrix.payoff, -matrix.payoff.T)
    np.testing.assert_allclose(mixture, np.full(3, 1.0 / 3.0), atol=0.03)
    np.testing.assert_allclose(robust.sum(), 1.0)


def test_payoff_matrix_uses_reverse_evidence_for_missing_cell() -> None:
    matrix = build_payoff_matrix([MatchOutcome("a", "b", 3, 8, 0.75)])
    assert matrix.payoff[0, 1] == 0.75
    assert matrix.payoff[1, 0] == -0.75
