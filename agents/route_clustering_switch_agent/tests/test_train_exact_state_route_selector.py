from __future__ import annotations

import numpy as np

from train_exact_state_route_selector import fit_state_actions


def test_state_actions_use_mean_utility_and_fallback_for_unseen_state() -> None:
    state_ids = np.asarray([0, 0, 1])
    scores = np.asarray([[1, 0], [0, 1], [0, 1]], dtype=np.float32)
    margins = np.zeros_like(scores)
    actions = fit_state_actions(
        state_ids, scores, margins, np.asarray([0, 1]), 3, fallback=1
    )
    assert actions.tolist() == [0, 1, 1]


def test_state_actions_do_not_sacrifice_an_indistinguishable_opponent() -> None:
    state_ids = np.asarray([0, 0, 0, 0])
    opponent_ids = np.asarray([0, 0, 1, 1])
    scores = np.asarray([
        [1, 0, .75], [1, 0, .75],
        [0, 1, .75], [0, 1, .75],
    ], dtype=np.float32)
    actions = fit_state_actions(
        state_ids, scores, np.zeros_like(scores), np.arange(4), 1,
        fallback=0, opponent_ids=opponent_ids,
    )
    assert actions.tolist() == [2]
