import numpy as np

from select_dynamic_ga_portfolio import exact_state_cv_metrics


def test_exact_state_cv_uses_public_state_groups() -> None:
    scores = np.asarray([
        [1, 0], [1, 0], [0, 1], [0, 1],
        [1, 0], [1, 0], [0, 1], [0, 1],
    ], dtype=np.float32)
    states = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    metrics = exact_state_cv_metrics(scores, states, seed_count=4, folds=2)
    assert metrics["minimum_fold_raw_win_rate"] == 1.0
    assert metrics["cross_validation_raw_win_rate"] == 1.0
    assert metrics["training_raw_win_rate"] == 1.0
