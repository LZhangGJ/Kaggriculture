from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from train_compact_route_q_selector import (  # noqa: E402
    policy_metrics, trial_sort_key, zero_feature_prefixes,
)


def test_policy_metrics_use_selected_route_and_paired_seats() -> None:
    scores = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
    margins = np.asarray([[10, -5], [-2, 7], [3, 4], [-1, -2]], dtype=np.float32)
    predictions = np.asarray([0, 1, 0, 1])

    metrics = policy_metrics(scores, margins, predictions, (1, 2, 2))

    assert metrics["raw_win_rate"] == 0.75
    assert metrics["both_seats_win_rate"] == 0.5
    assert metrics["mean_margin"] == 4.5


def test_feature_ablation_preserves_shape_and_other_columns() -> None:
    matrix = np.asarray([[1.0, 2.0, 3.0]])
    result = zero_feature_prefixes(
        matrix, ["self_money", "opponent_money", "shop_yarn"], ("opponent_",)
    )

    assert result.tolist() == [[1.0, 0.0, 3.0]]
    assert matrix.tolist() == [[1.0, 2.0, 3.0]]


def test_trial_selection_prioritizes_the_worst_opponent() -> None:
    safer = {
        "min_leaf": 2, "max_features": .5, "margin_weight": 0,
        "cross_validation": {
            "minimum_opponent_raw_win_rate": .86,
            "paired_seed_raw_win_lower_95pct": .88,
            "raw_win_rate": .90,
            "both_seats_win_rate": .89,
            "mean_margin": 100,
        },
    }
    higher_mean = {
        "min_leaf": 2, "max_features": .5, "margin_weight": 0,
        "cross_validation": {
            "minimum_opponent_raw_win_rate": .80,
            "paired_seed_raw_win_lower_95pct": .94,
            "raw_win_rate": .96,
            "both_seats_win_rate": .95,
            "mean_margin": 200,
        },
    }
    assert trial_sort_key(safer) < trial_sort_key(higher_mean)
