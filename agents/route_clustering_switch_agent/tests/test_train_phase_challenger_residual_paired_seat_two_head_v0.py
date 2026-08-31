from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_phase_challenger_residual_paired_seat_two_head_v0 as paired


def test_robustify_uses_group_worst_and_preserves_realized_values() -> None:
    arrays = {
        "features": np.zeros((3, 2), np.float32),
        "decision": np.asarray([0, 1, 2]),
        "delta_margin": np.asarray([54.0, -42.0, 9.0]),
        paired.GROUP_KEY: np.asarray([b"a", b"a", b"b"], dtype="S1"),
    }
    result, audit = paired.robustify(arrays, "delta_margin")
    assert result["paired_worst_value"].tolist() == [-42.0, -42.0, 9.0]
    assert result["target_nonpositive"].tolist() == [1, 1, 0]
    assert result["delta_margin"].tolist() == [54.0, -42.0, 9.0]
    assert audit["positive_nonpositive_conflict_groups"] == 1
    assert audit["realized_values_preserved"] is True


def test_selection_filters_risky_top_utility_before_argmax() -> None:
    arrays = {
        "decision": np.asarray([0, 0, 0]),
        "edit": np.asarray([0, 1, 1]),
        "delta_margin": np.asarray([0.0, -5.0, 3.0]),
        "opponent": np.asarray([0, 0, 0]),
        "seed": np.asarray([1, 1, 1]),
    }
    metrics, choices = paired.selection_metrics(
        arrays,
        utility=np.asarray([0.0, 9.0, 2.0]),
        risk=np.asarray([1.0, 0.8, 0.1]),
        params={"utility_threshold": 0.0, "max_nonpositive_risk": 0.2},
        farmer_only=False,
    )
    assert choices.tolist() == [2]
    assert metrics["beneficial"] == 1
    assert metrics["harmful"] == 0


def test_exact_cache_blocks_known_nonpositive_group() -> None:
    train = {
        paired.GROUP_KEY: np.asarray([b"a", b"b"], dtype="S1"),
        "delta_margin": np.asarray([-1.0, 4.0]),
    }
    valid = {
        paired.GROUP_KEY: np.asarray([b"a", b"c"], dtype="S1"),
    }
    cache = paired.harmful_cache(train, "delta_margin")
    assert cache == {b"a"}
    assert paired.cache_mask(valid, cache).tolist() == [True, False]


def test_risk_quantile_is_upper_and_utility_quantile_is_lower() -> None:
    utility, risk = paired.two_head_scores(
        np.asarray([[1.0, 2.0, 3.0, 4.0]]),
        np.asarray([[0.0, 0.1, 0.8, 1.0]]),
    )
    assert utility.tolist() == [1.0]
    assert risk.tolist() == [1.0]
