from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_adaptive_tail_oracle_v0 as oracle


def _observation(market_price: int = 1) -> dict:
    empty = [[None] * 10 for _ in range(10)]
    farm = {
        "money": 1000, "hands": [], "unlocked_quadrants": ["NW"],
        "hires_today": 0, "tiles": empty,
    }
    return {
        "step": 216, "player": 0, "day": 9, "hour": 0,
        "farms": [farm, {**farm, "money": 900}],
        "private": {"shed": {}, "seeds": {}, "inventories": []},
        "market": {
            "inventory": {item: market_price for item in (
                "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                "EGG", "MILK", "WOOL", "FERTILIZER",
            )},
            "prices": {item: market_price for item in (
                "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                "EGG", "MILK", "WOOL", "FERTILIZER",
            )},
        },
        "town": {"unlocked_shops": []},
    }


def _tape() -> list[dict]:
    return [
        {"farmer": ["PASS"], "hands": [], "market": []}
        for _ in range(719)
    ]


def test_choose_candidate_keeps_current_on_exact_tie() -> None:
    rewards = np.asarray([[10.0, 5.0], [10.0, 5.0], [9.0, 5.0]])
    route, index, rank = oracle.choose_candidate(
        ["current", "a", "b"], rewards, 0, "current"
    )
    assert (route, index, rank) == ("current", 0, (2, 5.0))


def test_choose_candidate_uses_outcome_before_margin() -> None:
    rewards = np.asarray([[9.0, 10.0], [20.0, 19.0], [30.0, 10.0]])
    route, _, rank = oracle.choose_candidate(
        ["current", "small_win", "large_win"], rewards, 0, "current"
    )
    assert route == "large_win"
    assert rank == (2, 20.0)


def test_snapshot_contract_excludes_market_environment() -> None:
    left = oracle.snapshot_contract(_observation(1), _tape())
    right = oracle.snapshot_contract(_observation(999), _tape())
    assert left[0].shape == (97,)
    assert left[1].shape == (18,)
    assert np.array_equal(left[0], right[0])
    assert np.array_equal(left[1], right[1])
    assert not any(name.startswith("market_") for name in oracle.STATE_NAMES)
    assert not any("history" in name for name in oracle.STATE_NAMES)
