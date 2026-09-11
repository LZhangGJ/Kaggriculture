from __future__ import annotations

import numpy as np

from kaggriculture_jax.constants import MAX_MARKET_ORDERS, MAX_UNITS, UnitOp
from project_route_search_v2.null_opponent import (
    combine_with_null_opponent,
    jax_null_player_action,
    official_null_agent,
)


def test_official_null_agent_passes_farmer_and_all_existing_hands() -> None:
    obs = {
        "player": 1,
        "farms": [
            {"hands": [{"x": 0}]},
            {"hands": [{"x": 1}, {"x": 2}, {"x": 3}]},
        ],
    }
    action = official_null_agent(obs)
    assert action == {
        "farmer": ["PASS"],
        "hands": [["PASS"], ["PASS"], ["PASS"]],
        "market": [],
    }


def test_batched_jax_null_action_has_only_pass_and_no_market_orders() -> None:
    action = jax_null_player_action((3, 2))
    assert action["unit_op"].shape == (3, 2, MAX_UNITS)
    assert action["market_op"].shape == (3, 2, MAX_MARKET_ORDERS)
    np.testing.assert_array_equal(np.asarray(action["unit_op"]), UnitOp.PASS)
    np.testing.assert_array_equal(np.asarray(action["unit_count"]), 1)
    np.testing.assert_array_equal(np.asarray(action["market_count"]), 0)


def test_null_opponent_can_be_combined_in_both_seats() -> None:
    player = jax_null_player_action(4)
    for seat in (0, 1):
        joint = combine_with_null_opponent(player, player_seat=seat)
        assert joint.unit_op.shape == (4, 2, MAX_UNITS)
        assert joint.market_op.shape == (4, 2, MAX_MARKET_ORDERS)
        np.testing.assert_array_equal(np.asarray(joint.unit_count), 1)
        np.testing.assert_array_equal(np.asarray(joint.market_count), 0)

