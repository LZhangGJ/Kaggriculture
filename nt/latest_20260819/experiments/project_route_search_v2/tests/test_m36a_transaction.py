from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import MarketOp, TileKind, UnitOp
from kaggriculture_jax.policy import combine_player_actions
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.state import load_tables, reset
from kaggriculture_jax.types import Events
from project_route_search_v2.m36_calendar import (
    kawashigi_opening_calendar_v3,
    validate_route_calendar_v3,
)
from project_route_search_v2.m36_schema import (
    M36IntentFailureV3,
    M36IntentStatusV3,
)
from project_route_search_v2.m36_transaction import (
    m36_player_action_dict_v3,
    m36a_policy_step_v3,
    reconcile_commitment_bundle_v3,
)


def _events(batch_size: int) -> Events:
    return Events(
        jnp.zeros((batch_size, 30, 100), dtype=jnp.bool_),
        jnp.zeros((batch_size, 30, 101), dtype=jnp.int8),
    )


def test_kawashigi_opening_calendar_is_high_level_and_valid() -> None:
    calendar = kawashigi_opening_calendar_v3(2)
    assert validate_route_calendar_v3(calendar) == []
    np.testing.assert_array_equal(
        np.asarray(calendar.hand_target_by_day[:, 0]), np.asarray((5, 5))
    )
    np.testing.assert_array_equal(
        np.asarray(calendar.crop_target_by_day[0, 0]),
        np.asarray((7, 0, 0, 0, 12)),
    )
    np.testing.assert_array_equal(
        np.asarray(calendar.animal_purchase_additions_by_day[0, 0]),
        np.asarray((0, 2, 2)),
    )


def test_m36a_generates_and_reconciles_exact_gold_opening() -> None:
    states = jax.vmap(reset)(jnp.asarray((17,), dtype=jnp.int32))
    calendar = kawashigi_opening_calendar_v3(1)
    decide0 = jax.jit(lambda s, c: m36a_policy_step_v3(s, c, 0))
    decide1 = jax.jit(lambda s, c: m36a_policy_step_v3(s, c, 1))
    (player0, projected0), (player1, projected1) = (
        decide0(states, calendar),
        decide1(states, calendar),
    )
    assert int(player0.unit_op[0, 0]) == UnitOp.BUILD_PASTURE
    assert int(player1.unit_op[0, 0]) == UnitOp.BUILD_PASTURE
    assert int(player1.market_count[0]) == 10
    np.testing.assert_array_equal(
        np.asarray(player1.market_op[0]),
        np.asarray(
            (
                MarketOp.HIRE,
                MarketOp.HIRE,
                MarketOp.HIRE,
                MarketOp.HIRE,
                MarketOp.HIRE,
                MarketOp.BUY_ANIMAL,
                MarketOp.BUY_ANIMAL,
                MarketOp.BUY_SEED,
                MarketOp.BUY_SEED,
                MarketOp.BUY_PRODUCT,
            )
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(player1.market_item[0]),
        np.asarray((-1, -1, -1, -1, -1, 10, 11, 0, 4, 0)),
    )
    np.testing.assert_array_equal(
        np.asarray(player1.bundle.requested_quantity[0]),
        np.asarray((1, 1, 1, 1, 1, 2, 2, 7, 12, 6)),
    )
    np.testing.assert_array_equal(
        np.asarray(player1.market_amount[0]),
        np.asarray((0, 0, 0, 0, 0, 2, 2, 7, 12, 6)),
    )
    assert int(player1.diagnostics.hard_error_count[0]) == 0
    assert int(player1.bundle.pending_structures[0, 1]) == 3
    np.testing.assert_array_equal(
        np.asarray(player1.bundle.future_place_plan[0]), np.asarray((0, 2, 2))
    )

    action = combine_player_actions(
        m36_player_action_dict_v3(player0), m36_player_action_dict_v3(player1)
    )
    next_states = batched_step_sync(states, action, _events(1), load_tables())
    reconciled, diagnostics = reconcile_commitment_bundle_v3(
        projected1, next_states, player1.bundle, 1
    )
    pasture_count = np.sum(
        np.asarray(next_states.tile_kind[0, 1]) == TileKind.PASTURE
    )
    assert pasture_count == 1
    assert int(next_states.money[0, 1]) == 22
    assert int(next_states.hires_today[0, 1]) == 5
    assert int(next_states.shed[0, 1, 10]) == 2
    assert int(next_states.shed[0, 1, 11]) == 2
    assert int(next_states.seeds[0, 1, 0]) == 7
    assert int(next_states.seeds[0, 1, 4]) == 12
    assert int(next_states.shed[0, 1, 0]) == 5
    assert int(reconciled.requested_quantity[0, 9]) == 6
    assert int(reconciled.filled_quantity[0, 9]) == 5
    assert int(reconciled.post_step_quantity[0, 9]) == 5
    assert int(reconciled.intent_status[0, 9]) == M36IntentStatusV3.PARTIAL
    assert (
        int(reconciled.failure_code[0, 9])
        == M36IntentFailureV3.INSUFFICIENT_CASH
    )
    assert int(diagnostics.hard_error_count[0]) == 0

