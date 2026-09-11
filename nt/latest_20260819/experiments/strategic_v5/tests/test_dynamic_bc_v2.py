from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax import load_tables
from kaggriculture_jax.constants import MAX_MARKET_ORDERS, MAX_UNITS, NUM_PLAYERS
from kaggriculture_jax.state import reset

from strategic_v5.dynamic_bc_v2 import (
    actor_visible_arrays_to_state_v2,
    build_dynamic_replay_teacher_batch_v2,
    market_order_slots_v2,
)
from minimal_legal_bc.observation import state_to_actor_visible_jax_v1


def _batch_reset(batch_size: int):
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape), reset(7)
    )


def _arrays(batch_size: int = 4):
    states = _batch_reset(batch_size)
    arrays = state_to_actor_visible_jax_v1(states, 0)
    arrays.update(
        {
            "target_unit_op": jnp.zeros((batch_size, MAX_UNITS), dtype=jnp.int8),
            "target_unit_item": jnp.full((batch_size, MAX_UNITS), -1, dtype=jnp.int8),
            "target_unit_amount": jnp.ones((batch_size, MAX_UNITS), dtype=jnp.int32),
            "target_unit_count": jnp.ones((batch_size,), dtype=jnp.int8),
            "target_market_op": jnp.zeros((batch_size, MAX_MARKET_ORDERS), dtype=jnp.int8),
            "target_market_item": jnp.full((batch_size, MAX_MARKET_ORDERS), -1, dtype=jnp.int8),
            "target_market_amount": jnp.zeros((batch_size, MAX_MARKET_ORDERS), dtype=jnp.int32),
            "target_market_count": jnp.zeros((batch_size,), dtype=jnp.int8),
        }
    )
    return states, arrays


def test_actor_visible_round_trip_public_and_own_private():
    expected, arrays = _arrays()
    actual = actor_visible_arrays_to_state_v2(arrays)
    round_trip = state_to_actor_visible_jax_v1(actual, 0)
    for name, value in arrays.items():
        if name.startswith("obs_"):
            assert jnp.array_equal(round_trip[name], value), name
    assert jnp.all(actual.unit_inventory[:, 1] == 0)
    assert jnp.all(actual.shed[:, 1] == 0)
    assert jnp.all(actual.seeds[:, 1] == 0)
    assert actual.money.shape == (4, NUM_PLAYERS)


def test_actor_visible_1327_hinge_price_is_not_narrowed_to_int16():
    _, arrays = _arrays(1)
    arrays["obs_market_price"] = arrays["obs_market_price"].at[0, 2].set(300_000)
    actual = actor_visible_arrays_to_state_v2(arrays)
    assert actual.market_price.dtype == jnp.int32
    assert int(actual.market_price[0, 2]) == 300_000


def test_official_market_schema_maps_to_21_slots():
    op = jnp.asarray([[2, 4, 4, 5, 5, 5, 3, 1, 6]], dtype=jnp.int8)
    item = jnp.asarray([[-1, 8, 0, 9, 10, 11, 4, -1, 8]], dtype=jnp.int8)
    assert market_order_slots_v2(op, item).tolist() == [
        [0, 1, 2, 3, 4, 5, 10, 11, 20]
    ]


def test_teacher_trace_uses_one_stop_label_and_no_future_labels():
    _, arrays = _arrays(2)
    # Row 0 buys one wheat seed then STOP; row 1 immediately STOPs.
    arrays["target_market_op"] = arrays["target_market_op"].at[0, 0].set(3)
    arrays["target_market_item"] = arrays["target_market_item"].at[0, 0].set(0)
    arrays["target_market_amount"] = arrays["target_market_amount"].at[0, 0].set(1)
    arrays["target_market_count"] = arrays["target_market_count"].at[0].set(1)
    teacher = build_dynamic_replay_teacher_batch_v2(arrays, load_tables())
    assert teacher.target_candidate_slot.shape == (2, MAX_MARKET_ORDERS)
    assert int(teacher.target_candidate_slot[0, 0]) == 6
    assert int(teacher.target_candidate_slot[0, 1]) == 21
    assert int(teacher.target_candidate_slot[1, 0]) == 21
    assert float(teacher.selection_weight[0, 0]) == 1.0
    assert float(teacher.selection_weight[0, 1]) == 0.5
    assert float(teacher.selection_weight[1, 0]) == 0.25
    assert float(jnp.sum(teacher.selection_weight[0, 2:])) == 0.0
    assert float(jnp.sum(teacher.selection_weight[1, 1:])) == 0.0
