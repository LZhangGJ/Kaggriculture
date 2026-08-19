from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import load_tables, reset
from kaggriculture_jax.constants import MAX_MARKET_ORDERS, MAX_UNITS, UnitOp
from strategic_v5.dynamic_replay_audit_v2 import (
    DynamicReplayAuditInputV2,
    audit_dynamic_replay_batch_v2,
    summarize_dynamic_replay_audit_v2,
)


def test_teacher_forced_prefix_opens_drop_sell_then_buy_without_future_label_input() -> None:
    states = jax.vmap(reset)(jnp.asarray((9201,), dtype=jnp.int32))
    states = states._replace(
        money=states.money.at[0, 0].set(100),
        unit_inventory=states.unit_inventory.at[0, 0, 0, 7].set(3),
    )
    unit_op = jnp.full((1, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8)
    unit_op = unit_op.at[0, 0].set(UnitOp.DROP)
    slots = jnp.full((1, MAX_MARKET_ORDERS), -1, dtype=jnp.int16)
    slots = slots.at[0, :3].set(jnp.asarray((19, 4, 11), dtype=jnp.int16))
    quantity = jnp.zeros((1, MAX_MARKET_ORDERS), dtype=jnp.int16)
    quantity = quantity.at[0, :3].set(jnp.asarray((3, 1, 1), dtype=jnp.int16))
    valid = jnp.zeros((1, MAX_MARKET_ORDERS), dtype=jnp.bool_)
    valid = valid.at[0, :3].set(True)
    data = DynamicReplayAuditInputV2(
        states=states,
        unit_op=unit_op,
        unit_item=jnp.full((1, MAX_UNITS), -1, dtype=jnp.int8),
        unit_quantity=jnp.ones((1, MAX_UNITS), dtype=jnp.int16),
        unit_count=jnp.ones((1,), dtype=jnp.int8),
        market_slot=slots,
        market_quantity=quantity,
        market_valid=valid,
    )
    audit = audit_dynamic_replay_batch_v2(data, load_tables())

    np.testing.assert_array_equal(
        np.asarray(audit.static_present[0, :3]),
        np.asarray((False, False, True)),
    )
    np.testing.assert_array_equal(
        np.asarray(audit.after_unit_present[0, :3]),
        np.asarray((True, False, True)),
    )
    np.testing.assert_array_equal(
        np.asarray(audit.dynamic_present[0, :3]),
        np.asarray((True, True, True)),
    )
    np.testing.assert_array_equal(
        np.asarray(audit.filled_amount[0, :3]), np.asarray((3, 1, 1))
    )
    assert int(audit.final_market_count[0]) == 3
    summary = summarize_dynamic_replay_audit_v2(audit)
    assert summary["coverage"]["opened_after_unit_phase"] == 1
    assert summary["coverage"]["opened_after_prior_market_prefix"] == 1
    assert summary["coverage"]["still_not_present_orders"] == 0
    assert summary["coverage"]["dynamic_core_gap_executable_orders"] == 0
    assert summary["coverage"]["dynamic_representation_rate_executable"] == 1.0
