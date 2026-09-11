from pathlib import Path

import jax
import jax.numpy as jnp

from hasegawa_jax_v2 import (
    hasegawa_step_with_external_v2,
    initialize_hasegawa_carry_v2,
    load_hasegawa_trace_bank_v2,
    select_hasegawa_branch_v2,
)
from kaggriculture_jax.constants import (
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.state import load_event_bank, load_tables, reset
from kaggriculture_jax.types import Action, Events


ROOT = Path(__file__).resolve().parents[3]
BANK_PATH = ROOT / "experiments/hasegawa_jax_v2/artifacts/hasegawa_trace_bank_v2.npz"


def null_action(batch: int):
    return Action(
        jnp.full((batch, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8),
        jnp.full((batch, MAX_UNITS), -1, dtype=jnp.int8),
        jnp.ones((batch, MAX_UNITS), dtype=jnp.int32),
        jnp.ones((batch,), dtype=jnp.int8),
        jnp.full((batch, MAX_MARKET_ORDERS), MarketOp.NONE, dtype=jnp.int8),
        jnp.full((batch, MAX_MARKET_ORDERS), -1, dtype=jnp.int8),
        jnp.zeros((batch, MAX_MARKET_ORDERS), dtype=jnp.int32),
        jnp.zeros((batch,), dtype=jnp.int8),
    )


def events(batch: int):
    _, bank = load_event_bank()
    return Events(bank.weed_spawn[:batch], bank.shop_choice[:batch])


def test_bank_is_nine_coherent_full_programs():
    bank = load_hasegawa_trace_bank_v2(BANK_PATH)
    assert bank.unit_op.shape == (9, 719, MAX_UNITS)
    assert bank.market_op.shape == (9, 719, MAX_MARKET_ORDERS)
    assert bank.source_episode_id.shape == (9,)


def test_branch_locks_once_and_never_daily_switches():
    state = jax.vmap(reset)(jnp.arange(1, dtype=jnp.int32))._replace(
        town_count=jnp.ones((1,), dtype=jnp.int8),
        town_shops=jnp.zeros((1, 8), dtype=jnp.int8).at[0, 0].set(4),
        step=jnp.asarray([72], dtype=jnp.int16),
    )
    carry = initialize_hasegawa_carry_v2(1)
    branch, locked, lock_step = select_hasegawa_branch_v2(state, carry)
    assert branch.tolist() == [5]
    carry = carry._replace(branch_id=branch, branch_locked=locked, branch_lock_step=lock_step)
    changed = state._replace(town_shops=state.town_shops.at[0, 0].set(1), step=jnp.asarray([240], dtype=jnp.int16))
    branch2, locked2, lock_step2 = select_hasegawa_branch_v2(changed, carry)
    assert branch2.tolist() == [5]
    assert locked2.tolist() == [True]
    assert lock_step2.tolist() == [72]


def test_opening_is_hasegawa_atomic_transaction():
    state = jax.vmap(reset)(jnp.arange(1, dtype=jnp.int32))
    bank = load_hasegawa_trace_bank_v2(BANK_PATH)
    next_state, _, diagnostics, joint = hasegawa_step_with_external_v2(
        state,
        initialize_hasegawa_carry_v2(1),
        bank,
        null_action(1),
        0,
        events(1),
        load_tables(),
    )
    assert int(joint.unit_op[0, 0, 0]) == int(UnitOp.BUILD_PASTURE)
    assert joint.market_op[0, 0].tolist() == [
        int(MarketOp.HIRE), int(MarketOp.HIRE), int(MarketOp.HIRE),
        int(MarketOp.HIRE), int(MarketOp.HIRE), int(MarketOp.BUY_ANIMAL),
        int(MarketOp.BUY_ANIMAL), int(MarketOp.BUY_SEED),
        int(MarketOp.BUY_SEED), int(MarketOp.BUY_PRODUCT),
    ]
    assert int(next_state.hires_today[0, 0]) == 5
    assert int(diagnostics.hard_counter_delta[0]) == 0


def test_weed_repair_changes_only_blocked_atomic_action():
    state = jax.vmap(reset)(jnp.arange(1, dtype=jnp.int32))
    state = state._replace(tile_kind=state.tile_kind.at[0, 0, 4, 4].set(TileKind.WEED))
    bank = load_hasegawa_trace_bank_v2(BANK_PATH)
    _, carry, _, joint = hasegawa_step_with_external_v2(
        state,
        initialize_hasegawa_carry_v2(1),
        bank,
        null_action(1),
        0,
        events(1),
        load_tables(),
    )
    assert int(joint.unit_op[0, 0, 0]) == int(UnitOp.DIG)
    assert bool(carry.weed_active[0, 0])


def test_runtime_does_not_import_old_compound_task_kernel():
    source = (ROOT / "experiments/hasegawa_jax_v2/src/hasegawa_jax_v2/agent.py").read_text(encoding="utf-8")
    assert "kaggriculture_execution_core" not in source
    assert "general_project_planner_v1" not in source
    assert "fulfillment_step" not in source
