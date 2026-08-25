from pathlib import Path

import jax
import jax.numpy as jnp

from hasegawa_jax_v1 import (
    hasegawa_step_with_external_v1,
    initialize_hasegawa_carry_v1,
    load_hasegawa_plan_bank_v1,
    select_hasegawa_route_v1,
)
from kaggriculture_jax.constants import (
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    SHOP_NAMES,
    MarketOp,
    UnitOp,
)
from kaggriculture_jax.state import load_event_bank, load_tables, reset
from kaggriculture_jax.types import Action, Events


ROOT = Path(__file__).resolve().parents[3]
BANK = (
    ROOT
    / "experiments"
    / "hasegawa_jax_v1"
    / "artifacts"
    / "hasegawa_plan_bank_v1.npz"
)


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


def test_plan_bank_contains_only_high_level_fields():
    bank = load_hasegawa_plan_bank_v1(BANK)
    assert bank.episode_id.shape == (102,)
    assert bank.plan.crop_target_by_day.shape == (102, 30, 5)
    assert not any("action" in field or "coordinate" in field for field in bank._fields)


def test_opening_matches_rank1_transaction():
    batch = 1
    bank = load_hasegawa_plan_bank_v1(BANK)
    state = jax.vmap(reset)(jnp.arange(batch, dtype=jnp.int32))
    _, event_bank = load_event_bank()
    events = Events(event_bank.weed_spawn[:batch], event_bank.shop_choice[:batch])
    next_state, _, diagnostics, joint = hasegawa_step_with_external_v1(
        state,
        initialize_hasegawa_carry_v1(batch),
        bank,
        null_action(batch),
        0,
        events,
        load_tables(),
    )
    assert int(joint.unit_op[0, 0, 0]) == int(UnitOp.BUILD_PASTURE)
    assert joint.market_op[0, 0].tolist() == [
        int(MarketOp.HIRE),
        int(MarketOp.HIRE),
        int(MarketOp.HIRE),
        int(MarketOp.HIRE),
        int(MarketOp.HIRE),
        int(MarketOp.BUY_ANIMAL),
        int(MarketOp.BUY_ANIMAL),
        int(MarketOp.BUY_SEED),
        int(MarketOp.BUY_SEED),
        int(MarketOp.BUY_PRODUCT),
    ]
    assert int(next_state.hires_today[0, 0]) == 5
    assert int(next_state.shed[0, 0, 0]) == 4
    assert int(diagnostics.hard_error_count[0]) == 0


def test_first_shop_routes_yarn_and_farmers_market_to_different_calendars():
    bank = load_hasegawa_plan_bank_v1(BANK)
    state = jax.vmap(reset)(jnp.arange(2, dtype=jnp.int32))._replace(
        step=jnp.full((2,), 72, dtype=jnp.int16),
        town_count=jnp.ones((2,), dtype=jnp.int8),
    )
    ids = {name: index for index, name in enumerate(SHOP_NAMES)}
    town_shops = state.town_shops.at[0, 0].set(ids["YARN_STORE"])
    town_shops = town_shops.at[1, 0].set(ids["FARMERS_MARKET"])
    route = select_hasegawa_route_v1(
        state._replace(town_shops=town_shops), bank, 0
    )
    selected_demand = bank.ref_shop_demand[route, 3]
    assert int(selected_demand[0, 7]) == 2  # WOOL x2 for Yarn Store.
    assert selected_demand[1, :5].tolist() == [1, 1, 1, 1, 0]
