from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import (
    batched_project_action_phases,
    empty_action,
    load_tables,
    reset,
)
from kaggriculture_jax.constants import (
    MARKET_MAX_INVENTORY,
    MARKET_MIN_INVENTORY,
    MAX_MARKET_ORDERS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    SHED_CAPACITY,
    MarketOp,
    UnitOp,
)
from strategic_v5.turn_ledger_v2 import (
    TurnPhaseV2,
    apply_dynamic_market_candidate_v2,
    apply_dynamic_market_matrix_slot_v2,
    apply_turn_market_order_v2,
    close_market_phase_v2,
    close_unit_phase_v2,
    initialize_turn_ledger_v2,
    ledger_action_v2,
    ordered_market_label_to_order_v2,
    refresh_dynamic_market_candidates_v2,
    refresh_dynamic_market_feasibility_v2,
    refresh_dynamic_market_matrix_v2,
)
from strategic_v5.e4_core import (
    build_full_core_candidates_v1,
    evaluate_full_core_feasibility_v1,
)
from strategic_v5.lifecycle import reset_controller_state_v1
from strategic_v5.replay_bc_v2 import broad_replay_bc_candidate_program_v2


TABLES = load_tables()
WOOL = 7
TOMATO = 2
COW_ITEM = NUM_PRODUCTS + 1


def _states(count: int = 1):
    return jax.vmap(reset)(jnp.arange(9100, 9100 + count, dtype=jnp.int32))


def _actions(count: int = 1):
    action = empty_action()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (count,) + value.shape), action
    )


def _drop_wool_action(count: int = 1):
    action = _actions(count)
    return action._replace(
        unit_op=action.unit_op.at[:, 0, 0].set(UnitOp.DROP),
        unit_count=action.unit_count.at[:, 0].set(1),
    )


def _controllers(count: int = 1):
    one = reset_controller_state_v1()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (count,) + value.shape), one
    )


def _assert_actor_market_parity(states, ledger, player: int = 0) -> None:
    action = ledger_action_v2(ledger, player)
    official = batched_project_action_phases(states, action, TABLES)
    np.testing.assert_array_equal(np.asarray(ledger.money_nominal), np.asarray(official.money[:, player]))
    np.testing.assert_array_equal(np.asarray(ledger.shed), np.asarray(official.shed[:, player]))
    np.testing.assert_array_equal(np.asarray(ledger.seeds), np.asarray(official.seeds[:, player]))
    np.testing.assert_array_equal(
        np.asarray(ledger.unit_inventory), np.asarray(official.unit_inventory[:, player])
    )
    np.testing.assert_array_equal(
        np.asarray(ledger.hires_today), np.asarray(official.hires_today[:, player])
    )
    np.testing.assert_array_equal(
        np.asarray(ledger.unlocked_count), np.asarray(official.unlocked_count[:, player])
    )
    np.testing.assert_array_equal(
        np.asarray(ledger.market_inventory), np.asarray(official.market_inventory)
    )
    np.testing.assert_array_equal(np.asarray(ledger.market_price), np.asarray(official.market_price))


def test_unit_phase_drop_opens_same_turn_sell_then_buy_then_hire() -> None:
    states = _states()._replace(
        money=_states().money.at[0, 0].set(100),
        unit_inventory=_states().unit_inventory.at[0, 0, 0, WOOL].set(3),
    )
    ledger, projected = close_unit_phase_v2(states, _drop_wool_action(), 0)
    assert int(ledger.phase[0]) == TurnPhaseV2.MARKET
    assert int(projected.shed[0, 0, WOOL]) == 3
    assert int(ledger.shed[0, WOOL]) == 3
    assert int(ledger.unit_inventory[0, 0, WOOL]) == 0

    ledger = apply_turn_market_order_v2(ledger, MarketOp.SELL, WOOL, 3, TABLES)
    cash_after_sell = int(ledger.money_nominal[0])
    assert cash_after_sell > 500
    ledger = apply_turn_market_order_v2(
        ledger, MarketOp.BUY_ANIMAL, COW_ITEM, 1, TABLES
    )
    ledger = apply_turn_market_order_v2(ledger, MarketOp.HIRE, -1, 1, TABLES)
    assert int(ledger.shed[0, COW_ITEM]) == 1
    assert int(ledger.hires_today[0]) == 1
    assert int(ledger.market_count[0]) == 3
    _assert_actor_market_parity(states, ledger)


def test_reverse_buy_before_sell_fails_like_official() -> None:
    states = _states()._replace(
        money=_states().money.at[0, 0].set(100),
        shed=_states().shed.at[0, 0, WOOL].set(3),
    )
    ledger, _ = close_unit_phase_v2(states, _actions(), 0)
    ledger = apply_turn_market_order_v2(
        ledger, MarketOp.BUY_ANIMAL, COW_ITEM, 1, TABLES
    )
    assert int(ledger.market_filled_amount[0, 0]) == 0
    ledger = apply_turn_market_order_v2(ledger, MarketOp.SELL, WOOL, 3, TABLES)
    assert int(ledger.shed[0, COW_ITEM]) == 0
    assert int(ledger.money_nominal[0]) > 500
    _assert_actor_market_parity(states, ledger)


def test_sell_releases_capacity_for_later_animal_buy() -> None:
    states = _states()
    shed = states.shed.at[0, 0, 0].set(SHED_CAPACITY - 1).at[0, 0, WOOL].set(1)
    states = states._replace(money=states.money.at[0, 0].set(1000), shed=shed)
    ledger, _ = close_unit_phase_v2(states, _actions(), 0)
    assert int(jnp.sum(ledger.shed[0])) == SHED_CAPACITY
    ledger = apply_turn_market_order_v2(ledger, MarketOp.SELL, WOOL, 1, TABLES)
    ledger = apply_turn_market_order_v2(
        ledger, MarketOp.BUY_ANIMAL, COW_ITEM, 1, TABLES
    )
    assert int(jnp.sum(ledger.shed[0])) == SHED_CAPACITY
    assert int(ledger.shed[0, COW_ITEM]) == 1
    _assert_actor_market_parity(states, ledger)


def test_repeated_hires_and_land_use_updated_prices() -> None:
    states = _states()._replace(money=_states().money.at[0, 0].set(10_000))
    ledger, _ = close_unit_phase_v2(states, _actions(), 0)
    for _ in range(3):
        ledger = apply_turn_market_order_v2(ledger, MarketOp.HIRE, -1, 1, TABLES)
    for _ in range(3):
        ledger = apply_turn_market_order_v2(ledger, MarketOp.BUY_LAND, -1, 1, TABLES)
    assert int(ledger.hires_today[0]) == 3
    assert int(ledger.unlocked_count[0]) == 4
    assert int(ledger.money_nominal[0]) == 10_000 - (1 + 1 + 2) - (1000 + 2000 + 4000)
    _assert_actor_market_parity(states, ledger)


def test_price_one_sale_does_not_add_market_supply() -> None:
    states = _states()
    inventory = states.market_inventory.at[0, WOOL].set(65_535)
    price = TABLES.market_price[WOOL, 65_535 - (-32_768)]
    assert int(price) == 1
    states = states._replace(
        shed=states.shed.at[0, 0, WOOL].set(2),
        market_inventory=inventory,
        market_price=states.market_price.at[0, WOOL].set(price),
    )
    ledger, _ = close_unit_phase_v2(states, _actions(), 0)
    ledger = apply_turn_market_order_v2(ledger, MarketOp.SELL, WOOL, 2, TABLES)
    assert int(ledger.market_inventory[0, WOOL]) == 65_535
    assert int(ledger.money_nominal[0]) == int(states.money[0, 0]) + 2
    _assert_actor_market_parity(states, ledger)


def test_1327_hinge_price_remains_int32_through_ordered_ledger() -> None:
    states = _states()
    price = TABLES.market_price[TOMATO, 0]
    assert int(price) > np.iinfo(np.int16).max
    states = states._replace(
        shed=states.shed.at[0, 0, TOMATO].set(1),
        market_inventory=states.market_inventory.at[0, TOMATO].set(
            MARKET_MIN_INVENTORY
        ),
        market_price=states.market_price.at[0, TOMATO].set(price),
    )
    ledger, _ = close_unit_phase_v2(states, _actions(), 0)
    assert ledger.market_price.dtype == jnp.int32
    assert int(ledger.market_price[0, TOMATO]) == int(price)
    ledger = apply_turn_market_order_v2(ledger, MarketOp.SELL, TOMATO, 1, TABLES)
    assert ledger.market_price.dtype == jnp.int32
    _assert_actor_market_parity(states, ledger)


def test_ten_order_cap_repeat_orders_and_closed_phase() -> None:
    states = _states()._replace(money=_states().money.at[0, 0].set(10_000))
    ledger, _ = close_unit_phase_v2(states, _actions(), 0)
    for _ in range(MAX_MARKET_ORDERS):
        ledger = apply_turn_market_order_v2(ledger, MarketOp.BUY_SEED, 0, 1, TABLES)
    assert int(ledger.market_count[0]) == MAX_MARKET_ORDERS
    assert int(ledger.seeds[0, 0]) == MAX_MARKET_ORDERS
    ledger = apply_turn_market_order_v2(ledger, MarketOp.BUY_SEED, 0, 1, TABLES)
    assert int(ledger.market_count[0]) == MAX_MARKET_ORDERS
    assert int(ledger.overflow_order_count[0]) == 1
    ledger = close_market_phase_v2(ledger)
    assert int(ledger.phase[0]) == TurnPhaseV2.CLOSED
    ledger = apply_turn_market_order_v2(ledger, MarketOp.BUY_SEED, 1, 1, TABLES)
    assert int(ledger.market_count[0]) == MAX_MARKET_ORDERS
    _assert_actor_market_parity(states, ledger)


def test_market_cannot_start_before_unit_phase_is_closed() -> None:
    states = _states()
    ledger = initialize_turn_ledger_v2(states, 0)
    ledger = apply_turn_market_order_v2(ledger, MarketOp.BUY_SEED, 0, 1, TABLES)
    assert int(ledger.market_count[0]) == 0
    assert int(ledger.seeds[0, 0]) == 0


def test_dynamic_refresh_opens_drop_sell_buy_hire_chain_without_combo_cards() -> None:
    states = _states()
    states = states._replace(
        money=states.money.at[0, 0].set(100),
        unit_inventory=states.unit_inventory.at[0, 0, 0, WOOL].set(3),
    )
    candidates = build_full_core_candidates_v1(
        states,
        _controllers(),
        TABLES,
        0,
        broad_replay_bc_candidate_program_v2(),
    )
    static_feasibility = evaluate_full_core_feasibility_v1(
        states, candidates, TABLES, 0
    )
    cow_slot = 4
    wool_sell_slot = 12 + WOOL
    assert not bool(candidates.present[0, cow_slot])
    assert not bool(candidates.present[0, wool_sell_slot])

    ledger, _ = close_unit_phase_v2(states, _drop_wool_action(), 0)
    dynamic = refresh_dynamic_market_candidates_v2(candidates, ledger, TABLES)
    feasibility = refresh_dynamic_market_feasibility_v2(
        static_feasibility, dynamic, ledger, TABLES
    )
    assert bool(dynamic.present[0, wool_sell_slot])
    assert int(dynamic.quantity[0, wool_sell_slot]) == 3
    assert bool(feasibility.legal_now[0, wool_sell_slot])
    assert not bool(dynamic.present[0, cow_slot])

    ledger = apply_dynamic_market_candidate_v2(
        ledger, dynamic, jnp.asarray((wool_sell_slot,), dtype=jnp.int16), TABLES
    )
    dynamic = refresh_dynamic_market_candidates_v2(candidates, ledger, TABLES)
    assert bool(dynamic.present[0, cow_slot])
    ledger = apply_dynamic_market_candidate_v2(
        ledger, dynamic, jnp.asarray((cow_slot,), dtype=jnp.int16), TABLES
    )
    dynamic = refresh_dynamic_market_candidates_v2(candidates, ledger, TABLES)
    assert bool(dynamic.present[0, 11])
    ledger = apply_dynamic_market_candidate_v2(
        ledger, dynamic, jnp.asarray((11,), dtype=jnp.int16), TABLES
    )
    assert int(ledger.market_count[0]) == 3
    assert int(ledger.shed[0, COW_ITEM]) == 1
    assert int(ledger.hires_today[0]) == 1
    _assert_actor_market_parity(states, ledger)


def test_dynamic_refresh_changes_only_market_schema_prefix() -> None:
    states = _states()
    candidates = build_full_core_candidates_v1(
        states,
        _controllers(),
        TABLES,
        0,
        broad_replay_bc_candidate_program_v2(),
    )
    ledger, _ = close_unit_phase_v2(states, _actions(), 0)
    dynamic = refresh_dynamic_market_candidates_v2(candidates, ledger, TABLES)
    for old, new in zip(candidates, dynamic, strict=True):
        np.testing.assert_array_equal(np.asarray(old[:, 21:]), np.asarray(new[:, 21:]))


def test_dynamic_stop_does_not_consume_slot_or_raise_invalid_counter() -> None:
    states = _states()
    candidates = build_full_core_candidates_v1(
        states,
        _controllers(),
        TABLES,
        0,
        broad_replay_bc_candidate_program_v2(),
    )
    ledger, _ = close_unit_phase_v2(states, _actions(), 0)
    dynamic = refresh_dynamic_market_candidates_v2(candidates, ledger, TABLES)
    ledger = apply_dynamic_market_candidate_v2(
        ledger, dynamic, jnp.asarray((-1,), dtype=jnp.int16), TABLES
    )
    assert int(ledger.market_count[0]) == 0
    assert int(ledger.invalid_order_count[0]) == 0


def test_ordered_replay_slot_mapping_is_parameterized_and_repeatable() -> None:
    slots = jnp.asarray((0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 20), dtype=jnp.int16)
    amounts = jnp.asarray((9, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), dtype=jnp.int16)
    valid, op, item, quantity = ordered_market_label_to_order_v2(slots, amounts)
    np.testing.assert_array_equal(np.asarray(valid), np.ones(len(slots), dtype=np.bool_))
    np.testing.assert_array_equal(
        np.asarray(op),
        np.asarray(
            (
                MarketOp.BUY_LAND,
                MarketOp.BUY_PRODUCT,
                MarketOp.BUY_PRODUCT,
                MarketOp.BUY_ANIMAL,
                MarketOp.BUY_ANIMAL,
                MarketOp.BUY_ANIMAL,
                MarketOp.BUY_SEED,
                MarketOp.BUY_SEED,
                MarketOp.HIRE,
                MarketOp.SELL,
                MarketOp.SELL,
            ),
            dtype=np.int8,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(item),
        np.asarray((-1, 8, 0, 9, 10, 11, 0, 4, -1, 0, 8), dtype=np.int8),
    )
    np.testing.assert_array_equal(
        np.asarray(quantity),
        np.asarray((1, 2, 3, 4, 5, 6, 7, 8, 1, 10, 11), dtype=np.int32),
    )


def test_dynamic_buy_product_cash_required_uses_exact_prefix_quote() -> None:
    states = _states()._replace(money=_states().money.at[0, 0].set(100_000))
    candidates = build_full_core_candidates_v1(
        states,
        _controllers(),
        TABLES,
        0,
        broad_replay_bc_candidate_program_v2(),
    )
    candidates = candidates._replace(quantity=candidates.quantity.at[0, 2].set(7))
    static = evaluate_full_core_feasibility_v1(states, candidates, TABLES, 0)
    ledger, _ = close_unit_phase_v2(states, _actions(), 0)
    dynamic = refresh_dynamic_market_candidates_v2(candidates, ledger, TABLES)
    feasibility = refresh_dynamic_market_feasibility_v2(
        static, dynamic, ledger, TABLES
    )
    bought = apply_turn_market_order_v2(
        ledger, MarketOp.BUY_PRODUCT, 0, 7, TABLES
    )
    actual_cost = int(ledger.money_nominal[0] - bought.money_nominal[0])
    assert int(feasibility.cash_required[0, 2]) == actual_cost


def test_vectorized_market_settlement_matches_official_on_random_order_books() -> None:
    """Exercise all market op families and LUT boundaries in one batched diff."""

    batch_size = 64
    rng = np.random.default_rng(20260813)
    states = _states(batch_size)

    money = rng.integers(0, 20_001, size=batch_size, dtype=np.int32)
    shed = np.zeros((batch_size, NUM_SHED_ITEMS), dtype=np.int16)
    shed[:, :NUM_PRODUCTS] = rng.integers(
        0, 6, size=(batch_size, NUM_PRODUCTS), dtype=np.int16
    )
    shed[:, NUM_PRODUCTS:] = rng.integers(
        0, 3, size=(batch_size, NUM_ANIMALS), dtype=np.int16
    )
    seeds = rng.integers(
        0, 11, size=(batch_size, NUM_CROPS), dtype=np.int16
    )
    market_inventory = rng.integers(
        MARKET_MIN_INVENTORY,
        MARKET_MAX_INVENTORY + 1,
        size=(batch_size, NUM_PRODUCTS),
        dtype=np.int32,
    )

    # Pin explicit edge cases that pure random sampling is unlikely to hit.
    floor_inventory = (
        MARKET_MIN_INVENTORY + np.asarray(TABLES.market_first_floor_index)
    )
    wool_floor = int(floor_inventory[WOOL])
    market_inventory[0, WOOL] = wool_floor - 2  # crosses the price-1 floor
    market_inventory[1, 0] = MARKET_MIN_INVENTORY + 1  # BUY clips below LUT
    market_inventory[2, 8] = MARKET_MAX_INVENTORY  # high LUT boundary
    market_inventory[3, 0] = MARKET_MAX_INVENTORY  # SELL clips above LUT
    market_inventory[4, WOOL] = wool_floor  # already at price 1
    shed[0, WOOL] = 7
    shed[3, 0] = 7
    shed[4, WOOL] = 7
    shed[1:3] = 0
    money[1:3] = 100_000

    price_lut = np.asarray(TABLES.market_price)
    price_index = np.clip(
        market_inventory - MARKET_MIN_INVENTORY, 0, price_lut.shape[1] - 1
    )
    market_price = price_lut[np.arange(NUM_PRODUCTS)[None, :], price_index]
    states = states._replace(
        money=states.money.at[:, 0].set(jnp.asarray(money)),
        shed=states.shed.at[:, 0].set(jnp.asarray(shed, dtype=states.shed.dtype)),
        seeds=states.seeds.at[:, 0].set(
            jnp.asarray(seeds, dtype=states.seeds.dtype)
        ),
        hires_today=states.hires_today.at[:, 0].set(
            jnp.asarray(rng.integers(0, 6, size=batch_size), dtype=jnp.int8)
        ),
        unlocked_count=states.unlocked_count.at[:, 0].set(
            jnp.asarray(rng.integers(1, 5, size=batch_size), dtype=jnp.int8)
        ),
        market_inventory=jnp.asarray(
            market_inventory, dtype=states.market_inventory.dtype
        ),
        market_price=jnp.asarray(market_price, dtype=states.market_price.dtype),
    )

    op_choices = np.asarray(
        (
            MarketOp.SELL,
            MarketOp.BUY_PRODUCT,
            MarketOp.BUY_SEED,
            MarketOp.BUY_ANIMAL,
            MarketOp.HIRE,
            MarketOp.BUY_LAND,
        ),
        dtype=np.int8,
    )
    ops = rng.choice(op_choices, size=(MAX_MARKET_ORDERS, batch_size))
    items = np.full((MAX_MARKET_ORDERS, batch_size), -1, dtype=np.int8)
    amounts = np.ones((MAX_MARKET_ORDERS, batch_size), dtype=np.int32)
    for slot in range(MAX_MARKET_ORDERS):
        for row in range(batch_size):
            op = int(ops[slot, row])
            if op == MarketOp.SELL:
                items[slot, row] = rng.integers(0, NUM_PRODUCTS)
                amounts[slot, row] = rng.integers(1, 26)
            elif op == MarketOp.BUY_PRODUCT:
                items[slot, row] = int(rng.choice((0, 8)))
                amounts[slot, row] = rng.integers(1, 26)
            elif op == MarketOp.BUY_SEED:
                items[slot, row] = rng.integers(0, NUM_CROPS)
                amounts[slot, row] = rng.integers(1, 26)
            elif op == MarketOp.BUY_ANIMAL:
                items[slot, row] = NUM_PRODUCTS + rng.integers(0, NUM_ANIMALS)
                amounts[slot, row] = rng.integers(1, 7)

    ops[0, :5] = np.asarray(
        (
            MarketOp.SELL,
            MarketOp.BUY_PRODUCT,
            MarketOp.BUY_PRODUCT,
            MarketOp.SELL,
            MarketOp.SELL,
        ),
        dtype=np.int8,
    )
    items[0, :5] = np.asarray((WOOL, 0, 8, 0, WOOL), dtype=np.int8)
    amounts[0, :5] = 7

    ledger, _ = close_unit_phase_v2(states, _actions(batch_size), 0)
    for slot in range(MAX_MARKET_ORDERS):
        ledger = apply_turn_market_order_v2(
            ledger,
            jnp.asarray(ops[slot]),
            jnp.asarray(items[slot]),
            jnp.asarray(amounts[slot]),
            TABLES,
        )

    _assert_actor_market_parity(states, ledger)
    assert int(ledger.invalid_order_count.sum()) == 0
    assert int(ledger.overflow_order_count.sum()) == 0


def test_compact_market_matrix_matches_legacy_96_slot_refresh_and_apply() -> None:
    batch_size = 32
    rng = np.random.default_rng(20260814)
    states = _states(batch_size)
    states = states._replace(
        money=states.money.at[:, 0].set(
            jnp.asarray(rng.integers(0, 20_000, batch_size), dtype=jnp.int32)
        ),
        shed=states.shed.at[:, 0].set(
            jnp.asarray(
                rng.integers(0, 5, (batch_size, NUM_SHED_ITEMS)),
                dtype=states.shed.dtype,
            )
        ),
        hires_today=states.hires_today.at[:, 0].set(
            jnp.asarray(rng.integers(0, 6, batch_size), dtype=jnp.int8)
        ),
        unlocked_count=states.unlocked_count.at[:, 0].set(
            jnp.asarray(rng.integers(1, 5, batch_size), dtype=jnp.int8)
        ),
    )
    candidates = build_full_core_candidates_v1(
        states,
        _controllers(batch_size),
        TABLES,
        0,
        broad_replay_bc_candidate_program_v2(),
    )
    candidates = candidates._replace(
        quantity=candidates.quantity.at[:, :21].set(
            jnp.asarray(rng.integers(1, 26, (batch_size, 21)), dtype=jnp.int16)
        )
    )
    static = evaluate_full_core_feasibility_v1(states, candidates, TABLES, 0)
    ledger, _ = close_unit_phase_v2(states, _actions(batch_size), 0)
    legacy = refresh_dynamic_market_candidates_v2(candidates, ledger, TABLES)
    legacy_feasibility = refresh_dynamic_market_feasibility_v2(
        static, legacy, ledger, TABLES
    )
    compact = refresh_dynamic_market_matrix_v2(candidates.quantity, ledger, TABLES)
    np.testing.assert_array_equal(
        np.asarray(compact.quantity), np.asarray(legacy.quantity[:, :21])
    )
    np.testing.assert_array_equal(
        np.asarray(compact.present), np.asarray(legacy.present[:, :21])
    )
    np.testing.assert_array_equal(
        np.asarray(compact.cash_required),
        np.asarray(legacy_feasibility.cash_required[:, :21]),
    )

    selected = jnp.asarray(rng.integers(-1, 21, batch_size), dtype=jnp.int16)
    legacy_applied = apply_dynamic_market_candidate_v2(
        ledger, legacy, selected, TABLES
    )
    compact_applied = apply_dynamic_market_matrix_slot_v2(
        ledger, compact, selected, TABLES
    )
    for legacy_value, compact_value in zip(
        jax.tree.leaves(legacy_applied),
        jax.tree.leaves(compact_applied),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(compact_value), np.asarray(legacy_value)
        )
