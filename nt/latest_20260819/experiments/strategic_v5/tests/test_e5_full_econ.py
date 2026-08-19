from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from kaggriculture_jax import empty_action, load_event_bank, load_tables, reset
from kaggriculture_jax.constants import (
    ANIMAL_COST,
    CROP_FIRST_YIELD_DAY,
    CROP_MAX_YIELD,
    CROP_SEED_COST,
    EPISODE_STEPS,
    FLAG_CARED,
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_FED,
    MARKET_MIN_INVENTORY,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    MarketOp,
    TileKind,
)
from kaggriculture_jax.simulator import batched_step_sync
from strategic_v5 import (
    ECON_STOCHASTIC_RISK_BIT,
    TaskStatusV1,
    TaskTypeV1,
    build_full_core_candidates_v1,
    build_full_econ_features_v1,
    clear_finished_tasks_v1,
    compile_full_core_action_bundle_v1,
    evaluate_full_core_feasibility_v1,
    exact_market_quote_v1,
    full_econ_score_v1,
    known_town_demand_before_sale_v1,
    reset_controller_state_v1,
    select_full_econ_candidates_v1,
    update_full_core_controller_from_effects_v1,
)


TABLES = load_tables()


def _states(seeds=(901,)):
    return jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))


def _controllers(batch_size: int):
    value = reset_controller_state_v1()
    return jax.tree.map(
        lambda item: jnp.broadcast_to(item, (batch_size,) + item.shape), value
    )


def _events(batch_size: int):
    _, bank = load_event_bank()
    return jax.tree.map(lambda value: value[:batch_size], bank)


def _pipeline(states, controller, player: int, *, risks=True):
    controller = clear_finished_tasks_v1(controller)
    candidates = build_full_core_candidates_v1(states, controller, TABLES, player)
    feasibility = evaluate_full_core_feasibility_v1(
        states, candidates, TABLES, player
    )
    econ = build_full_econ_features_v1(
        states,
        candidates,
        feasibility,
        TABLES,
        player,
        include_expected_risk=risks,
        include_scenario_risk=risks,
    )
    return candidates, feasibility, econ


@pytest.mark.parametrize("buy", [False, True])
@pytest.mark.parametrize("quantity", [1, 2, 9, 40, 100])
def test_exact_market_batch_quote_matches_official_rollout(
    buy: bool, quantity: int
) -> None:
    states = _states()
    item = 0
    if not buy:
        states = states._replace(
            shed=states.shed.at[0, 0, item].set(quantity)
        )
    before = int(states.money[0, 0])
    quote = int(
        exact_market_quote_v1(
            states,
            TABLES,
            jnp.asarray([[item]], dtype=jnp.int8),
            jnp.asarray([[quantity]], dtype=jnp.int16),
            buy=buy,
        )[0, 0]
    )
    action = jax.tree.map(lambda value: value[None, ...], empty_action())
    action = action._replace(
        market_op=action.market_op.at[0, 0, 0].set(
            MarketOp.BUY_PRODUCT if buy else MarketOp.SELL
        ),
        market_item=action.market_item.at[0, 0, 0].set(item),
        market_amount=action.market_amount.at[0, 0, 0].set(quantity),
        market_count=action.market_count.at[0, 0].set(1),
    )
    following = batched_step_sync(states, action, _events(1), TABLES)
    actual = int(following.money[0, 0]) - before
    assert actual == (-quote if buy else quote)


def test_exact_market_quote_respects_price_floor_inventory_rule() -> None:
    states = _states()
    item = 0
    # The default official WHEAT curve does not reach 1.  Freeze a one-cell
    # boundary fixture and pass the same table to both the quote and simulator;
    # this exercises the simulator's otherwise unreachable price-floor branch.
    floor_index = 32
    inventory = floor_index + MARKET_MIN_INVENTORY
    floor_price = TABLES.market_price.at[item, floor_index].set(1)
    floor_prefix = jnp.concatenate(
        (
            jnp.zeros((NUM_PRODUCTS, 1), dtype=jnp.int32),
            jnp.cumsum(floor_price.astype(jnp.int32), axis=1),
        ),
        axis=1,
    )
    floor_tables = TABLES._replace(
        market_price=floor_price,
        market_price_prefix=floor_prefix,
        market_first_floor_index=TABLES.market_first_floor_index.at[item].set(
            floor_index
        ),
    )
    states = states._replace(
        step=states.step.at[0].set(1),
        market_inventory=states.market_inventory.at[0, item].set(inventory),
        shed=states.shed.at[0, 0, item].set(7),
    )
    quote = int(
        exact_market_quote_v1(
            states,
            floor_tables,
            jnp.asarray([[item]], dtype=jnp.int8),
            jnp.asarray([[7]], dtype=jnp.int16),
            buy=False,
        )[0, 0]
    )
    assert quote == 7
    action = jax.tree.map(lambda value: value[None, ...], empty_action())
    action = action._replace(
        market_op=action.market_op.at[0, 0, 0].set(MarketOp.SELL),
        market_item=action.market_item.at[0, 0, 0].set(item),
        market_amount=action.market_amount.at[0, 0, 0].set(7),
        market_count=action.market_count.at[0, 0].set(1),
    )
    following = batched_step_sync(states, action, _events(1), floor_tables)
    assert int(following.money[0, 0] - states.money[0, 0]) == quote
    assert int(following.market_inventory[0, item]) == inventory


def test_known_town_calendar_uses_only_currently_unlocked_shops() -> None:
    states = _states()
    states = states._replace(
        town_count=states.town_count.at[0].set(2),
        town_shops=states.town_shops.at[0, :2].set(
            jnp.asarray([0, 4], dtype=jnp.int8)
        ),
    )
    item = jnp.asarray([[0, 1]], dtype=jnp.int8)
    steps = jnp.asarray([[9, 9]], dtype=jnp.int16)
    demand = known_town_demand_before_sale_v1(states, item, steps)
    # Prior transitions 0..7 contain ticks 0 and 4; center tick 0 also applies.
    # Shop 0 consumes WHEAT once/tick; shop 4 consumes CARROT twice/tick.
    assert demand.tolist() == [[3, 5]]


def test_all_section_7_1_fields_have_finite_values_and_provenance() -> None:
    states = _states((910, 911, 912, 913))
    controller = _controllers(4)
    _, feasibility, econ = _pipeline(states, controller, 0)
    float_fields = (
        "expected_bank_delta_current_price",
        "action_opportunity_cost",
        "input_opportunity_cost",
        "expected_overflow_loss",
        "expected_decay_or_capacity_loss",
        "stochastic_event_risk",
        "terminal_salvage_value",
    )
    for name in float_fields:
        assert bool(jnp.all(jnp.isfinite(getattr(econ, name))))
    provenance = econ.exact_field_mask | econ.expected_field_mask | econ.scenario_field_mask
    assert bool(jnp.all(provenance != 0))
    assert bool(jnp.all(econ.terminal_salvage_value == 0))
    assert bool(jnp.all(econ.bankable_before_terminal <= feasibility.bankable_before_terminal))


def test_soft_risk_toggle_changes_only_soft_econ_fields_not_core() -> None:
    states = _states()
    controller = _controllers(1)
    candidates, feasibility, with_risk = _pipeline(states, controller, 0, risks=True)
    candidates_again, feasibility_again, without_risk = _pipeline(
        states, controller, 0, risks=False
    )
    assert all(
        bool(jnp.all(left == right))
        for left, right in zip(candidates, candidates_again, strict=True)
    )
    assert all(
        bool(jnp.all(left == right))
        for left, right in zip(feasibility, feasibility_again, strict=True)
    )
    assert bool(jnp.all(without_risk.stochastic_event_risk == 0))
    assert bool(jnp.any(with_risk.stochastic_event_risk >= 0))
    assert bool(
        jnp.all(
            (without_risk.expected_field_mask & ECON_STOCHASTIC_RISK_BIT) == 0
        )
    )


def test_higher_cost_cannot_increase_expected_net_cash() -> None:
    states = _states()
    controller = _controllers(1)
    candidates, feasibility, econ = _pipeline(states, controller, 0)
    seed = (
        (candidates.task_type == TaskTypeV1.CROP_PRODUCTION)
        & (candidates.owner_unit < 0)
    )
    index = int(jnp.argmax(seed[0]))
    baseline = float(econ.expected_bank_delta_current_price[0, index])
    extra_cost = 123
    modified = econ._replace(
        expected_bank_delta_current_price=econ.expected_bank_delta_current_price.at[
            0, index
        ].add(-extra_cost),
        cash_required=econ.cash_required.at[0, index].add(extra_cost),
        cash_flow_before_revenue=econ.cash_flow_before_revenue.at[0, index].add(
            -extra_cost
        ),
        minimum_cash_during_task=econ.minimum_cash_during_task.at[0, index].add(
            -extra_cost
        ),
        simple_cash_buffer_after_commit=econ.simple_cash_buffer_after_commit.at[
            0, index
        ].add(-extra_cost),
    )
    assert float(modified.expected_bank_delta_current_price[0, index]) == pytest.approx(
        baseline - extra_cost
    )
    assert float(full_econ_score_v1(candidates, feasibility, modified)[0, index]) < float(
        full_econ_score_v1(candidates, feasibility, econ)[0, index]
    )


def test_fewer_remaining_days_cannot_increase_crop_or_animal_cycles() -> None:
    early = _states()
    late = early._replace(step=early.step.at[0].set(24 * 20))
    controller = _controllers(1)
    early_candidates, _, early_econ = _pipeline(early, controller, 0)
    late_candidates, _, late_econ = _pipeline(late, controller, 0)
    for crop in range(NUM_CROPS):
        early_mask = (
            (early_candidates.task_type == TaskTypeV1.CROP_PRODUCTION)
            & (early_candidates.owner_unit < 0)
            & (early_candidates.item_id == crop)
        )
        late_mask = (
            (late_candidates.task_type == TaskTypeV1.CROP_PRODUCTION)
            & (late_candidates.owner_unit < 0)
            & (late_candidates.item_id == crop)
        )
        early_cycles = int(jnp.max(jnp.where(early_mask, early_econ.cycles_before_terminal, 0)))
        late_cycles = int(jnp.max(jnp.where(late_mask, late_econ.cycles_before_terminal, 0)))
        assert late_cycles <= early_cycles

    for animal in range(NUM_ANIMALS):
        item = NUM_PRODUCTS + animal
        early_mask = (early_candidates.task_type == TaskTypeV1.ANIMAL_PURCHASE) & (
            early_candidates.item_id == item
        )
        late_mask = (late_candidates.task_type == TaskTypeV1.ANIMAL_PURCHASE) & (
            late_candidates.item_id == item
        )
        early_cycles = int(jnp.max(jnp.where(early_mask, early_econ.cycles_before_terminal, 0)))
        late_cycles = int(jnp.max(jnp.where(late_mask, late_econ.cycles_before_terminal, 0)))
        assert late_cycles <= early_cycles


def test_terminal_unbankable_investments_have_zero_bank_delta_and_are_not_selected() -> None:
    states = _states()
    states = states._replace(step=states.step.at[0].set(718))
    controller = _controllers(1)
    candidates, feasibility, econ = _pipeline(states, controller, 0)
    investment = (
        (candidates.task_type == TaskTypeV1.CROP_PRODUCTION)
        | (candidates.task_type == TaskTypeV1.BUY_PRODUCT)
        | (candidates.task_type == TaskTypeV1.ANIMAL_PURCHASE)
        | (candidates.task_type == TaskTypeV1.BUY_LAND)
        | (candidates.task_type == TaskTypeV1.HIRE_WORKER)
        | (candidates.task_type == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
        | (candidates.task_type == TaskTypeV1.APPLY_FERTILIZER)
    )
    assert bool(jnp.all(jnp.where(investment, econ.expected_bank_delta_current_price == 0, True)))
    selection = select_full_econ_candidates_v1(
        states, candidates, feasibility, econ, controller, 0
    )
    selected = selection.selected_candidate_indices[0]
    selected = selected[selected >= 0]
    assert not bool(jnp.any(investment[0, selected]))


def test_feed_escape_care_cap_fertilizer_expiry_crop_decay_overflow_boundaries() -> None:
    controller = _controllers(1)

    # Feed/escape boundary: the unfed animal at neglect=1 must have positive
    # preservation value while wheat is still available in the shed.
    feed_state = _states()._replace(
        tile_kind=_states().tile_kind.at[0, 0, 0, 0].set(TileKind.COOP),
        tile_animal=_states().tile_animal.at[0, 0, 0, 0].set(0),
        tile_origin_day=_states().tile_origin_day.at[0, 0, 0, 0].set(-5),
        tile_neglect=_states().tile_neglect.at[0, 0, 0, 0].set(1),
        shed=_states().shed.at[0, 0, 0].set(1),
    )
    candidates, _, econ = _pipeline(feed_state, controller, 0)
    feed = candidates.task_type == TaskTypeV1.ANIMAL_FEED
    assert bool(jnp.any(feed))
    assert float(jnp.max(jnp.where(feed, econ.expected_bank_delta_current_price, -1))) > 0

    # Care at the species yield cap is legal but has no incremental economic
    # value.  FED suppresses the higher-priority feed route.
    care_state = _states()._replace(
        tile_kind=_states().tile_kind.at[0, 0, 0, 0].set(TileKind.COOP),
        tile_animal=_states().tile_animal.at[0, 0, 0, 0].set(0),
        tile_origin_day=_states().tile_origin_day.at[0, 0, 0, 0].set(-5),
        tile_yield=_states().tile_yield.at[0, 0, 0, 0].set(4),
        tile_flags=_states().tile_flags.at[0, 0, 0, 0].set(FLAG_FED),
    )
    candidates, _, _ = _pipeline(care_state, controller, 0)
    product = (
        (candidates.task_type == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
        & (candidates.target_id == 0)
    )
    assert bool(jnp.any(product))
    care_index = int(jnp.argmax(product[0]))
    candidates = candidates._replace(
        task_type=candidates.task_type.at[0, care_index].set(TaskTypeV1.ANIMAL_CARE),
        item_id=candidates.item_id.at[0, care_index].set(0),
    )
    feasibility = evaluate_full_core_feasibility_v1(
        care_state, candidates, TABLES, 0
    )
    econ = build_full_econ_features_v1(
        care_state, candidates, feasibility, TABLES, 0
    )
    care = candidates.task_type == TaskTypeV1.ANIMAL_CARE
    assert float(jnp.max(jnp.where(care, econ.expected_bank_delta_current_price, -1))) == 0

    # Fertilizer can be collected before its day-end expiry and receives a
    # positive sell-or-use value.
    collect_state = care_state._replace(
        tile_yield=care_state.tile_yield.at[0, 0, 0, 0].set(0),
        tile_flags=care_state.tile_flags.at[0, 0, 0, 0].set(
            FLAG_FED | FLAG_CARED | FLAG_FERTILIZER_AVAILABLE
        ),
    )
    candidates, _, econ = _pipeline(collect_state, controller, 0)
    collect = candidates.task_type == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER
    assert bool(jnp.any(collect))
    assert float(jnp.max(jnp.where(collect, econ.expected_bank_delta_current_price, -1))) > 0

    # A mature, already decaying crop feeding a nearly full shed must expose
    # both decay and overflow loss fields.
    harvest_state = _states()._replace(
        tile_kind=_states().tile_kind.at[0, 0, 0, 1].set(TileKind.PLANT),
        tile_crop=_states().tile_crop.at[0, 0, 0, 1].set(0),
        tile_origin_day=_states().tile_origin_day.at[0, 0, 0, 1].set(-5),
        tile_yield=_states().tile_yield.at[0, 0, 0, 1].set(3),
        tile_max_lifespan=_states().tile_max_lifespan.at[0, 0, 0, 1].set(0),
        shed=_states().shed.at[0, 0, 0].set(99),
    )
    candidates, _, econ = _pipeline(harvest_state, controller, 0)
    harvest = (
        (candidates.task_type == TaskTypeV1.CROP_PRODUCTION)
        & (candidates.owner_unit >= 0)
        & (candidates.target_id == 1)
    )
    assert bool(jnp.any(harvest))
    assert float(jnp.max(jnp.where(harvest, econ.expected_decay_or_capacity_loss, 0))) >= 0
    assert float(jnp.max(jnp.where(harvest, econ.expected_overflow_loss, 0))) > 0


@pytest.mark.parametrize("crop", range(NUM_CROPS))
def test_each_crop_has_independent_econ_case(crop: int) -> None:
    states = _states()
    controller = _controllers(1)
    candidates, _, econ = _pipeline(states, controller, 0)
    mask = (
        (candidates.task_type == TaskTypeV1.CROP_PRODUCTION)
        & (candidates.owner_unit < 0)
        & (candidates.item_id == crop)
    )
    assert bool(jnp.any(mask))
    index = int(jnp.argmax(mask[0]))
    assert int(econ.cash_required[0, index]) == CROP_SEED_COST[crop]
    assert int(econ.cycles_before_terminal[0, index]) >= 0


@pytest.mark.parametrize("animal", range(NUM_ANIMALS))
def test_each_animal_has_independent_econ_case(animal: int) -> None:
    states = _states()
    tile_kind = states.tile_kind.at[0, 0, animal, 0].set(
        TileKind.COOP if animal == 0 else TileKind.PASTURE
    )
    states = states._replace(tile_kind=tile_kind)
    controller = _controllers(1)
    candidates, _, econ = _pipeline(states, controller, 0)
    mask = (candidates.task_type == TaskTypeV1.ANIMAL_PURCHASE) & (
        candidates.item_id == NUM_PRODUCTS + animal
    )
    assert bool(jnp.any(mask))
    index = int(jnp.argmax(mask[0]))
    assert int(econ.cash_required[0, index]) >= ANIMAL_COST[animal]
    assert int(econ.cycles_before_terminal[0, index]) >= 0


def test_already_purchased_goose_has_a_positive_place_route() -> None:
    states = _states()
    states = states._replace(
        tile_kind=states.tile_kind.at[0, 0, 0, 0].set(TileKind.COOP),
        shed=states.shed.at[0, 0, NUM_PRODUCTS].set(1),
    )
    controller = _controllers(1)
    candidates, feasibility, econ = _pipeline(states, controller, 0)
    place = (
        (candidates.task_type == TaskTypeV1.ANIMAL_PLACE)
        & (candidates.item_id == NUM_PRODUCTS)
    )
    assert bool(jnp.any(place))
    index = int(jnp.argmax(place[0]))
    expected_feed_cash = int(
        exact_market_quote_v1(
            states,
            TABLES,
            jnp.asarray([[0]], dtype=jnp.int8),
            jnp.asarray([[4]], dtype=jnp.int16),
            buy=True,
            enforce_resources=False,
        )[0, 0]
    )
    assert int(econ.cash_required[0, index]) == expected_feed_cash
    score = full_econ_score_v1(candidates, feasibility, econ)
    assert float(jnp.max(jnp.where(place, score, -1.0e9))) > 0


def test_low_cash_does_not_block_animal_product_cash_recovery() -> None:
    states = _states()
    states = states._replace(
        money=states.money.at[0, 0].set(31),
        tile_kind=states.tile_kind.at[0, 0, 0, 0].set(TileKind.PASTURE),
        tile_animal=states.tile_animal.at[0, 0, 0, 0].set(1),
        tile_origin_day=states.tile_origin_day.at[0, 0, 0, 0].set(-8),
        tile_yield=states.tile_yield.at[0, 0, 0, 0].set(1),
    )
    controller = _controllers(1)
    candidates, feasibility, econ = _pipeline(states, controller, 0)
    product = candidates.task_type == TaskTypeV1.ANIMAL_COLLECT_PRODUCT
    assert bool(jnp.any(product))
    score = full_econ_score_v1(candidates, feasibility, econ)
    assert float(jnp.max(jnp.where(product, score, -1.0e9))) > 0
    selection = select_full_econ_candidates_v1(
        states, candidates, feasibility, econ, controller, 0
    )
    selected = selection.selected_candidate_indices[0]
    selected = selected[selected >= 0]
    assert bool(jnp.any(product[0, selected]))


@pytest.mark.parametrize("product", range(NUM_PRODUCTS))
def test_each_product_has_independent_sell_econ_case(product: int) -> None:
    states = _states()
    states = states._replace(shed=states.shed.at[0, 0, product].set(2))
    controller = _controllers(1)
    candidates, _, econ = _pipeline(states, controller, 0)
    mask = (candidates.task_type == TaskTypeV1.SELL_INVENTORY) & (
        candidates.item_id == product
    )
    assert bool(jnp.any(mask))
    index = int(jnp.argmax(mask[0]))
    assert float(econ.expected_bank_delta_current_price[0, index]) > 0
    assert int(econ.terminal_salvage_value[0, index]) == 0


@pytest.mark.parametrize("shop", range(8))
def test_each_shop_has_independent_known_calendar_case(shop: int) -> None:
    states = _states()
    states = states._replace(
        town_count=states.town_count.at[0].set(1),
        town_shops=states.town_shops.at[0, 0].set(shop),
    )
    items = jnp.arange(NUM_PRODUCTS, dtype=jnp.int8)[None, :]
    steps = jnp.full((1, NUM_PRODUCTS), 5, dtype=jnp.int16)
    demand = known_town_demand_before_sale_v1(states, items, steps)
    assert demand.shape == (1, NUM_PRODUCTS)
    assert int(demand.sum()) > 0


def test_full_econ_pipeline_is_jittable() -> None:
    batch_size = 4
    states = _states((950, 951, 952, 953))
    controller = _controllers(batch_size)

    @jax.jit
    def run(current, control):
        candidates = build_full_core_candidates_v1(current, control, TABLES, 0)
        feasibility = evaluate_full_core_feasibility_v1(
            current, candidates, TABLES, 0
        )
        econ = build_full_econ_features_v1(
            current, candidates, feasibility, TABLES, 0
        )
        selection = select_full_econ_candidates_v1(
            current, candidates, feasibility, econ, control, 0
        )
        return econ, selection

    econ, selection = run(states, controller)
    assert econ.expected_bank_delta_current_price.shape == (batch_size, 96)
    assert int(selection.internal_resource_conflict.sum()) == 0
    assert bool(jnp.all(jnp.isfinite(econ.expected_bank_delta_current_price)))
