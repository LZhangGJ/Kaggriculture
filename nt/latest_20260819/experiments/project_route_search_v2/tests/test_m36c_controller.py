from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import FLAG_FERTILIZER_AVAILABLE, TileKind, UnitOp
from kaggriculture_jax.state import load_tables, reset
from strategic_v5.constants import TaskPhaseV1, TaskStatusV1, TaskTypeV1
from strategic_v5.e4_executor import compile_full_core_player_action_v1

from project_route_search_v2.lifecycle import initialize_project_controller_v2
from project_route_search_v2.m35_controller import (
    ensure_m35_projects_v2,
    materialize_m35_market_tasks_v2,
)
from project_route_search_v2.m35_genome import default_m35_farm_genome_v2
from project_route_search_v2.m36_calendar import empty_route_calendar_v3
from project_route_search_v2.m36_controller import (
    continue_m36_fertilizer_batch_after_effects_v3,
    materialize_m36_fertilizer_ledger_v3,
    materialize_m36_reactive_sales_v3,
    materialize_m37_emergency_feed_buy_v1,
)
from project_route_search_v2.m36_genome_adapter import calendar_step_genome_v3
from project_route_search_v2.m3_constants import M3AnimalFertilizerPolicyV2
from project_route_search_v2.m3_controller import (
    clear_invalidated_m3_tasks_v2,
    derive_m3_commitment_ledger_v2,
)


def test_reactive_sales_preserve_live_animal_feed_commitment():
    states = jax.vmap(reset)(jnp.asarray((43,), dtype=jnp.int32))
    animals = states.tile_animal.at[0, 0, 0, 0].set(jnp.int8(1))
    animals = animals.at[0, 0, 0, 1].set(jnp.int8(2))
    shed = states.shed.at[0, 0, 0].set(jnp.int16(5))
    states = states._replace(tile_animal=animals, shed=shed)
    controller = initialize_project_controller_v2(states, 0)
    calendar = empty_route_calendar_v3(1)
    genome = default_m35_farm_genome_v2(1)

    ledger = derive_m3_commitment_ledger_v2(
        states, controller, genome.animal, 0
    )
    committed = int(np.asarray(ledger.planned_feed_horizon).sum())
    updated = materialize_m36_reactive_sales_v3(
        states, controller, calendar, genome, 0
    )
    market = updated.market_tasks
    wheat_sell = (
        (np.asarray(market.status[0]) == int(TaskStatusV1.ACTIVE))
        & (np.asarray(market.task_type[0]) == int(TaskTypeV1.SELL_INVENTORY))
        & (np.asarray(market.item_id[0]) == 0)
    )
    sold = int(np.asarray(market.quantity[0])[wheat_sell].sum())

    assert committed > 0
    assert 5 - sold >= min(committed, 5)


def test_emergency_feed_buy_reacts_to_realized_cash_before_next_full_plan():
    states = jax.vmap(reset)(jnp.asarray((45,), dtype=jnp.int32))
    states = states._replace(
        step=jnp.asarray((76,), dtype=jnp.int16),
        money=states.money.at[0, 0].set(jnp.int32(106)),
        tile_kind=states.tile_kind.at[0, 0, 0, 0].set(
            jnp.int8(TileKind.PASTURE)
        ),
        tile_animal=states.tile_animal.at[0, 0, 0, 0].set(jnp.int8(2)),
        tile_neglect=states.tile_neglect.at[0, 0, 0, 0].set(jnp.int8(1)),
    )
    controller = initialize_project_controller_v2(states, 0)
    genome = default_m35_farm_genome_v2(1)
    planned = materialize_m37_emergency_feed_buy_v1(
        states, controller, genome, 0
    )
    market = planned.market_tasks
    wheat_buy = (
        (market.status[0] == TaskStatusV1.ACTIVE)
        & (market.task_type[0] == TaskTypeV1.BUY_PRODUCT)
        & (market.item_id[0] == 0)
    )

    assert int(jnp.sum(wheat_buy)) == 1
    assert int(market.quantity[0, jnp.argmax(wheat_buy)]) == 1


def test_emergency_feed_buy_does_not_create_optional_feed_demand():
    states = jax.vmap(reset)(jnp.asarray((46,), dtype=jnp.int32))
    states = states._replace(
        money=states.money.at[0, 0].set(jnp.int32(100_000)),
        tile_kind=states.tile_kind.at[0, 0, 0, 0].set(
            jnp.int8(TileKind.PASTURE)
        ),
        tile_animal=states.tile_animal.at[0, 0, 0, 0].set(jnp.int8(2)),
    )
    controller = initialize_project_controller_v2(states, 0)
    planned = materialize_m37_emergency_feed_buy_v1(
        states, controller, default_m35_farm_genome_v2(1), 0
    )
    assert not bool(
        jnp.any(
            (planned.market_tasks.status[0] == TaskStatusV1.ACTIVE)
            & (planned.market_tasks.task_type[0] == TaskTypeV1.BUY_PRODUCT)
            & (planned.market_tasks.item_id[0] == 0)
        )
    )


def test_random_weed_invalidates_reserved_animal_build_target():
    states = jax.vmap(reset)(jnp.asarray((47,), dtype=jnp.int32))
    kind = states.tile_kind.at[0, 0, 0, 0].set(jnp.int8(TileKind.WEED))
    states = states._replace(tile_kind=kind)
    controller = initialize_project_controller_v2(states, 0)
    tasks = controller.unit_tasks._replace(
        task_type=controller.unit_tasks.task_type.at[0, 0].set(
            jnp.int8(TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
        ),
        target_id=controller.unit_tasks.target_id.at[0, 0].set(jnp.int16(0)),
        target_x=controller.unit_tasks.target_x.at[0, 0].set(jnp.int8(0)),
        target_y=controller.unit_tasks.target_y.at[0, 0].set(jnp.int8(0)),
        item_id=controller.unit_tasks.item_id.at[0, 0].set(jnp.int8(1)),
        phase=controller.unit_tasks.phase.at[0, 0].set(
            jnp.int8(TaskPhaseV1.MOVE_TO_TARGET)
        ),
        status=controller.unit_tasks.status.at[0, 0].set(
            jnp.int8(TaskStatusV1.ACTIVE)
        ),
    )
    controller = controller._replace(unit_tasks=tasks)
    genome = default_m35_farm_genome_v2(1)
    cleared = clear_invalidated_m3_tasks_v2(
        states, controller, genome.animal, 0
    )
    assert int(cleared.unit_tasks.status[0, 0]) == TaskStatusV1.EMPTY


def test_market_template_orders_candidates_before_ten_slot_truncation():
    states = jax.vmap(reset)(jnp.asarray((71,), dtype=jnp.int32))
    money = states.money.at[0, 0].set(jnp.int32(100_000))
    kind = states.tile_kind
    for x in range(6):
        kind = kind.at[0, 0, 0, x].set(jnp.int8(TileKind.PASTURE))
    # Eight product sells are enough to crowd one animal species out under the
    # legacy sell-first order, making this a regression for pre-truncation
    # ordering rather than a superficial post-pack sort.
    shed = states.shed.at[0, 0, :9].set(jnp.int16(3))
    states = states._replace(money=money, tile_kind=kind, shed=shed)
    calendar = empty_route_calendar_v3(1)
    additions = calendar.animal_purchase_additions_by_day.at[0, 0].set(
        jnp.asarray((0, 2, 2), dtype=jnp.int16)
    )
    crops = calendar.crop_target_by_day.at[0, 0].set(
        jnp.asarray((7, 0, 0, 0, 12), dtype=jnp.int16)
    )
    hands = calendar.hand_target_by_day.at[0, 0].set(jnp.int8(5))
    calendar = calendar._replace(
        animal_purchase_additions_by_day=additions,
        crop_target_by_day=crops,
        hand_target_by_day=hands,
    )
    genome = calendar_step_genome_v3(
        states, calendar, default_m35_farm_genome_v2(1)
    )
    controller = ensure_m35_projects_v2(
        states, initialize_project_controller_v2(states, 0), genome, 0
    )
    legacy = materialize_m35_market_tasks_v2(
        states, controller, genome, 0, load_tables()
    )
    templated = materialize_m35_market_tasks_v2(
        states,
        controller,
        genome,
        0,
        load_tables(),
        market_template=jnp.asarray((0,), dtype=jnp.int8),
    )

    def animal_species(planned_controller):
        tasks = planned_controller.market_tasks
        active = np.asarray(tasks.status[0]) == int(TaskStatusV1.ACTIVE)
        animal = (
            np.asarray(tasks.task_type[0]) == int(TaskTypeV1.ANIMAL_PURCHASE)
        ) & active
        return set(np.asarray(tasks.item_id[0])[animal].tolist())

    assert animal_species(legacy) != {10, 11}
    assert animal_species(templated) == {10, 11}, {
        "task_type": np.asarray(templated.market_tasks.task_type[0]).tolist(),
        "item_id": np.asarray(templated.market_tasks.item_id[0]).tolist(),
        "quantity": np.asarray(templated.market_tasks.quantity[0]).tolist(),
        "status": np.asarray(templated.market_tasks.status[0]).tolist(),
    }


def test_successful_fertilizer_collection_releases_return_leg_for_batching():
    states = jax.vmap(reset)(jnp.asarray((79,), dtype=jnp.int32))
    states = states._replace(
        tile_kind=states.tile_kind.at[0, 0, 0, 0].set(
            jnp.int8(TileKind.PASTURE)
        ),
        tile_animal=states.tile_animal.at[0, 0, 0, 0].set(jnp.int8(1)),
        tile_flags=states.tile_flags.at[0, 0, 0, 0].set(
            jnp.uint8(FLAG_FERTILIZER_AVAILABLE)
        ),
        unit_pos=states.unit_pos.at[0, 0, 0].set(
            jnp.asarray((0, 0), dtype=jnp.int8)
        ),
    )
    planned = initialize_project_controller_v2(states, 0)
    tasks = planned.unit_tasks._replace(
        task_type=planned.unit_tasks.task_type.at[0, 0].set(
            jnp.int8(TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
        ),
        target_id=planned.unit_tasks.target_id.at[0, 0].set(jnp.int16(0)),
        target_x=planned.unit_tasks.target_x.at[0, 0].set(jnp.int8(0)),
        target_y=planned.unit_tasks.target_y.at[0, 0].set(jnp.int8(0)),
        item_id=planned.unit_tasks.item_id.at[0, 0].set(jnp.int8(8)),
        phase=planned.unit_tasks.phase.at[0, 0].set(
            jnp.int8(TaskPhaseV1.MOVE_TO_TARGET)
        ),
        status=planned.unit_tasks.status.at[0, 0].set(
            jnp.int8(TaskStatusV1.ACTIVE)
        ),
    )
    planned = planned._replace(unit_tasks=tasks)
    action = compile_full_core_player_action_v1(states, planned, 0)
    assert int(action.unit_op[0, 0]) == UnitOp.COLLECT_FERTILIZER
    next_states = states._replace(
        unit_inventory=states.unit_inventory.at[0, 0, 0, 8].set(jnp.int16(1))
    )
    generic_updated = planned._replace(
        unit_tasks=planned.unit_tasks._replace(
            phase=planned.unit_tasks.phase.at[0, 0].set(
                jnp.int8(TaskPhaseV1.MOVE_TO_DEPOT)
            )
        )
    )
    continued = continue_m36_fertilizer_batch_after_effects_v3(
        states, next_states, planned, generic_updated, action, 0
    )
    assert int(continued.unit_tasks.status[0, 0]) == TaskStatusV1.EMPTY


def test_fertilizer_sale_replaces_tenth_slot_and_skips_day_end_refresh():
    states = jax.vmap(reset)(jnp.asarray((83,), dtype=jnp.int32))
    states = states._replace(
        shed=states.shed.at[0, 0, 8].set(jnp.int16(3))
    )
    controller = initialize_project_controller_v2(states, 0)
    full_market = controller.market_tasks._replace(
        task_type=jnp.full((1, 10), TaskTypeV1.HIRE_WORKER, dtype=jnp.int8),
        quantity=jnp.ones((1, 10), dtype=jnp.int16),
        status=jnp.full((1, 10), TaskStatusV1.ACTIVE, dtype=jnp.int8),
    )
    controller = controller._replace(market_tasks=full_market)
    calendar = empty_route_calendar_v3(1)
    planned, diagnostics = materialize_m36_fertilizer_ledger_v3(
        states, controller, calendar, 0
    )
    market = planned.market_tasks
    fertilizer_sell = (
        (market.status[0] == TaskStatusV1.ACTIVE)
        & (market.task_type[0] == TaskTypeV1.SELL_INVENTORY)
        & (market.item_id[0] == 8)
    )
    assert int(jnp.sum(market.status[0] == TaskStatusV1.ACTIVE)) == 10
    assert int(jnp.sum(fertilizer_sell)) == 1
    assert int(market.quantity[0, jnp.argmax(fertilizer_sell)]) == 3
    assert int(diagnostics.market_slot_overflow_count[0]) == 0

    day_end = states._replace(step=jnp.asarray((23,), dtype=jnp.int16))
    blank = initialize_project_controller_v2(day_end, 0)
    planned, diagnostics = materialize_m36_fertilizer_ledger_v3(
        day_end, blank, calendar, 0
    )
    assert not bool(
        jnp.any(
            (planned.market_tasks.status[0] == TaskStatusV1.ACTIVE)
            & (planned.market_tasks.item_id[0] == 8)
        )
    )
    assert int(diagnostics.sell_units[0]) == 0


def test_fertilizer_ledger_does_not_sell_carried_stock_before_deposit():
    states = jax.vmap(reset)(jnp.asarray((97,), dtype=jnp.int32))
    states = states._replace(
        unit_inventory=states.unit_inventory.at[0, 0, 0, 8].set(jnp.int16(3))
    )
    controller = initialize_project_controller_v2(states, 0)
    calendar = empty_route_calendar_v3(1)

    planned, diagnostics = materialize_m36_fertilizer_ledger_v3(
        states, controller, calendar, 0
    )

    fertilizer_sell = (
        (planned.market_tasks.status[0] == TaskStatusV1.ACTIVE)
        & (planned.market_tasks.task_type[0] == TaskTypeV1.SELL_INVENTORY)
        & (planned.market_tasks.item_id[0] == 8)
    )
    assert int(diagnostics.held_units[0]) == 3
    assert int(diagnostics.surplus_units[0]) == 3
    assert int(diagnostics.sell_units[0]) == 0
    assert not bool(jnp.any(fertilizer_sell))


def test_fertilizer_ledger_reserves_carried_stock_before_shed_stock():
    states = jax.vmap(reset)(jnp.asarray((101,), dtype=jnp.int32))
    states = states._replace(
        shed=states.shed.at[0, 0, 8].set(jnp.int16(2)),
        unit_inventory=states.unit_inventory.at[0, 0, 0, 8].set(jnp.int16(1)),
    )
    controller = initialize_project_controller_v2(states, 0)
    calendar = empty_route_calendar_v3(1)._replace(
        fertilizer_safety_stock_by_day=jnp.ones((1, 30), dtype=jnp.int16)
    )

    planned, diagnostics = materialize_m36_fertilizer_ledger_v3(
        states, controller, calendar, 0
    )
    fertilizer_sell = (
        (planned.market_tasks.status[0] == TaskStatusV1.ACTIVE)
        & (planned.market_tasks.task_type[0] == TaskTypeV1.SELL_INVENTORY)
        & (planned.market_tasks.item_id[0] == 8)
    )
    slot = jnp.argmax(fertilizer_sell)
    assert int(diagnostics.held_units[0]) == 3
    assert int(diagnostics.reserved_units[0]) == 1
    assert int(diagnostics.surplus_units[0]) == 2
    assert int(diagnostics.sell_units[0]) == 2
    assert int(planned.market_tasks.quantity[0, slot]) == 2


def test_reactive_sales_defer_existing_fertilizer_sell_at_day_end():
    states = jax.vmap(reset)(jnp.asarray((89,), dtype=jnp.int32))
    states = states._replace(
        step=jnp.asarray((383,), dtype=jnp.int16),
        shed=states.shed.at[0, 0, 8].set(jnp.int16(1)),
    )
    controller = initialize_project_controller_v2(states, 0)
    market = controller.market_tasks._replace(
        task_type=controller.market_tasks.task_type.at[0, 0].set(
            jnp.int8(TaskTypeV1.SELL_INVENTORY)
        ),
        item_id=controller.market_tasks.item_id.at[0, 0].set(jnp.int8(8)),
        quantity=controller.market_tasks.quantity.at[0, 0].set(jnp.int16(1)),
        status=controller.market_tasks.status.at[0, 0].set(
            jnp.int8(TaskStatusV1.ACTIVE)
        ),
    )
    controller = controller._replace(market_tasks=market)
    updated = materialize_m36_reactive_sales_v3(
        states,
        controller,
        empty_route_calendar_v3(1),
        default_m35_farm_genome_v2(1),
        0,
    )
    active_fertilizer_sell = (
        (updated.market_tasks.status[0] == TaskStatusV1.ACTIVE)
        & (updated.market_tasks.task_type[0] == TaskTypeV1.SELL_INVENTORY)
        & (updated.market_tasks.item_id[0] == 8)
    )
    assert not bool(jnp.any(active_fertilizer_sell))


def test_calendar_animal_commitment_remains_catchable_after_observed_buy_day():
    states = jax.vmap(reset)(jnp.asarray((97,), dtype=jnp.int32))
    states = states._replace(step=jnp.asarray((10 * 24,), dtype=jnp.int16))
    calendar = empty_route_calendar_v3(1)
    additions = calendar.animal_purchase_additions_by_day.at[0, 0, 1].set(
        jnp.int16(2)
    )
    calendar = calendar._replace(animal_purchase_additions_by_day=additions)
    genome = calendar_step_genome_v3(states, calendar)

    assert int(genome.animal.animal_target[0, 0, 1]) == 2
    assert int(genome.animal.animal_investment_stop_step[0, 1]) == 27 * 24
    assert int(states.step[0]) < int(
        genome.animal.animal_investment_stop_step[0, 1]
    )


def test_zero_safety_fertilizer_sale_can_finance_same_bundle_animal() -> None:
    states = jax.vmap(reset)(jnp.asarray((103,), dtype=jnp.int32))
    states = states._replace(
        step=jnp.asarray((4 * 24 + 20,), dtype=jnp.int16),
        money=states.money.at[0, 0].set(jnp.int32(450)),
        shed=(
            states.shed.at[0, 0, 0]
            .set(jnp.int16(10))
            .at[0, 0, 8]
            .set(jnp.int16(1))
        ),
        tile_kind=states.tile_kind.at[0, 0, 0, 0].set(
            jnp.int8(TileKind.PASTURE)
        ),
    )
    calendar = empty_route_calendar_v3(1)
    additions = calendar.animal_purchase_additions_by_day.at[0, 4, 1].set(
        jnp.int16(1)
    )
    service = calendar.animal_service_target_by_day.at[0, 4, 1].set(
        jnp.int16(1)
    )
    calendar = calendar._replace(
        animal_purchase_additions_by_day=additions,
        animal_service_target_by_day=service,
    )
    genome = calendar_step_genome_v3(states, calendar)
    assert int(genome.animal.animal_fertilizer_policy[0]) == int(
        M3AnimalFertilizerPolicyV2.ACTIVE_COLLECT_AND_SELL
    )
    assert int(genome.animal.sell_interval[0, 8]) == 1

    controller = ensure_m35_projects_v2(
        states, initialize_project_controller_v2(states, 0), genome, 0
    )
    planned = materialize_m35_market_tasks_v2(
        states,
        controller,
        genome,
        0,
        load_tables(),
        market_template=jnp.asarray((1,), dtype=jnp.int8),
        reserve_future_hires=False,
    )
    tasks = planned.market_tasks
    active = tasks.status[0] == TaskStatusV1.ACTIVE
    fertilizer_sell = (
        active
        & (tasks.task_type[0] == TaskTypeV1.SELL_INVENTORY)
        & (tasks.item_id[0] == 8)
    )
    cow_buy = (
        active
        & (tasks.task_type[0] == TaskTypeV1.ANIMAL_PURCHASE)
        & (tasks.item_id[0] == 10)
    )
    assert int(jnp.sum(fertilizer_sell)) == 1
    assert int(jnp.sum(cow_buy)) == 1


def test_positive_fertilizer_safety_keeps_market_projection_conservative() -> None:
    states = jax.vmap(reset)(jnp.asarray((107,), dtype=jnp.int32))
    calendar = empty_route_calendar_v3(1)._replace(
        fertilizer_safety_stock_by_day=jnp.ones((1, 30), dtype=jnp.int16)
    )
    genome = calendar_step_genome_v3(states, calendar)
    assert int(genome.animal.animal_fertilizer_policy[0]) == int(
        M3AnimalFertilizerPolicyV2.RESERVE_FOR_CROPS
    )
