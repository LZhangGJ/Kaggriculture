from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import (
    MarketOp,
    NUM_CROPS,
    NUM_PRODUCTS,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.state import load_tables, reset
from strategic_v5.constants import TaskPhaseV1, TaskStatusV1, TaskTypeV1
from strategic_v5.e3_executor import (
    compile_e3_player_action_v1,
    update_e3_controller_from_effects_v1,
)

from project_route_search_v2.lifecycle import initialize_project_controller_v2
from project_route_search_v2.m3_controller import (
    _via_one_shed_access_distance,
    derive_m3_commitment_ledger_v2,
    ensure_m3_projects_v2,
    m3_policy_step_v2,
    materialize_m3_market_tasks_v2,
    materialize_m3_unit_tasks_v2,
)
from project_route_search_v2.m3_genome import default_m3_animal_genome_v2


def _states(seeds=(2701, 2702)):
    return jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))


def test_pickup_route_uses_one_physical_shed_access_cell() -> None:
    # From (5, 5) to (0, 1), pickup at the current access cell costs
    # 1 + Manhattan(5,5 -> 0,1) = 10.  Independently minimizing the two
    # route halves used to return 0 + 1 + Manhattan(4,4 -> 0,1) = 8, which
    # is physically impossible because pickup does not teleport the unit.
    distances = _via_one_shed_access_distance(
        jnp.asarray(((5, 5), (4, 4)), dtype=jnp.int16)
    )
    target = 1 * 10 + 0
    np.testing.assert_array_equal(np.asarray(distances[:, target]), [10, 8])


def test_batch_pickup_and_same_step_wheat_buy_reconcile_exact_units() -> None:
    states = _states((2717,))
    states = states._replace(
        shed=states.shed.at[0, 0, 0].set(jnp.int16(5))
    )
    controller = initialize_project_controller_v2(states, 0)
    units = controller.unit_tasks._replace(
        task_type=controller.unit_tasks.task_type.at[0, 0].set(
            jnp.int8(TaskTypeV1.ANIMAL_FEED)
        ),
        target_id=controller.unit_tasks.target_id.at[0, 0].set(jnp.int16(0)),
        target_x=controller.unit_tasks.target_x.at[0, 0].set(jnp.int8(0)),
        target_y=controller.unit_tasks.target_y.at[0, 0].set(jnp.int8(0)),
        item_id=controller.unit_tasks.item_id.at[0, 0].set(jnp.int8(0)),
        quantity=controller.unit_tasks.quantity.at[0, 0].set(jnp.int16(4)),
        phase=controller.unit_tasks.phase.at[0, 0].set(
            jnp.int8(TaskPhaseV1.MOVE_TO_TARGET)
        ),
        deadline_step=controller.unit_tasks.deadline_step.at[0, 0].set(
            jnp.int16(23)
        ),
        status=controller.unit_tasks.status.at[0, 0].set(
            jnp.int8(TaskStatusV1.ACTIVE)
        ),
    )
    market = controller.market_tasks._replace(
        task_type=controller.market_tasks.task_type.at[0, 0].set(
            jnp.int8(TaskTypeV1.BUY_PRODUCT)
        ),
        item_id=controller.market_tasks.item_id.at[0, 0].set(jnp.int8(0)),
        quantity=controller.market_tasks.quantity.at[0, 0].set(jnp.int16(3)),
        deadline_step=controller.market_tasks.deadline_step.at[0, 0].set(
            jnp.int16(1)
        ),
        status=controller.market_tasks.status.at[0, 0].set(
            jnp.int8(TaskStatusV1.ACTIVE)
        ),
    )
    controller = controller._replace(unit_tasks=units, market_tasks=market)
    action = compile_e3_player_action_v1(states, controller, 0)
    next_states = states._replace(
        step=states.step + 1,
        unit_inventory=states.unit_inventory.at[0, 0, 0, 0].set(
            jnp.int16(4)
        ),
        # PICKUP 4 followed by BUY_PRODUCT 3: 5 - 4 + 3 = 4.
        shed=states.shed.at[0, 0, 0].set(jnp.int16(4)),
    )
    _, diagnostics = update_e3_controller_from_effects_v1(
        states, next_states, controller, action, 0
    )

    assert int(action.unit_amount[0, 0]) == 4
    assert int(diagnostics.effect_mismatch_count[0]) == 0
    assert int(diagnostics.pickup_success_count[0]) == 1
    assert int(diagnostics.wheat_purchase_success_count[0]) == 1


def test_feed_ledger_keeps_survival_bonus_and_care_obligations_distinct() -> None:
    base = _states((2701,))
    states = base._replace(
        step=jnp.asarray((7 * 24,), dtype=jnp.int16),
        tile_kind=base.tile_kind.at[0, 0, 0, 0].set(TileKind.COOP)
        .at[0, 0, 0, 1]
        .set(TileKind.PASTURE)
        .at[0, 0, 0, 2]
        .set(TileKind.PASTURE),
        tile_animal=base.tile_animal.at[0, 0, 0, 0].set(0)
        .at[0, 0, 0, 1]
        .set(1)
        .at[0, 0, 0, 2]
        .set(2),
        tile_origin_day=base.tile_origin_day.at[0, 0, 0, 0].set(0)
        .at[0, 0, 0, 1]
        .set(0)
        .at[0, 0, 0, 2]
        .set(7),
        tile_neglect=base.tile_neglect.at[0, 0, 0, 0].set(1),
        tile_pending_care=base.tile_pending_care.at[0, 0, 0, 1].set(2),
        shed=base.shed.at[0, 0, NUM_PRODUCTS + 0].set(2),
        unit_inventory=base.unit_inventory.at[0, 0, 0, NUM_PRODUCTS + 1].set(1),
    )
    genome = default_m3_animal_genome_v2(1)._replace(
        care_policy=jnp.asarray(((0, 0, 2),), dtype=jnp.int8)
    )
    controller = initialize_project_controller_v2(states, 0)
    market = controller.market_tasks._replace(
        task_type=controller.market_tasks.task_type.at[0, 0].set(
            TaskTypeV1.ANIMAL_PURCHASE
        ),
        item_id=controller.market_tasks.item_id.at[0, 0].set(NUM_PRODUCTS + 2),
        quantity=controller.market_tasks.quantity.at[0, 0].set(3),
        status=controller.market_tasks.status.at[0, 0].set(TaskStatusV1.ACTIVE),
    )
    ledger = derive_m3_commitment_ledger_v2(
        states, controller._replace(market_tasks=market), genome, 0
    )
    np.testing.assert_array_equal(ledger.active, [[1, 1, 1]])
    np.testing.assert_array_equal(ledger.in_shed, [[2, 0, 0]])
    np.testing.assert_array_equal(ledger.carried, [[0, 1, 0]])
    np.testing.assert_array_equal(ledger.pending_purchase, [[0, 0, 3]])
    np.testing.assert_array_equal(ledger.committed, [[3, 2, 4]])
    np.testing.assert_array_equal(ledger.survival_feed_due_today, [[1, 0, 0]])
    np.testing.assert_array_equal(ledger.bonus_feed_due_today, [[0, 1, 0]])
    np.testing.assert_array_equal(ledger.care_feed_due_today, [[0, 0, 1]])
    np.testing.assert_array_equal(ledger.feed_required_today, [[1, 1, 1]])


def test_zero_animal_target_adds_no_animal_actions() -> None:
    states = _states()
    genome = default_m3_animal_genome_v2(2)._replace(
        animal_target=jnp.zeros((2, 6, 3), dtype=jnp.int16),
        hand_target=jnp.zeros((2, 6), dtype=jnp.int8),
        land_target=jnp.ones((2, 6), dtype=jnp.int8),
    )
    controller = initialize_project_controller_v2(states, 0)
    action, _ = jax.jit(m3_policy_step_v2, static_argnums=(3,))(
        states, controller, genome, 0, load_tables()
    )
    animal_unit_ops = np.isin(
        np.asarray(action.unit_op),
        [
            UnitOp.BUILD_COOP,
            UnitOp.BUILD_PASTURE,
            UnitOp.PLACE,
            UnitOp.FEED,
            UnitOp.CARE,
            UnitOp.HARVEST,
            UnitOp.COLLECT_FERTILIZER,
        ],
    )
    assert not np.any(animal_unit_ops)
    assert not np.any(np.asarray(action.market_op) == MarketOp.BUY_ANIMAL)


def test_cow_and_sheep_cannot_double_book_one_empty_pasture() -> None:
    base = _states((2710,))
    states = base._replace(
        money=base.money.at[0, 0].set(100_000),
        tile_kind=base.tile_kind.at[0, 0, 4, 4].set(TileKind.PASTURE),
    )
    genome = default_m3_animal_genome_v2(1)._replace(
        phase_count=jnp.asarray((1,), dtype=jnp.int8),
        animal_target=jnp.zeros((1, 6, 3), dtype=jnp.int16)
        .at[:, :, 1]
        .set(1)
        .at[:, :, 2]
        .set(1),
        # Keep this fixture focused on the shared-pasture invariant.  Two hands
        # give the calibrated CARE-every-cycle maintenance model enough route
        # budget to admit both raw requests before the pasture allocator rejects
        # one of them.  (The old one-hand fixture predates the 22-step complete
        # feed/care/harvest/deposit route cost and now correctly caps both out.)
        hand_target=jnp.full((1, 6), 2, dtype=jnp.int8),
        land_target=jnp.ones((1, 6), dtype=jnp.int8),
    )
    controller = initialize_project_controller_v2(states, 0)
    controller = ensure_m3_projects_v2(states, controller, genome, 0)
    market = materialize_m3_market_tasks_v2(
        states, controller, genome, 0, load_tables()
    ).market_tasks
    active = np.asarray(market.status) == TaskStatusV1.ACTIVE
    animal_orders = active & (
        np.asarray(market.task_type) == TaskTypeV1.ANIMAL_PURCHASE
    )
    assert int(np.sum(animal_orders)) == 1


def test_complete_calendar_rolling_replan_can_admit_exact_cost_cow() -> None:
    """Rolling replans must not prepay a cow's whole first-yield feed runway."""

    base = _states((2721, 2722))
    states = base._replace(
        money=base.money.at[:, 0].set(jnp.asarray((400, 399), dtype=jnp.int32)),
        tile_kind=base.tile_kind.at[:, 0, 4, 4].set(TileKind.PASTURE),
    )
    genome = default_m3_animal_genome_v2(2)._replace(
        phase_count=jnp.ones((2,), dtype=jnp.int8),
        animal_target=jnp.zeros((2, 6, 3), dtype=jnp.int16)
        .at[:, :, 1]
        .set(1),
        hand_target=jnp.zeros((2, 6), dtype=jnp.int8),
        land_target=jnp.ones((2, 6), dtype=jnp.int8),
        cash_floor=jnp.zeros((2,), dtype=jnp.int32),
        enforce_productive_cap=jnp.zeros((2,), dtype=jnp.bool_),
    )
    controller = initialize_project_controller_v2(states, 0)
    controller = ensure_m3_projects_v2(states, controller, genome, 0)

    conservative = materialize_m3_market_tasks_v2(
        states, controller, genome, 0, load_tables()
    ).market_tasks
    rolling = materialize_m3_market_tasks_v2(
        states,
        controller,
        genome,
        0,
        load_tables(),
        reserve_future_hires=False,
        rolling_replan=True,
    ).market_tasks

    def cow_quantity(tasks):
        mask = (
            (tasks.status == TaskStatusV1.ACTIVE)
            & (tasks.task_type == TaskTypeV1.ANIMAL_PURCHASE)
            & (tasks.item_id == NUM_PRODUCTS + 1)
        )
        return jnp.sum(jnp.where(mask, tasks.quantity, 0), axis=-1)

    np.testing.assert_array_equal(np.asarray(cow_quantity(conservative)), [0, 0])
    # Exact purchase cost is legal; one dollar below it is still rejected.
    np.testing.assert_array_equal(np.asarray(cow_quantity(rolling)), [1, 0])


def test_feed_task_batches_two_wheat_into_one_official_pickup() -> None:
    base = _states((2724,))
    states = base._replace(
        unit_active=base.unit_active.at[0, 0, :4].set(True),
        unit_pos=base.unit_pos.at[0, 0, :4].set(
            jnp.asarray(((4, 4), (4, 4), (4, 4), (4, 4)), dtype=jnp.int8)
        ),
        shed=base.shed.at[0, 0, 0].set(jnp.int16(4)),
        tile_kind=base.tile_kind.at[0, 0, 0, 0]
        .set(TileKind.PASTURE)
        .at[0, 0, 0, 1]
        .set(TileKind.PASTURE)
        .at[0, 0, 0, 2]
        .set(TileKind.PASTURE)
        .at[0, 0, 0, 3]
        .set(TileKind.PASTURE),
        tile_animal=base.tile_animal.at[0, 0, 0, 0]
        .set(jnp.int8(1))
        .at[0, 0, 0, 1]
        .set(jnp.int8(1))
        .at[0, 0, 0, 2]
        .set(jnp.int8(1))
        .at[0, 0, 0, 3]
        .set(jnp.int8(1)),
        tile_neglect=base.tile_neglect.at[0, 0, 0, 0]
        .set(jnp.int8(1))
        .at[0, 0, 0, 1]
        .set(jnp.int8(1))
        .at[0, 0, 0, 2]
        .set(jnp.int8(1))
        .at[0, 0, 0, 3]
        .set(jnp.int8(1)),
    )
    genome = default_m3_animal_genome_v2(1)._replace(
        phase_count=jnp.ones((1,), dtype=jnp.int8),
        animal_target=jnp.zeros((1, 6, 3), dtype=jnp.int16)
        .at[:, :, 1]
        .set(4),
        hand_target=jnp.full((1, 6), 3, dtype=jnp.int8),
        enforce_productive_cap=jnp.zeros((1,), dtype=jnp.bool_),
    )
    controller = ensure_m3_projects_v2(
        states, initialize_project_controller_v2(states, 0), genome, 0
    )
    planned = materialize_m3_unit_tasks_v2(states, controller, genome, 0)
    tasks = planned.unit_tasks
    feed = (
        (tasks.status[0] == TaskStatusV1.ACTIVE)
        & (tasks.task_type[0] == TaskTypeV1.ANIMAL_FEED)
    )
    assert int(jnp.sum(feed)) == 2
    np.testing.assert_array_equal(
        np.sort(np.asarray(tasks.quantity[0, feed])), np.asarray((2, 2))
    )

    action = compile_e3_player_action_v1(states, planned, 0)
    pickup = action.unit_op[0] == UnitOp.PICKUP
    assert int(jnp.sum(pickup)) == 2
    assert int(jnp.sum(action.unit_amount[0, pickup])) == 4


def test_sheep_tagged_empty_pasture_remains_legal_for_cow_placement() -> None:
    base = _states((2723,))
    states = base._replace(
        tile_kind=base.tile_kind.at[0, 0, 0, 0].set(TileKind.PASTURE),
        shed=base.shed.at[0, 0, NUM_PRODUCTS + 1].set(jnp.int16(1)),
    )
    genome = default_m3_animal_genome_v2(1)._replace(
        phase_count=jnp.asarray((1,), dtype=jnp.int8),
        animal_target=jnp.zeros((1, 6, 3), dtype=jnp.int16)
        .at[:, :, 1]
        .set(1),
        hand_target=jnp.zeros((1, 6), dtype=jnp.int8),
    )
    controller = initialize_project_controller_v2(states, 0)
    controller = ensure_m3_projects_v2(states, controller, genome, 0)
    controller = controller._replace(
        tile_project_id=controller.tile_project_id.at[0, 0, 0].set(
            jnp.int16(NUM_CROPS + 2)
        )
    )
    planned = materialize_m3_unit_tasks_v2(
        states, controller, genome, 0
    )
    active = planned.unit_tasks.status[0] == TaskStatusV1.ACTIVE
    place = active & (
        planned.unit_tasks.task_type[0] == TaskTypeV1.ANIMAL_PLACE
    )
    build = active & (
        planned.unit_tasks.task_type[0] == TaskTypeV1.BUILD_ANIMAL_STRUCTURE
    )
    assert int(jnp.sum(place)) == 1
    assert int(planned.unit_tasks.target_id[0, jnp.argmax(place)]) == 0
    assert int(jnp.sum(build)) == 0


def test_shared_pasture_build_wave_preserves_cow_and_sheep_capacity() -> None:
    base = _states((2729,))
    states = base._replace(
        unit_active=base.unit_active.at[0, 0, :4].set(True),
        shed=base.shed.at[0, 0, NUM_PRODUCTS + 1].set(jnp.int16(2))
        .at[0, 0, NUM_PRODUCTS + 2]
        .set(jnp.int16(2)),
    )
    genome = default_m3_animal_genome_v2(1)._replace(
        phase_count=jnp.asarray((1,), dtype=jnp.int8),
        animal_target=jnp.zeros((1, 6, 3), dtype=jnp.int16)
        .at[:, :, 1]
        .set(2)
        .at[:, :, 2]
        .set(2),
        animal_place_wave_size=jnp.ones((1, 3), dtype=jnp.int8)
        .at[:, 1]
        .set(2)
        .at[:, 2]
        .set(2),
        hand_target=jnp.full((1, 6), 4, dtype=jnp.int8),
    )
    controller = initialize_project_controller_v2(states, 0)
    controller = ensure_m3_projects_v2(states, controller, genome, 0)
    planned = materialize_m3_unit_tasks_v2(
        states,
        controller,
        genome,
        0,
        unlock_commitment_mask=jnp.asarray(((False, True, True),)),
    )
    active = planned.unit_tasks.status[0] == TaskStatusV1.ACTIVE
    build = active & (
        planned.unit_tasks.task_type[0] == TaskTypeV1.BUILD_ANIMAL_STRUCTURE
    )
    build_items = planned.unit_tasks.item_id[0, build]
    build_targets = planned.unit_tasks.target_id[0, build]

    assert int(jnp.sum(build)) == 4
    assert set(np.asarray(build_items).tolist()).issubset({1, 2})
    assert len(set(np.asarray(build_targets).tolist())) == 4


def test_animal_place_route_is_not_started_when_day_reset_would_erase_progress() -> None:
    base = _states((2731,))
    states = base._replace(
        step=jnp.asarray((22,), dtype=jnp.int16),
        unit_active=base.unit_active.at[0, 0, :2].set(True),
        unit_pos=base.unit_pos.at[0, 0, :2].set(
            jnp.asarray(((9, 9), (9, 9)), dtype=jnp.int8)
        ),
        tile_kind=base.tile_kind.at[0, 0, 0, 0].set(TileKind.PASTURE),
        shed=base.shed.at[0, 0, NUM_PRODUCTS + 1].set(jnp.int16(1)),
    )
    genome = default_m3_animal_genome_v2(1)._replace(
        phase_count=jnp.asarray((1,), dtype=jnp.int8),
        animal_target=jnp.zeros((1, 6, 3), dtype=jnp.int16)
        .at[:, :, 1]
        .set(1),
        hand_target=jnp.full((1, 6), 1, dtype=jnp.int8),
    )
    controller = ensure_m3_projects_v2(
        states, initialize_project_controller_v2(states, 0), genome, 0
    )
    planned = materialize_m3_unit_tasks_v2(states, controller, genome, 0)
    active_place = (
        (planned.unit_tasks.status[0] == TaskStatusV1.ACTIVE)
        & (planned.unit_tasks.task_type[0] == TaskTypeV1.ANIMAL_PLACE)
    )
    assert int(jnp.sum(active_place)) == 0


def test_carried_animal_can_be_placed_on_final_action_of_day() -> None:
    base = _states((2737,))
    states = base._replace(
        step=jnp.asarray((23,), dtype=jnp.int16),
        unit_pos=base.unit_pos.at[0, 0, 0].set(
            jnp.asarray((0, 0), dtype=jnp.int8)
        ),
        unit_inventory=base.unit_inventory.at[
            0, 0, 0, NUM_PRODUCTS + 1
        ].set(jnp.int16(1)),
        tile_kind=base.tile_kind.at[0, 0, 0, 0].set(TileKind.PASTURE),
    )
    genome = default_m3_animal_genome_v2(1)._replace(
        phase_count=jnp.asarray((1,), dtype=jnp.int8),
        animal_target=jnp.zeros((1, 6, 3), dtype=jnp.int16)
        .at[:, :, 1]
        .set(1),
        hand_target=jnp.zeros((1, 6), dtype=jnp.int8),
    )
    controller = ensure_m3_projects_v2(
        states, initialize_project_controller_v2(states, 0), genome, 0
    )
    planned = materialize_m3_unit_tasks_v2(states, controller, genome, 0)
    active_place = (
        (planned.unit_tasks.status[0] == TaskStatusV1.ACTIVE)
        & (planned.unit_tasks.task_type[0] == TaskTypeV1.ANIMAL_PLACE)
    )
    assert int(jnp.sum(active_place)) == 1
    assert int(planned.unit_tasks.target_id[0, jnp.argmax(active_place)]) == 0
