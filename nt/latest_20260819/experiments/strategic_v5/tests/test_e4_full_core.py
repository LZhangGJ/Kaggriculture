from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import load_event_bank, load_tables, reset
from kaggriculture_jax.constants import (
    MAX_UNITS,
    NUM_PRODUCTS,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.simulator import batched_step_sync
from strategic_v5 import (
    BUILD_COOP_SLOT,
    BUILD_PASTURE_SLOT,
    DEFAULT_REPLAY_TASK_CARD_PROGRAM_V1,
    FERTILIZER_ITEM,
    PLANT_START,
    PRIMARY_UNIT_START,
    SECONDARY_UNIT_START,
    TaskPhaseV1,
    TaskStatusV1,
    TaskTypeV1,
    build_full_core_candidates_v1,
    candidate_mask_from_e2_ledger_v1,
    clear_finished_tasks_v1,
    clear_invalidated_full_core_tasks_v1,
    compile_crop_inventory_player_action_v1,
    compile_full_core_action_bundle_v1,
    evaluate_full_core_feasibility_v1,
    empty_replay_task_card_program_v1,
    initialize_e2_ledger_v1,
    reset_controller_state_v1,
    select_candidates_with_scores_v1,
    select_full_core_candidates_v1,
    update_full_core_controller_from_effects_v1,
)


TABLES = load_tables()


def _two_active_tasks(states, task_type, targets):
    states = states._replace(
        unit_active=states.unit_active.at[0, 0, :2].set(True),
        unit_pos=states.unit_pos.at[0, 0, 0].set(
            jnp.asarray(targets[0], dtype=jnp.int8)
        ).at[0, 0, 1].set(jnp.asarray(targets[1], dtype=jnp.int8)),
    )
    controller = _controllers(1)
    tasks = controller.unit_tasks._replace(
        task_type=controller.unit_tasks.task_type.at[0, :2].set(task_type),
        owner_unit=controller.unit_tasks.owner_unit.at[0, :2].set(
            jnp.asarray([0, 1], dtype=jnp.int8)
        ),
        target_id=controller.unit_tasks.target_id.at[0, :2].set(
            jnp.asarray([y * 10 + x for x, y in targets], dtype=jnp.int16)
        ),
        target_x=controller.unit_tasks.target_x.at[0, :2].set(
            jnp.asarray([x for x, _ in targets], dtype=jnp.int8)
        ),
        target_y=controller.unit_tasks.target_y.at[0, :2].set(
            jnp.asarray([y for _, y in targets], dtype=jnp.int8)
        ),
        item_id=controller.unit_tasks.item_id.at[0, :2].set(0),
        phase=controller.unit_tasks.phase.at[0, :2].set(
            TaskPhaseV1.MOVE_TO_TARGET
        ),
        deadline_step=controller.unit_tasks.deadline_step.at[0, :2].set(718),
        status=controller.unit_tasks.status.at[0, :2].set(TaskStatusV1.ACTIVE),
    )
    return states, controller._replace(unit_tasks=tasks)


def test_crop_compiler_limits_simultaneous_plants_to_pre_market_seed_stock() -> None:
    states, controller = _two_active_tasks(
        _states(), TaskTypeV1.CROP_PRODUCTION, ((0, 0), (1, 0))
    )
    states = states._replace(seeds=states.seeds.at[0, 0, 0].set(1))
    action = compile_crop_inventory_player_action_v1(states, controller, 0)
    assert np.asarray(action.unit_op[0, :2]).tolist() == [UnitOp.PLANT, UnitOp.PASS]
    assert int(action.diagnostics.unexpected_pass_count[0]) == 0


def test_crop_compiler_serializes_same_tile_water_actions() -> None:
    states, controller = _two_active_tasks(
        _states(), TaskTypeV1.WATER_CROP, ((0, 0), (0, 0))
    )
    states = states._replace(
        tile_kind=states.tile_kind.at[0, 0, 0, 0].set(TileKind.PLANT),
        tile_crop=states.tile_crop.at[0, 0, 0, 0].set(0),
    )
    action = compile_crop_inventory_player_action_v1(states, controller, 0)
    assert np.asarray(action.unit_op[0, :2]).tolist() == [UnitOp.WATER, UnitOp.PASS]
    assert int(action.diagnostics.unexpected_pass_count[0]) == 0


def _states(seeds=(101,)):
    return jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))


def _controllers(batch_size: int):
    value = reset_controller_state_v1()
    return jax.tree.map(
        lambda item: jnp.broadcast_to(item, (batch_size,) + item.shape), value
    )


def _events(batch_size: int):
    _, bank = load_event_bank()
    return jax.tree.map(lambda value: value[:batch_size], bank)


def _decide(states, controller, player: int):
    controller = clear_finished_tasks_v1(controller)
    candidates = build_full_core_candidates_v1(states, controller, TABLES, player)
    feasibility = evaluate_full_core_feasibility_v1(
        states, candidates, TABLES, player
    )
    selection = select_full_core_candidates_v1(
        states, candidates, feasibility, controller, player
    )
    return candidates, feasibility, selection


def test_fixed_pool_layout_and_initial_cross_module_step() -> None:
    states = _states()
    controller0 = _controllers(1)
    controller1 = _controllers(1)
    candidates0, feasibility0, selection0 = _decide(states, controller0, 0)
    candidates1, feasibility1, selection1 = _decide(states, controller1, 1)

    assert candidates0.task_type.shape == (1, 96)
    assert int(candidates0.present.sum()) > 0
    assert bool(candidates0.present[0, BUILD_COOP_SLOT])
    assert bool(candidates0.present[0, BUILD_PASTURE_SLOT])
    assert PLANT_START > SECONDARY_UNIT_START > PRIMARY_UNIT_START
    assert int(selection0.internal_resource_conflict.sum()) == 0
    assert int(selection1.internal_resource_conflict.sum()) == 0
    assert int(selection0.ledger.cash_reserved[0]) <= int(
        selection0.ledger.cash_available[0]
    )
    assert int(selection0.ledger.market_slots_used[0]) <= 10
    assert int(selection0.ledger.land_purchases_reserved[0]) <= 1
    assert bool(jnp.all(feasibility0.legal_now | (~candidates0.hard_mask)))
    assert bool(jnp.all(feasibility1.legal_now | (~candidates1.hard_mask)))

    bundle = compile_full_core_action_bundle_v1(
        states, selection0.controller, selection1.controller
    )
    assert int(bundle.diagnostics.invalid_raw_action_count.sum()) == 0
    assert int(bundle.diagnostics.unit_compiler_overlap_count.sum()) == 0
    assert int(bundle.diagnostics.market_compiler_overlap_count.sum()) == 0
    following = batched_step_sync(states, bundle.action, _events(1), TABLES)
    _, diagnostics0 = update_full_core_controller_from_effects_v1(
        states, following, selection0.controller, bundle.player0, 0
    )
    _, diagnostics1 = update_full_core_controller_from_effects_v1(
        states, following, selection1.controller, bundle.player1, 1
    )
    assert int(diagnostics0.effect_mismatch_count.sum()) == 0
    assert int(diagnostics1.effect_mismatch_count.sum()) == 0
    assert int(following.step[0]) == 1


def test_kawashigi_opening_is_expressible_with_seven_task_cards() -> None:
    states = _states((81201,))
    controller = _controllers(1)
    candidates = build_full_core_candidates_v1(states, controller, TABLES, 0)
    feasibility = evaluate_full_core_feasibility_v1(states, candidates, TABLES, 0)

    # Five hires are one model-side macro card but five official atomic orders.
    assert int(candidates.quantity[0, 11]) == 5
    assert int(feasibility.cash_required[0, 11]) == 12
    assert int(feasibility.market_slots_required[0, 11]) == 5
    assert int(candidates.quantity[0, 2]) == 6  # opening feed reserve
    assert int(candidates.quantity[0, 4]) == 2  # cows
    assert int(candidates.quantity[0, 5]) == 2  # sheep
    assert int(candidates.quantity[0, 6]) == 7  # wheat seeds
    assert int(candidates.quantity[0, 10]) == 12  # melon seeds
    assert bool(candidates.present[0, BUILD_PASTURE_SLOT])

    # Force the seven replay cards in official order.  This checks action-space
    # expressivity independently of the untrained scorer's preferences.
    scores = jnp.full((1, 96), -1.0e9, dtype=jnp.float32)
    for score, slot in zip(
        (700.0, 600.0, 500.0, 400.0, 300.0, 200.0, 100.0),
        (11, 4, 5, 6, 10, 2, BUILD_PASTURE_SLOT),
        strict=True,
    ):
        scores = scores.at[0, slot].set(score)
    selection = select_candidates_with_scores_v1(
        states,
        candidates,
        feasibility,
        scores,
        controller,
        initialize_e2_ledger_v1(states, controller, 0),
        0,
    )
    assert selection.selected_candidate_indices[0].tolist() == [
        11,
        4,
        5,
        6,
        10,
        2,
        BUILD_PASTURE_SLOT,
        -1,
    ]
    assert int(selection.ledger.market_slots_used[0]) == 10
    assert int(selection.ledger.cash_reserved[0]) < int(states.money[0, 0])

    bundle = compile_full_core_action_bundle_v1(
        states, selection.controller, _controllers(1)
    )
    assert bundle.action.market_op[0, 0].tolist() == [
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
    ]
    assert bundle.action.market_item[0, 0].tolist() == [
        -1,
        -1,
        -1,
        -1,
        -1,
        NUM_PRODUCTS + 1,
        NUM_PRODUCTS + 2,
        0,
        4,
        0,
    ]
    assert bundle.action.market_amount[0, 0].tolist() == [1, 1, 1, 1, 1, 2, 2, 7, 12, 6]
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.BUILD_PASTURE
    assert int(bundle.diagnostics.invalid_raw_action_count.sum()) == 0
    assert int(bundle.diagnostics.market_compiler_overlap_count.sum()) == 0

    following = batched_step_sync(states, bundle.action, _events(1), TABLES)
    _, diagnostics = update_full_core_controller_from_effects_v1(
        states, following, selection.controller, bundle.player0, 0
    )
    assert int(diagnostics.effect_mismatch_count.sum()) == 0
    assert int(following.hires_today[0, 0]) == 5
    # The requested six wheat are partially filled to five by the official
    # market because the sixth marginal unit is unaffordable.
    assert int(following.shed[0, 0, 0]) == 5
    assert int(following.shed[0, 0, NUM_PRODUCTS + 1]) == 2
    assert int(following.shed[0, 0, NUM_PRODUCTS + 2]) == 2
    assert int(following.seeds[0, 0, 0]) == 7
    assert int(following.seeds[0, 0, 4]) == 12
    # With a passive opponent the five fills are cheaper than in the source
    # replay (whose opponent simultaneously bought wheat), leaving 25 cash.
    assert int(following.money[0, 0]) == 25
    assert int(jnp.sum(following.tile_kind[0, 0] == TileKind.PASTURE)) == 1


def test_opening_macros_come_from_profile_not_player_specific_code() -> None:
    states = _states((81202,))
    controller = _controllers(1)
    candidates = build_full_core_candidates_v1(
        states,
        controller,
        TABLES,
        0,
        task_card_program=empty_replay_task_card_program_v1(),
    )
    assert int(candidates.quantity[0, 11]) == 1
    assert int(candidates.quantity[0, 6]) == 1
    assert int(candidates.quantity[0, 10]) == 1
    assert not bool(candidates.present[0, 2])
    assert not bool(candidates.present[0, 4])
    assert not bool(candidates.present[0, 5])


def test_task_card_profiles_are_dynamic_fixed_shape_jit_inputs() -> None:
    states = _states((81203,))
    controller = _controllers(1)

    @jax.jit
    def build(current, control, program):
        return build_full_core_candidates_v1(
            current, control, TABLES, 0, task_card_program=program
        )

    expert = build(states, controller, DEFAULT_REPLAY_TASK_CARD_PROGRAM_V1)
    empty = build(states, controller, empty_replay_task_card_program_v1())
    assert expert.quantity.shape == empty.quantity.shape == (1, 96)
    assert int(expert.quantity[0, 11]) == 5
    assert int(empty.quantity[0, 11]) == 1


def test_stale_water_route_is_cleared_before_it_waters_a_weed() -> None:
    states = _states()
    states = states._replace(
        step=states.step.at[0].set(699),
        unit_pos=states.unit_pos.at[0, 0, 0].set(
            jnp.asarray([4, 1], dtype=jnp.int8)
        ),
        tile_kind=states.tile_kind.at[0, 0, 1, 4].set(TileKind.WEED),
        tile_crop=states.tile_crop.at[0, 0, 1, 4].set(-1),
    )
    controller0 = _controllers(1)
    tasks = controller0.unit_tasks
    tasks = tasks._replace(
        task_type=tasks.task_type.at[0, 0].set(TaskTypeV1.WATER_CROP),
        target_id=tasks.target_id.at[0, 0].set(14),
        target_x=tasks.target_x.at[0, 0].set(4),
        target_y=tasks.target_y.at[0, 0].set(1),
        item_id=tasks.item_id.at[0, 0].set(2),
        phase=tasks.phase.at[0, 0].set(TaskPhaseV1.MOVE_TO_TARGET),
        status=tasks.status.at[0, 0].set(TaskStatusV1.ACTIVE),
    )
    controller0 = controller0._replace(unit_tasks=tasks)
    sanitized = clear_invalidated_full_core_tasks_v1(states, controller0, 0)
    assert int(sanitized.unit_tasks.status[0, 0]) == TaskStatusV1.EMPTY

    bundle = compile_full_core_action_bundle_v1(
        states, sanitized, _controllers(1)
    )
    assert int(bundle.player0.unit_op[0, 0]) == UnitOp.PASS
    assert int(bundle.player0.diagnostics.unexpected_pass_count[0]) == 0

    live_states = states._replace(
        tile_kind=states.tile_kind.at[0, 0, 1, 4].set(TileKind.PLANT),
        tile_crop=states.tile_crop.at[0, 0, 1, 4].set(2),
    )
    preserved = clear_invalidated_full_core_tasks_v1(live_states, controller0, 0)
    assert int(preserved.unit_tasks.status[0, 0]) == TaskStatusV1.ACTIVE


def test_terminal_recovery_drop_and_liquidation_are_atomic() -> None:
    states = _states()
    states = states._replace(
        step=states.step.at[0].set(718),
        shed=states.shed.at[0, 0, 0].set(2),
        unit_inventory=states.unit_inventory.at[0, 0, 0, 0].set(3),
        unit_inventory_order=states.unit_inventory_order.at[0, 0, 0, 0].set(0),
        unit_inventory_next_order=states.unit_inventory_next_order.at[0, 0, 0].set(1),
    )
    controller0 = _controllers(1)
    controller1 = _controllers(1)
    candidates, feasibility, selection0 = _decide(states, controller0, 0)
    _, _, selection1 = _decide(states, controller1, 1)
    terminal = candidates.task_type == TaskTypeV1.TERMINAL_LIQUIDATION
    recovery = candidates.task_type == TaskTypeV1.SAFE_RECOVERY
    assert bool(jnp.any(terminal & feasibility.legal_now))
    assert bool(jnp.any(recovery & feasibility.legal_now))
    bundle = compile_full_core_action_bundle_v1(
        states, selection0.controller, selection1.controller
    )
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.DROP
    assert bool(jnp.any(bundle.action.market_op[0, 0] == MarketOp.SELL))
    following = batched_step_sync(states, bundle.action, _events(1), TABLES)
    updated, diagnostics = update_full_core_controller_from_effects_v1(
        states, following, selection0.controller, bundle.player0, 0
    )
    assert int(following.shed[0, 0, 0]) == 0
    assert int(following.unit_inventory[0, 0, 0, 0]) == 0
    assert int(diagnostics.effect_mismatch_count[0]) == 0
    assert int(diagnostics.crop_inventory.sold_product_units[0]) == 5
    assert bool(jnp.all(updated.market_tasks.status != TaskStatusV1.ACTIVE))


def test_day_end_hand_auto_deposit_does_not_hide_same_step_sell_fill() -> None:
    states = _states()._replace(
        step=jnp.asarray((23,), dtype=jnp.int16),
        shed=_states().shed.at[0, 0, 2].set(jnp.int16(3)),
        hires_today=_states().hires_today.at[0, 0].set(jnp.int8(1)),
        unit_active=_states().unit_active.at[0, 0, 1].set(True),
        unit_inventory=_states().unit_inventory.at[0, 0, 1, 2].set(
            jnp.int16(3)
        ),
    )
    controller0 = _controllers(1)
    market = controller0.market_tasks._replace(
        task_type=controller0.market_tasks.task_type.at[0, 0].set(
            TaskTypeV1.SELL_INVENTORY
        ),
        item_id=controller0.market_tasks.item_id.at[0, 0].set(jnp.int8(2)),
        quantity=controller0.market_tasks.quantity.at[0, 0].set(jnp.int16(3)),
        deadline_step=controller0.market_tasks.deadline_step.at[0, 0].set(
            jnp.int16(718)
        ),
        status=controller0.market_tasks.status.at[0, 0].set(
            TaskStatusV1.ACTIVE
        ),
    )
    controller0 = controller0._replace(market_tasks=market)
    controller1 = _controllers(1)
    bundle = compile_full_core_action_bundle_v1(states, controller0, controller1)
    assert int(bundle.player0.market_op[0, 0]) == MarketOp.SELL

    following = batched_step_sync(states, bundle.action, _events(1), TABLES)
    _, diagnostics = update_full_core_controller_from_effects_v1(
        states, following, controller0, bundle.player0, 0
    )

    # The three pre-existing shed units were sold and the hand's three carried
    # units were then deposited, so shed quantity is unchanged despite a real
    # market fill.
    assert int(following.shed[0, 0, 2]) == 3
    assert int(following.unit_inventory[0, 0, 1, 2]) == 0
    assert int(diagnostics.crop_inventory.sold_product_units[0]) == 3
    assert int(diagnostics.effect_mismatch_count[0]) == 0


def test_day_end_farmer_auto_deposit_does_not_hide_same_step_sell_fill() -> None:
    states = _states()._replace(
        step=jnp.asarray((23,), dtype=jnp.int16),
        shed=_states().shed.at[0, 0, 4].set(jnp.int16(6)),
        unit_inventory=_states().unit_inventory.at[0, 0, 0, 4].set(
            jnp.int16(6)
        ),
    )
    controller0 = _controllers(1)
    market = controller0.market_tasks._replace(
        task_type=controller0.market_tasks.task_type.at[0, 0].set(
            TaskTypeV1.SELL_INVENTORY
        ),
        item_id=controller0.market_tasks.item_id.at[0, 0].set(jnp.int8(4)),
        quantity=controller0.market_tasks.quantity.at[0, 0].set(jnp.int16(6)),
        deadline_step=controller0.market_tasks.deadline_step.at[0, 0].set(
            jnp.int16(718)
        ),
        status=controller0.market_tasks.status.at[0, 0].set(
            TaskStatusV1.ACTIVE
        ),
    )
    controller0 = controller0._replace(market_tasks=market)
    bundle = compile_full_core_action_bundle_v1(
        states, controller0, _controllers(1)
    )
    following = batched_step_sync(states, bundle.action, _events(1), TABLES)
    _, diagnostics = update_full_core_controller_from_effects_v1(
        states, following, controller0, bundle.player0, 0
    )

    # The six shed units were sold while the farmer's six carried units were
    # deposited, so the post-step shed quantity remains six.
    assert int(following.shed[0, 0, 4]) == 6
    assert int(following.unit_inventory[0, 0, 0, 4]) == 0
    assert int(diagnostics.crop_inventory.sold_product_units[0]) == 6
    assert int(diagnostics.effect_mismatch_count[0]) == 0


def test_cross_module_unit_and_plot_reservations_have_no_duplicates() -> None:
    states = _states()
    unit_active = states.unit_active.at[0, 0, 1:3].set(True)
    unit_pos = states.unit_pos.at[0, 0, 0].set(
        jnp.asarray([0, 0], dtype=jnp.int8)
    )
    unit_pos = unit_pos.at[0, 0, 1].set(jnp.asarray([1, 0], dtype=jnp.int8))
    unit_pos = unit_pos.at[0, 0, 2].set(jnp.asarray([2, 0], dtype=jnp.int8))
    tile_kind = states.tile_kind.at[0, 0, 0, 0].set(TileKind.PLANT)
    tile_kind = tile_kind.at[0, 0, 0, 1].set(TileKind.COOP)
    tile_yield = states.tile_yield.at[0, 0, 0, 0].set(2)
    tile_yield = tile_yield.at[0, 0, 0, 1].set(2)
    shed = states.shed.at[0, 0, 0].set(1)
    shed = shed.at[0, 0, 8].set(1)
    states = states._replace(
        unit_active=unit_active,
        unit_pos=unit_pos,
        tile_kind=tile_kind,
        tile_crop=states.tile_crop.at[0, 0, 0, 0].set(0),
        tile_origin_day=states.tile_origin_day.at[0, 0, 0, 0].set(-3),
        tile_yield=tile_yield,
        tile_animal=states.tile_animal.at[0, 0, 0, 1].set(0),
        shed=shed,
    )
    controller = _controllers(1)
    candidates, feasibility, selection = _decide(states, controller, 0)
    selected = selection.selected_candidate_indices[0]
    selected = selected[selected >= 0]
    owners = candidates.owner_unit[0, selected]
    unit_owners = owners[owners >= 0]
    assert int(jnp.unique(unit_owners).shape[0]) == int(unit_owners.shape[0])
    target_ids = candidates.target_id[0, selected][owners >= 0]
    assert int(jnp.unique(target_ids).shape[0]) == int(target_ids.shape[0])
    assert int(selection.internal_resource_conflict[0]) == 0
    assert bool(jnp.all(feasibility.legal_now[0, selected]))


def test_active_crop_route_keeps_seed_reserved_across_steps() -> None:
    states = _states()
    states = states._replace(
        unit_active=states.unit_active.at[0, 0, 1].set(True),
        seeds=states.seeds.at[0, 0, 0].set(1),
    )
    controller = _controllers(1)
    tasks = controller.unit_tasks._replace(
        task_type=controller.unit_tasks.task_type.at[0, 0].set(
            TaskTypeV1.CROP_PRODUCTION
        ),
        owner_unit=controller.unit_tasks.owner_unit.at[0, 0].set(0),
        target_id=controller.unit_tasks.target_id.at[0, 0].set(0),
        target_x=controller.unit_tasks.target_x.at[0, 0].set(0),
        target_y=controller.unit_tasks.target_y.at[0, 0].set(0),
        item_id=controller.unit_tasks.item_id.at[0, 0].set(0),
        quantity=controller.unit_tasks.quantity.at[0, 0].set(1),
        phase=controller.unit_tasks.phase.at[0, 0].set(TaskPhaseV1.MOVE_TO_TARGET),
        deadline_step=controller.unit_tasks.deadline_step.at[0, 0].set(24),
        status=controller.unit_tasks.status.at[0, 0].set(TaskStatusV1.ACTIVE),
    )
    controller = controller._replace(unit_tasks=tasks)
    candidates = build_full_core_candidates_v1(states, controller, TABLES, 0)
    feasibility = evaluate_full_core_feasibility_v1(
        states, candidates, TABLES, 0
    )
    ledger = initialize_e2_ledger_v1(states, controller, 0)
    mask = candidate_mask_from_e2_ledger_v1(
        states, candidates, feasibility, ledger, 0
    )
    competing_seed = (
        (candidates.task_type == TaskTypeV1.CROP_PRODUCTION)
        & (candidates.owner_unit >= 0)
        & (candidates.item_id == 0)
    )
    assert int(ledger.seeds_reserved[0, 0]) == 1
    assert not bool(jnp.any(mask & competing_seed))


def test_returning_crop_route_does_not_reserve_seed_again() -> None:
    states = _states()
    controller = _controllers(1)
    tasks = controller.unit_tasks._replace(
        task_type=controller.unit_tasks.task_type.at[0, 0].set(
            TaskTypeV1.CROP_PRODUCTION
        ),
        owner_unit=controller.unit_tasks.owner_unit.at[0, 0].set(0),
        target_id=controller.unit_tasks.target_id.at[0, 0].set(0),
        target_x=controller.unit_tasks.target_x.at[0, 0].set(0),
        target_y=controller.unit_tasks.target_y.at[0, 0].set(0),
        item_id=controller.unit_tasks.item_id.at[0, 0].set(4),
        quantity=controller.unit_tasks.quantity.at[0, 0].set(1),
        phase=controller.unit_tasks.phase.at[0, 0].set(TaskPhaseV1.MOVE_TO_DEPOT),
        deadline_step=controller.unit_tasks.deadline_step.at[0, 0].set(719),
        status=controller.unit_tasks.status.at[0, 0].set(TaskStatusV1.ACTIVE),
    )
    controller = controller._replace(unit_tasks=tasks)
    ledger = initialize_e2_ledger_v1(states, controller, 0)
    assert int(jnp.sum(ledger.seeds_reserved)) == 0


def test_same_product_cannot_be_bought_and_sold_in_one_selection() -> None:
    states = _states()
    states = states._replace(
        tile_kind=states.tile_kind.at[0, 0, 0, 0].set(TileKind.PLANT),
        tile_crop=states.tile_crop.at[0, 0, 0, 0].set(0),
        shed=states.shed.at[0, 0, 8].set(1),
    )
    controller = _controllers(1)
    candidates, _, selection = _decide(states, controller, 0)
    selected = selection.selected_candidate_indices[0]
    selected = selected[selected >= 0]
    selected_task = candidates.task_type[0, selected]
    selected_item = candidates.item_id[0, selected]
    buys_fertilizer = jnp.any(
        (selected_task == TaskTypeV1.BUY_PRODUCT) & (selected_item == 8)
    )
    sells_fertilizer = jnp.any(
        (
            (selected_task == TaskTypeV1.SELL_INVENTORY)
            | (selected_task == TaskTypeV1.TERMINAL_LIQUIDATION)
        )
        & (selected_item == 8)
    )
    assert not bool(buys_fertilizer & sells_fertilizer)


def test_fertilizer_purchase_requires_unmet_need_and_no_held_unit() -> None:
    states = _states()
    states = states._replace(
        tile_kind=states.tile_kind.at[0, 0, 0, 0].set(TileKind.PLANT),
        tile_crop=states.tile_crop.at[0, 0, 0, 0].set(0),
    )
    controller = _controllers(1)
    candidates = build_full_core_candidates_v1(states, controller, TABLES, 0)
    assert bool(candidates.present[0, 1])

    held = states._replace(
        shed=states.shed.at[0, 0, FERTILIZER_ITEM].set(1)
    )
    candidates = build_full_core_candidates_v1(held, controller, TABLES, 0)
    assert not bool(candidates.present[0, 1])

    already_fertilized = states._replace(
        tile_fertilized_until=states.tile_fertilized_until.at[0, 0, 0, 0].set(3)
    )
    candidates = build_full_core_candidates_v1(
        already_fertilized, controller, TABLES, 0
    )
    assert not bool(candidates.present[0, 1])


def test_purchased_animal_placement_is_a_mandatory_committed_route() -> None:
    states = _states()
    states = states._replace(
        tile_kind=states.tile_kind.at[0, 0, 0, 0].set(TileKind.COOP),
        shed=states.shed.at[0, 0, NUM_PRODUCTS].set(1),
    )
    candidates = build_full_core_candidates_v1(
        states, _controllers(1), TABLES, 0
    )
    place = candidates.task_type == TaskTypeV1.ANIMAL_PLACE
    assert bool(jnp.any(place))
    assert bool(jnp.all(jnp.where(place, candidates.mandatory, True)))


def test_animal_purchase_in_market_slot_seven_is_not_clobbered() -> None:
    states = _states()
    controller = _controllers(1)
    market = controller.market_tasks._replace(
        task_type=controller.market_tasks.task_type.at[0, 7].set(
            TaskTypeV1.ANIMAL_PURCHASE
        ),
        item_id=controller.market_tasks.item_id.at[0, 7].set(NUM_PRODUCTS),
        quantity=controller.market_tasks.quantity.at[0, 7].set(1),
        deadline_step=controller.market_tasks.deadline_step.at[0, 7].set(718),
        status=controller.market_tasks.status.at[0, 7].set(TaskStatusV1.ACTIVE),
    )
    controller = controller._replace(market_tasks=market)
    bundle = compile_full_core_action_bundle_v1(
        states, controller, _controllers(1)
    )
    assert int(bundle.action.market_op[0, 0, 7]) == MarketOp.BUY_ANIMAL
    assert int(bundle.diagnostics.market_compiler_overlap_count[0, 0]) == 0


def test_unified_compiler_exposes_all_18_unit_and_6_market_operations() -> None:
    states = _states()
    states = states._replace(
        unit_active=states.unit_active.at[0, 0, :18].set(True)
    )
    positions = states.unit_pos
    positions = positions.at[0, 0, 0].set(jnp.asarray([4, 4], dtype=jnp.int8))
    positions = positions.at[0, 0, 1].set(jnp.asarray([4, 4], dtype=jnp.int8))
    positions = positions.at[0, 0, 2].set(jnp.asarray([4, 4], dtype=jnp.int8))
    positions = positions.at[0, 0, 3].set(jnp.asarray([4, 4], dtype=jnp.int8))
    positions = positions.at[0, 0, 4].set(jnp.asarray([4, 4], dtype=jnp.int8))
    operation_targets = (
        (4, 4),  # DROP at shed
        (1, 2),  # PICKUP at shed, target elsewhere
        (0, 0),  # PLACE
        (1, 0),  # PLANT
        (2, 0),  # WATER
        (3, 0),  # HARVEST
        (4, 0),  # FERTILIZE
        (0, 1),  # DIG
        (1, 1),  # BUILD_COOP
        (2, 1),  # BUILD_PASTURE
        (3, 1),  # FEED
        (4, 1),  # COLLECT_FERTILIZER
        (0, 2),  # CARE
    )
    for unit, (x, y) in enumerate(operation_targets, start=5):
        positions = positions.at[0, 0, unit].set(
            jnp.asarray([x, y], dtype=jnp.int8)
        )
    positions = positions.at[0, 0, 6].set(
        jnp.asarray([4, 4], dtype=jnp.int8)
    )
    states = states._replace(unit_pos=positions)
    tile_kind = states.tile_kind
    tile_kind = tile_kind.at[0, 0, 0, 0].set(TileKind.COOP)
    tile_kind = tile_kind.at[0, 0, 0, 2].set(TileKind.PLANT)
    tile_kind = tile_kind.at[0, 0, 0, 3].set(TileKind.PLANT)
    tile_kind = tile_kind.at[0, 0, 0, 4].set(TileKind.PLANT)
    tile_kind = tile_kind.at[0, 0, 1, 0].set(TileKind.WEED)
    animal_tiles = states.tile_animal.at[0, 0, 0, 0].set(0)
    animal_tiles = animal_tiles.at[0, 0, 1, 3].set(0)
    animal_tiles = animal_tiles.at[0, 0, 1, 4].set(0)
    animal_tiles = animal_tiles.at[0, 0, 2, 0].set(0)
    states = states._replace(
        tile_kind=tile_kind,
        tile_crop=states.tile_crop.at[0, 0, 0, 2:5].set(0),
        tile_origin_day=states.tile_origin_day.at[0, 0, 0, 3].set(-3),
        tile_yield=states.tile_yield.at[0, 0, 0, 3].set(1),
        tile_animal=animal_tiles,
        tile_flags=states.tile_flags.at[0, 0, 1, 4].set(8),
        shed=states.shed.at[0, 0, 0].set(1),
        seeds=states.seeds.at[0, 0, 0].set(1),
    )
    inventory = states.unit_inventory
    inventory = inventory.at[0, 0, 5, 1].set(1)
    inventory = inventory.at[0, 0, 7, NUM_PRODUCTS].set(1)
    inventory = inventory.at[0, 0, 11, 8].set(1)
    inventory = inventory.at[0, 0, 15, 0].set(1)
    states = states._replace(unit_inventory=inventory)

    controller = _controllers(1)
    tasks = controller.unit_tasks
    unit_task_types = jnp.asarray(
        [
            0,
            TaskTypeV1.CROP_PRODUCTION,
            TaskTypeV1.CROP_PRODUCTION,
            TaskTypeV1.CROP_PRODUCTION,
            TaskTypeV1.CROP_PRODUCTION,
            TaskTypeV1.SAFE_RECOVERY,
            TaskTypeV1.ANIMAL_FEED,
            TaskTypeV1.ANIMAL_PLACE,
            TaskTypeV1.CROP_PRODUCTION,
            TaskTypeV1.WATER_CROP,
            TaskTypeV1.CROP_PRODUCTION,
            TaskTypeV1.APPLY_FERTILIZER,
            TaskTypeV1.CLEAR_OR_REMOVE_TILE,
            TaskTypeV1.BUILD_ANIMAL_STRUCTURE,
            TaskTypeV1.BUILD_ANIMAL_STRUCTURE,
            TaskTypeV1.ANIMAL_FEED,
            TaskTypeV1.ANIMAL_COLLECT_FERTILIZER,
            TaskTypeV1.ANIMAL_CARE,
        ],
        dtype=jnp.int8,
    )
    targets = jnp.asarray(
        [
            (4, 4),
            (4, 3),
            (4, 5),
            (5, 4),
            (3, 4),
            *operation_targets,
        ],
        dtype=jnp.int8,
    )
    items = jnp.asarray(
        [-1, 0, 0, 0, 0, -1, 0, NUM_PRODUCTS, 0, 0, 0, 8, -1, 0, 1, 0, 8, 0],
        dtype=jnp.int8,
    )
    active_task = jnp.arange(18) > 0
    tasks = tasks._replace(
        task_type=tasks.task_type.at[0, :18].set(unit_task_types),
        owner_unit=tasks.owner_unit.at[0, :18].set(jnp.arange(18, dtype=jnp.int8)),
        target_x=tasks.target_x.at[0, :18].set(targets[:, 0]),
        target_y=tasks.target_y.at[0, :18].set(targets[:, 1]),
        target_id=tasks.target_id.at[0, :18].set(
            targets[:, 1].astype(jnp.int16) * 10 + targets[:, 0]
        ),
        item_id=tasks.item_id.at[0, :18].set(items),
        quantity=tasks.quantity.at[0, :18].set(1),
        phase=tasks.phase.at[0, :18].set(TaskPhaseV1.MOVE_TO_TARGET),
        deadline_step=tasks.deadline_step.at[0, :18].set(24),
        status=tasks.status.at[0, :18].set(
            jnp.where(active_task, TaskStatusV1.ACTIVE, TaskStatusV1.EMPTY)
        ),
    )
    market = controller.market_tasks
    market = market._replace(
        task_type=market.task_type.at[0, :6].set(
            jnp.asarray(
                [
                    TaskTypeV1.HIRE_WORKER,
                    TaskTypeV1.BUY_LAND,
                    TaskTypeV1.CROP_PRODUCTION,
                    TaskTypeV1.BUY_PRODUCT,
                    TaskTypeV1.ANIMAL_PURCHASE,
                    TaskTypeV1.SELL_INVENTORY,
                ],
                dtype=jnp.int8,
            )
        ),
        item_id=market.item_id.at[0, :6].set(
            jnp.asarray([-1, -1, 0, 8, NUM_PRODUCTS, 0], dtype=jnp.int8)
        ),
        quantity=market.quantity.at[0, :6].set(1),
        deadline_step=market.deadline_step.at[0, :6].set(718),
        status=market.status.at[0, :6].set(TaskStatusV1.ACTIVE),
    )
    controller = controller._replace(unit_tasks=tasks, market_tasks=market)
    bundle = compile_full_core_action_bundle_v1(
        states, controller, _controllers(1)
    )
    assert bundle.action.unit_op[0, 0, :18].tolist() == list(range(18))
    assert bundle.action.market_op[0, 0, :6].tolist() == [
        MarketOp.HIRE,
        MarketOp.BUY_LAND,
        MarketOp.BUY_SEED,
        MarketOp.BUY_ANIMAL,
        MarketOp.SELL,
        MarketOp.BUY_PRODUCT,
    ]
    assert int(bundle.diagnostics.unit_compiler_overlap_count[0, 0]) == 0
    assert int(bundle.diagnostics.market_compiler_overlap_count[0, 0]) == 0


def test_unified_decision_and_compiler_are_jittable() -> None:
    batch_size = 4
    states = _states((801, 802, 803, 804))
    controller0 = _controllers(batch_size)
    controller1 = _controllers(batch_size)

    @jax.jit
    def decide_and_compile(current, left, right):
        left = clear_finished_tasks_v1(left)
        right = clear_finished_tasks_v1(right)
        left_candidates = build_full_core_candidates_v1(current, left, TABLES, 0)
        left_feasibility = evaluate_full_core_feasibility_v1(
            current, left_candidates, TABLES, 0
        )
        left_selection = select_full_core_candidates_v1(
            current, left_candidates, left_feasibility, left, 0
        )
        right_candidates = build_full_core_candidates_v1(current, right, TABLES, 1)
        right_feasibility = evaluate_full_core_feasibility_v1(
            current, right_candidates, TABLES, 1
        )
        right_selection = select_full_core_candidates_v1(
            current, right_candidates, right_feasibility, right, 1
        )
        bundle = compile_full_core_action_bundle_v1(
            current, left_selection.controller, right_selection.controller
        )
        return (
            bundle.action,
            left_selection.internal_resource_conflict,
            right_selection.internal_resource_conflict,
            bundle.diagnostics,
        )

    action, left_conflict, right_conflict, diagnostics = decide_and_compile(
        states, controller0, controller1
    )
    assert action.unit_op.shape == (batch_size, 2, MAX_UNITS)
    assert int(left_conflict.sum() + right_conflict.sum()) == 0
    assert int(diagnostics.invalid_raw_action_count.sum()) == 0
    assert int(diagnostics.unit_compiler_overlap_count.sum()) == 0
    assert int(diagnostics.market_compiler_overlap_count.sum()) == 0


def test_batch_fuzz_unified_controller_has_no_internal_conflict_or_compiler_overlap() -> None:
    batch_size = 32
    states = _states(tuple(range(700, 700 + batch_size)))
    events = _events(batch_size)
    controller0 = _controllers(batch_size)
    controller1 = _controllers(batch_size)
    conflict = jnp.zeros((batch_size,), dtype=jnp.int32)
    invalid = jnp.zeros((batch_size,), dtype=jnp.int32)
    overlap = jnp.zeros((batch_size,), dtype=jnp.int32)
    mismatch = jnp.zeros((batch_size,), dtype=jnp.int32)

    for _ in range(12):
        _, _, selection0 = _decide(states, controller0, 0)
        _, _, selection1 = _decide(states, controller1, 1)
        conflict = conflict + selection0.internal_resource_conflict + selection1.internal_resource_conflict
        bundle = compile_full_core_action_bundle_v1(
            states, selection0.controller, selection1.controller
        )
        invalid = invalid + jnp.sum(bundle.diagnostics.invalid_raw_action_count, axis=1)
        overlap = overlap + jnp.sum(
            bundle.diagnostics.unit_compiler_overlap_count
            + bundle.diagnostics.market_compiler_overlap_count,
            axis=1,
        )
        following = batched_step_sync(states, bundle.action, events, TABLES)
        controller0, diagnostics0 = update_full_core_controller_from_effects_v1(
            states, following, selection0.controller, bundle.player0, 0
        )
        controller1, diagnostics1 = update_full_core_controller_from_effects_v1(
            states, following, selection1.controller, bundle.player1, 1
        )
        mismatch = mismatch + diagnostics0.effect_mismatch_count + diagnostics1.effect_mismatch_count
        states = following

    assert int(conflict.sum()) == 0
    assert int(invalid.sum()) == 0
    assert int(overlap.sum()) == 0
    assert int(mismatch.sum()) == 0


def test_adversary_can_submit_all_raw_action_families_without_corrupting_player0() -> None:
    states = _states()
    states = states._replace(
        unit_active=states.unit_active.at[0, 1, :18].set(True)
    )
    controller0 = _controllers(1)
    controller1 = _controllers(1)
    _, _, selection0 = _decide(states, controller0, 0)
    bundle = compile_full_core_action_bundle_v1(
        states, selection0.controller, controller1
    )
    adversary_unit_ops = jnp.arange(18, dtype=jnp.int8)
    adversary_market_ops = jnp.arange(1, 7, dtype=jnp.int8)
    action = bundle.action._replace(
        unit_op=bundle.action.unit_op.at[0, 1, :18].set(adversary_unit_ops),
        unit_item=bundle.action.unit_item.at[0, 1, :18].set(0),
        unit_amount=bundle.action.unit_amount.at[0, 1, :18].set(1),
        unit_count=bundle.action.unit_count.at[0, 1].set(18),
        market_op=bundle.action.market_op.at[0, 1, :6].set(adversary_market_ops),
        market_item=bundle.action.market_item.at[0, 1, :6].set(
            jnp.asarray([-1, -1, 0, 0, NUM_PRODUCTS, 0], dtype=jnp.int8)
        ),
        market_amount=bundle.action.market_amount.at[0, 1, :6].set(1),
        market_count=bundle.action.market_count.at[0, 1].set(6),
    )
    following = batched_step_sync(states, action, _events(1), TABLES)
    updated, diagnostics = update_full_core_controller_from_effects_v1(
        states, following, selection0.controller, bundle.player0, 0
    )
    assert int(following.step[0]) == 1
    assert not bool(jnp.any(jnp.isnan(following.money.astype(jnp.float32))))
    assert int(bundle.player0.diagnostics.unit_compiler_overlap_count[0]) == 0
    assert int(bundle.player0.diagnostics.market_compiler_overlap_count[0]) == 0
    assert updated.schema_version.shape == (1,)
    # Shared-market interference may legitimately make a selected order fail;
    # it must be explicit rather than corrupting task state.
    assert bool(
        jnp.all(
            (updated.market_tasks.status != TaskStatusV1.ACTIVE)
            | (updated.market_tasks.failure_code == 0)
        )
    )
