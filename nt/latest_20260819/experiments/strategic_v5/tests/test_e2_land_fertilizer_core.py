from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax import load_event_bank, load_tables, reset
from kaggriculture_jax.constants import (
    FLAG_WATERED,
    MARKET_MIN_INVENTORY,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.simulator import batched_step_sync
from strategic_v5 import (
    FERTILIZER_ITEM,
    TaskStatusV1,
    TaskTypeV1,
    build_e2_candidates_v1,
    compile_e2_action_bundle_v1,
    compile_e2_player_action_v1,
    evaluate_e2_feasibility_v1,
    initialize_e2_ledger_v1,
    reset_controller_state_v1,
    select_e2_candidates_v1,
    update_e2_controller_from_effects_v1,
)


TABLES = load_tables()


def _states(seed: int = 17):
    return jax.vmap(reset)(jnp.asarray([seed], dtype=jnp.int32))


def _controller():
    value = reset_controller_state_v1()
    return jax.tree.map(lambda item: item[None, ...], value)


def _events(batch_size: int = 1):
    _, bank = load_event_bank()
    return jax.tree.map(lambda value: value[:batch_size], bank)


def _market_task(task_type: int, item: int = -1, quantity: int = 1):
    controller = _controller()
    tasks = controller.market_tasks._replace(
        task_type=controller.market_tasks.task_type.at[0, 0].set(task_type),
        item_id=controller.market_tasks.item_id.at[0, 0].set(item),
        quantity=controller.market_tasks.quantity.at[0, 0].set(quantity),
        start_step=controller.market_tasks.start_step.at[0, 0].set(0),
        deadline_step=controller.market_tasks.deadline_step.at[0, 0].set(718),
        status=controller.market_tasks.status.at[0, 0].set(TaskStatusV1.ACTIVE),
    )
    return controller._replace(market_tasks=tasks)


def _apply_task(x: int, y: int):
    controller = _controller()
    tasks = controller.unit_tasks._replace(
        task_type=controller.unit_tasks.task_type.at[0, 0].set(
            TaskTypeV1.APPLY_FERTILIZER
        ),
        target_id=controller.unit_tasks.target_id.at[0, 0].set(y * 10 + x),
        target_x=controller.unit_tasks.target_x.at[0, 0].set(x),
        target_y=controller.unit_tasks.target_y.at[0, 0].set(y),
        item_id=controller.unit_tasks.item_id.at[0, 0].set(FERTILIZER_ITEM),
        quantity=controller.unit_tasks.quantity.at[0, 0].set(1),
        start_step=controller.unit_tasks.start_step.at[0, 0].set(0),
        deadline_step=controller.unit_tasks.deadline_step.at[0, 0].set(718),
        status=controller.unit_tasks.status.at[0, 0].set(TaskStatusV1.ACTIVE),
    )
    return controller._replace(unit_tasks=tasks)


def _step(states, controller0, controller1=None):
    if controller1 is None:
        controller1 = _controller()
    bundle = compile_e2_action_bundle_v1(states, controller0, controller1)
    next_states = batched_step_sync(states, bundle.action, _events(states.step.shape[0]), TABLES)
    next_controller, diagnostics = update_e2_controller_from_effects_v1(
        states, next_states, controller0, bundle.player0, 0
    )
    return bundle, next_states, next_controller, diagnostics


def test_buy_land_compiles_and_unlocks_exact_next_quadrant() -> None:
    states = _states()
    controller = _market_task(TaskTypeV1.BUY_LAND)
    bundle, next_states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.market_op[0, 0, 0]) == MarketOp.BUY_LAND
    assert int(next_states.money[0, 0]) == 2000
    assert int(next_states.unlocked_count[0, 0]) == 2
    assert int(next_states.tile_kind[0, 0, 0, 5]) == TileKind.EMPTY
    assert int(next_states.tile_kind[0, 0, 5, 0]) == TileKind.LOCKED
    assert int(controller.market_tasks.status[0, 0]) == TaskStatusV1.DONE
    assert int(diagnostics.land_purchase_success_count[0]) == 1
    assert int(diagnostics.effect_mismatch_count[0]) == 0


def test_buy_fertilizer_uses_official_post_buy_quote() -> None:
    states = _states()
    controller = _market_task(TaskTypeV1.BUY_PRODUCT, FERTILIZER_ITEM)
    index = int(states.market_inventory[0, FERTILIZER_ITEM]) - 1 - MARKET_MIN_INVENTORY
    expected_price = int(TABLES.market_price[FERTILIZER_ITEM, index])
    bundle, next_states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.market_op[0, 0, 0]) == MarketOp.BUY_PRODUCT
    assert int(next_states.shed[0, 0, FERTILIZER_ITEM]) == 1
    assert int(next_states.money[0, 0]) == 3000 - expected_price
    assert int(controller.market_tasks.status[0, 0]) == TaskStatusV1.DONE
    assert int(diagnostics.fertilizer_buy_success_count[0]) == 1
    assert int(diagnostics.effect_mismatch_count[0]) == 0


def test_fertilizer_task_picks_up_moves_applies_and_checks_effect() -> None:
    base = _states()
    states = base._replace(
        shed=base.shed.at[0, 0, FERTILIZER_ITEM].set(1),
        tile_kind=base.tile_kind.at[0, 0, 3, 3].set(TileKind.PLANT),
        tile_crop=base.tile_crop.at[0, 0, 3, 3].set(0),
        tile_origin_day=base.tile_origin_day.at[0, 0, 3, 3].set(0),
        tile_flags=base.tile_flags.at[0, 0, 3, 3].set(FLAG_WATERED),
    )
    controller = _apply_task(3, 3)

    bundle, states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.PICKUP
    assert int(states.unit_inventory[0, 0, 0, FERTILIZER_ITEM]) == 1
    assert int(diagnostics.fertilizer_pickup_success_count[0]) == 1

    bundle, states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.WEST
    bundle, states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.NORTH
    bundle, states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.FERTILIZE
    assert int(states.unit_inventory[0, 0, 0, FERTILIZER_ITEM]) == 0
    assert int(states.tile_fertilized_until[0, 0, 3, 3]) == 2
    assert int(controller.unit_tasks.status[0, 0]) == TaskStatusV1.DONE
    assert int(diagnostics.fertilizer_apply_success_count[0]) == 1
    assert int(diagnostics.effect_mismatch_count[0]) == 0


def test_repeat_fertilize_extends_to_current_day_plus_two() -> None:
    base = _states()
    states = base._replace(
        step=jnp.asarray([24], dtype=jnp.int16),
        unit_pos=base.unit_pos.at[0, 0, 0].set(jnp.asarray([3, 3], dtype=jnp.int8)),
        unit_inventory=base.unit_inventory.at[0, 0, 0, FERTILIZER_ITEM].set(1),
        tile_kind=base.tile_kind.at[0, 0, 3, 3].set(TileKind.PLANT),
        tile_crop=base.tile_crop.at[0, 0, 3, 3].set(0),
        tile_origin_day=base.tile_origin_day.at[0, 0, 3, 3].set(0),
        tile_fertilized_until=base.tile_fertilized_until.at[0, 0, 3, 3].set(2),
    )
    controller = _apply_task(3, 3)
    bundle, next_states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.FERTILIZE
    assert int(next_states.tile_fertilized_until[0, 0, 3, 3]) == 3
    assert int(controller.unit_tasks.status[0, 0]) == TaskStatusV1.DONE
    assert int(diagnostics.effect_mismatch_count[0]) == 0


def test_one_shed_fertilizer_cannot_be_reserved_by_two_units() -> None:
    base = _states()
    states = base._replace(
        unit_active=base.unit_active.at[0, 0, 1].set(True),
        unit_pos=base.unit_pos.at[0, 0, 1].set(jnp.asarray([5, 4], dtype=jnp.int8)),
        shed=base.shed.at[0, 0, FERTILIZER_ITEM].set(1),
        tile_kind=base.tile_kind.at[0, 0, 3, 3].set(TileKind.PLANT),
        tile_crop=base.tile_crop.at[0, 0, 3, 3].set(0),
    )
    controller = _controller()
    candidates = build_e2_candidates_v1(states, controller, TABLES, 0)
    feasibility = evaluate_e2_feasibility_v1(states, candidates, TABLES, 0)
    selected = select_e2_candidates_v1(
        states,
        candidates,
        feasibility,
        controller,
        initialize_e2_ledger_v1(states, controller, 0),
        0,
    )
    indices = selected.selected_candidate_indices[0]
    safe = jnp.clip(indices, 0, candidates.task_type.shape[1] - 1)
    task_types = candidates.task_type[0, safe]
    apply_count = jnp.sum((indices >= 0) & (task_types == TaskTypeV1.APPLY_FERTILIZER))
    assert int(apply_count) == 1
    assert int(selected.ledger.shed_reserved_out[0, FERTILIZER_ITEM]) == 1
    assert int(selected.internal_resource_conflict[0]) == 0


def test_e2_candidate_and_compile_paths_are_jittable() -> None:
    base = _states()
    states = base._replace(
        shed=base.shed.at[0, 0, FERTILIZER_ITEM].set(1),
        tile_kind=base.tile_kind.at[0, 0, 3, 3].set(TileKind.PLANT),
        tile_crop=base.tile_crop.at[0, 0, 3, 3].set(0),
    )
    controller = _controller()
    build = jax.jit(build_e2_candidates_v1, static_argnums=3)
    first = build(states, controller, TABLES, 0)
    second = build(states, controller, TABLES, 0)
    equal = jax.tree.map(jnp.array_equal, first, second)
    assert all(bool(value) for value in jax.tree.leaves(equal))
    compile_player = jax.jit(compile_e2_player_action_v1, static_argnums=2)
    action = compile_player(states, _apply_task(3, 3), 0)
    assert int(action.diagnostics.invalid_raw_action_count[0]) == 0


def test_apply_route_that_cannot_finish_before_day_reset_is_masked() -> None:
    base = _states()
    states = base._replace(
        step=jnp.asarray([23], dtype=jnp.int16),
        shed=base.shed.at[0, 0, FERTILIZER_ITEM].set(1),
        tile_kind=base.tile_kind.at[0, 0, 3, 3].set(TileKind.PLANT),
        tile_crop=base.tile_crop.at[0, 0, 3, 3].set(0),
    )
    controller = _controller()
    candidates = build_e2_candidates_v1(states, controller, TABLES, 0)
    feasibility = evaluate_e2_feasibility_v1(states, candidates, TABLES, 0)
    apply = candidates.task_type == TaskTypeV1.APPLY_FERTILIZER
    assert not bool(jnp.any(feasibility.legal_now & apply))


def test_last_turn_apply_at_target_is_legal_and_persists() -> None:
    base = _states()
    states = base._replace(
        step=jnp.asarray([23], dtype=jnp.int16),
        unit_pos=base.unit_pos.at[0, 0, 0].set(jnp.asarray([3, 3], dtype=jnp.int8)),
        unit_inventory=base.unit_inventory.at[0, 0, 0, FERTILIZER_ITEM].set(1),
        tile_kind=base.tile_kind.at[0, 0, 3, 3].set(TileKind.PLANT),
        tile_crop=base.tile_crop.at[0, 0, 3, 3].set(0),
        tile_origin_day=base.tile_origin_day.at[0, 0, 3, 3].set(0),
    )
    controller = _apply_task(3, 3)
    bundle, following, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.FERTILIZE
    assert int(following.step[0]) == 24
    assert int(following.tile_fertilized_until[0, 0, 3, 3]) == 2
    assert int(controller.unit_tasks.status[0, 0]) == TaskStatusV1.DONE
    assert int(diagnostics.effect_mismatch_count[0]) == 0
