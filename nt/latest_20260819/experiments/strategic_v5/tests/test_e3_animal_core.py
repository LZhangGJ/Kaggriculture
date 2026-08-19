from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from kaggriculture_jax import load_event_bank, load_tables, reset
from kaggriculture_jax.constants import (
    ANIMAL_COST,
    ANIMAL_PRODUCT,
    ANIMAL_STRUCTURE,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    NUM_PRODUCTS,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.simulator import batched_step_sync
from strategic_v5 import (
    FailureCodeV1,
    TaskPhaseV1,
    TaskStatusV1,
    TaskTypeV1,
    build_e3_candidates_v1,
    compile_e3_action_bundle_v1,
    evaluate_e3_feasibility_v1,
    reset_controller_state_v1,
    select_e3_candidates_v1,
    update_e3_controller_from_effects_v1,
)


TABLES = load_tables()


def _states(seed: int = 41):
    return jax.vmap(reset)(jnp.asarray([seed], dtype=jnp.int32))


def _controller():
    value = reset_controller_state_v1()
    return jax.tree.map(lambda item: item[None, ...], value)


def _events(batch_size: int = 1):
    _, bank = load_event_bank()
    return jax.tree.map(lambda value: value[:batch_size], bank)


def _unit_task(task_type: int, *, x: int, y: int, item: int, phase: int = TaskPhaseV1.MOVE_TO_TARGET):
    controller = _controller()
    tasks = controller.unit_tasks._replace(
        task_type=controller.unit_tasks.task_type.at[0, 0].set(task_type),
        target_id=controller.unit_tasks.target_id.at[0, 0].set(y * 10 + x),
        target_x=controller.unit_tasks.target_x.at[0, 0].set(x),
        target_y=controller.unit_tasks.target_y.at[0, 0].set(y),
        item_id=controller.unit_tasks.item_id.at[0, 0].set(item),
        quantity=controller.unit_tasks.quantity.at[0, 0].set(1),
        phase=controller.unit_tasks.phase.at[0, 0].set(phase),
        start_step=controller.unit_tasks.start_step.at[0, 0].set(0),
        deadline_step=controller.unit_tasks.deadline_step.at[0, 0].set(24),
        status=controller.unit_tasks.status.at[0, 0].set(TaskStatusV1.ACTIVE),
    )
    return controller._replace(unit_tasks=tasks)


def _market_task(task_type: int, *, item: int, quantity: int = 1):
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


def _step(states, controller0, controller1=None):
    if controller1 is None:
        controller1 = _controller()
    bundle = compile_e3_action_bundle_v1(states, controller0, controller1)
    following = batched_step_sync(states, bundle.action, _events(states.step.shape[0]), TABLES)
    controller0, diagnostics = update_e3_controller_from_effects_v1(
        states, following, controller0, bundle.player0, 0
    )
    return bundle, following, controller0, diagnostics


@pytest.mark.parametrize("animal", [0, 1, 2])
def test_buy_each_animal_uses_exact_fixed_cost(animal: int) -> None:
    states = _states()
    item = NUM_PRODUCTS + animal
    controller = _market_task(TaskTypeV1.ANIMAL_PURCHASE, item=item)
    bundle, following, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.market_op[0, 0, 0]) == MarketOp.BUY_ANIMAL
    assert int(bundle.action.market_item[0, 0, 0]) == item
    assert int(following.money[0, 0]) == 3000 - ANIMAL_COST[animal]
    assert int(following.shed[0, 0, item]) == 1
    assert int(controller.market_tasks.status[0, 0]) == TaskStatusV1.DONE
    assert int(diagnostics.animal_purchase_success_count[0]) == 1
    assert int(diagnostics.effect_mismatch_count[0]) == 0


@pytest.mark.parametrize("animal", [0, 1, 2])
def test_build_and_place_each_animal_type(animal: int) -> None:
    base = _states()
    states = base._replace(
        unit_pos=base.unit_pos.at[0, 0, 0].set(jnp.asarray([3, 3], dtype=jnp.int8))
    )
    controller = _unit_task(
        TaskTypeV1.BUILD_ANIMAL_STRUCTURE, x=3, y=3, item=animal
    )
    bundle, states, controller, diagnostics = _step(states, controller)
    expected_kind = ANIMAL_STRUCTURE[animal]
    expected_op = UnitOp.BUILD_COOP if expected_kind == TileKind.COOP else UnitOp.BUILD_PASTURE
    assert int(bundle.action.unit_op[0, 0, 0]) == expected_op
    assert int(states.tile_kind[0, 0, 3, 3]) == expected_kind
    assert int(diagnostics.build_success_count[0]) == 1

    item = NUM_PRODUCTS + animal
    states = states._replace(
        unit_inventory=states.unit_inventory.at[0, 0, 0, item].set(1)
    )
    controller = _unit_task(TaskTypeV1.ANIMAL_PLACE, x=3, y=3, item=item)
    bundle, states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.PLACE
    assert int(states.tile_animal[0, 0, 3, 3]) == animal
    assert int(states.unit_inventory[0, 0, 0, item]) == 0
    assert int(controller.unit_tasks.status[0, 0]) == TaskStatusV1.DONE
    assert int(diagnostics.place_success_count[0]) == 1
    assert int(diagnostics.effect_mismatch_count[0]) == 0


def test_feed_and_care_are_effect_checked() -> None:
    base = _states()
    states = base._replace(
        unit_pos=base.unit_pos.at[0, 0, 0].set(jnp.asarray([3, 3], dtype=jnp.int8)),
        unit_inventory=base.unit_inventory.at[0, 0, 0, 0].set(1),
        tile_kind=base.tile_kind.at[0, 0, 3, 3].set(TileKind.COOP),
        tile_animal=base.tile_animal.at[0, 0, 3, 3].set(0),
        tile_origin_day=base.tile_origin_day.at[0, 0, 3, 3].set(0),
    )
    controller = _unit_task(TaskTypeV1.ANIMAL_FEED, x=3, y=3, item=0)
    bundle, states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.FEED
    assert int(states.tile_flags[0, 0, 3, 3]) & FLAG_FED
    assert int(states.unit_inventory[0, 0, 0, 0]) == 0
    assert int(diagnostics.feed_success_count[0]) == 1

    controller = _unit_task(TaskTypeV1.ANIMAL_CARE, x=3, y=3, item=0)
    bundle, states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.CARE
    assert int(states.tile_flags[0, 0, 3, 3]) & FLAG_CARED
    assert int(diagnostics.care_success_count[0]) == 1
    assert int(diagnostics.effect_mismatch_count[0]) == 0


@pytest.mark.parametrize("task_type", [TaskTypeV1.ANIMAL_FEED, TaskTypeV1.ANIMAL_CARE])
def test_feed_and_care_on_day_end_are_not_false_effect_mismatches(
    task_type: int,
) -> None:
    base = _states()
    inventory = base.unit_inventory
    if task_type == TaskTypeV1.ANIMAL_FEED:
        inventory = inventory.at[0, 0, 0, 0].set(1)
    states = base._replace(
        step=jnp.asarray([23], dtype=jnp.int16),
        unit_pos=base.unit_pos.at[0, 0, 0].set(
            jnp.asarray([3, 3], dtype=jnp.int8)
        ),
        unit_inventory=inventory,
        tile_kind=base.tile_kind.at[0, 0, 3, 3].set(TileKind.COOP),
        tile_animal=base.tile_animal.at[0, 0, 3, 3].set(0),
        tile_origin_day=base.tile_origin_day.at[0, 0, 3, 3].set(0),
    )
    controller = _unit_task(task_type, x=3, y=3, item=0)
    bundle, following, controller, diagnostics = _step(states, controller)
    expected_op = (
        UnitOp.FEED if task_type == TaskTypeV1.ANIMAL_FEED else UnitOp.CARE
    )
    assert int(bundle.action.unit_op[0, 0, 0]) == expected_op
    assert int(following.step[0]) == 24
    assert int(following.tile_animal[0, 0, 3, 3]) == 0
    assert int(controller.unit_tasks.status[0, 0]) == TaskStatusV1.DONE
    assert int(diagnostics.effect_mismatch_count[0]) == 0


def test_care_on_escape_boundary_is_counted_as_executed_not_effect_mismatch() -> None:
    base = _states()
    states = base._replace(
        step=jnp.asarray([23], dtype=jnp.int16),
        unit_pos=base.unit_pos.at[0, 0, 0].set(
            jnp.asarray([3, 3], dtype=jnp.int8)
        ),
        tile_kind=base.tile_kind.at[0, 0, 3, 3].set(TileKind.COOP),
        tile_animal=base.tile_animal.at[0, 0, 3, 3].set(0),
        tile_origin_day=base.tile_origin_day.at[0, 0, 3, 3].set(0),
        tile_neglect=base.tile_neglect.at[0, 0, 3, 3].set(1),
    )
    controller = _unit_task(TaskTypeV1.ANIMAL_CARE, x=3, y=3, item=0)
    bundle, following, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.CARE
    assert int(following.tile_animal[0, 0, 3, 3]) == -1
    assert int(controller.unit_tasks.status[0, 0]) == TaskStatusV1.DONE
    assert int(diagnostics.effect_mismatch_count[0]) == 0


def test_collect_product_deposit_and_sell_closes_cash_loop() -> None:
    base = _states()
    states = base._replace(
        step=jnp.asarray([100], dtype=jnp.int16),
        unit_pos=base.unit_pos.at[0, 0, 0].set(jnp.asarray([4, 3], dtype=jnp.int8)),
        tile_kind=base.tile_kind.at[0, 0, 3, 4].set(TileKind.COOP),
        tile_animal=base.tile_animal.at[0, 0, 3, 4].set(0),
        tile_origin_day=base.tile_origin_day.at[0, 0, 3, 4].set(0),
        tile_yield=base.tile_yield.at[0, 0, 3, 4].set(2),
    )
    product = ANIMAL_PRODUCT[0]
    controller = _unit_task(
        TaskTypeV1.ANIMAL_COLLECT_PRODUCT, x=4, y=3, item=product
    )
    controller = controller._replace(
        unit_tasks=controller.unit_tasks._replace(
            deadline_step=controller.unit_tasks.deadline_step.at[0, 0].set(120)
        )
    )
    bundle, states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.HARVEST
    assert int(states.unit_inventory[0, 0, 0, product]) == 2
    assert int(controller.unit_tasks.phase[0, 0]) == TaskPhaseV1.MOVE_TO_DEPOT
    assert int(diagnostics.harvest_success_count[0]) == 1

    bundle, states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.SOUTH
    money_before_sale = int(states.money[0, 0])
    bundle, states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.DROP
    assert int(bundle.action.market_op[0, 0, 7]) == MarketOp.SELL
    assert int(bundle.action.market_amount[0, 0, 7]) == 2
    assert int(states.unit_inventory[0, 0, 0, product]) == 0
    assert int(states.shed[0, 0, product]) == 0
    assert int(states.money[0, 0]) > money_before_sale
    assert int(controller.unit_tasks.status[0, 0]) == TaskStatusV1.DONE
    assert int(diagnostics.deposit_success_count[0]) == 1
    assert int(diagnostics.sold_product_units[0]) == 2


def test_collect_fertilizer_then_deposit() -> None:
    base = _states()
    states = base._replace(
        step=jnp.asarray([48], dtype=jnp.int16),
        unit_pos=base.unit_pos.at[0, 0, 0].set(jnp.asarray([4, 3], dtype=jnp.int8)),
        tile_kind=base.tile_kind.at[0, 0, 3, 4].set(TileKind.COOP),
        tile_animal=base.tile_animal.at[0, 0, 3, 4].set(0),
        tile_flags=base.tile_flags.at[0, 0, 3, 4].set(FLAG_FERTILIZER_AVAILABLE),
    )
    controller = _unit_task(
        TaskTypeV1.ANIMAL_COLLECT_FERTILIZER, x=4, y=3, item=8
    )
    controller = controller._replace(
        unit_tasks=controller.unit_tasks._replace(
            deadline_step=controller.unit_tasks.deadline_step.at[0, 0].set(72)
        )
    )
    bundle, states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.COLLECT_FERTILIZER
    assert int(states.unit_inventory[0, 0, 0, 8]) == 1
    assert int(diagnostics.collect_fertilizer_success_count[0]) == 1
    bundle, states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.SOUTH
    bundle, states, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.DROP
    assert int(states.shed[0, 0, 8]) == 1
    assert int(controller.unit_tasks.status[0, 0]) == TaskStatusV1.DONE


def test_missing_feed_is_explicit_resource_failure_not_hidden_progress() -> None:
    base = _states()
    states = base._replace(
        unit_pos=base.unit_pos.at[0, 0, 0].set(jnp.asarray([3, 3], dtype=jnp.int8)),
        tile_kind=base.tile_kind.at[0, 0, 3, 3].set(TileKind.COOP),
        tile_animal=base.tile_animal.at[0, 0, 3, 3].set(0),
    )
    controller = _unit_task(TaskTypeV1.ANIMAL_FEED, x=3, y=3, item=0)
    bundle, _, controller, diagnostics = _step(states, controller)
    assert int(bundle.action.unit_op[0, 0, 0]) == UnitOp.PASS
    assert int(controller.unit_tasks.status[0, 0]) == TaskStatusV1.FAILED
    assert int(controller.unit_tasks.failure_code[0, 0]) == FailureCodeV1.RESOURCE_UNAVAILABLE
    assert int(diagnostics.resource_unavailable_count[0]) == 1


def test_one_pasture_cannot_reserve_both_cow_and_sheep_purchase() -> None:
    base = _states()
    states = base._replace(
        tile_kind=base.tile_kind.at[0, 0, 3, 3].set(TileKind.PASTURE)
    )
    controller = _controller()
    candidates = build_e3_candidates_v1(states, controller, TABLES, 0)
    feasibility = evaluate_e3_feasibility_v1(states, candidates, TABLES, 0)
    selected = select_e3_candidates_v1(states, candidates, feasibility, controller, 0)
    indices = selected.selected_candidate_indices[0]
    safe = jnp.clip(indices, 0, candidates.task_type.shape[1] - 1)
    purchased = (indices >= 0) & (
        candidates.task_type[0, safe] == TaskTypeV1.ANIMAL_PURCHASE
    )
    assert int(jnp.sum(purchased)) == 1
    assert int(selected.internal_resource_conflict[0]) == 0


def test_e3_candidate_feasibility_and_compiler_are_jittable() -> None:
    base = _states()
    states = base._replace(
        tile_kind=base.tile_kind.at[0, 0, 3, 3].set(TileKind.COOP),
        tile_animal=base.tile_animal.at[0, 0, 3, 3].set(0),
        shed=base.shed.at[0, 0, 0].set(2),
    )
    controller = _controller()
    build = jax.jit(build_e3_candidates_v1, static_argnums=3)
    candidates = build(states, controller, TABLES, 0)
    feasible = jax.jit(evaluate_e3_feasibility_v1, static_argnums=3)(
        states, candidates, TABLES, 0
    )
    assert bool(jnp.any(feasible.legal_now))
    compile_player = jax.jit(
        lambda current, control: compile_e3_action_bundle_v1(
            current, control, _controller()
        ).player0
    )
    action = compile_player(
        states, _unit_task(TaskTypeV1.ANIMAL_FEED, x=3, y=3, item=0)
    )
    assert int(action.diagnostics.invalid_raw_action_count[0]) == 0
