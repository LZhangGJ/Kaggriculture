"""Lossless behavior diagnostics for H1B RULE_ONLY variants.

The accepted staged evaluator remains the source of environment/controller
semantics.  This module wraps its execute kernel and accumulates only small,
fixed-shape counters for the rule-controlled seat.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    MAX_MARKET_ORDERS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    TileKind,
)
from kaggriculture_jax.types import StaticTables

from .constants import TaskStatusV1, TaskTypeV1
from .e4_executor import (
    compile_full_core_action_bundle_v1,
    update_full_core_controller_from_effects_v1,
)
from .e5_rollout import _task_successes
from .h_arena import HHybridCarryV1
from .h_eval_v2 import h_execute_selected_step_v2
from .schema import ControllerStateV1


TASK_TYPE_COUNT_V1 = max(int(value) for value in TaskTypeV1) + 1
PHASE_COUNT_V1 = 4
MARKET_OP_COUNT_V1 = 7
PHASE_END_STEPS_V1 = jnp.asarray((239, 479, 647, 718), dtype=jnp.int32)


class RuleBehaviorDiagnosticsV1(NamedTuple):
    active_slot_steps_by_phase_task: jax.Array
    new_assignments_by_phase_task: jax.Array
    successful_effects_by_phase_task: jax.Array
    market_orders_by_phase_op: jax.Array
    market_units_by_phase_op: jax.Array
    market_orders_by_phase_op_item: jax.Array
    market_units_by_phase_op_item: jax.Array
    terminal_orders_by_phase: jax.Array
    terminal_units_by_phase: jax.Array
    realized_sold_product_units_by_phase: jax.Array
    gross_cash_in_by_phase: jax.Array
    gross_cash_out_by_phase: jax.Array
    phase_end_game_count: jax.Array
    phase_end_money_sum: jax.Array
    phase_end_unlocked_count_sum: jax.Array
    phase_end_active_units_sum: jax.Array
    phase_end_crop_tiles_sum: jax.Array
    phase_end_animal_tiles_sum: jax.Array
    phase_end_occupied_tiles_sum: jax.Array
    phase_end_product_units_sum: jax.Array
    phase_end_nonproduct_units_sum: jax.Array
    phase_end_product_mark_value_sum: jax.Array


def empty_rule_behavior_diagnostics_v1() -> RuleBehaviorDiagnosticsV1:
    phase_task = lambda: jnp.zeros(
        (PHASE_COUNT_V1, TASK_TYPE_COUNT_V1), dtype=jnp.int32
    )
    phase_op = lambda: jnp.zeros(
        (PHASE_COUNT_V1, MARKET_OP_COUNT_V1), dtype=jnp.int32
    )
    phase_op_item = lambda: jnp.zeros(
        (PHASE_COUNT_V1, MARKET_OP_COUNT_V1, NUM_SHED_ITEMS), dtype=jnp.int32
    )
    phase = lambda: jnp.zeros((PHASE_COUNT_V1,), dtype=jnp.int32)
    return RuleBehaviorDiagnosticsV1(
        active_slot_steps_by_phase_task=phase_task(),
        new_assignments_by_phase_task=phase_task(),
        successful_effects_by_phase_task=phase_task(),
        market_orders_by_phase_op=phase_op(),
        market_units_by_phase_op=phase_op(),
        market_orders_by_phase_op_item=phase_op_item(),
        market_units_by_phase_op_item=phase_op_item(),
        terminal_orders_by_phase=phase(),
        terminal_units_by_phase=phase(),
        realized_sold_product_units_by_phase=phase(),
        gross_cash_in_by_phase=phase(),
        gross_cash_out_by_phase=phase(),
        phase_end_game_count=phase(),
        phase_end_money_sum=phase(),
        phase_end_unlocked_count_sum=phase(),
        phase_end_active_units_sum=phase(),
        phase_end_crop_tiles_sum=phase(),
        phase_end_animal_tiles_sum=phase(),
        phase_end_occupied_tiles_sum=phase(),
        phase_end_product_units_sum=phase(),
        phase_end_nonproduct_units_sum=phase(),
        phase_end_product_mark_value_sum=phase(),
    )


def _phase_index(step: jax.Array) -> jax.Array:
    scalar = step[0].astype(jnp.int32)
    return jnp.where(
        scalar < 240,
        0,
        jnp.where(scalar < 480, 1, jnp.where(scalar < 648, 2, 3)),
    ).astype(jnp.int32)


def _one_hot_task_counts(task_type: jax.Array, mask: jax.Array) -> jax.Array:
    safe = jnp.clip(task_type.astype(jnp.int32), 0, TASK_TYPE_COUNT_V1 - 1)
    encoded = jax.nn.one_hot(safe, TASK_TYPE_COUNT_V1, dtype=jnp.int32)
    return jnp.sum(encoded * mask[..., None].astype(jnp.int32), axis=(0, 1))


def _active_task_counts(controller: ControllerStateV1) -> jax.Array:
    unit = _one_hot_task_counts(
        controller.unit_tasks.task_type,
        controller.unit_tasks.status == TaskStatusV1.ACTIVE,
    )
    market = _one_hot_task_counts(
        controller.market_tasks.task_type,
        controller.market_tasks.status == TaskStatusV1.ACTIVE,
    )
    return unit + market


def _new_assignment_counts(
    prior: ControllerStateV1, selected: ControllerStateV1
) -> jax.Array:
    prior_unit = prior.unit_tasks
    unit = selected.unit_tasks
    unit_same = (
        (prior_unit.status == TaskStatusV1.ACTIVE)
        & (unit.task_type == prior_unit.task_type)
        & (unit.owner_unit == prior_unit.owner_unit)
        & (unit.target_id == prior_unit.target_id)
        & (unit.target_x == prior_unit.target_x)
        & (unit.target_y == prior_unit.target_y)
        & (unit.item_id == prior_unit.item_id)
        & (unit.quantity == prior_unit.quantity)
        & (unit.start_step == prior_unit.start_step)
    )
    unit_new = (unit.status == TaskStatusV1.ACTIVE) & (~unit_same)

    prior_market = prior.market_tasks
    market = selected.market_tasks
    market_same = (
        (prior_market.status == TaskStatusV1.ACTIVE)
        & (market.task_type == prior_market.task_type)
        & (market.item_id == prior_market.item_id)
        & (market.quantity == prior_market.quantity)
        & (market.start_step == prior_market.start_step)
        & (market.deadline_step == prior_market.deadline_step)
    )
    market_new = (market.status == TaskStatusV1.ACTIVE) & (~market_same)
    return _one_hot_task_counts(unit.task_type, unit_new) + _one_hot_task_counts(
        market.task_type, market_new
    )


def _market_counts(player_action) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    op = player_action.market_op.astype(jnp.int32)
    item = player_action.market_item.astype(jnp.int32)
    amount = player_action.market_amount.astype(jnp.int32)
    slot = jnp.arange(MAX_MARKET_ORDERS, dtype=jnp.int32)[None, :]
    in_count = slot < player_action.market_count[:, None].astype(jnp.int32)
    valid_op = in_count & (op > 0) & (op < MARKET_OP_COUNT_V1)
    op_hot = jax.nn.one_hot(
        jnp.clip(op, 0, MARKET_OP_COUNT_V1 - 1),
        MARKET_OP_COUNT_V1,
        dtype=jnp.int32,
    )
    item_hot = jax.nn.one_hot(
        jnp.clip(item, 0, NUM_SHED_ITEMS - 1), NUM_SHED_ITEMS, dtype=jnp.int32
    )
    valid_item = valid_op & (item >= 0) & (item < NUM_SHED_ITEMS)
    orders_by_op = jnp.sum(op_hot * valid_op[..., None], axis=(0, 1))
    units_by_op = jnp.sum(
        op_hot * valid_op[..., None] * amount[..., None], axis=(0, 1)
    )
    pair_hot = op_hot[..., :, None] * item_hot[..., None, :]
    orders_by_pair = jnp.sum(pair_hot * valid_item[..., None, None], axis=(0, 1))
    units_by_pair = jnp.sum(
        pair_hot * valid_item[..., None, None] * amount[..., None, None],
        axis=(0, 1),
    )
    return orders_by_op, units_by_op, orders_by_pair, units_by_pair


def _terminal_order_counts(controller: ControllerStateV1) -> tuple[jax.Array, jax.Array]:
    tasks = controller.market_tasks
    active = (tasks.status == TaskStatusV1.ACTIVE) & (
        tasks.task_type == TaskTypeV1.TERMINAL_LIQUIDATION
    )
    return (
        jnp.sum(active, dtype=jnp.int32),
        jnp.sum(jnp.where(active, jnp.maximum(tasks.quantity, 1), 0), dtype=jnp.int32),
    )


def _add_behavior_step(
    prior: RuleBehaviorDiagnosticsV1,
    carry: HHybridCarryV1,
    next_carry: HHybridCarryV1,
    selected: ControllerStateV1,
    prior_controller: ControllerStateV1,
    player_action,
    effect,
    player: int,
) -> RuleBehaviorDiagnosticsV1:
    states = carry.environment_state
    next_states = next_carry.environment_state
    phase = _phase_index(states.step)
    active = _active_task_counts(selected)
    new = _new_assignment_counts(prior_controller, selected)
    successes = jnp.sum(
        _task_successes(
            effect, player_action.crop_inventory.diagnostics.terminal_sell_orders
        ),
        axis=0,
        dtype=jnp.int32,
    )
    orders_op, units_op, orders_pair, units_pair = _market_counts(player_action)
    terminal_orders, terminal_units = _terminal_order_counts(selected)
    sold = jnp.sum(
        effect.e3.sold_product_units + effect.crop_inventory.sold_product_units,
        dtype=jnp.int32,
    )

    delta = (
        next_states.money[:, player].astype(jnp.int32)
        - states.money[:, player].astype(jnp.int32)
    )
    cash_in = jnp.sum(jnp.maximum(delta, 0), dtype=jnp.int32)
    cash_out = jnp.sum(jnp.maximum(-delta, 0), dtype=jnp.int32)

    phase_end = states.step[0].astype(jnp.int32) == PHASE_END_STEPS_V1[phase]
    phase_end_i32 = phase_end.astype(jnp.int32)
    products = (
        jnp.sum(next_states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int32), axis=-1)
        + jnp.sum(
            next_states.unit_inventory[:, player, :, :NUM_PRODUCTS].astype(jnp.int32),
            axis=(1, 2),
        )
    )
    nonproducts = (
        jnp.sum(next_states.shed[:, player, NUM_PRODUCTS:].astype(jnp.int32), axis=-1)
        + jnp.sum(
            next_states.unit_inventory[:, player, :, NUM_PRODUCTS:].astype(jnp.int32),
            axis=(1, 2),
        )
    )
    product_by_item = (
        next_states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int32)
        + jnp.sum(
            next_states.unit_inventory[:, player, :, :NUM_PRODUCTS].astype(jnp.int32),
            axis=1,
        )
    )
    product_mark_value = jnp.sum(
        product_by_item * next_states.market_price.astype(jnp.int32), axis=-1
    )
    kind = next_states.tile_kind[:, player]
    crop_tiles = jnp.sum(kind == TileKind.PLANT, axis=(1, 2), dtype=jnp.int32)
    animal_tiles = jnp.sum(
        (kind == TileKind.COOP) | (kind == TileKind.PASTURE),
        axis=(1, 2),
        dtype=jnp.int32,
    )
    occupied_tiles = jnp.sum(
        (kind != TileKind.EMPTY) & (kind != TileKind.LOCKED),
        axis=(1, 2),
        dtype=jnp.int32,
    )
    batch_size = states.step.shape[0]

    def add_at(array: jax.Array, value: jax.Array) -> jax.Array:
        return array.at[phase].add(value)

    return prior._replace(
        active_slot_steps_by_phase_task=add_at(
            prior.active_slot_steps_by_phase_task, active
        ),
        new_assignments_by_phase_task=add_at(
            prior.new_assignments_by_phase_task, new
        ),
        successful_effects_by_phase_task=add_at(
            prior.successful_effects_by_phase_task, successes
        ),
        market_orders_by_phase_op=add_at(prior.market_orders_by_phase_op, orders_op),
        market_units_by_phase_op=add_at(prior.market_units_by_phase_op, units_op),
        market_orders_by_phase_op_item=add_at(
            prior.market_orders_by_phase_op_item, orders_pair
        ),
        market_units_by_phase_op_item=add_at(
            prior.market_units_by_phase_op_item, units_pair
        ),
        terminal_orders_by_phase=add_at(prior.terminal_orders_by_phase, terminal_orders),
        terminal_units_by_phase=add_at(prior.terminal_units_by_phase, terminal_units),
        realized_sold_product_units_by_phase=add_at(
            prior.realized_sold_product_units_by_phase, sold
        ),
        gross_cash_in_by_phase=add_at(prior.gross_cash_in_by_phase, cash_in),
        gross_cash_out_by_phase=add_at(prior.gross_cash_out_by_phase, cash_out),
        phase_end_game_count=add_at(
            prior.phase_end_game_count, phase_end_i32 * batch_size
        ),
        phase_end_money_sum=add_at(
            prior.phase_end_money_sum,
            phase_end_i32 * jnp.sum(next_states.money[:, player], dtype=jnp.int32),
        ),
        phase_end_unlocked_count_sum=add_at(
            prior.phase_end_unlocked_count_sum,
            phase_end_i32
            * jnp.sum(next_states.unlocked_count[:, player], dtype=jnp.int32),
        ),
        phase_end_active_units_sum=add_at(
            prior.phase_end_active_units_sum,
            phase_end_i32
            * jnp.sum(next_states.unit_active[:, player], dtype=jnp.int32),
        ),
        phase_end_crop_tiles_sum=add_at(
            prior.phase_end_crop_tiles_sum,
            phase_end_i32 * jnp.sum(crop_tiles, dtype=jnp.int32),
        ),
        phase_end_animal_tiles_sum=add_at(
            prior.phase_end_animal_tiles_sum,
            phase_end_i32 * jnp.sum(animal_tiles, dtype=jnp.int32),
        ),
        phase_end_occupied_tiles_sum=add_at(
            prior.phase_end_occupied_tiles_sum,
            phase_end_i32 * jnp.sum(occupied_tiles, dtype=jnp.int32),
        ),
        phase_end_product_units_sum=add_at(
            prior.phase_end_product_units_sum,
            phase_end_i32 * jnp.sum(products, dtype=jnp.int32),
        ),
        phase_end_nonproduct_units_sum=add_at(
            prior.phase_end_nonproduct_units_sum,
            phase_end_i32 * jnp.sum(nonproducts, dtype=jnp.int32),
        ),
        phase_end_product_mark_value_sum=add_at(
            prior.phase_end_product_mark_value_sum,
            phase_end_i32 * jnp.sum(product_mark_value, dtype=jnp.int32),
        ),
    )


def make_h_execute_rule_diagnostic_kernel_v1(learned_player: int):
    """Return an execute kernel that preserves H evaluator results exactly."""

    if learned_player not in (0, 1):
        raise ValueError("learned_player must be 0 or 1")
    rule_player = 1 - learned_player

    def execute(
        carry: HHybridCarryV1,
        behavior: RuleBehaviorDiagnosticsV1,
        tables: StaticTables,
        selected0: ControllerStateV1,
        selected1: ControllerStateV1,
        conflict0: jax.Array,
        conflict1: jax.Array,
        next_key: jax.Array,
    ) -> tuple[HHybridCarryV1, RuleBehaviorDiagnosticsV1]:
        next_carry = h_execute_selected_step_v2(
            carry,
            tables,
            selected0,
            selected1,
            conflict0,
            conflict1,
            next_key,
        )
        selected = selected1 if learned_player == 0 else selected0
        prior_controller = (
            carry.player1_controller if learned_player == 0 else carry.player0_controller
        )
        bundle = compile_full_core_action_bundle_v1(
            carry.environment_state, selected0, selected1
        )
        player_action = bundle.player1 if learned_player == 0 else bundle.player0
        updated, rule_effect = update_full_core_controller_from_effects_v1(
            carry.environment_state,
            next_carry.environment_state,
            selected,
            player_action,
            rule_player,
        )
        del updated
        next_behavior = _add_behavior_step(
            behavior,
            carry,
            next_carry,
            selected,
            prior_controller,
            player_action,
            rule_effect,
            rule_player,
        )
        return next_carry, next_behavior

    return execute
