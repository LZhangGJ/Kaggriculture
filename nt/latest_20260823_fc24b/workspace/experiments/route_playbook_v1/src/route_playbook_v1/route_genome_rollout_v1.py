"""Batched exact rollout and behavior metrics for fixed-shape RouteGenomeV1."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_FIRST_YIELD_DAY,
    ANIMAL_PRODUCT,
    EPISODE_STEPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    TURNS_PER_DAY,
    MarketOp,
)
from kaggriculture_jax.types import Events, State, StaticTables

from .market_ledger import batched_step_with_sales_ledger_sync_v1
from .route_genome_v1 import (
    RouteGenomeCarryV1,
    RouteGenomeV1,
    initialize_route_genome_carry_v1,
    route_genome_player_action_v1,
)
from .trace_core import CalibrationTraceV1, pair_trace_with_null_v1


class RouteGenomeMetricsV1(NamedTuple):
    productive_animal_days: jax.Array
    sold_units: jax.Array
    gross_revenue: jax.Array
    min_cash: jax.Array
    max_hands: jax.Array
    max_land: jax.Array
    first_target_sale_step: jax.Array
    action_hash_a: jax.Array
    action_hash_b: jax.Array
    target_sale_attempt_units: jax.Array
    target_animal_buy_attempt_units: jax.Array
    target_animal_buy_success_units: jax.Array
    land_buy_attempts: jax.Array
    land_buy_successes: jax.Array
    hire_attempts: jax.Array
    hire_successes: jax.Array
    deferred_cash: jax.Array
    deferred_shop: jax.Array
    worker_task_attempts: jax.Array
    worker_task_confirmed: jax.Array
    worker_task_retries: jax.Array
    worker_task_hard_failures: jax.Array


class RouteGenomeRolloutCarryV1(NamedTuple):
    environment_state: State
    genome_carry: RouteGenomeCarryV1
    events: Events
    metrics: RouteGenomeMetricsV1


class RouteGenomeRolloutResultV1(NamedTuple):
    final_state: State
    genome_carry: RouteGenomeCarryV1
    metrics: RouteGenomeMetricsV1
    terminal_product_inventory: jax.Array
    terminal_animal_inventory: jax.Array
    hard_failures: jax.Array


_ANIMAL_FIRST = jnp.asarray(ANIMAL_FIRST_YIELD_DAY, dtype=jnp.int16)
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int8)


def initialize_route_genome_metrics_v1(states: State) -> RouteGenomeMetricsV1:
    batch_size = states.step.shape[0]
    return RouteGenomeMetricsV1(
        productive_animal_days=jnp.zeros((batch_size,), dtype=jnp.int32),
        sold_units=jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int32),
        gross_revenue=jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int32),
        min_cash=states.money[:, 0].astype(jnp.int32),
        max_hands=jnp.zeros((batch_size,), dtype=jnp.int16),
        max_land=states.unlocked_count[:, 0].astype(jnp.int8),
        first_target_sale_step=jnp.full((batch_size,), -1, dtype=jnp.int16),
        action_hash_a=jnp.full((batch_size,), 2166136261, dtype=jnp.uint32),
        action_hash_b=jnp.full((batch_size,), 2246822519, dtype=jnp.uint32),
        target_sale_attempt_units=jnp.zeros((batch_size,), dtype=jnp.int32),
        target_animal_buy_attempt_units=jnp.zeros((batch_size,), dtype=jnp.int32),
        target_animal_buy_success_units=jnp.zeros((batch_size,), dtype=jnp.int32),
        land_buy_attempts=jnp.zeros((batch_size,), dtype=jnp.int32),
        land_buy_successes=jnp.zeros((batch_size,), dtype=jnp.int32),
        hire_attempts=jnp.zeros((batch_size,), dtype=jnp.int32),
        hire_successes=jnp.zeros((batch_size,), dtype=jnp.int32),
        deferred_cash=jnp.zeros((batch_size,), dtype=jnp.int32),
        deferred_shop=jnp.zeros((batch_size,), dtype=jnp.int32),
        worker_task_attempts=jnp.zeros((batch_size,), dtype=jnp.int32),
        worker_task_confirmed=jnp.zeros((batch_size,), dtype=jnp.int32),
        worker_task_retries=jnp.zeros((batch_size,), dtype=jnp.int32),
        worker_task_hard_failures=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


def initialize_route_genome_rollout_carry_v1(
    states: State, events: Events
) -> RouteGenomeRolloutCarryV1:
    return RouteGenomeRolloutCarryV1(
        environment_state=states,
        genome_carry=initialize_route_genome_carry_v1(states.step.shape[0]),
        events=events,
        metrics=initialize_route_genome_metrics_v1(states),
    )


def _target_assets(states: State, genome: RouteGenomeV1, player: int = 0):
    target = genome.target_animal_id.astype(jnp.int8)
    placed = jnp.sum(
        states.tile_animal[:, player] == target[:, None, None], axis=(1, 2)
    ).astype(jnp.int32)
    item = NUM_PRODUCTS + target.astype(jnp.int32)
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)
    shed = states.shed[batch, player, item].astype(jnp.int32)
    carried = jnp.sum(
        states.unit_inventory[:, player]
        * jax.nn.one_hot(item, NUM_SHED_ITEMS, dtype=jnp.int16)[:, None, :],
        axis=(1, 2),
        dtype=jnp.int32,
    )
    return placed + shed + carried


def _productive_animals(states: State, genome: RouteGenomeV1) -> jax.Array:
    target = genome.target_animal_id.astype(jnp.int8)
    present = states.tile_animal[:, 0] == target[:, None, None]
    current_day = states.step.astype(jnp.int16) // TURNS_PER_DAY
    first = _ANIMAL_FIRST[target.astype(jnp.int32)]
    mature = (
        current_day[:, None, None]
        - states.tile_origin_day[:, 0].astype(jnp.int16)
        >= first[:, None, None]
    )
    day_tick = (states.step % TURNS_PER_DAY) == 0
    return jnp.where(
        day_tick,
        jnp.sum(present & mature, axis=(1, 2), dtype=jnp.int32),
        0,
    )


def _action_hash(
    hash_a: jax.Array, hash_b: jax.Array, action
) -> tuple[jax.Array, jax.Array]:
    unit_index = jnp.arange(MAX_UNITS, dtype=jnp.uint32)[None, :] + 1
    market_index = jnp.arange(MAX_MARKET_ORDERS, dtype=jnp.uint32)[None, :] + 1
    unit_signal = jnp.sum(
        (action.unit_op.astype(jnp.uint32) + 1) * unit_index * jnp.uint32(131)
        + (action.unit_item.astype(jnp.int32) + 2).astype(jnp.uint32)
        * unit_index
        * jnp.uint32(17)
        + action.unit_amount.astype(jnp.uint32) * unit_index * jnp.uint32(7),
        axis=1,
        dtype=jnp.uint32,
    )
    market_signal = jnp.sum(
        (action.market_op.astype(jnp.uint32) + 1)
        * market_index
        * jnp.uint32(257)
        + (action.market_item.astype(jnp.int32) + 2).astype(jnp.uint32)
        * market_index
        * jnp.uint32(29)
        + action.market_amount.astype(jnp.uint32)
        * market_index
        * jnp.uint32(11),
        axis=1,
        dtype=jnp.uint32,
    )
    signal = unit_signal ^ (market_signal * jnp.uint32(374761393))
    return (
        (hash_a ^ signal) * jnp.uint32(16777619),
        (hash_b + signal + jnp.uint32(0x9E3779B9)) * jnp.uint32(2246822519),
    )


def route_genome_step_v1(
    carry: RouteGenomeRolloutCarryV1,
    tables: StaticTables,
    bank: CalibrationTraceV1,
    genome: RouteGenomeV1,
) -> tuple[RouteGenomeRolloutCarryV1, None]:
    states = carry.environment_state
    player_action, genome_carry, diagnostics = route_genome_player_action_v1(
        states, tables, bank, genome, carry.genome_carry, 0
    )
    joint = pair_trace_with_null_v1(player_action, 0)
    previous_assets = _target_assets(states, genome)
    result = batched_step_with_sales_ledger_sync_v1(
        states, joint, carry.events, tables
    )
    next_states = result.final_state
    following_assets = _target_assets(next_states, genome)
    target_product = _ANIMAL_PRODUCT[genome.target_animal_id.astype(jnp.int32)]
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)
    target_sold = result.sales.units[batch, 0, target_product].astype(jnp.int32)
    active = (
        jnp.arange(MAX_MARKET_ORDERS)[None, :]
        < player_action.market_count[:, None]
    )
    target_sale_attempt = jnp.sum(
        jnp.where(
            active
            & (player_action.market_op == MarketOp.SELL)
            & (player_action.market_item == target_product[:, None]),
            jnp.maximum(player_action.market_amount, 0),
            0,
        ),
        axis=1,
    )
    animal_buy_attempt = jnp.sum(
        jnp.where(
            active & (player_action.market_op == MarketOp.BUY_ANIMAL),
            jnp.maximum(player_action.market_amount, 0),
            0,
        ),
        axis=1,
    )
    land_attempt = jnp.sum(
        active & (player_action.market_op == MarketOp.BUY_LAND),
        axis=1,
        dtype=jnp.int32,
    )
    hire_attempt = jnp.sum(
        active & (player_action.market_op == MarketOp.HIRE),
        axis=1,
        dtype=jnp.int32,
    )
    hands = jnp.maximum(
        jnp.sum(next_states.unit_active[:, 0], axis=1).astype(jnp.int16) - 1, 0
    )
    first_sale = jnp.where(
        (carry.metrics.first_target_sale_step < 0) & (target_sold > 0),
        states.step.astype(jnp.int16),
        carry.metrics.first_target_sale_step,
    )
    hash_a, hash_b = _action_hash(
        carry.metrics.action_hash_a,
        carry.metrics.action_hash_b,
        player_action,
    )
    metrics = RouteGenomeMetricsV1(
        productive_animal_days=carry.metrics.productive_animal_days
        + _productive_animals(states, genome),
        sold_units=carry.metrics.sold_units + result.sales.units[:, 0],
        gross_revenue=carry.metrics.gross_revenue
        + result.sales.gross_revenue[:, 0],
        min_cash=jnp.minimum(carry.metrics.min_cash, next_states.money[:, 0]),
        max_hands=jnp.maximum(carry.metrics.max_hands, hands),
        max_land=jnp.maximum(
            carry.metrics.max_land, next_states.unlocked_count[:, 0]
        ),
        first_target_sale_step=first_sale,
        action_hash_a=hash_a,
        action_hash_b=hash_b,
        target_sale_attempt_units=carry.metrics.target_sale_attempt_units
        + target_sale_attempt,
        target_animal_buy_attempt_units=carry.metrics.target_animal_buy_attempt_units
        + animal_buy_attempt,
        target_animal_buy_success_units=carry.metrics.target_animal_buy_success_units
        + jnp.maximum(following_assets - previous_assets, 0),
        land_buy_attempts=carry.metrics.land_buy_attempts + land_attempt,
        land_buy_successes=carry.metrics.land_buy_successes
        + jnp.maximum(
            next_states.unlocked_count[:, 0].astype(jnp.int32)
            - states.unlocked_count[:, 0].astype(jnp.int32),
            0,
        ),
        hire_attempts=carry.metrics.hire_attempts + hire_attempt,
        hire_successes=carry.metrics.hire_successes
        + jnp.maximum(
            jnp.sum(next_states.unit_active[:, 0], axis=1).astype(jnp.int32)
            - jnp.sum(states.unit_active[:, 0], axis=1).astype(jnp.int32),
            0,
        ),
        deferred_cash=carry.metrics.deferred_cash + diagnostics.deferred_cash,
        deferred_shop=carry.metrics.deferred_shop + diagnostics.deferred_shop,
        worker_task_attempts=carry.metrics.worker_task_attempts
        + diagnostics.attempted_tasks,
        worker_task_confirmed=carry.metrics.worker_task_confirmed
        + diagnostics.confirmed_tasks,
        worker_task_retries=carry.metrics.worker_task_retries
        + diagnostics.retried_tasks,
        worker_task_hard_failures=carry.metrics.worker_task_hard_failures
        + diagnostics.hard_failures,
    )
    return RouteGenomeRolloutCarryV1(
        environment_state=next_states,
        genome_carry=genome_carry,
        events=carry.events,
        metrics=metrics,
    ), None


def make_route_genome_rollout_v1(rollout_steps: int = EPISODE_STEPS - 1):
    """Return one shape-stable function for all same-batch genome values."""

    def rollout(
        initial: RouteGenomeRolloutCarryV1,
        tables: StaticTables,
        bank: CalibrationTraceV1,
        genome: RouteGenomeV1,
    ) -> RouteGenomeRolloutResultV1:
        def body(value, _):
            return route_genome_step_v1(value, tables, bank, genome)

        final, _ = jax.lax.scan(body, initial, xs=None, length=rollout_steps)
        state = final.environment_state
        terminal_product_inventory = (
            jnp.sum(state.shed[:, 0, :NUM_PRODUCTS].astype(jnp.int32), axis=1)
            + jnp.sum(
                state.unit_inventory[:, 0, :, :NUM_PRODUCTS].astype(jnp.int32),
                axis=(1, 2),
            )
        )
        terminal_animal_inventory = jnp.sum(
            state.shed[:, 0, NUM_PRODUCTS:].astype(jnp.int32), axis=1
        ) + jnp.sum(
            state.unit_inventory[:, 0, :, NUM_PRODUCTS:].astype(jnp.int32),
            axis=(1, 2),
        )
        simulator_hard = (
            jnp.sum(state.hand_cap_hits, axis=1, dtype=jnp.int32)
            + state.market_loop_cap_hits
            + state.price_lut_oob
        ).astype(jnp.int32)
        return RouteGenomeRolloutResultV1(
            final_state=state,
            genome_carry=final.genome_carry,
            metrics=final.metrics,
            terminal_product_inventory=terminal_product_inventory,
            terminal_animal_inventory=terminal_animal_inventory,
            hard_failures=final.genome_carry.hard_failures + simulator_hard,
        )

    return rollout
