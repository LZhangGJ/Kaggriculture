"""State-driven Full-core rollout for fixed-shape R6/R7 Route Genomes.

The first Route Genome smoke test proved the JIT value contract but also
proved that retiming investments while replaying a frozen 719-step worker
script is structurally invalid.  This executor keeps the genome's macro
targets and delegates worker planning/recovery to the already-tested Full-core
controller on every visible state.
"""

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
    UnitOp,
)
from kaggriculture_jax.simulator import batched_project_unit_phase
from kaggriculture_jax.types import Action, Events, State, StaticTables
from strategic_v5.e5_rollout import (
    FullRuleCarryV1,
    initialize_full_rule_carry_v1,
)

from .market_ledger import batch_market_sales_ledger_sync_v1
from .route_genome_v1 import RouteGenomeV1, route_genome_schedule_v1
from .route_rollout import route_rule_step_with_action_v1
from .schema import empty_route_schedule_v1


class RouteGenomeFullCoreMetricsV1(NamedTuple):
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


class RouteGenomeFullCoreCarryV1(NamedTuple):
    rule: FullRuleCarryV1
    metrics: RouteGenomeFullCoreMetricsV1


class RouteGenomeFullCoreResultV1(NamedTuple):
    final_state: State
    rule_carry: FullRuleCarryV1
    metrics: RouteGenomeFullCoreMetricsV1
    terminal_product_inventory: jax.Array
    terminal_animal_inventory: jax.Array
    hard_failures: jax.Array


_ANIMAL_FIRST = jnp.asarray(ANIMAL_FIRST_YIELD_DAY, dtype=jnp.int16)
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int8)


def _empty_metrics(states: State) -> RouteGenomeFullCoreMetricsV1:
    batch = states.step.shape[0]
    zeros = jnp.zeros((batch,), dtype=jnp.int32)
    return RouteGenomeFullCoreMetricsV1(
        productive_animal_days=zeros,
        sold_units=jnp.zeros((batch, NUM_PRODUCTS), dtype=jnp.int32),
        gross_revenue=jnp.zeros((batch, NUM_PRODUCTS), dtype=jnp.int32),
        min_cash=states.money[:, 0].astype(jnp.int32),
        max_hands=jnp.zeros((batch,), dtype=jnp.int16),
        max_land=states.unlocked_count[:, 0].astype(jnp.int8),
        first_target_sale_step=jnp.full((batch,), -1, dtype=jnp.int16),
        action_hash_a=jnp.full((batch,), 2166136261, dtype=jnp.uint32),
        action_hash_b=jnp.full((batch,), 2246822519, dtype=jnp.uint32),
        target_sale_attempt_units=zeros,
        target_animal_buy_attempt_units=zeros,
        target_animal_buy_success_units=zeros,
        land_buy_attempts=zeros,
        land_buy_successes=zeros,
        hire_attempts=zeros,
        hire_successes=zeros,
        deferred_cash=zeros,
        deferred_shop=zeros,
        worker_task_attempts=zeros,
        worker_task_confirmed=zeros,
        worker_task_retries=zeros,
        worker_task_hard_failures=zeros,
    )


def initialize_route_genome_full_core_carry_v1(
    states: State, events: Events
) -> RouteGenomeFullCoreCarryV1:
    dummy_seeds = jnp.zeros_like(states.step, dtype=jnp.int32)
    rule = initialize_full_rule_carry_v1(dummy_seeds, events)._replace(
        environment_state=states
    )
    return RouteGenomeFullCoreCarryV1(rule=rule, metrics=_empty_metrics(states))


def _target_assets(states: State, genome: RouteGenomeV1) -> jax.Array:
    target = genome.target_animal_id.astype(jnp.int8)
    placed = jnp.sum(
        states.tile_animal[:, 0] == target[:, None, None], axis=(1, 2)
    ).astype(jnp.int32)
    item = NUM_PRODUCTS + target.astype(jnp.int32)
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)
    shed = states.shed[batch, 0, item].astype(jnp.int32)
    carried = jnp.sum(
        states.unit_inventory[:, 0]
        * jax.nn.one_hot(item, NUM_SHED_ITEMS, dtype=jnp.int16)[:, None, :],
        axis=(1, 2),
        dtype=jnp.int32,
    )
    return placed + shed + carried


def _productive_animals(states: State, genome: RouteGenomeV1) -> jax.Array:
    target = genome.target_animal_id.astype(jnp.int8)
    present = states.tile_animal[:, 0] == target[:, None, None]
    day = states.step.astype(jnp.int16) // TURNS_PER_DAY
    first = _ANIMAL_FIRST[target.astype(jnp.int32)]
    mature = (
        day[:, None, None]
        - states.tile_origin_day[:, 0].astype(jnp.int16)
        >= first[:, None, None]
    )
    return jnp.where(
        (states.step % TURNS_PER_DAY) == 0,
        jnp.sum(present & mature, axis=(1, 2), dtype=jnp.int32),
        0,
    )


def _player_action(action: Action, player: int = 0) -> Action:
    return Action(
        unit_op=action.unit_op[:, player],
        unit_item=action.unit_item[:, player],
        unit_amount=action.unit_amount[:, player],
        unit_count=action.unit_count[:, player],
        market_op=action.market_op[:, player],
        market_item=action.market_item[:, player],
        market_amount=action.market_amount[:, player],
        market_count=action.market_count[:, player],
    )


def _action_hash(
    hash_a: jax.Array, hash_b: jax.Array, action: Action
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
        (hash_b + signal + jnp.uint32(0x9E3779B9))
        * jnp.uint32(2246822519),
    )


def _severe_diagnostic_increment(before, after) -> jax.Array:
    delta = jax.tree.map(lambda new, old: new - old, after, before)
    return (
        delta.invalid_raw_action_count[:, 0]
        + delta.internal_resource_conflict_count[:, 0]
        + delta.unexpected_silent_noop_count[:, 0]
        + delta.effect_mismatch_count[:, 0]
        + delta.owner_inactive_count[:, 0]
        + delta.deadline_missed_count[:, 0]
        + delta.cross_episode_task_contamination_count[:, 0]
        + delta.nan_or_inf_count[:, 0]
    ).astype(jnp.int32)


def make_route_genome_full_core_rollout_v1(
    rollout_steps: int = EPISODE_STEPS - 1,
):
    """Return one shape-stable Full-core rollout for all genome values."""

    def rollout(
        initial: RouteGenomeFullCoreCarryV1,
        tables: StaticTables,
        unused_trace_bank,
        genome: RouteGenomeV1,
    ) -> RouteGenomeFullCoreResultV1:
        del unused_trace_bank
        schedule = route_genome_schedule_v1(genome)
        null = empty_route_schedule_v1(genome.candidate_id.shape[0])

        def body(carry: RouteGenomeFullCoreCarryV1, _):
            states = carry.rule.environment_state
            previous_assets = _target_assets(states, genome)
            following_rule, joint_action = route_rule_step_with_action_v1(
                carry.rule,
                tables,
                schedule,
                null,
                audit=True,
            )
            next_states = following_rule.environment_state
            action = _player_action(joint_action, 0)
            # Exact sales are read from the same official synchronized market
            # semantics.  This diagnostic replay does not mutate the rollout.
            projected = batched_project_unit_phase(states, joint_action)
            sales = batch_market_sales_ledger_sync_v1(
                projected, joint_action, tables
            ).sales
            target_product = _ANIMAL_PRODUCT[
                genome.target_animal_id.astype(jnp.int32)
            ]
            batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)
            target_sold = sales.units[batch, 0, target_product]
            active_market = (
                jnp.arange(MAX_MARKET_ORDERS)[None, :]
                < action.market_count[:, None]
            )
            sale_attempt = jnp.sum(
                jnp.where(
                    active_market
                    & (action.market_op == MarketOp.SELL)
                    & (action.market_item == target_product[:, None]),
                    jnp.maximum(action.market_amount, 0),
                    0,
                ),
                axis=1,
                dtype=jnp.int32,
            )
            animal_attempt = jnp.sum(
                jnp.where(
                    active_market & (action.market_op == MarketOp.BUY_ANIMAL),
                    jnp.maximum(action.market_amount, 0),
                    0,
                ),
                axis=1,
                dtype=jnp.int32,
            )
            land_attempt = jnp.sum(
                active_market & (action.market_op == MarketOp.BUY_LAND),
                axis=1,
                dtype=jnp.int32,
            )
            hire_attempt = jnp.sum(
                active_market & (action.market_op == MarketOp.HIRE),
                axis=1,
                dtype=jnp.int32,
            )
            active_unit = (
                jnp.arange(MAX_UNITS)[None, :] < action.unit_count[:, None]
            )
            task_attempt = jnp.sum(
                active_unit & (action.unit_op != UnitOp.PASS),
                axis=1,
                dtype=jnp.int32,
            )
            diag_delta = jax.tree.map(
                lambda new, old: new - old,
                following_rule.diagnostics,
                carry.rule.diagnostics,
            )
            confirmed = jnp.sum(
                diag_delta.success_by_task_type[:, 0],
                axis=1,
                dtype=jnp.int32,
            )
            retries = diag_delta.resource_unavailable_count[:, 0].astype(
                jnp.int32
            )
            hard = _severe_diagnostic_increment(
                carry.rule.diagnostics, following_rule.diagnostics
            )
            day = states.step.astype(jnp.int32) // TURNS_PER_DAY
            target_animals = jnp.take_along_axis(
                schedule.animal_target_by_day,
                jnp.clip(day, 0, 29)[:, None, None],
                axis=1,
            )[:, 0]
            target_animals = jnp.take_along_axis(
                target_animals,
                genome.target_animal_id.astype(jnp.int32)[:, None],
                axis=1,
            )[:, 0]
            deferred_cash = (
                (_target_assets(states, genome) < target_animals)
                & (states.money[:, 0] <= genome.cash_reserve)
            ).astype(jnp.int32)
            first_sale = jnp.where(
                (carry.metrics.first_target_sale_step < 0)
                & (target_sold > 0),
                states.step.astype(jnp.int16),
                carry.metrics.first_target_sale_step,
            )
            hash_a, hash_b = _action_hash(
                carry.metrics.action_hash_a,
                carry.metrics.action_hash_b,
                action,
            )
            hands = jnp.maximum(
                jnp.sum(next_states.unit_active[:, 0], axis=1).astype(jnp.int16)
                - 1,
                0,
            )
            following_assets = _target_assets(next_states, genome)
            metrics = RouteGenomeFullCoreMetricsV1(
                productive_animal_days=carry.metrics.productive_animal_days
                + _productive_animals(states, genome),
                sold_units=carry.metrics.sold_units + sales.units[:, 0],
                gross_revenue=carry.metrics.gross_revenue
                + sales.gross_revenue[:, 0],
                min_cash=jnp.minimum(
                    carry.metrics.min_cash, next_states.money[:, 0]
                ),
                max_hands=jnp.maximum(carry.metrics.max_hands, hands),
                max_land=jnp.maximum(
                    carry.metrics.max_land, next_states.unlocked_count[:, 0]
                ),
                first_target_sale_step=first_sale,
                action_hash_a=hash_a,
                action_hash_b=hash_b,
                target_sale_attempt_units=carry.metrics.target_sale_attempt_units
                + sale_attempt,
                target_animal_buy_attempt_units=carry.metrics.target_animal_buy_attempt_units
                + animal_attempt,
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
                deferred_cash=carry.metrics.deferred_cash + deferred_cash,
                deferred_shop=carry.metrics.deferred_shop,
                worker_task_attempts=carry.metrics.worker_task_attempts
                + task_attempt,
                worker_task_confirmed=carry.metrics.worker_task_confirmed
                + confirmed,
                worker_task_retries=carry.metrics.worker_task_retries + retries,
                worker_task_hard_failures=carry.metrics.worker_task_hard_failures
                + hard,
            )
            return RouteGenomeFullCoreCarryV1(
                rule=following_rule, metrics=metrics
            ), None

        final, _ = jax.lax.scan(body, initial, xs=None, length=rollout_steps)
        state = final.rule.environment_state
        terminal_products = (
            jnp.sum(state.shed[:, 0, :NUM_PRODUCTS].astype(jnp.int32), axis=1)
            + jnp.sum(
                state.unit_inventory[:, 0, :, :NUM_PRODUCTS].astype(jnp.int32),
                axis=(1, 2),
            )
        )
        terminal_animals = jnp.sum(
            state.shed[:, 0, NUM_PRODUCTS:].astype(jnp.int32), axis=1
        ) + jnp.sum(
            state.unit_inventory[:, 0, :, NUM_PRODUCTS:].astype(jnp.int32),
            axis=(1, 2),
        )
        simulator_hard = (
            jnp.sum(state.hand_cap_hits, axis=1, dtype=jnp.int32)
            + state.market_loop_cap_hits
            + state.price_lut_oob
        )
        return RouteGenomeFullCoreResultV1(
            final_state=state,
            rule_carry=final.rule,
            metrics=final.metrics,
            terminal_product_inventory=terminal_products,
            terminal_animal_inventory=terminal_animals,
            hard_failures=final.metrics.worker_task_hard_failures
            + simulator_hard,
        )

    return rollout
