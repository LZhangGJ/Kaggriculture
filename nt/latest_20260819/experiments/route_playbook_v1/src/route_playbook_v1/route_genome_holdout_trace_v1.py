"""Seat-aware action tracing for official Route Genome holdout judging."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import EPISODE_STEPS, NUM_PRODUCTS
from kaggriculture_jax.types import Action, Events, State, StaticTables
from strategic_v5.e5_rollout import initialize_full_rule_carry_v1

from .route_genome_v1 import RouteGenomeV1, route_genome_schedule_v1
from .route_rollout import route_rule_step_with_action_v1
from .schema import empty_route_schedule_v1


class RouteGenomeSeatTraceResultV1(NamedTuple):
    final_state: State
    candidate_actions: Action
    terminal_cash: jax.Array
    terminal_product_inventory: jax.Array
    terminal_visible_yield: jax.Array
    hard_failures: jax.Array


def _choose_schedule(route, null, choose_route: jax.Array):
    return jax.tree.map(
        lambda route_value, null_value: jnp.where(
            choose_route.reshape(
                (choose_route.shape[0],)
                + (1,) * (route_value.ndim - 1)
            ),
            route_value,
            null_value,
        ),
        route,
        null,
    )


def _candidate_action(joint: Action, seats: jax.Array) -> Action:
    batch = jnp.arange(seats.shape[0], dtype=jnp.int32)
    return Action(
        unit_op=joint.unit_op[batch, seats],
        unit_item=joint.unit_item[batch, seats],
        unit_amount=joint.unit_amount[batch, seats],
        unit_count=joint.unit_count[batch, seats],
        market_op=joint.market_op[batch, seats],
        market_item=joint.market_item[batch, seats],
        market_amount=joint.market_amount[batch, seats],
        market_count=joint.market_count[batch, seats],
    )


def make_route_genome_seat_trace_rollout_v1(
    rollout_steps: int = EPISODE_STEPS - 1,
):
    """Return a fixed-shape rollout that records the candidate's two-seat actions."""

    def rollout(
        states: State,
        events: Events,
        tables: StaticTables,
        genome: RouteGenomeV1,
        seats: jax.Array,
    ) -> RouteGenomeSeatTraceResultV1:
        seats = seats.astype(jnp.int32)
        dummy_seeds = jnp.zeros_like(states.step, dtype=jnp.int32)
        carry = initialize_full_rule_carry_v1(dummy_seeds, events)._replace(
            environment_state=states
        )
        route = route_genome_schedule_v1(genome)
        null = empty_route_schedule_v1(genome.candidate_id.shape[0])
        schedule0 = _choose_schedule(route, null, seats == 0)
        schedule1 = _choose_schedule(route, null, seats == 1)

        def body(current, _):
            following, joint = route_rule_step_with_action_v1(
                current,
                tables,
                schedule0,
                schedule1,
                audit=True,
            )
            return following, _candidate_action(joint, seats)

        final, actions = jax.lax.scan(body, carry, xs=None, length=rollout_steps)
        state = final.environment_state
        batch = jnp.arange(seats.shape[0], dtype=jnp.int32)
        products = (
            jnp.sum(
                state.shed[batch, seats, :NUM_PRODUCTS].astype(jnp.int32),
                axis=1,
            )
            + jnp.sum(
                state.unit_inventory[batch, seats, :, :NUM_PRODUCTS].astype(jnp.int32),
                axis=(1, 2),
            )
        )
        diagnostics = final.diagnostics
        hard = (
            diagnostics.invalid_raw_action_count[batch, seats]
            + diagnostics.internal_resource_conflict_count[batch, seats]
            + diagnostics.unexpected_silent_noop_count[batch, seats]
            + diagnostics.effect_mismatch_count[batch, seats]
            + diagnostics.owner_inactive_count[batch, seats]
            + diagnostics.deadline_missed_count[batch, seats]
            + diagnostics.cross_episode_task_contamination_count[batch, seats]
            + diagnostics.nan_or_inf_count[batch, seats]
            + state.hand_cap_hits[batch, seats]
            + state.market_loop_cap_hits
            + state.price_lut_oob
        ).astype(jnp.int32)
        return RouteGenomeSeatTraceResultV1(
            final_state=state,
            candidate_actions=actions,
            terminal_cash=state.money[batch, seats].astype(jnp.int32),
            terminal_product_inventory=products,
            terminal_visible_yield=jnp.sum(
                state.tile_yield[batch, seats].astype(jnp.int32), axis=(1, 2)
            ),
            hard_failures=hard,
        )

    return rollout
