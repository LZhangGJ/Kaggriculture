"""Seat-balanced Dynamic Full-core V2 versus frozen Boatlee V16-RC2."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax import reset
from kaggriculture_jax.constants import TURNS_PER_DAY
from kaggriculture_jax.simulator import (
    batched_project_unit_phase,
    batched_step_from_projected_unit_phase_sync,
)
from kaggriculture_jax.types import Action, Events, State, StaticTables

from .boatlee_v16_gpu import (
    BoatleeCarryV1,
    BoatleeTraceV1,
    boatlee_actions_v1,
    initialize_boatlee_carry_v1,
)
from .dynamic_policy_v2 import (
    combine_dynamic_actions_v2,
    select_dynamic_unit_phase_v2,
)
from .dynamic_rollout_v2 import select_dynamic_full_core_pair_v2
from .e4_executor import update_full_core_controller_from_effects_v1
from .e5_rollout import cleanup_full_controller_day_end_v1
from .lifecycle import clear_invalidated_full_core_tasks_v1, reset_controller_state_v1
from .opponent_v2 import (
    OpponentHistoryV2,
    initialize_opponent_history_v2,
    update_opponent_history_v2,
)
from .schema import ControllerStateV1
from .task_cards import ReplayTaskCardProgramV1


class DynamicBoatleeArenaCarryV2(NamedTuple):
    environment_state: State
    player0_controller: ControllerStateV1
    player1_controller: ControllerStateV1
    opponent_history: OpponentHistoryV2
    boatlee_carry: BoatleeCarryV1
    current_events: Events
    learner_player: jax.Array
    rng: jax.Array


class DynamicBoatleeStepTraceV2(NamedTuple):
    invalid_market_orders: jax.Array
    overflow_market_orders: jax.Array


class DynamicBoatleeArenaResultV2(NamedTuple):
    final_carry: DynamicBoatleeArenaCarryV2
    learner_margin: jax.Array
    learner_outcome: jax.Array
    invalid_market_orders: jax.Array
    overflow_market_orders: jax.Array


def _controller_batch(batch_size: int) -> ControllerStateV1:
    one = reset_controller_state_v1()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape), one
    )


def initialize_dynamic_boatlee_arena_carry_v2(
    seeds: jax.Array,
    events: Events,
    learner_player: jax.Array,
    key: jax.Array,
) -> DynamicBoatleeArenaCarryV2:
    seeds = jnp.asarray(seeds, dtype=jnp.int32)
    learner_player = jnp.asarray(learner_player, dtype=jnp.int8)
    if seeds.shape != learner_player.shape:
        raise ValueError("seeds and learner_player must have the same shape")
    states = jax.vmap(reset)(seeds)
    batch_size = seeds.shape[0]
    return DynamicBoatleeArenaCarryV2(
        environment_state=states,
        player0_controller=_controller_batch(batch_size),
        player1_controller=_controller_batch(batch_size),
        opponent_history=initialize_opponent_history_v2(states),
        boatlee_carry=initialize_boatlee_carry_v1(batch_size),
        current_events=events,
        learner_player=learner_player,
        rng=key,
    )


def _select_controller_update(
    updated: ControllerStateV1,
    prior: ControllerStateV1,
    selected: jax.Array,
) -> ControllerStateV1:
    def choose(new, old):
        mask = selected.reshape((selected.shape[0],) + (1,) * (new.ndim - 1))
        return jnp.where(mask, new, old)

    return jax.tree.map(choose, updated, prior)


def _combine_actions(
    learned: Action, boatlee: Action, learner_player: jax.Array
) -> Action:
    selected = jnp.arange(2, dtype=jnp.int8)[None, :] == learner_player[:, None]

    def choose(learned_value, boatlee_value):
        mask = selected.reshape(
            selected.shape + (1,) * (learned_value.ndim - selected.ndim)
        )
        return jnp.where(mask, learned_value, boatlee_value)

    return jax.tree.map(choose, learned, boatlee)


def _select_seat(left: jax.Array, right: jax.Array, seat: jax.Array) -> jax.Array:
    values = jnp.stack((left, right), axis=1)
    return values[jnp.arange(seat.shape[0]), seat.astype(jnp.int32)]


def dynamic_boatlee_step_v2(
    carry: DynamicBoatleeArenaCarryV2,
    tables: StaticTables,
    boatlee_trace: BoatleeTraceV1,
    params: object,
    *,
    deterministic: bool = True,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = False,
) -> tuple[DynamicBoatleeArenaCarryV2, DynamicBoatleeStepTraceV2]:
    states = carry.environment_state
    next_key, key0, key1 = jax.random.split(carry.rng, 3)
    controller0 = clear_invalidated_full_core_tasks_v1(
        states, carry.player0_controller, 0
    )
    controller1 = clear_invalidated_full_core_tasks_v1(
        states, carry.player1_controller, 1
    )
    market_key0, unit_key0 = jax.random.split(key0)
    market_key1, unit_key1 = jax.random.split(key1)

    boatlee_action, next_boatlee_carry = boatlee_actions_v1(
        states, tables, boatlee_trace, carry.boatlee_carry
    )
    unit0 = select_dynamic_unit_phase_v2(
        states,
        controller0,
        carry.opponent_history,
        tables,
        params,
        0,
        unit_key0,
        deterministic=deterministic,
        include_opponent=include_opponent,
        include_history=include_history,
        task_card_program=task_card_program,
        allow_nonpositive_econ=allow_nonpositive_econ,
    )
    unit1 = select_dynamic_unit_phase_v2(
        states,
        controller1,
        carry.opponent_history,
        tables,
        params,
        1,
        unit_key1,
        deterministic=deterministic,
        include_opponent=include_opponent,
        include_history=include_history,
        task_card_program=task_card_program,
        allow_nonpositive_econ=allow_nonpositive_econ,
    )
    dynamic_unit_action = combine_dynamic_actions_v2(unit0.action, unit1.action)
    actual_unit_action = _combine_actions(
        dynamic_unit_action, boatlee_action, carry.learner_player
    )
    projected = batched_project_unit_phase(states, actual_unit_action)
    decision0, decision1 = select_dynamic_full_core_pair_v2(
        states,
        controller0,
        controller1,
        carry.opponent_history,
        tables,
        params,
        market_key0,
        market_key1,
        unit0,
        unit1,
        projected,
        deterministic=deterministic,
        include_opponent=include_opponent,
        include_history=include_history,
        task_card_program=task_card_program,
        allow_nonpositive_econ=allow_nonpositive_econ,
    )
    dynamic_action = combine_dynamic_actions_v2(decision0.action, decision1.action)
    action = _combine_actions(dynamic_action, boatlee_action, carry.learner_player)
    next_states = batched_step_from_projected_unit_phase_sync(
        projected, action, carry.current_events, tables
    )

    updated0, _ = update_full_core_controller_from_effects_v1(
        states, next_states, decision0.controller, decision0.effect_action, 0
    )
    updated1, _ = update_full_core_controller_from_effects_v1(
        states, next_states, decision1.controller, decision1.effect_action, 1
    )
    controller0 = _select_controller_update(
        updated0, controller0, carry.learner_player == 0
    )
    controller1 = _select_controller_update(
        updated1, controller1, carry.learner_player == 1
    )
    day_end = ((next_states.step % TURNS_PER_DAY) == 0) & (~next_states.done)
    controller0 = cleanup_full_controller_day_end_v1(
        controller0, next_states.unit_active[:, 0], day_end
    )
    controller1 = cleanup_full_controller_day_end_v1(
        controller1, next_states.unit_active[:, 1], day_end
    )
    invalid = _select_seat(
        decision0.ledger.invalid_order_count,
        decision1.ledger.invalid_order_count,
        carry.learner_player,
    )
    overflow = _select_seat(
        decision0.ledger.overflow_order_count,
        decision1.ledger.overflow_order_count,
        carry.learner_player,
    )
    return (
        DynamicBoatleeArenaCarryV2(
            environment_state=next_states,
            player0_controller=controller0,
            player1_controller=controller1,
            opponent_history=(
                update_opponent_history_v2(carry.opponent_history, next_states)
                if include_history
                else carry.opponent_history
            ),
            boatlee_carry=next_boatlee_carry,
            current_events=carry.current_events,
            learner_player=carry.learner_player,
            rng=next_key,
        ),
        DynamicBoatleeStepTraceV2(invalid, overflow),
    )


def make_dynamic_boatlee_arena_rollout_v2(
    *,
    rollout_steps: int,
    deterministic: bool = True,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = False,
):
    def rollout(
        carry: DynamicBoatleeArenaCarryV2,
        tables: StaticTables,
        boatlee_trace: BoatleeTraceV1,
        params: object,
    ) -> DynamicBoatleeArenaResultV2:
        def body(current, _):
            return dynamic_boatlee_step_v2(
                current,
                tables,
                boatlee_trace,
                params,
                deterministic=deterministic,
                include_opponent=include_opponent,
                include_history=include_history,
                task_card_program=task_card_program,
                allow_nonpositive_econ=allow_nonpositive_econ,
            )

        final, traces = jax.lax.scan(body, carry, xs=None, length=rollout_steps)
        learner_money = _select_seat(
            final.environment_state.money[:, 0],
            final.environment_state.money[:, 1],
            final.learner_player,
        )
        opponent_money = _select_seat(
            final.environment_state.money[:, 1],
            final.environment_state.money[:, 0],
            final.learner_player,
        )
        margin = learner_money - opponent_money
        return DynamicBoatleeArenaResultV2(
            final_carry=final,
            learner_margin=margin,
            learner_outcome=jnp.sign(margin).astype(jnp.int8),
            invalid_market_orders=jnp.sum(traces.invalid_market_orders, axis=0),
            overflow_market_orders=jnp.sum(traces.overflow_market_orders, axis=0),
        )

    return rollout
