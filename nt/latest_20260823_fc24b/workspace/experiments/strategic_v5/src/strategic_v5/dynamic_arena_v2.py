"""Seat-balanced two-checkpoint arena for Dynamic Full-core V2 policies."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import EPISODE_STEPS, TURNS_PER_DAY
from kaggriculture_jax.simulator import (
    batched_project_unit_phase,
    batched_step_from_projected_unit_phase_sync,
)
from kaggriculture_jax.types import StaticTables

from .dynamic_policy_v2 import (
    combine_dynamic_actions_v2,
    select_dynamic_full_core_v2,
    select_dynamic_unit_phase_v2,
)
from .e4_executor import update_full_core_controller_from_effects_v1
from .e5_rollout import cleanup_full_controller_day_end_v1
from .lifecycle import clear_invalidated_full_core_tasks_v1
from .opponent_v2 import update_opponent_history_v2
from .rollout_v2 import FullLearnedCarryV2
from .task_cards import ReplayTaskCardProgramV1


class DynamicPairedArenaResultV2(NamedTuple):
    final_carry: FullLearnedCarryV2
    player0_outcome: jax.Array
    invalid_market_orders: jax.Array
    overflow_market_orders: jax.Array


def dynamic_two_policy_step_v2(
    carry: FullLearnedCarryV2,
    tables: StaticTables,
    player0_params: object,
    player1_params: object,
    *,
    deterministic: bool = True,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = False,
    player0_allow_nonpositive_econ: bool | None = None,
    player1_allow_nonpositive_econ: bool | None = None,
):
    gate0 = (
        allow_nonpositive_econ
        if player0_allow_nonpositive_econ is None
        else player0_allow_nonpositive_econ
    )
    gate1 = (
        allow_nonpositive_econ
        if player1_allow_nonpositive_econ is None
        else player1_allow_nonpositive_econ
    )
    states = carry.environment_state
    next_key, key0, key1 = jax.random.split(carry.rng, 3)
    market_key0, unit_key0 = jax.random.split(key0)
    market_key1, unit_key1 = jax.random.split(key1)
    controller0 = clear_invalidated_full_core_tasks_v1(
        states, carry.player0_controller, 0
    )
    controller1 = clear_invalidated_full_core_tasks_v1(
        states, carry.player1_controller, 1
    )
    unit0 = select_dynamic_unit_phase_v2(
        states,
        controller0,
        carry.opponent_history,
        tables,
        player0_params,
        0,
        unit_key0,
        deterministic=deterministic,
        include_opponent=include_opponent,
        include_history=include_history,
        task_card_program=task_card_program,
        allow_nonpositive_econ=gate0,
    )
    unit1 = select_dynamic_unit_phase_v2(
        states,
        controller1,
        carry.opponent_history,
        tables,
        player1_params,
        1,
        unit_key1,
        deterministic=deterministic,
        include_opponent=include_opponent,
        include_history=include_history,
        task_card_program=task_card_program,
        allow_nonpositive_econ=gate1,
    )
    unit_action = combine_dynamic_actions_v2(unit0.action, unit1.action)
    projected_unit_state = batched_project_unit_phase(states, unit_action)
    decision0 = select_dynamic_full_core_v2(
        states,
        controller0,
        carry.opponent_history,
        tables,
        player0_params,
        0,
        market_key0,
        deterministic=deterministic,
        include_opponent=include_opponent,
        include_history=include_history,
        task_card_program=task_card_program,
        allow_nonpositive_econ=gate0,
        unit_decision=unit0,
        projected_unit_state=projected_unit_state,
    )
    decision1 = select_dynamic_full_core_v2(
        states,
        controller1,
        carry.opponent_history,
        tables,
        player1_params,
        1,
        market_key1,
        deterministic=deterministic,
        include_opponent=include_opponent,
        include_history=include_history,
        task_card_program=task_card_program,
        allow_nonpositive_econ=gate1,
        unit_decision=unit1,
        projected_unit_state=projected_unit_state,
    )
    action = combine_dynamic_actions_v2(decision0.action, decision1.action)
    next_states = batched_step_from_projected_unit_phase_sync(
        projected_unit_state, action, carry.current_events, tables
    )
    controller0, _ = update_full_core_controller_from_effects_v1(
        states, next_states, decision0.controller, decision0.effect_action, 0
    )
    controller1, _ = update_full_core_controller_from_effects_v1(
        states, next_states, decision1.controller, decision1.effect_action, 1
    )
    day_end = ((next_states.step % TURNS_PER_DAY) == 0) & (~next_states.done)
    controller0 = cleanup_full_controller_day_end_v1(
        controller0, next_states.unit_active[:, 0], day_end
    )
    controller1 = cleanup_full_controller_day_end_v1(
        controller1, next_states.unit_active[:, 1], day_end
    )
    next_carry = FullLearnedCarryV2(
        environment_state=next_states,
        player0_controller=controller0,
        player1_controller=controller1,
        current_events=carry.current_events,
        opponent_history=(
            update_opponent_history_v2(carry.opponent_history, next_states)
            if include_history
            else carry.opponent_history
        ),
        rng=next_key,
    )
    invalid = jnp.stack(
        (decision0.ledger.invalid_order_count, decision1.ledger.invalid_order_count),
        axis=1,
    )
    overflow = jnp.stack(
        (decision0.ledger.overflow_order_count, decision1.ledger.overflow_order_count),
        axis=1,
    )
    return next_carry, (invalid, overflow)


def make_dynamic_paired_arena_rollout_v2(
    *,
    rollout_steps: int = EPISODE_STEPS - 1,
    deterministic: bool = True,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = False,
    player0_allow_nonpositive_econ: bool | None = None,
    player1_allow_nonpositive_econ: bool | None = None,
):
    def rollout(
        initial_carry: FullLearnedCarryV2,
        tables: StaticTables,
        player0_params: object,
        player1_params: object,
    ) -> DynamicPairedArenaResultV2:
        def body(carry, _):
            return dynamic_two_policy_step_v2(
                carry,
                tables,
                player0_params,
                player1_params,
                deterministic=deterministic,
                include_opponent=include_opponent,
                include_history=include_history,
                task_card_program=task_card_program,
                allow_nonpositive_econ=allow_nonpositive_econ,
                player0_allow_nonpositive_econ=player0_allow_nonpositive_econ,
                player1_allow_nonpositive_econ=player1_allow_nonpositive_econ,
            )

        final_carry, traces = jax.lax.scan(
            body, initial_carry, xs=None, length=rollout_steps
        )
        money = final_carry.environment_state.money
        return DynamicPairedArenaResultV2(
            final_carry=final_carry,
            player0_outcome=jnp.sign(money[:, 0] - money[:, 1]).astype(jnp.int8),
            invalid_market_orders=traces[0],
            overflow_market_orders=traces[1],
        )

    return rollout


def make_dynamic_paired_arena_rollout_runtime_gates_v2(
    *,
    rollout_steps: int = EPISODE_STEPS - 1,
    deterministic: bool = True,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
):
    """Build one arena executable whose two econ gates are runtime scalars.

    The original factory intentionally keeps the gate in the Python closure, which
    is convenient for normal training but forces a separate XLA compilation for
    hard/hard, soft/soft and the two mixed-seat evaluation panels.  An ablation
    needs all four combinations with identical simulator semantics, so this
    variant accepts the two boolean gates as scalar arguments and reuses one
    compiled executable.
    """

    def rollout(
        initial_carry: FullLearnedCarryV2,
        tables: StaticTables,
        player0_params: object,
        player1_params: object,
        player0_allow_nonpositive_econ: jax.Array,
        player1_allow_nonpositive_econ: jax.Array,
    ) -> DynamicPairedArenaResultV2:
        def body(carry, _):
            return dynamic_two_policy_step_v2(
                carry,
                tables,
                player0_params,
                player1_params,
                deterministic=deterministic,
                include_opponent=include_opponent,
                include_history=include_history,
                task_card_program=task_card_program,
                player0_allow_nonpositive_econ=player0_allow_nonpositive_econ,
                player1_allow_nonpositive_econ=player1_allow_nonpositive_econ,
            )

        final_carry, traces = jax.lax.scan(
            body, initial_carry, xs=None, length=rollout_steps
        )
        money = final_carry.environment_state.money
        return DynamicPairedArenaResultV2(
            final_carry=final_carry,
            player0_outcome=jnp.sign(money[:, 0] - money[:, 1]).astype(jnp.int8),
            invalid_market_orders=traces[0],
            overflow_market_orders=traces[1],
        )

    return rollout
