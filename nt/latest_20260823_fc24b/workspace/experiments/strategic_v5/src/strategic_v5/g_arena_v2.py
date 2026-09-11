"""Seat-balanced two-checkpoint arena for opponent-aware V2 policies."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import EPISODE_STEPS, TURNS_PER_DAY
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.types import StaticTables

from .e4_executor import (
    compile_full_core_action_bundle_v1,
    update_full_core_controller_from_effects_v1,
)
from .e5_rollout import cleanup_full_controller_day_end_v1
from .lifecycle import clear_invalidated_full_core_tasks_v1
from .rollout_v2 import FullLearnedCarryV2, _player_decision_v2
from .task_cards import ReplayTaskCardProgramV1
from .opponent_v2 import update_opponent_history_v2


class FullPairedArenaResultV2(NamedTuple):
    final_carry: FullLearnedCarryV2
    player0_outcome: jax.Array


def full_two_policy_step_v2(
    carry: FullLearnedCarryV2,
    tables: StaticTables,
    player0_params: object,
    player1_params: object,
    *,
    deterministic: bool = True,
    decision_interval: int = 1,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = False,
) -> FullLearnedCarryV2:
    states = carry.environment_state
    next_key, key0, key1 = jax.random.split(carry.rng, 3)
    controller0 = clear_invalidated_full_core_tasks_v1(
        states, carry.player0_controller, 0
    )
    controller1 = clear_invalidated_full_core_tasks_v1(
        states, carry.player1_controller, 1
    )
    should_decide = ((states.step[0] % decision_interval) == 0) | (
        states.step[0] >= EPISODE_STEPS - 2
    )

    def decide(_):
        selection0, _, _, _, _ = _player_decision_v2(
            states,
            controller0,
            carry.opponent_history,
            tables,
            player0_params,
            0,
            key0,
            deterministic,
            include_opponent,
            include_history,
            task_card_program,
            allow_nonpositive_econ,
        )
        selection1, _, _, _, _ = _player_decision_v2(
            states,
            controller1,
            carry.opponent_history,
            tables,
            player1_params,
            1,
            key1,
            deterministic,
            include_opponent,
            include_history,
            task_card_program,
            allow_nonpositive_econ,
        )
        return selection0.controller, selection1.controller

    selected0, selected1 = jax.lax.cond(
        should_decide,
        decide,
        lambda _: (controller0, controller1),
        operand=None,
    )
    bundle = compile_full_core_action_bundle_v1(states, selected0, selected1)
    next_states = batched_step_sync(
        states, bundle.action, carry.current_events, tables
    )
    controller0, _ = update_full_core_controller_from_effects_v1(
        states, next_states, selected0, bundle.player0, 0
    )
    controller1, _ = update_full_core_controller_from_effects_v1(
        states, next_states, selected1, bundle.player1, 1
    )
    day_end = ((next_states.step % TURNS_PER_DAY) == 0) & (~next_states.done)
    controller0 = cleanup_full_controller_day_end_v1(
        controller0, next_states.unit_active[:, 0], day_end
    )
    controller1 = cleanup_full_controller_day_end_v1(
        controller1, next_states.unit_active[:, 1], day_end
    )
    return FullLearnedCarryV2(
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


def make_full_paired_arena_rollout_v2(
    *,
    rollout_steps: int = EPISODE_STEPS - 1,
    deterministic: bool = True,
    decision_interval: int = 1,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = False,
):
    def rollout(
        initial_carry: FullLearnedCarryV2,
        tables: StaticTables,
        player0_params: object,
        player1_params: object,
    ) -> FullPairedArenaResultV2:
        def body(carry, _):
            return (
                full_two_policy_step_v2(
                    carry,
                    tables,
                    player0_params,
                    player1_params,
                    deterministic=deterministic,
                    decision_interval=decision_interval,
                    include_opponent=include_opponent,
                    include_history=include_history,
                    task_card_program=task_card_program,
                    allow_nonpositive_econ=allow_nonpositive_econ,
                ),
                None,
            )

        final_carry, _ = jax.lax.scan(
            body, initial_carry, xs=None, length=rollout_steps
        )
        money = final_carry.environment_state.money
        return FullPairedArenaResultV2(
            final_carry=final_carry,
            player0_outcome=jnp.sign(money[:, 0] - money[:, 1]).astype(jnp.int8),
        )

    return rollout
