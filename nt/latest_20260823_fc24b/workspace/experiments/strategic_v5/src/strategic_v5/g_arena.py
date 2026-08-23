"""Seat-balanced two-checkpoint arena for V5 G and later promotion gates.

This module deliberately lives outside the F training hot path.  It uses the
same Full-core candidate builder, Full-econ features and persistent executor,
but allows player 0 and player 1 to use different parameter trees.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import EPISODE_STEPS, TURNS_PER_DAY
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.types import State, StaticTables

from .e4_executor import (
    compile_full_core_action_bundle_v1,
    update_full_core_controller_from_effects_v1,
)
from .e5_rollout import cleanup_full_controller_day_end_v1
from .learned_v1 import FullLearnedCarryV1, _player_decision_v1
from .lifecycle import clear_invalidated_full_core_tasks_v1


class FullPairedArenaResultV1(NamedTuple):
    final_carry: FullLearnedCarryV1
    player0_outcome: jax.Array


def full_two_policy_step_v1(
    carry: FullLearnedCarryV1,
    tables: StaticTables,
    player0_params: object,
    player1_params: object,
    *,
    deterministic: bool = True,
    decision_interval: int = 8,
) -> FullLearnedCarryV1:
    """Advance one official step with an independently frozen policy per seat."""

    states = carry.environment_state
    next_key, key0, key1 = jax.random.split(carry.rng, 3)
    controller0 = clear_invalidated_full_core_tasks_v1(
        states, carry.player0_controller, 0
    )
    controller1 = clear_invalidated_full_core_tasks_v1(
        states, carry.player1_controller, 1
    )
    should_decide = (
        (states.step[0] % decision_interval) == 0
    ) | (states.step[0] >= EPISODE_STEPS - 2)

    def decide(_):
        selection0, _, _, _, _ = _player_decision_v1(
            states,
            controller0,
            tables,
            player0_params,
            0,
            key0,
            deterministic,
        )
        selection1, _, _, _, _ = _player_decision_v1(
            states,
            controller1,
            tables,
            player1_params,
            1,
            key1,
            deterministic,
        )
        return selection0.controller, selection1.controller

    selected_controller0, selected_controller1 = jax.lax.cond(
        should_decide,
        decide,
        lambda _: (controller0, controller1),
        operand=None,
    )
    bundle = compile_full_core_action_bundle_v1(
        states, selected_controller0, selected_controller1
    )
    next_states = batched_step_sync(
        states, bundle.action, carry.current_events, tables
    )
    controller0, _ = update_full_core_controller_from_effects_v1(
        states, next_states, selected_controller0, bundle.player0, 0
    )
    controller1, _ = update_full_core_controller_from_effects_v1(
        states, next_states, selected_controller1, bundle.player1, 1
    )
    day_end = ((next_states.step % TURNS_PER_DAY) == 0) & (~next_states.done)
    controller0 = cleanup_full_controller_day_end_v1(
        controller0, next_states.unit_active[:, 0], day_end
    )
    controller1 = cleanup_full_controller_day_end_v1(
        controller1, next_states.unit_active[:, 1], day_end
    )
    return FullLearnedCarryV1(
        next_states,
        controller0,
        controller1,
        carry.current_events,
        next_key,
    )


def make_full_paired_arena_rollout_v1(
    *,
    rollout_steps: int = EPISODE_STEPS - 1,
    deterministic: bool = True,
    decision_interval: int = 8,
):
    """Return a JAX-native complete arena rollout for two parameter trees."""

    def rollout(
        initial_carry: FullLearnedCarryV1,
        tables: StaticTables,
        player0_params: object,
        player1_params: object,
    ) -> FullPairedArenaResultV1:
        def body(carry, _):
            return (
                full_two_policy_step_v1(
                    carry,
                    tables,
                    player0_params,
                    player1_params,
                    deterministic=deterministic,
                    decision_interval=decision_interval,
                ),
                None,
            )

        final_carry, _ = jax.lax.scan(
            body, initial_carry, xs=None, length=rollout_steps
        )
        money = final_carry.environment_state.money
        return FullPairedArenaResultV1(
            final_carry=final_carry,
            player0_outcome=jnp.sign(money[:, 0] - money[:, 1]).astype(jnp.int8),
        )

    return rollout
