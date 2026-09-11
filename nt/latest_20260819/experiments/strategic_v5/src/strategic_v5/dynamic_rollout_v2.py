"""End-to-end GPU rollout for the dynamic Full-core V2 policy."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import TURNS_PER_DAY
from kaggriculture_jax.simulator import (
    batched_project_unit_phase,
    batched_step_from_projected_unit_phase_sync,
)
from kaggriculture_jax.types import Action, State, StaticTables

from .dynamic_policy_v2 import (
    DynamicFullDecisionV2,
    DynamicUnitDecisionV2,
    combine_dynamic_actions_v2,
    select_dynamic_full_core_v2,
    select_dynamic_unit_phase_v2,
)
from .e4_executor import (
    update_full_core_controller_from_effects_v1,
)
from .e5_rollout import cleanup_full_controller_day_end_v1
from .rollout_v2 import FullLearnedCarryV2
from .lifecycle import clear_invalidated_full_core_tasks_v1
from .opponent_v2 import update_opponent_history_v2
from .task_cards import ReplayTaskCardProgramV1


class DynamicFullStepTraceV2(NamedTuple):
    joint_logprob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    market_count: jax.Array
    invalid_market_orders: jax.Array
    overflow_market_orders: jax.Array


class DynamicFullRolloutV2(NamedTuple):
    final_carry: FullLearnedCarryV2
    transitions: DynamicFullStepTraceV2


_PLAYER_STATE_FIELDS_V2 = {
    "money",
    "tile_kind",
    "tile_crop",
    "tile_animal",
    "tile_origin_day",
    "tile_yield",
    "tile_neglect",
    "tile_max_lifespan",
    "tile_fertilized_until",
    "tile_pending_care",
    "tile_flags",
    "unit_pos",
    "unit_active",
    "unit_inventory",
    "unit_inventory_order",
    "unit_inventory_next_order",
    "shed",
    "seeds",
    "hires_today",
    "unlocked_count",
    "reward",
    "hand_cap_hits",
}


def _swap_state_players_v2(states: State) -> State:
    replacements = {
        name: jnp.flip(value, axis=1)
        for name, value in zip(states._fields, states, strict=True)
        if name in _PLAYER_STATE_FIELDS_V2
    }
    return states._replace(**replacements)


def _swap_action_players_v2(action: Action) -> Action:
    return jax.tree.map(lambda value: jnp.flip(value, axis=1), action)


def _concat_batch_v2(left, right):
    return jax.tree.map(
        lambda left_value, right_value: jnp.concatenate(
            (left_value, right_value), axis=0
        ),
        left,
        right,
    )


def _slice_batch_v2(value, start: int, stop: int):
    return jax.tree.map(lambda leaf: leaf[start:stop], value)


def select_dynamic_full_core_pair_v2(
    states: State,
    controller0,
    controller1,
    history,
    tables: StaticTables,
    params: object,
    key0: jax.Array,
    key1: jax.Array,
    unit0: DynamicUnitDecisionV2,
    unit1: DynamicUnitDecisionV2,
    projected_unit_state: State,
    *,
    deterministic: bool,
    include_opponent: bool,
    include_history: bool,
    task_card_program: ReplayTaskCardProgramV1 | None,
    allow_nonpositive_econ: bool,
) -> tuple[DynamicFullDecisionV2, DynamicFullDecisionV2]:
    """Decode both seats in one 2B market batch.

    The second half is seat-normalized so both halves use the actor-0 ledger
    path.  Unit/full-econ work remains actor-specific and is only concatenated;
    the learned market scorer is then invoked once per ordinal instead of once
    per ordinal *and* seat.  The result is split and seat 1 is restored before
    the official shared-market simulator consumes the actions.
    """

    batch_size = states.step.shape[0]
    paired_unit1 = unit1._replace(action=_swap_action_players_v2(unit1.action))
    paired_unit = _concat_batch_v2(unit0, paired_unit1)
    paired_states = _concat_batch_v2(states, states)
    paired_projected = _concat_batch_v2(
        projected_unit_state, _swap_state_players_v2(projected_unit_state)
    )
    paired_controller = _concat_batch_v2(controller0, controller1)
    paired_history = _concat_batch_v2(history, history)
    key1_data = jax.random.key_data(key1)
    paired_key = jax.random.fold_in(key0, jnp.bitwise_xor.reduce(key1_data))
    paired = select_dynamic_full_core_v2(
        paired_states,
        paired_controller,
        paired_history,
        tables,
        params,
        0,
        paired_key,
        deterministic=deterministic,
        include_opponent=include_opponent,
        include_history=include_history,
        task_card_program=task_card_program,
        allow_nonpositive_econ=allow_nonpositive_econ,
        unit_decision=paired_unit,
        projected_unit_state=paired_projected,
    )
    decision0 = _slice_batch_v2(paired, 0, batch_size)
    decision1 = _slice_batch_v2(paired, batch_size, batch_size * 2)
    decision1 = decision1._replace(
        action=_swap_action_players_v2(decision1.action),
        ledger=decision1.ledger._replace(
            player=jnp.ones_like(decision1.ledger.player)
        ),
    )
    return decision0, decision1


def dynamic_full_step_v2(
    carry: FullLearnedCarryV2,
    tables: StaticTables,
    params: object,
    *,
    deterministic: bool = False,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = False,
) -> tuple[FullLearnedCarryV2, DynamicFullStepTraceV2]:
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
    unit_action = combine_dynamic_actions_v2(unit0.action, unit1.action)
    projected_unit_state = batched_project_unit_phase(states, unit_action)
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
        projected_unit_state,
        deterministic=deterministic,
        include_opponent=include_opponent,
        include_history=include_history,
        task_card_program=task_card_program,
        allow_nonpositive_econ=allow_nonpositive_econ,
    )
    action = combine_dynamic_actions_v2(decision0.action, decision1.action)
    next_states = batched_step_from_projected_unit_phase_sync(
        projected_unit_state, action, carry.current_events, tables
    )

    # Effect updaters consume only the compiled unit-op metadata; market
    # success is inferred from before/after official state.  Reuse the exact
    # unit compilation that was executed instead of compiling the final
    # controller again (which could also reorder dynamic market orders).
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
    outcome = jnp.sign(next_states.money[:, 0] - next_states.money[:, 1]).astype(
        jnp.float32
    )
    just_finished = next_states.done & (~states.done)
    reward = jnp.stack((outcome, -outcome), axis=1) * just_finished[:, None]
    trace = DynamicFullStepTraceV2(
        joint_logprob=jnp.stack(
            (
                decision0.unit_selection.joint_logprob
                + decision0.market_trace.joint_logprob,
                decision1.unit_selection.joint_logprob
                + decision1.market_trace.joint_logprob,
            ),
            axis=1,
        ),
        value=jnp.stack((decision0.value, decision1.value), axis=1),
        reward=reward,
        done=jnp.broadcast_to(next_states.done[:, None], reward.shape),
        market_count=jnp.stack(
            (decision0.ledger.market_count, decision1.ledger.market_count), axis=1
        ),
        invalid_market_orders=jnp.stack(
            (
                decision0.ledger.invalid_order_count,
                decision1.ledger.invalid_order_count,
            ),
            axis=1,
        ),
        overflow_market_orders=jnp.stack(
            (
                decision0.ledger.overflow_order_count,
                decision1.ledger.overflow_order_count,
            ),
            axis=1,
        ),
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
    ), trace


def make_dynamic_full_collector_v2(
    *,
    rollout_steps: int,
    deterministic: bool = False,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = False,
):
    def collect(carry: FullLearnedCarryV2, tables: StaticTables, params: object):
        def body(current, _):
            return dynamic_full_step_v2(
                current,
                tables,
                params,
                deterministic=deterministic,
                include_opponent=include_opponent,
                include_history=include_history,
                task_card_program=task_card_program,
                allow_nonpositive_econ=allow_nonpositive_econ,
            )

        final, transitions = jax.lax.scan(body, carry, xs=None, length=rollout_steps)
        return DynamicFullRolloutV2(final, transitions)

    return collect
