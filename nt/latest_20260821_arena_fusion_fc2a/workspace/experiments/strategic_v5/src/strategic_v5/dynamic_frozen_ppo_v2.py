"""Learner-only strided PPO rollouts against a frozen Dynamic V2 opponent."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import TURNS_PER_DAY
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
from .dynamic_ppo_v2 import (
    POLICY_STEPS_V2,
    DynamicPPOCollectorConfigV2,
    DynamicPPOTransitionV2,
    dynamic_sample_steps_v2,
)
from .e4_executor import update_full_core_controller_from_effects_v1
from .e5_rollout import cleanup_full_controller_day_end_v1
from .learned_v1 import FullTimelineV1
from .lifecycle import clear_invalidated_full_core_tasks_v1
from .opponent_v2 import update_opponent_history_v2
from .rollout_v2 import FullLearnedCarryV2


class DynamicFrozenStepAuditV2(NamedTuple):
    invalid_market_orders: jax.Array
    overflow_market_orders: jax.Array


class DynamicFrozenStridedRolloutV2(NamedTuple):
    final_carry: FullLearnedCarryV2
    transitions: DynamicPPOTransitionV2
    timeline: FullTimelineV1
    bootstrap_value: jax.Array
    invalid_market_orders: jax.Array
    overflow_market_orders: jax.Array


def _seat_axis(value: jax.Array) -> jax.Array:
    return value[:, None, ...]


def _dynamic_frozen_decisions_v2(
    carry: FullLearnedCarryV2,
    tables: StaticTables,
    learner_params: object,
    frozen_params: object,
    *,
    learner_player: int,
    config: DynamicPPOCollectorConfigV2,
):
    if learner_player not in (0, 1):
        raise ValueError("learner_player must be 0 or 1")
    player0_params = learner_params if learner_player == 0 else frozen_params
    player1_params = frozen_params if learner_player == 0 else learner_params
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
        deterministic=config.deterministic,
        include_opponent=config.include_opponent,
        include_history=config.include_history,
        task_card_program=config.task_card_program,
        allow_nonpositive_econ=config.allow_nonpositive_econ,
    )
    unit1 = select_dynamic_unit_phase_v2(
        states,
        controller1,
        carry.opponent_history,
        tables,
        player1_params,
        1,
        unit_key1,
        deterministic=config.deterministic,
        include_opponent=config.include_opponent,
        include_history=config.include_history,
        task_card_program=config.task_card_program,
        allow_nonpositive_econ=config.allow_nonpositive_econ,
    )
    projected = batched_project_unit_phase(
        states, combine_dynamic_actions_v2(unit0.action, unit1.action)
    )
    decision0 = select_dynamic_full_core_v2(
        states,
        controller0,
        carry.opponent_history,
        tables,
        player0_params,
        0,
        market_key0,
        deterministic=config.deterministic,
        include_opponent=config.include_opponent,
        include_history=config.include_history,
        task_card_program=config.task_card_program,
        allow_nonpositive_econ=config.allow_nonpositive_econ,
        unit_decision=unit0,
        projected_unit_state=projected,
    )
    decision1 = select_dynamic_full_core_v2(
        states,
        controller1,
        carry.opponent_history,
        tables,
        player1_params,
        1,
        market_key1,
        deterministic=config.deterministic,
        include_opponent=config.include_opponent,
        include_history=config.include_history,
        task_card_program=config.task_card_program,
        allow_nonpositive_econ=config.allow_nonpositive_econ,
        unit_decision=unit1,
        projected_unit_state=projected,
    )
    action = combine_dynamic_actions_v2(decision0.action, decision1.action)
    next_states = batched_step_from_projected_unit_phase_sync(
        projected, action, carry.current_events, tables
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
            if config.include_history
            else carry.opponent_history
        ),
        rng=next_key,
    )
    learner = decision0 if learner_player == 0 else decision1
    outcome0 = jnp.sign(next_states.money[:, 0] - next_states.money[:, 1]).astype(
        jnp.float32
    )
    learner_outcome = outcome0 if learner_player == 0 else -outcome0
    just_finished = next_states.done & (~states.done)
    reward = learner_outcome * just_finished
    audit = DynamicFrozenStepAuditV2(
        invalid_market_orders=learner.ledger.invalid_order_count,
        overflow_market_orders=learner.ledger.overflow_order_count,
    )
    return next_carry, learner, reward, next_states.done, audit


def dynamic_frozen_sample_step_v2(
    carry: FullLearnedCarryV2,
    tables: StaticTables,
    learner_params: object,
    frozen_params: object,
    *,
    learner_player: int,
    config: DynamicPPOCollectorConfigV2,
):
    following, learner, reward, done, audit = _dynamic_frozen_decisions_v2(
        carry,
        tables,
        learner_params,
        frozen_params,
        learner_player=learner_player,
        config=config,
    )
    transition = DynamicPPOTransitionV2(
        unit_global_features=_seat_axis(learner.unit_global_features),
        unit_candidate_features=_seat_axis(learner.unit_candidate_features),
        unit_candidate_task_type=_seat_axis(learner.unit_candidate_task_type),
        unit_task_masks=_seat_axis(learner.unit_selection.masks),
        unit_selected_indices=_seat_axis(
            learner.unit_selection.selected_candidate_indices
        ),
        market_candidate_features=_seat_axis(
            learner.market_trace.candidate_features
        ),
        market_candidate_task_type=_seat_axis(
            learner.market_trace.candidate_task_type
        ),
        market_masks=_seat_axis(learner.market_trace.masks),
        market_selected_indices=_seat_axis(learner.market_trace.selected_indices),
        old_logprob=_seat_axis(
            learner.unit_selection.joint_logprob
            + learner.market_trace.joint_logprob
        ),
        old_value=_seat_axis(learner.value),
        reward=_seat_axis(reward),
        done=_seat_axis(done),
    )
    return following, transition, audit


def dynamic_frozen_light_step_v2(
    carry: FullLearnedCarryV2,
    tables: StaticTables,
    learner_params: object,
    frozen_params: object,
    *,
    learner_player: int,
    config: DynamicPPOCollectorConfigV2,
):
    following, learner, reward, done, audit = _dynamic_frozen_decisions_v2(
        carry,
        tables,
        learner_params,
        frozen_params,
        learner_player=learner_player,
        config=config,
    )
    timeline = FullTimelineV1(
        old_value=_seat_axis(learner.value),
        reward=_seat_axis(reward),
        done=_seat_axis(done),
    )
    return following, timeline, audit


def make_dynamic_frozen_strided_collector_v2(
    config: DynamicPPOCollectorConfigV2, *, learner_player: int
):
    """Collect one learner seat; call once per seat and concatenate batches."""

    if learner_player not in (0, 1):
        raise ValueError("learner_player must be 0 or 1")
    sample_steps = dynamic_sample_steps_v2(config.sample_stride)
    regular_steps = tuple(range(0, POLICY_STEPS_V2, config.sample_stride))
    prefix_groups = len(regular_steps) - 1
    tail_start = regular_steps[-1]
    tail_gap = (POLICY_STEPS_V2 - 1) - tail_start

    def collect(carry, tables, learner_params, frozen_params):
        def prefix_body(current, _):
            current, heavy, heavy_audit = dynamic_frozen_sample_step_v2(
                current,
                tables,
                learner_params,
                frozen_params,
                learner_player=learner_player,
                config=config,
            )

            def light_body(inner, __):
                following, timeline, audit = dynamic_frozen_light_step_v2(
                    inner,
                    tables,
                    learner_params,
                    frozen_params,
                    learner_player=learner_player,
                    config=config,
                )
                return following, (timeline, audit)

            current, (light_timeline, light_audit) = jax.lax.scan(
                light_body, current, xs=None, length=config.sample_stride - 1
            )
            group_timeline = jax.tree.map(
                lambda first, rest: jnp.concatenate((first[None], rest), axis=0),
                FullTimelineV1(heavy.old_value, heavy.reward, heavy.done),
                light_timeline,
            )
            invalid = jnp.sum(heavy_audit.invalid_market_orders) + jnp.sum(
                light_audit.invalid_market_orders
            )
            overflow = jnp.sum(heavy_audit.overflow_market_orders) + jnp.sum(
                light_audit.overflow_market_orders
            )
            return current, (heavy, group_timeline, invalid, overflow)

        carry, (prefix_heavy, prefix_timeline, invalid, overflow) = jax.lax.scan(
            prefix_body, carry, xs=None, length=prefix_groups
        )
        invalid_total = jnp.sum(invalid)
        overflow_total = jnp.sum(overflow)
        carry, tail0, tail0_audit = dynamic_frozen_sample_step_v2(
            carry,
            tables,
            learner_params,
            frozen_params,
            learner_player=learner_player,
            config=config,
        )
        tail_heavy = [tail0]
        tail_timeline = [
            jax.tree.map(
                lambda value: value[None],
                FullTimelineV1(tail0.old_value, tail0.reward, tail0.done),
            )
        ]
        invalid_total += jnp.sum(tail0_audit.invalid_market_orders)
        overflow_total += jnp.sum(tail0_audit.overflow_market_orders)
        if tail_gap > 0:
            between = tail_gap - 1
            if between > 0:
                def tail_light_body(inner, _):
                    following, timeline, audit = dynamic_frozen_light_step_v2(
                        inner,
                        tables,
                        learner_params,
                        frozen_params,
                        learner_player=learner_player,
                        config=config,
                    )
                    return following, (timeline, audit)

                carry, (middle_timeline, middle_audit) = jax.lax.scan(
                    tail_light_body, carry, xs=None, length=between
                )
                tail_timeline.append(middle_timeline)
                invalid_total += jnp.sum(middle_audit.invalid_market_orders)
                overflow_total += jnp.sum(middle_audit.overflow_market_orders)
            carry, final_heavy, final_audit = dynamic_frozen_sample_step_v2(
                carry,
                tables,
                learner_params,
                frozen_params,
                learner_player=learner_player,
                config=config,
            )
            tail_heavy.append(final_heavy)
            tail_timeline.append(
                jax.tree.map(
                    lambda value: value[None],
                    FullTimelineV1(
                        final_heavy.old_value, final_heavy.reward, final_heavy.done
                    ),
                )
            )
            invalid_total += jnp.sum(final_audit.invalid_market_orders)
            overflow_total += jnp.sum(final_audit.overflow_market_orders)
        transitions = jax.tree.map(
            lambda prefix, *tail: jnp.concatenate(
                (prefix, *[value[None] for value in tail]), axis=0
            ),
            prefix_heavy,
            *tail_heavy,
        )
        prefix_timeline = jax.tree.map(
            lambda value: value.reshape((-1,) + value.shape[2:]), prefix_timeline
        )
        timeline = jax.tree.map(
            lambda prefix, *tail: jnp.concatenate((prefix, *tail), axis=0),
            prefix_timeline,
            *tail_timeline,
        )
        bootstrap = jnp.zeros_like(timeline.old_value[-1])
        return DynamicFrozenStridedRolloutV2(
            carry,
            transitions,
            timeline,
            bootstrap,
            invalid_total,
            overflow_total,
        )

    collect.sample_steps = sample_steps
    return collect
