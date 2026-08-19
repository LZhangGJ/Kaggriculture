"""Seat-balanced learned-policy versus frozen Boatlee V16 GPU arena."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax import reset
from kaggriculture_jax.constants import EPISODE_STEPS, TURNS_PER_DAY
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.types import Action, Events, State, StaticTables

from .boatlee_v16_gpu import (
    BoatleeCarryV1,
    BoatleeTraceV1,
    boatlee_actions_v1,
    initialize_boatlee_carry_v1,
)
from .constants import MAX_CANDIDATES_V1, MAX_SELECTIONS_V1
from .e4_executor import (
    compile_full_core_action_bundle_v1,
    update_full_core_controller_from_effects_v1,
)
from .e5_rollout import cleanup_full_controller_day_end_v1
from .learned_v1 import (
    CANDIDATE_FEATURE_DIM_V1,
    GLOBAL_FEATURE_DIM_V1,
    FullLearnedTransitionV1,
    FullTimelineV1,
    _bootstrap_value_v1,
    _player_decision_v1,
)
from .lifecycle import clear_invalidated_full_core_tasks_v1, reset_controller_state_v1
from .schema import ControllerStateV1


class BoatleeArenaCarryV1(NamedTuple):
    environment_state: State
    player0_controller: ControllerStateV1
    player1_controller: ControllerStateV1
    boatlee_carry: BoatleeCarryV1
    current_events: Events
    learner_player: jax.Array
    rng: jax.Array


class BoatleeStridedRolloutV1(NamedTuple):
    final_carry: BoatleeArenaCarryV1
    transitions: FullLearnedTransitionV1
    timeline: FullTimelineV1
    bootstrap_value: jax.Array


class BoatleeArenaResultV1(NamedTuple):
    final_carry: BoatleeArenaCarryV1
    learner_margin: jax.Array
    learner_outcome: jax.Array


def _controller_batch(batch_size: int) -> ControllerStateV1:
    one = reset_controller_state_v1()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape), one
    )


def initialize_boatlee_arena_carry_v1(
    seeds: jax.Array,
    events: Events,
    learner_player: jax.Array,
    key: jax.Array,
) -> BoatleeArenaCarryV1:
    seeds = jnp.asarray(seeds, dtype=jnp.int32)
    learner_player = jnp.asarray(learner_player, dtype=jnp.int8)
    if seeds.shape != learner_player.shape:
        raise ValueError("seeds and learner_player must have the same shape")
    batch_size = seeds.shape[0]
    return BoatleeArenaCarryV1(
        environment_state=jax.vmap(reset)(seeds),
        player0_controller=_controller_batch(batch_size),
        player1_controller=_controller_batch(batch_size),
        boatlee_carry=initialize_boatlee_carry_v1(batch_size),
        current_events=events,
        learner_player=learner_player,
        rng=key,
    )


def _select_seat(left: jax.Array, right: jax.Array, seat: jax.Array) -> jax.Array:
    values = jnp.stack((left, right), axis=1)
    return values[jnp.arange(seat.shape[0]), seat.astype(jnp.int32)]


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
    players = jnp.arange(2)[None, :]
    selected = players == learner_player[:, None]

    def choose(learned_value, boatlee_value):
        mask = selected.reshape(
            selected.shape + (1,) * (learned_value.ndim - selected.ndim)
        )
        return jnp.where(mask, learned_value, boatlee_value)

    return jax.tree.map(choose, learned, boatlee)


def boatlee_learned_step_v1(
    carry: BoatleeArenaCarryV1,
    tables: StaticTables,
    trace: BoatleeTraceV1,
    params: object,
    *,
    deterministic: bool = False,
    decision_interval: int = 8,
) -> tuple[BoatleeArenaCarryV1, FullLearnedTransitionV1]:
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
    batch_size = states.step.shape[0]

    def decide(_):
        selection0, global0, features0, task0, value0 = _player_decision_v1(
            states, controller0, tables, params, 0, key0, deterministic
        )
        selection1, global1, features1, task1, value1 = _player_decision_v1(
            states, controller1, tables, params, 1, key1, deterministic
        )
        return (
            selection0.controller,
            selection1.controller,
            _select_seat(global0, global1, carry.learner_player),
            _select_seat(features0, features1, carry.learner_player),
            _select_seat(task0, task1, carry.learner_player),
            _select_seat(selection0.masks, selection1.masks, carry.learner_player),
            _select_seat(
                selection0.selected_candidate_indices,
                selection1.selected_candidate_indices,
                carry.learner_player,
            ),
            _select_seat(
                selection0.joint_logprob,
                selection1.joint_logprob,
                carry.learner_player,
            ),
            _select_seat(value0, value1, carry.learner_player),
        )

    def continue_existing(_):
        stop_masks = jnp.zeros(
            (batch_size, MAX_SELECTIONS_V1, MAX_CANDIDATES_V1 + 1),
            dtype=jnp.bool_,
        ).at[..., -1].set(True)
        return (
            controller0,
            controller1,
            jnp.zeros((batch_size, GLOBAL_FEATURE_DIM_V1), dtype=jnp.float32),
            jnp.zeros(
                (batch_size, MAX_CANDIDATES_V1, CANDIDATE_FEATURE_DIM_V1),
                dtype=jnp.float16,
            ),
            jnp.zeros((batch_size, MAX_CANDIDATES_V1), dtype=jnp.int8),
            stop_masks,
            jnp.full((batch_size, MAX_SELECTIONS_V1), -1, dtype=jnp.int16),
            jnp.zeros((batch_size,), dtype=jnp.float32),
            jnp.zeros((batch_size,), dtype=jnp.float32),
        )

    (
        selected0,
        selected1,
        global_features,
        candidate_features,
        candidate_task_type,
        task_masks,
        task_selected_indices,
        old_logprob,
        old_value,
    ) = jax.lax.cond(should_decide, decide, continue_existing, operand=None)
    learned_bundle = compile_full_core_action_bundle_v1(states, selected0, selected1)
    boatlee_action, next_boatlee_carry = boatlee_actions_v1(
        states, tables, trace, carry.boatlee_carry
    )
    action = _combine_actions(
        learned_bundle.action, boatlee_action, carry.learner_player
    )
    next_states = batched_step_sync(states, action, carry.current_events, tables)
    updated0, _ = update_full_core_controller_from_effects_v1(
        states, next_states, selected0, learned_bundle.player0, 0
    )
    updated1, _ = update_full_core_controller_from_effects_v1(
        states, next_states, selected1, learned_bundle.player1, 1
    )
    updated0 = _select_controller_update(
        updated0, selected0, carry.learner_player == 0
    )
    updated1 = _select_controller_update(
        updated1, selected1, carry.learner_player == 1
    )
    day_end = ((next_states.step % TURNS_PER_DAY) == 0) & (~next_states.done)
    updated0 = cleanup_full_controller_day_end_v1(
        updated0, next_states.unit_active[:, 0], day_end
    )
    updated1 = cleanup_full_controller_day_end_v1(
        updated1, next_states.unit_active[:, 1], day_end
    )
    learner_money = _select_seat(
        next_states.money[:, 0], next_states.money[:, 1], carry.learner_player
    )
    opponent_money = _select_seat(
        next_states.money[:, 1], next_states.money[:, 0], carry.learner_player
    )
    margin = learner_money.astype(jnp.float32) - opponent_money.astype(jnp.float32)
    just_finished = next_states.done & (~states.done)
    reward = jnp.where(
        just_finished, jnp.clip(margin / 100_000.0, -1.0, 1.0), 0.0
    )[:, None]
    transition = FullLearnedTransitionV1(
        global_features=global_features[:, None],
        candidate_features=candidate_features[:, None],
        candidate_task_type=candidate_task_type[:, None],
        task_masks=task_masks[:, None],
        task_selected_indices=task_selected_indices[:, None],
        old_logprob=old_logprob[:, None],
        old_value=old_value[:, None],
        reward=reward,
        done=next_states.done[:, None],
    )
    return (
        BoatleeArenaCarryV1(
            next_states,
            updated0,
            updated1,
            next_boatlee_carry,
            carry.current_events,
            carry.learner_player,
            next_key,
        ),
        transition,
    )


def _boatlee_bootstrap_value_v1(
    carry: BoatleeArenaCarryV1, tables: StaticTables, params: object
) -> jax.Array:
    from .learned_v1 import FullLearnedCarryV1

    proxy = FullLearnedCarryV1(
        carry.environment_state,
        carry.player0_controller,
        carry.player1_controller,
        carry.current_events,
        carry.rng,
    )
    both = _bootstrap_value_v1(proxy, tables, params)
    selected = both[
        jnp.arange(carry.learner_player.shape[0]),
        carry.learner_player.astype(jnp.int32),
    ]
    return selected[:, None]


def make_boatlee_strided_collector_v1(
    *,
    rollout_steps: int,
    sample_stride: int,
    deterministic: bool = False,
    decision_interval: int = 8,
    include_final_sample: bool = False,
):
    groups, remainder = divmod(rollout_steps, sample_stride)

    def collect(initial_carry, tables, trace, params):
        def one_step(carry, _):
            return boatlee_learned_step_v1(
                carry,
                tables,
                trace,
                params,
                deterministic=deterministic,
                decision_interval=decision_interval,
            )

        def one_group(carry, _):
            following, transitions = jax.lax.scan(
                one_step, carry, xs=None, length=sample_stride
            )
            sample = jax.tree.map(lambda value: value[0], transitions)
            timeline = FullTimelineV1(
                transitions.old_value, transitions.reward, transitions.done
            )
            return following, (sample, timeline)

        carry = initial_carry
        samples = []
        timelines = []
        if groups:
            carry, (group_samples, group_timeline) = jax.lax.scan(
                one_group, carry, xs=None, length=groups
            )
            samples.append(group_samples)
            timelines.append(
                jax.tree.map(
                    lambda value: value.reshape(
                        (groups * sample_stride,) + value.shape[2:]
                    ),
                    group_timeline,
                )
            )
        if remainder:
            carry, tail = jax.lax.scan(one_step, carry, xs=None, length=remainder)
            samples.append(jax.tree.map(lambda value: value[0:1], tail))
            timelines.append(FullTimelineV1(tail.old_value, tail.reward, tail.done))
            if include_final_sample and remainder > 1:
                samples.append(jax.tree.map(lambda value: value[-1:], tail))
        elif include_final_sample and rollout_steps > 1:
            raise ValueError("include_final_sample requires a non-zero remainder")

        transitions = (
            samples[0]
            if len(samples) == 1
            else jax.tree.map(lambda *values: jnp.concatenate(values), *samples)
        )
        timeline = (
            timelines[0]
            if len(timelines) == 1
            else jax.tree.map(lambda *values: jnp.concatenate(values), *timelines)
        )
        return BoatleeStridedRolloutV1(
            carry,
            transitions,
            timeline,
            _boatlee_bootstrap_value_v1(carry, tables, params),
        )

    return collect


def make_boatlee_arena_rollout_v1(
    *, rollout_steps: int = EPISODE_STEPS - 1, decision_interval: int = 8
):
    def rollout(initial_carry, tables, trace, params):
        def body(carry, _):
            following, _ = boatlee_learned_step_v1(
                carry,
                tables,
                trace,
                params,
                deterministic=True,
                decision_interval=decision_interval,
            )
            return following, None

        final_carry, _ = jax.lax.scan(
            body, initial_carry, xs=None, length=rollout_steps
        )
        final = final_carry.environment_state
        learner_money = _select_seat(
            final.money[:, 0], final.money[:, 1], final_carry.learner_player
        )
        opponent_money = _select_seat(
            final.money[:, 1], final.money[:, 0], final_carry.learner_player
        )
        margin = learner_money.astype(jnp.int32) - opponent_money.astype(jnp.int32)
        return BoatleeArenaResultV1(
            final_carry, margin, jnp.sign(margin).astype(jnp.int8)
        )

    return rollout
