"""GPU-native opponent-aware rollout for the Strategic V5 V2 policy."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax import reset
from kaggriculture_jax.constants import EPISODE_STEPS, TURNS_PER_DAY
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.types import Events, State, StaticTables

from .constants import MAX_CANDIDATES_V1, MAX_SELECTIONS_V1
from .e4_core import (
    build_full_core_candidates_v1,
    evaluate_full_core_feasibility_v1,
)
from .e4_executor import (
    compile_full_core_action_bundle_v1,
    update_full_core_controller_from_effects_v1,
)
from .e5_econ import build_full_econ_features_v1
from .e5_rollout import cleanup_full_controller_day_end_v1
from .learned_v1 import (
    FullLearnedTransitionV1,
    FullTimelineV1,
    select_full_learned_candidates_v1,
)
from .learned_v2 import apply_full_learned_model_v2
from .lifecycle import clear_invalidated_full_core_tasks_v1, reset_controller_state_v1
from .opponent_v2 import (
    CANDIDATE_FEATURE_DIM_V2,
    GLOBAL_FEATURE_DIM_V2,
    OpponentHistoryV2,
    build_candidate_features_v2,
    build_global_features_v2,
    initialize_opponent_history_v2,
    update_opponent_history_v2,
)
from .schema import ControllerStateV1
from .task_cards import ReplayTaskCardProgramV1


class FullLearnedCarryV2(NamedTuple):
    environment_state: State
    player0_controller: ControllerStateV1
    player1_controller: ControllerStateV1
    current_events: Events
    opponent_history: OpponentHistoryV2
    rng: jax.Array


class FullLearnedRolloutV2(NamedTuple):
    final_carry: FullLearnedCarryV2
    transitions: FullLearnedTransitionV1
    bootstrap_value: jax.Array


class FullStridedRolloutV2(NamedTuple):
    final_carry: FullLearnedCarryV2
    transitions: FullLearnedTransitionV1
    timeline: FullTimelineV1
    bootstrap_value: jax.Array


def _controller_batch_v2(batch_size: int) -> ControllerStateV1:
    one = reset_controller_state_v1()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape), one
    )


def initialize_full_learned_carry_v2(
    seeds: jax.Array, events: Events, key: jax.Array
) -> FullLearnedCarryV2:
    seeds = jnp.asarray(seeds, dtype=jnp.int32)
    batch_size = seeds.shape[0]
    states = jax.vmap(reset)(seeds)
    return FullLearnedCarryV2(
        environment_state=states,
        player0_controller=_controller_batch_v2(batch_size),
        player1_controller=_controller_batch_v2(batch_size),
        current_events=events,
        opponent_history=initialize_opponent_history_v2(states),
        rng=key,
    )


def _player_decision_v2(
    states: State,
    controller: ControllerStateV1,
    history: OpponentHistoryV2,
    tables: StaticTables,
    params: object,
    player: int,
    key: jax.Array,
    deterministic: bool,
    include_opponent: bool,
    include_history: bool,
    task_card_program: ReplayTaskCardProgramV1 | None,
    allow_nonpositive_econ: bool,
):
    candidates = build_full_core_candidates_v1(
        states, controller, tables, player, task_card_program
    )
    feasibility = evaluate_full_core_feasibility_v1(
        states, candidates, tables, player
    )
    econ = build_full_econ_features_v1(
        states, candidates, feasibility, tables, player
    )
    global_features = build_global_features_v2(
        states,
        player,
        history,
        include_opponent=include_opponent,
        include_history=include_history,
    )
    candidate_features = build_candidate_features_v2(
        states,
        candidates,
        feasibility,
        econ,
        player,
        tables,
        include_opponent=include_opponent,
    )
    output = apply_full_learned_model_v2(
        params, global_features, candidate_features, candidates.task_type
    )
    selection = select_full_learned_candidates_v1(
        states,
        candidates,
        feasibility,
        econ,
        output.candidate_logits,
        output.stop_logit,
        controller,
        player,
        key,
        deterministic=deterministic,
        allow_nonpositive_econ=allow_nonpositive_econ,
    )
    return (
        selection,
        global_features,
        candidate_features,
        candidates.task_type,
        output.value,
    )


def full_learned_step_v2(
    carry: FullLearnedCarryV2,
    tables: StaticTables,
    params: object,
    *,
    deterministic: bool = False,
    decision_interval: int = 1,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = False,
) -> tuple[FullLearnedCarryV2, FullLearnedTransitionV1]:
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
    batch_size = states.step.shape[0]

    def decide(_):
        selection0, global0, features0, task0, value0 = _player_decision_v2(
            states,
            controller0,
            carry.opponent_history,
            tables,
            params,
            0,
            key0,
            deterministic,
            include_opponent,
            include_history,
            task_card_program,
            allow_nonpositive_econ,
        )
        selection1, global1, features1, task1, value1 = _player_decision_v2(
            states,
            controller1,
            carry.opponent_history,
            tables,
            params,
            1,
            key1,
            deterministic,
            include_opponent,
            include_history,
            task_card_program,
            allow_nonpositive_econ,
        )
        return (
            selection0.controller,
            selection1.controller,
            jnp.stack((global0, global1), axis=1),
            jnp.stack((features0, features1), axis=1),
            jnp.stack((task0, task1), axis=1),
            jnp.stack((selection0.masks, selection1.masks), axis=1),
            jnp.stack(
                (
                    selection0.selected_candidate_indices,
                    selection1.selected_candidate_indices,
                ),
                axis=1,
            ),
            jnp.stack((selection0.joint_logprob, selection1.joint_logprob), axis=1),
            jnp.stack((value0, value1), axis=1),
        )

    def continue_existing(_):
        stop_masks = jnp.zeros(
            (
                batch_size,
                2,
                MAX_SELECTIONS_V1,
                MAX_CANDIDATES_V1 + 1,
            ),
            dtype=jnp.bool_,
        ).at[..., -1].set(True)
        return (
            controller0,
            controller1,
            jnp.zeros((batch_size, 2, GLOBAL_FEATURE_DIM_V2), dtype=jnp.float32),
            jnp.zeros(
                (batch_size, 2, MAX_CANDIDATES_V1, CANDIDATE_FEATURE_DIM_V2),
                dtype=jnp.float16,
            ),
            jnp.zeros((batch_size, 2, MAX_CANDIDATES_V1), dtype=jnp.int8),
            stop_masks,
            jnp.full(
                (batch_size, 2, MAX_SELECTIONS_V1), -1, dtype=jnp.int16
            ),
            jnp.zeros((batch_size, 2), dtype=jnp.float32),
            jnp.zeros((batch_size, 2), dtype=jnp.float32),
        )

    (
        selected_controller0,
        selected_controller1,
        global_features,
        candidate_features,
        candidate_task_type,
        task_masks,
        task_selected_indices,
        old_logprob,
        old_value,
    ) = jax.lax.cond(should_decide, decide, continue_existing, operand=None)
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
    outcome = jnp.sign(next_states.money[:, 0] - next_states.money[:, 1]).astype(
        jnp.float32
    )
    just_finished = next_states.done & (~states.done)
    reward = jnp.stack((outcome, -outcome), axis=1) * just_finished[:, None]
    transition = FullLearnedTransitionV1(
        global_features=global_features,
        candidate_features=candidate_features,
        candidate_task_type=candidate_task_type,
        task_masks=task_masks,
        task_selected_indices=task_selected_indices,
        old_logprob=old_logprob,
        old_value=old_value,
        reward=reward,
        done=jnp.broadcast_to(next_states.done[:, None], reward.shape),
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
    ), transition


def _bootstrap_value_v2(
    carry: FullLearnedCarryV2,
    tables: StaticTables,
    params: object,
    *,
    include_opponent: bool,
    include_history: bool,
    task_card_program: ReplayTaskCardProgramV1 | None,
) -> jax.Array:
    values = []
    for player, controller in (
        (0, carry.player0_controller),
        (1, carry.player1_controller),
    ):
        controller = clear_invalidated_full_core_tasks_v1(
            carry.environment_state, controller, player
        )
        candidates = build_full_core_candidates_v1(
            carry.environment_state,
            controller,
            tables,
            player,
            task_card_program,
        )
        feasibility = evaluate_full_core_feasibility_v1(
            carry.environment_state, candidates, tables, player
        )
        econ = build_full_econ_features_v1(
            carry.environment_state, candidates, feasibility, tables, player
        )
        global_features = build_global_features_v2(
            carry.environment_state,
            player,
            carry.opponent_history,
            include_opponent=include_opponent,
            include_history=include_history,
        )
        candidate_features = build_candidate_features_v2(
            carry.environment_state,
            candidates,
            feasibility,
            econ,
            player,
            tables,
            include_opponent=include_opponent,
        )
        values.append(
            apply_full_learned_model_v2(
                params, global_features, candidate_features, candidates.task_type
            ).value
        )
    result = jnp.stack(values, axis=1)
    return jnp.where(carry.environment_state.done[:, None], 0.0, result)


def make_full_learned_collector_v2(
    *,
    rollout_steps: int,
    deterministic: bool = False,
    decision_interval: int = 1,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = False,
):
    def collect(
        initial_carry: FullLearnedCarryV2,
        tables: StaticTables,
        params: object,
    ) -> FullLearnedRolloutV2:
        def body(carry, _):
            return full_learned_step_v2(
                carry,
                tables,
                params,
                deterministic=deterministic,
                decision_interval=decision_interval,
                include_opponent=include_opponent,
                include_history=include_history,
                task_card_program=task_card_program,
                allow_nonpositive_econ=allow_nonpositive_econ,
            )

        final_carry, transitions = jax.lax.scan(
            body, initial_carry, xs=None, length=rollout_steps
        )
        return FullLearnedRolloutV2(
            final_carry=final_carry,
            transitions=transitions,
            bootstrap_value=_bootstrap_value_v2(
                final_carry,
                tables,
                params,
                include_opponent=include_opponent,
                include_history=include_history,
                task_card_program=task_card_program,
            ),
        )

    return collect


def make_full_strided_collector_v2(
    *,
    rollout_steps: int,
    sample_stride: int,
    deterministic: bool = False,
    decision_interval: int = 1,
    include_final_sample: bool = False,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = False,
):
    if rollout_steps <= 0 or sample_stride <= 0:
        raise ValueError("rollout_steps and sample_stride must be positive")
    groups, remainder = divmod(rollout_steps, sample_stride)

    def collect(
        initial_carry: FullLearnedCarryV2,
        tables: StaticTables,
        params: object,
    ) -> FullStridedRolloutV2:
        def one_step(carry, _):
            return full_learned_step_v2(
                carry,
                tables,
                params,
                deterministic=deterministic,
                decision_interval=decision_interval,
                include_opponent=include_opponent,
                include_history=include_history,
                task_card_program=task_card_program,
                allow_nonpositive_econ=allow_nonpositive_econ,
            )

        def one_group(carry, _):
            next_carry, transitions = jax.lax.scan(
                one_step, carry, xs=None, length=sample_stride
            )
            sample = jax.tree.map(lambda value: value[0], transitions)
            timeline = FullTimelineV1(
                old_value=transitions.old_value,
                reward=transitions.reward,
                done=transitions.done,
            )
            return next_carry, (sample, timeline)

        carry = initial_carry
        sample_parts = []
        timeline_parts = []
        if groups:
            carry, (group_samples, group_timeline) = jax.lax.scan(
                one_group, carry, xs=None, length=groups
            )
            sample_parts.append(group_samples)
            timeline_parts.append(
                jax.tree.map(
                    lambda value: value.reshape(
                        (groups * sample_stride,) + value.shape[2:]
                    ),
                    group_timeline,
                )
            )
        if remainder:
            carry, tail = jax.lax.scan(
                one_step, carry, xs=None, length=remainder
            )
            sample_parts.append(jax.tree.map(lambda value: value[0:1], tail))
            timeline_parts.append(FullTimelineV1(tail.old_value, tail.reward, tail.done))
            if include_final_sample and remainder > 1:
                sample_parts.append(jax.tree.map(lambda value: value[-1:], tail))
        elif include_final_sample and rollout_steps > 1:
            raise ValueError("include_final_sample requires a non-zero remainder")

        transitions = (
            sample_parts[0]
            if len(sample_parts) == 1
            else jax.tree.map(
                lambda *values: jnp.concatenate(values, axis=0), *sample_parts
            )
        )
        timeline = (
            timeline_parts[0]
            if len(timeline_parts) == 1
            else jax.tree.map(
                lambda *values: jnp.concatenate(values, axis=0), *timeline_parts
            )
        )
        return FullStridedRolloutV2(
            final_carry=carry,
            transitions=transitions,
            timeline=timeline,
            bootstrap_value=_bootstrap_value_v2(
                carry,
                tables,
                params,
                include_opponent=include_opponent,
                include_history=include_history,
                task_card_program=task_card_program,
            ),
        )

    return collect
