"""Memory-bounded joint unit plus ordered-market PPO for Dynamic Full-core V2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from kaggriculture_jax.constants import EPISODE_STEPS
from kaggriculture_jax.simulator import batched_project_unit_phase
from kaggriculture_jax.types import StaticTables

from .dynamic_policy_v2 import (
    MARKET_STOP_INDEX_V2,
    combine_dynamic_actions_v2,
    select_dynamic_unit_phase_v2,
)
from .dynamic_rollout_v2 import (
    DynamicFullStepTraceV2,
    dynamic_full_step_v2,
    select_dynamic_full_core_pair_v2,
)
from .learned_v1 import recompute_full_logprob_v1
from .learned_v2 import MODEL_FULL_V2, apply_full_learned_model_v2
from .lifecycle import clear_invalidated_full_core_tasks_v1
from .ppo_v1 import FullPPOConfigV1, compute_timeline_gae_v1
from .rollout_v2 import FullLearnedCarryV2
from .learned_v1 import FullTimelineV1
from .task_cards import ReplayTaskCardProgramV1


POLICY_STEPS_V2 = EPISODE_STEPS - 1


class DynamicPPOObservationV2(NamedTuple):
    unit_global_features: jax.Array
    unit_candidate_features: jax.Array
    unit_candidate_task_type: jax.Array
    unit_task_masks: jax.Array
    unit_selected_indices: jax.Array
    market_candidate_features: jax.Array
    market_candidate_task_type: jax.Array
    market_masks: jax.Array
    market_selected_indices: jax.Array
    old_logprob: jax.Array
    old_value: jax.Array


class DynamicPPOTransitionV2(NamedTuple):
    unit_global_features: jax.Array
    unit_candidate_features: jax.Array
    unit_candidate_task_type: jax.Array
    unit_task_masks: jax.Array
    unit_selected_indices: jax.Array
    market_candidate_features: jax.Array
    market_candidate_task_type: jax.Array
    market_masks: jax.Array
    market_selected_indices: jax.Array
    old_logprob: jax.Array
    old_value: jax.Array
    reward: jax.Array
    done: jax.Array


class DynamicStridedRolloutV2(NamedTuple):
    final_carry: FullLearnedCarryV2
    transitions: DynamicPPOTransitionV2
    timeline: FullTimelineV1
    bootstrap_value: jax.Array


class DynamicPPOSamplesV2(NamedTuple):
    unit_global_features: jax.Array
    unit_candidate_features: jax.Array
    unit_candidate_task_type: jax.Array
    unit_task_masks: jax.Array
    unit_selected_indices: jax.Array
    market_candidate_features: jax.Array
    market_candidate_task_type: jax.Array
    market_masks: jax.Array
    market_selected_indices: jax.Array
    old_logprob: jax.Array
    old_value: jax.Array
    advantage: jax.Array
    returns: jax.Array


class DynamicPPOMetricsV2(NamedTuple):
    loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    unit_entropy: jax.Array
    market_entropy: jax.Array
    approx_kl: jax.Array
    clip_fraction: jax.Array
    grad_norm: jax.Array
    mean_ratio: jax.Array


@dataclass(frozen=True)
class DynamicPPOCollectorConfigV2:
    sample_stride: int = 32
    deterministic: bool = False
    include_opponent: bool = True
    include_history: bool = False
    task_card_program: ReplayTaskCardProgramV1 | None = None
    allow_nonpositive_econ: bool = False


def dynamic_sample_steps_v2(sample_stride: int) -> tuple[int, ...]:
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    steps = list(range(0, POLICY_STEPS_V2, sample_stride))
    if steps[-1] != POLICY_STEPS_V2 - 1:
        steps.append(POLICY_STEPS_V2 - 1)
    return tuple(steps)


def _stack_seats(left: jax.Array, right: jax.Array) -> jax.Array:
    return jnp.stack((left, right), axis=1)


def dynamic_ppo_snapshot_v2(
    carry: FullLearnedCarryV2,
    tables: StaticTables,
    params: object,
    *,
    deterministic: bool = False,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = False,
) -> DynamicPPOObservationV2:
    """Recreate the exact policy choices without mutating the environment.

    The following real environment step receives the same carry and keys, so
    its sampled joint action and old log-probability are identical.  Replaying
    only the sparsely sampled policy states avoids retaining 719 heavy feature
    tensors on the GPU.
    """

    states = carry.environment_state
    _, key0, key1 = jax.random.split(carry.rng, 3)
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
    projected = batched_project_unit_phase(
        states, combine_dynamic_actions_v2(unit0.action, unit1.action)
    )
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
    return DynamicPPOObservationV2(
        unit_global_features=_stack_seats(
            decision0.unit_global_features, decision1.unit_global_features
        ),
        unit_candidate_features=_stack_seats(
            decision0.unit_candidate_features, decision1.unit_candidate_features
        ),
        unit_candidate_task_type=_stack_seats(
            decision0.unit_candidate_task_type, decision1.unit_candidate_task_type
        ),
        unit_task_masks=_stack_seats(
            decision0.unit_selection.masks, decision1.unit_selection.masks
        ),
        unit_selected_indices=_stack_seats(
            decision0.unit_selection.selected_candidate_indices,
            decision1.unit_selection.selected_candidate_indices,
        ),
        market_candidate_features=_stack_seats(
            decision0.market_trace.candidate_features,
            decision1.market_trace.candidate_features,
        ),
        market_candidate_task_type=_stack_seats(
            decision0.market_trace.candidate_task_type,
            decision1.market_trace.candidate_task_type,
        ),
        market_masks=_stack_seats(
            decision0.market_trace.masks, decision1.market_trace.masks
        ),
        market_selected_indices=_stack_seats(
            decision0.market_trace.selected_indices,
            decision1.market_trace.selected_indices,
        ),
        old_logprob=_stack_seats(
            decision0.unit_selection.joint_logprob
            + decision0.market_trace.joint_logprob,
            decision1.unit_selection.joint_logprob
            + decision1.market_trace.joint_logprob,
        ),
        old_value=_stack_seats(decision0.value, decision1.value),
    )


def dynamic_ppo_sample_step_v2(
    carry: FullLearnedCarryV2,
    tables: StaticTables,
    params: object,
    *,
    deterministic: bool = False,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = False,
) -> tuple[FullLearnedCarryV2, DynamicPPOTransitionV2]:
    observation = dynamic_ppo_snapshot_v2(
        carry,
        tables,
        params,
        deterministic=deterministic,
        include_opponent=include_opponent,
        include_history=include_history,
        task_card_program=task_card_program,
        allow_nonpositive_econ=allow_nonpositive_econ,
    )
    following, actual = dynamic_full_step_v2(
        carry,
        tables,
        params,
        deterministic=deterministic,
        include_opponent=include_opponent,
        include_history=include_history,
        task_card_program=task_card_program,
        allow_nonpositive_econ=allow_nonpositive_econ,
    )
    transition = DynamicPPOTransitionV2(
        *observation[:-2],
        old_logprob=actual.joint_logprob,
        old_value=actual.value,
        reward=actual.reward,
        done=actual.done,
    )
    return following, transition


def _timeline_from_light(trace: DynamicFullStepTraceV2) -> FullTimelineV1:
    return FullTimelineV1(trace.value, trace.reward, trace.done)


def _timeline_from_heavy(trace: DynamicPPOTransitionV2) -> FullTimelineV1:
    return FullTimelineV1(trace.old_value, trace.reward, trace.done)


def make_dynamic_strided_collector_v2(config: DynamicPPOCollectorConfigV2):
    sample_steps = dynamic_sample_steps_v2(config.sample_stride)
    regular_steps = tuple(range(0, POLICY_STEPS_V2, config.sample_stride))
    prefix_groups = len(regular_steps) - 1
    tail_start = regular_steps[-1]
    tail_gap = (POLICY_STEPS_V2 - 1) - tail_start

    def light_step(carry, tables, params):
        return dynamic_full_step_v2(
            carry,
            tables,
            params,
            deterministic=config.deterministic,
            include_opponent=config.include_opponent,
            include_history=config.include_history,
            task_card_program=config.task_card_program,
            allow_nonpositive_econ=config.allow_nonpositive_econ,
        )

    def sample_step(carry, tables, params):
        return dynamic_ppo_sample_step_v2(
            carry,
            tables,
            params,
            deterministic=config.deterministic,
            include_opponent=config.include_opponent,
            include_history=config.include_history,
            task_card_program=config.task_card_program,
            allow_nonpositive_econ=config.allow_nonpositive_econ,
        )

    def collect(carry: FullLearnedCarryV2, tables: StaticTables, params: object):
        def prefix_body(current, _):
            following, heavy = sample_step(current, tables, params)

            def one_light(inner, __):
                next_inner, trace = light_step(inner, tables, params)
                return next_inner, _timeline_from_light(trace)

            following, light_timeline = jax.lax.scan(
                one_light,
                following,
                xs=None,
                length=config.sample_stride - 1,
            )
            group_timeline = jax.tree.map(
                lambda first, rest: jnp.concatenate((first[None], rest), axis=0),
                _timeline_from_heavy(heavy),
                light_timeline,
            )
            return following, (heavy, group_timeline)

        carry, (prefix_heavy, prefix_timeline) = jax.lax.scan(
            prefix_body, carry, xs=None, length=prefix_groups
        )
        tail_heavy0_carry, tail_heavy0 = sample_step(carry, tables, params)
        tail_heavy = [tail_heavy0]
        tail_timeline_parts = [
            jax.tree.map(lambda value: value[None], _timeline_from_heavy(tail_heavy0))
        ]
        carry = tail_heavy0_carry
        if tail_gap > 0:
            between = tail_gap - 1
            if between > 0:
                def tail_light_body(inner, _):
                    following, trace = light_step(inner, tables, params)
                    return following, _timeline_from_light(trace)

                carry, middle_timeline = jax.lax.scan(
                    tail_light_body, carry, xs=None, length=between
                )
                tail_timeline_parts.append(middle_timeline)
            carry, final_heavy = sample_step(carry, tables, params)
            tail_heavy.append(final_heavy)
            tail_timeline_parts.append(
                jax.tree.map(lambda value: value[None], _timeline_from_heavy(final_heavy))
            )
        transitions = jax.tree.map(
            lambda prefix, *tail: jnp.concatenate((prefix, *[value[None] for value in tail]), axis=0),
            prefix_heavy,
            *tail_heavy,
        )
        prefix_timeline = jax.tree.map(
            lambda value: value.reshape((-1,) + value.shape[2:]), prefix_timeline
        )
        timeline = jax.tree.map(
            lambda prefix, *tail: jnp.concatenate((prefix, *tail), axis=0),
            prefix_timeline,
            *tail_timeline_parts,
        )
        bootstrap = jnp.zeros_like(timeline.old_value[-1])
        return DynamicStridedRolloutV2(carry, transitions, timeline, bootstrap)

    collect.sample_steps = sample_steps
    return collect


def flatten_dynamic_ppo_samples_v2(
    transitions: DynamicPPOTransitionV2,
    advantage: jax.Array,
    returns: jax.Array,
) -> DynamicPPOSamplesV2:
    sample_count = int(
        transitions.old_logprob.shape[0]
        * transitions.old_logprob.shape[1]
        * transitions.old_logprob.shape[2]
    )

    def flatten(value):
        return value.reshape((sample_count,) + value.shape[3:])

    return DynamicPPOSamplesV2(
        unit_global_features=flatten(transitions.unit_global_features),
        unit_candidate_features=flatten(transitions.unit_candidate_features),
        unit_candidate_task_type=flatten(transitions.unit_candidate_task_type),
        unit_task_masks=flatten(transitions.unit_task_masks),
        unit_selected_indices=flatten(transitions.unit_selected_indices),
        market_candidate_features=flatten(transitions.market_candidate_features),
        market_candidate_task_type=flatten(transitions.market_candidate_task_type),
        market_masks=flatten(transitions.market_masks),
        market_selected_indices=flatten(transitions.market_selected_indices),
        old_logprob=flatten(transitions.old_logprob),
        old_value=flatten(transitions.old_value),
        advantage=advantage.reshape((sample_count,)),
        returns=returns.reshape((sample_count,)),
    )


def _masked_entropy(logits: jax.Array, masks: jax.Array) -> jax.Array:
    masked = jnp.where(masks, logits, jnp.finfo(jnp.float32).min)
    log_probability = jax.nn.log_softmax(masked, axis=-1)
    probability = jnp.exp(log_probability)
    return -jnp.sum(probability * log_probability, axis=-1)


def dynamic_ppo_loss_v2(
    params: object,
    samples: DynamicPPOSamplesV2,
    config: FullPPOConfigV1,
) -> tuple[jax.Array, DynamicPPOMetricsV2]:
    unit = apply_full_learned_model_v2(
        params,
        samples.unit_global_features,
        samples.unit_candidate_features,
        samples.unit_candidate_task_type,
    )
    unit_logprob = recompute_full_logprob_v1(
        unit.candidate_logits,
        unit.stop_logit,
        samples.unit_task_masks,
        samples.unit_selected_indices,
    )
    count, ordinals = samples.market_selected_indices.shape
    market_global = jnp.broadcast_to(
        samples.unit_global_features[:, None, :],
        (count, ordinals, samples.unit_global_features.shape[-1]),
    ).reshape((count * ordinals, -1))
    market = apply_full_learned_model_v2(
        params,
        market_global,
        samples.market_candidate_features.reshape(
            (count * ordinals,) + samples.market_candidate_features.shape[2:]
        ),
        samples.market_candidate_task_type.reshape(
            (count * ordinals,) + samples.market_candidate_task_type.shape[2:]
        ),
    )
    market_logits = jnp.concatenate(
        (market.candidate_logits, market.stop_logit[:, None]), axis=-1
    ).reshape((count, ordinals, -1))
    selected = jnp.where(
        samples.market_selected_indices >= 0,
        samples.market_selected_indices,
        MARKET_STOP_INDEX_V2,
    ).astype(jnp.int32)
    masked_market = jnp.where(
        samples.market_masks, market_logits, jnp.finfo(jnp.float32).min
    )
    market_logprob = jnp.sum(
        jnp.take_along_axis(
            jax.nn.log_softmax(masked_market, axis=-1), selected[..., None], axis=-1
        )[..., 0],
        axis=1,
    )
    new_logprob = unit_logprob + market_logprob
    advantage = (samples.advantage - jnp.mean(samples.advantage)) / (
        jnp.std(samples.advantage) + 1e-8
    )
    ratio = jnp.exp(jnp.clip(new_logprob - samples.old_logprob, -20.0, 20.0))
    unclipped = ratio * advantage
    clipped = jnp.clip(
        ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
    ) * advantage
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    value_loss = 0.5 * jnp.mean(jnp.square(unit.value - samples.returns))
    unit_logits = jnp.concatenate(
        (unit.candidate_logits, unit.stop_logit[:, None]), axis=-1
    )
    unit_entropy = jnp.mean(
        jnp.sum(
            _masked_entropy(
                jnp.broadcast_to(
                    unit_logits[:, None, :], samples.unit_task_masks.shape
                ),
                samples.unit_task_masks,
            ),
            axis=1,
        )
    )
    market_entropy = jnp.mean(
        jnp.sum(_masked_entropy(market_logits, samples.market_masks), axis=1)
    )
    entropy = unit_entropy + market_entropy
    loss = (
        policy_loss
        + config.value_coefficient * value_loss
        - config.entropy_coefficient * entropy
    )
    return loss, DynamicPPOMetricsV2(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        unit_entropy=unit_entropy,
        market_entropy=market_entropy,
        approx_kl=jnp.mean(samples.old_logprob - new_logprob),
        clip_fraction=jnp.mean(
            (jnp.abs(ratio - 1.0) > config.clip_epsilon).astype(jnp.float32)
        ),
        grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
        mean_ratio=jnp.mean(ratio),
    )


def initialize_dynamic_ppo_state_v2(params: object, config: FullPPOConfigV1):
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )
    return train_state.TrainState.create(
        apply_fn=MODEL_FULL_V2.apply, params=params, tx=optimizer
    )


def make_dynamic_strided_ppo_update_v2(
    config: FullPPOConfigV1,
    *,
    sample_count: int,
    sample_steps: tuple[int, ...],
):
    if sample_count % config.minibatch_size != 0:
        raise ValueError("sample_count must be divisible by minibatch_size")
    minibatches = sample_count // config.minibatch_size
    step_index = jnp.asarray(sample_steps, dtype=jnp.int32)

    def update(state, transitions, timeline, bootstrap_value, key):
        full_advantage, full_returns = compute_timeline_gae_v1(
            timeline,
            bootstrap_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        samples = flatten_dynamic_ppo_samples_v2(
            transitions,
            full_advantage[step_index],
            full_returns[step_index],
        )

        def epoch_body(carry, _):
            current_state, current_key = carry
            current_key, permutation_key = jax.random.split(current_key)
            order = jax.random.permutation(permutation_key, sample_count).reshape(
                (minibatches, config.minibatch_size)
            )

            def minibatch_body(one_state, indices):
                batch = jax.tree.map(lambda value: value[indices], samples)
                (loss, metrics), gradient = jax.value_and_grad(
                    dynamic_ppo_loss_v2, has_aux=True
                )(one_state.params, batch, config)
                del loss
                grad_norm = optax.global_norm(gradient)
                one_state = one_state.apply_gradients(grads=gradient)
                return one_state, metrics._replace(grad_norm=grad_norm)

            current_state, metrics = jax.lax.scan(
                minibatch_body, current_state, order
            )
            return (current_state, current_key), metrics

        (state, key), metrics = jax.lax.scan(
            epoch_body, (state, key), xs=None, length=config.epochs
        )
        return state, jax.tree.map(lambda value: jnp.mean(value), metrics), key

    return update
