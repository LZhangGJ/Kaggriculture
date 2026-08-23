"""Teacher-forced task-card BC for the opponent-aware V2 policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from .constants import MAX_CANDIDATES_V1
from .dynamic_policy_v2 import (
    MARKET_CANDIDATE_COUNT_V2,
    MARKET_STOP_INDEX_V2,
    DynamicMarketTraceV2,
)
from .learned_v2 import MODEL_FULL_V2, apply_full_learned_model_v2
from .replay_bc_v2 import ORDERED_STOP_SLOT_V3


class FullBCBatchV2(NamedTuple):
    global_features: jax.Array
    candidate_features: jax.Array
    candidate_task_type: jax.Array
    task_masks: jax.Array
    task_selected_indices: jax.Array
    selection_weight: jax.Array
    opponent_task_type: jax.Array
    sample_weight: jax.Array


class FullBCMetricsV2(NamedTuple):
    loss: jax.Array
    task_loss: jax.Array
    opponent_loss: jax.Array
    task_accuracy: jax.Array
    opponent_accuracy: jax.Array
    illegal_target_count: jax.Array
    grad_norm: jax.Array


class FullBCListwiseBatchV2(NamedTuple):
    """Multi-label strategic-card supervision from replay actions.

    This batch intentionally excludes low-level movement/maintenance actions.
    They remain Full-core responsibilities instead of being mislabeled STOP.
    """

    global_features: jax.Array
    candidate_features: jax.Array
    candidate_task_type: jax.Array
    positive_candidate_mask: jax.Array
    supervised_candidate_mask: jax.Array
    opponent_task_type: jax.Array
    sample_weight: jax.Array


class FullBCListwiseMetricsV2(NamedTuple):
    loss: jax.Array
    positive_loss: jax.Array
    negative_loss: jax.Array
    opponent_loss: jax.Array
    pairwise_accuracy: jax.Array
    opponent_accuracy: jax.Array
    empty_positive_count: jax.Array
    grad_norm: jax.Array


class DynamicMarketBCBatchV2(NamedTuple):
    """Leak-free teacher-forced ordered market supervision.

    The feature/mask prefix must be generated before applying the current
    expert label.  ``target_quantity`` is retained for the parameterized
    executor/ledger; the policy classifies only the shared 21 task identities
    plus STOP, so a different quantity never creates a new expert class.
    """

    global_features: jax.Array
    candidate_features: jax.Array
    candidate_task_type: jax.Array
    masks: jax.Array
    target_candidate_slot: jax.Array
    target_quantity: jax.Array
    selection_weight: jax.Array
    sample_weight: jax.Array


class DynamicMarketBCMetricsV2(NamedTuple):
    loss: jax.Array
    accuracy: jax.Array
    illegal_target_count: jax.Array
    supervised_substeps: jax.Array
    positive_quantity_substeps: jax.Array
    grad_norm: jax.Array


class AnchoredDynamicMarketBCBatchV2(NamedTuple):
    market: DynamicMarketBCBatchV2
    unit_global_features: jax.Array
    unit_candidate_features: jax.Array
    unit_candidate_task_type: jax.Array
    unit_anchor_mask: jax.Array
    anchor_candidate_logits: jax.Array
    anchor_stop_logit: jax.Array
    anchor_value: jax.Array


class AnchoredDynamicMarketBCMetricsV2(NamedTuple):
    loss: jax.Array
    market_loss: jax.Array
    market_accuracy: jax.Array
    unit_anchor_loss: jax.Array
    stop_anchor_loss: jax.Array
    value_anchor_loss: jax.Array
    illegal_target_count: jax.Array
    supervised_substeps: jax.Array
    grad_norm: jax.Array


@dataclass(frozen=True)
class FullBCConfigV2:
    learning_rate: float = 3e-4
    opponent_aux_weight: float = 0.1
    max_grad_norm: float = 1.0


def initialize_full_bc_state_v2(params: object, config: FullBCConfigV2):
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )
    return train_state.TrainState.create(
        apply_fn=MODEL_FULL_V2.apply, params=params, tx=optimizer
    )


def full_bc_loss_v2(
    params: object,
    batch: FullBCBatchV2,
    config: FullBCConfigV2,
) -> tuple[jax.Array, FullBCMetricsV2]:
    output = MODEL_FULL_V2.apply(
        {"params": params},
        batch.global_features,
        batch.candidate_features,
        batch.candidate_task_type,
    )
    extended = jnp.concatenate(
        (output.candidate_logits, output.stop_logit[:, None]), axis=-1
    )
    targets = jnp.where(
        batch.task_selected_indices >= 0,
        batch.task_selected_indices,
        MAX_CANDIDATES_V1,
    ).astype(jnp.int32)
    masked = jnp.where(
        batch.task_masks,
        extended[:, None, :],
        jnp.finfo(jnp.float32).min,
    )
    log_probability = jax.nn.log_softmax(masked, axis=-1)
    selected_log_probability = jnp.take_along_axis(
        log_probability, targets[..., None], axis=-1
    )[..., 0]
    allowed = jnp.take_along_axis(
        batch.task_masks, targets[..., None], axis=-1
    )[..., 0]
    step_weight = (
        batch.selection_weight.astype(jnp.float32)
        * batch.sample_weight.astype(jnp.float32)[:, None]
    )
    denominator = jnp.maximum(jnp.sum(step_weight), 1.0)
    task_loss = -jnp.sum(selected_log_probability * step_weight) / denominator
    predicted = jnp.argmax(masked, axis=-1)
    task_accuracy = jnp.sum((predicted == targets) * step_weight) / denominator
    illegal_target_count = jnp.sum(
        (~allowed) & (batch.selection_weight > 0), dtype=jnp.float32
    )

    opponent_valid = batch.opponent_task_type >= 0
    opponent_target = jnp.clip(batch.opponent_task_type, 0, output.opponent_task_logits.shape[-1] - 1)
    opponent_ce = optax.softmax_cross_entropy_with_integer_labels(
        output.opponent_task_logits, opponent_target
    )
    opponent_weight = (
        opponent_valid.astype(jnp.float32) * batch.sample_weight.astype(jnp.float32)
    )
    opponent_denominator = jnp.maximum(jnp.sum(opponent_weight), 1.0)
    opponent_loss = jnp.sum(opponent_ce * opponent_weight) / opponent_denominator
    opponent_accuracy = jnp.sum(
        (
            jnp.argmax(output.opponent_task_logits, axis=-1) == opponent_target
        ).astype(jnp.float32)
        * opponent_weight
    ) / opponent_denominator
    loss = task_loss + config.opponent_aux_weight * opponent_loss
    return loss, FullBCMetricsV2(
        loss=loss,
        task_loss=task_loss,
        opponent_loss=opponent_loss,
        task_accuracy=task_accuracy,
        opponent_accuracy=opponent_accuracy,
        illegal_target_count=illegal_target_count,
        grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
    )


def make_full_bc_update_v2(config: FullBCConfigV2):
    def update(state, batch: FullBCBatchV2):
        (loss, metrics), gradient = jax.value_and_grad(
            full_bc_loss_v2, has_aux=True
        )(state.params, batch, config)
        del loss
        grad_norm = optax.global_norm(gradient)
        state = state.apply_gradients(grads=gradient)
        return state, metrics._replace(grad_norm=grad_norm)

    return update


def full_bc_listwise_loss_v2(
    params: object,
    batch: FullBCListwiseBatchV2,
    config: FullBCConfigV2,
) -> tuple[jax.Array, FullBCListwiseMetricsV2]:
    output = MODEL_FULL_V2.apply(
        {"params": params},
        batch.global_features,
        batch.candidate_features,
        batch.candidate_task_type,
    )
    positive = batch.positive_candidate_mask & batch.supervised_candidate_mask
    negative = (~batch.positive_candidate_mask) & batch.supervised_candidate_mask
    positive_weight = positive.astype(jnp.float32)
    negative_weight = negative.astype(jnp.float32)
    sample_weight = batch.sample_weight.astype(jnp.float32)
    positive_denominator = jnp.maximum(
        jnp.sum(positive_weight, axis=-1), 1.0
    )
    negative_denominator = jnp.maximum(
        jnp.sum(negative_weight, axis=-1), 1.0
    )
    positive_per_sample = jnp.sum(
        jax.nn.softplus(-output.candidate_logits) * positive_weight, axis=-1
    ) / positive_denominator
    negative_per_sample = jnp.sum(
        jax.nn.softplus(output.candidate_logits) * negative_weight, axis=-1
    ) / negative_denominator
    denominator = jnp.maximum(jnp.sum(sample_weight), 1.0)
    positive_loss = jnp.sum(positive_per_sample * sample_weight) / denominator
    negative_loss = jnp.sum(negative_per_sample * sample_weight) / denominator

    pair_valid = positive[:, :, None] & negative[:, None, :]
    pair_correct = (
        output.candidate_logits[:, :, None]
        > output.candidate_logits[:, None, :]
    ) & pair_valid
    pair_denominator = jnp.maximum(jnp.sum(pair_valid), 1.0)
    pairwise_accuracy = jnp.sum(pair_correct, dtype=jnp.float32) / pair_denominator

    opponent_valid = batch.opponent_task_type >= 0
    opponent_target = jnp.clip(
        batch.opponent_task_type, 0, output.opponent_task_logits.shape[-1] - 1
    )
    opponent_ce = optax.softmax_cross_entropy_with_integer_labels(
        output.opponent_task_logits, opponent_target
    )
    opponent_weight = opponent_valid.astype(jnp.float32) * sample_weight
    opponent_denominator = jnp.maximum(jnp.sum(opponent_weight), 1.0)
    opponent_loss = jnp.sum(opponent_ce * opponent_weight) / opponent_denominator
    opponent_accuracy = jnp.sum(
        (
            jnp.argmax(output.opponent_task_logits, axis=-1) == opponent_target
        ).astype(jnp.float32)
        * opponent_weight
    ) / opponent_denominator
    loss = (
        0.5 * positive_loss
        + 0.5 * negative_loss
        + config.opponent_aux_weight * opponent_loss
    )
    return loss, FullBCListwiseMetricsV2(
        loss=loss,
        positive_loss=positive_loss,
        negative_loss=negative_loss,
        opponent_loss=opponent_loss,
        pairwise_accuracy=pairwise_accuracy,
        opponent_accuracy=opponent_accuracy,
        empty_positive_count=jnp.sum(
            jnp.sum(positive_weight, axis=-1) == 0, dtype=jnp.float32
        ),
        grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
    )


def make_full_bc_listwise_update_v2(config: FullBCConfigV2):
    def update(state, batch: FullBCListwiseBatchV2):
        (loss, metrics), gradient = jax.value_and_grad(
            full_bc_listwise_loss_v2, has_aux=True
        )(state.params, batch, config)
        del loss
        grad_norm = optax.global_norm(gradient)
        state = state.apply_gradients(grads=gradient)
        return state, metrics._replace(grad_norm=grad_norm)

    return update


def dynamic_market_bc_batch_from_trace_v2(
    trace: DynamicMarketTraceV2,
    target_candidate_slot: jax.Array,
    target_quantity: jax.Array,
    target_valid: jax.Array,
    *,
    sample_weight: jax.Array | None = None,
) -> DynamicMarketBCBatchV2:
    """Bind ordered Replay V3 labels to a teacher-forced dynamic trace.

    Replay's frozen STOP id is 96 for compatibility with the 96-card schema;
    the dynamic market head uses local STOP id 21.  Both are normalized here.
    Invalid/padded labels receive zero selection weight.  Expert orders that
    the official engine executes as no-ops should likewise be marked invalid
    by the teacher-forced audit rather than trained as impossible actions.
    """

    target = jnp.asarray(target_candidate_slot, dtype=jnp.int32)
    normalized = jnp.where(
        (target == ORDERED_STOP_SLOT_V3) | (target < 0),
        MARKET_STOP_INDEX_V2,
        target,
    ).astype(jnp.int16)
    batch_size = normalized.shape[0]
    weights = jnp.asarray(target_valid, dtype=jnp.float32)
    samples = (
        jnp.ones((batch_size,), dtype=jnp.float32)
        if sample_weight is None
        else jnp.asarray(sample_weight, dtype=jnp.float32)
    )
    return DynamicMarketBCBatchV2(
        global_features=trace.global_features,
        candidate_features=trace.candidate_features,
        candidate_task_type=trace.candidate_task_type,
        masks=trace.masks,
        target_candidate_slot=normalized,
        target_quantity=jnp.asarray(target_quantity, dtype=jnp.int16),
        selection_weight=weights,
        sample_weight=samples,
    )


def dynamic_market_bc_loss_v2(
    params: object,
    batch: DynamicMarketBCBatchV2,
) -> tuple[jax.Array, DynamicMarketBCMetricsV2]:
    """Cross entropy over an ordered 21-market-card + STOP sequence."""

    batch_size, substeps = batch.target_candidate_slot.shape
    global_features = batch.global_features.reshape(
        (batch_size * substeps, batch.global_features.shape[-1])
    )
    candidate_features = batch.candidate_features.reshape(
        (
            batch_size * substeps,
            MARKET_CANDIDATE_COUNT_V2,
            batch.candidate_features.shape[-1],
        )
    )
    task_type = batch.candidate_task_type.reshape(
        (batch_size * substeps, MARKET_CANDIDATE_COUNT_V2)
    )
    output = apply_full_learned_model_v2(
        params, global_features, candidate_features, task_type
    )
    extended = jnp.concatenate(
        (output.candidate_logits, output.stop_logit[:, None]), axis=-1
    ).reshape((batch_size, substeps, MARKET_CANDIDATE_COUNT_V2 + 1))
    masked = jnp.where(
        batch.masks, extended, jnp.finfo(jnp.float32).min
    )
    target = jnp.clip(
        batch.target_candidate_slot.astype(jnp.int32),
        0,
        MARKET_STOP_INDEX_V2,
    )
    allowed = jnp.take_along_axis(
        batch.masks, target[..., None], axis=-1
    )[..., 0]
    selected_logprob = jnp.take_along_axis(
        jax.nn.log_softmax(masked, axis=-1), target[..., None], axis=-1
    )[..., 0]
    weight = (
        batch.selection_weight.astype(jnp.float32)
        * batch.sample_weight.astype(jnp.float32)[:, None]
        * allowed.astype(jnp.float32)
    )
    denominator = jnp.maximum(jnp.sum(weight), 1.0)
    loss = -jnp.sum(selected_logprob * weight) / denominator
    predicted = jnp.argmax(masked, axis=-1)
    accuracy = jnp.sum((predicted == target).astype(jnp.float32) * weight) / denominator
    illegal = jnp.sum(
        (~allowed) & (batch.selection_weight > 0), dtype=jnp.float32
    )
    return loss, DynamicMarketBCMetricsV2(
        loss=loss,
        accuracy=accuracy,
        illegal_target_count=illegal,
        supervised_substeps=jnp.sum(weight > 0, dtype=jnp.float32),
        positive_quantity_substeps=jnp.sum(
            (batch.target_quantity > 0) & (weight > 0), dtype=jnp.float32
        ),
        grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
    )


def make_dynamic_market_bc_update_v2():
    """Use the same optimizer state as the existing V2 BC pipeline."""

    def update(state, batch: DynamicMarketBCBatchV2):
        (loss, metrics), gradient = jax.value_and_grad(
            dynamic_market_bc_loss_v2, has_aux=True
        )(state.params, batch)
        del loss
        norm = optax.global_norm(gradient)
        return state.apply_gradients(grads=gradient), metrics._replace(
            grad_norm=norm
        )

    return update


def anchored_dynamic_market_bc_loss_v2(
    params: object,
    batch: AnchoredDynamicMarketBCBatchV2,
    *,
    unit_anchor_weight: float = 1.0,
    stop_anchor_weight: float = 1.0,
    value_anchor_weight: float = 1.0,
) -> tuple[jax.Array, AnchoredDynamicMarketBCMetricsV2]:
    market_loss, market_metrics = dynamic_market_bc_loss_v2(params, batch.market)
    output = apply_full_learned_model_v2(
        params,
        batch.unit_global_features,
        batch.unit_candidate_features,
        batch.unit_candidate_task_type,
    )
    anchor_mask = batch.unit_anchor_mask.astype(jnp.float32)
    anchor_denominator = jnp.maximum(jnp.sum(anchor_mask), 1.0)
    unit_anchor_loss = jnp.sum(
        jnp.square(output.candidate_logits - batch.anchor_candidate_logits)
        * anchor_mask
    ) / anchor_denominator
    stop_anchor_loss = jnp.mean(jnp.square(output.stop_logit - batch.anchor_stop_logit))
    value_anchor_loss = jnp.mean(jnp.square(output.value - batch.anchor_value))
    loss = (
        market_loss
        + unit_anchor_weight * unit_anchor_loss
        + stop_anchor_weight * stop_anchor_loss
        + value_anchor_weight * value_anchor_loss
    )
    return loss, AnchoredDynamicMarketBCMetricsV2(
        loss=loss,
        market_loss=market_loss,
        market_accuracy=market_metrics.accuracy,
        unit_anchor_loss=unit_anchor_loss,
        stop_anchor_loss=stop_anchor_loss,
        value_anchor_loss=value_anchor_loss,
        illegal_target_count=market_metrics.illegal_target_count,
        supervised_substeps=market_metrics.supervised_substeps,
        grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
    )


def make_anchored_dynamic_market_bc_update_v2(
    *,
    unit_anchor_weight: float = 1.0,
    stop_anchor_weight: float = 1.0,
    value_anchor_weight: float = 1.0,
):
    def update(state, batch: AnchoredDynamicMarketBCBatchV2):
        (loss, metrics), gradient = jax.value_and_grad(
            anchored_dynamic_market_bc_loss_v2, has_aux=True
        )(
            state.params,
            batch,
            unit_anchor_weight=unit_anchor_weight,
            stop_anchor_weight=stop_anchor_weight,
            value_anchor_weight=value_anchor_weight,
        )
        del loss
        norm = optax.global_norm(gradient)
        return state.apply_gradients(grads=gradient), metrics._replace(grad_norm=norm)

    return update
