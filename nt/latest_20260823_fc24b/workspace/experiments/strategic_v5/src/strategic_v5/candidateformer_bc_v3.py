"""Replay feature compilation and BC losses for CandidateFormer V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from .candidateformer_v3 import (
    apply_candidateformer_market_v3,
    build_candidate_features_v3,
    build_plan_ledger_features_v3,
    encode_candidateformer_v3,
)
from .constants import MAX_CANDIDATES_V1
from .dynamic_bc_v2 import actor_visible_arrays_to_state_v2
from .dynamic_policy_v2 import MARKET_STOP_INDEX_V2
from .e4_core import build_full_core_candidates_v1, evaluate_full_core_feasibility_v1
from .e5_econ import build_full_econ_features_v1
from .lifecycle import reset_controller_state_v1
from .opponent_v2 import build_candidate_features_v2, build_global_features_v2
from .replay_bc_v2 import broad_replay_bc_candidate_program_v2


class CandidateFormerReplayFeaturesV3(NamedTuple):
    global_features: jax.Array
    plan_features: jax.Array
    candidate_features: jax.Array
    candidate_task_type: jax.Array
    official_legal_mask: jax.Array


class CandidateFormerUnitBatchV3(NamedTuple):
    global_features: jax.Array
    plan_features: jax.Array
    candidate_features: jax.Array
    candidate_task_type: jax.Array
    positive_mask: jax.Array
    supervised_mask: jax.Array
    opponent_task_type: jax.Array
    sample_weight: jax.Array


class CandidateFormerMarketBatchV3(NamedTuple):
    global_features: jax.Array
    plan_features: jax.Array
    full_candidate_features: jax.Array
    full_candidate_task_type: jax.Array
    market_candidate_features: jax.Array
    market_candidate_task_type: jax.Array
    mask: jax.Array
    target: jax.Array
    sample_weight: jax.Array


class CandidateFormerCombinedBatchV3(NamedTuple):
    unit: CandidateFormerUnitBatchV3
    market: CandidateFormerMarketBatchV3


class CandidateFormerBCMetricsV3(NamedTuple):
    loss: jax.Array
    unit_loss: jax.Array
    unit_positive_loss: jax.Array
    unit_negative_loss: jax.Array
    unit_pairwise_accuracy: jax.Array
    market_loss: jax.Array
    market_accuracy: jax.Array
    opponent_loss: jax.Array
    opponent_accuracy: jax.Array
    illegal_target_count: jax.Array
    grad_norm: jax.Array


@dataclass(frozen=True)
class CandidateFormerBCConfigV3:
    learning_rate: float = 1.0e-4
    unit_weight: float = 1.0
    market_weight: float = 1.0
    opponent_weight: float = 0.05
    max_grad_norm: float = 1.0


def build_candidateformer_replay_features_v3(
    arrays, tables
) -> CandidateFormerReplayFeaturesV3:
    """Build actor-only relation inputs; expert targets are not accepted here."""

    states = actor_visible_arrays_to_state_v2(arrays)
    batch_size = states.step.shape[0]
    one = reset_controller_state_v1()
    controller = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape), one
    )
    candidates = build_full_core_candidates_v1(
        states,
        controller,
        tables,
        0,
        broad_replay_bc_candidate_program_v2(),
    )
    candidates = candidates._replace(
        replay_priority=jnp.zeros_like(candidates.replay_priority)
    )
    feasibility = evaluate_full_core_feasibility_v1(
        states, candidates, tables, 0
    )
    econ = build_full_econ_features_v1(
        states, candidates, feasibility, tables, 0
    )
    global_features = build_global_features_v2(
        states, 0, None, include_opponent=True, include_history=False
    )
    base = build_candidate_features_v2(
        states,
        candidates,
        feasibility,
        econ,
        0,
        tables,
        include_opponent=True,
    )
    return CandidateFormerReplayFeaturesV3(
        global_features=global_features,
        plan_features=build_plan_ledger_features_v3(
            states, controller, candidates, feasibility, econ, 0
        ),
        candidate_features=build_candidate_features_v3(
            states, candidates, feasibility, econ, base
        ),
        candidate_task_type=candidates.task_type,
        official_legal_mask=candidates.present & feasibility.legal_now,
    )


def _masked_mean(value: jax.Array, weight: jax.Array) -> jax.Array:
    weight = weight.astype(jnp.float32)
    return jnp.sum(value * weight) / jnp.maximum(jnp.sum(weight), 1.0)


def candidateformer_bc_loss_v3(
    params: object,
    batch: CandidateFormerCombinedBatchV3,
    config: CandidateFormerBCConfigV3,
) -> tuple[jax.Array, CandidateFormerBCMetricsV3]:
    unit_context, unit_output = encode_candidateformer_v3(
        params,
        batch.unit.global_features,
        batch.unit.plan_features,
        batch.unit.candidate_features,
        batch.unit.candidate_task_type,
    )
    del unit_context
    positive = batch.unit.positive_mask & batch.unit.supervised_mask
    negative = (~batch.unit.positive_mask) & batch.unit.supervised_mask
    sample = batch.unit.sample_weight.astype(jnp.float32)
    positive_per = jnp.sum(
        jax.nn.softplus(-unit_output.candidate_logits) * positive, axis=-1
    ) / jnp.maximum(jnp.sum(positive, axis=-1), 1.0)
    negative_per = jnp.sum(
        jax.nn.softplus(unit_output.candidate_logits) * negative, axis=-1
    ) / jnp.maximum(jnp.sum(negative, axis=-1), 1.0)
    positive_loss = _masked_mean(positive_per, sample)
    negative_loss = _masked_mean(negative_per, sample)
    unit_loss = 0.5 * (positive_loss + negative_loss)
    pairs = positive[:, :, None] & negative[:, None, :]
    pairwise_accuracy = jnp.sum(
        (
            unit_output.candidate_logits[:, :, None]
            > unit_output.candidate_logits[:, None, :]
        )
        & pairs,
        dtype=jnp.float32,
    ) / jnp.maximum(jnp.sum(pairs), 1.0)

    opponent_valid = batch.unit.opponent_task_type >= 0
    opponent_target = jnp.clip(
        batch.unit.opponent_task_type.astype(jnp.int32),
        0,
        unit_output.opponent_task_logits.shape[-1] - 1,
    )
    opponent_ce = optax.softmax_cross_entropy_with_integer_labels(
        unit_output.opponent_task_logits, opponent_target
    )
    opponent_weight = opponent_valid.astype(jnp.float32) * sample
    opponent_loss = _masked_mean(opponent_ce, opponent_weight)
    opponent_accuracy = _masked_mean(
        (
            jnp.argmax(unit_output.opponent_task_logits, axis=-1)
            == opponent_target
        ).astype(jnp.float32),
        opponent_weight,
    )

    market_context, _ = encode_candidateformer_v3(
        params,
        batch.market.global_features,
        batch.market.plan_features,
        batch.market.full_candidate_features,
        batch.market.full_candidate_task_type,
    )
    market_output = apply_candidateformer_market_v3(
        params,
        market_context,
        batch.market.market_candidate_features,
        batch.market.market_candidate_task_type,
    )
    extended = jnp.concatenate(
        (market_output.candidate_logits, market_output.stop_logit[:, None]), axis=-1
    )
    masked = jnp.where(
        batch.market.mask, extended, jnp.finfo(jnp.float32).min
    )
    target = jnp.clip(
        batch.market.target.astype(jnp.int32), 0, MARKET_STOP_INDEX_V2
    )
    allowed = jnp.take_along_axis(batch.market.mask, target[:, None], axis=-1)[:, 0]
    selected = jnp.take_along_axis(
        jax.nn.log_softmax(masked, axis=-1), target[:, None], axis=-1
    )[:, 0]
    market_weight = batch.market.sample_weight.astype(jnp.float32) * allowed
    market_loss = _masked_mean(-selected, market_weight)
    market_accuracy = _masked_mean(
        (jnp.argmax(masked, axis=-1) == target).astype(jnp.float32), market_weight
    )
    illegal = jnp.sum(
        (~allowed) & (batch.market.sample_weight > 0), dtype=jnp.float32
    )

    loss = (
        config.unit_weight * unit_loss
        + config.market_weight * market_loss
        + config.opponent_weight * opponent_loss
    )
    return loss, CandidateFormerBCMetricsV3(
        loss=loss,
        unit_loss=unit_loss,
        unit_positive_loss=positive_loss,
        unit_negative_loss=negative_loss,
        unit_pairwise_accuracy=pairwise_accuracy,
        market_loss=market_loss,
        market_accuracy=market_accuracy,
        opponent_loss=opponent_loss,
        opponent_accuracy=opponent_accuracy,
        illegal_target_count=illegal,
        grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
    )


def initialize_candidateformer_bc_state_v3(
    params: object, config: CandidateFormerBCConfigV3
):
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(config.learning_rate, weight_decay=1.0e-5),
    )
    return train_state.TrainState.create(
        apply_fn=lambda *_args, **_kwargs: None,
        params=params,
        tx=optimizer,
    )


def make_candidateformer_accumulated_update_v3(
    config: CandidateFormerBCConfigV3,
):
    """One Adam update from a leading axis of physical microbatches."""

    def update(state, microbatches: CandidateFormerCombinedBatchV3):
        count = microbatches.unit.global_features.shape[0]
        zero = jax.tree.map(jnp.zeros_like, state.params)
        metric_zero = CandidateFormerBCMetricsV3(
            *(jnp.asarray(0.0, dtype=jnp.float32) for _ in range(11))
        )

        def one(carry, batch):
            gradient_sum, metric_sum = carry
            (_, metrics), gradient = jax.value_and_grad(
                candidateformer_bc_loss_v3, has_aux=True
            )(state.params, batch, config)
            gradient_sum = jax.tree.map(
                lambda total, value: total + value, gradient_sum, gradient
            )
            metric_sum = jax.tree.map(lambda a, b: a + b, metric_sum, metrics)
            return (gradient_sum, metric_sum), None

        (gradient, metrics), _ = jax.lax.scan(one, (zero, metric_zero), microbatches)
        gradient = jax.tree.map(lambda value: value / float(count), gradient)
        # The inherited 50K route is a frozen safety/residual baseline.
        gradient = dict(gradient)
        gradient["baseline"] = jax.tree.map(jnp.zeros_like, gradient["baseline"])
        grad_norm = optax.global_norm(gradient["transformer"])
        state = state.apply_gradients(grads=gradient)
        metrics = jax.tree.map(lambda value: value / float(count), metrics)
        return state, metrics._replace(grad_norm=grad_norm)

    return update
