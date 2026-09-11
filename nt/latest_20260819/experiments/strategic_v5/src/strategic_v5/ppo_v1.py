"""PPO update for the compact V5 Full-core candidate policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from .learned_v1 import (
    MAX_CANDIDATES_V1,
    MODEL_FULL_V1,
    FullLearnedTransitionV1,
    FullTimelineV1,
    recompute_full_logprob_v1,
)


class FullPPOMetricsV1(NamedTuple):
    loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    clip_fraction: jax.Array
    grad_norm: jax.Array


class FullPPOSamplesV1(NamedTuple):
    global_features: jax.Array
    candidate_features: jax.Array
    candidate_task_type: jax.Array
    task_masks: jax.Array
    task_selected_indices: jax.Array
    old_logprob: jax.Array
    old_value: jax.Array
    advantage: jax.Array
    returns: jax.Array


@dataclass(frozen=True)
class FullPPOConfigV1:
    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 1.0
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 1.0
    epochs: int = 1
    minibatch_size: int = 4096


def initialize_full_ppo_state_v1(params: object, config: FullPPOConfigV1):
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )
    return train_state.TrainState.create(
        apply_fn=MODEL_FULL_V1.apply, params=params, tx=optimizer
    )


def compute_full_gae_v1(
    transitions: FullLearnedTransitionV1,
    bootstrap_value: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    next_values = jnp.concatenate(
        (transitions.old_value[1:], bootstrap_value[None]), axis=0
    )
    not_done = 1.0 - transitions.done.astype(jnp.float32)
    delta = transitions.reward + gamma * not_done * next_values - transitions.old_value

    def body(carry, values):
        one_delta, one_not_done = values
        advantage = one_delta + gamma * gae_lambda * one_not_done * carry
        return advantage, advantage

    _, advantage = jax.lax.scan(
        body,
        jnp.zeros_like(bootstrap_value),
        (delta, not_done),
        reverse=True,
    )
    return advantage.astype(jnp.float32), (advantage + transitions.old_value).astype(
        jnp.float32
    )


def compute_timeline_gae_v1(
    timeline: FullTimelineV1,
    bootstrap_value: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    next_values = jnp.concatenate(
        (timeline.old_value[1:], bootstrap_value[None]), axis=0
    )
    not_done = 1.0 - timeline.done.astype(jnp.float32)
    delta = timeline.reward + gamma * not_done * next_values - timeline.old_value

    def body(carry, values):
        one_delta, one_not_done = values
        advantage = one_delta + gamma * gae_lambda * one_not_done * carry
        return advantage, advantage

    _, advantage = jax.lax.scan(
        body,
        jnp.zeros_like(bootstrap_value),
        (delta, not_done),
        reverse=True,
    )
    return advantage.astype(jnp.float32), (advantage + timeline.old_value).astype(
        jnp.float32
    )


def flatten_full_ppo_samples_v1(
    transitions: FullLearnedTransitionV1,
    advantage: jax.Array,
    returns: jax.Array,
) -> FullPPOSamplesV1:
    sample_count = int(
        transitions.old_logprob.shape[0]
        * transitions.old_logprob.shape[1]
        * transitions.old_logprob.shape[2]
    )
    return FullPPOSamplesV1(
        global_features=transitions.global_features.reshape((sample_count, -1)),
        candidate_features=transitions.candidate_features.reshape(
            (sample_count,) + transitions.candidate_features.shape[3:]
        ),
        candidate_task_type=transitions.candidate_task_type.reshape(
            (sample_count,) + transitions.candidate_task_type.shape[3:]
        ),
        task_masks=transitions.task_masks.reshape(
            (sample_count,) + transitions.task_masks.shape[3:]
        ),
        task_selected_indices=transitions.task_selected_indices.reshape(
            (sample_count,) + transitions.task_selected_indices.shape[3:]
        ),
        old_logprob=transitions.old_logprob.reshape((sample_count,)),
        old_value=transitions.old_value.reshape((sample_count,)),
        advantage=advantage.reshape((sample_count,)),
        returns=returns.reshape((sample_count,)),
    )


def _full_entropy_v1(logits: jax.Array, masks: jax.Array) -> jax.Array:
    def one(mask):
        masked = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
        log_probability = jax.nn.log_softmax(masked, axis=-1)
        probability = jnp.exp(log_probability)
        return -jnp.sum(probability * log_probability, axis=-1)

    return jnp.sum(jax.vmap(one, in_axes=1, out_axes=1)(masks), axis=-1)


def full_ppo_loss_v1(
    params: object, samples: FullPPOSamplesV1, config: FullPPOConfigV1
) -> tuple[jax.Array, FullPPOMetricsV1]:
    output = MODEL_FULL_V1.apply(
        {"params": params},
        samples.global_features,
        samples.candidate_features,
        samples.candidate_task_type,
    )
    new_logprob = recompute_full_logprob_v1(
        output.candidate_logits,
        output.stop_logit,
        samples.task_masks,
        samples.task_selected_indices,
    )
    advantage = (samples.advantage - jnp.mean(samples.advantage)) / (
        jnp.std(samples.advantage) + 1e-8
    )
    ratio = jnp.exp(new_logprob - samples.old_logprob)
    unclipped = ratio * advantage
    clipped = jnp.clip(
        ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
    ) * advantage
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    value_loss = 0.5 * jnp.mean(jnp.square(output.value - samples.returns))
    extended_logits = jnp.concatenate(
        (output.candidate_logits, output.stop_logit[:, None]), axis=-1
    )
    entropy = jnp.mean(_full_entropy_v1(extended_logits, samples.task_masks))
    loss = (
        policy_loss
        + config.value_coefficient * value_loss
        - config.entropy_coefficient * entropy
    )
    return loss, FullPPOMetricsV1(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        approx_kl=jnp.mean(samples.old_logprob - new_logprob),
        clip_fraction=jnp.mean(
            (jnp.abs(ratio - 1.0) > config.clip_epsilon).astype(jnp.float32)
        ),
        grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
    )


def make_full_ppo_update_v1(config: FullPPOConfigV1, *, sample_count: int):
    if sample_count % config.minibatch_size != 0:
        raise ValueError("sample_count must be divisible by minibatch_size")
    minibatches = sample_count // config.minibatch_size

    def update(state, transitions, bootstrap_value, key):
        advantage, returns = compute_full_gae_v1(
            transitions,
            bootstrap_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        samples = flatten_full_ppo_samples_v1(transitions, advantage, returns)

        def epoch_body(carry, _):
            current_state, current_key = carry
            current_key, permutation_key = jax.random.split(current_key)
            order = jax.random.permutation(permutation_key, sample_count)
            shuffled = jax.tree.map(lambda value: value[order], samples)
            batches = jax.tree.map(
                lambda value: value.reshape(
                    (minibatches, config.minibatch_size) + value.shape[1:]
                ),
                shuffled,
            )

            def minibatch_body(one_state, batch):
                (loss, metrics), gradient = jax.value_and_grad(
                    full_ppo_loss_v1, has_aux=True
                )(one_state.params, batch, config)
                del loss
                grad_norm = optax.global_norm(gradient)
                one_state = one_state.apply_gradients(grads=gradient)
                return one_state, metrics._replace(grad_norm=grad_norm)

            current_state, metrics = jax.lax.scan(
                minibatch_body, current_state, batches
            )
            return (current_state, current_key), metrics

        (state, key), metrics = jax.lax.scan(
            epoch_body, (state, key), xs=None, length=config.epochs
        )
        return state, jax.tree.map(lambda value: jnp.mean(value), metrics), key

    return update


def make_strided_full_ppo_update_v1(
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
        samples = flatten_full_ppo_samples_v1(
            transitions,
            full_advantage[step_index],
            full_returns[step_index],
        )

        def epoch_body(carry, _):
            current_state, current_key = carry
            current_key, permutation_key = jax.random.split(current_key)
            order = jax.random.permutation(permutation_key, sample_count)
            shuffled = jax.tree.map(lambda value: value[order], samples)
            batches = jax.tree.map(
                lambda value: value.reshape(
                    (minibatches, config.minibatch_size) + value.shape[1:]
                ),
                shuffled,
            )

            def minibatch_body(one_state, batch):
                (loss, metrics), gradient = jax.value_and_grad(
                    full_ppo_loss_v1, has_aux=True
                )(one_state.params, batch, config)
                del loss
                grad_norm = optax.global_norm(gradient)
                one_state = one_state.apply_gradients(grads=gradient)
                return one_state, metrics._replace(grad_norm=grad_norm)

            current_state, metrics = jax.lax.scan(
                minibatch_body, current_state, batches
            )
            return (current_state, current_key), metrics

        (state, key), metrics = jax.lax.scan(
            epoch_body, (state, key), xs=None, length=config.epochs
        )
        return state, jax.tree.map(lambda value: jnp.mean(value), metrics), key

    return update
