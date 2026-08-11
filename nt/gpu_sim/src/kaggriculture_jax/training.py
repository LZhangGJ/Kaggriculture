"""Independent GPU-resident self-play collection and clipped PPO update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from .policy import (
    AMOUNT_VALUES,
    POLICY_MARKET_SLOTS,
    PolicyHeads,
    combine_player_actions,
    encode_observations,
    heads_to_player_action,
)
from .simulator import batched_step_sync
from .types import Action, Events, State, StaticTables


PolicyApply = Callable[[object, jax.Array], PolicyHeads]


class TrainingActions(NamedTuple):
    unit_op: jax.Array
    unit_item: jax.Array
    unit_amount_index: jax.Array
    unit_mask: jax.Array
    market_op: jax.Array
    market_item: jax.Array
    market_amount_index: jax.Array


class Transition(NamedTuple):
    observations: jax.Array
    actions: TrainingActions
    old_logprob: jax.Array
    old_value: jax.Array
    reward: jax.Array
    done: jax.Array


class SelfPlayRollout(NamedTuple):
    final_state: State
    transitions: Transition
    bootstrap_value: jax.Array


class PPOMetrics(NamedTuple):
    loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    clip_fraction: jax.Array
    grad_norm: jax.Array


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 1.0


def _selected_logprob(logits: jax.Array, index: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, index[..., None], axis=-1)[..., 0]


def _categorical_entropy(logits: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probabilities = jnp.exp(log_probs)
    return -jnp.sum(probabilities * log_probs, axis=-1)


def player_logprob_entropy(
    heads: PolicyHeads,
    action: dict[str, jax.Array],
    unit_active: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    unit_amount_index = jnp.argmax(
        action["unit_amount"][..., None] == AMOUNT_VALUES, axis=-1
    )
    market_amount_index = jnp.argmax(
        action["market_amount"][..., :POLICY_MARKET_SLOTS, None]
        == AMOUNT_VALUES,
        axis=-1,
    )
    unit_logprob = (
        _selected_logprob(heads.unit_op_logits, action["unit_op"].astype(jnp.int32))
        + _selected_logprob(
            heads.unit_item_logits,
            jnp.clip(action["unit_item"], 0, heads.unit_item_logits.shape[-1] - 1),
        )
        + _selected_logprob(heads.unit_amount_logits, unit_amount_index)
    )
    unit_entropy = (
        _categorical_entropy(heads.unit_op_logits)
        + _categorical_entropy(heads.unit_item_logits)
        + _categorical_entropy(heads.unit_amount_logits)
    )
    market_op = action["market_op"][..., :POLICY_MARKET_SLOTS].astype(jnp.int32)
    market_item = action["market_item"][..., :POLICY_MARKET_SLOTS]
    market_item_active = market_op != 0
    market_logprob = _selected_logprob(heads.market_op_logits, market_op)
    market_logprob = market_logprob + market_item_active * (
        _selected_logprob(
            heads.market_item_logits,
            jnp.clip(market_item, 0, heads.market_item_logits.shape[-1] - 1),
        )
        + _selected_logprob(heads.market_amount_logits, market_amount_index)
    )
    market_entropy = _categorical_entropy(heads.market_op_logits)
    market_entropy = market_entropy + market_item_active * (
        _categorical_entropy(heads.market_item_logits)
        + _categorical_entropy(heads.market_amount_logits)
    )
    return (
        jnp.sum(unit_logprob * unit_active, axis=-1)
        + jnp.sum(market_logprob, axis=-1),
        jnp.sum(unit_entropy * unit_active, axis=-1)
        + jnp.sum(market_entropy, axis=-1),
    )


def _training_actions(action: Action) -> TrainingActions:
    unit_amount_index = jnp.argmax(
        action.unit_amount[..., None] == AMOUNT_VALUES, axis=-1
    )
    market_amount_index = jnp.argmax(
        action.market_amount[..., :POLICY_MARKET_SLOTS, None] == AMOUNT_VALUES,
        axis=-1,
    )
    unit_mask = (
        jnp.arange(action.unit_op.shape[-1])[None, None, :]
        < action.unit_count[..., None]
    )
    return TrainingActions(
        unit_op=action.unit_op,
        unit_item=action.unit_item,
        unit_amount_index=unit_amount_index,
        unit_mask=unit_mask,
        market_op=action.market_op[..., :POLICY_MARKET_SLOTS],
        market_item=action.market_item[..., :POLICY_MARKET_SLOTS],
        market_amount_index=market_amount_index,
    )


def make_selfplay_collector(
    policy_apply: PolicyApply,
    *,
    rollout_steps: int,
    deterministic: bool = False,
):
    """Return a JIT-able homogeneous self-play collector with no host callbacks."""

    def collect(
        initial_states: State,
        events: Events,
        tables: StaticTables,
        params: object,
        key: jax.Array,
    ) -> SelfPlayRollout:
        def body(carry, _):
            states, rng = carry
            rng, key0, key1 = jax.random.split(rng, 3)
            observations = jax.vmap(encode_observations)(states)
            heads = policy_apply(params, observations)
            heads0 = jax.tree.map(lambda value: value[:, 0], heads)
            heads1 = jax.tree.map(lambda value: value[:, 1], heads)
            action0, value0 = heads_to_player_action(
                heads0, key0, states.unit_active[:, 0], deterministic
            )
            action1, value1 = heads_to_player_action(
                heads1, key1, states.unit_active[:, 1], deterministic
            )
            action = combine_player_actions(action0, action1)
            logprob0, _ = player_logprob_entropy(
                heads0, action0, states.unit_active[:, 0]
            )
            logprob1, _ = player_logprob_entropy(
                heads1, action1, states.unit_active[:, 1]
            )
            next_states = batched_step_sync(states, action, events, tables)
            outcome = jnp.sign(
                next_states.money[:, 0] - next_states.money[:, 1]
            ).astype(jnp.float32)
            just_finished = next_states.done & (~states.done)
            rewards = jnp.stack((outcome, -outcome), axis=1) * just_finished[:, None]
            transition = Transition(
                observations=observations,
                actions=_training_actions(action),
                old_logprob=jnp.stack((logprob0, logprob1), axis=1),
                old_value=jnp.stack((value0, value1), axis=1),
                reward=rewards,
                done=jnp.broadcast_to(next_states.done[:, None], rewards.shape),
            )
            return (next_states, rng), transition

        (final_states, _), transitions = jax.lax.scan(
            body, (initial_states, key), xs=None, length=rollout_steps
        )
        final_observations = jax.vmap(encode_observations)(final_states)
        bootstrap = policy_apply(params, final_observations).value
        bootstrap = jnp.where(final_states.done[:, None], 0.0, bootstrap)
        return SelfPlayRollout(
            final_state=final_states,
            transitions=transitions,
            bootstrap_value=bootstrap,
        )

    return collect


def generalized_advantage_estimate(
    transitions: Transition,
    bootstrap_value: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    def body(carry, inputs):
        next_advantage, next_value = carry
        reward, value, done = inputs
        not_done = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * not_done * next_value - value
        advantage = delta + gamma * gae_lambda * not_done * next_advantage
        return (advantage, value), advantage

    (_, _), reversed_advantages = jax.lax.scan(
        body,
        (jnp.zeros_like(bootstrap_value), bootstrap_value),
        (
            transitions.reward[::-1],
            transitions.old_value[::-1],
            transitions.done[::-1],
        ),
    )
    advantages = reversed_advantages[::-1]
    return advantages, advantages + transitions.old_value


def create_train_state(
    policy_apply: PolicyApply,
    params: object,
    config: PPOConfig,
) -> train_state.TrainState:
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )
    return train_state.TrainState.create(
        apply_fn=policy_apply, params=params, tx=optimizer
    )


def _heads_from_training_actions(
    heads: PolicyHeads, actions: TrainingActions
) -> tuple[jax.Array, jax.Array]:
    unit_item = jnp.clip(actions.unit_item, 0, heads.unit_item_logits.shape[-1] - 1)
    unit_logprob = (
        _selected_logprob(heads.unit_op_logits, actions.unit_op.astype(jnp.int32))
        + _selected_logprob(heads.unit_item_logits, unit_item)
        + _selected_logprob(heads.unit_amount_logits, actions.unit_amount_index)
    )
    unit_entropy = (
        _categorical_entropy(heads.unit_op_logits)
        + _categorical_entropy(heads.unit_item_logits)
        + _categorical_entropy(heads.unit_amount_logits)
    )
    market_op = actions.market_op.astype(jnp.int32)
    market_item_active = market_op != 0
    market_logprob = _selected_logprob(heads.market_op_logits, market_op)
    market_logprob += market_item_active * (
        _selected_logprob(
            heads.market_item_logits,
            jnp.clip(actions.market_item, 0, heads.market_item_logits.shape[-1] - 1),
        )
        + _selected_logprob(heads.market_amount_logits, actions.market_amount_index)
    )
    market_entropy = _categorical_entropy(heads.market_op_logits)
    market_entropy += market_item_active * (
        _categorical_entropy(heads.market_item_logits)
        + _categorical_entropy(heads.market_amount_logits)
    )
    logprob = jnp.sum(unit_logprob * actions.unit_mask, axis=-1) + jnp.sum(
        market_logprob, axis=-1
    )
    entropy = jnp.sum(unit_entropy * actions.unit_mask, axis=-1) + jnp.sum(
        market_entropy, axis=-1
    )
    return logprob, entropy


def make_ppo_update(policy_apply: PolicyApply, config: PPOConfig):
    """Return one full-batch clipped PPO gradient update."""

    def update(
        state: train_state.TrainState,
        transitions: Transition,
        advantages: jax.Array,
        returns: jax.Array,
    ) -> tuple[train_state.TrainState, PPOMetrics]:
        normalized_advantage = (advantages - jnp.mean(advantages)) / (
            jnp.std(advantages) + 1e-8
        )

        def loss_fn(params):
            heads = policy_apply(params, transitions.observations)
            logprob, entropy = _heads_from_training_actions(
                heads, transitions.actions
            )
            ratio = jnp.exp(logprob - transitions.old_logprob)
            unclipped = ratio * normalized_advantage
            clipped = (
                jnp.clip(
                    ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
                )
                * normalized_advantage
            )
            policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
            value_loss = 0.5 * jnp.mean(jnp.square(heads.value - returns))
            mean_entropy = jnp.mean(entropy)
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * mean_entropy
            )
            approx_kl = jnp.mean(transitions.old_logprob - logprob)
            clip_fraction = jnp.mean(
                (jnp.abs(ratio - 1.0) > config.clip_epsilon).astype(jnp.float32)
            )
            return loss, (
                policy_loss,
                value_loss,
                mean_entropy,
                approx_kl,
                clip_fraction,
            )

        (loss, auxiliary), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        grad_norm = optax.tree.norm(grads)
        new_state = state.apply_gradients(grads=grads)
        metrics = PPOMetrics(loss, *auxiliary, grad_norm)
        return new_state, metrics

    return update
