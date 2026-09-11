"""Exact GPU controller for public G15 V18 closed-loop router."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import NUM_PRODUCTS
from kaggriculture_jax.types import Action, State


def _literal_action(states: State, bank, skeleton_id: jax.Array) -> Action:
    """Return the submitted tape verbatim, including surplus hand actions.

    V18 does not align its returned hand list to the live workforce.  Official
    Python silently ignores surplus entries, but strict action parity must keep
    them in the encoded submission.
    """

    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    return Action(
        bank.unit_op[skeleton_id, step],
        bank.unit_item[skeleton_id, step],
        bank.unit_amount[skeleton_id, step],
        bank.unit_count[skeleton_id, step],
        bank.market_op[skeleton_id, step],
        bank.market_item[skeleton_id, step],
        bank.market_amount[skeleton_id, step],
        bank.market_count[skeleton_id, step],
    )


class V18ClosedLoopCarryV1(NamedTuple):
    selected_market: jax.Array
    selected_day: jax.Array


def initialize_v18_closed_loop_carry_v1(
    batch_size: int,
) -> V18ClosedLoopCarryV1:
    return V18ClosedLoopCarryV1(
        selected_market=jnp.full((batch_size,), -1, dtype=jnp.int8),
        selected_day=jnp.full((batch_size,), -1, dtype=jnp.int8),
    )


def _state_features_v18(states: State, player: int) -> jax.Array:
    """Reproduce the 29 public own-state features used by Python V18."""

    money = jnp.log1p(jnp.maximum(states.money[:, player], 0).astype(jnp.float32))
    hands = jnp.sum(
        states.unit_active[:, player, 1:].astype(jnp.float32), axis=1
    ) / 16.0
    unlocked = states.unlocked_count[:, player].astype(jnp.float32) / 4.0
    crops = states.tile_crop[:, player]
    animals = states.tile_animal[:, player]
    counts = jnp.stack(
        [
            *(jnp.sum(crops == crop, axis=(1, 2)) for crop in range(5)),
            jnp.sum(animals == 1, axis=(1, 2)),  # COW
            jnp.sum(animals == 2, axis=(1, 2)),  # SHEEP
            jnp.sum(animals == 0, axis=(1, 2)),  # GOOSE
        ],
        axis=1,
    ).astype(jnp.float32) / 50.0
    shed = jnp.log1p(
        jnp.maximum(states.shed[:, player, :NUM_PRODUCTS], 0).astype(jnp.float32)
    )
    prices = jnp.maximum(
        states.market_price[:, :NUM_PRODUCTS].astype(jnp.float32), 1.0
    )
    price_features = jnp.log(prices / jnp.mean(prices, axis=1, keepdims=True))
    return jnp.concatenate(
        (money[:, None], hands[:, None], unlocked[:, None], counts, shed, price_features),
        axis=1,
    )


def _last_argmax(scores: jax.Array) -> jax.Array:
    """Match Python max((score, name)) for alphabetically ordered experts."""

    best = jnp.max(scores, axis=1, keepdims=True)
    indices = jnp.broadcast_to(jnp.arange(scores.shape[1])[None, :], scores.shape)
    return jnp.max(jnp.where(scores == best, indices, -1), axis=1).astype(jnp.int8)


def public_v18_closed_loop_player_action_v1(
    states: State,
    bank,
    board_skeleton_id: jax.Array,
    expert_skeleton_ids: jax.Array,
    feature_scale: jax.Array,
    market_bias_by_seat: jax.Array,
    prototypes_by_day: jax.Array,
    distance_strength: jax.Array,
    stay_bonus: jax.Array,
    carry: V18ClosedLoopCarryV1,
    player: int,
) -> tuple[Action, V18ClosedLoopCarryV1]:
    step = states.step.astype(jnp.int32)
    day = jnp.clip(step // 24, 0, 29).astype(jnp.int8)
    reset = step == 0
    selected = jnp.where(reset, -1, carry.selected_market).astype(jnp.int8)
    selected_day = jnp.where(reset, -1, carry.selected_day).astype(jnp.int8)
    choose = (selected < 0) | (selected_day != day)

    features = _state_features_v18(states, player)
    # prototypes: [expert, day, feature] -> [batch, expert, feature]
    at_day = jnp.transpose(
        jnp.take(prototypes_by_day, day.astype(jnp.int32), axis=1), (1, 0, 2)
    )
    distance = jnp.mean(
        ((features[:, None, :] - at_day) / jnp.maximum(feature_scale, 1e-12)) ** 2,
        axis=2,
    )
    scores = market_bias_by_seat[player][None, :] - distance_strength * distance
    scores = scores + stay_bonus * (
        jnp.arange(scores.shape[1])[None, :] == selected[:, None]
    )
    choice = _last_argmax(scores)
    selected = jnp.where(choose, choice, selected).astype(jnp.int8)
    selected_day = jnp.where(choose, day, selected_day).astype(jnp.int8)

    board_action = _literal_action(states, bank, board_skeleton_id)
    market_skeleton = expert_skeleton_ids[selected.astype(jnp.int32)]
    market_action = _literal_action(states, bank, market_skeleton)
    return (
        board_action._replace(
            market_op=market_action.market_op,
            market_item=market_action.market_item,
            market_amount=market_action.market_amount,
            market_count=market_action.market_count,
        ),
        V18ClosedLoopCarryV1(selected, selected_day),
    )
