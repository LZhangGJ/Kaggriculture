"""GPU-native observation encoder and compact multi-head policy/value network."""

from __future__ import annotations

from typing import NamedTuple

from flax import linen as nn
import jax
import jax.numpy as jnp

from .constants import (
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_WATERED,
    MAX_MARKET_ORDERS,
    MAX_SHOPS,
    MAX_UNITS,
    NUM_PLAYERS,
    NUM_SHED_ITEMS,
    MarketOp,
    UnitOp,
)
from .types import Action, State


NUM_UNIT_OPS = len(UnitOp)
NUM_MARKET_OPS = len(MarketOp)
POLICY_MARKET_SLOTS = 2
AMOUNT_VALUES = jnp.asarray((0, 1, 2, 3, 4, 8, 16, 32), dtype=jnp.int32)


class PolicyHeads(NamedTuple):
    unit_op_logits: jax.Array
    unit_item_logits: jax.Array
    unit_amount_logits: jax.Array
    market_op_logits: jax.Array
    market_item_logits: jax.Array
    market_amount_logits: jax.Array
    value: jax.Array


def _tile_features(state: State, order: jax.Array) -> jax.Array:
    kind = state.tile_kind[order].astype(jnp.float32) / 5.0
    crop = (state.tile_crop[order].astype(jnp.float32) + 1.0) / 5.0
    animal = (state.tile_animal[order].astype(jnp.float32) + 1.0) / 3.0
    origin = (state.tile_origin_day[order].astype(jnp.float32) + 1.0) / 30.0
    yield_units = state.tile_yield[order].astype(jnp.float32) / 10.0
    neglect = state.tile_neglect[order].astype(jnp.float32) / 2.0
    max_life = (state.tile_max_lifespan[order].astype(jnp.float32) + 1.0) / 720.0
    fertilized = (
        state.tile_fertilized_until[order].astype(jnp.float32) + 1.0
    ) / 30.0
    pending = state.tile_pending_care[order].astype(jnp.float32) / 8.0
    flags = state.tile_flags[order]
    flag_features = jnp.stack(
        (
            (flags & FLAG_WATERED) != 0,
            (flags & FLAG_FED) != 0,
            (flags & FLAG_CARED) != 0,
            (flags & FLAG_FERTILIZER_AVAILABLE) != 0,
        ),
        axis=-1,
    ).astype(jnp.float32)
    scalar = jnp.stack(
        (
            kind,
            crop,
            animal,
            origin,
            yield_units,
            neglect,
            max_life,
            fertilized,
            pending,
        ),
        axis=-1,
    )
    return jnp.concatenate((scalar, flag_features), axis=-1).reshape((-1,))


def encode_observations(state: State) -> jax.Array:
    """Return actor-visible observations for both viewpoints, shape ``[2, D]``.

    Public farms are ordered self/opponent.  Shed, seeds and carried inventories
    include only the viewing player's private state.
    """

    rows = []
    for player in range(NUM_PLAYERS):
        opponent = 1 - player
        order = jnp.asarray((player, opponent), dtype=jnp.int32)
        unit_pos = state.unit_pos[order].astype(jnp.float32) / 9.0
        unit_active = state.unit_active[order].astype(jnp.float32)
        town_safe = jnp.clip(state.town_shops, 0, MAX_SHOPS - 1)
        town_active = jnp.arange(MAX_SHOPS) < state.town_count
        town = (
            jax.nn.one_hot(town_safe, MAX_SHOPS, dtype=jnp.float32)
            * town_active[:, None]
        ).reshape((-1,))
        public_and_private = (
            jnp.asarray((state.step / 719.0,), dtype=jnp.float32),
            state.money[order].astype(jnp.float32) / 100_000.0,
            _tile_features(state, order),
            unit_pos.reshape((-1,)),
            unit_active.reshape((-1,)),
            state.unlocked_count[order].astype(jnp.float32) / 4.0,
            state.hires_today[order].astype(jnp.float32) / MAX_UNITS,
            state.shed[player].astype(jnp.float32) / 100.0,
            state.seeds[player].astype(jnp.float32) / 100.0,
            state.unit_inventory[player].astype(jnp.float32).reshape((-1,)) / 100.0,
            (state.market_inventory.astype(jnp.float32) - 10_000.0) / 2_000.0,
            state.market_price.astype(jnp.float32) / 500.0,
            town,
            jnp.asarray((state.town_count / MAX_SHOPS,), dtype=jnp.float32),
        )
        rows.append(jnp.concatenate(public_and_private))
    return jnp.stack(rows)


class PolicyValueNet(nn.Module):
    hidden_sizes: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, observations: jax.Array) -> PolicyHeads:
        x = observations
        for width in self.hidden_sizes:
            x = nn.Dense(width)(x)
            x = nn.tanh(x)
        unit_op = nn.Dense(MAX_UNITS * NUM_UNIT_OPS)(x).reshape(
            (*x.shape[:-1], MAX_UNITS, NUM_UNIT_OPS)
        )
        unit_item = nn.Dense(MAX_UNITS * NUM_SHED_ITEMS)(x).reshape(
            (*x.shape[:-1], MAX_UNITS, NUM_SHED_ITEMS)
        )
        unit_amount = nn.Dense(MAX_UNITS * len(AMOUNT_VALUES))(x).reshape(
            (*x.shape[:-1], MAX_UNITS, len(AMOUNT_VALUES))
        )
        market_op = nn.Dense(POLICY_MARKET_SLOTS * NUM_MARKET_OPS)(x).reshape(
            (*x.shape[:-1], POLICY_MARKET_SLOTS, NUM_MARKET_OPS)
        )
        market_item = nn.Dense(POLICY_MARKET_SLOTS * NUM_SHED_ITEMS)(x).reshape(
            (*x.shape[:-1], POLICY_MARKET_SLOTS, NUM_SHED_ITEMS)
        )
        market_amount = nn.Dense(POLICY_MARKET_SLOTS * len(AMOUNT_VALUES))(x).reshape(
            (*x.shape[:-1], POLICY_MARKET_SLOTS, len(AMOUNT_VALUES))
        )
        value = nn.Dense(1)(x)[..., 0]
        return PolicyHeads(
            unit_op_logits=unit_op,
            unit_item_logits=unit_item,
            unit_amount_logits=unit_amount,
            market_op_logits=market_op,
            market_item_logits=market_item,
            market_amount_logits=market_amount,
            value=value,
        )


def _choose(logits: jax.Array, key: jax.Array, deterministic: bool) -> jax.Array:
    if deterministic:
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)
    return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)


def heads_to_player_action(
    heads: PolicyHeads,
    key: jax.Array,
    unit_active: jax.Array,
    deterministic: bool = False,
) -> tuple[dict[str, jax.Array], jax.Array]:
    keys = jax.random.split(key, 6)
    unit_op = _choose(heads.unit_op_logits, keys[0], deterministic)
    unit_item = _choose(heads.unit_item_logits, keys[1], deterministic)
    unit_amount_index = _choose(heads.unit_amount_logits, keys[2], deterministic)
    market_op_small = _choose(heads.market_op_logits, keys[3], deterministic)
    market_item_small = _choose(heads.market_item_logits, keys[4], deterministic)
    market_amount_index = _choose(heads.market_amount_logits, keys[5], deterministic)

    unit_op = jnp.where(unit_active, unit_op, UnitOp.PASS).astype(jnp.int8)
    unit_item = jnp.where(unit_active, unit_item, -1).astype(jnp.int8)
    unit_amount = jnp.where(
        unit_active, AMOUNT_VALUES[unit_amount_index], 1
    ).astype(jnp.int32)
    batch_shape = market_op_small.shape[:-1]
    market_op = jnp.zeros((*batch_shape, MAX_MARKET_ORDERS), dtype=jnp.int8)
    market_item = jnp.full((*batch_shape, MAX_MARKET_ORDERS), -1, dtype=jnp.int8)
    market_amount = jnp.zeros((*batch_shape, MAX_MARKET_ORDERS), dtype=jnp.int32)
    market_op = market_op.at[..., :POLICY_MARKET_SLOTS].set(
        market_op_small.astype(jnp.int8)
    )
    market_item = market_item.at[..., :POLICY_MARKET_SLOTS].set(
        market_item_small.astype(jnp.int8)
    )
    market_amount = market_amount.at[..., :POLICY_MARKET_SLOTS].set(
        AMOUNT_VALUES[market_amount_index]
    )
    action = {
        "unit_op": unit_op,
        "unit_item": unit_item,
        "unit_amount": unit_amount,
        "unit_count": jnp.sum(unit_active, axis=-1).astype(jnp.int8),
        "market_op": market_op,
        "market_item": market_item,
        "market_amount": market_amount,
        "market_count": jnp.full(
            batch_shape, POLICY_MARKET_SLOTS, dtype=jnp.int8
        ),
    }
    return action, heads.value


def combine_player_actions(player0: dict, player1: dict) -> Action:
    return Action(
        unit_op=jnp.stack((player0["unit_op"], player1["unit_op"]), axis=1),
        unit_item=jnp.stack((player0["unit_item"], player1["unit_item"]), axis=1),
        unit_amount=jnp.stack(
            (player0["unit_amount"], player1["unit_amount"]), axis=1
        ),
        unit_count=jnp.stack(
            (player0["unit_count"], player1["unit_count"]), axis=1
        ),
        market_op=jnp.stack(
            (player0["market_op"], player1["market_op"]), axis=1
        ),
        market_item=jnp.stack(
            (player0["market_item"], player1["market_item"]), axis=1
        ),
        market_amount=jnp.stack(
            (player0["market_amount"], player1["market_amount"]), axis=1
        ),
        market_count=jnp.stack(
            (player0["market_count"], player1["market_count"]), axis=1
        ),
    )
