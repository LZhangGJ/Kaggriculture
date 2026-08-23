"""GPU controller for public G11 V21.1 conditional-memory agent."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    FLAG_FERTILIZER_AVAILABLE,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_PRODUCTS,
    PRODUCTS,
    MarketOp,
    TileKind,
)
from kaggriculture_jax.types import Action, State
from strategic_v5.public_g02_gpu import (
    PublicG02CarryV1,
    _compact_market,
    _project_shed,
    _raw_action,
    _weed_repair,
)


_SELLABLE_IDS = jnp.asarray(
    [PRODUCTS.index(name) for name in (
        "STRAWBERRY", "MELON", "MILK", "WOOL", "EGG",
        "TOMATO", "CARROT", "WHEAT", "FERTILIZER",
    )],
    dtype=jnp.int8,
)
_GLUT_WEIGHT = jnp.asarray((2.0, 3.6, 2.0, 3.2, 1.5, 1.3, 1.0, 1.0, 1.0), dtype=jnp.float32)
_MAX_MEMORY_ACTORS = 13
_POSITION_WEIGHT = jnp.asarray([0.8, *([0.25] * (_MAX_MEMORY_ACTORS - 1))], dtype=jnp.float32)
_COUNT_WEIGHT = jnp.asarray([*([3.0] * 10), 0.25], dtype=jnp.float32)


def _safe_market_v21(states: State, action: Action, player: int) -> Action:
    remaining = _project_shed(states, action, player)
    amount = action.market_amount.astype(jnp.int32)
    keep = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    for slot in range(MAX_MARKET_ORDERS):
        active = keep[:, slot]
        sell = active & (action.market_op[:, slot] == MarketOp.SELL)
        item = action.market_item[:, slot].astype(jnp.int32)
        valid = (item >= 0) & (item < NUM_PRODUCTS)
        safe_item = jnp.clip(item, 0, NUM_PRODUCTS - 1)
        requested = jnp.maximum(amount[:, slot], 0)
        quantity = jnp.minimum(requested, remaining[jnp.arange(item.shape[0]), safe_item])
        keep_sell = sell & valid & (quantity > 0)
        amount = amount.at[:, slot].set(jnp.where(sell, quantity, amount[:, slot]))
        keep = keep.at[:, slot].set(jnp.where(sell, keep_sell, active))
        remaining = remaining.at[jnp.arange(item.shape[0]), safe_item].add(
            jnp.where(keep_sell, -quantity, 0)
        )
    return _compact_market(action._replace(market_amount=amount), keep)


def _current_signature(states: State, opponent: int) -> jax.Array:
    batch = states.step.shape[0]
    active = states.unit_active[:, opponent, :_MAX_MEMORY_ACTORS]
    workers = jnp.sum(active[:, 1:].astype(jnp.int16), axis=1, keepdims=True)
    unlock_count = states.unlocked_count[:, opponent].astype(jnp.int16)
    unlock_mask = ((1 << unlock_count) - 1)[:, None]
    positions = jnp.where(
        active[..., None], states.unit_pos[:, opponent, :_MAX_MEMORY_ACTORS].astype(jnp.int16), -1
    ).reshape(batch, -1)
    crops = states.tile_crop[:, opponent]
    animals = states.tile_animal[:, opponent]
    kinds = states.tile_kind[:, opponent]
    tile_yield = jnp.maximum(states.tile_yield[:, opponent], 0).astype(jnp.int16)
    counts = jnp.stack(
        [
            *(jnp.sum(crops == crop, axis=(1, 2)) for crop in range(5)),
            jnp.sum(animals == 1, axis=(1, 2)),  # COW
            jnp.sum(animals == 2, axis=(1, 2)),  # SHEEP
            jnp.sum(animals == 0, axis=(1, 2)),  # GOOSE
            jnp.sum(kinds == TileKind.PASTURE, axis=(1, 2)),
            jnp.sum(kinds == TileKind.COOP, axis=(1, 2)),
            jnp.sum(kinds == TileKind.WEED, axis=(1, 2)),
        ],
        axis=1,
    ).astype(jnp.int16)
    yields = jnp.stack(
        [
            *(jnp.sum(jnp.where(crops == crop, tile_yield, 0), axis=(1, 2)) for crop in range(5)),
            jnp.sum(jnp.where(animals == 1, tile_yield, 0), axis=(1, 2)),
            jnp.sum(jnp.where(animals == 2, tile_yield, 0), axis=(1, 2)),
            jnp.sum(jnp.where(animals == 0, tile_yield, 0), axis=(1, 2)),
        ],
        axis=1,
    ).astype(jnp.int16)
    return jnp.concatenate((workers, unlock_mask, positions, counts, yields), axis=1)


def _prototype_distance(current: jax.Array, prototype: jax.Array) -> jax.Array:
    # current: [B,47], prototype: [B,P,47]
    workers = 12.0 * jnp.abs(current[:, None, 0] - prototype[:, :, 0])
    xor = jnp.bitwise_xor(
        current[:, None, 1].astype(jnp.int32), prototype[:, :, 1].astype(jnp.int32)
    )
    bits = sum(((xor >> bit) & 1) for bit in range(4)).astype(jnp.float32)
    positions = jnp.abs(
        current[:, None, 2:28].reshape(current.shape[0], 1, _MAX_MEMORY_ACTORS, 2)
        - prototype[:, :, 2:28].reshape(current.shape[0], prototype.shape[1], _MAX_MEMORY_ACTORS, 2)
    )
    position_distance = jnp.sum(
        positions * _POSITION_WEIGHT[None, None, :, None], axis=(2, 3)
    )
    count_distance = jnp.sum(
        jnp.abs(current[:, None, 28:39] - prototype[:, :, 28:39])
        * _COUNT_WEIGHT[None, None, :],
        axis=2,
    )
    yield_distance = 0.15 * jnp.sum(
        jnp.abs(current[:, None, 39:47] - prototype[:, :, 39:47]), axis=2
    )
    return workers.astype(jnp.float32) + 7.0 * bits + position_distance + count_distance + yield_distance


def _conditional_reorder_v21(
    states: State,
    action: Action,
    prototype_signature: jax.Array,
    prototype_sales: jax.Array,
    player: int,
) -> Action:
    step = jnp.clip(states.step.astype(jnp.int32), 0, 718)
    opponent = 1 - player
    current = _current_signature(states, opponent)
    # [P,B,F] -> [B,P,F]
    at_step = jnp.transpose(jnp.take(prototype_signature, step, axis=1), (1, 0, 2))
    distance = _prototype_distance(current, at_step)
    closest = jnp.argmin(distance, axis=1)
    best_distance = jnp.take_along_axis(distance, closest[:, None], axis=1)[:, 0]
    sales_at_step = jnp.transpose(jnp.take(prototype_sales, step, axis=1), (1, 0, 2))
    predicted = sales_at_step[jnp.arange(step.shape[0]), closest]
    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    active = slot < action.market_count[:, None]
    sell = active & (action.market_op == MarketOp.SELL)
    safe_item = jnp.clip(action.market_item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    collided = sell & jnp.take_along_axis(predicted, safe_item, axis=1)
    enable = (best_distance <= 48.0) & jnp.any(sell, axis=1)
    collided = collided & enable[:, None]
    order = jnp.argsort(
        jnp.where(collided, slot, slot + MAX_MARKET_ORDERS), axis=1, stable=True
    )
    return action._replace(
        market_op=jnp.take_along_axis(action.market_op, order, axis=1),
        market_item=jnp.take_along_axis(action.market_item, order, axis=1),
        market_amount=jnp.take_along_axis(action.market_amount, order, axis=1),
    )


def _opponent_exposure(states: State, opponent: int) -> jax.Array:
    crops = states.tile_crop[:, opponent]
    animals = states.tile_animal[:, opponent]
    tile_yield = jnp.maximum(states.tile_yield[:, opponent], 0).astype(jnp.float32)
    exposure = jnp.zeros((states.step.shape[0], NUM_PRODUCTS), dtype=jnp.float32)
    for crop in range(5):
        exposure = exposure.at[:, crop].set(
            jnp.sum(jnp.where(crops == crop, jnp.maximum(tile_yield, 1.0), 0.0), axis=(1, 2))
        )
    for animal, product in ((1, 6), (2, 7), (0, 5)):  # cow/milk, sheep/wool, goose/egg
        exposure = exposure.at[:, product].add(
            jnp.sum(jnp.where(animals == animal, 1.0 + tile_yield, 0.0), axis=(1, 2))
        )
    fertilizer = (states.tile_flags[:, opponent] & jnp.uint8(FLAG_FERTILIZER_AVAILABLE)) != 0
    return exposure.at[:, 8].add(jnp.sum(fertilizer, axis=(1, 2)))


def _terminal_market_v21(states: State, action: Action, player: int) -> Action:
    shed = _project_shed(states, action, player)
    exposure = _opponent_exposure(states, 1 - player)
    quantities = shed[:, _SELLABLE_IDS].astype(jnp.int32)
    prices = states.market_price[:, _SELLABLE_IDS].astype(jnp.float32)
    scores = (
        (1.0 + exposure[:, _SELLABLE_IDS])
        * _GLUT_WEIGHT[None, :]
        * jnp.maximum(prices, 1.0)
        * jnp.log1p(quantities.astype(jnp.float32))
    )
    valid = quantities > 0
    order = jnp.argsort(jnp.where(valid, -scores, jnp.inf), axis=1, stable=True)
    sorted_valid = jnp.take_along_axis(valid, order, axis=1)
    sorted_item = jnp.take_along_axis(
        jnp.broadcast_to(_SELLABLE_IDS[None, :], quantities.shape), order, axis=1
    )
    sorted_quantity = jnp.take_along_axis(quantities, order, axis=1)
    count = jnp.sum(sorted_valid, axis=1).astype(jnp.int8)
    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    present = slot < count[:, None]
    pad_item = jnp.pad(sorted_item, ((0, 0), (0, MAX_MARKET_ORDERS - NUM_PRODUCTS)), constant_values=-1)
    pad_quantity = jnp.pad(sorted_quantity, ((0, 0), (0, MAX_MARKET_ORDERS - NUM_PRODUCTS)))
    terminal = states.step == 718
    replacement = Action(
        action.unit_op,
        action.unit_item,
        action.unit_amount,
        action.unit_count,
        jnp.where(present, MarketOp.SELL, MarketOp.NONE).astype(jnp.int8),
        jnp.where(present, pad_item, -1).astype(jnp.int8),
        jnp.where(present, pad_quantity, 0).astype(jnp.int32),
        count,
    )
    return Action(
        *(
            jnp.where(
                terminal.reshape((terminal.shape[0],) + (1,) * (left.ndim - 1)),
                left,
                right,
            )
            for left, right in zip(replacement, action, strict=True)
        )
    )


def public_v21_player_action_v1(
    states: State,
    bank,
    skeleton_id: jax.Array,
    prototype_signature: jax.Array,
    prototype_sales: jax.Array,
    carry: PublicG02CarryV1,
    player: int,
) -> tuple[Action, PublicG02CarryV1]:
    action = _raw_action(states, bank, skeleton_id, player)
    action, carry = _weed_repair(states, bank, skeleton_id, action, carry, player)
    action = _safe_market_v21(states, action, player)
    action = _safe_market_v21(states, action, player)
    action = _conditional_reorder_v21(
        states, action, prototype_signature, prototype_sales, player
    )
    action = _terminal_market_v21(states, action, player)
    return action, carry
