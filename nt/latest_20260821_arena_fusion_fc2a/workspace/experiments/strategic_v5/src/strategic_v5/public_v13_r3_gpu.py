"""Exact GPU controller for public G12 V13-R3 order-safe preemption."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import MAX_MARKET_ORDERS, NUM_PRODUCTS, PRODUCTS, MarketOp
from kaggriculture_jax.types import Action, State
from strategic_v5.public_g02_gpu import (
    PublicG02CarryV1,
    _append_sell,
    _project_shed,
    _raw_action,
    _repay,
    _weed_repair,
)
from strategic_v5.public_v14_gpu import _clone_distance_v14, _future_quantity
from strategic_v5.public_v21_gpu import _safe_market_v21


_PREMIUM_IDS = tuple(PRODUCTS.index(name) for name in ("STRAWBERRY", "MELON", "MILK", "WOOL"))
_SELLABLE_IDS = jnp.asarray(
    [PRODUCTS.index(name) for name in (
        "STRAWBERRY", "MELON", "MILK", "WOOL", "WHEAT",
        "FERTILIZER", "EGG", "TOMATO", "CARROT",
    )],
    dtype=jnp.int8,
)


def _remaining_shed(states: State, action: Action, player: int) -> jax.Array:
    remaining = _project_shed(states, action, player)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    for product_id in range(NUM_PRODUCTS):
        sold = jnp.sum(
            jnp.where(
                active
                & (action.market_op == MarketOp.SELL)
                & (action.market_item == product_id),
                jnp.maximum(action.market_amount, 0),
                0,
            ),
            axis=1,
        )
        remaining = remaining.at[:, product_id].set(
            jnp.maximum(remaining[:, product_id] - sold, 0)
        )
    return remaining


def _preempt_v13_r3(
    states: State,
    bank,
    skeleton_id,
    hazard_enabled: jax.Array,
    hazard_cap: jax.Array,
    action: Action,
    carry: PublicG02CarryV1,
    player: int,
) -> tuple[Action, PublicG02CarryV1]:
    step = states.step.astype(jnp.int32)
    future = jnp.clip(step + 1, 0, 718)
    enabled_base = (
        (step >= 120)
        & (step < 680)
        & (carry.due_step < 0)
        & ((step - carry.last_preempt) >= 1)
        & (_clone_distance_v14(states) <= 6)
        & (step + 1 < 719)
    )
    remaining = _remaining_shed(states, action, player)
    due = jnp.zeros_like(carry.due, dtype=jnp.int32)
    changed = jnp.zeros_like(enabled_base)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    for ordinal, product_id in enumerate(_PREMIUM_IDS):
        future_quantity = _future_quantity(bank, skeleton_id, step, 1, product_id)
        cap = hazard_cap[future, ordinal].astype(jnp.int32)
        target = jnp.minimum(
            jnp.minimum(
                jnp.minimum(remaining[:, product_id], future_quantity), 30
            ),
            cap,
        )
        eligible = (
            enabled_base
            & hazard_enabled[future, ordinal]
            & (future_quantity > 0)
            & (target > 0)
        )
        locations = jnp.where(
            active
            & (action.market_op == MarketOp.SELL)
            & (action.market_item == product_id),
            jnp.arange(MAX_MARKET_ORDERS)[None, :],
            MAX_MARKET_ORDERS,
        )
        location = jnp.min(locations, axis=1)
        merge = eligible & (location < MAX_MARKET_ORDERS)
        safe_location = jnp.clip(location, 0, MAX_MARKET_ORDERS - 1)
        batch = jnp.arange(step.shape[0])
        amount = action.market_amount.at[batch, safe_location].set(
            jnp.where(
                merge,
                action.market_amount[batch, safe_location] + target,
                action.market_amount[batch, safe_location],
            )
        )
        action = action._replace(market_amount=amount)
        append = eligible & (~merge) & (action.market_count < MAX_MARKET_ORDERS)
        action = _append_sell(
            action,
            append,
            jnp.full_like(step, product_id, dtype=jnp.int8),
            target,
        )
        success = merge | append
        remaining = remaining.at[:, product_id].set(
            jnp.where(
                success,
                jnp.maximum(remaining[:, product_id] - target, 0),
                remaining[:, product_id],
            )
        )
        due = due.at[:, product_id].set(jnp.where(success, target, 0))
        changed = changed | success
        active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    return action, carry._replace(
        due_step=jnp.where(changed, step + 1, carry.due_step).astype(jnp.int16),
        due=jnp.where(changed[:, None], due, carry.due).astype(jnp.int16),
        last_preempt=jnp.where(changed, step, carry.last_preempt).astype(jnp.int32),
    )


def _terminal_market_v13(states: State, action: Action, player: int) -> Action:
    terminal = states.step == 718
    shed = _project_shed(states, action, player)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    existing_sell = jnp.stack(
        [
            jnp.any(
                active
                & (action.market_op == MarketOp.SELL)
                & (action.market_item == product_id),
                axis=1,
            )
            for product_id in range(NUM_PRODUCTS)
        ],
        axis=1,
    )
    quantities = shed[:, _SELLABLE_IDS].astype(jnp.int32)
    missing = ~existing_sell[:, _SELLABLE_IDS]
    valid = (quantities > 0) & missing
    prices = states.market_price[:, _SELLABLE_IDS].astype(jnp.float32)
    order = jnp.argsort(jnp.where(valid, -prices, jnp.inf), axis=1, stable=True)
    sorted_valid = jnp.take_along_axis(valid, order, axis=1)
    sorted_item = jnp.take_along_axis(
        jnp.broadcast_to(_SELLABLE_IDS[None, :], quantities.shape), order, axis=1
    )
    sorted_quantity = jnp.take_along_axis(quantities, order, axis=1)
    for ordinal in range(NUM_PRODUCTS):
        append = terminal & sorted_valid[:, ordinal] & (action.market_count < MAX_MARKET_ORDERS)
        action = _append_sell(
            action,
            append,
            sorted_item[:, ordinal],
            sorted_quantity[:, ordinal],
        )
    return action


def public_v13_r3_player_action_v1(
    states: State,
    bank,
    skeleton_id: jax.Array,
    hazard_enabled: jax.Array,
    hazard_cap: jax.Array,
    carry: PublicG02CarryV1,
    player: int,
) -> tuple[Action, PublicG02CarryV1]:
    action = _raw_action(states, bank, skeleton_id, player)
    action, carry = _weed_repair(
        states, bank, skeleton_id, action, carry, player, replay_steps=2
    )
    action, due_step, due = _repay(
        action, states.step.astype(jnp.int32), carry.due_step, carry.due
    )
    carry = carry._replace(due_step=due_step, due=due)
    action = _safe_market_v21(states, action, player)
    action, carry = _preempt_v13_r3(
        states,
        bank,
        skeleton_id,
        hazard_enabled,
        hazard_cap,
        action,
        carry,
        player,
    )
    action = _safe_market_v21(states, action, player)
    action = _terminal_market_v13(states, action, player)
    return action, carry
