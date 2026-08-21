"""Exact GPU controller for public G08 V14-H6 preemption agent."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    MAX_MARKET_ORDERS,
    NUM_PRODUCTS,
    PRODUCTS,
    MarketOp,
    TileKind,
)
from kaggriculture_jax.types import Action, State, StaticTables
from route_playbook_v1.trace_core import _append_terminal_liquidation_v1
from strategic_v5.boatlee_v16_gpu import _rank_sell_slots
from strategic_v5.public_g02_gpu import (
    PublicG02CarryV1,
    _append_sell,
    _project_shed,
    _raw_action,
    _repay,
    _weed_repair,
)


_PREMIUM_IDS = tuple(PRODUCTS.index(name) for name in ("STRAWBERRY", "MELON", "MILK", "WOOL"))
_HORIZONS = (6, 5, 4, 3, 2, 1)


def _clone_distance_v14(states: State) -> jax.Array:
    kinds = states.tile_kind.astype(jnp.int32)
    crops = states.tile_crop.astype(jnp.int32)
    animals = states.tile_animal.astype(jnp.int32)
    # Exact alphabetical order of Python's sorted signature keys.  The order
    # does not affect the L1 sum but documenting it prevents field omissions.
    columns = (
        jnp.sum(crops == 1, axis=(2, 3)),  # CARROT
        jnp.sum((crops < 0) & (animals < 0) & (kinds == TileKind.COOP), axis=(2, 3)),
        jnp.sum(animals == 1, axis=(2, 3)),  # COW
        jnp.sum(animals == 0, axis=(2, 3)),  # GOOSE
        jnp.sum(crops == 4, axis=(2, 3)),  # MELON
        jnp.sum((crops < 0) & (animals < 0) & (kinds == TileKind.PASTURE), axis=(2, 3)),
        jnp.sum(animals == 2, axis=(2, 3)),  # SHEEP
        jnp.sum(crops == 3, axis=(2, 3)),  # STRAWBERRY
        jnp.sum(crops == 2, axis=(2, 3)),  # TOMATO
        jnp.sum((crops < 0) & (animals < 0) & (kinds == TileKind.WEED), axis=(2, 3)),
        jnp.sum(crops == 0, axis=(2, 3)),  # WHEAT
    )
    counts = jnp.stack(columns, axis=-1).astype(jnp.int32)
    hands = jnp.sum(states.unit_active[:, :, 1:].astype(jnp.int32), axis=2)
    unlocked = states.unlocked_count.astype(jnp.int32)
    return (
        jnp.abs(hands[:, 0] - hands[:, 1])
        + 3 * jnp.abs(unlocked[:, 0] - unlocked[:, 1])
        + jnp.sum(jnp.abs(counts[:, 0] - counts[:, 1]), axis=1)
    )


def _future_quantity(bank, skeleton_id, step, horizon: int, product_id: int) -> jax.Array:
    future = jnp.clip(step + horizon, 0, 718)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < bank.market_count[skeleton_id, future][:, None]
    return jnp.where(
        step + horizon < 719,
        jnp.sum(
            jnp.where(
                active
                & (bank.market_op[skeleton_id, future] == MarketOp.SELL)
                & (bank.market_item[skeleton_id, future] == product_id),
                jnp.maximum(bank.market_amount[skeleton_id, future], 0),
                0,
            ),
            axis=1,
        ),
        0,
    ).astype(jnp.int32)


def _preempt_v14(
    states: State,
    bank,
    skeleton_id,
    action: Action,
    carry: PublicG02CarryV1,
    player: int,
) -> tuple[Action, PublicG02CarryV1]:
    step = states.step.astype(jnp.int32)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    remaining = _project_shed(states, action, player)
    for product_id in range(NUM_PRODUCTS):
        planned = jnp.sum(
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
            jnp.maximum(remaining[:, product_id] - planned, 0)
        )

    eligible_base = (
        (step >= 120)
        & (step < 680)
        & (carry.due_step < 0)
        & (_clone_distance_v14(states) <= 6)
        & (action.market_count < MAX_MARKET_ORDERS)
    )
    selected = jnp.zeros_like(eligible_base)
    selected_due_step = carry.due_step
    selected_due = carry.due.astype(jnp.int32)
    result = action
    for horizon in _HORIZONS:
        trial = action
        trial_remaining = remaining
        trial_due = jnp.zeros_like(selected_due)
        shifted = jnp.zeros_like(eligible_base)
        for product_id in _PREMIUM_IDS:
            future_quantity = _future_quantity(
                bank, skeleton_id, step, horizon, product_id
            )
            target = jnp.minimum(
                jnp.minimum(trial_remaining[:, product_id], future_quantity), 30
            )
            append = (
                eligible_base
                & (~selected)
                & (future_quantity >= 4)
                & (states.market_price[:, product_id] > 1)
                & (target > 0)
                & (trial.market_count < MAX_MARKET_ORDERS)
            )
            trial = _append_sell(
                trial,
                append,
                jnp.full_like(step, product_id, dtype=jnp.int8),
                target,
            )
            trial_remaining = trial_remaining.at[:, product_id].set(
                jnp.where(
                    append,
                    jnp.maximum(trial_remaining[:, product_id] - target, 0),
                    trial_remaining[:, product_id],
                )
            )
            trial_due = trial_due.at[:, product_id].set(
                jnp.where(append, target, trial_due[:, product_id])
            )
            shifted = shifted | append
        use = shifted & (~selected)
        result = Action(
            *(
                jnp.where(
                    use.reshape((use.shape[0],) + (1,) * (left.ndim - 1)),
                    left,
                    right,
                )
                for left, right in zip(trial, result, strict=True)
            )
        )
        selected_due_step = jnp.where(use, step + horizon, selected_due_step).astype(jnp.int16)
        selected_due = jnp.where(use[:, None], trial_due, selected_due)
        selected = selected | shifted
    return result, carry._replace(
        due_step=selected_due_step,
        due=selected_due.astype(jnp.int16),
    )


def public_v14_player_action_v1(
    states: State,
    tables: StaticTables,
    bank,
    skeleton_id: jax.Array,
    carry: PublicG02CarryV1,
    player: int,
) -> tuple[Action, PublicG02CarryV1]:
    action = _raw_action(states, bank, skeleton_id, player)
    action, carry = _weed_repair(states, bank, skeleton_id, action, carry, player)
    action, due_step, due = _repay(
        action, states.step.astype(jnp.int32), carry.due_step, carry.due
    )
    carry = carry._replace(due_step=due_step, due=due)
    market_op, market_item, market_amount = _rank_sell_slots(
        states,
        tables,
        action.market_op,
        action.market_item,
        action.market_amount,
        action.market_count,
    )
    action = action._replace(
        market_op=market_op,
        market_item=market_item,
        market_amount=market_amount,
    )
    action, carry = _preempt_v14(states, bank, skeleton_id, action, carry, player)
    return _append_terminal_liquidation_v1(states, action, player), carry
