"""Exact GPU controller for public G09 C68 THUNDER."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    MAX_MARKET_ORDERS,
    NUM_PRODUCTS,
    PRODUCTS,
    SHOP_NAMES,
    MarketOp,
)
from kaggriculture_jax.types import Action, State, StaticTables
from route_playbook_v1.trace_core import _append_terminal_liquidation_v1
from strategic_v5.boatlee_v16_gpu import _rank_sell_slots
from strategic_v5.public_g02_gpu import (
    PublicG02CarryV1,
    _SHOP_DEMAND,
    _append_sell,
    _project_shed,
    _raw_action,
    _repay,
    _weed_repair,
)
from strategic_v5.public_v14_gpu import _clone_distance_v14


_PREMIUM_IDS = tuple(PRODUCTS.index(name) for name in ("STRAWBERRY", "MELON", "MILK", "WOOL"))


def _planned(bank, skeleton_id, step, product_id: int) -> jax.Array:
    safe_step = jnp.clip(step, 0, 718)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < bank.market_count[skeleton_id, safe_step][:, None]
    return jnp.where(
        (step >= 0) & (step < 719),
        jnp.sum(
            jnp.where(
                active
                & (bank.market_op[skeleton_id, safe_step] == MarketOp.SELL)
                & (bank.market_item[skeleton_id, safe_step] == product_id),
                jnp.maximum(bank.market_amount[skeleton_id, safe_step], 0),
                0,
            ),
            axis=1,
        ),
        0,
    ).astype(jnp.int32)


def _current_shop_demand(states: State) -> jax.Array:
    slot = jnp.arange(states.town_shops.shape[1])[None, :]
    active = slot < states.town_count[:, None]
    shops = jnp.clip(states.town_shops.astype(jnp.int32), 0, len(SHOP_NAMES) - 1)
    return jnp.sum(
        jnp.where(active[..., None], _SHOP_DEMAND[shops], 0), axis=1
    ).astype(jnp.int16)


def _observe_race(
    states: State,
    bank,
    skeleton_id,
    carry: PublicG02CarryV1,
) -> PublicG02CarryV1:
    step = states.step.astype(jnp.int32)
    previous_step = carry.race_last_step.astype(jnp.int32)
    sequential = previous_step == step - 1
    near_clone = _clone_distance_v14(states) <= 6
    scores = carry.race_scores
    events = carry.race_events.astype(jnp.int32)
    qualify_all = []
    for product_id in _PREMIUM_IDS:
        delta = (
            states.market_inventory[:, product_id].astype(jnp.int32)
            - carry.race_prev_inventory[:, product_id].astype(jnp.int32)
        )
        town_drain = jnp.where(
            previous_step % 4 == 0,
            carry.race_shop_demand[:, product_id].astype(jnp.int32),
            0,
        ) + jnp.where(previous_step % 24 == 0, 1, 0)
        inferred = (
            delta
            + town_drain
            - carry.race_own_sells[:, product_id].astype(jnp.int32)
            - _planned(bank, skeleton_id, previous_step, product_id)
        )
        qualify = sequential & near_clone & (inferred >= 4)
        qualify_all.append(qualify)
        for horizon_index, horizon in enumerate(range(1, 7)):
            expected = _planned(
                bank, skeleton_id, previous_step + horizon, product_id
            )
            similarity = jnp.minimum(inferred, expected).astype(jnp.float32) / jnp.maximum(
                jnp.maximum(inferred, expected), 1
            ).astype(jnp.float32)
            delta_score = jnp.where(expected > 0, 1.0 + similarity, -0.15)
            scores = scores.at[:, horizon_index].add(
                jnp.where(qualify, delta_score, 0.0)
            )
    events = events + jnp.sum(jnp.stack(qualify_all, axis=1), axis=1)
    best = jnp.argmax(scores, axis=1)
    learned_horizon = jnp.clip(best + 2, 2, 7).astype(jnp.int8)
    horizon = jnp.where(events >= 2, learned_horizon, carry.race_horizon).astype(jnp.int8)
    return carry._replace(
        race_prev_inventory=states.market_inventory.astype(jnp.int32),
        race_shop_demand=_current_shop_demand(states),
        race_scores=scores,
        race_events=events.astype(jnp.int16),
        race_horizon=horizon,
        race_last_step=step.astype(jnp.int16),
    )


def _repay_debt_ring(
    action: Action, states: State, carry: PublicG02CarryV1
) -> tuple[Action, PublicG02CarryV1]:
    step = states.step.astype(jnp.int32)
    slot = step % carry.debt_ring.shape[1]
    due = carry.debt_ring[jnp.arange(step.shape[0]), slot].astype(jnp.int16)
    action, _unused_step, _unused_due = _repay(
        action, step, step.astype(jnp.int16), due
    )
    ring = carry.debt_ring.at[jnp.arange(step.shape[0]), slot].set(0)
    return action, carry._replace(debt_ring=ring)


def _future_quantity_dynamic(
    bank, skeleton_id, step, horizon, product_id: int
) -> jax.Array:
    future = jnp.clip(step + horizon.astype(jnp.int32), 0, 718)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < bank.market_count[skeleton_id, future][:, None]
    return jnp.where(
        step + horizon.astype(jnp.int32) < 719,
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


def _preempt_c68(
    states: State,
    bank,
    skeleton_id,
    action: Action,
    carry: PublicG02CarryV1,
    player: int,
) -> tuple[Action, PublicG02CarryV1]:
    step = states.step.astype(jnp.int32)
    horizon = carry.race_horizon.astype(jnp.int32)
    enabled = (
        (step >= 120)
        & (step < 680)
        & (_clone_distance_v14(states) <= 6)
        & (action.market_count < MAX_MARKET_ORDERS)
    )
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
    shifted = jnp.zeros_like(carry.race_own_sells, dtype=jnp.int32)
    changed = jnp.zeros_like(enabled)
    for product_id in _PREMIUM_IDS:
        future_quantity = _future_quantity_dynamic(
            bank, skeleton_id, step, horizon, product_id
        )
        target = jnp.minimum(
            jnp.minimum(remaining[:, product_id], future_quantity), 30
        )
        append = (
            enabled
            & (future_quantity >= 4)
            & (target > 0)
            & (action.market_count < MAX_MARKET_ORDERS)
        )
        action = _append_sell(
            action,
            append,
            jnp.full_like(step, product_id, dtype=jnp.int8),
            target,
        )
        remaining = remaining.at[:, product_id].set(
            jnp.where(
                append,
                jnp.maximum(remaining[:, product_id] - target, 0),
                remaining[:, product_id],
            )
        )
        shifted = shifted.at[:, product_id].set(jnp.where(append, target, 0))
        changed = changed | append
    due_slot = (step + horizon) % carry.debt_ring.shape[1]
    old_due = carry.debt_ring[jnp.arange(step.shape[0]), due_slot].astype(jnp.int32)
    ring = carry.debt_ring.at[jnp.arange(step.shape[0]), due_slot].set(
        jnp.where(changed[:, None], old_due + shifted, old_due).astype(jnp.int16)
    )
    return action, carry._replace(debt_ring=ring)


def _record_own_sells(action: Action, carry: PublicG02CarryV1) -> PublicG02CarryV1:
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    sold = jnp.zeros_like(carry.race_own_sells, dtype=jnp.int32)
    for product_id in _PREMIUM_IDS:
        quantity = jnp.sum(
            jnp.where(
                active
                & (action.market_op == MarketOp.SELL)
                & (action.market_item == product_id),
                jnp.maximum(action.market_amount, 0),
                0,
            ),
            axis=1,
        )
        sold = sold.at[:, product_id].set(quantity)
    return carry._replace(race_own_sells=sold.astype(jnp.int16))


def public_c68_player_action_v1(
    states: State,
    tables: StaticTables,
    bank,
    skeleton_id: jax.Array,
    carry: PublicG02CarryV1,
    player: int,
) -> tuple[Action, PublicG02CarryV1]:
    carry = _observe_race(states, bank, skeleton_id, carry)
    action = _raw_action(states, bank, skeleton_id, player)
    action, carry = _weed_repair(states, bank, skeleton_id, action, carry, player)
    action, carry = _repay_debt_ring(action, states, carry)
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
    action, carry = _preempt_c68(
        states, bank, skeleton_id, action, carry, player
    )
    action = _append_terminal_liquidation_v1(states, action, player)
    carry = _record_own_sells(action, carry)
    return action, carry

