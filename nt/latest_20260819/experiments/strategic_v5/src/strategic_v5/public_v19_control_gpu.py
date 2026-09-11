"""Exact GPU controller for public G14 V19 replication-to-control.

The frozen V19 artifact carries four experts, but its fitted distance and
stay coefficients are both zero.  The highest-bias expert is therefore
``himanshu_kumar`` for both seats.  Runtime behaviour reduces exactly to that
expert's fixed 719-step tape plus V19's public clone-gated late liquidation
and exact final projected liquidation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import MAX_MARKET_ORDERS, NUM_PRODUCTS, MarketOp
from kaggriculture_jax.types import Action, State
from strategic_v5.public_g02_gpu import _append_sell, _project_shed, _raw_action
from strategic_v5.public_v14_gpu import _clone_distance_v14
from strategic_v5.public_v21_gpu import (
    _GLUT_WEIGHT,
    _SELLABLE_IDS,
    _opponent_exposure,
    _terminal_market_v21,
)


def _late_inventory_market_v19(
    states: State, action: Action, player: int
) -> Action:
    """Append public V19 sell candidates without replacing planned orders."""

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

    result = action
    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    for ordinal in range(NUM_PRODUCTS):
        item = sorted_item[:, ordinal].astype(jnp.int8)
        quantity = sorted_quantity[:, ordinal].astype(jnp.int32)
        active = slot < result.market_count[:, None]
        already = jnp.any(
            active
            & (result.market_op == MarketOp.SELL)
            & (result.market_item == item[:, None]),
            axis=1,
        )
        enabled = (
            sorted_valid[:, ordinal]
            & (~already)
            & (result.market_count < MAX_MARKET_ORDERS)
        )
        result = _append_sell(result, enabled, item, quantity)
    return result


def public_v19_control_player_action_v1(
    states: State,
    bank,
    skeleton_id: jax.Array,
    player: int,
) -> Action:
    action = _raw_action(states, bank, skeleton_id, player)
    late = (states.step >= 680) & (_clone_distance_v14(states) <= 2)
    late_action = _late_inventory_market_v19(states, action, player)
    action = Action(
        *(
            jnp.where(
                late.reshape((late.shape[0],) + (1,) * (left.ndim - 1)),
                left,
                right,
            )
            for left, right in zip(late_action, action, strict=True)
        )
    )
    # V19 replaces, rather than appends, the market list on the final step.
    return _terminal_market_v21(states, action, player)
