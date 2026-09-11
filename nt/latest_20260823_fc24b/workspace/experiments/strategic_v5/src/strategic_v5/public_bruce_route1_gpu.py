"""Exact GPU controller for public G13 Bruce Route1."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import MAX_UNITS, TileKind, UnitOp
from kaggriculture_jax.types import Action, State
from strategic_v5.public_tran_cashflow_gpu import (
    _replace_unit,
    _terminal_action,
    _tile_kind_at,
    _trace_action,
)


class BruceRoute1CarryV1(NamedTuple):
    pending_active: jax.Array
    pending_pos: jax.Array
    pending_op: jax.Array
    pending_item: jax.Array
    pending_amount: jax.Array


def initialize_bruce_route1_carry_v1(batch_size: int) -> BruceRoute1CarryV1:
    actors = (batch_size, MAX_UNITS)
    return BruceRoute1CarryV1(
        pending_active=jnp.zeros(actors, dtype=jnp.bool_),
        pending_pos=jnp.full((*actors, 2), -1, dtype=jnp.int8),
        pending_op=jnp.full(actors, UnitOp.PASS, dtype=jnp.int8),
        pending_item=jnp.full(actors, -1, dtype=jnp.int8),
        pending_amount=jnp.ones(actors, dtype=jnp.int32),
    )


def public_bruce_route1_player_action_v1(
    states: State,
    bank,
    skeleton_id: jax.Array,
    carry: BruceRoute1CarryV1,
    player: int,
) -> tuple[Action, BruceRoute1CarryV1]:
    step = states.step.astype(jnp.int32)
    reset = step == 0
    fresh = initialize_bruce_route1_carry_v1(states.step.shape[0])
    carry = BruceRoute1CarryV1(
        *(
            jnp.where(
                reset.reshape((reset.shape[0],) + (1,) * (new.ndim - 1)),
                new,
                old,
            )
            for new, old in zip(fresh, carry, strict=True)
        )
    )
    action = _trace_action(states, bank, skeleton_id)
    actual_count = jnp.sum(
        states.unit_active[:, player].astype(jnp.int32), axis=1
    )
    submitted_count = action.unit_count.astype(jnp.int32)
    active = carry.pending_active
    pos = carry.pending_pos
    op = carry.pending_op
    item = carry.pending_item
    amount = carry.pending_amount

    # apply_pending: a successful retry may extend a short returned hand list;
    # the bank's unused slots already contain canonical PASS values.
    for actor in range(MAX_UNITS):
        present = actor < actual_count
        current_pos = states.unit_pos[:, player, actor]
        retry = (
            active[:, actor]
            & present
            & jnp.all(current_pos == pos[:, actor], axis=1)
            & (_tile_kind_at(states, player, current_pos) == TileKind.EMPTY)
        )
        action = _replace_unit(
            action,
            jnp.full(step.shape, actor, dtype=jnp.int32),
            retry,
            op[:, actor],
            item[:, actor],
            amount[:, actor],
        )
        submitted_count = jnp.where(
            retry, jnp.maximum(submitted_count, actor + 1), submitted_count
        )
        active = active.at[:, actor].set(active[:, actor] & (~retry))
    action = action._replace(unit_count=submitted_count.astype(jnp.int8))

    # repair_weed_actions: every existing actor whose returned BUILD_PASTURE or
    # PLANT is blocked receives DIG now and its intended action is remembered.
    for actor in range(MAX_UNITS):
        present = (actor < actual_count) & (actor < submitted_count)
        current_pos = states.unit_pos[:, player, actor]
        requested = action.unit_op[:, actor]
        blocked = (
            present
            & (
                (requested == UnitOp.BUILD_PASTURE)
                | (requested == UnitOp.PLANT)
            )
            & (_tile_kind_at(states, player, current_pos) == TileKind.WEED)
        )
        active = active.at[:, actor].set(active[:, actor] | blocked)
        pos = pos.at[:, actor].set(
            jnp.where(blocked[:, None], current_pos, pos[:, actor])
        )
        op = op.at[:, actor].set(jnp.where(blocked, requested, op[:, actor]))
        item = item.at[:, actor].set(
            jnp.where(blocked, action.unit_item[:, actor], item[:, actor])
        )
        amount = amount.at[:, actor].set(
            jnp.where(blocked, action.unit_amount[:, actor], amount[:, actor])
        )
        action = _replace_unit(
            action,
            jnp.full(step.shape, actor, dtype=jnp.int32),
            blocked,
            UnitOp.DIG,
        )

    action = _terminal_action(
        states,
        action,
        jnp.zeros(step.shape, dtype=jnp.bool_),
        player,
    )
    return action, BruceRoute1CarryV1(active, pos, op, item, amount)
