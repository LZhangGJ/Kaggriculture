"""Deterministic inactive opponents for official Python and batched JAX use."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from kaggriculture_jax.constants import MAX_MARKET_ORDERS, MAX_UNITS, UnitOp
from kaggriculture_jax.policy import combine_player_actions
from kaggriculture_jax.types import Action


def official_null_agent(obs: dict, configuration=None) -> dict:
    """Return only legal PASS actions for the currently controlled official seat."""

    del configuration
    farms = obs.get("farms", []) or []
    seat = 1 if int(obs.get("player", 0) or 0) == 1 else 0
    farm = farms[seat] if seat < len(farms) else {}
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in (farm.get("hands", []) or [])],
        "market": [],
    }


def _normalize_batch_shape(batch_shape: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(batch_shape, int):
        if batch_shape < 0:
            raise ValueError("batch_shape must be non-negative")
        return (batch_shape,) if batch_shape else ()
    normalized = tuple(int(value) for value in batch_shape)
    if any(value < 0 for value in normalized):
        raise ValueError("batch_shape entries must be non-negative")
    return normalized


def jax_null_player_action(batch_shape: int | Sequence[int] = ()) -> dict:
    """Return a player action compatible with ``combine_player_actions``.

    The farmer is provided as the sole active unit and executes PASS. There are
    no market orders. Values are constants and therefore cheap to broadcast in
    a batched JIT graph.
    """

    shape = _normalize_batch_shape(batch_shape)
    return {
        "unit_op": jnp.full((*shape, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8),
        "unit_item": jnp.full((*shape, MAX_UNITS), -1, dtype=jnp.int8),
        "unit_amount": jnp.ones((*shape, MAX_UNITS), dtype=jnp.int32),
        "unit_count": jnp.ones(shape, dtype=jnp.int8),
        "market_op": jnp.zeros((*shape, MAX_MARKET_ORDERS), dtype=jnp.int8),
        "market_item": jnp.full(
            (*shape, MAX_MARKET_ORDERS), -1, dtype=jnp.int8
        ),
        "market_amount": jnp.zeros(
            (*shape, MAX_MARKET_ORDERS), dtype=jnp.int32
        ),
        "market_count": jnp.zeros(shape, dtype=jnp.int8),
    }


def combine_with_null_opponent(
    player_action: dict,
    *,
    player_seat: int,
) -> Action:
    """Combine a player action with the constant null action in either seat."""

    if player_seat not in (0, 1):
        raise ValueError("player_seat must be 0 or 1")
    batch_shape = tuple(player_action["unit_op"].shape[:-1])
    null_action = jax_null_player_action(batch_shape)
    if player_seat == 0:
        return combine_player_actions(player_action, null_action)
    return combine_player_actions(null_action, player_action)


__all__ = [
    "combine_with_null_opponent",
    "jax_null_player_action",
    "official_null_agent",
]

