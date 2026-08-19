"""Exact GPU controller for public G10 Four-Hire.

The Python submission is a stack of deterministic public-state controllers:

* a V25/C68 branch selected from three opponent-farm checkpoints;
* two narrow market corrections at steps 500 and 525;
* a second C68 continuation selected from the opponent's step-25 farm and the
  public step-161 WHEAT quote;
* removal of exactly one opening HIRE order.

The source embeds two byte-identical C68 modules.  Both receive the same
observation once per step, start from identical state, and have no external
side effects; by induction their actions and state are identical.  The JAX
port evaluates that transition once and reuses its result for both selectors.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    MAX_MARKET_ORDERS,
    PRODUCTS,
    SHOP_NAMES,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.types import Action, State, StaticTables
from strategic_v5.public_c68_gpu import public_c68_player_action_v1
from strategic_v5.public_g02_gpu import (
    PublicG02CarryV1,
    _append_sell,
    _compact_market,
    _project_shed,
    initialize_public_g02_carry_v1,
)
from strategic_v5.public_v25_gpu import public_v25_player_action_v1


_WHEAT = PRODUCTS.index("WHEAT")
_MILK = PRODUCTS.index("MILK")
_FERTILIZER = PRODUCTS.index("FERTILIZER")
_BRUNCH_SPOT = SHOP_NAMES.index("BRUNCH_SPOT")
_BAKERY = SHOP_NAMES.index("BAKERY")
_SMOOTHIE_SHOP = SHOP_NAMES.index("SMOOTHIE_SHOP")

_CHECKPOINT_STEPS = (25, 49, 145)
_CHECKPOINT_HANDS = (0, 0, 0)
_CHECKPOINT_FARMER = ((4, 4), (4, 3), (4, 4))
_CHECKPOINT_LAND = (1, 1, 1)
_CHECKPOINT_CROPS = (
    (5, 0, 0, 0, 5),
    (5, 0, 0, 0, 5),
    (6, 0, 0, 6, 5),
)
_CHECKPOINT_ANIMALS = ((0, 1, 3), (0, 1, 4), (0, 2, 4))
_CHECKPOINT_STRUCTURES = ((0, 1), (0, 0), (0, 0))


class FourHireCarryV1(NamedTuple):
    outer_family: jax.Array
    outer_locked: jax.Array
    parent_matches: jax.Array
    parent_locked: jax.Array
    parent_ray_allowed: jax.Array
    parent_kaito: PublicG02CarryV1
    shared_ray: PublicG02CarryV1


def initialize_four_hire_carry_v1(batch_size: int) -> FourHireCarryV1:
    return FourHireCarryV1(
        outer_family=jnp.zeros((batch_size,), dtype=jnp.bool_),
        outer_locked=jnp.zeros((batch_size,), dtype=jnp.bool_),
        parent_matches=jnp.zeros((batch_size,), dtype=jnp.int8),
        parent_locked=jnp.zeros((batch_size,), dtype=jnp.bool_),
        parent_ray_allowed=jnp.full((batch_size,), -1, dtype=jnp.int8),
        parent_kaito=initialize_public_g02_carry_v1(batch_size),
        shared_ray=initialize_public_g02_carry_v1(batch_size),
    )


def _select_action(mask: jax.Array, left: Action, right: Action) -> Action:
    return Action(
        *(
            jnp.where(
                mask.reshape((mask.shape[0],) + (1,) * (lhs.ndim - 1)),
                lhs,
                rhs,
            )
            for lhs, rhs in zip(left, right, strict=True)
        )
    )


def _farm_components(states: State, player: int):
    kinds = states.tile_kind[:, player].astype(jnp.int32)
    crops = states.tile_crop[:, player].astype(jnp.int32)
    animals = states.tile_animal[:, player].astype(jnp.int32)
    hands = jnp.sum(states.unit_active[:, player, 1:].astype(jnp.int32), axis=1)
    farmer = states.unit_pos[:, player, 0].astype(jnp.int32)
    land = states.unlocked_count[:, player].astype(jnp.int32)
    crop_counts = jnp.stack(
        [jnp.sum(crops == product, axis=(1, 2)) for product in range(5)], axis=1
    ).astype(jnp.int32)
    animal_counts = jnp.stack(
        [jnp.sum(animals == animal, axis=(1, 2)) for animal in range(3)], axis=1
    ).astype(jnp.int32)
    structures = jnp.stack(
        [
            jnp.sum(
                (kinds == TileKind.COOP) & (crops < 0) & (animals < 0),
                axis=(1, 2),
            ),
            jnp.sum(
                (kinds == TileKind.PASTURE) & (crops < 0) & (animals < 0),
                axis=(1, 2),
            ),
        ],
        axis=1,
    ).astype(jnp.int32)
    return hands, farmer, land, crop_counts, animal_counts, structures


def _distance_to_checkpoint(components, checkpoint: int) -> jax.Array:
    hands, farmer, land, crops, animals, structures = components
    return (
        4 * jnp.abs(hands - _CHECKPOINT_HANDS[checkpoint])
        + jnp.sum(
            jnp.abs(
                farmer
                - jnp.asarray(_CHECKPOINT_FARMER[checkpoint], dtype=jnp.int32)
            ),
            axis=1,
        )
        + 4 * jnp.abs(land - _CHECKPOINT_LAND[checkpoint])
        + 2
        * jnp.sum(
            jnp.abs(
                crops - jnp.asarray(_CHECKPOINT_CROPS[checkpoint], dtype=jnp.int32)
            ),
            axis=1,
        )
        + 3
        * jnp.sum(
            jnp.abs(
                animals
                - jnp.asarray(_CHECKPOINT_ANIMALS[checkpoint], dtype=jnp.int32)
            ),
            axis=1,
        )
        + 2
        * jnp.sum(
            jnp.abs(
                structures
                - jnp.asarray(
                    _CHECKPOINT_STRUCTURES[checkpoint], dtype=jnp.int32
                )
            ),
            axis=1,
        )
    )


def _farm_pair_distance(left, right) -> jax.Array:
    return (
        4 * jnp.abs(left[0] - right[0])
        + jnp.sum(jnp.abs(left[1] - right[1]), axis=1)
        + 4 * jnp.abs(left[2] - right[2])
        + 2 * jnp.sum(jnp.abs(left[3] - right[3]), axis=1)
        + 3 * jnp.sum(jnp.abs(left[4] - right[4]), axis=1)
        + 2 * jnp.sum(jnp.abs(left[5] - right[5]), axis=1)
    )


def _has_shop(states: State, shop_id: int) -> jax.Array:
    slot = jnp.arange(states.town_shops.shape[1])[None, :]
    return jnp.any(
        (slot < states.town_count[:, None]) & (states.town_shops == shop_id), axis=1
    )


def _has_drop(action: Action) -> jax.Array:
    active = jnp.arange(action.unit_op.shape[1])[None, :] < action.unit_count[:, None]
    return jnp.any(active & (action.unit_op == UnitOp.DROP), axis=1)


def _parent_action(
    states: State,
    tables: StaticTables,
    bank,
    kaito_skeleton_id: jax.Array,
    ray_skeleton_id: jax.Array,
    carry: FourHireCarryV1,
    player: int,
    opponent_components,
) -> tuple[Action, Action, FourHireCarryV1]:
    step = states.step.astype(jnp.int32)
    reset = step == 0
    matches = jnp.where(reset, 0, carry.parent_matches).astype(jnp.int8)
    locked = jnp.where(reset, False, carry.parent_locked)
    ray_allowed = jnp.where(reset, -1, carry.parent_ray_allowed).astype(jnp.int8)

    for checkpoint, checkpoint_step in enumerate(_CHECKPOINT_STEPS):
        matched = (step == checkpoint_step) & (
            _distance_to_checkpoint(opponent_components, checkpoint) <= 10
        )
        matches = (matches + matched.astype(jnp.int8)).astype(jnp.int8)
        locked = locked | (matches >= 3)

    decide = locked & (step >= 161) & (ray_allowed < 0)
    allowed = ~_has_shop(states, _BRUNCH_SPOT)
    ray_allowed = jnp.where(decide, allowed.astype(jnp.int8), ray_allowed).astype(
        jnp.int8
    )

    kaito_action, parent_kaito = public_v25_player_action_v1(
        states, tables, bank, kaito_skeleton_id, carry.parent_kaito, player
    )
    ray_action, shared_ray = public_c68_player_action_v1(
        states, tables, bank, ray_skeleton_id, carry.shared_ray, player
    )
    use_ray = locked & (ray_allowed == 1) & (step >= 161)
    action = _select_action(use_ray, ray_action, kaito_action)
    return action, ray_action, carry._replace(
        parent_matches=matches,
        parent_locked=locked,
        parent_ray_allowed=ray_allowed,
        parent_kaito=parent_kaito,
        shared_ray=shared_ray,
    )


def _v6_step525(
    states: State,
    action: Action,
    projected: jax.Array,
    before: jax.Array,
    pair_distance: jax.Array,
) -> Action:
    eligible = (
        (states.step == 525)
        & (action.market_count == 0)
        & _has_drop(action)
        & ((projected[:, _WHEAT] - before[:, _WHEAT]) == 3)
        & (projected[:, _WHEAT] >= 25)
        & (pair_distance <= 5)
    )
    return _append_sell(
        action,
        eligible,
        jnp.full(states.step.shape, _WHEAT, dtype=jnp.int8),
        jnp.full(states.step.shape, 3, dtype=jnp.int32),
    )


def _v7_step500(
    states: State,
    action: Action,
    player: int,
    projected: jax.Array,
    before: jax.Array,
    pair_distance: jax.Array,
) -> Action:
    eligible = (
        (player == 1)
        & (states.step == 500)
        & _has_shop(states, _BAKERY)
        & _has_shop(states, _SMOOTHIE_SHOP)
        & (pair_distance <= 10)
        & _has_drop(action)
        & (action.market_count == 0)
    )
    milk = jnp.maximum(projected[:, _MILK] - before[:, _MILK], 0)
    fertilizer = jnp.maximum(
        projected[:, _FERTILIZER] - before[:, _FERTILIZER], 0
    )
    action = _append_sell(
        action,
        eligible & (milk > 0),
        jnp.full(states.step.shape, _MILK, dtype=jnp.int8),
        milk,
    )
    action = _append_sell(
        action,
        eligible & (fertilizer > 0),
        jnp.full(states.step.shape, _FERTILIZER, dtype=jnp.int8),
        fertilizer,
    )
    return action


def _remove_first_opening_hire(states: State, action: Action) -> Action:
    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    active = slot < action.market_count[:, None]
    hire = active & (action.market_op == MarketOp.HIRE)
    first = hire & (jnp.cumsum(hire.astype(jnp.int8), axis=1) == 1)
    remove = (states.step == 0)[:, None] & first
    return _compact_market(action, ~remove)


def public_four_hire_player_action_v1(
    states: State,
    tables: StaticTables,
    bank,
    kaito_skeleton_id: jax.Array,
    ray_skeleton_id: jax.Array,
    carry: FourHireCarryV1,
    player: int,
) -> tuple[Action, FourHireCarryV1]:
    """Emit the exact public G10 action and updated nested controller state."""

    left_components = _farm_components(states, 0)
    right_components = _farm_components(states, 1)
    opponent_components = right_components if player == 0 else left_components
    pair_distance = _farm_pair_distance(left_components, right_components)
    v7_action, ray_action, carry = _parent_action(
        states,
        tables,
        bank,
        kaito_skeleton_id,
        ray_skeleton_id,
        carry,
        player,
        opponent_components,
    )
    before = states.shed[:, player].astype(jnp.int32)
    needs_projection = jnp.any((states.step == 500) | (states.step == 525))
    projected = jax.lax.cond(
        needs_projection,
        lambda _: _project_shed(states, v7_action, player),
        lambda _: before,
        operand=None,
    )
    v7_action = _v6_step525(
        states, v7_action, projected, before, pair_distance
    )
    v7_action = _v7_step500(
        states, v7_action, player, projected, before, pair_distance
    )

    step = states.step.astype(jnp.int32)
    reset = step == 0
    family = jnp.where(reset, False, carry.outer_family)
    locked = jnp.where(reset, False, carry.outer_locked)
    opponent = 1 - player
    wheat = jnp.sum(
        states.tile_crop[:, opponent].astype(jnp.int32) == _WHEAT, axis=(1, 2)
    )
    pasture = jnp.sum(
        states.tile_kind[:, opponent].astype(jnp.int32) == TileKind.PASTURE,
        axis=(1, 2),
    )
    family = jnp.where((step == 25), (wheat == 14) & (pasture == 8), family)
    locked = jnp.where(
        step == 161,
        family & (states.market_price[:, _WHEAT].astype(jnp.int32) <= 30),
        locked,
    )
    action = _select_action(locked & (step >= 161), ray_action, v7_action)
    action = _remove_first_opening_hire(states, action)
    return action, carry._replace(
        outer_family=family,
        outer_locked=locked,
    )
