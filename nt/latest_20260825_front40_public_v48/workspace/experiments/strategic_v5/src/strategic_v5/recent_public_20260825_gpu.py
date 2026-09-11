"""Exact JAX runtime for promising agents from the 2026-08-25 recent scan.

The first accepted controller is Kaito V48.  Its six public 719-step tapes
live in ``recent_v48_route_bank_v1.npz``; this module owns only observable
shop routing, sparse weed repair, clone-gated sell preemption, and terminal
collision liquidation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import (
    MARKET_MAX_INVENTORY,
    MARKET_MIN_INVENTORY,
    MAX_MARKET_ORDERS,
    NUM_PRODUCTS,
    PRODUCTS,
    SHOP_NAMES,
    MarketOp,
)
from kaggriculture_jax.types import Action, State
from strategic_v5.high_potential_v20_gpu import (
    HighPotentialRuntimeTablesV1,
    _planned_sales,
    _projected_shed_without_pickups,
)
from strategic_v5.latest_public8_gpu import _append_market, _kaito_clone_distance
from strategic_v5.public_g02_gpu import (
    PublicG02CarryV1,
    _compact_market,
    _raw_action,
    _weed_repair,
    initialize_public_g02_carry_v1,
)
from strategic_v5.public_v21_gpu import _terminal_market_v21


K48_DEFAULT = 20
K48_YARN_FAST = 21
K48_FARM_FAST = 22
K48_YARN_SECOND = 23
K48_YARN_THIRD = 24
K48_BAKERY_CAPITAL = 25

_YARN = SHOP_NAMES.index("YARN_STORE")
_FARMERS = SHOP_NAMES.index("FARMERS_MARKET")
_BAKERY = SHOP_NAMES.index("BAKERY")
_PIZZA = SHOP_NAMES.index("PIZZA_SHOP")
_BRUNCH = SHOP_NAMES.index("BRUNCH_SPOT")
_PET = SHOP_NAMES.index("PET_CAFE")
_WHEAT = PRODUCTS.index("WHEAT")
_MELON = PRODUCTS.index("MELON")

# Python sorts clone candidates by ``(price * quantity, item, quantity)`` in
# reverse order.  Product ids are not lexical, so preserve the string tie-break.
_LEXICAL_RANK = jnp.asarray(
    [sorted(PRODUCTS).index(name) for name in PRODUCTS], dtype=jnp.int32
)
_RING = 4


def _build_v48_price_lut() -> jax.Array:
    """Literal public V48 transcription of official 1.32.7 market curves."""

    params = (
        (25, 10000, 400, "sqrt", 0.80, "log", 0.20),
        (35, 10000, 450, "hinge", 1.00, "sqrt", 0.70),
        (60, 10000, 200, "hinge", 0.40, "sqrt", 0.60),
        (120, 10000, 100, "sqrt", 0.70, "linear", 1.60),
        (250, 10000, 300, "log", 0.20, "sq", 3.60),
        (50, 10000, 332, "hinge", 0.40, "log", 0.20),
        (160, 10000, 122, "sqrt", 0.60, "linear", 1.60),
        (200, 10000, 105, "log", 0.20, "sq", 3.20),
        (100, 10000, 200, "linear", 0.40, "linear", 0.40),
    )
    inventory = np.arange(
        MARKET_MIN_INVENTORY, MARKET_MAX_INVENTORY + 1, dtype=np.float64
    )

    def shape(name: str, value, scale: float):
        value = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
        if name == "linear":
            return value
        if name == "sq":
            return value * value
        if name == "sqrt":
            return np.sqrt(value)
        if name == "log":
            return np.log1p(value)
        if name == "hinge":
            normalized = value / scale
            return normalized + 8.0 * np.maximum(normalized - 1.0, 0.0) ** 2
        raise ValueError(name)

    rows = []
    for base, equilibrium, scale, below, below_target, above, above_target in params:
        below_amplitude = below_target * base / shape(below, scale, scale)
        above_amplitude = above_target * base / shape(above, scale, scale)
        price = np.where(
            inventory < equilibrium,
            base
            + below_amplitude
            * shape(below, equilibrium - inventory, scale),
            base
            - above_amplitude
            * shape(above, inventory - equilibrium, scale),
        )
        rows.append(np.maximum(np.rint(price), 1.0).astype(np.int32))
    return jnp.asarray(np.stack(rows, axis=0), dtype=jnp.int32)


_V48_PRICE_LUT = _build_v48_price_lut()
_V48_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
_V48_DEMAND_TABLE = jnp.asarray(
    [
        [
            (2 if len(_V48_SHOP_PRODUCTS[shop]) == 1 else 1)
            if product in _V48_SHOP_PRODUCTS[shop]
            else 0
            for product in PRODUCTS
        ]
        for shop in SHOP_NAMES
    ],
    dtype=jnp.float32,
)


def _rank_sell_slots_v48(states: State, action: Action) -> Action:
    """Exact public V48 sell-slot ordering under the 1.32.7 hinge curves."""

    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    active = slot < action.market_count[:, None]
    sell = active & (action.market_op == MarketOp.SELL) & (action.market_item >= 0) & (
        action.market_item < NUM_PRODUCTS
    )
    safe_item = jnp.clip(action.market_item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    quantity = jnp.maximum(action.market_amount, 0).astype(jnp.int32)
    inventory = jnp.take_along_axis(states.market_inventory, safe_item, axis=1)
    current_quote = jnp.take_along_axis(states.market_price, safe_item, axis=1)
    later_index = jnp.clip(
        inventory + quantity - MARKET_MIN_INVENTORY,
        0,
        _V48_PRICE_LUT.shape[1] - 1,
    )
    later_quote = _V48_PRICE_LUT[safe_item, later_index]
    impact = quantity.astype(jnp.float32) * jnp.maximum(
        current_quote - later_quote, 0
    ).astype(jnp.float32)
    active_shops = (
        jnp.arange(states.town_shops.shape[1])[None, :] < states.town_count[:, None]
    )
    safe_shops = jnp.clip(
        states.town_shops.astype(jnp.int32), 0, len(SHOP_NAMES) - 1
    )
    demand_by_item = jnp.sum(
        _V48_DEMAND_TABLE[safe_shops] * active_shops[..., None], axis=1
    ) * 6.0
    demand_by_item = demand_by_item + jnp.asarray(
        [1.0] * (NUM_PRODUCTS - 1) + [0.0], dtype=jnp.float32
    )[None, :]
    demand = jnp.maximum(
        jnp.take_along_axis(demand_by_item, safe_item, axis=1), 0.25
    )
    excess = jnp.maximum(inventory + quantity - 10000, 0).astype(jnp.float32)
    urgency = jnp.minimum(1.0, (excess / demand) / 10.0)
    score = jnp.where(sell, impact * (1.0 + 0.25 * urgency), -jnp.inf)
    ranked_index = jnp.argsort(-score, axis=1, stable=True)
    ranked_op = jnp.take_along_axis(action.market_op, ranked_index, axis=1)
    ranked_item = jnp.take_along_axis(action.market_item, ranked_index, axis=1)
    ranked_amount = jnp.take_along_axis(action.market_amount, ranked_index, axis=1)
    ordinal = jnp.maximum(jnp.cumsum(sell, axis=1) - 1, 0)
    return action._replace(
        market_op=jnp.where(
            sell, jnp.take_along_axis(ranked_op, ordinal, axis=1), action.market_op
        ),
        market_item=jnp.where(
            sell, jnp.take_along_axis(ranked_item, ordinal, axis=1), action.market_item
        ),
        market_amount=jnp.where(
            sell,
            jnp.take_along_axis(ranked_amount, ordinal, axis=1),
            action.market_amount,
        ),
    )


class KaitoV48CarryV1(NamedTuple):
    child: PublicG02CarryV1
    route_id: jax.Array
    preempt_near_streak: jax.Array
    preempt_latched: jax.Array
    clone_veto: jax.Array
    due_ring: jax.Array


def initialize_kaito_v48_carry_v1(batch_size: int) -> KaitoV48CarryV1:
    return KaitoV48CarryV1(
        child=initialize_public_g02_carry_v1(batch_size),
        route_id=jnp.full((batch_size,), K48_DEFAULT, dtype=jnp.int8),
        preempt_near_streak=jnp.zeros((batch_size,), dtype=jnp.int16),
        preempt_latched=jnp.zeros((batch_size,), dtype=jnp.bool_),
        clone_veto=jnp.zeros((batch_size,), dtype=jnp.bool_),
        due_ring=jnp.zeros((batch_size, _RING, NUM_PRODUCTS), dtype=jnp.int16),
    )


def _choose(mask: jax.Array, left: jax.Array, right: jax.Array) -> jax.Array:
    shape = (mask.shape[0],) + (1,) * (left.ndim - 1)
    return jnp.where(mask.reshape(shape), left, right)


def _reset_child_on_switch(
    child: PublicG02CarryV1, switched: jax.Array
) -> PublicG02CarryV1:
    """A newly selected Python child receives its first call at the switch."""

    fresh = initialize_public_g02_carry_v1(switched.shape[0])
    # Only the weed transaction belongs to the sparse child.  The other
    # historical PublicG02 fields are unused by V48 and remain unchanged.
    return child._replace(
        repair_start=_choose(switched, fresh.repair_start, child.repair_start),
        repair_op=_choose(switched, fresh.repair_op, child.repair_op),
        repair_item=_choose(switched, fresh.repair_item, child.repair_item),
        repair_amount=_choose(switched, fresh.repair_amount, child.repair_amount),
    )


def _opponent_asset_counts(states: State, player: int):
    opponent = 1 - player
    crops = states.tile_crop[:, opponent]
    animals = states.tile_animal[:, opponent]
    return (
        jnp.sum(animals == 1, axis=(1, 2)),  # COW
        jnp.sum(animals == 2, axis=(1, 2)),  # SHEEP
        jnp.sum(animals == 0, axis=(1, 2)),  # GOOSE
        jnp.sum(crops == _WHEAT, axis=(1, 2)),
        jnp.sum(crops == _MELON, axis=(1, 2)),
    )


def _select_kaito_v48_route(
    states: State, carry: KaitoV48CarryV1, player: int
) -> tuple[jax.Array, KaitoV48CarryV1, jax.Array]:
    step = states.step.astype(jnp.int32)
    count = states.town_count.astype(jnp.int32)
    shops = states.town_shops.astype(jnp.int32)
    first = shops[:, 0]
    second = shops[:, 1]
    third = shops[:, 2]
    undecided = carry.route_id == K48_DEFAULT

    first_yarn = undecided & (count >= 1) & (first == _YARN) & (step >= 88)
    first_farm = (
        undecided
        & (~first_yarn)
        & (count >= 1)
        & (first == _FARMERS)
        & (step >= 120)
    )
    second_yarn = (
        undecided
        & (~first_yarn)
        & (~first_farm)
        & (count >= 2)
        & (first != _YARN)
        & (first != _FARMERS)
        & (second == _YARN)
        & (step >= 153)
    )
    allowed_third = ((first == _BRUNCH) & (second == _PET)) | (
        (first == _PET) & (second == _FARMERS)
    )
    third_yarn = (
        undecided
        & (~first_yarn)
        & (~first_farm)
        & (~second_yarn)
        & (count >= 3)
        & (first != _YARN)
        & (first != _FARMERS)
        & (second != _YARN)
        & (third == _YARN)
        & allowed_third
        & (step >= 216)
    )

    cows, sheep, geese, _wheat, melons = _opponent_asset_counts(states, player)
    bakery = (
        undecided
        & (~first_yarn)
        & (~first_farm)
        & (~second_yarn)
        & (~third_yarn)
        & (step == 160)
        & (count >= 2)
        & (first == _BAKERY)
        & (second == _PIZZA)
        & (cows >= 3)
        & (sheep >= 2)
        & (melons >= 10)
        & (geese <= 0)
    )
    route = carry.route_id.astype(jnp.int32)
    route = jnp.where(first_yarn, K48_YARN_FAST, route)
    route = jnp.where(first_farm, K48_FARM_FAST, route)
    route = jnp.where(second_yarn, K48_YARN_SECOND, route)
    route = jnp.where(third_yarn, K48_YARN_THIRD, route)
    route = jnp.where(bakery, K48_BAKERY_CAPITAL, route).astype(jnp.int8)
    switched = route != carry.route_id
    return route.astype(jnp.int32), carry._replace(route_id=route), switched


def _clone_veto_candidate(states: State, player: int) -> jax.Array:
    step = states.step.astype(jnp.int32)
    count = states.town_count.astype(jnp.int32)
    first = states.town_shops[:, 0].astype(jnp.int32)
    cows, sheep, geese, wheat, melons = _opponent_asset_counts(states, player)
    return (
        (step == 120)
        & (count >= 1)
        & (first == _BAKERY)
        & (sheep >= 4)
        & (cows <= 1)
        & (wheat >= 8)
        & (melons >= 7)
        & (geese <= 0)
    )


def _repay_ring(
    action: Action, step: jax.Array, ring: jax.Array
) -> tuple[Action, jax.Array]:
    batch = jnp.arange(step.shape[0])
    slot_id = jnp.mod(step, _RING).astype(jnp.int32)
    due_now = ring[batch, slot_id].astype(jnp.int32)
    ring = ring.at[batch, slot_id].set(0)
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    keep = active.copy()
    amounts = action.market_amount.astype(jnp.int32)
    for slot in range(MAX_MARKET_ORDERS):
        is_sell = active[:, slot] & (action.market_op[:, slot] == MarketOp.SELL)
        item = jnp.clip(
            action.market_item[:, slot].astype(jnp.int32), 0, NUM_PRODUCTS - 1
        )
        owed = due_now[batch, item]
        reduction = jnp.where(
            is_sell, jnp.minimum(jnp.maximum(amounts[:, slot], 0), owed), 0
        )
        amounts = amounts.at[:, slot].add(-reduction)
        due_now = due_now.at[batch, item].add(-reduction)
        keep = keep.at[:, slot].set(~(is_sell & (amounts[:, slot] <= 0)))
    action = _compact_market(action._replace(market_amount=amounts), keep)
    next_slot = jnp.mod(step + 1, _RING).astype(jnp.int32)
    ring = ring.at[batch, next_slot].add(due_now.astype(jnp.int16))
    return action, ring


def _future_sales(bank, route: jax.Array, future_step: jax.Array) -> jax.Array:
    active = jnp.arange(MAX_MARKET_ORDERS)[None, :] < bank.market_count[
        route, future_step
    ][:, None]
    result = jnp.zeros((route.shape[0], NUM_PRODUCTS), dtype=jnp.int32)
    batch = jnp.arange(route.shape[0])
    for slot in range(MAX_MARKET_ORDERS):
        sell = active[:, slot] & (
            bank.market_op[route, future_step, slot] == MarketOp.SELL
        )
        item = jnp.clip(
            bank.market_item[route, future_step, slot].astype(jnp.int32),
            0,
            NUM_PRODUCTS - 1,
        )
        amount = jnp.maximum(
            bank.market_amount[route, future_step, slot], 0
        ).astype(jnp.int32)
        result = result.at[batch, item].add(jnp.where(sell, amount, 0))
    return result


def _kaito_v48_preempt(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    bank,
    route: jax.Array,
    action: Action,
    carry: KaitoV48CarryV1,
    player: int,
    switched: jax.Array,
) -> tuple[Action, KaitoV48CarryV1]:
    step = states.step.astype(jnp.int32)
    distance = _kaito_clone_distance(states)
    near_streak = jnp.where(
        distance <= 2, carry.preempt_near_streak.astype(jnp.int32) + 1, 0
    ).astype(jnp.int16)
    latched = carry.preempt_latched | (
        (step >= 48) & (near_streak.astype(jnp.int32) >= 24)
    )
    veto = carry.clone_veto | _clone_veto_candidate(states, player)
    ring = jnp.where(switched[:, None, None], 0, carry.due_ring).astype(jnp.int16)
    action, ring = _repay_ring(action, step, ring)

    future_step = jnp.clip(step + 2, 0, 718)
    future = _future_sales(bank, route, future_step)
    planned = _planned_sales(action).astype(jnp.int32)
    projected = _projected_shed_without_pickups(states, action, player).astype(
        jnp.int32
    )
    available = jnp.maximum(projected[:, :NUM_PRODUCTS] - planned, 0)
    requested = jnp.maximum(future - planned, 0)
    quantity = jnp.minimum(jnp.minimum(requested, available), 10)
    value = states.market_price[:, :NUM_PRODUCTS].astype(jnp.int32) * quantity
    key = value * 16 + _LEXICAL_RANK[None, :]
    valid = quantity > 0
    order = jnp.argsort(jnp.where(valid, -key, jnp.iinfo(jnp.int32).max), axis=1)

    eligible = (
        latched
        & (~veto)
        & (step >= 160)
        & (step < 700)
        & (step + 2 < 719)
        & (action.market_count < MAX_MARKET_ORDERS)
    )
    total = jnp.zeros(step.shape, dtype=jnp.int32)
    shifted = jnp.zeros((step.shape[0], NUM_PRODUCTS), dtype=jnp.int32)
    batch = jnp.arange(step.shape[0])
    for ordinal in range(NUM_PRODUCTS):
        item = order[:, ordinal].astype(jnp.int32)
        requested_item = quantity[batch, item]
        use = jnp.minimum(requested_item, jnp.maximum(10 - total, 0))
        append = (
            eligible
            & (use > 0)
            & (action.market_count < MAX_MARKET_ORDERS)
        )
        action = _append_market(
            action, append, MarketOp.SELL, item.astype(jnp.int8), use
        )
        used = jnp.where(append, use, 0)
        shifted = shifted.at[batch, item].add(used)
        total = total + used

    changed = total > 0
    due_slot = jnp.mod(step + 2, _RING).astype(jnp.int32)
    ring = ring.at[batch, due_slot].add(shifted.astype(jnp.int16))
    ranked = _rank_sell_slots_v48(states, action)
    action = Action(
        *(_choose(changed, new, old) for new, old in zip(ranked, action, strict=True))
    )
    return action, carry._replace(
        preempt_near_streak=near_streak,
        preempt_latched=latched,
        clone_veto=veto,
        due_ring=ring,
    )


def kaito_v48_player_action_v1(
    states: State,
    runtime: HighPotentialRuntimeTablesV1,
    bank,
    carry: KaitoV48CarryV1,
    player: int,
) -> tuple[Action, KaitoV48CarryV1]:
    """Public V48 semantics using only current official public state."""

    route, carry, switched = _select_kaito_v48_route(states, carry, player)
    child = _reset_child_on_switch(carry.child, switched)
    action = _raw_action(states, bank, route, player)
    action, child = _weed_repair(states, bank, route, action, child, player)
    action = _rank_sell_slots_v48(states, action)
    carry = carry._replace(child=child)
    action, carry = _kaito_v48_preempt(
        states, runtime, bank, route, action, carry, player, switched
    )
    action = _terminal_market_v21(states, action, player)
    return action, carry


__all__ = [
    "KaitoV48CarryV1",
    "initialize_kaito_v48_carry_v1",
    "kaito_v48_player_action_v1",
]
