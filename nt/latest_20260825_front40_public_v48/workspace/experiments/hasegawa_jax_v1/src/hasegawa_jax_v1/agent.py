"""Demand-routed Hasegawa V1 controller that owns the JAX environment step."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from general_project_planner_v1.fulfillment import (
    fulfillment_step_v1,
    reset_fulfillment_controller_v1,
)
from general_project_planner_v1.fulfillment_schema import (
    FulfillmentControllerV1,
    FulfillmentStepDiagnosticsV1,
    ObligationPlanV1,
)
from kaggriculture_jax.constants import (
    ANIMALS,
    CROPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    PRODUCTS,
    SHOP_NAMES,
    TURNS_PER_DAY,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.types import Action, Events, State, StaticTables

from .plan_bank import HasegawaPlanBankV1


_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
_SHOP_DEMAND = jnp.asarray(
    [
        [
            (2 if len(_SHOP_PRODUCTS[shop]) == 1 else 1)
            if product in _SHOP_PRODUCTS[shop]
            else 0
            for product in PRODUCTS
        ]
        for shop in SHOP_NAMES
    ],
    dtype=jnp.int16,
)

_CROP_OP_PLANT = 0
_CROP_OP_WATER = 1
_CROP_OP_HARVEST = 2
_CROP_OP_FERTILIZE = 3
_ANIMAL_OP_FEED = 0
_ANIMAL_OP_CARE = 1
_ANIMAL_OP_PRODUCT = 2
_ANIMAL_OP_FERTILIZER = 3


class HasegawaCarryV1(NamedTuple):
    controller: FulfillmentControllerV1
    selected_route: jax.Array
    selected_day: jax.Array
    route_switch_count: jax.Array


def initialize_hasegawa_carry_v1(batch_size: int) -> HasegawaCarryV1:
    return HasegawaCarryV1(
        controller=reset_fulfillment_controller_v1(batch_size),
        selected_route=jnp.zeros((batch_size,), dtype=jnp.int16),
        selected_day=jnp.full((batch_size,), -1, dtype=jnp.int8),
        route_switch_count=jnp.zeros((batch_size,), dtype=jnp.int16),
    )


def _current_shop_demand(states: State) -> jax.Array:
    slot = jnp.arange(len(SHOP_NAMES), dtype=jnp.int8)[None, :]
    active = slot < states.town_count[:, None]
    shop = jnp.clip(states.town_shops.astype(jnp.int32), 0, len(SHOP_NAMES) - 1)
    return jnp.sum(
        jnp.where(active[:, :, None], _SHOP_DEMAND[shop], 0),
        axis=1,
        dtype=jnp.int16,
    )


def _physical_profile(states: State, player: int):
    crop_ids = jnp.arange(NUM_CROPS, dtype=jnp.int8)
    animal_ids = jnp.arange(NUM_ANIMALS, dtype=jnp.int8)
    crop = jnp.sum(
        states.tile_crop[:, player, :, :, None] == crop_ids[None, None, None, :],
        axis=(1, 2),
        dtype=jnp.int16,
    )
    animal = jnp.sum(
        states.tile_animal[:, player, :, :, None]
        == animal_ids[None, None, None, :],
        axis=(1, 2),
        dtype=jnp.int16,
    )
    return crop, animal


def _route_day(reference: jax.Array, day: jax.Array) -> jax.Array:
    """Return reference[route, day[batch], ...] as [batch, route, ...]."""

    selected = reference[:, day, ...]
    axes = (1, 0, *range(2, selected.ndim))
    return jnp.transpose(selected, axes)


def select_hasegawa_route_v1(
    states: State,
    bank: HasegawaPlanBankV1,
    player: int,
) -> jax.Array:
    """Choose a winning calendar using only information visible *now*.

    Shop identity dominates because the Replay audit found that this is the
    main route branch.  Price, cash, assets and capacity select among routes
    that saw the same unlocked demand.  Final reward is only a tiny tie-break.
    No future shop/event column participates in this score.
    """

    day = jnp.clip(states.step // TURNS_PER_DAY, 0, 29).astype(jnp.int32)
    crop, animal = _physical_profile(states, player)
    demand = _current_shop_demand(states)

    ref_demand = _route_day(bank.ref_shop_demand, day).astype(jnp.float32)
    ref_price = _route_day(bank.ref_market_price, day).astype(jnp.float32)
    ref_money = _route_day(bank.ref_money, day).astype(jnp.float32)
    ref_crop = _route_day(bank.ref_crop, day).astype(jnp.float32)
    ref_animal = _route_day(bank.ref_animal, day).astype(jnp.float32)
    ref_hires = _route_day(bank.ref_hires, day).astype(jnp.float32)
    ref_unlocked = _route_day(bank.ref_unlocked, day).astype(jnp.float32)

    score = (
        5000.0
        * jnp.sum(
            jnp.abs(ref_demand - demand[:, None, :].astype(jnp.float32)), axis=2
        )
        + 12.0
        * jnp.sum(
            jnp.abs(ref_price - states.market_price[:, None, :].astype(jnp.float32)),
            axis=2,
        )
        + 30.0
        * jnp.sum(
            jnp.abs(ref_animal - animal[:, None, :].astype(jnp.float32)), axis=2
        )
        + 8.0
        * jnp.sum(
            jnp.abs(ref_crop - crop[:, None, :].astype(jnp.float32)), axis=2
        )
        + jnp.abs(ref_money - states.money[:, player, None].astype(jnp.float32))
        / 50.0
        + 20.0
        * jnp.abs(ref_hires - states.hires_today[:, player, None].astype(jnp.float32))
        + 100.0
        * jnp.abs(
            ref_unlocked - states.unlocked_count[:, player, None].astype(jnp.float32)
        )
        - bank.final_reward[None, :].astype(jnp.float32) / 100000.0
    )
    return jnp.argmin(score, axis=1).astype(jnp.int16)


def _gather_plan(plan: ObligationPlanV1, route: jax.Array) -> ObligationPlanV1:
    return ObligationPlanV1(*(field[route] for field in plan))


def _prior_cumulative(field: jax.Array, day: jax.Array) -> jax.Array:
    cumulative = jnp.cumsum(field.astype(jnp.int32), axis=1)
    batch = jnp.arange(field.shape[0], dtype=jnp.int32)
    prior_day = jnp.maximum(day - 1, 0)
    prior = cumulative[batch, prior_day]
    mask_shape = (field.shape[0],) + (1,) * (prior.ndim - 1)
    return jnp.where((day > 0).reshape(mask_shape), prior, 0).astype(jnp.int16)


def _rebase_controller_for_day(
    controller: FulfillmentControllerV1,
    plan: ObligationPlanV1,
    day: jax.Array,
) -> FulfillmentControllerV1:
    """Rebase cumulative ledgers when a new daily route is selected.

    This makes a route switch request only the selected route's *current-day*
    obligations.  It prevents the controller from trying to repay historical
    work belonging to the newly selected Replay calendar.
    """

    crop = jnp.zeros_like(controller.crop_operation_count)
    crop = crop.at[:, _CROP_OP_PLANT].set(_prior_cumulative(plan.crop_plant_by_day, day))
    crop = crop.at[:, _CROP_OP_HARVEST].set(
        _prior_cumulative(plan.crop_harvest_by_day, day)
    )
    crop = crop.at[:, _CROP_OP_FERTILIZE].set(
        _prior_cumulative(plan.crop_fertilize_by_day, day)
    )
    animal = jnp.zeros_like(controller.animal_operation_count)
    animal = animal.at[:, _ANIMAL_OP_PRODUCT].set(
        _prior_cumulative(plan.animal_product_by_day, day)
    )
    animal = animal.at[:, _ANIMAL_OP_FERTILIZER].set(
        _prior_cumulative(plan.animal_fertilizer_by_day, day)
    )
    return controller._replace(
        operation_day=day.astype(jnp.int8),
        crop_operation_count=crop,
        animal_operation_count=animal,
        crop_clear_count=_prior_cumulative(plan.crop_clear_by_day, day),
    )


def _opening_action(batch_size: int) -> Action:
    unit_op = jnp.full((batch_size, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8)
    unit_op = unit_op.at[:, 0].set(UnitOp.BUILD_PASTURE)
    market_op_row = jnp.asarray(
        [
            MarketOp.HIRE,
            MarketOp.HIRE,
            MarketOp.HIRE,
            MarketOp.HIRE,
            MarketOp.HIRE,
            MarketOp.BUY_ANIMAL,
            MarketOp.BUY_ANIMAL,
            MarketOp.BUY_SEED,
            MarketOp.BUY_SEED,
            MarketOp.BUY_PRODUCT,
        ],
        dtype=jnp.int8,
    )
    market_item_row = jnp.asarray(
        [-1, -1, -1, -1, -1, NUM_PRODUCTS + 2, NUM_PRODUCTS + 1, 4, 0, 0],
        dtype=jnp.int8,
    )
    market_amount_row = jnp.asarray(
        [0, 0, 0, 0, 0, 2, 2, 11, 6, 4], dtype=jnp.int32
    )
    return Action(
        unit_op=unit_op,
        unit_item=jnp.full((batch_size, MAX_UNITS), -1, dtype=jnp.int8),
        unit_amount=jnp.ones((batch_size, MAX_UNITS), dtype=jnp.int32),
        unit_count=jnp.ones((batch_size,), dtype=jnp.int8),
        market_op=jnp.broadcast_to(market_op_row, (batch_size, MAX_MARKET_ORDERS)),
        market_item=jnp.broadcast_to(
            market_item_row, (batch_size, MAX_MARKET_ORDERS)
        ),
        market_amount=jnp.broadcast_to(
            market_amount_row, (batch_size, MAX_MARKET_ORDERS)
        ),
        market_count=jnp.full((batch_size,), MAX_MARKET_ORDERS, dtype=jnp.int8),
    )


def _pair(controlled: Action, external: Action, player: int) -> Action:
    if player == 0:
        return Action(
            *(jnp.stack((left, right), axis=1) for left, right in zip(controlled, external, strict=True))
        )
    return Action(
        *(jnp.stack((left, right), axis=1) for left, right in zip(external, controlled, strict=True))
    )


def _zero_diagnostics(batch_size: int) -> FulfillmentStepDiagnosticsV1:
    zeros = jnp.zeros((batch_size,), dtype=jnp.int32)
    return FulfillmentStepDiagnosticsV1(
        active_unit_obligations=zeros,
        active_market_obligations=zeros,
        unmet_crop_units=zeros,
        unmet_animal_units=zeros,
        hard_error_count=zeros,
        completed_this_step=zeros,
    )


def hasegawa_step_with_external_v1(
    states: State,
    carry: HasegawaCarryV1,
    bank: HasegawaPlanBankV1,
    external_action: Action,
    player: int,
    events: Events,
    tables: StaticTables,
):
    """Advance one synchronized JAX batch and return the executed joint action."""

    day = jnp.clip(states.step // TURNS_PER_DAY, 0, 29).astype(jnp.int32)
    day_changed = carry.selected_day != day.astype(jnp.int8)
    proposed = select_hasegawa_route_v1(states, bank, player)
    route = jnp.where(day_changed, proposed, carry.selected_route).astype(jnp.int16)
    selected_plan = _gather_plan(bank.plan, route.astype(jnp.int32))
    controller = jax.lax.cond(
        jnp.any(day_changed),
        lambda value: _rebase_controller_for_day(value, selected_plan, day),
        lambda value: value,
        carry.controller,
    )
    next_carry = HasegawaCarryV1(
        controller=controller,
        selected_route=route,
        selected_day=day.astype(jnp.int8),
        route_switch_count=(
            carry.route_switch_count
            + (day_changed & (carry.selected_day >= 0) & (route != carry.selected_route))
        ).astype(jnp.int16),
    )

    def opening_branch(value):
        current_states, current_carry = value
        action = _opening_action(current_states.step.shape[0])
        joint = _pair(action, external_action, player)
        next_state = batched_step_sync(current_states, joint, events, tables)
        return next_state, current_carry.controller, _zero_diagnostics(
            current_states.step.shape[0]
        ), joint

    def normal_branch(value):
        current_states, current_carry = value
        return fulfillment_step_v1(
            current_states,
            current_carry.controller,
            selected_plan,
            external_action,
            player,
            events,
            tables,
        )

    next_state, next_controller, diagnostics, joint = jax.lax.cond(
        states.step[0] == 0,
        opening_branch,
        normal_branch,
        (states, next_carry),
    )
    next_carry = next_carry._replace(controller=next_controller)
    return next_state, next_carry, diagnostics, joint


__all__ = [
    "HasegawaCarryV1",
    "hasegawa_step_with_external_v1",
    "initialize_hasegawa_carry_v1",
    "select_hasegawa_route_v1",
]
