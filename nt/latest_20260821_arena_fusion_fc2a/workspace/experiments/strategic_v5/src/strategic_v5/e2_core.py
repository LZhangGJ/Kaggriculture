"""E2 Land-core and Fertilizer-core strategic implementation.

The physics remain in :mod:`kaggriculture_jax.simulator`.  This module adds
the missing V5 strategic layer: candidates, exact feasibility, reservations,
persistent unit/market tasks, compilation, and effect-checked completion.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from kaggriculture_jax.constants import (
    BOARD_SIZE,
    EPISODE_STEPS,
    LAND_PRICES,
    MARKET_MIN_INVENTORY,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_CROPS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    SHED_ACCESS,
    SHED_CAPACITY,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.types import Action, State, StaticTables

from .constants import (
    CandidateSourceV1,
    FailureCodeV1,
    MAX_CANDIDATES_V1,
    MAX_SELECTIONS_V1,
    TaskPhaseV1,
    TaskStatusV1,
    TaskTypeV1,
)
from .geometry import movement_op_toward_v1, nearest_shed_access_v1
from .schema import (
    CandidateV1,
    CompileDiagnosticsV1,
    ControllerStateV1,
    EffectDiagnosticsV1,
    FeasibilityV1,
    LedgerV1,
)


FERTILIZER_ITEM = 8
_LAND_PRICES = jnp.asarray(LAND_PRICES, dtype=jnp.int32)
_TILE_X = jnp.tile(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
_TILE_Y = jnp.repeat(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)
_INVALID_DISTANCE = jnp.int32(1_000_000)


class E2SelectionV1(NamedTuple):
    controller: ControllerStateV1
    ledger: LedgerV1
    selected_candidate_indices: jax.Array
    masks: jax.Array
    internal_resource_conflict: jax.Array


class E2PlayerActionV1(NamedTuple):
    unit_op: jax.Array
    unit_item: jax.Array
    unit_amount: jax.Array
    unit_count: jax.Array
    market_op: jax.Array
    market_item: jax.Array
    market_amount: jax.Array
    market_count: jax.Array
    diagnostics: CompileDiagnosticsV1


def _empty_candidates(batch_size: int) -> CandidateV1:
    shape = (batch_size, MAX_CANDIDATES_V1)
    i8 = lambda value=0: jnp.full(shape, value, dtype=jnp.int8)
    i16 = lambda value=0: jnp.full(shape, value, dtype=jnp.int16)
    boolean = lambda value=False: jnp.full(shape, value, dtype=jnp.bool_)
    return CandidateV1(
        task_type=i8(),
        owner_unit=i8(-1),
        target_id=i16(-1),
        target_x=i8(-1),
        target_y=i8(-1),
        item_id=i8(-1),
        quantity=i16(),
        source=i8(),
        mandatory=boolean(),
        present=boolean(),
        hard_mask=boolean(),
        source_slot=jnp.broadcast_to(
            jnp.arange(MAX_CANDIDATES_V1, dtype=jnp.int16), shape
        ),
        replay_priority=jnp.zeros(shape, dtype=jnp.float32),
    )


def _empty_feasibility(batch_size: int) -> FeasibilityV1:
    shape = (batch_size, MAX_CANDIDATES_V1)
    i8 = lambda value=0: jnp.full(shape, value, dtype=jnp.int8)
    i16 = lambda value=0: jnp.full(shape, value, dtype=jnp.int16)
    i32 = lambda value=0: jnp.full(shape, value, dtype=jnp.int32)
    boolean = lambda value=False: jnp.full(shape, value, dtype=jnp.bool_)
    return FeasibilityV1(
        legal_now=boolean(),
        unit_required=boolean(),
        plot_required=boolean(),
        path_steps=i16(),
        operation_steps=i16(),
        expected_finish_step=i16(-1),
        deadline_step=i16(EPISODE_STEPS - 2),
        bankable_before_terminal=boolean(),
        cash_required=i32(),
        seed_item=i8(-1),
        seed_required=i16(),
        shed_item=i8(-1),
        shed_item_required=i16(),
        unit_item=i8(-1),
        unit_item_required=i16(),
        shed_reserved_in=i16(),
        market_slots_required=i8(),
        land_purchases_required=i8(),
    )


def _buy_quote(states: State, tables: StaticTables, item: int) -> jax.Array:
    inventory_after_buy = states.market_inventory[:, item].astype(jnp.int32) - 1
    index = jnp.clip(
        inventory_after_buy - MARKET_MIN_INVENTORY,
        0,
        tables.market_price.shape[1] - 1,
    )
    return tables.market_price[item, index].astype(jnp.int32)


def build_e2_candidates_v1(
    states: State,
    controller: ControllerStateV1,
    tables: StaticTables,
    player: int,
) -> CandidateV1:
    """Generate one land, one fertilizer-buy, and one apply target per unit."""

    batch_size = states.step.shape[0]
    candidates = _empty_candidates(batch_size)
    terminal = states.step >= EPISODE_STEPS - 2
    free_market_slot = jnp.any(
        controller.market_tasks.status != TaskStatusV1.ACTIVE, axis=-1
    )
    shed_used = jnp.sum(states.shed[:, player].astype(jnp.int32), axis=-1)

    unlocked = states.unlocked_count[:, player].astype(jnp.int32)
    land_index = jnp.clip(unlocked - 1, 0, len(LAND_PRICES) - 1)
    land_cost = _LAND_PRICES[land_index]
    land_present = (
        (unlocked < 4)
        & (states.money[:, player] >= land_cost)
        & free_market_slot
        & (~terminal)
    )
    candidates = candidates._replace(
        task_type=candidates.task_type.at[:, 0].set(jnp.int8(TaskTypeV1.BUY_LAND)),
        quantity=candidates.quantity.at[:, 0].set(jnp.int16(1)),
        source=candidates.source.at[:, 0].set(jnp.int8(CandidateSourceV1.LAND)),
        present=candidates.present.at[:, 0].set(land_present),
        hard_mask=candidates.hard_mask.at[:, 0].set(land_present),
    )

    plant_exists = jnp.any(states.tile_kind[:, player] == TileKind.PLANT, axis=(1, 2))
    buy_price = _buy_quote(states, tables, FERTILIZER_ITEM)
    buy_present = (
        plant_exists
        & (shed_used < SHED_CAPACITY)
        & (states.money[:, player] >= buy_price)
        & free_market_slot
        & (~terminal)
    )
    candidates = candidates._replace(
        task_type=candidates.task_type.at[:, 1].set(jnp.int8(TaskTypeV1.BUY_PRODUCT)),
        item_id=candidates.item_id.at[:, 1].set(jnp.int8(FERTILIZER_ITEM)),
        quantity=candidates.quantity.at[:, 1].set(jnp.int16(1)),
        source=candidates.source.at[:, 1].set(
            jnp.int8(CandidateSourceV1.FERTILIZER_BUY)
        ),
        present=candidates.present.at[:, 1].set(buy_present),
        hard_mask=candidates.hard_mask.at[:, 1].set(buy_present),
    )

    unit_active = states.unit_active[:, player]
    unit_free = unit_active & (controller.unit_tasks.status != TaskStatusV1.ACTIVE)
    positions = states.unit_pos[:, player].astype(jnp.int16)
    kinds = states.tile_kind[:, player].reshape(batch_size, -1)
    fertilized_until = states.tile_fertilized_until[:, player].reshape(batch_size, -1)
    day = (states.step // 24).astype(jnp.int16)
    useful_plant = (kinds == TileKind.PLANT) & (
        fertilized_until.astype(jnp.int16) < (day[:, None] + 2)
    )
    targets = jnp.stack((_TILE_X, _TILE_Y), axis=-1)
    distance = jnp.sum(
        jnp.abs(positions[:, :, None, :] - targets[None, None, :, :]), axis=-1
    ).astype(jnp.int32)
    key = jnp.where(
        useful_plant[:, None, :], distance * 100 + jnp.arange(100), _INVALID_DISTANCE
    )
    nearest_index = jnp.argmin(key, axis=-1)
    nearest_key = jnp.take_along_axis(key, nearest_index[..., None], axis=-1)[..., 0]
    target_x = _TILE_X[nearest_index].astype(jnp.int8)
    target_y = _TILE_Y[nearest_index].astype(jnp.int8)
    unit_has = states.unit_inventory[:, player, :, FERTILIZER_ITEM] > 0
    shed_has = states.shed[:, player, FERTILIZER_ITEM] > 0
    resource_exists = unit_has | shed_has[:, None]
    apply_present = unit_free & resource_exists & (nearest_key < _INVALID_DISTANCE) & (~terminal[:, None])
    slots = jnp.arange(MAX_UNITS, dtype=jnp.int32) + 2
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    candidates = candidates._replace(
        task_type=candidates.task_type.at[batch, slots].set(
            jnp.full((batch_size, MAX_UNITS), TaskTypeV1.APPLY_FERTILIZER, dtype=jnp.int8)
        ),
        owner_unit=candidates.owner_unit.at[batch, slots].set(
            jnp.broadcast_to(jnp.arange(MAX_UNITS, dtype=jnp.int8), (batch_size, MAX_UNITS))
        ),
        target_id=candidates.target_id.at[batch, slots].set(nearest_index.astype(jnp.int16)),
        target_x=candidates.target_x.at[batch, slots].set(target_x),
        target_y=candidates.target_y.at[batch, slots].set(target_y),
        item_id=candidates.item_id.at[batch, slots].set(jnp.int8(FERTILIZER_ITEM)),
        quantity=candidates.quantity.at[batch, slots].set(jnp.int16(1)),
        source=candidates.source.at[batch, slots].set(
            jnp.full((batch_size, MAX_UNITS), CandidateSourceV1.FERTILIZER_APPLY, dtype=jnp.int8)
        ),
        present=candidates.present.at[batch, slots].set(apply_present),
        hard_mask=candidates.hard_mask.at[batch, slots].set(apply_present),
    )
    return candidates


def evaluate_e2_feasibility_v1(
    states: State,
    candidates: CandidateV1,
    tables: StaticTables,
    player: int,
) -> FeasibilityV1:
    batch_size = states.step.shape[0]
    result = _empty_feasibility(batch_size)
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    task = candidates.task_type
    is_land = task == TaskTypeV1.BUY_LAND
    is_buy = (task == TaskTypeV1.BUY_PRODUCT) & (candidates.item_id == FERTILIZER_ITEM)
    is_apply = task == TaskTypeV1.APPLY_FERTILIZER
    supported = is_land | is_buy | is_apply
    owner = jnp.clip(candidates.owner_unit.astype(jnp.int32), 0, MAX_UNITS - 1)
    x = jnp.clip(candidates.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(candidates.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    position = states.unit_pos[batch, player, owner].astype(jnp.int16)
    target = jnp.stack((x, y), axis=-1).astype(jnp.int16)
    direct_distance = jnp.sum(jnp.abs(position - target), axis=-1).astype(jnp.int16)
    via = (
        jnp.sum(jnp.abs(position[..., None, :] - _SHED_ACCESS), axis=-1)
        + jnp.sum(jnp.abs(_SHED_ACCESS - target[..., None, :]), axis=-1)
    )
    via_distance = jnp.min(via, axis=-1).astype(jnp.int16)
    unit_fertilizer = states.unit_inventory[batch, player, owner, FERTILIZER_ITEM]
    shed_fertilizer = states.shed[:, player, FERTILIZER_ITEM][:, None]
    pickup_required = is_apply & (unit_fertilizer <= 0)
    path_steps = jnp.where(
        is_apply,
        jnp.where(pickup_required, via_distance, direct_distance),
        0,
    ).astype(jnp.int16)
    operation_steps = jnp.where(is_apply, 1 + pickup_required.astype(jnp.int16), 1).astype(jnp.int16)
    expected_finish = (
        states.step[:, None].astype(jnp.int16) + path_steps + operation_steps
    ).astype(jnp.int16)
    terminal_step = jnp.int16(EPISODE_STEPS - 2)
    bankable = expected_finish <= terminal_step
    day_end_step = (
        ((states.step[:, None].astype(jnp.int16) // 24) + 1) * 24
    ).astype(jnp.int16)
    # Every unit returns to its spawn at day end and hands disappear.  An E2
    # route selected now must therefore finish within the current workday;
    # otherwise it is regenerated from the next day's actual state.
    finishes_before_day_reset = expected_finish <= day_end_step

    unlocked = states.unlocked_count[:, player].astype(jnp.int32)[:, None]
    land_index = jnp.clip(unlocked - 1, 0, len(LAND_PRICES) - 1)
    land_cost = _LAND_PRICES[land_index]
    buy_price = _buy_quote(states, tables, FERTILIZER_ITEM)[:, None]
    cash_required = jnp.where(is_land, land_cost, jnp.where(is_buy, buy_price, 0)).astype(jnp.int32)
    target_plant = states.tile_kind[batch, player, y, x] == TileKind.PLANT
    unit_active = states.unit_active[batch, player, owner]
    shed_used = jnp.sum(states.shed[:, player].astype(jnp.int32), axis=-1)[:, None]
    land_ok = is_land & (unlocked < 4) & (states.money[:, player, None] >= land_cost)
    buy_ok = (
        is_buy
        & (candidates.quantity > 0)
        & (shed_used + candidates.quantity <= SHED_CAPACITY)
        & (states.money[:, player, None] >= buy_price)
    )
    apply_ok = (
        is_apply
        & unit_active
        & target_plant
        & ((unit_fertilizer > 0) | (shed_fertilizer > 0))
        & finishes_before_day_reset
    )
    legal = candidates.present & candidates.hard_mask & supported & bankable & (land_ok | buy_ok | apply_ok)
    return result._replace(
        legal_now=legal,
        unit_required=is_apply,
        plot_required=is_apply,
        path_steps=path_steps,
        operation_steps=operation_steps,
        expected_finish_step=expected_finish,
        deadline_step=jnp.where(is_apply, day_end_step, terminal_step).astype(jnp.int16),
        bankable_before_terminal=bankable,
        cash_required=cash_required,
        shed_item=jnp.where(is_apply | is_buy, FERTILIZER_ITEM, -1).astype(jnp.int8),
        shed_item_required=(pickup_required & is_apply).astype(jnp.int16),
        unit_item=jnp.where(is_apply, FERTILIZER_ITEM, -1).astype(jnp.int8),
        unit_item_required=(is_apply & (~pickup_required)).astype(jnp.int16),
        shed_reserved_in=jnp.where(is_buy, candidates.quantity, 0).astype(jnp.int16),
        market_slots_required=(is_land | is_buy).astype(jnp.int8),
        land_purchases_required=is_land.astype(jnp.int8),
    )


def initialize_e2_ledger_v1(
    states: State, controller: ControllerStateV1, player: int
) -> LedgerV1:
    batch_size = states.step.shape[0]
    active_units = controller.unit_tasks.status == TaskStatusV1.ACTIVE
    active_market = controller.market_tasks.status == TaskStatusV1.ACTIVE
    unit_free = states.unit_active[:, player] & (~active_units)
    unit_tasks = controller.unit_tasks
    task = unit_tasks.task_type
    plot_task = active_units & (
        (task == TaskTypeV1.APPLY_FERTILIZER)
        | (task == TaskTypeV1.CROP_PRODUCTION)
        | (task == TaskTypeV1.WATER_CROP)
        | (task == TaskTypeV1.CLEAR_OR_REMOVE_TILE)
        | (task == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
        | (task == TaskTypeV1.ANIMAL_PLACE)
        | (task == TaskTypeV1.ANIMAL_FEED)
        | (task == TaskTypeV1.ANIMAL_CARE)
        | (task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
        | (task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
    )
    x = jnp.clip(unit_tasks.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(unit_tasks.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    plot_reserved = jnp.zeros((batch_size, BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    unit = jnp.arange(MAX_UNITS, dtype=jnp.int32)[None, :]
    plot_reserved = plot_reserved.at[batch, y, x].max(plot_task)
    target_kind = states.tile_kind[batch, player, y, x]
    target_yield = states.tile_yield[batch, player, y, x].astype(jnp.int16)
    target_crop = jnp.clip(
        states.tile_crop[batch, player, y, x].astype(jnp.int32), 0, NUM_CROPS - 1
    )
    target_animal = jnp.clip(
        states.tile_animal[batch, player, y, x].astype(jnp.int32), 0, 2
    )
    task_item = jnp.clip(unit_tasks.item_id.astype(jnp.int32), 0, NUM_SHED_ITEMS - 1)
    task_quantity = jnp.maximum(unit_tasks.quantity.astype(jnp.int16), 1)
    unit_item_count = states.unit_inventory[batch, player, unit, task_item]
    needs_item = active_units & (
        (task == TaskTypeV1.APPLY_FERTILIZER)
        | (task == TaskTypeV1.ANIMAL_PLACE)
        | (task == TaskTypeV1.ANIMAL_FEED)
        | (task == TaskTypeV1.SHED_PICKUP)
    )
    pickup_required = needs_item & (unit_item_count <= 0)
    shed_reserved_out = jnp.zeros(
        (batch_size, NUM_SHED_ITEMS), dtype=jnp.int16
    )
    shed_reserved_out = shed_reserved_out.at[batch, task_item].add(
        jnp.where(pickup_required, task_quantity, 0).astype(jnp.int16)
    )
    unit_inventory_reserved = jnp.zeros(
        (batch_size, MAX_UNITS, NUM_SHED_ITEMS), dtype=jnp.int16
    )
    unit_inventory_reserved = unit_inventory_reserved.at[
        batch, unit, task_item
    ].add(jnp.where(needs_item & (~pickup_required), task_quantity, 0).astype(jnp.int16))

    active_plant = (
        active_units
        & (task == TaskTypeV1.CROP_PRODUCTION)
        & (unit_tasks.phase != TaskPhaseV1.MOVE_TO_DEPOT)
        & (target_kind == TileKind.EMPTY)
    )
    seed_item = jnp.clip(unit_tasks.item_id.astype(jnp.int32), 0, NUM_CROPS - 1)
    seeds_reserved = jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int16)
    seeds_reserved = seeds_reserved.at[batch, seed_item].add(
        active_plant.astype(jnp.int16)
    )

    # Active return routes keep their future shed capacity reservation across
    # environment steps.  Before harvest, reserve the target yield; after
    # harvest, reserve the actual carried inventory by item.
    returning = active_units & (
        (unit_tasks.phase == TaskPhaseV1.MOVE_TO_DEPOT)
        | (task == TaskTypeV1.SAFE_RECOVERY)
        | (task == TaskTypeV1.SHED_DEPOSIT)
    )
    shed_reserved_in = jnp.sum(
        states.unit_inventory[:, player].astype(jnp.int16)
        * returning[..., None].astype(jnp.int16),
        axis=1,
        dtype=jnp.int16,
    )
    crop_future = (
        active_units
        & (task == TaskTypeV1.CROP_PRODUCTION)
        & (unit_tasks.phase == TaskPhaseV1.MOVE_TO_TARGET)
        & (target_kind == TileKind.PLANT)
        & (target_yield > 0)
    )
    animal_future = active_units & (
        task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT
    ) & (target_yield > 0)
    animal_product = jnp.asarray((5, 6, 7), dtype=jnp.int32)[target_animal]
    future_item = jnp.where(crop_future, target_crop, animal_product)
    future_quantity = jnp.where(
        crop_future | animal_future, target_yield, 0
    ).astype(jnp.int16)
    shed_reserved_in = shed_reserved_in.at[batch, future_item].add(future_quantity)
    collect_future = active_units & (
        task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER
    )
    shed_reserved_in = shed_reserved_in.at[:, FERTILIZER_ITEM].add(
        jnp.sum(collect_future, axis=-1, dtype=jnp.int16)
    )

    market_tasks = controller.market_tasks
    market_item = jnp.clip(
        market_tasks.item_id.astype(jnp.int32), 0, NUM_SHED_ITEMS - 1
    )
    market_buy_in = active_market & (
        (market_tasks.task_type == TaskTypeV1.BUY_PRODUCT)
        | (market_tasks.task_type == TaskTypeV1.ANIMAL_PURCHASE)
    )
    shed_reserved_in = shed_reserved_in.at[batch, market_item].add(
        jnp.where(market_buy_in, market_tasks.quantity, 0).astype(jnp.int16)
    )
    market_sell = active_market & (
        (market_tasks.task_type == TaskTypeV1.SELL_INVENTORY)
        | (market_tasks.task_type == TaskTypeV1.TERMINAL_LIQUIDATION)
    )
    shed_reserved_out = shed_reserved_out.at[batch, market_item].add(
        jnp.where(market_sell, market_tasks.quantity, 0).astype(jnp.int16)
    )
    market_slots = jnp.sum(active_market, axis=-1, dtype=jnp.int8)
    land_reserved = jnp.sum(
        active_market & (controller.market_tasks.task_type == TaskTypeV1.BUY_LAND),
        axis=-1,
        dtype=jnp.int8,
    )
    return LedgerV1(
        unit_free=unit_free,
        plot_reserved=plot_reserved,
        cash_available=states.money[:, player],
        cash_reserved=jnp.zeros((batch_size,), dtype=jnp.int32),
        seeds_available=states.seeds[:, player],
        seeds_reserved=seeds_reserved,
        shed_available=states.shed[:, player],
        shed_reserved_out=shed_reserved_out,
        shed_reserved_in=shed_reserved_in,
        unit_inventory_available=states.unit_inventory[:, player],
        unit_inventory_reserved=unit_inventory_reserved,
        market_slots_used=market_slots,
        market_product_buy_reserved=jnp.zeros(
            (batch_size, NUM_PRODUCTS), dtype=jnp.bool_
        ).at[
            batch,
            jnp.clip(market_tasks.item_id.astype(jnp.int32), 0, NUM_PRODUCTS - 1),
        ].max(active_market & (market_tasks.task_type == TaskTypeV1.BUY_PRODUCT)),
        market_product_sell_reserved=jnp.zeros(
            (batch_size, NUM_PRODUCTS), dtype=jnp.bool_
        ).at[
            batch,
            jnp.clip(market_tasks.item_id.astype(jnp.int32), 0, NUM_PRODUCTS - 1),
        ].max(market_sell),
        land_purchases_reserved=land_reserved,
        maintenance_reserved=plot_reserved,
    )


def candidate_mask_from_e2_ledger_v1(
    states: State,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    ledger: LedgerV1,
    player: int,
) -> jax.Array:
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    owner = jnp.clip(candidates.owner_unit.astype(jnp.int32), 0, MAX_UNITS - 1)
    x = jnp.clip(candidates.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(candidates.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    shed_item = jnp.clip(feasibility.shed_item.astype(jnp.int32), 0, NUM_SHED_ITEMS - 1)
    unit_item = jnp.clip(feasibility.unit_item.astype(jnp.int32), 0, NUM_SHED_ITEMS - 1)
    seed_item = jnp.clip(feasibility.seed_item.astype(jnp.int32), 0, NUM_CROPS - 1)
    market_product = jnp.clip(candidates.item_id.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    is_product_buy = candidates.task_type == TaskTypeV1.BUY_PRODUCT
    is_product_sell = (
        (candidates.task_type == TaskTypeV1.SELL_INVENTORY)
        | (candidates.task_type == TaskTypeV1.TERMINAL_LIQUIDATION)
    )
    unit_ok = (~feasibility.unit_required) | ledger.unit_free[batch, owner]
    plot_ok = (~feasibility.plot_required) | (~ledger.plot_reserved[batch, y, x])
    cash_ok = (
        ledger.cash_available[:, None] - ledger.cash_reserved[:, None]
        >= feasibility.cash_required
    )
    shed_item_ok = (
        ledger.shed_available[batch, shed_item]
        - ledger.shed_reserved_out[batch, shed_item]
        >= feasibility.shed_item_required
    )
    unit_item_ok = (
        ledger.unit_inventory_available[batch, owner, unit_item]
        - ledger.unit_inventory_reserved[batch, owner, unit_item]
        >= feasibility.unit_item_required
    )
    seed_ok = (
        ledger.seeds_available[batch, seed_item]
        - ledger.seeds_reserved[batch, seed_item]
        >= feasibility.seed_required
    )
    shed_used = jnp.sum(ledger.shed_available.astype(jnp.int32), axis=-1)[:, None]
    shed_in = jnp.sum(ledger.shed_reserved_in.astype(jnp.int32), axis=-1)[:, None]
    shed_out = jnp.sum(ledger.shed_reserved_out.astype(jnp.int32), axis=-1)[:, None]
    capacity_ok = shed_used + shed_in - shed_out + feasibility.shed_reserved_in <= SHED_CAPACITY
    market_ok = (
        ledger.market_slots_used[:, None] + feasibility.market_slots_required
        <= MAX_MARKET_ORDERS
    )
    market_direction_ok = (
        ((~is_product_buy) | (~ledger.market_product_sell_reserved[batch, market_product]))
        & ((~is_product_sell) | (~ledger.market_product_buy_reserved[batch, market_product]))
    )
    land_ok = (
        states.unlocked_count[:, player, None].astype(jnp.int16)
        + ledger.land_purchases_reserved[:, None].astype(jnp.int16)
        + feasibility.land_purchases_required.astype(jnp.int16)
        <= 4
    )
    return (
        candidates.present
        & candidates.hard_mask
        & feasibility.legal_now
        & unit_ok
        & plot_ok
        & cash_ok
        & shed_item_ok
        & unit_item_ok
        & seed_ok
        & capacity_ok
        & market_ok
        & market_direction_ok
        & land_ok
    )


def _reserve_one_v1(
    ledger: LedgerV1,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    selected: jax.Array,
) -> LedgerV1:
    chosen = selected >= 0
    safe = jnp.clip(selected, 0, MAX_CANDIDATES_V1 - 1)
    owner = jnp.clip(candidates.owner_unit[safe].astype(jnp.int32), 0, MAX_UNITS - 1)
    x = jnp.clip(candidates.target_x[safe].astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(candidates.target_y[safe].astype(jnp.int32), 0, BOARD_SIZE - 1)
    shed_item = jnp.clip(feasibility.shed_item[safe].astype(jnp.int32), 0, NUM_SHED_ITEMS - 1)
    unit_item = jnp.clip(feasibility.unit_item[safe].astype(jnp.int32), 0, NUM_SHED_ITEMS - 1)
    seed_item = jnp.clip(feasibility.seed_item[safe].astype(jnp.int32), 0, NUM_CROPS - 1)
    market_product = jnp.clip(candidates.item_id[safe].astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    is_product_buy = candidates.task_type[safe] == TaskTypeV1.BUY_PRODUCT
    is_product_sell = (
        (candidates.task_type[safe] == TaskTypeV1.SELL_INVENTORY)
        | (candidates.task_type[safe] == TaskTypeV1.TERMINAL_LIQUIDATION)
    )

    def reserve(value: LedgerV1) -> LedgerV1:
        value = value._replace(
            unit_free=value.unit_free.at[owner].set(
                value.unit_free[owner] & (~feasibility.unit_required[safe])
            ),
            plot_reserved=value.plot_reserved.at[y, x].set(
                value.plot_reserved[y, x] | feasibility.plot_required[safe]
            ),
            maintenance_reserved=value.maintenance_reserved.at[y, x].set(
                value.maintenance_reserved[y, x] | feasibility.plot_required[safe]
            ),
            cash_reserved=value.cash_reserved + feasibility.cash_required[safe],
            shed_reserved_out=value.shed_reserved_out.at[shed_item].add(
                feasibility.shed_item_required[safe]
            ),
            shed_reserved_in=value.shed_reserved_in.at[shed_item].add(
                feasibility.shed_reserved_in[safe]
            ),
            unit_inventory_reserved=value.unit_inventory_reserved.at[owner, unit_item].add(
                feasibility.unit_item_required[safe]
            ),
            seeds_reserved=value.seeds_reserved.at[seed_item].add(
                feasibility.seed_required[safe]
            ),
            market_slots_used=(
                value.market_slots_used + feasibility.market_slots_required[safe]
            ).astype(jnp.int8),
            market_product_buy_reserved=value.market_product_buy_reserved.at[
                market_product
            ].set(
                value.market_product_buy_reserved[market_product] | is_product_buy
            ),
            market_product_sell_reserved=value.market_product_sell_reserved.at[
                market_product
            ].set(
                value.market_product_sell_reserved[market_product] | is_product_sell
            ),
            land_purchases_reserved=(
                value.land_purchases_reserved + feasibility.land_purchases_required[safe]
            ).astype(jnp.int8),
        )
        return value

    return lax.cond(chosen, reserve, lambda value: value, ledger)


def reserve_e2_candidate_batched_v1(
    ledger: LedgerV1,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    selected: jax.Array,
) -> LedgerV1:
    return jax.vmap(_reserve_one_v1)(ledger, candidates, feasibility, selected)


def _attach_one_v1(
    controller: ControllerStateV1,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    selected: jax.Array,
    current_step: jax.Array,
) -> ControllerStateV1:
    chosen = selected >= 0
    safe = jnp.clip(selected, 0, MAX_CANDIDATES_V1 - 1)
    task_type = candidates.task_type[safe]
    is_market = (candidates.owner_unit[safe] < 0) | (
        (task_type == TaskTypeV1.BUY_LAND)
        | (task_type == TaskTypeV1.BUY_PRODUCT)
        | (task_type == TaskTypeV1.ANIMAL_PURCHASE)
        | (task_type == TaskTypeV1.HIRE_WORKER)
        | (task_type == TaskTypeV1.SELL_INVENTORY)
        | (task_type == TaskTypeV1.TERMINAL_LIQUIDATION)
    )
    owner = jnp.clip(candidates.owner_unit[safe].astype(jnp.int32), 0, MAX_UNITS - 1)

    def attach_unit(value: ControllerStateV1) -> ControllerStateV1:
        needs_shed_pickup = feasibility.shed_item_required[safe] > 0
        phase = jnp.where(
            needs_shed_pickup, TaskPhaseV1.MOVE_TO_SHED, TaskPhaseV1.MOVE_TO_TARGET
        ).astype(jnp.int8)
        tasks = value.unit_tasks._replace(
            task_type=value.unit_tasks.task_type.at[owner].set(task_type),
            target_id=value.unit_tasks.target_id.at[owner].set(candidates.target_id[safe]),
            target_x=value.unit_tasks.target_x.at[owner].set(candidates.target_x[safe]),
            target_y=value.unit_tasks.target_y.at[owner].set(candidates.target_y[safe]),
            item_id=value.unit_tasks.item_id.at[owner].set(candidates.item_id[safe]),
            quantity=value.unit_tasks.quantity.at[owner].set(candidates.quantity[safe]),
            phase=value.unit_tasks.phase.at[owner].set(phase),
            start_step=value.unit_tasks.start_step.at[owner].set(current_step),
            last_progress_step=value.unit_tasks.last_progress_step.at[owner].set(current_step),
            expected_finish_step=value.unit_tasks.expected_finish_step.at[owner].set(
                feasibility.expected_finish_step[safe]
            ),
            deadline_step=value.unit_tasks.deadline_step.at[owner].set(
                feasibility.deadline_step[safe]
            ),
            status=value.unit_tasks.status.at[owner].set(TaskStatusV1.ACTIVE),
            failure_code=value.unit_tasks.failure_code.at[owner].set(FailureCodeV1.NONE),
        )
        return value._replace(unit_tasks=tasks)

    def attach_market(value: ControllerStateV1) -> ControllerStateV1:
        free = value.market_tasks.status != TaskStatusV1.ACTIVE
        slots_required = jnp.maximum(
            feasibility.market_slots_required[safe].astype(jnp.int32), 1
        )
        free_rank = jnp.cumsum(free.astype(jnp.int32))
        write = free & (free_rank <= slots_required)
        atomic_quantity = jnp.where(
            task_type == TaskTypeV1.HIRE_WORKER,
            jnp.int16(1),
            candidates.quantity[safe],
        )
        tasks = value.market_tasks._replace(
            task_type=jnp.where(write, task_type, value.market_tasks.task_type),
            item_id=jnp.where(write, candidates.item_id[safe], value.market_tasks.item_id),
            quantity=jnp.where(write, atomic_quantity, value.market_tasks.quantity),
            start_step=jnp.where(write, current_step, value.market_tasks.start_step),
            deadline_step=jnp.where(
                write, feasibility.deadline_step[safe], value.market_tasks.deadline_step
            ),
            status=jnp.where(write, TaskStatusV1.ACTIVE, value.market_tasks.status),
            failure_code=jnp.where(
                write, FailureCodeV1.NONE, value.market_tasks.failure_code
            ),
        )
        return value._replace(market_tasks=tasks)

    def attach(value: ControllerStateV1) -> ControllerStateV1:
        return lax.cond(is_market, attach_market, attach_unit, value)

    return lax.cond(chosen, attach, lambda value: value, controller)


def attach_e2_candidate_batched_v1(
    controller: ControllerStateV1,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    selected: jax.Array,
    current_step: jax.Array,
) -> ControllerStateV1:
    return jax.vmap(_attach_one_v1)(
        controller, candidates, feasibility, selected, current_step
    )


def e2_priority_v1(candidates: CandidateV1, feasibility: FeasibilityV1) -> jax.Array:
    priority = jnp.where(
        candidates.task_type == TaskTypeV1.APPLY_FERTILIZER,
        300.0,
        jnp.where(candidates.task_type == TaskTypeV1.BUY_PRODUCT, 200.0, 100.0),
    )
    score = priority - feasibility.path_steps.astype(jnp.float32)
    return jnp.where(feasibility.legal_now, score, -1.0e9)


def select_e2_candidates_v1(
    states: State,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    controller: ControllerStateV1,
    ledger: LedgerV1,
    player: int,
    max_selections: int = MAX_SELECTIONS_V1,
) -> E2SelectionV1:
    """Greedy deterministic selection with a reservation/remask after each edit."""

    scores = e2_priority_v1(candidates, feasibility)
    return select_candidates_with_scores_v1(
        states,
        candidates,
        feasibility,
        scores,
        controller,
        ledger,
        player,
        max_selections,
    )


def select_candidates_with_scores_v1(
    states: State,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    scores: jax.Array,
    controller: ControllerStateV1,
    ledger: LedgerV1,
    player: int,
    max_selections: int = MAX_SELECTIONS_V1,
    eligible_mask: jax.Array | None = None,
) -> E2SelectionV1:
    """Shared V5 greedy selector; the caller supplies module-specific scores."""

    batch_size = states.step.shape[0]
    selected = jnp.full((batch_size, max_selections), -1, dtype=jnp.int16)
    masks = jnp.zeros((batch_size, max_selections, MAX_CANDIDATES_V1), dtype=jnp.bool_)
    conflicts = jnp.zeros((batch_size,), dtype=jnp.int32)
    if eligible_mask is None:
        eligible_mask = jnp.ones_like(candidates.present)

    def body(index, carry):
        current_controller, current_ledger, choices, saved_masks, conflict = carry
        mask = candidate_mask_from_e2_ledger_v1(
            states, candidates, feasibility, current_ledger, player
        )
        mask = mask & eligible_mask
        candidate_index = jnp.arange(MAX_CANDIDATES_V1, dtype=jnp.int16)
        already_selected = jnp.any(
            choices[:, :, None] == candidate_index[None, None, :], axis=1
        )
        mask = mask & (~already_selected)
        logits = jnp.where(mask, scores, -1.0e9)
        choice = jnp.argmax(logits, axis=-1).astype(jnp.int16)
        has_choice = jnp.any(mask, axis=-1)
        choice = jnp.where(has_choice, choice, -1).astype(jnp.int16)
        repeated = jnp.any(choices == choice[:, None], axis=-1) & (choice >= 0)
        conflict = conflict + repeated.astype(jnp.int32)
        choices = choices.at[:, index].set(choice)
        saved_masks = saved_masks.at[:, index].set(mask)
        current_controller = attach_e2_candidate_batched_v1(
            current_controller, candidates, feasibility, choice, states.step
        )
        current_ledger = reserve_e2_candidate_batched_v1(
            current_ledger, candidates, feasibility, choice
        )
        return current_controller, current_ledger, choices, saved_masks, conflict

    controller, ledger, selected, masks, conflicts = lax.fori_loop(
        0,
        max_selections,
        body,
        (controller, ledger, selected, masks, conflicts),
    )
    return E2SelectionV1(controller, ledger, selected, masks, conflicts)
