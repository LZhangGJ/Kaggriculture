"""Route-conditioned candidate overlay on the accepted V5 executor stack."""

from __future__ import annotations

import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_COST,
    ANIMAL_PRODUCT,
    BOARD_SIZE,
    CROP_SEED_COST,
    CROP_FIRST_YIELD_DAY,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_WATERED,
    HIRE_COST,
    LAND_PRICES,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_DAYS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    SHED_CAPACITY,
    TURNS_PER_DAY,
    TileKind,
)
from kaggriculture_jax.types import State
from strategic_v5.constants import TaskTypeV1
from strategic_v5.e4_core import (
    BUILD_COOP_SLOT,
    BUILD_PASTURE_SLOT,
    MARKET_SLOT_COUNT,
    PLANT_START,
    PRIMARY_UNIT_START,
    SECONDARY_UNIT_COUNT,
    SECONDARY_UNIT_START,
)
from strategic_v5.constants import CandidateSourceV1
from strategic_v5.schema import CandidateV1

from .schema import FertilizerPolicyV1, RouteScheduleV1


FERTILIZER_ITEM = 8
WHEAT_ITEM = 0
LAND_SLOT = 0
FERTILIZER_BUY_SLOT = 1
WHEAT_BUY_SLOT = 2
ANIMAL_BUY_START = 3
SEED_BUY_START = ANIMAL_BUY_START + NUM_ANIMALS
HIRE_SLOT = SEED_BUY_START + NUM_CROPS
SELL_START = HIRE_SLOT + 1
CAPACITY_PRESSURE = 85

# Gold A-family pasture cluster around the fixed shed access.  The thirteenth
# slot is a late spare/replacement position observed in the accepted trace.
_PASTURE_X = jnp.asarray(
    [4, 3, 4, 3, 4, 2, 5, 6, 5, 5, 6, 7, 7], dtype=jnp.int8
)
_PASTURE_Y = jnp.asarray(
    [2, 3, 3, 4, 4, 4, 4, 4, 2, 3, 3, 4, 3], dtype=jnp.int8
)
_PASTURE_ID = (
    _PASTURE_Y.astype(jnp.int16) * BOARD_SIZE + _PASTURE_X.astype(jnp.int16)
)
_TILE_ID = jnp.arange(BOARD_SIZE * BOARD_SIZE, dtype=jnp.int16)
_TILE_X = (_TILE_ID % BOARD_SIZE).astype(jnp.int8)
_TILE_Y = (_TILE_ID // BOARD_SIZE).astype(jnp.int8)
_TILE_XY = jnp.stack((_TILE_X, _TILE_Y), axis=-1)
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int8)
_ANIMAL_COST = jnp.asarray(ANIMAL_COST, dtype=jnp.int32)
_CROP_SEED_COST = jnp.asarray(CROP_SEED_COST, dtype=jnp.int32)
_HIRE_COST = jnp.asarray(HIRE_COST, dtype=jnp.int32)
_HIRE_COST_PREFIX = jnp.concatenate(
    (jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(_HIRE_COST, dtype=jnp.int32))
)
_LAND_PRICES = jnp.asarray(LAND_PRICES, dtype=jnp.int32)
MAINTENANCE_START = MARKET_SLOT_COUNT
MAINTENANCE_COUNT = PRIMARY_UNIT_START - MARKET_SLOT_COUNT

assert SELL_START + NUM_PRODUCTS == MARKET_SLOT_COUNT
assert MAINTENANCE_COUNT == 9


def _daily_targets(
    states: State, schedule: RouteScheduleV1
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    day = jnp.clip(states.step.astype(jnp.int32) // TURNS_PER_DAY, 0, NUM_DAYS - 1)
    return (
        schedule.crop_target_by_day[batch, day],
        schedule.animal_target_by_day[batch, day],
        schedule.land_target_by_day[batch, day],
        schedule.hire_target_by_day[batch, day],
    )


def _owned_counts(
    states: State, player: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    tile_kind = states.tile_kind[:, player]
    tile_crop = states.tile_crop[:, player]
    tile_animal = states.tile_animal[:, player]
    crop_ids = jnp.arange(NUM_CROPS, dtype=jnp.int8)
    animal_ids = jnp.arange(NUM_ANIMALS, dtype=jnp.int8)
    crop_count = jnp.sum(
        (tile_kind[..., None] == TileKind.PLANT)
        & (tile_crop[..., None] == crop_ids),
        axis=(1, 2),
        dtype=jnp.int16,
    )
    placed_animals = jnp.sum(
        tile_animal[..., None] == animal_ids,
        axis=(1, 2),
        dtype=jnp.int16,
    )
    shed_animals = states.shed[
        :, player, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
    ].astype(jnp.int16)
    carried_animals = jnp.sum(
        states.unit_inventory[
            :, player, :, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
        ].astype(jnp.int16),
        axis=1,
        dtype=jnp.int16,
    )
    animal_total = placed_animals + shed_animals + carried_animals
    coop_count = jnp.sum(
        tile_kind == TileKind.COOP, axis=(1, 2), dtype=jnp.int16
    )
    pasture_count = jnp.sum(
        tile_kind == TileKind.PASTURE, axis=(1, 2), dtype=jnp.int16
    )
    return crop_count, animal_total, coop_count, pasture_count


def apply_route_schedule_v1(
    states: State,
    candidates: CandidateV1,
    schedule: RouteScheduleV1,
    player: int,
) -> CandidateV1:
    """Mask generic spending and replace it with batched route commitments."""

    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    enabled = schedule.enabled.astype(jnp.bool_)
    crop_target, animal_target, land_target, hire_target = _daily_targets(
        states, schedule
    )
    crop_count, animal_total, coop_count, pasture_count = _owned_counts(
        states, player
    )
    investing = states.step < schedule.investment_stop_step
    liquidating = states.step >= schedule.liquidation_start_step

    present = candidates.present
    hard_mask = candidates.hard_mask
    mandatory = candidates.mandatory
    task_type = candidates.task_type
    item_id = candidates.item_id
    source = candidates.source
    quantity = candidates.quantity
    priority = jnp.zeros_like(candidates.replay_priority)

    # Route-controlled land expansion.
    land_need = (
        enabled
        & investing
        & (states.unlocked_count[:, player].astype(jnp.int16) < land_target)
    )
    present = present.at[:, LAND_SLOT].set(land_need)
    hard_mask = hard_mask.at[:, LAND_SLOT].set(land_need)
    quantity = quantity.at[:, LAND_SLOT].set(jnp.int16(1))
    priority = priority.at[:, LAND_SLOT].set(jnp.where(land_need, 72.0, 0.0))

    # Feed reserve is route-specific and remains available after investment stops.
    placed_animals = jnp.sum(
        states.tile_animal[:, player] >= 0, axis=(1, 2), dtype=jnp.int16
    )
    wheat_available = (
        states.shed[:, player, WHEAT_ITEM].astype(jnp.int16)
        + jnp.sum(
            states.unit_inventory[:, player, :, WHEAT_ITEM].astype(jnp.int16),
            axis=1,
            dtype=jnp.int16,
        )
    )
    wanted_wheat = placed_animals * jnp.maximum(
        schedule.feed_reserve_days.astype(jnp.int16), 1
    )
    wheat_deficit = jnp.maximum(wanted_wheat - wheat_available, 0).astype(jnp.int16)
    wheat_need = enabled & (~liquidating) & (wheat_deficit > 0)
    present = present.at[:, WHEAT_BUY_SLOT].set(wheat_need)
    hard_mask = hard_mask.at[:, WHEAT_BUY_SLOT].set(wheat_need)
    quantity = quantity.at[:, WHEAT_BUY_SLOT].set(wheat_deficit)
    priority = priority.at[:, WHEAT_BUY_SLOT].set(
        jnp.where(wheat_need, 98.0, 0.0)
    )

    # Fertilizer purchases are allowed only for APPLY/MIXED routes.
    fertilizer_mode = schedule.fertilizer_policy
    fertilizer_buy = (
        enabled
        & (~liquidating)
        & (jnp.sum(crop_target.astype(jnp.int16), axis=-1) > 0)
        & (fertilizer_mode == FertilizerPolicyV1.APPLY)
        & candidates.present[:, FERTILIZER_BUY_SLOT]
    )
    present = present.at[:, FERTILIZER_BUY_SLOT].set(fertilizer_buy)
    hard_mask = hard_mask.at[:, FERTILIZER_BUY_SLOT].set(fertilizer_buy)
    priority = priority.at[:, FERTILIZER_BUY_SLOT].set(
        jnp.where(fertilizer_buy, 52.0, 0.0)
    )

    # Animal purchases are one-at-a-time into already available structures.
    animal_deficit = jnp.maximum(animal_target.astype(jnp.int16) - animal_total, 0)
    animal_slots = ANIMAL_BUY_START + jnp.arange(NUM_ANIMALS, dtype=jnp.int32)
    animal_structural = candidates.target_id[batch[:, None], animal_slots] >= 0
    animal_need = enabled[:, None] & investing[:, None] & (animal_deficit > 0)
    animal_need &= animal_structural
    present = present.at[batch[:, None], animal_slots].set(animal_need)
    hard_mask = hard_mask.at[batch[:, None], animal_slots].set(animal_need)
    quantity = quantity.at[batch[:, None], animal_slots].set(
        jnp.ones_like(animal_deficit, dtype=jnp.int16)
    )
    priority = priority.at[batch[:, None], animal_slots].set(
        jnp.where(animal_need, 78.0, 0.0)
    )

    # Buy enough seed stock to fill the current crop target, in bounded batches.
    seed_stock = states.seeds[:, player].astype(jnp.int16)
    seed_deficit = jnp.maximum(
        crop_target.astype(jnp.int16) - crop_count - seed_stock, 0
    )
    seed_quantity = jnp.minimum(
        seed_deficit,
        jnp.maximum(schedule.seed_batch.astype(jnp.int16), 1),
    )
    seed_need = enabled[:, None] & (~liquidating[:, None]) & (seed_quantity > 0)
    seed_slots = SEED_BUY_START + jnp.arange(NUM_CROPS, dtype=jnp.int32)
    present = present.at[batch[:, None], seed_slots].set(seed_need)
    hard_mask = hard_mask.at[batch[:, None], seed_slots].set(seed_need)
    quantity = quantity.at[batch[:, None], seed_slots].set(seed_quantity)
    priority = priority.at[batch[:, None], seed_slots].set(
        jnp.where(seed_need, 68.0, 0.0)
    )

    # Buy the affordable part of the daily labor target immediately.  A
    # quantity-N hire candidate compiles to N atomic ordered HIRE cards, up to
    # the ten official market slots.  The previous quantity-one loop reached
    # the same end-of-day hand count as gold but threw away thousands of early
    # worker steps.  Protect the higher-priority feed purchase before sizing
    # this batch; the shared ledger remains the final cash/slot authority.
    hires_so_far = states.hires_today[:, player].astype(jnp.int32)
    hire_deficit = jnp.maximum(
        hire_target.astype(jnp.int32) - hires_so_far, 0
    )
    hire_choices = jnp.arange(1, MAX_MARKET_ORDERS + 1, dtype=jnp.int32)[
        None, :
    ]
    hire_batch_max = jnp.clip(
        schedule.hire_batch_max.astype(jnp.int32), 1, MAX_MARKET_ORDERS
    )[:, None]
    hire_end = jnp.clip(
        hires_so_far[:, None] + hire_choices, 0, len(HIRE_COST)
    )
    hire_batch_cost = (
        _HIRE_COST_PREFIX[hire_end] - _HIRE_COST_PREFIX[hires_so_far[:, None]]
    )
    feed_cash_reserve = (
        wheat_deficit.astype(jnp.int32)
        * states.market_price[:, WHEAT_ITEM].astype(jnp.int32)
    )
    hire_cash_budget = jnp.maximum(
        states.money[:, player].astype(jnp.int32) - feed_cash_reserve, 0
    )
    affordable_hire = (
        (hire_choices <= hire_deficit[:, None])
        & (hire_choices <= hire_batch_max)
        & (hire_batch_cost <= hire_cash_budget[:, None])
    )
    affordable_hire_count = jnp.max(
        jnp.where(affordable_hire, hire_choices, 0), axis=-1
    )
    hire_quantity = jnp.where(
        affordable_hire_count > 0,
        affordable_hire_count,
        jnp.minimum(hire_deficit, 1),
    ).astype(jnp.int16)
    hire_need = enabled & (hire_deficit > 0)
    present = present.at[:, HIRE_SLOT].set(hire_need)
    hard_mask = hard_mask.at[:, HIRE_SLOT].set(hire_need)
    quantity = quantity.at[:, HIRE_SLOT].set(hire_quantity)
    priority = priority.at[:, HIRE_SLOT].set(jnp.where(hire_need, 88.0, 0.0))

    # Only sell on the configured cadence, under capacity pressure, or in liquidation.
    products = jnp.arange(NUM_PRODUCTS, dtype=jnp.int32)
    sell_slots = SELL_START + products
    available = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int16)
    carried_wheat = jnp.sum(
        states.unit_inventory[:, player, :, WHEAT_ITEM].astype(jnp.int16),
        axis=1,
        dtype=jnp.int16,
    )
    wanted_wheat_to_hold = placed_animals * jnp.maximum(
        schedule.feed_sell_reserve_days.astype(jnp.int16), 0
    )
    shed_wheat_reserve = jnp.maximum(
        wanted_wheat_to_hold - carried_wheat, 0
    )
    product_reserve = jnp.zeros_like(available).at[:, WHEAT_ITEM].set(
        jnp.where(liquidating, 0, shed_wheat_reserve)
    )
    sellable = jnp.maximum(available - product_reserve, 0).astype(jnp.int16)
    interval = jnp.maximum(schedule.sell_interval.astype(jnp.int32), 1)
    phase = jnp.mod(schedule.sell_phase.astype(jnp.int32), interval)
    cadence = jnp.mod(states.step.astype(jnp.int32)[:, None], interval) == phase
    price_ok = states.market_price >= schedule.sell_price_floor
    shed_used = jnp.sum(
        states.shed[:, player].astype(jnp.int32), axis=-1
    )
    pressure = shed_used >= CAPACITY_PRESSURE
    investment_need = (
        land_need
        | wheat_need
        | hire_need
        | jnp.any(animal_need, axis=-1)
        | jnp.any(seed_need, axis=-1)
    )
    # Financing is a one-step bridge, not a blanket liquidation rule.  The
    # selector evaluates purchase legality from the pre-sale state, so a sale
    # now can only fund the highest-priority pending purchase on the following
    # step.  Estimate that purchase's exact entry cost, protect feed for the
    # current animal target, and expose only the highest-priced sellable item in
    # the smallest useful quantity.  This prevents sell/buy churn and avoids
    # filling all eight selection slots with simultaneous sale cards.
    hire_index = jnp.clip(
        states.hires_today[:, player].astype(jnp.int32), 0, len(HIRE_COST) - 1
    )
    hire_cost = _HIRE_COST[hire_index]
    land_index = jnp.clip(
        states.unlocked_count[:, player].astype(jnp.int32) - 1,
        0,
        len(LAND_PRICES) - 1,
    )
    land_cost = _LAND_PRICES[land_index]
    animal_cost = jnp.min(
        jnp.where(animal_need, _ANIMAL_COST[None, :], jnp.int32(1_000_000)),
        axis=-1,
    )
    seed_cost = jnp.min(
        jnp.where(
            seed_need,
            seed_quantity.astype(jnp.int32) * _CROP_SEED_COST[None, :],
            jnp.int32(1_000_000),
        ),
        axis=-1,
    )
    next_investment_cost = jnp.where(
        wheat_need,
        states.market_price[:, WHEAT_ITEM].astype(jnp.int32),
        jnp.where(
            hire_need,
            hire_cost,
            jnp.where(
                jnp.any(animal_need, axis=-1),
                animal_cost,
                jnp.where(land_need, land_cost, seed_cost),
            ),
        ),
    )
    financing_target = jnp.maximum(
        next_investment_cost, schedule.financing_cash_floor
    )
    financing_sale = (
        investment_need
        & (schedule.financing_cash_floor > 0)
        & (states.money[:, player] < financing_target)
    )
    target_animals = jnp.sum(animal_target.astype(jnp.int16), axis=-1)
    financing_feed_reserve = jnp.maximum(placed_animals, target_animals) * jnp.maximum(
        schedule.feed_reserve_days.astype(jnp.int16), 1
    )
    financing_wheat_reserve = jnp.maximum(
        financing_feed_reserve - carried_wheat, 0
    ).astype(jnp.int16)
    financing_reserve = jnp.zeros_like(available).at[:, WHEAT_ITEM].set(
        financing_wheat_reserve
    )
    financing_sellable = jnp.maximum(
        available - financing_reserve, 0
    ).astype(jnp.int16)
    financing_value = jnp.where(
        financing_sellable > 0,
        states.market_price.astype(jnp.int32),
        jnp.int32(-1),
    )
    financing_product = jnp.argmax(financing_value, axis=-1).astype(jnp.int32)
    has_financing_product = jnp.max(financing_value, axis=-1) >= 0
    financing_price = jnp.maximum(
        states.market_price[batch, financing_product].astype(jnp.int32), 1
    )
    financing_gap = jnp.maximum(
        financing_target - states.money[:, player], 0
    )
    financing_quantity = jnp.minimum(
        financing_sellable[batch, financing_product].astype(jnp.int32),
        (financing_gap + financing_price - 1) // financing_price,
    ).astype(jnp.int16)
    financing_item = products[None, :] == financing_product[:, None]
    financing_need = (
        financing_sale[:, None]
        & has_financing_product[:, None]
        & financing_item
        & (financing_quantity[:, None] > 0)
    )
    regular_sell_need = (
        enabled[:, None]
        & (sellable > 0)
        & (
            liquidating[:, None]
            | pressure[:, None]
            | (cadence & price_ok)
        )
    )
    sell_need = regular_sell_need | financing_need
    present = present.at[batch[:, None], sell_slots].set(sell_need)
    hard_mask = hard_mask.at[batch[:, None], sell_slots].set(sell_need)
    sell_quantity = jnp.where(
        financing_need,
        financing_quantity[:, None],
        sellable,
    )
    quantity = quantity.at[batch[:, None], sell_slots].set(sell_quantity)
    sell_priority = jnp.where(
        financing_need,
        100.0,
        jnp.where(liquidating[:, None] | pressure[:, None], 96.0, 58.0),
    )
    priority = priority.at[batch[:, None], sell_slots].set(
        jnp.where(sell_need, sell_priority, 0.0)
    )

    # Explicit plant slots follow the current daily target. Retarget planting
    # away from the reserved A-family pasture cluster so crops cannot block the
    # animal logistics layout before its structures are built.
    plant_slots = PLANT_START + jnp.arange(NUM_CROPS, dtype=jnp.int32)
    crop_deficit = jnp.maximum(crop_target.astype(jnp.int16) - crop_count, 0)
    pasture_target = animal_target[:, 1].astype(jnp.int16) + animal_target[
        :, 2
    ].astype(jnp.int16)
    pasture_rank = jnp.arange(_PASTURE_ID.shape[0], dtype=jnp.int16)[None, :]
    reserved_pasture = pasture_rank < jnp.minimum(
        pasture_target[:, None] + 1, _PASTURE_ID.shape[0]
    )
    reserved_tile = jnp.any(
        reserved_pasture[:, :, None]
        & (_PASTURE_ID[None, :, None] == _TILE_ID[None, None, :]),
        axis=1,
    )
    flat_kind = states.tile_kind[:, player].reshape(batch_size, -1)
    plant_owner = jnp.clip(
        candidates.owner_unit[batch[:, None], plant_slots], 0, MAX_UNITS - 1
    ).astype(jnp.int32)
    owner_pos = states.unit_pos[batch[:, None], player, plant_owner]
    plant_distance = jnp.sum(
        jnp.abs(
            owner_pos[:, :, None, :].astype(jnp.int16)
            - _TILE_XY[None, None, :, :].astype(jnp.int16)
        ),
        axis=-1,
    )
    plant_eligible = (flat_kind == TileKind.EMPTY) & (~reserved_tile)
    plant_key = jnp.where(
        plant_eligible[:, None, :],
        plant_distance * 100 + _TILE_ID[None, None, :],
        32767,
    )
    plant_target_id = jnp.argmin(plant_key, axis=-1).astype(jnp.int16)
    plant_target_valid = jnp.min(plant_key, axis=-1) < 32767
    plant_need = (
        enabled[:, None]
        & (~liquidating[:, None])
        & (crop_deficit > 0)
        & candidates.present[batch[:, None], plant_slots]
        & plant_target_valid
    )
    present = present.at[batch[:, None], plant_slots].set(plant_need)
    hard_mask = hard_mask.at[batch[:, None], plant_slots].set(plant_need)
    target_id = candidates.target_id.at[batch[:, None], plant_slots].set(
        plant_target_id
    )
    target_x = candidates.target_x.at[batch[:, None], plant_slots].set(
        _TILE_X[plant_target_id]
    )
    target_y = candidates.target_y.at[batch[:, None], plant_slots].set(
        _TILE_Y[plant_target_id]
    )
    priority = priority.at[batch[:, None], plant_slots].set(
        jnp.where(plant_need, 74.0, 0.0)
    )

    # Structures track animal targets. Cows and sheep share pasture capacity.
    coop_need = (
        enabled
        & investing
        & (coop_count < animal_target[:, 0].astype(jnp.int16))
        & candidates.present[:, BUILD_COOP_SLOT]
    )
    pasture_kind = states.tile_kind[
        :, player, _PASTURE_Y.astype(jnp.int32), _PASTURE_X.astype(jnp.int32)
    ]
    pasture_candidate = (
        (pasture_rank < jnp.minimum(pasture_target[:, None] + 1, _PASTURE_ID.shape[0]))
        & (pasture_kind == TileKind.EMPTY)
    )
    pasture_choice_key = jnp.where(pasture_candidate, pasture_rank, 127)
    pasture_choice = jnp.argmin(pasture_choice_key, axis=-1)
    pasture_target_valid = jnp.min(pasture_choice_key, axis=-1) < 127
    pasture_need = (
        enabled
        & investing
        & (pasture_count < pasture_target)
        & pasture_target_valid
        & candidates.present[:, BUILD_PASTURE_SLOT]
    )
    present = present.at[:, BUILD_COOP_SLOT].set(coop_need)
    hard_mask = hard_mask.at[:, BUILD_COOP_SLOT].set(coop_need)
    priority = priority.at[:, BUILD_COOP_SLOT].set(
        jnp.where(coop_need, 82.0, 0.0)
    )
    present = present.at[:, BUILD_PASTURE_SLOT].set(pasture_need)
    hard_mask = hard_mask.at[:, BUILD_PASTURE_SLOT].set(pasture_need)
    target_id = target_id.at[:, BUILD_PASTURE_SLOT].set(
        _PASTURE_ID[pasture_choice]
    )
    target_x = target_x.at[:, BUILD_PASTURE_SLOT].set(
        _PASTURE_X[pasture_choice]
    )
    target_y = target_y.at[:, BUILD_PASTURE_SLOT].set(
        _PASTURE_Y[pasture_choice]
    )
    priority = priority.at[:, BUILD_PASTURE_SLOT].set(
        jnp.where(pasture_need, 82.0, 0.0)
    )

    # The generic core exposes only the two highest-ranked tile jobs per free
    # worker.  With animals and mature crops on the board those two slots can
    # both be consumed by feed/harvest work, making watering invisible to the
    # selector.  Reserve one secondary slot per free worker and distribute all
    # currently unwatered crops across those workers.  Missing-day crops are
    # promoted to rescue priority; ordinary crops are handled early enough to
    # avoid an impossible next-day rescue spike.  This is a generic survival
    # invariant, not a replay coordinate or action sequence.
    primary_units = jnp.arange(SECONDARY_UNIT_COUNT, dtype=jnp.int32)[None, :]
    primary_slots = PRIMARY_UNIT_START + primary_units
    secondary_slots = SECONDARY_UNIT_START + primary_units
    primary_available = (
        present[batch[:, None], primary_slots]
        & (
            task_type[batch[:, None], primary_slots]
            != TaskTypeV1.SAFE_RECOVERY
        )
    )
    worker_rank = (
        jnp.cumsum(primary_available.astype(jnp.int16), axis=-1) - 1
    ).astype(jnp.int32)
    flat_flags = states.tile_flags[:, player].reshape(batch_size, -1)
    flat_neglect = states.tile_neglect[:, player].reshape(batch_size, -1)
    unwatered_crop = (
        (flat_kind == TileKind.PLANT)
        & ((flat_flags & jnp.uint8(FLAG_WATERED)) == 0)
    )
    free_worker_count = jnp.maximum(
        jnp.sum(primary_available, axis=-1, dtype=jnp.int16), 1
    )
    crop_rank = jnp.cumsum(unwatered_crop.astype(jnp.int16), axis=-1) - 1
    crop_owner_rank = jnp.mod(crop_rank, free_worker_count[:, None])
    unit_position = states.unit_pos[
        :, player, :SECONDARY_UNIT_COUNT
    ].astype(jnp.int16)
    water_distance = jnp.sum(
        jnp.abs(
            unit_position[:, :, None, :]
            - _TILE_XY[None, None, :, :].astype(jnp.int16)
        ),
        axis=-1,
    )
    assigned_water = (
        unwatered_crop[:, None, :]
        & primary_available[:, :, None]
        & (crop_owner_rank[:, None, :] == worker_rank[:, :, None])
    )
    water_key = jnp.where(
        assigned_water,
        water_distance * 100 + _TILE_ID[None, None, :],
        32767,
    )
    water_target_id = jnp.argmin(water_key, axis=-1).astype(jnp.int16)
    water_target_valid = jnp.min(water_key, axis=-1) < 32767
    water_need = enabled[:, None] & water_target_valid
    water_x = _TILE_X[water_target_id]
    water_y = _TILE_Y[water_target_id]
    water_crop_id = states.tile_crop[
        batch[:, None], player, water_y.astype(jnp.int32), water_x.astype(jnp.int32)
    ]
    water_urgent = (
        flat_neglect[batch[:, None], water_target_id.astype(jnp.int32)] >= 1
    ) & water_need
    existing_secondary_task = task_type[batch[:, None], secondary_slots]
    existing_secondary_target = target_id[batch[:, None], secondary_slots]
    existing_secondary_x = target_x[batch[:, None], secondary_slots]
    existing_secondary_y = target_y[batch[:, None], secondary_slots]
    existing_secondary_item = item_id[batch[:, None], secondary_slots]
    existing_secondary_source = source[batch[:, None], secondary_slots]
    existing_secondary_quantity = quantity[batch[:, None], secondary_slots]
    task_type = task_type.at[batch[:, None], secondary_slots].set(
        jnp.where(water_need, TaskTypeV1.WATER_CROP, existing_secondary_task)
    )
    target_id = target_id.at[batch[:, None], secondary_slots].set(
        jnp.where(water_need, water_target_id, existing_secondary_target)
    )
    target_x = target_x.at[batch[:, None], secondary_slots].set(
        jnp.where(water_need, water_x, existing_secondary_x)
    )
    target_y = target_y.at[batch[:, None], secondary_slots].set(
        jnp.where(water_need, water_y, existing_secondary_y)
    )
    item_id = item_id.at[batch[:, None], secondary_slots].set(
        jnp.where(water_need, water_crop_id, existing_secondary_item)
    )
    source = source.at[batch[:, None], secondary_slots].set(
        jnp.where(water_need, CandidateSourceV1.CROP, existing_secondary_source)
    )
    quantity = quantity.at[batch[:, None], secondary_slots].set(
        jnp.where(water_need, 1, existing_secondary_quantity)
    )
    present = present.at[batch[:, None], secondary_slots].set(
        present[batch[:, None], secondary_slots] | water_need
    )
    hard_mask = hard_mask.at[batch[:, None], secondary_slots].set(
        hard_mask[batch[:, None], secondary_slots] | water_need
    )
    mandatory = mandatory.at[batch[:, None], secondary_slots].set(
        mandatory[batch[:, None], secondary_slots] | water_urgent
    )
    priority = priority.at[batch[:, None], secondary_slots].set(
        jnp.where(
            water_urgent,
            129.0,
            jnp.where(
                water_need,
                123.0,
                priority[batch[:, None], secondary_slots],
            ),
        )
    )

    # Reclaim route capacity continuously.  Finite-life crops and random weed
    # spawns otherwise turn target acreage into permanent obstacles because a
    # low-ranked generic DIG proposal is rarely visible in a dense economy.
    crop_shortfall_now = (
        jnp.sum(crop_deficit, axis=-1, dtype=jnp.int16) > 0
    )
    clear_weed = (
        (flat_kind == TileKind.WEED)
        & (crop_shortfall_now[:, None] | reserved_tile)
        & (~liquidating[:, None])
    )
    clear_weed_count = jnp.sum(clear_weed, axis=-1, dtype=jnp.int16)
    reserved_weed_count = jnp.sum(
        clear_weed & reserved_tile, axis=-1, dtype=jnp.int16
    )
    clear_worker_count = jnp.minimum(
        free_worker_count,
        jnp.maximum(
            (clear_weed_count + 5) // 6,
            (reserved_weed_count > 0).astype(jnp.int16),
        ),
    )
    clear_worker = (
        primary_available
        & (worker_rank >= 0)
        & (worker_rank < clear_worker_count[:, None])
    )
    clear_rank = jnp.cumsum(clear_weed.astype(jnp.int16), axis=-1) - 1
    clear_owner_rank = jnp.mod(
        clear_rank, jnp.maximum(clear_worker_count, 1)[:, None]
    )
    assigned_clear = (
        clear_weed[:, None, :]
        & clear_worker[:, :, None]
        & (clear_owner_rank[:, None, :] == worker_rank[:, :, None])
    )
    clear_key = jnp.where(
        assigned_clear,
        water_distance * 100
        + jnp.where(reserved_tile[:, None, :], 0, 10000)
        + _TILE_ID[None, None, :],
        32767,
    )
    clear_target_id = jnp.argmin(clear_key, axis=-1).astype(jnp.int16)
    clear_need = enabled[:, None] & (jnp.min(clear_key, axis=-1) < 32767)
    clear_x = _TILE_X[clear_target_id]
    clear_y = _TILE_Y[clear_target_id]
    clear_reserved = (
        reserved_tile[batch[:, None], clear_target_id.astype(jnp.int32)]
        & clear_need
    )
    task_type = task_type.at[batch[:, None], secondary_slots].set(
        jnp.where(
            clear_need,
            TaskTypeV1.CLEAR_OR_REMOVE_TILE,
            task_type[batch[:, None], secondary_slots],
        )
    )
    target_id = target_id.at[batch[:, None], secondary_slots].set(
        jnp.where(
            clear_need,
            clear_target_id,
            target_id[batch[:, None], secondary_slots],
        )
    )
    target_x = target_x.at[batch[:, None], secondary_slots].set(
        jnp.where(clear_need, clear_x, target_x[batch[:, None], secondary_slots])
    )
    target_y = target_y.at[batch[:, None], secondary_slots].set(
        jnp.where(clear_need, clear_y, target_y[batch[:, None], secondary_slots])
    )
    item_id = item_id.at[batch[:, None], secondary_slots].set(
        jnp.where(clear_need, -1, item_id[batch[:, None], secondary_slots])
    )
    source = source.at[batch[:, None], secondary_slots].set(
        jnp.where(
            clear_need,
            CandidateSourceV1.CROP,
            source[batch[:, None], secondary_slots],
        )
    )
    quantity = quantity.at[batch[:, None], secondary_slots].set(
        jnp.where(clear_need, 1, quantity[batch[:, None], secondary_slots])
    )
    present = present.at[batch[:, None], secondary_slots].set(
        present[batch[:, None], secondary_slots] | clear_need
    )
    hard_mask = hard_mask.at[batch[:, None], secondary_slots].set(
        hard_mask[batch[:, None], secondary_slots] | clear_need
    )
    mandatory = mandatory.at[batch[:, None], secondary_slots].set(
        jnp.where(
            clear_need,
            clear_reserved,
            mandatory[batch[:, None], secondary_slots],
        )
    )
    priority = priority.at[batch[:, None], secondary_slots].set(
        jnp.where(
            clear_reserved,
            129.5,
            jnp.where(
                clear_need,
                128.0,
                priority[batch[:, None], secondary_slots],
            ),
        )
    )

    # Convert collected fertilizer into crop yield on APPLY/MIXED routes.
    # A small worker band owns this job so fertilizer is not perpetually sold
    # before the low-ranked generic APPLY candidate becomes visible.
    apply_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)
    flat_fertilized_until = states.tile_fertilized_until[:, player].reshape(
        batch_size, -1
    ).astype(jnp.int16)
    fertilizer_target = (
        (flat_kind == TileKind.PLANT)
        & (flat_fertilized_until < apply_day[:, None] + 2)
        & (apply_day[:, None] >= 10)
        & (states.money[:, player, None] >= 5000)
        & (states.unlocked_count[:, player, None] >= 3)
        & (~liquidating[:, None])
    )
    fertilizer_target_count = jnp.sum(
        fertilizer_target, axis=-1, dtype=jnp.int16
    )
    fertilizer_stock = (
        states.shed[:, player, FERTILIZER_ITEM].astype(jnp.int16)
        + jnp.sum(
            states.unit_inventory[:, player, :, FERTILIZER_ITEM].astype(
                jnp.int16
            ),
            axis=-1,
            dtype=jnp.int16,
        )
    )
    apply_capacity = jnp.maximum(free_worker_count - clear_worker_count, 0)
    apply_worker_count = jnp.minimum(
        apply_capacity,
        jnp.minimum(
            fertilizer_stock,
            (fertilizer_target_count + 7) // 8,
        ),
    )
    apply_worker_rank = worker_rank - clear_worker_count[:, None]
    apply_worker = (
        primary_available
        & (apply_worker_rank >= 0)
        & (apply_worker_rank < apply_worker_count[:, None])
        & (fertilizer_mode[:, None] != FertilizerPolicyV1.SELL)
    )
    fertilizer_rank = (
        jnp.cumsum(fertilizer_target.astype(jnp.int16), axis=-1) - 1
    )
    fertilizer_owner_rank = jnp.mod(
        fertilizer_rank, jnp.maximum(apply_worker_count, 1)[:, None]
    )
    assigned_fertilizer = (
        fertilizer_target[:, None, :]
        & apply_worker[:, :, None]
        & (
            fertilizer_owner_rank[:, None, :]
            == apply_worker_rank[:, :, None]
        )
    )
    apply_key = jnp.where(
        assigned_fertilizer,
        water_distance * 100 + _TILE_ID[None, None, :],
        32767,
    )
    apply_target_id = jnp.argmin(apply_key, axis=-1).astype(jnp.int16)
    apply_need = (
        enabled[:, None]
        & (jnp.min(apply_key, axis=-1) < 32767)
        & (~water_urgent)
    )
    apply_x = _TILE_X[apply_target_id]
    apply_y = _TILE_Y[apply_target_id]
    task_type = task_type.at[batch[:, None], secondary_slots].set(
        jnp.where(
            apply_need,
            TaskTypeV1.APPLY_FERTILIZER,
            task_type[batch[:, None], secondary_slots],
        )
    )
    target_id = target_id.at[batch[:, None], secondary_slots].set(
        jnp.where(
            apply_need,
            apply_target_id,
            target_id[batch[:, None], secondary_slots],
        )
    )
    target_x = target_x.at[batch[:, None], secondary_slots].set(
        jnp.where(apply_need, apply_x, target_x[batch[:, None], secondary_slots])
    )
    target_y = target_y.at[batch[:, None], secondary_slots].set(
        jnp.where(apply_need, apply_y, target_y[batch[:, None], secondary_slots])
    )
    item_id = item_id.at[batch[:, None], secondary_slots].set(
        jnp.where(
            apply_need,
            FERTILIZER_ITEM,
            item_id[batch[:, None], secondary_slots],
        )
    )
    source = source.at[batch[:, None], secondary_slots].set(
        jnp.where(
            apply_need,
            CandidateSourceV1.FERTILIZER_APPLY,
            source[batch[:, None], secondary_slots],
        )
    )
    quantity = quantity.at[batch[:, None], secondary_slots].set(
        jnp.where(apply_need, 1, quantity[batch[:, None], secondary_slots])
    )
    present = present.at[batch[:, None], secondary_slots].set(
        present[batch[:, None], secondary_slots] | apply_need
    )
    hard_mask = hard_mask.at[batch[:, None], secondary_slots].set(
        hard_mask[batch[:, None], secondary_slots] | apply_need
    )
    mandatory = mandatory.at[batch[:, None], secondary_slots].set(
        jnp.where(
            apply_need,
            False,
            mandatory[batch[:, None], secondary_slots],
        )
    )
    priority = priority.at[batch[:, None], secondary_slots].set(
        jnp.where(
            apply_need,
            127.5,
            priority[batch[:, None], secondary_slots],
        )
    )

    # Fill the nine previously-unused pool slots with a balanced, global
    # production dispatcher.  Per-worker nearest-two proposals often collide
    # on the same animal and hide care/fertilizer/product work behind crop
    # tasks.  These slots expose nine distinct board jobs to nine distinct free
    # workers while the existing feasibility and shared ledger remain the
    # authority on what can actually execute.
    maintenance_slots = MAINTENANCE_START + jnp.arange(
        MAINTENANCE_COUNT, dtype=jnp.int32
    )[None, :]
    free_unit_key = jnp.where(
        primary_available,
        jnp.arange(SECONDARY_UNIT_COUNT, dtype=jnp.int16)[None, :],
        127,
    )
    maintenance_owner = jnp.argsort(free_unit_key, axis=-1)[
        :, :MAINTENANCE_COUNT
    ].astype(jnp.int8)
    maintenance_owner_valid = (
        jnp.arange(MAINTENANCE_COUNT, dtype=jnp.int16)[None, :]
        < jnp.sum(primary_available, axis=-1, dtype=jnp.int16)[:, None]
    )
    flat_animal = states.tile_animal[:, player].reshape(batch_size, -1)
    flat_yield = states.tile_yield[:, player].reshape(batch_size, -1)
    animal_present = flat_animal >= 0
    unfed = animal_present & ((flat_flags & jnp.uint8(FLAG_FED)) == 0)
    uncared = (
        animal_present
        & ((flat_flags & jnp.uint8(FLAG_CARED)) == 0)
        & (~liquidating[:, None])
    )
    fertilizer_ready = animal_present & (
        (flat_flags & jnp.uint8(FLAG_FERTILIZER_AVAILABLE)) != 0
    )
    product_ready = animal_present & (flat_yield > 0)
    urgent_feed = unfed & (flat_neglect >= 1)
    crop_shortfall = jnp.sum(crop_deficit, axis=-1, dtype=jnp.int16) > 0
    useful_weed = (flat_kind == TileKind.WEED) & crop_shortfall[:, None]
    maintenance_code = jnp.full_like(flat_kind, 99, dtype=jnp.int16)
    maintenance_code = jnp.where(useful_weed, 5, maintenance_code)
    maintenance_code = jnp.where(product_ready, 4, maintenance_code)
    maintenance_code = jnp.where(
        unfed & (~liquidating[:, None]), 3, maintenance_code
    )
    maintenance_code = jnp.where(uncared, 2, maintenance_code)
    maintenance_code = jnp.where(fertilizer_ready, 1, maintenance_code)
    maintenance_code = jnp.where(urgent_feed, 0, maintenance_code)
    maintenance_key = maintenance_code.astype(jnp.int32) * 1000 + _TILE_ID[
        None, :
    ].astype(jnp.int32)
    maintenance_order = jnp.argsort(maintenance_key, axis=-1)[
        :, :MAINTENANCE_COUNT
    ]
    selected_maintenance_code = jnp.take_along_axis(
        maintenance_code, maintenance_order, axis=-1
    )
    # Product peaks are sparse but decisive (for example a first wool batch
    # funding land and labor).  Guarantee up to three visible product jobs so
    # the generic ranking cannot leave a whole ripe batch until the next day.
    maintenance_slot_rank = jnp.arange(
        MAINTENANCE_COUNT, dtype=jnp.int16
    )[None, :]
    product_key = jnp.where(product_ready, _TILE_ID[None, :], 32767)
    product_order = jnp.argsort(product_key, axis=-1)[:, :MAINTENANCE_COUNT]
    product_rank = jnp.clip(
        maintenance_slot_rank.astype(jnp.int32) - 1,
        0,
        MAINTENANCE_COUNT - 1,
    )
    product_target = jnp.take_along_axis(product_order, product_rank, axis=-1)
    product_valid = (
        jnp.take_along_axis(product_key, product_target, axis=-1) < 32767
    )
    product_slots = (
        (maintenance_slot_rank >= 1) & (maintenance_slot_rank <= 3)
    ) & product_valid
    maintenance_order = jnp.where(
        product_slots, product_target, maintenance_order
    )
    selected_maintenance_code = jnp.where(
        product_slots, 4, selected_maintenance_code
    )
    reserved_weed_key = jnp.where(
        reserved_pasture & (pasture_kind == TileKind.WEED),
        pasture_rank,
        127,
    )
    reserved_weed_choice = jnp.argmin(reserved_weed_key, axis=-1)
    reserved_weed_valid = jnp.min(reserved_weed_key, axis=-1) < 127
    reserved_weed_target = _PASTURE_ID[reserved_weed_choice]
    reserved_clear_slot = (
        (maintenance_slot_rank == 0) & reserved_weed_valid[:, None]
    )
    maintenance_order = jnp.where(
        reserved_clear_slot, reserved_weed_target[:, None], maintenance_order
    )
    selected_maintenance_code = jnp.where(
        reserved_clear_slot, 5, selected_maintenance_code
    )
    general_weed_key = jnp.where(
        useful_weed & (~reserved_tile), _TILE_ID[None, :], 32767
    )
    general_weed_target = jnp.argmin(general_weed_key, axis=-1)
    general_weed_valid = jnp.min(general_weed_key, axis=-1) < 32767
    general_clear_slot = (
        (maintenance_slot_rank == MAINTENANCE_COUNT - 1)
        & general_weed_valid[:, None]
        & (~reserved_clear_slot)
    )
    maintenance_order = jnp.where(
        general_clear_slot, general_weed_target[:, None], maintenance_order
    )
    selected_maintenance_code = jnp.where(
        general_clear_slot, 5, selected_maintenance_code
    )
    flat_crop_id = states.tile_crop[:, player].reshape(batch_size, -1)
    flat_origin_day = states.tile_origin_day[:, player].reshape(batch_size, -1)
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)
    crop_first_day = jnp.asarray(CROP_FIRST_YIELD_DAY, dtype=jnp.int16)
    safe_flat_crop = jnp.clip(flat_crop_id.astype(jnp.int32), 0, NUM_CROPS - 1)
    harvest_ready = (
        (flat_kind == TileKind.PLANT)
        & (flat_yield > 0)
        & (
            current_day[:, None] - flat_origin_day.astype(jnp.int16)
            >= crop_first_day[safe_flat_crop]
        )
    )
    harvest_key = jnp.where(harvest_ready, _TILE_ID[None, :], 32767)
    harvest_order = jnp.argsort(harvest_key, axis=-1)[:, :MAINTENANCE_COUNT]
    harvest_valid = jnp.take_along_axis(
        harvest_key, harvest_order, axis=-1
    ) < 32767
    # The generic nearest-two pool exposes too few mature crops once animal
    # and watering work becomes dense.  Route schedules may reserve a bounded
    # tail band of the nine global dispatcher lanes for distinct ripe crops.
    # Preserve urgent feed, ready animal product, and the reserved pasture
    # clear; same-tile animal chaining can finish the remaining animal service.
    harvest_lane_start = MAINTENANCE_COUNT - jnp.clip(
        schedule.harvest_dispatch_lanes.astype(jnp.int16),
        0,
        MAINTENANCE_COUNT,
    )
    harvest_lane = maintenance_slot_rank >= harvest_lane_start[:, None]
    harvest_lane_rank = jnp.maximum(
        maintenance_slot_rank - harvest_lane_start[:, None], 0
    ).astype(jnp.int32)
    harvest_lane_target = jnp.take_along_axis(
        harvest_order, harvest_lane_rank, axis=-1
    )
    harvest_lane_valid = jnp.take_along_axis(
        harvest_key, harvest_lane_target, axis=-1
    ) < 32767
    continuous_crop_slot = (
        (~liquidating[:, None])
        & (states.step[:, None] >= schedule.harvest_dispatch_start_step[:, None])
        & harvest_lane
        & harvest_lane_valid
        & (selected_maintenance_code != 0)
        & (selected_maintenance_code != 4)
        & (~reserved_clear_slot)
    )
    maintenance_order = jnp.where(
        continuous_crop_slot, harvest_lane_target, maintenance_order
    )
    selected_maintenance_code = jnp.where(
        continuous_crop_slot, 6, selected_maintenance_code
    )
    liquidation_crop_slot = liquidating[:, None] & harvest_valid
    maintenance_order = jnp.where(
        liquidation_crop_slot, harvest_order, maintenance_order
    )
    selected_maintenance_code = jnp.where(
        liquidation_crop_slot, 6, selected_maintenance_code
    )
    harvest_dispatch_slot = continuous_crop_slot | liquidation_crop_slot
    liquidation_available_workers = primary_available
    liquidation_owner_parts = []
    liquidation_owner_valid_parts = []
    liquidation_unit_ids = jnp.arange(
        SECONDARY_UNIT_COUNT, dtype=jnp.int32
    )[None, :]
    for liquidation_index in range(MAINTENANCE_COUNT):
        liquidation_target = maintenance_order[:, liquidation_index]
        liquidation_xy = jnp.stack(
            (
                _TILE_X[liquidation_target],
                _TILE_Y[liquidation_target],
            ),
            axis=-1,
        ).astype(jnp.int16)
        liquidation_distance = jnp.sum(
            jnp.abs(unit_position - liquidation_xy[:, None, :]), axis=-1
        ).astype(jnp.int32)
        liquidation_worker_key = jnp.where(
            liquidation_available_workers,
            liquidation_distance * 100 + liquidation_unit_ids,
            32767,
        )
        liquidation_owner = jnp.argmin(
            liquidation_worker_key, axis=-1
        ).astype(jnp.int8)
        liquidation_owner_valid = (
            jnp.min(liquidation_worker_key, axis=-1) < 32767
        )
        liquidation_owner_parts.append(liquidation_owner)
        liquidation_owner_valid_parts.append(liquidation_owner_valid)
        liquidation_available_workers = liquidation_available_workers & ~(
            liquidation_unit_ids
            == liquidation_owner[:, None].astype(jnp.int32)
        )
    liquidation_owner = jnp.stack(liquidation_owner_parts, axis=-1)
    liquidation_owner_valid = jnp.stack(
        liquidation_owner_valid_parts, axis=-1
    )
    maintenance_owner = jnp.where(
        liquidating[:, None], liquidation_owner, maintenance_owner
    )
    maintenance_owner_valid = jnp.where(
        liquidating[:, None], liquidation_owner_valid, maintenance_owner_valid
    )
    maintenance_valid = (
        enabled[:, None]
        & maintenance_owner_valid
        & (selected_maintenance_code < 99)
    )
    maintenance_animal = jnp.take_along_axis(
        flat_animal, maintenance_order, axis=-1
    )
    safe_maintenance_animal = jnp.clip(
        maintenance_animal.astype(jnp.int32), 0, NUM_ANIMALS - 1
    )
    maintenance_task = jnp.where(
        selected_maintenance_code <= 4,
        jnp.where(
            selected_maintenance_code == 0,
            TaskTypeV1.ANIMAL_FEED,
            jnp.where(
                selected_maintenance_code == 1,
                TaskTypeV1.ANIMAL_COLLECT_FERTILIZER,
                jnp.where(
                    selected_maintenance_code == 2,
                    TaskTypeV1.ANIMAL_CARE,
                    jnp.where(
                        selected_maintenance_code == 3,
                        TaskTypeV1.ANIMAL_FEED,
                        TaskTypeV1.ANIMAL_COLLECT_PRODUCT,
                    ),
                ),
            ),
        ),
        jnp.where(
            selected_maintenance_code == 6,
            TaskTypeV1.CROP_PRODUCTION,
            TaskTypeV1.CLEAR_OR_REMOVE_TILE,
        ),
    ).astype(jnp.int8)
    maintenance_item = jnp.where(
        (selected_maintenance_code == 0) | (selected_maintenance_code == 3),
        WHEAT_ITEM,
        jnp.where(
            selected_maintenance_code == 1,
            FERTILIZER_ITEM,
            jnp.where(
                selected_maintenance_code == 4,
                _ANIMAL_PRODUCT[safe_maintenance_animal],
                jnp.where(
                    selected_maintenance_code == 2,
                    maintenance_animal,
                    jnp.where(
                        selected_maintenance_code == 6,
                        jnp.take_along_axis(
                            flat_crop_id, maintenance_order, axis=-1
                        ),
                        -1,
                    ),
                ),
            ),
        ),
    ).astype(jnp.int8)
    maintenance_source = jnp.where(
        (selected_maintenance_code == 0) | (selected_maintenance_code == 3),
        CandidateSourceV1.MAINTENANCE,
        jnp.where(
            (selected_maintenance_code == 1)
            | (selected_maintenance_code == 4),
            CandidateSourceV1.ANIMAL,
            jnp.where(
                selected_maintenance_code == 2,
                CandidateSourceV1.MAINTENANCE,
                CandidateSourceV1.CROP,
            ),
        ),
    ).astype(jnp.int8)
    maintenance_priority = jnp.where(
        harvest_dispatch_slot,
        131.0,
        jnp.where(
        reserved_clear_slot,
        129.5,
        jnp.where(
        general_clear_slot,
        124.5,
        jnp.where(
        selected_maintenance_code == 0,
        129.0,
        jnp.where(
            selected_maintenance_code == 1,
            126.0,
            jnp.where(
            selected_maintenance_code == 2,
            125.5,
            jnp.where(
                selected_maintenance_code == 3,
                124.0,
                jnp.where(selected_maintenance_code == 4, 125.0, 118.0),
                ),
            ),
        ),
        ),
        ),
        ),
    )
    maintenance_x = _TILE_X[maintenance_order]
    maintenance_y = _TILE_Y[maintenance_order]
    task_type = task_type.at[batch[:, None], maintenance_slots].set(
        maintenance_task
    )
    item_id = item_id.at[batch[:, None], maintenance_slots].set(
        maintenance_item
    )
    source = source.at[batch[:, None], maintenance_slots].set(
        maintenance_source
    )
    quantity = quantity.at[batch[:, None], maintenance_slots].set(jnp.int16(1))
    target_id = target_id.at[batch[:, None], maintenance_slots].set(
        maintenance_order.astype(jnp.int16)
    )
    target_x = target_x.at[batch[:, None], maintenance_slots].set(
        maintenance_x.astype(jnp.int8)
    )
    target_y = target_y.at[batch[:, None], maintenance_slots].set(
        maintenance_y.astype(jnp.int8)
    )
    owner_unit = candidates.owner_unit.at[batch[:, None], maintenance_slots].set(
        maintenance_owner
    )
    present = present.at[batch[:, None], maintenance_slots].set(maintenance_valid)
    hard_mask = hard_mask.at[batch[:, None], maintenance_slots].set(
        maintenance_valid
    )
    mandatory = mandatory.at[batch[:, None], maintenance_slots].set(
        maintenance_valid
        & (
            (selected_maintenance_code == 0)
            | reserved_clear_slot
            | harvest_dispatch_slot
        )
    )
    priority = priority.at[batch[:, None], maintenance_slots].set(
        jnp.where(maintenance_valid, maintenance_priority, 0.0)
    )

    # Large route expansions need more than the single generic PLANT card per
    # crop.  Gold-like schedules can add twenty-plus plots inside a short
    # phase; serial path tasks leave purchased seed idle for many days.  Reuse
    # the final four global-dispatch lanes as parallel planting jobs when their
    # current maintenance work is neither urgent feed, fertilizer collection,
    # care, nor ready-product collection.  Each lane gets a distinct empty
    # tile and crops are assigned round-robin across current deficits.  The
    # ordinary feasibility/resource ledger still limits seed and worker use.
    expansion_lane_start = MAINTENANCE_COUNT - jnp.clip(
        schedule.parallel_plant_lanes.astype(jnp.int16), 0, 4
    )
    expansion_lane = maintenance_slot_rank >= expansion_lane_start[:, None]
    expansion_rank = jnp.maximum(
        maintenance_slot_rank - expansion_lane_start[:, None], 0
    ).astype(jnp.int16)
    expansion_crop = (crop_deficit > 0) & (seed_stock > 0)
    expansion_crop_count = jnp.sum(
        expansion_crop, axis=-1, dtype=jnp.int16
    )
    expansion_crop_rank = (
        jnp.cumsum(expansion_crop.astype(jnp.int16), axis=-1) - 1
    )
    wanted_crop_rank = jnp.mod(
        expansion_rank,
        jnp.maximum(expansion_crop_count, 1)[:, None],
    )
    expansion_crop_key = jnp.where(
        expansion_crop[:, None, :]
        & (
            expansion_crop_rank[:, None, :]
            == wanted_crop_rank[:, :, None]
        ),
        jnp.arange(NUM_CROPS, dtype=jnp.int16)[None, None, :],
        127,
    )
    expansion_crop_id = jnp.argmin(
        expansion_crop_key, axis=-1
    ).astype(jnp.int8)
    expansion_crop_valid = jnp.min(expansion_crop_key, axis=-1) < 127
    expansion_ordinal = expansion_rank // jnp.maximum(
        expansion_crop_count, 1
    )[:, None]
    expansion_seed = jnp.take_along_axis(
        seed_stock,
        expansion_crop_id.astype(jnp.int32),
        axis=-1,
    )
    expansion_deficit = jnp.take_along_axis(
        crop_deficit,
        expansion_crop_id.astype(jnp.int32),
        axis=-1,
    )
    expansion_resource_ok = (
        (expansion_seed > expansion_ordinal)
        & (expansion_deficit > expansion_ordinal)
    )
    expansion_replaceable = (
        (
            selected_maintenance_code
            >= schedule.parallel_plant_min_maintenance_code[:, None]
        )
        & (selected_maintenance_code != 4)
        & (~reserved_clear_slot)
        & (~harvest_dispatch_slot)
    )
    expansion_unit_key = jnp.where(
        states.unit_active[:, player, :SECONDARY_UNIT_COUNT],
        jnp.arange(SECONDARY_UNIT_COUNT, dtype=jnp.int16)[None, :],
        127,
    )
    expansion_owner = jnp.argsort(expansion_unit_key, axis=-1)[
        :, :MAINTENANCE_COUNT
    ].astype(jnp.int8)
    expansion_owner_valid = (
        jnp.arange(MAINTENANCE_COUNT, dtype=jnp.int16)[None, :]
        < jnp.sum(
            states.unit_active[:, player, :SECONDARY_UNIT_COUNT],
            axis=-1,
            dtype=jnp.int16,
        )[:, None]
    )

    used_expansion_tile = jnp.zeros(
        (batch_size, BOARD_SIZE * BOARD_SIZE), dtype=jnp.bool_
    )
    for crop_index in range(NUM_CROPS):
        used_expansion_tile = used_expansion_tile.at[
            batch, plant_target_id[:, crop_index].astype(jnp.int32)
        ].set(plant_need[:, crop_index])
    expansion_target_parts = []
    expansion_valid_parts = []
    for lane_index in range(MAINTENANCE_COUNT):
        lane_owner = expansion_owner[:, lane_index].astype(jnp.int32)
        lane_position = states.unit_pos[
            batch, player, lane_owner
        ].astype(jnp.int16)
        lane_distance = jnp.sum(
            jnp.abs(
                lane_position[:, None, :]
                - _TILE_XY[None, :, :].astype(jnp.int16)
            ),
            axis=-1,
        )
        lane_key = jnp.where(
            plant_eligible & (~used_expansion_tile),
            lane_distance * 100 + _TILE_ID[None, :],
            32767,
        )
        lane_target = jnp.argmin(lane_key, axis=-1).astype(jnp.int16)
        lane_target_valid = jnp.min(lane_key, axis=-1) < 32767
        lane_valid = (
            expansion_lane[:, lane_index]
            & expansion_owner_valid[:, lane_index]
            & expansion_crop_valid[:, lane_index]
            & expansion_resource_ok[:, lane_index]
            & expansion_replaceable[:, lane_index]
            & lane_target_valid
            & (~liquidating)
        )
        expansion_target_parts.append(lane_target)
        expansion_valid_parts.append(lane_valid)
        used_expansion_tile = used_expansion_tile.at[
            batch, lane_target.astype(jnp.int32)
        ].set(lane_valid)
    expansion_target = jnp.stack(expansion_target_parts, axis=-1)
    expansion_valid = jnp.stack(expansion_valid_parts, axis=-1)
    expansion_x = _TILE_X[expansion_target]
    expansion_y = _TILE_Y[expansion_target]
    task_type = task_type.at[batch[:, None], maintenance_slots].set(
        jnp.where(
            expansion_valid,
            TaskTypeV1.CROP_PRODUCTION,
            task_type[batch[:, None], maintenance_slots],
        )
    )
    target_id = target_id.at[batch[:, None], maintenance_slots].set(
        jnp.where(
            expansion_valid,
            expansion_target,
            target_id[batch[:, None], maintenance_slots],
        )
    )
    target_x = target_x.at[batch[:, None], maintenance_slots].set(
        jnp.where(
            expansion_valid,
            expansion_x,
            target_x[batch[:, None], maintenance_slots],
        )
    )
    target_y = target_y.at[batch[:, None], maintenance_slots].set(
        jnp.where(
            expansion_valid,
            expansion_y,
            target_y[batch[:, None], maintenance_slots],
        )
    )
    item_id = item_id.at[batch[:, None], maintenance_slots].set(
        jnp.where(
            expansion_valid,
            expansion_crop_id,
            item_id[batch[:, None], maintenance_slots],
        )
    )
    source = source.at[batch[:, None], maintenance_slots].set(
        jnp.where(
            expansion_valid,
            CandidateSourceV1.CROP,
            source[batch[:, None], maintenance_slots],
        )
    )
    owner_unit = owner_unit.at[batch[:, None], maintenance_slots].set(
        jnp.where(
            expansion_valid,
            expansion_owner,
            owner_unit[batch[:, None], maintenance_slots],
        )
    )
    quantity = quantity.at[batch[:, None], maintenance_slots].set(
        jnp.where(
            expansion_valid,
            1,
            quantity[batch[:, None], maintenance_slots],
        )
    )
    present = present.at[batch[:, None], maintenance_slots].set(
        present[batch[:, None], maintenance_slots] | expansion_valid
    )
    hard_mask = hard_mask.at[batch[:, None], maintenance_slots].set(
        hard_mask[batch[:, None], maintenance_slots] | expansion_valid
    )
    mandatory = mandatory.at[batch[:, None], maintenance_slots].set(
        jnp.where(
            expansion_valid,
            False,
            mandatory[batch[:, None], maintenance_slots],
        )
    )
    priority = priority.at[batch[:, None], maintenance_slots].set(
        jnp.where(
            expansion_valid,
            122.5,
            priority[batch[:, None], maintenance_slots],
        )
    )

    # A released collector may keep a small batch on the worker and take
    # another nearby production job.  Re-enable the generic depot trip once
    # that worker reaches the route's batching threshold.  Threshold one is
    # the legacy one-collection-per-trip behavior.
    deposit_task = (
        (task_type == TaskTypeV1.SAFE_RECOVERY)
        | (task_type == TaskTypeV1.SHED_DEPOSIT)
    )
    deposit_owner = jnp.clip(owner_unit.astype(jnp.int32), 0, MAX_UNITS - 1)
    deposit_carried = jnp.sum(
        states.unit_inventory[batch[:, None], player, deposit_owner].astype(
            jnp.int32
        ),
        axis=-1,
    )
    batch_ready = deposit_carried >= jnp.maximum(
        schedule.deposit_batch_units.astype(jnp.int32), 1
    )[:, None]
    suppress_small_deposit = (
        enabled[:, None]
        & (~liquidating[:, None])
        & deposit_task
        & (~batch_ready)
    )
    present = present & (~suppress_small_deposit)
    hard_mask = hard_mask & (~suppress_small_deposit)
    mandatory = mandatory & (~suppress_small_deposit)
    priority = jnp.where(suppress_small_deposit, 0.0, priority)

    # Do not apply fertilizer on routes that explicitly sell it.
    apply_fertilizer = task_type == TaskTypeV1.APPLY_FERTILIZER
    allow_apply = fertilizer_mode != FertilizerPolicyV1.SELL
    present = jnp.where(
        apply_fertilizer,
        present & allow_apply[:, None],
        present,
    )
    hard_mask = jnp.where(
        apply_fertilizer,
        hard_mask & allow_apply[:, None],
        hard_mask,
    )

    # E5's generic short-horizon economics intentionally undervalues care,
    # watering and collection.  A RouteSchedule is a long-horizon commitment,
    # so its already-owned production chain must outrank new spending.  Legal,
    # cash, capacity and shared-ledger checks still remask every task.
    task = task_type
    operational = owner_unit >= 0
    safe_candidate_x = jnp.clip(target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    safe_candidate_y = jnp.clip(target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    candidate_neglect = states.tile_neglect[
        batch[:, None], player, safe_candidate_y, safe_candidate_x
    ].astype(jnp.int16)
    urgent_feed_candidate = (
        (task == TaskTypeV1.ANIMAL_FEED) & (candidate_neglect >= 1)
    )
    liquidation_harvest = (
        liquidating[:, None]
        & (task == TaskTypeV1.CROP_PRODUCTION)
        & operational
    )
    route_task_priority = jnp.where(
        liquidation_harvest,
        131.0,
        jnp.where(
            task == TaskTypeV1.BUILD_ANIMAL_STRUCTURE,
            130.0,
            jnp.where(
                urgent_feed_candidate,
                129.0,
                jnp.where(
                    task == TaskTypeV1.ANIMAL_PLACE,
                    128.0,
                    jnp.where(
                        task == TaskTypeV1.SAFE_RECOVERY,
                        127.0,
                        jnp.where(
                            task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER,
                            126.0,
                            jnp.where(
                                task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT,
                                125.0,
                                jnp.where(
                                    task == TaskTypeV1.ANIMAL_FEED,
                                    124.0,
                                    jnp.where(
                                        task == TaskTypeV1.WATER_CROP,
                                        123.0,
                                        jnp.where(
                                            (task == TaskTypeV1.CROP_PRODUCTION)
                                            & operational,
                                            122.0,
                                            jnp.where(
                                                task == TaskTypeV1.SHED_DEPOSIT,
                                                121.0,
                                                jnp.where(
                                                    task == TaskTypeV1.SHED_PICKUP,
                                                    120.0,
                                                    jnp.where(
                                                        task == TaskTypeV1.ANIMAL_CARE,
                                                        123.5,
                                                        jnp.where(
                                                            task
                                                            == TaskTypeV1.CLEAR_OR_REMOVE_TILE,
                                                            118.0,
                                                            jnp.where(
                                                                task
                                                                == TaskTypeV1.APPLY_FERTILIZER,
                                                                117.0,
                                                                0.0,
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    priority = jnp.maximum(
        priority,
        jnp.where(enabled[:, None], route_task_priority, 0.0),
    )

    # A disabled schedule is a true JAX NullAgent: no candidate can be selected.
    present = present & enabled[:, None]
    hard_mask = hard_mask & enabled[:, None]
    return candidates._replace(
        task_type=task_type,
        owner_unit=owner_unit,
        present=present,
        hard_mask=hard_mask,
        mandatory=mandatory,
        target_id=target_id,
        target_x=target_x,
        target_y=target_y,
        item_id=item_id,
        source=source,
        quantity=quantity.astype(jnp.int16),
        replay_priority=priority.astype(jnp.float32),
    )
