"""E5 Full-econ simple in terminal bank-cash units.

This is deliberately a transparent heuristic, not a learned price predictor.
Exact current state and deterministic calendars are kept separate from soft
expected/scenario risk terms, which can be disabled without changing Core.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_COST,
    ANIMAL_FIRST_YIELD_DAY,
    ANIMAL_INTERVAL,
    ANIMAL_MAX_HELD,
    ANIMAL_PRODUCT,
    BOARD_SIZE,
    CROP_FIRST_YIELD_DAY,
    CROP_INTERVAL,
    CROP_MAX_YIELD,
    CROP_ONGOING,
    CROP_SEED_COST,
    EPISODE_STEPS,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_WATERED,
    LAND_PRICES,
    MARKET_LUT_SIZE,
    MARKET_MIN_INVENTORY,
    MAX_HANDS,
    MAX_SHOPS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    SHED_ACCESS,
    SHED_CAPACITY,
    TURNS_PER_DAY,
    TileKind,
)
from kaggriculture_jax.types import State, StaticTables

from .constants import MAX_SELECTIONS_V1, TaskTypeV1
from .e2_core import E2SelectionV1, initialize_e2_ledger_v1, select_candidates_with_scores_v1
from .schema import CandidateV1, ControllerStateV1, EconFeaturesV1, FeasibilityV1


WHEAT_ITEM = 0
FERTILIZER_ITEM = 8
SIMPLE_CASH_BUFFER = 500
ACTION_OPPORTUNITY_COST = 2.0
WEED_SPAWN_CHANCE = 0.005

_CROP_FIRST = jnp.asarray(CROP_FIRST_YIELD_DAY, dtype=jnp.int16)
_CROP_INTERVAL = jnp.asarray(CROP_INTERVAL, dtype=jnp.int16)
_CROP_MAX = jnp.asarray(CROP_MAX_YIELD, dtype=jnp.int16)
_CROP_ONGOING = jnp.asarray(CROP_ONGOING, dtype=jnp.bool_)
_SEED_COST = jnp.asarray(CROP_SEED_COST, dtype=jnp.int32)
_ANIMAL_COST = jnp.asarray(ANIMAL_COST, dtype=jnp.int32)
_ANIMAL_FIRST = jnp.asarray(ANIMAL_FIRST_YIELD_DAY, dtype=jnp.int16)
_ANIMAL_INTERVAL = jnp.asarray(ANIMAL_INTERVAL, dtype=jnp.int16)
_ANIMAL_MAX = jnp.asarray(ANIMAL_MAX_HELD, dtype=jnp.int16)
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int8)
_LAND_PRICES = jnp.asarray(LAND_PRICES, dtype=jnp.int32)
_HIRE_COST = jnp.asarray(
    (
        1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
        987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368,
        75025, 121393, 196418, 317811, 514229, 832040, 1346269, 2178309,
    ),
    dtype=jnp.int32,
)
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)
_SHOP_DEMAND = jnp.asarray(
    (
        (1, 0, 0, 0, 0, 1, 0, 0, 0),
        (1, 0, 0, 1, 0, 1, 0, 0, 0),
        (1, 1, 1, 1, 0, 0, 0, 0, 0),
        (1, 0, 0, 1, 0, 0, 1, 0, 0),
        (0, 2, 0, 0, 0, 0, 0, 0, 0),
        (1, 0, 1, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 2, 0),
    ),
    dtype=jnp.int16,
)
_CENTER_DEMAND = jnp.asarray((1, 1, 1, 1, 1, 1, 1, 1, 0), dtype=jnp.int16)

# Section 7.1 bit positions.  Every feature tensor carries a provenance mask.
ECON_FIELD_COUNT = 20
ECON_ALL_FIELDS_MASK = (1 << ECON_FIELD_COUNT) - 1
ECON_STOCHASTIC_RISK_BIT = 1 << 18
ECON_EXPECTED_BANK_BIT = 1 << 0


def _count_multiples(start: jax.Array, count: jax.Array, divisor: int) -> jax.Array:
    end = start + count - 1
    return jnp.where(
        count > 0,
        end // divisor - (start - 1) // divisor,
        0,
    ).astype(jnp.int16)


def known_town_demand_before_sale_v1(
    states: State, item: jax.Array, steps_to_revenue: jax.Array
) -> jax.Array:
    """Exact demand from currently unlocked shops before a future market action."""

    batch_size = states.step.shape[0]
    safe_shops = jnp.clip(states.town_shops.astype(jnp.int32), 0, MAX_SHOPS - 1)
    active = jnp.arange(MAX_SHOPS)[None, :] < states.town_count[:, None]
    shop_demand = jnp.sum(
        _SHOP_DEMAND[safe_shops] * active[..., None], axis=1, dtype=jnp.int16
    )
    prior_transitions = jnp.maximum(steps_to_revenue.astype(jnp.int32) - 1, 0)
    start = states.step[:, None].astype(jnp.int32)
    shop_ticks = _count_multiples(start, prior_transitions, 4)
    center_ticks = _count_multiples(start, prior_transitions, 24)
    safe_item = jnp.clip(item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    demand = (
        shop_ticks * shop_demand[batch, safe_item]
        + center_ticks * _CENTER_DEMAND[safe_item]
    )
    return jnp.where(
        (item >= 0) & (item < NUM_PRODUCTS), demand, 0
    ).astype(jnp.int16)


def exact_market_quote_v1(
    states: State,
    tables: StaticTables,
    item: jax.Array,
    quantity: jax.Array,
    *,
    buy: bool,
    known_town_demand: jax.Array | None = None,
    player: int = 0,
    enforce_resources: bool = True,
) -> jax.Array:
    """Exact current sequential quote for one product order per candidate.

    This reproduces the official sequential price path and price-floor rule.
    ``enforce_resources=True`` additionally applies current cash, shed capacity,
    and current sell inventory.  Lifecycle estimates set it to False because
    their products and inputs do not exist in the current state yet.
    """

    batch_size = states.step.shape[0]
    safe_item = jnp.clip(item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    quantity = jnp.broadcast_to(
        jnp.clip(quantity.astype(jnp.int32), 0, SHED_CAPACITY), item.shape
    )
    town = (
        jnp.zeros_like(quantity)
        if known_town_demand is None
        else jnp.broadcast_to(known_town_demand.astype(jnp.int32), item.shape)
    )
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    inventory = states.market_inventory[batch, safe_item] - town
    current_cash = jnp.broadcast_to(
        states.money[:, player, None].astype(jnp.int32), item.shape
    )
    shed_used = jnp.sum(
        states.shed[:, player].astype(jnp.int32), axis=-1
    )[:, None]
    current_room = jnp.maximum(SHED_CAPACITY - shed_used, 0)
    current_available = states.shed[batch, player, safe_item].astype(jnp.int32)
    if enforce_resources:
        cash = current_cash
        room = current_room
        available = current_available
    else:
        cash = jnp.full_like(current_cash, jnp.iinfo(jnp.int32).max)
        room = quantity
        available = quantity
    def forward_sum(q: jax.Array) -> jax.Array:
        """Official clipped quotes at inventory .. inventory+q-1."""

        q = jnp.maximum(q.astype(jnp.int32), 0)
        raw_start = inventory.astype(jnp.int32) - MARKET_MIN_INVENTORY
        low = jnp.minimum(q, jnp.maximum(-raw_start, 0))
        middle_start = jnp.clip(raw_start, 0, MARKET_LUT_SIZE)
        middle = jnp.minimum(
            q - low, jnp.maximum(MARKET_LUT_SIZE - middle_start, 0)
        )
        middle_end = middle_start + middle
        tail = q - low - middle
        return (
            low * tables.market_price[safe_item, 0].astype(jnp.int32)
            + tables.market_price_prefix[safe_item, middle_end]
            - tables.market_price_prefix[safe_item, middle_start]
            + tail * tables.market_price[safe_item, -1].astype(jnp.int32)
        ).astype(jnp.int32)

    def backward_sum(q: jax.Array) -> jax.Array:
        """Official BUY_PRODUCT quotes at inventory-1 .. inventory-q."""

        q = jnp.maximum(q.astype(jnp.int32), 0)
        raw_end = inventory.astype(jnp.int32) - MARKET_MIN_INVENTORY
        high = jnp.minimum(q, jnp.maximum(raw_end - MARKET_LUT_SIZE, 0))
        middle_end = jnp.clip(raw_end, 0, MARKET_LUT_SIZE)
        middle = jnp.minimum(q - high, middle_end)
        middle_start = middle_end - middle
        low = q - high - middle
        return (
            high * tables.market_price[safe_item, -1].astype(jnp.int32)
            + tables.market_price_prefix[safe_item, middle_end]
            - tables.market_price_prefix[safe_item, middle_start]
            + low * tables.market_price[safe_item, 0].astype(jnp.int32)
        ).astype(jnp.int32)

    if buy:
        limit = jnp.minimum(quantity, room).astype(jnp.int32)

        def affordable_body(_, bounds):
            lower, upper = bounds
            middle = (lower + upper + 1) // 2
            affordable = backward_sum(middle) <= cash
            return (
                jnp.where(affordable, middle, lower),
                jnp.where(affordable, upper, middle - 1),
            )

        filled, _ = jax.lax.fori_loop(
            0,
            8,
            affordable_body,
            (jnp.zeros_like(limit), limit),
        )
        value = backward_sum(filled)
    else:
        filled = jnp.minimum(quantity, available).astype(jnp.int32)
        raw_index = inventory.astype(jnp.int32) - MARKET_MIN_INVENTORY
        current_index = jnp.clip(raw_index, 0, MARKET_LUT_SIZE - 1)
        current_price = tables.market_price[
            safe_item, current_index
        ].astype(jnp.int32)
        floor_index = tables.market_first_floor_index[safe_item].astype(jnp.int32)
        floor_inventory = MARKET_MIN_INVENTORY + floor_index
        supplied = jnp.where(
            current_price > 1,
            jnp.minimum(
                filled,
                jnp.where(
                    floor_index < MARKET_LUT_SIZE,
                    jnp.maximum(floor_inventory - inventory, 0),
                    filled,
                ),
            ),
            0,
        ).astype(jnp.int32)
        value = forward_sum(supplied) + (filled - supplied)
    valid = (item >= 0) & (item < NUM_PRODUCTS) & (quantity > 0)
    return jnp.where(valid, value, 0).astype(jnp.int32)


def animal_feed_cash_reserve_v1(
    states: State, tables: StaticTables, player: int
) -> jax.Array:
    """Cash protected for existing/pending animals until first product revenue."""

    tile_animals = states.tile_animal[:, player]
    counts = jnp.stack(
        [
            jnp.sum(tile_animals == animal, axis=(1, 2), dtype=jnp.int32)
            + states.shed[:, player, NUM_PRODUCTS + animal].astype(jnp.int32)
            + jnp.sum(
                states.unit_inventory[
                    :, player, :, NUM_PRODUCTS + animal
                ].astype(jnp.int32),
                axis=-1,
            )
            for animal in range(NUM_ANIMALS)
        ],
        axis=-1,
    )
    day = (states.step // TURNS_PER_DAY).astype(jnp.int32)
    remaining_days = jnp.maximum(29 - day, 0)
    reserve_days = jnp.minimum(
        remaining_days[:, None], _ANIMAL_FIRST[None, :].astype(jnp.int32)
    )
    wheat_available = (
        states.shed[:, player, WHEAT_ITEM].astype(jnp.int32)
        + jnp.sum(
            states.unit_inventory[:, player, :, WHEAT_ITEM].astype(jnp.int32),
            axis=-1,
        )
    )
    required = jnp.maximum(
        jnp.sum(counts * reserve_days, axis=-1, dtype=jnp.int32)
        - wheat_available,
        0,
    )[:, None]
    return exact_market_quote_v1(
        states,
        tables,
        jnp.zeros_like(required, dtype=jnp.int8),
        required,
        buy=True,
        player=player,
        enforce_resources=False,
    )[:, 0]


def _cycles(remaining_days: jax.Array, first: jax.Array, interval: jax.Array, cap: jax.Array) -> jax.Array:
    interval = jnp.maximum(interval, 1)
    return jnp.where(
        remaining_days >= first,
        jnp.minimum((remaining_days - first) // interval + 1, cap),
        0,
    ).astype(jnp.int16)


def build_full_econ_features_v1(
    states: State,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    tables: StaticTables,
    player: int,
    *,
    include_expected_risk: bool = True,
    include_scenario_risk: bool = True,
) -> EconFeaturesV1:
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    owner = jnp.clip(candidates.owner_unit.astype(jnp.int32), 0, MAX_UNITS - 1)
    x = jnp.clip(candidates.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(candidates.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    task = candidates.task_type
    unit_task = candidates.owner_unit >= 0
    item = candidates.item_id.astype(jnp.int32)
    product_item = jnp.clip(item, 0, NUM_PRODUCTS - 1)
    crop_item = jnp.clip(item, 0, NUM_CROPS - 1)
    animal_item = jnp.clip(item - NUM_PRODUCTS, 0, NUM_ANIMALS - 1)
    current_step = states.step[:, None].astype(jnp.int16)
    day = (states.step[:, None] // TURNS_PER_DAY).astype(jnp.int16)
    remaining_days = jnp.maximum(29 - day, 0).astype(jnp.int16)
    remaining_turns = jnp.maximum(
        EPISODE_STEPS - 1 - states.step[:, None].astype(jnp.int32), 0
    ).astype(jnp.int16)
    tile_kind = states.tile_kind[batch, player, y, x]
    tile_crop = states.tile_crop[batch, player, y, x]
    tile_animal = states.tile_animal[batch, player, y, x]
    tile_yield = states.tile_yield[batch, player, y, x].astype(jnp.int16)
    tile_flags = states.tile_flags[batch, player, y, x]
    tile_neglect = states.tile_neglect[batch, player, y, x].astype(jnp.int16)
    tile_origin = states.tile_origin_day[batch, player, y, x].astype(jnp.int16)
    tile_max_life = states.tile_max_lifespan[batch, player, y, x].astype(jnp.int16)
    safe_existing_crop = jnp.clip(tile_crop.astype(jnp.int32), 0, NUM_CROPS - 1)
    safe_existing_animal = jnp.clip(tile_animal.astype(jnp.int32), 0, NUM_ANIMALS - 1)

    is_crop_route = unit_task & (task == TaskTypeV1.CROP_PRODUCTION)
    is_plant = is_crop_route & (tile_kind == TileKind.EMPTY)
    is_harvest_crop = is_crop_route & (tile_kind == TileKind.PLANT)
    is_seed_buy = (~unit_task) & (task == TaskTypeV1.CROP_PRODUCTION)
    is_water = unit_task & (task == TaskTypeV1.WATER_CROP)
    is_clear = unit_task & (task == TaskTypeV1.CLEAR_OR_REMOVE_TILE)
    is_apply_fertilizer = unit_task & (task == TaskTypeV1.APPLY_FERTILIZER)
    is_build = unit_task & (task == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
    is_animal_purchase = (~unit_task) & (task == TaskTypeV1.ANIMAL_PURCHASE)
    is_place = unit_task & (task == TaskTypeV1.ANIMAL_PLACE)
    is_feed = unit_task & (task == TaskTypeV1.ANIMAL_FEED)
    is_care = unit_task & (task == TaskTypeV1.ANIMAL_CARE)
    is_animal_product = unit_task & (task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
    is_collect_fert = unit_task & (task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
    is_land = (~unit_task) & (task == TaskTypeV1.BUY_LAND)
    is_buy_product = (~unit_task) & (task == TaskTypeV1.BUY_PRODUCT)
    is_hire = (~unit_task) & (task == TaskTypeV1.HIRE_WORKER)
    is_recovery = unit_task & (
        (task == TaskTypeV1.SAFE_RECOVERY)
        | (task == TaskTypeV1.SHED_DEPOSIT)
        | (task == TaskTypeV1.SHED_PICKUP)
    )
    is_sell = (~unit_task) & (
        (task == TaskTypeV1.SELL_INVENTORY)
        | (task == TaskTypeV1.TERMINAL_LIQUIDATION)
    )

    target = jnp.stack((x, y), axis=-1).astype(jnp.int16)
    target_to_shed = jnp.min(
        jnp.sum(jnp.abs(target[..., None, :] - _SHED_ACCESS), axis=-1), axis=-1
    ).astype(jnp.int16)
    base_actions = jnp.maximum(
        feasibility.path_steps + feasibility.operation_steps, 1
    ).astype(jnp.int16)
    transport = jnp.where(
        is_harvest_crop | is_animal_product | is_collect_fert | is_recovery,
        target_to_shed + 1,
        jnp.where(is_place | is_feed | is_apply_fertilizer, feasibility.path_steps, 0),
    ).astype(jnp.int16)

    crop_cycles = _cycles(
        remaining_days,
        _CROP_FIRST[crop_item],
        _CROP_INTERVAL[crop_item],
        _CROP_MAX[crop_item],
    )
    crop_quantity = jnp.where(
        _CROP_ONGOING[crop_item], crop_cycles, jnp.where(crop_cycles > 0, _CROP_MAX[crop_item], 0)
    ).astype(jnp.int16)
    crop_steps = (
        base_actions
        + _CROP_FIRST[crop_item] * TURNS_PER_DAY
        + target_to_shed
        + 2
    ).astype(jnp.int16)
    crop_town = known_town_demand_before_sale_v1(states, crop_item, crop_steps)
    crop_gross = exact_market_quote_v1(
        states, tables, crop_item, crop_quantity, buy=False,
        known_town_demand=crop_town, player=player, enforce_resources=False
    ).astype(jnp.float32)
    crop_seed_cost = _SEED_COST[crop_item].astype(jnp.float32)
    crop_maintenance = jnp.where(crop_cycles > 0, _CROP_FIRST[crop_item], 0).astype(jnp.int16)
    crop_actions = (base_actions + crop_maintenance + target_to_shed + 1).astype(jnp.int16)
    crop_buy_bank_delta = crop_gross - crop_seed_cost

    harvest_arrival = current_step + feasibility.path_steps
    first_decay = jnp.where(
        current_step <= tile_max_life,
        tile_max_life,
        current_step + ((current_step - tile_max_life) & jnp.int16(1)),
    )
    decay_count = jnp.where(
        (tile_max_life >= 0) & (first_decay < harvest_arrival),
        ((harvest_arrival - 1 - first_decay) // 2) + 1,
        0,
    ).astype(jnp.int16)
    harvest_quantity = jnp.maximum(tile_yield - decay_count, 0).astype(jnp.int16)
    harvest_steps = (base_actions + 1).astype(jnp.int16)
    harvest_town = known_town_demand_before_sale_v1(
        states, safe_existing_crop, harvest_steps
    )
    harvest_gross = exact_market_quote_v1(
        states,
        tables,
        safe_existing_crop,
        harvest_quantity,
        buy=False,
        known_town_demand=harvest_town,
        player=player,
        enforce_resources=False,
    ).astype(jnp.float32)

    animal_for_route = jnp.where(
        is_animal_product | is_feed | is_care | is_collect_fert,
        safe_existing_animal,
        jnp.where(is_build, jnp.clip(item, 0, NUM_ANIMALS - 1), animal_item),
    ).astype(jnp.int32)
    animal_cycles = _cycles(
        remaining_days,
        _ANIMAL_FIRST[animal_for_route],
        _ANIMAL_INTERVAL[animal_for_route],
        jnp.full_like(animal_for_route, 30),
    )
    animal_product_item = _ANIMAL_PRODUCT[animal_for_route].astype(jnp.int32)
    animal_steps = (
        base_actions
        + _ANIMAL_FIRST[animal_for_route] * TURNS_PER_DAY
        + target_to_shed
        + 2
    ).astype(jnp.int16)
    animal_town = known_town_demand_before_sale_v1(
        states, animal_product_item, animal_steps
    )
    animal_gross = exact_market_quote_v1(
        states,
        tables,
        animal_product_item,
        animal_cycles,
        buy=False,
        known_town_demand=animal_town,
        player=player,
        enforce_resources=False,
    ).astype(jnp.float32)
    wheat_days = jnp.minimum(remaining_days, jnp.int16(30)).astype(jnp.int16)
    wheat_cost = exact_market_quote_v1(
        states,
        tables,
        jnp.full_like(item, WHEAT_ITEM),
        wheat_days,
        buy=True,
        player=player,
        enforce_resources=False,
    ).astype(jnp.float32)
    wheat_cash_days = jnp.minimum(
        wheat_days, _ANIMAL_FIRST[animal_for_route]
    ).astype(jnp.int16)
    wheat_cash_reserve = exact_market_quote_v1(
        states,
        tables,
        jnp.full_like(item, WHEAT_ITEM),
        wheat_cash_days,
        buy=True,
        player=player,
        enforce_resources=False,
    ).astype(jnp.float32)
    animal_actions = (
        base_actions + wheat_days + animal_cycles * 2 + 2
    ).astype(jnp.int16)
    animal_bank_delta = (
        animal_gross
        - _ANIMAL_COST[animal_for_route].astype(jnp.float32)
        - wheat_cost
    )

    direct_steps = jnp.ones_like(feasibility.path_steps, dtype=jnp.int16)
    direct_town = known_town_demand_before_sale_v1(states, product_item, direct_steps)
    direct_sell_quote = exact_market_quote_v1(
        states,
        tables,
        product_item,
        candidates.quantity,
        buy=False,
        known_town_demand=direct_town,
        player=player,
    ).astype(jnp.float32)
    buy_product_cost = exact_market_quote_v1(
        states,
        tables,
        product_item,
        candidates.quantity,
        buy=True,
        player=player,
    ).astype(jnp.float32)

    animal_collect_steps = (base_actions + 1).astype(jnp.int16)
    animal_collect_town = known_town_demand_before_sale_v1(
        states, animal_product_item, animal_collect_steps
    )
    animal_collect_gross = exact_market_quote_v1(
        states,
        tables,
        animal_product_item,
        tile_yield,
        buy=False,
        known_town_demand=animal_collect_town,
        player=player,
        enforce_resources=False,
    ).astype(jnp.float32)
    fertilizer_sell_value = exact_market_quote_v1(
        states,
        tables,
        jnp.full_like(item, FERTILIZER_ITEM),
        jnp.ones_like(candidates.quantity),
        buy=False,
        player=player,
        enforce_resources=False,
    ).astype(jnp.float32)
    fertilizer_bonus_units = jnp.minimum(crop_cycles, 3).astype(jnp.int16)
    fertilizer_bonus_value = exact_market_quote_v1(
        states,
        tables,
        crop_item,
        fertilizer_bonus_units,
        buy=False,
        player=player,
        enforce_resources=False,
    ).astype(jnp.float32)

    watered = (tile_flags & jnp.uint8(FLAG_WATERED)) != 0
    water_preserved = jnp.where(
        tile_neglect >= 1,
        exact_market_quote_v1(
            states,
            tables,
            safe_existing_crop,
            jnp.maximum(tile_yield, 1),
            buy=False,
            player=player,
            enforce_resources=False,
        ).astype(jnp.float32),
        exact_market_quote_v1(
            states,
            tables,
            safe_existing_crop,
            jnp.ones_like(tile_yield),
            buy=False,
            player=player,
            enforce_resources=False,
        ).astype(jnp.float32),
    )
    feed_preserved = jnp.where(
        tile_neglect >= 1,
        _ANIMAL_COST[safe_existing_animal].astype(jnp.float32) + animal_gross * 0.5,
        exact_market_quote_v1(
            states,
            tables,
            _ANIMAL_PRODUCT[safe_existing_animal].astype(jnp.int32),
            jnp.ones_like(tile_yield),
            buy=False,
            player=player,
            enforce_resources=False,
        ).astype(jnp.float32),
    )
    wheat_opportunity = exact_market_quote_v1(
        states,
        tables,
        jnp.full_like(item, WHEAT_ITEM),
        jnp.ones_like(candidates.quantity),
        buy=False,
        player=player,
        enforce_resources=False,
    ).astype(jnp.float32)
    care_value = exact_market_quote_v1(
        states,
        tables,
        _ANIMAL_PRODUCT[safe_existing_animal].astype(jnp.int32),
        jnp.ones_like(tile_yield),
        buy=False,
        player=player,
        enforce_resources=False,
    ).astype(jnp.float32)
    care_realizable = (animal_cycles > 0) & (tile_yield < _ANIMAL_MAX[safe_existing_animal])
    care_value = jnp.where(care_realizable, care_value, 0.0)

    land_index = jnp.clip(states.unlocked_count[:, player, None].astype(jnp.int32) - 1, 0, 2)
    land_cost = _LAND_PRICES[land_index].astype(jnp.float32)
    best_crop_net = jnp.max(
        jnp.stack(
            [
                exact_market_quote_v1(
                    states,
                    tables,
                    jnp.full_like(item, crop_id),
                    jnp.full_like(candidates.quantity, CROP_MAX_YIELD[crop_id]),
                    buy=False,
                    player=player,
                    enforce_resources=False,
                ).astype(jnp.float32)
                - float(CROP_SEED_COST[crop_id])
                for crop_id in range(NUM_CROPS)
            ],
            axis=-1,
        ),
        axis=-1,
    )
    usable_land_routes = jnp.minimum(remaining_days // 3, 5).astype(jnp.float32)
    land_net = best_crop_net * usable_land_routes - land_cost

    hire_n = states.hires_today[:, player, None].astype(jnp.int32)
    hire_cost = _HIRE_COST[jnp.clip(hire_n, 0, MAX_HANDS - 1)].astype(jnp.float32)
    turns_left_today = (TURNS_PER_DAY - (states.step[:, None] % TURNS_PER_DAY) - 1).astype(jnp.float32)
    hire_net = turns_left_today * 6.0 - hire_cost

    feed_value = feed_preserved
    collect_fert_value = jnp.maximum(fertilizer_sell_value, fertilizer_bonus_value)
    apply_fert_value = fertilizer_bonus_value
    fertilizer_buy_route_value = (
        fertilizer_bonus_value
        - fertilizer_sell_value
        - ACTION_OPPORTUNITY_COST
        - buy_product_cost
    )
    build_value = animal_bank_delta
    purchase_value = animal_bank_delta
    place_value = jnp.maximum(animal_gross - wheat_cost, 0.0)
    recovery_inventory = states.unit_inventory[batch, player, owner, :NUM_PRODUCTS]
    recovery_value = jnp.sum(
        recovery_inventory.astype(jnp.float32)
        * states.market_price[:, None, :].astype(jnp.float32),
        axis=-1,
    )

    expected_bank = jnp.zeros_like(feasibility.path_steps, dtype=jnp.float32)
    expected_bank = jnp.where(is_sell, direct_sell_quote, expected_bank)
    expected_bank = jnp.where(is_seed_buy, crop_buy_bank_delta, expected_bank)
    expected_bank = jnp.where(is_plant, crop_gross, expected_bank)
    expected_bank = jnp.where(is_harvest_crop, harvest_gross, expected_bank)
    expected_bank = jnp.where(is_water, water_preserved, expected_bank)
    expected_bank = jnp.where(is_clear, jnp.maximum(best_crop_net * 0.25, 0.0), expected_bank)
    expected_bank = jnp.where(is_apply_fertilizer, apply_fert_value, expected_bank)
    expected_bank = jnp.where(is_build, build_value, expected_bank)
    expected_bank = jnp.where(is_animal_purchase, purchase_value, expected_bank)
    expected_bank = jnp.where(is_place, place_value, expected_bank)
    expected_bank = jnp.where(is_feed, feed_value, expected_bank)
    expected_bank = jnp.where(is_care, care_value, expected_bank)
    expected_bank = jnp.where(is_animal_product, animal_collect_gross, expected_bank)
    expected_bank = jnp.where(is_collect_fert, collect_fert_value, expected_bank)
    expected_bank = jnp.where(is_land, land_net, expected_bank)
    expected_bank = jnp.where(
        is_buy_product,
        jnp.where(
            item == WHEAT_ITEM,
            feed_preserved * candidates.quantity.astype(jnp.float32) - buy_product_cost,
            fertilizer_buy_route_value,
        ),
        expected_bank,
    )
    expected_bank = jnp.where(is_hire, hire_net, expected_bank)
    expected_bank = jnp.where(is_recovery, recovery_value, expected_bank)

    maintenance = jnp.where(
        is_seed_buy | is_plant,
        crop_maintenance,
        jnp.where(is_build | is_animal_purchase | is_place, wheat_days, jnp.where(is_feed | is_water, 1, 0)),
    ).astype(jnp.int16)
    total_actions = jnp.where(
        is_seed_buy | is_plant,
        crop_actions,
        jnp.where(
            is_build | is_animal_purchase | is_place,
            animal_actions,
            base_actions + transport,
        ),
    ).astype(jnp.int16)
    steps_to_revenue = jnp.where(
        is_sell,
        1,
        jnp.where(
            is_seed_buy | is_plant,
            crop_steps,
            jnp.where(
                is_build | is_animal_purchase | is_place | is_feed | is_care,
                animal_steps,
                base_actions + 1,
            ),
        ),
    ).astype(jnp.int16)
    cycles = jnp.where(
        is_seed_buy | is_plant,
        crop_cycles,
        jnp.where(is_build | is_animal_purchase | is_place | is_feed | is_care, animal_cycles, 1),
    ).astype(jnp.int16)
    plot_duration = jnp.where(
        is_seed_buy | is_plant,
        _CROP_FIRST[crop_item] * TURNS_PER_DAY,
        jnp.where(is_build | is_animal_purchase | is_place, remaining_turns, 0),
    ).astype(jnp.int16)
    storage = jnp.maximum(feasibility.shed_reserved_in, 0).astype(jnp.int16)

    route_cash = feasibility.cash_required.astype(jnp.int32)
    route_cash = jnp.where(is_seed_buy, _SEED_COST[crop_item], route_cash)
    route_cash = jnp.where(is_hire, _HIRE_COST[jnp.clip(hire_n, 0, MAX_HANDS - 1)], route_cash)
    route_cash = jnp.where(is_land, _LAND_PRICES[land_index], route_cash)
    route_cash = jnp.where(is_buy_product, buy_product_cost.astype(jnp.int32), route_cash)
    route_cash = jnp.where(is_animal_purchase, _ANIMAL_COST[animal_item], route_cash)
    future_route_cash = jnp.where(
        is_build | is_animal_purchase,
        _ANIMAL_COST[animal_for_route] + wheat_cash_reserve.astype(jnp.int32),
        jnp.where(is_place, wheat_cash_reserve.astype(jnp.int32), route_cash),
    )
    cash_after = states.money[:, player, None].astype(jnp.int32) - future_route_cash
    operating_feed_reserve = animal_feed_cash_reserve_v1(
        states, tables, player
    )[:, None]
    minimum_cash = cash_after - operating_feed_reserve
    input_opportunity = jnp.where(
        is_plant,
        crop_seed_cost,
        jnp.where(
            is_feed,
            wheat_opportunity,
            jnp.where(is_apply_fertilizer, fertilizer_sell_value, 0.0),
        ),
    ).astype(jnp.float32)
    action_cost = total_actions.astype(jnp.float32) * ACTION_OPPORTUNITY_COST
    shed_used = jnp.sum(states.shed[:, player].astype(jnp.int32), axis=-1)[:, None]
    overflow_units = jnp.maximum(shed_used + storage.astype(jnp.int32) - SHED_CAPACITY, 0)
    unit_price = states.market_price[batch, product_item].astype(jnp.float32)
    overflow_loss = overflow_units.astype(jnp.float32) * unit_price
    decay_loss = jnp.where(
        is_harvest_crop,
        decay_count.astype(jnp.float32) * states.market_price[batch, safe_existing_crop],
        jnp.where(
            is_animal_product & (tile_yield >= _ANIMAL_MAX[safe_existing_animal]),
            states.market_price[batch, animal_product_item].astype(jnp.float32),
            0.0,
        ),
    ).astype(jnp.float32)

    sale_item = jnp.where(
        is_seed_buy | is_plant | is_harvest_crop | is_water,
        jnp.where(is_harvest_crop | is_water, safe_existing_crop, crop_item),
        jnp.where(
            is_build | is_animal_purchase | is_place | is_feed | is_care | is_animal_product,
            animal_product_item,
            product_item,
        ),
    ).astype(jnp.int32)
    known_town = known_town_demand_before_sale_v1(states, sale_item, steps_to_revenue)
    known_market_impact = jnp.where(
        is_sell,
        jnp.where(direct_sell_quote > 0, candidates.quantity, 0),
        jnp.where(is_buy_product, -candidates.quantity, 0),
    ).astype(jnp.int32)

    expected_risk = jnp.where(
        is_land | is_seed_buy | is_plant,
        remaining_days.astype(jnp.float32) * WEED_SPAWN_CHANCE * ACTION_OPPORTUNITY_COST,
        0.0,
    )
    opponent = 1 - player
    opponent_visible_supply = jnp.sum(
        states.tile_yield[:, opponent].astype(jnp.float32), axis=(1, 2)
    )[:, None]
    scenario_risk = jnp.where(
        steps_to_revenue > 1,
        0.01 * opponent_visible_supply * unit_price,
        0.0,
    )
    expected_risk_enabled = jnp.asarray(include_expected_risk, dtype=jnp.bool_)
    scenario_risk_enabled = jnp.asarray(include_scenario_risk, dtype=jnp.bool_)
    stochastic_risk = (
        jnp.where(expected_risk_enabled, expected_risk, 0.0)
        + jnp.where(scenario_risk_enabled, scenario_risk, 0.0)
    ).astype(jnp.float32)

    bankable = feasibility.bankable_before_terminal & (
        current_step.astype(jnp.int32) + steps_to_revenue.astype(jnp.int32)
        <= EPISODE_STEPS - 1
    )
    expected_bank = jnp.where(bankable, expected_bank, 0.0)
    expected_bank = jnp.where(feasibility.legal_now, expected_bank, 0.0)
    long_route = (
        is_seed_buy | is_plant | is_build | is_animal_purchase | is_place | is_feed | is_care | is_land | is_apply_fertilizer
    )
    exact_mask = jnp.full_like(
        feasibility.cash_required, ECON_ALL_FIELDS_MASK, dtype=jnp.int32
    )
    exact_mask = jnp.where(
        long_route,
        exact_mask & jnp.int32(~ECON_EXPECTED_BANK_BIT),
        exact_mask,
    )
    expected_risk_bit = jnp.where(
        expected_risk_enabled,
        jnp.int32(ECON_STOCHASTIC_RISK_BIT),
        jnp.int32(0),
    )
    expected_mask = jnp.where(
        long_route,
        jnp.int32(ECON_EXPECTED_BANK_BIT) | expected_risk_bit,
        expected_risk_bit,
    ).astype(jnp.int32)
    scenario_mask = jnp.where(
        scenario_risk_enabled & (steps_to_revenue > 1),
        jnp.int32(ECON_STOCHASTIC_RISK_BIT),
        jnp.int32(0),
    ).astype(jnp.int32)
    terminal_salvage = jnp.zeros_like(expected_bank, dtype=jnp.float32)
    del watered, tile_origin
    return EconFeaturesV1(
        expected_bank_delta_current_price=expected_bank.astype(jnp.float32),
        cash_required=future_route_cash.astype(jnp.int32),
        cash_flow_before_revenue=(-future_route_cash).astype(jnp.int32),
        minimum_cash_during_task=minimum_cash.astype(jnp.int32),
        steps_to_first_revenue=steps_to_revenue,
        total_required_actions=total_actions,
        maintenance_actions=maintenance,
        plot_occupancy_duration=plot_duration,
        storage_required=storage,
        cycles_before_terminal=cycles,
        bankable_before_terminal=bankable,
        simple_cash_buffer_after_commit=cash_after.astype(jnp.int32),
        action_opportunity_cost=action_cost,
        transport_actions=transport,
        input_opportunity_cost=input_opportunity,
        expected_overflow_loss=overflow_loss,
        expected_decay_or_capacity_loss=decay_loss,
        known_market_impact=known_market_impact,
        known_town_demand_before_sale=known_town,
        stochastic_event_risk=stochastic_risk,
        terminal_salvage_value=terminal_salvage,
        exact_field_mask=exact_mask,
        expected_field_mask=expected_mask,
        scenario_field_mask=scenario_mask,
    )


def full_econ_score_v1(
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    econ: EconFeaturesV1,
) -> jax.Array:
    net = (
        econ.expected_bank_delta_current_price
        - econ.action_opportunity_cost
        - econ.input_opportunity_cost
        - econ.expected_overflow_loss
        - econ.expected_decay_or_capacity_loss
        - econ.stochastic_event_risk
    )
    buffer_shortfall = jnp.maximum(
        SIMPLE_CASH_BUFFER - econ.minimum_cash_during_task, 0
    ).astype(jnp.float32)
    optional_penalty = jnp.where(
        (~candidates.mandatory) & (econ.cash_required > 0),
        buffer_shortfall * 2.0,
        0.0,
    )
    score = net - optional_penalty - candidates.source_slot.astype(jnp.float32) * 1.0e-4
    score = score + jnp.where(candidates.mandatory, 100_000.0, 0.0)
    score = score + jnp.where(
        candidates.task_type == TaskTypeV1.TERMINAL_LIQUIDATION, 50_000.0, 0.0
    )
    # An explicitly activated replay profile is an expert scheduling prior.
    # It outranks the generic E5 heuristic, while legality, cash, capacity and
    # shared-ledger checks still remask every selection after each card.
    score = jnp.where(
        candidates.replay_priority > 0,
        200_000.0 + candidates.replay_priority + score * 1.0e-3,
        score,
    )
    return jnp.where(feasibility.legal_now, score, -1.0e9).astype(jnp.float32)


def select_full_econ_candidates_v1(
    states: State,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    econ: EconFeaturesV1,
    controller: ControllerStateV1,
    player: int,
    max_selections: int = MAX_SELECTIONS_V1,
) -> E2SelectionV1:
    score = full_econ_score_v1(candidates, feasibility, econ)
    eligible = (
        feasibility.legal_now
        & econ.bankable_before_terminal
        & (
            (score > 0.0)
            | candidates.mandatory
            | (candidates.task_type == TaskTypeV1.TERMINAL_LIQUIDATION)
            | (candidates.task_type == TaskTypeV1.SAFE_RECOVERY)
        )
    )
    # Core legality keeps only immediately payable cash.  The policy selector
    # additionally reserves the larger lifecycle cash commitment so that a
    # profitable-looking animal route cannot be selected together with other
    # spending that makes its future feed impossible.  This is an Econ policy
    # constraint only; it never changes ``feasibility.legal_now``.
    selection_feasibility = feasibility._replace(
        cash_required=jnp.where(
            candidates.mandatory,
            0,
            jnp.maximum(feasibility.cash_required, econ.cash_required),
        ).astype(jnp.int32)
    )
    ledger = initialize_e2_ledger_v1(states, controller, player)
    operating_feed_reserve = jnp.max(
        econ.simple_cash_buffer_after_commit - econ.minimum_cash_during_task,
        axis=-1,
    ).astype(jnp.int32)
    mandatory_cash = jnp.sum(
        jnp.where(
            candidates.mandatory & feasibility.legal_now,
            feasibility.cash_required,
            0,
        ),
        axis=-1,
        dtype=jnp.int32,
    )
    protected_cash = jnp.minimum(
        operating_feed_reserve,
        jnp.maximum(ledger.cash_available - mandatory_cash, 0),
    )
    ledger = ledger._replace(
        cash_reserved=ledger.cash_reserved + protected_cash
    )
    return select_candidates_with_scores_v1(
        states,
        candidates,
        selection_feasibility,
        score,
        controller,
        ledger,
        player,
        max_selections,
        eligible,
    )
