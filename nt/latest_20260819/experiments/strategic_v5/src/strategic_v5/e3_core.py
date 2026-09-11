"""E3 Animal-core candidate, feasibility, and reservation integration."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_COST,
    ANIMAL_MAX_HELD,
    ANIMAL_PRODUCT,
    ANIMAL_STRUCTURE,
    BOARD_SIZE,
    EPISODE_STEPS,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    MARKET_MIN_INVENTORY,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_PRODUCTS,
    SHED_ACCESS,
    SHED_CAPACITY,
    TileKind,
)
from kaggriculture_jax.types import State, StaticTables

from .constants import (
    CandidateSourceV1,
    MAX_CANDIDATES_V1,
    TaskStatusV1,
    TaskTypeV1,
)
from .e2_core import (
    E2SelectionV1,
    _empty_candidates,
    _empty_feasibility,
    initialize_e2_ledger_v1,
    select_candidates_with_scores_v1,
)
from .schema import CandidateV1, ControllerStateV1, FeasibilityV1


WHEAT_ITEM = 0
FERTILIZER_ITEM = 8
_ANIMAL_COST = jnp.asarray(ANIMAL_COST, dtype=jnp.int32)
_ANIMAL_STRUCTURE = jnp.asarray(ANIMAL_STRUCTURE, dtype=jnp.int8)
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int8)
_ANIMAL_MAX = jnp.asarray(ANIMAL_MAX_HELD, dtype=jnp.int16)
_TILE_X = jnp.tile(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
_TILE_Y = jnp.repeat(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
_TILE_ID = jnp.arange(BOARD_SIZE * BOARD_SIZE, dtype=jnp.int32)
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)
_INVALID_KEY = jnp.int32(1_000_000_000)


def _product_buy_total_quote(
    states: State, tables: StaticTables, item: int, quantity: jax.Array
) -> jax.Array:
    offsets = jnp.arange(MAX_UNITS, dtype=jnp.int32)[None, None, :]
    inventory = (
        states.market_inventory[:, item, None, None].astype(jnp.int32)
        - 1
        - offsets
    )
    index = jnp.clip(
        inventory - MARKET_MIN_INVENTORY, 0, tables.market_price.shape[1] - 1
    )
    quotes = tables.market_price[item, index].astype(jnp.int32)
    return jnp.sum(
        jnp.where(offsets < quantity[..., None], quotes, 0),
        axis=-1,
        dtype=jnp.int32,
    )


def _nearest_matching_tile(mask: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    key = jnp.where(mask, _TILE_ID[None, :], _INVALID_KEY)
    index = jnp.argmin(key, axis=-1)
    valid = jnp.take_along_axis(key, index[:, None], axis=-1)[:, 0] < _INVALID_KEY
    return index, _TILE_X[index].astype(jnp.int8), _TILE_Y[index].astype(jnp.int8)


def build_e3_candidates_v1(
    states: State,
    controller: ControllerStateV1,
    tables: StaticTables,
    player: int,
) -> CandidateV1:
    """Build market animal/feed candidates plus one urgent unit task per unit."""

    del tables
    batch_size = states.step.shape[0]
    candidates = _empty_candidates(batch_size)
    terminal = states.step >= EPISODE_STEPS - 2
    free_market = jnp.any(
        controller.market_tasks.status != TaskStatusV1.ACTIVE, axis=-1
    )
    shed_used = jnp.sum(states.shed[:, player].astype(jnp.int32), axis=-1)
    kinds = states.tile_kind[:, player].reshape(batch_size, -1)
    animals = states.tile_animal[:, player].reshape(batch_size, -1)
    flags = states.tile_flags[:, player].reshape(batch_size, -1)
    yields = states.tile_yield[:, player].reshape(batch_size, -1)
    neglect = states.tile_neglect[:, player].reshape(batch_size, -1)
    animal_present = animals >= 0

    unfed = animal_present & ((flags & jnp.uint8(FLAG_FED)) == 0)
    feed_demand = jnp.sum(unfed, axis=-1, dtype=jnp.int16)
    wheat_available = (
        states.shed[:, player, WHEAT_ITEM].astype(jnp.int16)
        + jnp.sum(
            states.unit_inventory[:, player, :, WHEAT_ITEM].astype(jnp.int16), axis=-1
        )
    )
    wheat_deficit = jnp.maximum(feed_demand - wheat_available, 0).astype(jnp.int16)
    wheat_buy = jnp.minimum(
        wheat_deficit, (SHED_CAPACITY - shed_used).astype(jnp.int16)
    )
    buy_wheat_present = (wheat_buy > 0) & free_market & (~terminal)
    candidates = candidates._replace(
        task_type=candidates.task_type.at[:, 0].set(jnp.int8(TaskTypeV1.BUY_PRODUCT)),
        item_id=candidates.item_id.at[:, 0].set(jnp.int8(WHEAT_ITEM)),
        quantity=candidates.quantity.at[:, 0].set(wheat_buy),
        source=candidates.source.at[:, 0].set(jnp.int8(CandidateSourceV1.ANIMAL)),
        mandatory=candidates.mandatory.at[:, 0].set(buy_wheat_present),
        present=candidates.present.at[:, 0].set(buy_wheat_present),
        hard_mask=candidates.hard_mask.at[:, 0].set(buy_wheat_present),
    )

    flattened_kind = kinds
    empty_structure_by_animal = (
        flattened_kind[:, None, :] == _ANIMAL_STRUCTURE[None, :, None]
    ) & (animals[:, None, :] < 0)
    structure_key = jnp.where(
        empty_structure_by_animal,
        _TILE_ID[None, None, :],
        _INVALID_KEY,
    )
    structure_index = jnp.argmin(structure_key, axis=-1)
    structure_valid = (
        jnp.take_along_axis(structure_key, structure_index[..., None], axis=-1)[..., 0]
        < _INVALID_KEY
    )
    structure_x = _TILE_X[structure_index].astype(jnp.int8)
    structure_y = _TILE_Y[structure_index].astype(jnp.int8)
    animal_shed = states.shed[:, player, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS]
    animal_carried = jnp.sum(
        states.unit_inventory[
            :, player, :, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
        ],
        axis=1,
        dtype=jnp.int16,
    )
    no_stock = (animal_shed + animal_carried) <= 0
    can_afford = states.money[:, player, None] >= _ANIMAL_COST[None, :]
    purchase_present = (
        structure_valid
        & no_stock
        & can_afford
        & (shed_used[:, None] < SHED_CAPACITY)
        & free_market[:, None]
        & (~terminal[:, None])
    )
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    purchase_slots = jnp.arange(NUM_ANIMALS, dtype=jnp.int32)[None, :] + 1
    candidates = candidates._replace(
        task_type=candidates.task_type.at[batch, purchase_slots].set(
            jnp.full((batch_size, NUM_ANIMALS), TaskTypeV1.ANIMAL_PURCHASE, dtype=jnp.int8)
        ),
        target_id=candidates.target_id.at[batch, purchase_slots].set(
            structure_index.astype(jnp.int16)
        ),
        target_x=candidates.target_x.at[batch, purchase_slots].set(structure_x),
        target_y=candidates.target_y.at[batch, purchase_slots].set(structure_y),
        item_id=candidates.item_id.at[batch, purchase_slots].set(
            NUM_PRODUCTS + jnp.arange(NUM_ANIMALS, dtype=jnp.int8)[None, :]
        ),
        quantity=candidates.quantity.at[batch, purchase_slots].set(jnp.int16(1)),
        source=candidates.source.at[batch, purchase_slots].set(
            jnp.full((batch_size, NUM_ANIMALS), CandidateSourceV1.ANIMAL, dtype=jnp.int8)
        ),
        present=candidates.present.at[batch, purchase_slots].set(purchase_present),
        hard_mask=candidates.hard_mask.at[batch, purchase_slots].set(purchase_present),
    )

    unit_active = states.unit_active[:, player]
    unit_free = unit_active & (controller.unit_tasks.status != TaskStatusV1.ACTIVE)
    positions = states.unit_pos[:, player].astype(jnp.int16)
    distance = jnp.sum(
        jnp.abs(
            positions[:, :, None, :]
            - jnp.stack((_TILE_X, _TILE_Y), axis=-1)[None, None, :, :]
        ),
        axis=-1,
    ).astype(jnp.int32)
    safe_animal = jnp.clip(animals.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    animal_product = _ANIMAL_PRODUCT[safe_animal]
    animal_max = _ANIMAL_MAX[safe_animal]
    product_ready = animal_present & (yields > 0)
    product_urgent = product_ready & (yields >= animal_max)
    fertilizer_ready = animal_present & (
        (flags & jnp.uint8(FLAG_FERTILIZER_AVAILABLE)) != 0
    )
    uncared = animal_present & ((flags & jnp.uint8(FLAG_CARED)) == 0)
    unit_wheat = states.unit_inventory[:, player, :, WHEAT_ITEM] > 0
    shed_wheat = states.shed[:, player, WHEAT_ITEM] > 0
    can_feed = unit_wheat | shed_wheat[:, None]

    available_animals = (
        states.shed[:, player, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS][:, None, :]
        + states.unit_inventory[
            :, player, :, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
        ]
    ) > 0
    coop_empty = (kinds == TileKind.COOP) & (animals < 0)
    pasture_empty = (kinds == TileKind.PASTURE) & (animals < 0)
    goose_available = available_animals[:, :, 0]
    cow_available = available_animals[:, :, 1]
    sheep_available = available_animals[:, :, 2]
    place_animal = jnp.where(
        coop_empty[:, None, :] & goose_available[..., None],
        0,
        jnp.where(
            pasture_empty[:, None, :] & cow_available[..., None],
            1,
            jnp.where(
                pasture_empty[:, None, :] & sheep_available[..., None], 2, -1
            ),
        ),
    ).astype(jnp.int8)
    place_possible = place_animal >= 0

    preferred_animal = (jnp.arange(MAX_UNITS, dtype=jnp.int8) % NUM_ANIMALS)[None, :, None]
    preferred_structure = _ANIMAL_STRUCTURE[preferred_animal]
    empty_for_build = kinds[:, None, :] == TileKind.EMPTY
    build_possible = empty_for_build & (~terminal[:, None, None])

    priority = jnp.full((batch_size, MAX_UNITS, BOARD_SIZE * BOARD_SIZE), 99, dtype=jnp.int32)
    task_for_tile = jnp.full_like(priority, TaskTypeV1.NONE, dtype=jnp.int8)
    item_for_tile = jnp.full_like(priority, -1, dtype=jnp.int8)

    feed_tile = unfed[:, None, :] & can_feed[..., None]
    priority = jnp.where(feed_tile, 0 + neglect[:, None, :] * -2, priority)
    task_for_tile = jnp.where(feed_tile, TaskTypeV1.ANIMAL_FEED, task_for_tile)
    item_for_tile = jnp.where(feed_tile, WHEAT_ITEM, item_for_tile)

    urgent_tile = product_urgent[:, None, :]
    replace = urgent_tile & (priority > 1)
    priority = jnp.where(replace, 1, priority)
    task_for_tile = jnp.where(replace, TaskTypeV1.ANIMAL_COLLECT_PRODUCT, task_for_tile)
    item_for_tile = jnp.where(replace, animal_product[:, None, :], item_for_tile)

    place_tile = place_possible
    replace = place_tile & (priority > 2)
    priority = jnp.where(replace, 2, priority)
    task_for_tile = jnp.where(replace, TaskTypeV1.ANIMAL_PLACE, task_for_tile)
    item_for_tile = jnp.where(replace, NUM_PRODUCTS + place_animal, item_for_tile)

    product_tile = product_ready[:, None, :]
    replace = product_tile & (priority > 3)
    priority = jnp.where(replace, 3, priority)
    task_for_tile = jnp.where(replace, TaskTypeV1.ANIMAL_COLLECT_PRODUCT, task_for_tile)
    item_for_tile = jnp.where(replace, animal_product[:, None, :], item_for_tile)

    fertilizer_tile = fertilizer_ready[:, None, :]
    replace = fertilizer_tile & (priority > 4)
    priority = jnp.where(replace, 4, priority)
    task_for_tile = jnp.where(
        replace, TaskTypeV1.ANIMAL_COLLECT_FERTILIZER, task_for_tile
    )
    item_for_tile = jnp.where(replace, FERTILIZER_ITEM, item_for_tile)

    care_tile = uncared[:, None, :]
    replace = care_tile & (priority > 5)
    priority = jnp.where(replace, 5, priority)
    task_for_tile = jnp.where(replace, TaskTypeV1.ANIMAL_CARE, task_for_tile)
    item_for_tile = jnp.where(replace, animals[:, None, :], item_for_tile)

    # Building is a low-priority expansion action and never masks maintenance.
    replace = build_possible & (priority > 8)
    priority = jnp.where(replace, 8, priority)
    task_for_tile = jnp.where(replace, TaskTypeV1.BUILD_ANIMAL_STRUCTURE, task_for_tile)
    item_for_tile = jnp.where(replace, preferred_animal, item_for_tile)
    del preferred_structure

    key = priority * 100_000 + distance * 100 + _TILE_ID
    key = jnp.where(priority < 99, key, _INVALID_KEY)
    target_index = jnp.argmin(key, axis=-1)
    target_key = jnp.take_along_axis(key, target_index[..., None], axis=-1)[..., 0]
    chosen_task = jnp.take_along_axis(task_for_tile, target_index[..., None], axis=-1)[..., 0]
    chosen_item = jnp.take_along_axis(item_for_tile, target_index[..., None], axis=-1)[..., 0]
    present = unit_free & (target_key < _INVALID_KEY) & (~terminal[:, None])
    slots = jnp.arange(MAX_UNITS, dtype=jnp.int32)[None, :] + 4
    candidates = candidates._replace(
        task_type=candidates.task_type.at[batch, slots].set(chosen_task),
        owner_unit=candidates.owner_unit.at[batch, slots].set(
            jnp.broadcast_to(jnp.arange(MAX_UNITS, dtype=jnp.int8), (batch_size, MAX_UNITS))
        ),
        target_id=candidates.target_id.at[batch, slots].set(target_index.astype(jnp.int16)),
        target_x=candidates.target_x.at[batch, slots].set(_TILE_X[target_index].astype(jnp.int8)),
        target_y=candidates.target_y.at[batch, slots].set(_TILE_Y[target_index].astype(jnp.int8)),
        item_id=candidates.item_id.at[batch, slots].set(chosen_item),
        quantity=candidates.quantity.at[batch, slots].set(jnp.int16(1)),
        source=candidates.source.at[batch, slots].set(
            jnp.full((batch_size, MAX_UNITS), CandidateSourceV1.ANIMAL, dtype=jnp.int8)
        ),
        mandatory=candidates.mandatory.at[batch, slots].set(chosen_task == TaskTypeV1.ANIMAL_FEED),
        present=candidates.present.at[batch, slots].set(present),
        hard_mask=candidates.hard_mask.at[batch, slots].set(present),
    )
    return candidates


def evaluate_e3_feasibility_v1(
    states: State,
    candidates: CandidateV1,
    tables: StaticTables,
    player: int,
) -> FeasibilityV1:
    batch_size = states.step.shape[0]
    result = _empty_feasibility(batch_size)
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    task = candidates.task_type
    is_buy_wheat = (task == TaskTypeV1.BUY_PRODUCT) & (candidates.item_id == WHEAT_ITEM)
    is_purchase = task == TaskTypeV1.ANIMAL_PURCHASE
    is_build = task == TaskTypeV1.BUILD_ANIMAL_STRUCTURE
    is_place = task == TaskTypeV1.ANIMAL_PLACE
    is_feed = task == TaskTypeV1.ANIMAL_FEED
    is_care = task == TaskTypeV1.ANIMAL_CARE
    is_harvest = task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT
    is_collect_fert = task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER
    unit_task = is_build | is_place | is_feed | is_care | is_harvest | is_collect_fert
    market_task = is_buy_wheat | is_purchase
    supported = unit_task | market_task
    owner = jnp.clip(candidates.owner_unit.astype(jnp.int32), 0, MAX_UNITS - 1)
    x = jnp.clip(candidates.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(candidates.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    position = states.unit_pos[batch, player, owner].astype(jnp.int16)
    target = jnp.stack((x, y), axis=-1).astype(jnp.int16)
    direct = jnp.sum(jnp.abs(position - target), axis=-1).astype(jnp.int16)
    via_shed = (
        jnp.sum(jnp.abs(position[..., None, :] - _SHED_ACCESS), axis=-1)
        + jnp.sum(jnp.abs(_SHED_ACCESS - target[..., None, :]), axis=-1)
    )
    via_shed = jnp.min(via_shed, axis=-1).astype(jnp.int16)
    target_to_shed = jnp.min(
        jnp.sum(jnp.abs(target[..., None, :] - _SHED_ACCESS), axis=-1), axis=-1
    ).astype(jnp.int16)

    item = jnp.clip(candidates.item_id.astype(jnp.int32), 0, NUM_PRODUCTS + NUM_ANIMALS - 1)
    unit_item_count = states.unit_inventory[batch, player, owner, item]
    shed_item_count = states.shed[batch, player, item]
    needs_item = is_place | is_feed
    pickup_required = needs_item & (unit_item_count <= 0)
    path = jnp.where(
        is_harvest | is_collect_fert,
        direct + target_to_shed,
        jnp.where(unit_task, jnp.where(pickup_required, via_shed, direct), 0),
    ).astype(jnp.int16)
    operation = jnp.where(
        is_harvest | is_collect_fert,
        2,
        jnp.where(unit_task, 1 + pickup_required.astype(jnp.int16), 1),
    ).astype(jnp.int16)
    expected_finish = (states.step[:, None].astype(jnp.int16) + path + operation).astype(jnp.int16)
    day_end = (((states.step[:, None] // 24) + 1) * 24).astype(jnp.int16)
    terminal = jnp.int16(EPISODE_STEPS - 2)
    deadline = jnp.where(unit_task, day_end, terminal).astype(jnp.int16)
    bankable = expected_finish <= terminal
    finishes_day = expected_finish <= day_end

    animal_id = candidates.item_id.astype(jnp.int32) - NUM_PRODUCTS
    safe_animal = jnp.clip(animal_id, 0, NUM_ANIMALS - 1)
    purchase_cost = (
        _ANIMAL_COST[safe_animal]
        * jnp.maximum(candidates.quantity.astype(jnp.int32), 1)
    )
    # Product orders are "buy up to N": the official engine commits one unit at
    # a time and legally stops when cash runs out.  Reserve the first unit so a
    # replay request such as BUY_PRODUCT WHEAT 6 is not rejected merely because
    # only five units can ultimately clear.
    wheat_first_quote = _product_buy_total_quote(
        states, tables, WHEAT_ITEM, jnp.ones_like(candidates.quantity, dtype=jnp.int32)
    )
    cash = jnp.where(
        is_purchase,
        purchase_cost,
        jnp.where(is_buy_wheat, wheat_first_quote, 0),
    ).astype(jnp.int32)
    tile_kind = states.tile_kind[batch, player, y, x]
    tile_animal = states.tile_animal[batch, player, y, x]
    tile_flags = states.tile_flags[batch, player, y, x]
    tile_yield = states.tile_yield[batch, player, y, x]
    unit_active = states.unit_active[batch, player, owner]
    structure_match = tile_kind == _ANIMAL_STRUCTURE[safe_animal]
    build_ok = is_build & unit_active & (tile_kind == TileKind.EMPTY)
    place_ok = (
        is_place
        & unit_active
        & structure_match
        & (tile_animal < 0)
        & ((unit_item_count > 0) | (shed_item_count > 0))
    )
    feed_ok = (
        is_feed
        & unit_active
        & (tile_animal >= 0)
        & ((tile_flags & jnp.uint8(FLAG_FED)) == 0)
        & ((unit_item_count > 0) | (shed_item_count > 0))
    )
    care_ok = (
        is_care
        & unit_active
        & (tile_animal >= 0)
        & ((tile_flags & jnp.uint8(FLAG_CARED)) == 0)
    )
    harvest_ok = is_harvest & unit_active & (tile_animal >= 0) & (tile_yield > 0)
    collect_ok = (
        is_collect_fert
        & unit_active
        & (tile_animal >= 0)
        & ((tile_flags & jnp.uint8(FLAG_FERTILIZER_AVAILABLE)) != 0)
    )
    shed_used = jnp.sum(states.shed[:, player].astype(jnp.int32), axis=-1)[:, None]
    market_capacity = shed_used + candidates.quantity <= SHED_CAPACITY
    purchase_ok = (
        is_purchase
        & (candidates.quantity > 0)
        & (states.money[:, player, None] >= purchase_cost)
        & market_capacity
    )
    buy_wheat_ok = (
        is_buy_wheat
        & (candidates.quantity > 0)
        & (states.money[:, player, None] >= wheat_first_quote)
        & market_capacity
    )
    task_ok = (
        purchase_ok
        | buy_wheat_ok
        | build_ok
        | place_ok
        | feed_ok
        | care_ok
        | harvest_ok
        | collect_ok
    )
    legal = (
        candidates.present
        & candidates.hard_mask
        & supported
        & task_ok
        & bankable
        & ((~unit_task) | finishes_day)
    )
    product_item = jnp.clip(candidates.item_id.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    harvest_quantity = jnp.where(is_harvest, tile_yield, 0).astype(jnp.int16)
    collect_quantity = is_collect_fert.astype(jnp.int16)
    market_in = jnp.where(market_task, candidates.quantity, 0).astype(jnp.int16)
    future_in = harvest_quantity + collect_quantity + market_in
    shed_item = jnp.where(
        is_collect_fert,
        FERTILIZER_ITEM,
        jnp.where(is_harvest, product_item, candidates.item_id),
    ).astype(jnp.int8)
    return result._replace(
        legal_now=legal,
        unit_required=unit_task,
        # Legacy single-animal purchase cards may reserve an intended structure;
        # opening shed-stock macros deliberately have target_id=-1 and do not.
        plot_required=unit_task | (is_purchase & (candidates.target_id >= 0)),
        path_steps=path,
        operation_steps=operation,
        expected_finish_step=expected_finish,
        deadline_step=deadline,
        bankable_before_terminal=bankable,
        cash_required=cash,
        shed_item=shed_item,
        shed_item_required=(pickup_required & needs_item).astype(jnp.int16),
        unit_item=candidates.item_id,
        unit_item_required=(needs_item & (~pickup_required)).astype(jnp.int16),
        shed_reserved_in=future_in,
        market_slots_required=market_task.astype(jnp.int8),
    )


def e3_priority_v1(candidates: CandidateV1, feasibility: FeasibilityV1) -> jax.Array:
    task = candidates.task_type
    base = jnp.where(
        candidates.mandatory,
        1000.0,
        jnp.where(
            task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT,
            800.0,
            jnp.where(
                task == TaskTypeV1.ANIMAL_PLACE,
                700.0,
                jnp.where(
                    task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER,
                    600.0,
                    jnp.where(
                        task == TaskTypeV1.ANIMAL_CARE,
                        500.0,
                        jnp.where(
                            task == TaskTypeV1.ANIMAL_PURCHASE,
                            400.0,
                            jnp.where(task == TaskTypeV1.BUILD_ANIMAL_STRUCTURE, 300.0, 900.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    score = base - feasibility.path_steps.astype(jnp.float32)
    return jnp.where(feasibility.legal_now, score, -1.0e9)


def select_e3_candidates_v1(
    states: State,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    controller: ControllerStateV1,
    player: int,
    max_selections: int = 8,
) -> E2SelectionV1:
    return select_candidates_with_scores_v1(
        states,
        candidates,
        feasibility,
        e3_priority_v1(candidates, feasibility),
        controller,
        initialize_e2_ledger_v1(states, controller, player),
        player,
        max_selections,
    )
