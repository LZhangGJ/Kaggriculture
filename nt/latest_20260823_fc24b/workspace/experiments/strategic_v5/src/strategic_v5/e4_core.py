"""E4 unified Full-core candidate pool, feasibility, and shared ledger selection.

The fixed pool exposes market operations and two operational routes per worker,
then arbitrates crop, land, fertilizer, animal, inventory, worker, and terminal
tasks through the single V5 ledger.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_COST,
    ANIMAL_MAX_HELD,
    ANIMAL_PRODUCT,
    ANIMAL_STRUCTURE,
    BOARD_SIZE,
    CROP_FIRST_YIELD_DAY,
    CROP_SEED_COST,
    EPISODE_STEPS,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_WATERED,
    LAND_PRICES,
    MAX_HANDS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    SHED_ACCESS,
    SHED_CAPACITY,
    TURNS_PER_DAY,
    TileKind,
)
from kaggriculture_jax.types import State, StaticTables

from .constants import (
    CandidateSourceV1,
    MAX_CANDIDATES_V1,
    MAX_SELECTIONS_V1,
    TaskStatusV1,
    TaskTypeV1,
)
from .e2_core import (
    E2SelectionV1,
    _empty_candidates,
    _empty_feasibility,
    evaluate_e2_feasibility_v1,
    initialize_e2_ledger_v1,
    select_candidates_with_scores_v1,
)
from .e3_core import evaluate_e3_feasibility_v1
from .schema import CandidateV1, ControllerStateV1, FeasibilityV1
from .task_cards import (
    DEFAULT_REPLAY_TASK_CARD_PROGRAM_V1,
    ReplayTaskCardProgramV1,
)


FERTILIZER_ITEM = 8
WHEAT_ITEM = 0
MARKET_SLOT_COUNT = 21
PRIMARY_UNIT_START = 30
SECONDARY_UNIT_START = PRIMARY_UNIT_START + MAX_UNITS
SECONDARY_UNIT_COUNT = MAX_CANDIDATES_V1 - SECONDARY_UNIT_START - NUM_CROPS - 2
PLANT_START = SECONDARY_UNIT_START + SECONDARY_UNIT_COUNT
BUILD_COOP_SLOT = PLANT_START + NUM_CROPS
BUILD_PASTURE_SLOT = BUILD_COOP_SLOT + 1

assert BUILD_PASTURE_SLOT == MAX_CANDIDATES_V1 - 1

_CROP_FIRST = jnp.asarray(CROP_FIRST_YIELD_DAY, dtype=jnp.int16)
_SEED_COST = jnp.asarray(CROP_SEED_COST, dtype=jnp.int32)
_ANIMAL_COST = jnp.asarray(ANIMAL_COST, dtype=jnp.int32)
_ANIMAL_STRUCTURE = jnp.asarray(ANIMAL_STRUCTURE, dtype=jnp.int8)
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int8)
_ANIMAL_MAX = jnp.asarray(ANIMAL_MAX_HELD, dtype=jnp.int16)
_LAND_PRICES = jnp.asarray(LAND_PRICES, dtype=jnp.int32)
_HIRE_COST = jnp.asarray(
    (
        1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
        987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368,
        75025, 121393, 196418, 317811, 514229, 832040, 1346269, 2178309,
    ),
    dtype=jnp.int32,
)
_HIRE_COST_PREFIX = jnp.concatenate(
    (jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(_HIRE_COST, dtype=jnp.int32))
)
_TILE_X = jnp.tile(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
_TILE_Y = jnp.repeat(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
_TILE_ID = jnp.arange(BOARD_SIZE * BOARD_SIZE, dtype=jnp.int32)
_TILE_DISTANCE = (
    jnp.abs(_TILE_X[:, None] - _TILE_X[None, :])
    + jnp.abs(_TILE_Y[:, None] - _TILE_Y[None, :])
).astype(jnp.int16)
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)
_INVALID_KEY = jnp.int32(1_000_000_000)


def build_full_core_candidates_v1(
    states: State,
    controller: ControllerStateV1,
    tables: StaticTables,
    player: int,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    *,
    route_profile: bool = False,
) -> CandidateV1:
    """Build one fixed unified candidate pool for every Full-core module."""

    del tables
    batch_size = states.step.shape[0]
    candidates = _empty_candidates(batch_size)
    terminal = states.step >= EPISODE_STEPS - 2
    day = (states.step // TURNS_PER_DAY).astype(jnp.int16)
    turn = states.step % TURNS_PER_DAY
    money = states.money[:, player].astype(jnp.int32)
    shed = states.shed[:, player]
    shed_used = jnp.sum(shed.astype(jnp.int32), axis=-1)
    unit_inventory = states.unit_inventory[:, player]
    unit_inventory_total = jnp.sum(unit_inventory.astype(jnp.int32), axis=-1)
    program = (
        DEFAULT_REPLAY_TASK_CARD_PROGRAM_V1
        if task_card_program is None
        else task_card_program
    )
    program_step = jnp.clip(states.step.astype(jnp.int32), 0, EPISODE_STEPS - 1)
    replay_enabled = program.enabled[program_step]
    replay_quantity = program.market_quantity[program_step]
    replay_priority = program.market_priority[program_step]
    replay_build_animal = program.build_animal_id[program_step]
    replay_build_priority = program.build_priority[program_step]

    # Market candidates: 0 land, 1 fertilizer, 2 wheat, 3..5 animals,
    # 6..10 seeds, 11 hire, 12..20 product sales.
    unlocked = states.unlocked_count[:, player].astype(jnp.int32)
    land_index = jnp.clip(unlocked - 1, 0, len(LAND_PRICES) - 1)
    land_cost = _LAND_PRICES[land_index]
    land_hint = replay_enabled & (replay_quantity[:, 0] > 0)
    plant_mask = states.tile_kind[:, player] == TileKind.PLANT
    fertilizer_needed = jnp.any(
        plant_mask
        & (
            states.tile_fertilized_until[:, player].astype(jnp.int16)
            < day[:, None, None] + 2
        ),
        axis=(1, 2),
    )
    fertilizer_available = (
        shed[:, FERTILIZER_ITEM].astype(jnp.int16)
        + jnp.sum(
            unit_inventory[:, :, FERTILIZER_ITEM].astype(jnp.int16), axis=-1
        )
    )
    fertilizer_hint = replay_enabled & (replay_quantity[:, 1] > 0)
    fertilizer_quantity = jnp.where(
        fertilizer_hint, replay_quantity[:, 1], jnp.int16(1)
    )
    kinds = states.tile_kind[:, player].reshape(batch_size, -1)
    animals = states.tile_animal[:, player].reshape(batch_size, -1)
    flags = states.tile_flags[:, player].reshape(batch_size, -1)
    yields = states.tile_yield[:, player].reshape(batch_size, -1)
    neglect = states.tile_neglect[:, player].reshape(batch_size, -1)
    animal_present = animals >= 0
    unfed = animal_present & ((flags & jnp.uint8(FLAG_FED)) == 0)
    wheat_available = shed[:, WHEAT_ITEM].astype(jnp.int16) + jnp.sum(
        unit_inventory[:, :, WHEAT_ITEM].astype(jnp.int16), axis=-1
    )
    wheat_deficit = jnp.maximum(
        jnp.sum(unfed, axis=-1, dtype=jnp.int16) - wheat_available, 0
    )
    wheat_hint = replay_enabled & (replay_quantity[:, 2] > 0)
    wheat_buy = jnp.where(wheat_hint, replay_quantity[:, 2], wheat_deficit)
    wheat_buy = jnp.minimum(
        wheat_buy, (SHED_CAPACITY - shed_used).astype(jnp.int16)
    )
    animal_shed = shed[:, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS]
    animal_carried = jnp.sum(
        unit_inventory[:, :, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS],
        axis=1,
        dtype=jnp.int16,
    )
    # Build the repeated animal/seed market slots as matrices.  These slots are
    # independent templates, so batching them avoids a chain of per-slot scatter
    # updates while preserving the exact fixed 96-candidate schema.
    matching_empty = (
        kinds[:, None, :] == _ANIMAL_STRUCTURE[None, :, None]
    ) & (animals[:, None, :] < 0)
    animal_key = jnp.where(
        matching_empty, _TILE_ID[None, None, :], _INVALID_KEY
    )
    animal_target = jnp.argmin(animal_key, axis=-1)
    animal_target_valid = jnp.min(animal_key, axis=-1) < _INVALID_KEY
    animal_stock = animal_shed + animal_carried
    replay_animal_quantity = replay_quantity[:, 3 : 3 + NUM_ANIMALS]
    animal_hint = replay_enabled[:, None] & (replay_animal_quantity > 0)
    animal_quantity = jnp.where(
        animal_hint, replay_animal_quantity, jnp.int16(1)
    ).astype(jnp.int16)
    animal_purchase_present = (
        (animal_hint | (animal_target_valid & (animal_stock <= 0)))
        & (
            money[:, None]
            >= _ANIMAL_COST[None, :] * animal_quantity.astype(jnp.int32)
        )
        & (
            shed_used[:, None] + animal_quantity.astype(jnp.int32)
            <= SHED_CAPACITY
        )
        & (~terminal[:, None])
    )
    animal_target_id = jnp.where(animal_hint, -1, animal_target).astype(jnp.int16)
    animal_target_x = jnp.where(
        animal_hint, -1, _TILE_X[animal_target]
    ).astype(jnp.int8)
    animal_target_y = jnp.where(
        animal_hint, -1, _TILE_Y[animal_target]
    ).astype(jnp.int8)

    remaining_days = 29 - day
    replay_seed_quantity = replay_quantity[:, 6 : 6 + NUM_CROPS]
    seed_hint = replay_enabled[:, None] & (replay_seed_quantity > 0)
    seed_quantity = jnp.where(
        seed_hint, replay_seed_quantity, jnp.int16(1)
    ).astype(jnp.int16)
    seed_present = (
        (seed_hint | (states.seeds[:, player] < 4))
        & (money[:, None] >= _SEED_COST[None, :] * seed_quantity.astype(jnp.int32))
        & (remaining_days[:, None] >= _CROP_FIRST[None, :])
        & (~terminal[:, None])
    )

    hire_n = states.hires_today[:, player].astype(jnp.int32)
    hire_hint = replay_enabled & (replay_quantity[:, 11] > 0)
    hire_quantity = jnp.where(
        hire_hint, replay_quantity[:, 11], jnp.int16(1)
    ).astype(jnp.int16)
    hire_end = jnp.clip(hire_n + hire_quantity.astype(jnp.int32), 0, MAX_HANDS)
    hire_cost = _HIRE_COST_PREFIX[hire_end] - _HIRE_COST_PREFIX[hire_n]
    hire_present = (
        (hire_quantity > 0)
        & (hire_end <= MAX_HANDS)
        & (money >= hire_cost)
        & (day < 29)
        & (turn < TURNS_PER_DAY - 1)
        & (~terminal)
    )
    inventory_by_product = jnp.sum(
        unit_inventory[:, :, :NUM_PRODUCTS].astype(jnp.int32), axis=1
    )
    available_quantity = jnp.where(
        terminal[:, None],
        shed[:, :NUM_PRODUCTS].astype(jnp.int32) + inventory_by_product,
        shed[:, :NUM_PRODUCTS].astype(jnp.int32),
    ).astype(jnp.int16)
    replay_sell_quantity = replay_quantity[:, 12 : 12 + NUM_PRODUCTS]
    sell_hint = replay_enabled[:, None] & (replay_sell_quantity > 0)
    sell_quantity = jnp.where(
        sell_hint,
        jnp.minimum(available_quantity, replay_sell_quantity),
        available_quantity,
    ).astype(jnp.int16)
    sell_present = sell_quantity > 0

    # Consolidate all 21 market templates into one matrix write: land,
    # fertilizer, wheat, animals, seeds, hire, and product sales.
    market_task_type = jnp.concatenate(
        (
            jnp.full((batch_size, 1), TaskTypeV1.BUY_LAND, dtype=jnp.int8),
            jnp.full((batch_size, 2), TaskTypeV1.BUY_PRODUCT, dtype=jnp.int8),
            jnp.full(
                (batch_size, NUM_ANIMALS),
                TaskTypeV1.ANIMAL_PURCHASE,
                dtype=jnp.int8,
            ),
            jnp.full(
                (batch_size, NUM_CROPS),
                TaskTypeV1.CROP_PRODUCTION,
                dtype=jnp.int8,
            ),
            jnp.full((batch_size, 1), TaskTypeV1.HIRE_WORKER, dtype=jnp.int8),
            jnp.where(
                terminal[:, None],
                jnp.int8(TaskTypeV1.TERMINAL_LIQUIDATION),
                jnp.int8(TaskTypeV1.SELL_INVENTORY),
            ).repeat(NUM_PRODUCTS, axis=1),
        ),
        axis=1,
    )
    market_item_id = jnp.broadcast_to(
        jnp.concatenate(
            (
                jnp.asarray((-1, FERTILIZER_ITEM, WHEAT_ITEM), dtype=jnp.int8),
                jnp.arange(
                    NUM_PRODUCTS, NUM_PRODUCTS + NUM_ANIMALS, dtype=jnp.int8
                ),
                jnp.arange(NUM_CROPS, dtype=jnp.int8),
                jnp.asarray((-1,), dtype=jnp.int8),
                jnp.arange(NUM_PRODUCTS, dtype=jnp.int8),
            )
        )[None, :],
        (batch_size, MARKET_SLOT_COUNT),
    )
    market_quantity = jnp.concatenate(
        (
            jnp.where(land_hint, replay_quantity[:, 0], 1)[:, None],
            fertilizer_quantity[:, None],
            wheat_buy[:, None],
            animal_quantity,
            seed_quantity,
            hire_quantity[:, None],
            sell_quantity,
        ),
        axis=1,
    ).astype(jnp.int16)
    market_source = jnp.broadcast_to(
        jnp.concatenate(
            (
                jnp.asarray(
                    (
                        CandidateSourceV1.LAND,
                        CandidateSourceV1.FERTILIZER_BUY,
                        CandidateSourceV1.ANIMAL,
                    ),
                    dtype=jnp.int8,
                ),
                jnp.full(
                    (NUM_ANIMALS,), CandidateSourceV1.ANIMAL, dtype=jnp.int8
                ),
                jnp.full((NUM_CROPS,), CandidateSourceV1.CROP, dtype=jnp.int8),
                jnp.asarray((CandidateSourceV1.INVENTORY,), dtype=jnp.int8),
                jnp.full(
                    (NUM_PRODUCTS,), CandidateSourceV1.INVENTORY, dtype=jnp.int8
                ),
            )
        )[None, :],
        (batch_size, MARKET_SLOT_COUNT),
    )
    market_source = market_source.at[:, 12:].set(
        jnp.where(
            terminal[:, None],
            jnp.int8(CandidateSourceV1.TERMINAL),
            jnp.int8(CandidateSourceV1.INVENTORY),
        )
    )
    market_present = jnp.concatenate(
        (
            ((unlocked < 4) & (money >= land_cost) & (~terminal))[:, None],
            (
                (
                    fertilizer_hint
                    | (fertilizer_needed & (fertilizer_available <= 0))
                )
                & (shed_used < SHED_CAPACITY)
                & (~terminal)
            )[:, None],
            ((wheat_buy > 0) & (~terminal))[:, None],
            animal_purchase_present,
            seed_present,
            hire_present[:, None],
            sell_present,
        ),
        axis=1,
    )
    market_mandatory = jnp.concatenate(
        (
            jnp.zeros((batch_size, 2), dtype=jnp.bool_),
            (wheat_buy > 0)[:, None],
            jnp.zeros(
                (batch_size, NUM_ANIMALS + NUM_CROPS + 1), dtype=jnp.bool_
            ),
            terminal[:, None] & sell_present,
        ),
        axis=1,
    )
    market_target_id = jnp.full(
        (batch_size, MARKET_SLOT_COUNT), -1, dtype=jnp.int16
    ).at[:, 3 : 3 + NUM_ANIMALS].set(animal_target_id)
    market_target_x = jnp.full(
        (batch_size, MARKET_SLOT_COUNT), -1, dtype=jnp.int8
    ).at[:, 3 : 3 + NUM_ANIMALS].set(animal_target_x)
    market_target_y = jnp.full(
        (batch_size, MARKET_SLOT_COUNT), -1, dtype=jnp.int8
    ).at[:, 3 : 3 + NUM_ANIMALS].set(animal_target_y)
    candidates = candidates._replace(
        task_type=candidates.task_type.at[:, :MARKET_SLOT_COUNT].set(market_task_type),
        owner_unit=candidates.owner_unit.at[:, :MARKET_SLOT_COUNT].set(jnp.int8(-1)),
        target_id=candidates.target_id.at[:, :MARKET_SLOT_COUNT].set(market_target_id),
        target_x=candidates.target_x.at[:, :MARKET_SLOT_COUNT].set(market_target_x),
        target_y=candidates.target_y.at[:, :MARKET_SLOT_COUNT].set(market_target_y),
        item_id=candidates.item_id.at[:, :MARKET_SLOT_COUNT].set(market_item_id),
        quantity=candidates.quantity.at[:, :MARKET_SLOT_COUNT].set(market_quantity),
        source=candidates.source.at[:, :MARKET_SLOT_COUNT].set(market_source),
        mandatory=candidates.mandatory.at[:, :MARKET_SLOT_COUNT].set(
            market_mandatory
        ),
        present=candidates.present.at[:, :MARKET_SLOT_COUNT].set(market_present),
        hard_mask=candidates.hard_mask.at[:, :MARKET_SLOT_COUNT].set(
            market_present
        ),
    )

    # Operational candidates: primary and secondary nearest routes per unit.
    active = states.unit_active[:, player]
    free = active & (controller.unit_tasks.status != TaskStatusV1.ACTIVE)
    positions = states.unit_pos[:, player].astype(jnp.int16)
    tile_position = jnp.stack((_TILE_X, _TILE_Y), axis=-1)
    position_id = jnp.clip(
        positions[..., 1].astype(jnp.int32) * BOARD_SIZE
        + positions[..., 0].astype(jnp.int32),
        0,
        BOARD_SIZE * BOARD_SIZE - 1,
    )
    distance = _TILE_DISTANCE[position_id].astype(jnp.int32)
    safe_animal = jnp.clip(animals.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    animal_product = _ANIMAL_PRODUCT[safe_animal]
    animal_max = _ANIMAL_MAX[safe_animal]
    product_ready = animal_present & (yields > 0)
    product_urgent = product_ready & (yields >= animal_max)
    fertilizer_ready = animal_present & (
        (flags & jnp.uint8(FLAG_FERTILIZER_AVAILABLE)) != 0
    )
    uncared = animal_present & ((flags & jnp.uint8(FLAG_CARED)) == 0)
    unit_wheat = unit_inventory[:, :, WHEAT_ITEM] > 0
    shed_wheat = shed[:, WHEAT_ITEM] > 0
    can_feed = unit_wheat | shed_wheat[:, None]
    unit_fertilizer = unit_inventory[:, :, FERTILIZER_ITEM] > 0
    shed_fertilizer = shed[:, FERTILIZER_ITEM] > 0
    can_fertilize = unit_fertilizer | shed_fertilizer[:, None]

    available_animals = (
        shed[:, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS][:, None, :]
        + unit_inventory[:, :, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS]
    ) > 0
    coop_empty = (kinds == TileKind.COOP) & (animals < 0)
    pasture_empty = (kinds == TileKind.PASTURE) & (animals < 0)
    place_animal = jnp.where(
        coop_empty[:, None, :] & available_animals[:, :, 0, None],
        0,
        jnp.where(
            pasture_empty[:, None, :] & available_animals[:, :, 1, None],
            1,
            jnp.where(
                pasture_empty[:, None, :] & available_animals[:, :, 2, None], 2, -1
            ),
        ),
    ).astype(jnp.int8)

    crop = states.tile_crop[:, player].reshape(batch_size, -1)
    origin = states.tile_origin_day[:, player].reshape(batch_size, -1)
    safe_crop = jnp.clip(crop.astype(jnp.int32), 0, NUM_CROPS - 1)
    mature = (day[:, None] - origin.astype(jnp.int16)) >= _CROP_FIRST[safe_crop]
    crop_harvest = (kinds == TileKind.PLANT) & mature & (yields > 0)
    crop_water = (kinds == TileKind.PLANT) & ((flags & jnp.uint8(FLAG_WATERED)) == 0)
    weed = kinds == TileKind.WEED
    fertilizer_useful = (kinds == TileKind.PLANT) & (
        states.tile_fertilized_until[:, player].reshape(batch_size, -1).astype(jnp.int16)
        < day[:, None] + 2
    )

    # Seven task classes depend only on the tile, so rank them once as Bx100.
    # Only feed/place/fertilize depend on the worker inventory; expand those
    # masks to BxU x100, retain one rank tensor, and recover task/item metadata
    # after Top-2 has reduced the tile axis to two entries.
    base_priority = jnp.full(
        (batch_size, BOARD_SIZE * BOARD_SIZE), 99, dtype=jnp.int32
    )
    base_priority = jnp.where(weed, 8, base_priority)
    base_priority = jnp.where(uncared & (base_priority > 7), 7, base_priority)
    base_priority = jnp.where(crop_water & (base_priority > 6), 6, base_priority)
    base_priority = jnp.where(
        fertilizer_ready & (base_priority > 5), 5, base_priority
    )
    base_priority = jnp.where(
        product_ready & (base_priority > 4), 4, base_priority
    )
    base_priority = jnp.where(
        product_urgent & (base_priority > 2), 2, base_priority
    )
    base_priority = jnp.where(
        crop_harvest & (base_priority > 1), 1, base_priority
    )
    priority = jnp.broadcast_to(
        base_priority[:, None, :],
        (batch_size, MAX_UNITS, BOARD_SIZE * BOARD_SIZE),
    )
    fertilizer_mask = fertilizer_useful[:, None, :] & can_fertilize[..., None]
    priority = jnp.where(
        fertilizer_mask & (priority > 9), jnp.int32(9), priority
    )
    priority = jnp.where(
        (place_animal >= 0) & (priority > 3), jnp.int32(3), priority
    )
    feed_mask = unfed[:, None, :] & can_feed[..., None]
    priority = jnp.where(feed_mask & (priority > 0), jnp.int32(0), priority)

    # Route exploration needs a production-chain ordering rather than the
    # legacy Full-core ordering.  Keep the task code unchanged (the executor
    # still receives the same exact task), but rank survival, fertilizer and
    # product realization ahead of routine harvesting.  The default path is
    # bit-for-bit the previous ordering.
    task_code = priority
    if route_profile:
        route_rank = jnp.where(
            task_code == 3,
            0,
            jnp.where(
                task_code == 2,
                2,
                jnp.where(
                    task_code == 5,
                    3,
                    jnp.where(
                        task_code == 4,
                        4,
                        jnp.where(
                            task_code == 0,
                            5,
                            jnp.where(
                                task_code == 7,
                                6,
                                jnp.where(
                                    task_code == 6,
                                    7,
                                    jnp.where(
                                        task_code == 1,
                                        8,
                                        jnp.where(
                                            task_code == 8,
                                            9,
                                            jnp.where(task_code == 9, 10, 99),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        urgent_feed = (task_code == 0) & (neglect[:, None, :] >= 1)
        urgent_water = (task_code == 6) & (neglect[:, None, :] >= 1)
        priority = jnp.where(urgent_feed | urgent_water, 1, route_rank)

    key = priority * 1_000_000 + distance * 100 + _TILE_ID
    key = jnp.where(priority < 99, key, _INVALID_KEY)
    negative_top, top = jax.lax.top_k(-key, 2)
    top_key = -negative_top
    top_priority = jnp.take_along_axis(priority, top, axis=-1)
    top_code = jnp.take_along_axis(task_code, top, axis=-1)
    tile_batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None, None]
    top_crop = crop[tile_batch, top]
    top_animal = animals[tile_batch, top]
    top_animal_product = animal_product[tile_batch, top]
    top_place_animal = jnp.take_along_axis(place_animal, top, axis=-1)
    top_task = jnp.where(
        top_code == 0,
        TaskTypeV1.ANIMAL_FEED,
        jnp.where(
            top_code == 1,
            TaskTypeV1.CROP_PRODUCTION,
            jnp.where(
                (top_code == 2) | (top_code == 4),
                TaskTypeV1.ANIMAL_COLLECT_PRODUCT,
                jnp.where(
                    top_code == 3,
                    TaskTypeV1.ANIMAL_PLACE,
                    jnp.where(
                        top_code == 5,
                        TaskTypeV1.ANIMAL_COLLECT_FERTILIZER,
                        jnp.where(
                            top_code == 6,
                            TaskTypeV1.WATER_CROP,
                            jnp.where(
                                top_code == 7,
                                TaskTypeV1.ANIMAL_CARE,
                                jnp.where(
                                    top_code == 8,
                                    TaskTypeV1.CLEAR_OR_REMOVE_TILE,
                                    jnp.where(
                                        top_code == 9,
                                        TaskTypeV1.APPLY_FERTILIZER,
                                        TaskTypeV1.NONE,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int8)
    top_item = jnp.where(
        top_code == 0,
        WHEAT_ITEM,
        jnp.where(
            (top_code == 1) | (top_code == 6),
            top_crop,
            jnp.where(
                (top_code == 2) | (top_code == 4),
                top_animal_product,
                jnp.where(
                    top_code == 3,
                    NUM_PRODUCTS + top_place_animal,
                    jnp.where(
                        (top_code == 5) | (top_code == 9),
                        FERTILIZER_ITEM,
                            jnp.where(top_code == 7, top_animal, -1),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int8)
    top_x = _TILE_X[top].astype(jnp.int8)
    top_y = _TILE_Y[top].astype(jnp.int8)
    recovery = free & (unit_inventory_total > 0)
    depot_distance = jnp.sum(
        jnp.abs(positions[..., None, :] - _SHED_ACCESS[None, None, :, :]), axis=-1
    )
    depot_index = jnp.argmin(depot_distance, axis=-1)
    depot = _SHED_ACCESS[depot_index]

    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    units = jnp.arange(MAX_UNITS, dtype=jnp.int32)[None, :]
    primary_slots = PRIMARY_UNIT_START + units
    primary_present = (
        free
        & ((top_key[..., 0] < _INVALID_KEY) | recovery)
        & ((~terminal[:, None]) | recovery)
    )
    primary_task = jnp.where(recovery, TaskTypeV1.SAFE_RECOVERY, top_task[..., 0])
    primary_item = jnp.where(recovery, -1, top_item[..., 0])
    primary_x = jnp.where(recovery, depot[..., 0], top_x[..., 0])
    primary_y = jnp.where(recovery, depot[..., 1], top_y[..., 0])
    primary_id = primary_y.astype(jnp.int16) * BOARD_SIZE + primary_x.astype(jnp.int16)
    primary_animal = (
        (primary_task == TaskTypeV1.ANIMAL_PLACE)
        | (primary_task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
        | (primary_task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
    )
    primary_maintenance = (
        (primary_task == TaskTypeV1.ANIMAL_FEED)
        | (primary_task == TaskTypeV1.ANIMAL_CARE)
    )
    primary_source = jnp.where(
        recovery,
        CandidateSourceV1.INVENTORY,
        jnp.where(
            primary_task == TaskTypeV1.APPLY_FERTILIZER,
            CandidateSourceV1.FERTILIZER_APPLY,
            jnp.where(
                primary_animal,
                CandidateSourceV1.ANIMAL,
                jnp.where(
                    primary_maintenance,
                    CandidateSourceV1.MAINTENANCE,
                    CandidateSourceV1.CROP,
                ),
            ),
        ),
    ).astype(jnp.int8)
    candidates = candidates._replace(
        task_type=candidates.task_type.at[batch, primary_slots].set(primary_task.astype(jnp.int8)),
        owner_unit=candidates.owner_unit.at[batch, primary_slots].set(
            jnp.broadcast_to(jnp.arange(MAX_UNITS, dtype=jnp.int8), (batch_size, MAX_UNITS))
        ),
        target_id=candidates.target_id.at[batch, primary_slots].set(primary_id),
        target_x=candidates.target_x.at[batch, primary_slots].set(primary_x.astype(jnp.int8)),
        target_y=candidates.target_y.at[batch, primary_slots].set(primary_y.astype(jnp.int8)),
        item_id=candidates.item_id.at[batch, primary_slots].set(primary_item.astype(jnp.int8)),
        quantity=candidates.quantity.at[batch, primary_slots].set(
            jnp.where(recovery, unit_inventory_total, 1).astype(jnp.int16)
        ),
        source=candidates.source.at[batch, primary_slots].set(primary_source),
        mandatory=candidates.mandatory.at[batch, primary_slots].set(
            recovery
            | (primary_task == TaskTypeV1.ANIMAL_FEED)
            | (primary_task == TaskTypeV1.ANIMAL_PLACE)
        ),
        present=candidates.present.at[batch, primary_slots].set(primary_present),
        hard_mask=candidates.hard_mask.at[batch, primary_slots].set(primary_present),
    )

    if SECONDARY_UNIT_COUNT > 0:
        secondary_units = jnp.arange(SECONDARY_UNIT_COUNT, dtype=jnp.int32)[None, :]
        secondary_slots = SECONDARY_UNIT_START + secondary_units
        secondary_present = (
            free[:, :SECONDARY_UNIT_COUNT]
            & (~recovery[:, :SECONDARY_UNIT_COUNT])
            & (top_key[:, :SECONDARY_UNIT_COUNT, 1] < _INVALID_KEY)
            & (~terminal[:, None])
        )
        secondary_task = top_task[:, :SECONDARY_UNIT_COUNT, 1]
        secondary_source = jnp.where(
            secondary_task == TaskTypeV1.APPLY_FERTILIZER,
            CandidateSourceV1.FERTILIZER_APPLY,
            jnp.where(
                (secondary_task == TaskTypeV1.ANIMAL_FEED)
                | (secondary_task == TaskTypeV1.ANIMAL_CARE),
                CandidateSourceV1.MAINTENANCE,
                jnp.where(
                    (secondary_task == TaskTypeV1.ANIMAL_PLACE)
                    | (secondary_task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
                    | (secondary_task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER),
                    CandidateSourceV1.ANIMAL,
                    CandidateSourceV1.CROP,
                ),
            ),
        ).astype(jnp.int8)
        candidates = candidates._replace(
            task_type=candidates.task_type.at[batch, secondary_slots].set(
                top_task[:, :SECONDARY_UNIT_COUNT, 1]
            ),
            owner_unit=candidates.owner_unit.at[batch, secondary_slots].set(
                jnp.broadcast_to(
                    jnp.arange(SECONDARY_UNIT_COUNT, dtype=jnp.int8),
                    (batch_size, SECONDARY_UNIT_COUNT),
                )
            ),
            target_id=candidates.target_id.at[batch, secondary_slots].set(
                top[:, :SECONDARY_UNIT_COUNT, 1].astype(jnp.int16)
            ),
            target_x=candidates.target_x.at[batch, secondary_slots].set(
                top_x[:, :SECONDARY_UNIT_COUNT, 1]
            ),
            target_y=candidates.target_y.at[batch, secondary_slots].set(
                top_y[:, :SECONDARY_UNIT_COUNT, 1]
            ),
            item_id=candidates.item_id.at[batch, secondary_slots].set(
                top_item[:, :SECONDARY_UNIT_COUNT, 1]
            ),
            quantity=candidates.quantity.at[batch, secondary_slots].set(jnp.int16(1)),
            source=candidates.source.at[batch, secondary_slots].set(secondary_source),
            mandatory=candidates.mandatory.at[batch, secondary_slots].set(
                (top_task[:, :SECONDARY_UNIT_COUNT, 1] == TaskTypeV1.ANIMAL_FEED)
                | (top_task[:, :SECONDARY_UNIT_COUNT, 1] == TaskTypeV1.ANIMAL_PLACE)
            ),
            present=candidates.present.at[batch, secondary_slots].set(secondary_present),
            hard_mask=candidates.hard_mask.at[batch, secondary_slots].set(secondary_present),
        )

    # Five crop alternatives remain visible even with one worker.
    empty = kinds == TileKind.EMPTY
    unit_ids = jnp.arange(MAX_UNITS, dtype=jnp.int32)[None, :, None]
    crop_ids = jnp.arange(NUM_CROPS, dtype=jnp.int32)[None, None, :]
    unit_choice_key = jnp.where(
        free[:, :, None], (unit_ids - crop_ids) % MAX_UNITS, MAX_UNITS + 1
    )
    plant_owner = jnp.argmin(unit_choice_key, axis=1)
    owner_valid = jnp.min(unit_choice_key, axis=1) <= MAX_UNITS
    selected_position = jnp.take_along_axis(
        positions[:, :, None, :], plant_owner[:, None, :, None], axis=1
    )[:, 0]
    plant_distance = jnp.sum(
        jnp.abs(selected_position[:, :, None, :] - tile_position[None, None, :, :]), axis=-1
    )
    plant_key = jnp.where(empty[:, None, :], plant_distance * 100 + _TILE_ID, _INVALID_KEY)
    plant_target = jnp.argmin(plant_key, axis=-1)
    plant_target_valid = jnp.min(plant_key, axis=-1) < _INVALID_KEY
    plant_present = (
        owner_valid
        & plant_target_valid
        & (states.seeds[:, player] > 0)
        & (~terminal[:, None])
    )
    plant_slots = PLANT_START + jnp.arange(NUM_CROPS, dtype=jnp.int32)[None, :]
    candidates = candidates._replace(
        task_type=candidates.task_type.at[batch, plant_slots].set(
            jnp.full((batch_size, NUM_CROPS), TaskTypeV1.CROP_PRODUCTION, dtype=jnp.int8)
        ),
        owner_unit=candidates.owner_unit.at[batch, plant_slots].set(plant_owner.astype(jnp.int8)),
        target_id=candidates.target_id.at[batch, plant_slots].set(plant_target.astype(jnp.int16)),
        target_x=candidates.target_x.at[batch, plant_slots].set(_TILE_X[plant_target].astype(jnp.int8)),
        target_y=candidates.target_y.at[batch, plant_slots].set(_TILE_Y[plant_target].astype(jnp.int8)),
        item_id=candidates.item_id.at[batch, plant_slots].set(
            jnp.broadcast_to(jnp.arange(NUM_CROPS, dtype=jnp.int8), (batch_size, NUM_CROPS))
        ),
        quantity=candidates.quantity.at[batch, plant_slots].set(jnp.int16(1)),
        source=candidates.source.at[batch, plant_slots].set(jnp.int8(CandidateSourceV1.CROP)),
        present=candidates.present.at[batch, plant_slots].set(plant_present),
        hard_mask=candidates.hard_mask.at[batch, plant_slots].set(plant_present),
    )

    # Explicit coop and pasture routes expose both official build operations.
    first_free_key = jnp.where(free, jnp.arange(MAX_UNITS)[None, :], MAX_UNITS + 1)
    build_owner = jnp.argmin(first_free_key, axis=-1)
    build_owner_valid = jnp.min(first_free_key, axis=-1) <= MAX_UNITS
    build_position = positions[jnp.arange(batch_size), build_owner]
    build_distance = jnp.sum(
        jnp.abs(build_position[:, None, :] - tile_position[None, :, :]), axis=-1
    )
    build_key = jnp.where(empty, build_distance * 100 + _TILE_ID, _INVALID_KEY)
    build_target = jnp.argmin(build_key, axis=-1)
    build_valid = build_owner_valid & (jnp.min(build_key, axis=-1) < _INVALID_KEY) & (~terminal)
    for slot, animal_id in ((BUILD_COOP_SLOT, 0), (BUILD_PASTURE_SLOT, 1)):
        candidates = candidates._replace(
            task_type=candidates.task_type.at[:, slot].set(jnp.int8(TaskTypeV1.BUILD_ANIMAL_STRUCTURE)),
            owner_unit=candidates.owner_unit.at[:, slot].set(build_owner.astype(jnp.int8)),
            target_id=candidates.target_id.at[:, slot].set(build_target.astype(jnp.int16)),
            target_x=candidates.target_x.at[:, slot].set(_TILE_X[build_target].astype(jnp.int8)),
            target_y=candidates.target_y.at[:, slot].set(_TILE_Y[build_target].astype(jnp.int8)),
            item_id=candidates.item_id.at[:, slot].set(jnp.int8(animal_id)),
            quantity=candidates.quantity.at[:, slot].set(jnp.int16(1)),
            source=candidates.source.at[:, slot].set(jnp.int8(CandidateSourceV1.ANIMAL)),
            present=candidates.present.at[:, slot].set(build_valid),
            hard_mask=candidates.hard_mask.at[:, slot].set(build_valid),
            replay_priority=candidates.replay_priority.at[:, slot].set(
                jnp.where(
                    replay_enabled & (replay_build_animal == animal_id),
                    replay_build_priority,
                    0.0,
                )
            ),
        )

    return candidates._replace(
        replay_priority=candidates.replay_priority.at[:, :MARKET_SLOT_COUNT].set(
            jnp.where(replay_enabled[:, None], replay_priority, 0.0)
        )
    )


def evaluate_crop_inventory_feasibility_v1(
    states: State,
    candidates: CandidateV1,
    player: int,
) -> FeasibilityV1:
    """Exact E4 feasibility for crop, worker, inventory, and terminal tasks."""

    batch_size = states.step.shape[0]
    result = _empty_feasibility(batch_size)
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    owner = jnp.clip(candidates.owner_unit.astype(jnp.int32), 0, MAX_UNITS - 1)
    x = jnp.clip(candidates.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(candidates.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    target_valid = (
        (candidates.target_x >= 0)
        & (candidates.target_x < BOARD_SIZE)
        & (candidates.target_y >= 0)
        & (candidates.target_y < BOARD_SIZE)
    )
    position = states.unit_pos[batch, player, owner].astype(jnp.int16)
    target = jnp.stack((x, y), axis=-1).astype(jnp.int16)
    direct = jnp.sum(jnp.abs(position - target), axis=-1).astype(jnp.int16)
    target_to_shed = jnp.min(
        jnp.sum(jnp.abs(target[..., None, :] - _SHED_ACCESS), axis=-1), axis=-1
    ).astype(jnp.int16)
    tile_kind = states.tile_kind[batch, player, y, x]
    tile_crop = states.tile_crop[batch, player, y, x]
    tile_yield = states.tile_yield[batch, player, y, x].astype(jnp.int16)
    tile_flags = states.tile_flags[batch, player, y, x]
    tile_origin = states.tile_origin_day[batch, player, y, x].astype(jnp.int16)
    tile_neglect = states.tile_neglect[batch, player, y, x].astype(jnp.int16)
    tile_max_lifespan = states.tile_max_lifespan[batch, player, y, x].astype(jnp.int16)
    task = candidates.task_type
    unit_task = candidates.owner_unit >= 0
    is_crop = unit_task & (task == TaskTypeV1.CROP_PRODUCTION)
    is_plant = is_crop & (tile_kind == TileKind.EMPTY)
    is_harvest = is_crop & (tile_kind == TileKind.PLANT)
    is_water = unit_task & (task == TaskTypeV1.WATER_CROP)
    is_clear = unit_task & (task == TaskTypeV1.CLEAR_OR_REMOVE_TILE)
    is_recovery = unit_task & (
        (task == TaskTypeV1.SAFE_RECOVERY) | (task == TaskTypeV1.SHED_DEPOSIT)
    )
    is_seed = (~unit_task) & (task == TaskTypeV1.CROP_PRODUCTION)
    is_hire = (~unit_task) & (task == TaskTypeV1.HIRE_WORKER)
    is_sell = (~unit_task) & (
        (task == TaskTypeV1.SELL_INVENTORY) | (task == TaskTypeV1.TERMINAL_LIQUIDATION)
    )
    supported_unit = is_plant | is_harvest | is_water | is_clear | is_recovery
    supported_market = is_seed | is_hire | is_sell
    supported = supported_unit | supported_market

    inventory_total = jnp.sum(
        states.unit_inventory[batch, player, owner].astype(jnp.int32), axis=-1
    ).astype(jnp.int16)
    path = jnp.where(
        is_harvest, direct + target_to_shed, jnp.where(supported_unit, direct, 0)
    ).astype(jnp.int16)
    operation = jnp.where(
        is_plant | is_harvest, 2, jnp.where(supported_unit | supported_market, 1, 0)
    ).astype(jnp.int16)
    expected = (states.step[:, None].astype(jnp.int16) + path + operation).astype(jnp.int16)
    current_day = (states.step[:, None] // TURNS_PER_DAY).astype(jnp.int16)
    day_end = ((current_day + 1) * TURNS_PER_DAY).astype(jnp.int16)
    # step 718 is the final ACTIVE observation; its action is committed into
    # DONE state step 719, so finish-time feasibility must allow 719.
    terminal_step = jnp.int16(EPISODE_STEPS - 1)
    deadline = jnp.where(supported_unit, day_end, terminal_step).astype(jnp.int16)
    safe_crop = jnp.clip(candidates.item_id.astype(jnp.int32), 0, NUM_CROPS - 1)
    existing_crop = jnp.clip(tile_crop.astype(jnp.int32), 0, NUM_CROPS - 1)
    mature = current_day - tile_origin >= _CROP_FIRST[existing_crop]
    seed_available = states.seeds[batch, player, safe_crop]
    watered = (tile_flags & jnp.uint8(FLAG_WATERED)) != 0
    current_step = states.step[:, None].astype(jnp.int16)
    arrival_step = current_step + direct
    first_decay_step = jnp.where(
        current_step <= tile_max_lifespan,
        tile_max_lifespan,
        current_step + ((current_step - tile_max_lifespan) & jnp.int16(1)),
    )
    decay_count_before_arrival = jnp.where(
        (tile_max_lifespan >= 0) & (first_decay_step < arrival_step),
        ((arrival_step - 1 - first_decay_step) // 2) + 1,
        0,
    ).astype(jnp.int16)
    harvest_yield_at_arrival = jnp.maximum(
        tile_yield - decay_count_before_arrival, 0
    ).astype(jnp.int16)
    dies_at_upcoming_day_end = (
        (~watered)
        & (tile_neglect >= 1)
        & (day_end <= arrival_step)
    )
    unit_active = states.unit_active[batch, player, owner]
    unit_ok = jnp.where(
        is_plant,
        (tile_kind == TileKind.EMPTY) & (seed_available > 0),
        jnp.where(
            is_harvest,
            (tile_kind == TileKind.PLANT)
            & mature
            & (harvest_yield_at_arrival > 0)
            & (~dies_at_upcoming_day_end),
            jnp.where(
                is_water,
                (tile_kind == TileKind.PLANT) & (~watered),
                jnp.where(
                    is_clear,
                    (tile_kind != TileKind.EMPTY)
                    & (tile_kind != TileKind.LOCKED)
                    & (states.tile_animal[batch, player, y, x] < 0),
                    is_recovery & (inventory_total > 0),
                ),
            ),
        ),
    )
    earliest_harvest = (
        (current_day + _CROP_FIRST[safe_crop]) * TURNS_PER_DAY + 2
    ).astype(jnp.int16)
    bankable = jnp.where(is_plant, earliest_harvest <= terminal_step, expected <= terminal_step)
    finishes_day = expected <= day_end
    incoming = jnp.where(
        is_harvest,
        harvest_yield_at_arrival,
        jnp.where(is_recovery, inventory_total, 0),
    ).astype(jnp.int16)
    shed_used = jnp.sum(states.shed[:, player].astype(jnp.int32), axis=-1)[:, None]
    capacity_ok = (~(is_harvest | is_recovery)) | (shed_used + incoming <= SHED_CAPACITY)

    hire_n = states.hires_today[:, player].astype(jnp.int32)[:, None]
    hire_quantity = jnp.maximum(candidates.quantity.astype(jnp.int32), 1)
    hire_end = jnp.clip(hire_n + hire_quantity, 0, MAX_HANDS)
    hire_cost = _HIRE_COST_PREFIX[hire_end] - _HIRE_COST_PREFIX[hire_n]
    seed_cost = _SEED_COST[safe_crop] * jnp.maximum(candidates.quantity.astype(jnp.int32), 1)
    cash = jnp.where(is_seed, seed_cost, jnp.where(is_hire, hire_cost, 0)).astype(jnp.int32)
    item = jnp.clip(candidates.item_id.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    shed_item_count = states.shed[batch, player, item].astype(jnp.int16)
    market_ok = jnp.where(
        is_seed,
        states.money[:, player, None] >= seed_cost,
        jnp.where(
            is_hire,
            (hire_n + hire_quantity <= MAX_HANDS)
            & (states.money[:, player, None] >= hire_cost),
            is_sell & (candidates.quantity > 0),
        ),
    )
    legal = (
        candidates.present
        & candidates.hard_mask
        & supported
        & bankable
        & jnp.where(
            supported_unit,
            target_valid & unit_active & unit_ok & finishes_day & capacity_ok,
            market_ok,
        )
    )
    return result._replace(
        legal_now=legal,
        unit_required=supported_unit,
        plot_required=supported_unit & (~is_recovery),
        path_steps=path,
        operation_steps=operation,
        expected_finish_step=expected,
        deadline_step=deadline,
        bankable_before_terminal=bankable,
        cash_required=cash,
        seed_item=jnp.where(is_plant, candidates.item_id, -1).astype(jnp.int8),
        seed_required=is_plant.astype(jnp.int16),
        shed_item=jnp.where(is_sell, candidates.item_id, -1).astype(jnp.int8),
        shed_item_required=jnp.where(is_sell, shed_item_count, 0).astype(jnp.int16),
        shed_reserved_in=incoming,
        market_slots_required=jnp.where(
            is_hire,
            hire_quantity,
            supported_market.astype(jnp.int32),
        ).astype(jnp.int8),
    )


def evaluate_full_core_feasibility_v1(
    states: State,
    candidates: CandidateV1,
    tables: StaticTables,
    player: int,
) -> FeasibilityV1:
    """Merge exact module feasibility without allowing one module to shadow another."""

    e2 = evaluate_e2_feasibility_v1(states, candidates, tables, player)
    e3 = evaluate_e3_feasibility_v1(states, candidates, tables, player)
    core = evaluate_crop_inventory_feasibility_v1(states, candidates, player)
    e2_mask = (
        (candidates.task_type == TaskTypeV1.BUY_LAND)
        | (
            (candidates.task_type == TaskTypeV1.BUY_PRODUCT)
            & (candidates.item_id == FERTILIZER_ITEM)
        )
        | (candidates.task_type == TaskTypeV1.APPLY_FERTILIZER)
    )
    e3_mask = (
        (candidates.task_type == TaskTypeV1.ANIMAL_PURCHASE)
        | (
            (candidates.task_type == TaskTypeV1.BUY_PRODUCT)
            & (candidates.item_id == WHEAT_ITEM)
        )
        | (candidates.task_type == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
        | (candidates.task_type == TaskTypeV1.ANIMAL_PLACE)
        | (candidates.task_type == TaskTypeV1.ANIMAL_FEED)
        | (candidates.task_type == TaskTypeV1.ANIMAL_CARE)
        | (candidates.task_type == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
        | (candidates.task_type == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
    )
    return jax.tree.map(
        lambda left, middle, right: jnp.where(e2_mask, left, jnp.where(e3_mask, middle, right)),
        e2,
        e3,
        core,
    )


def full_core_priority_v1(
    candidates: CandidateV1, feasibility: FeasibilityV1
) -> jax.Array:
    task = candidates.task_type
    base = jnp.where(
        candidates.mandatory,
        12_000.0,
        jnp.where(
            task == TaskTypeV1.TERMINAL_LIQUIDATION,
            11_000.0,
            jnp.where(
                task == TaskTypeV1.SAFE_RECOVERY,
                10_000.0,
                jnp.where(
                    task == TaskTypeV1.ANIMAL_FEED,
                    9_500.0,
                    jnp.where(
                        (task == TaskTypeV1.CROP_PRODUCTION) & (candidates.owner_unit >= 0),
                        8_500.0,
                        jnp.where(
                            task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT,
                            8_000.0,
                            jnp.where(
                                task == TaskTypeV1.ANIMAL_PLACE,
                                7_500.0,
                                jnp.where(
                                    task == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER,
                                    7_000.0,
                                    jnp.where(
                                        task == TaskTypeV1.WATER_CROP,
                                        6_500.0,
                                        jnp.where(
                                            task == TaskTypeV1.ANIMAL_CARE,
                                            6_000.0,
                                            jnp.where(
                                                task == TaskTypeV1.CLEAR_OR_REMOVE_TILE,
                                                5_500.0,
                                                jnp.where(
                                                    task == TaskTypeV1.APPLY_FERTILIZER,
                                                    5_000.0,
                                                    jnp.where(
                                                        task == TaskTypeV1.SELL_INVENTORY,
                                                        4_500.0,
                                                        jnp.where(
                                                            task == TaskTypeV1.BUY_PRODUCT,
                                                            4_000.0,
                                                            jnp.where(
                                                                task == TaskTypeV1.CROP_PRODUCTION,
                                                                3_500.0,
                                                                jnp.where(
                                                                    task == TaskTypeV1.HIRE_WORKER,
                                                                    3_000.0,
                                                                    jnp.where(
                                                                        task == TaskTypeV1.ANIMAL_PURCHASE,
                                                                        2_500.0,
                                                                        jnp.where(
                                                                            task == TaskTypeV1.BUILD_ANIMAL_STRUCTURE,
                                                                            2_000.0,
                                                                            1_500.0,
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
                ),
            ),
        ),
    )
    base = jnp.where(candidates.replay_priority > 0, candidates.replay_priority, base)
    score = base - feasibility.path_steps.astype(jnp.float32) - candidates.source_slot * 1.0e-4
    return jnp.where(feasibility.legal_now, score, -1.0e9)


def select_full_core_candidates_v1(
    states: State,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    controller: ControllerStateV1,
    player: int,
    max_selections: int = MAX_SELECTIONS_V1,
) -> E2SelectionV1:
    return select_candidates_with_scores_v1(
        states,
        candidates,
        feasibility,
        full_core_priority_v1(candidates, feasibility),
        controller,
        initialize_e2_ledger_v1(states, controller, player),
        player,
        max_selections,
    )
