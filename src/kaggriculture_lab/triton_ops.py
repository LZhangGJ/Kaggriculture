"""Fused CUDA kernels for common Kaggriculture unit interactions.

The public wrapper is intentionally tiny and optional. CPU execution and CUDA
installations without Triton continue to use the pure PyTorch transition.
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only installations
    triton = None
    tl = None


TRITON_AVAILABLE = triton is not None
COMMON_UNIT_OPS = tuple(range(6, 16))  # DIG through PLANT
INVENTORY_UNIT_OPS = (5, 16, 17)  # DROP, PICKUP, PLACE
SUPPORTED_UNIT_OPS = INVENTORY_UNIT_OPS + COMMON_UNIT_OPS


if TRITON_AVAILABLE:

    @triton.jit
    def _common_interaction_kernel(
        op_ptr,
        arg_ptr,
        active_ptr,
        plant_allowed_ptr,
        position_ptr,
        tile_ptr,
        planted_ptr,
        placed_ptr,
        watered_ptr,
        fed_ptr,
        cared_ptr,
        consecutive_ptr,
        yield_ptr,
        max_life_ptr,
        fert_until_ptr,
        fert_available_ptr,
        care_bonus_ptr,
        inventory_ptr,
        seeds_ptr,
        unit,
        day,
        pair_count,
        MAX_UNITS: tl.constexpr,
        ITEM_COUNT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid_pair = pair < pair_count
        unit_offset = pair * MAX_UNITS + unit
        active = tl.load(active_ptr + unit_offset, mask=valid_pair, other=0).to(tl.int1)
        op = tl.load(op_ptr + unit_offset, mask=valid_pair, other=0)
        supported = (op >= 6) & (op <= 15)
        execute = valid_pair & active & supported

        position_offset = unit_offset * 2
        x = tl.load(position_ptr + position_offset, mask=execute, other=0).to(tl.int32)
        y = tl.load(position_ptr + position_offset + 1, mask=execute, other=0).to(tl.int32)
        board_offset = pair * 100 + y * 10 + x
        tile = tl.load(tile_ptr + board_offset, mask=execute, other=1).to(tl.int32)
        owned = execute & (tile != 1)
        animal = (tile >= 10) & (tile < 13)
        plant_tile = (tile >= 3) & (tile < 8)
        crop = tile - 3
        crop_safe = tl.maximum(0, tl.minimum(crop, 4))
        crop_first = tl.where(crop_safe == 0, 2, tl.where(crop_safe == 1, 2, tl.where(crop_safe == 2, 8, 10)))
        crop_max_day = tl.where(
            crop_safe == 0,
            4,
            tl.where(crop_safe == 1, 3, tl.where(crop_safe == 2, 8, tl.where(crop_safe == 3, 10, 12))),
        )
        crop_max_yield = tl.where(crop_safe == 0, 6, tl.where((crop_safe == 1) | (crop_safe == 2) | (crop_safe == 3), 4, 6))
        ongoing = (crop_safe == 2) | (crop_safe == 3)

        dig = owned & (op == 6) & (tile != 0) & ~animal
        tl.store(tile_ptr + board_offset, 0, mask=dig)
        tl.store(planted_ptr + board_offset, -1, mask=dig)
        tl.store(placed_ptr + board_offset, -1, mask=dig)
        tl.store(max_life_ptr + board_offset, -1, mask=dig)
        tl.store(fert_until_ptr + board_offset, -1, mask=dig)
        tl.store(watered_ptr + board_offset, 0, mask=dig)
        tl.store(fed_ptr + board_offset, 0, mask=dig)
        tl.store(cared_ptr + board_offset, 0, mask=dig)
        tl.store(fert_available_ptr + board_offset, 0, mask=dig)
        tl.store(consecutive_ptr + board_offset, 0, mask=dig)
        tl.store(yield_ptr + board_offset, 0, mask=dig)
        tl.store(care_bonus_ptr + board_offset, 0, mask=dig)

        board_watered = tl.load(watered_ptr + board_offset, mask=execute, other=0).to(tl.int1)
        water = owned & (op == 7) & plant_tile & ~board_watered
        tl.store(watered_ptr + board_offset, 1, mask=water)
        planted_day = tl.load(planted_ptr + board_offset, mask=execute, other=-1).to(tl.int32)
        current_yield = tl.load(yield_ptr + board_offset, mask=execute, other=0).to(tl.int32)
        fert_until = tl.load(fert_until_ptr + board_offset, mask=execute, other=-1).to(tl.int32)
        age = day - planted_day
        window_start = (crop_max_day + 1) // 2
        bonus_window = ~ongoing & (age >= window_start) & (age <= crop_max_day)
        bonus = tl.where(fert_until >= day, 2, 1)
        watered_yield = tl.minimum(current_yield + bonus, crop_max_yield)
        tl.store(yield_ptr + board_offset, watered_yield, mask=water & bonus_window)

        inventory_base = unit_offset * ITEM_COUNT
        mature = age >= crop_first
        harvest_plant = owned & (op == 8) & plant_tile & (current_yield > 0) & mature
        crop_inventory_offset = inventory_base + crop_safe
        crop_inventory = tl.load(inventory_ptr + crop_inventory_offset, mask=harvest_plant, other=0).to(tl.int32)
        tl.store(inventory_ptr + crop_inventory_offset, crop_inventory + current_yield, mask=harvest_plant)
        tl.store(yield_ptr + board_offset, 0, mask=harvest_plant)
        clear_plant = harvest_plant & ~ongoing
        tl.store(tile_ptr + board_offset, 0, mask=clear_plant)
        tl.store(planted_ptr + board_offset, -1, mask=clear_plant)
        tl.store(placed_ptr + board_offset, -1, mask=clear_plant)
        tl.store(max_life_ptr + board_offset, -1, mask=clear_plant)
        tl.store(fert_until_ptr + board_offset, -1, mask=clear_plant)
        tl.store(watered_ptr + board_offset, 0, mask=clear_plant)
        tl.store(fed_ptr + board_offset, 0, mask=clear_plant)
        tl.store(cared_ptr + board_offset, 0, mask=clear_plant)
        tl.store(fert_available_ptr + board_offset, 0, mask=clear_plant)
        tl.store(consecutive_ptr + board_offset, 0, mask=clear_plant)
        tl.store(yield_ptr + board_offset, 0, mask=clear_plant)
        tl.store(care_bonus_ptr + board_offset, 0, mask=clear_plant)

        animal_index = tile - 10
        product_index = tl.where(animal_index == 0, 5, tl.where(animal_index == 1, 6, 7))
        harvest_animal = owned & (op == 8) & animal & (current_yield > 0)
        animal_inventory_offset = inventory_base + product_index
        animal_inventory = tl.load(inventory_ptr + animal_inventory_offset, mask=harvest_animal, other=0).to(tl.int32)
        tl.store(inventory_ptr + animal_inventory_offset, animal_inventory + current_yield, mask=harvest_animal)
        tl.store(yield_ptr + board_offset, 0, mask=harvest_animal)

        fertilizer_offset = inventory_base + 8
        fertilizer = tl.load(inventory_ptr + fertilizer_offset, mask=execute, other=0).to(tl.int32)
        fertilize = owned & (op == 9) & plant_tile & (fertilizer > 0)
        tl.store(inventory_ptr + fertilizer_offset, fertilizer - 1, mask=fertilize)
        tl.store(fert_until_ptr + board_offset, tl.maximum(fert_until, day + 2), mask=fertilize)

        build_coop = owned & (op == 10) & (tile == 0)
        build_pasture = owned & (op == 11) & (tile == 0)
        tl.store(tile_ptr + board_offset, 8, mask=build_coop)
        tl.store(tile_ptr + board_offset, 9, mask=build_pasture)

        wheat = tl.load(inventory_ptr + inventory_base, mask=execute, other=0).to(tl.int32)
        board_fed = tl.load(fed_ptr + board_offset, mask=execute, other=0).to(tl.int1)
        feed = owned & (op == 12) & animal & ~board_fed & (wheat > 0)
        tl.store(inventory_ptr + inventory_base, wheat - 1, mask=feed)
        tl.store(fed_ptr + board_offset, 1, mask=feed)

        available = tl.load(fert_available_ptr + board_offset, mask=execute, other=0).to(tl.int1)
        collect = owned & (op == 13) & animal & available
        fertilizer = tl.load(inventory_ptr + fertilizer_offset, mask=collect, other=0).to(tl.int32)
        tl.store(fert_available_ptr + board_offset, 0, mask=collect)
        tl.store(inventory_ptr + fertilizer_offset, fertilizer + 1, mask=collect)

        board_cared = tl.load(cared_ptr + board_offset, mask=execute, other=0).to(tl.int1)
        care = owned & (op == 14) & animal & ~board_cared
        tl.store(cared_ptr + board_offset, 1, mask=care)

        crop_arg = tl.load(arg_ptr + unit_offset, mask=execute, other=0).to(tl.int32)
        valid_crop_arg = (crop_arg >= 0) & (crop_arg < 5)
        crop_arg_safe = tl.maximum(0, tl.minimum(crop_arg, 4))
        seed_offset = pair * 5 + crop_arg_safe
        seed_count = tl.load(seeds_ptr + seed_offset, mask=execute, other=0).to(tl.int32)
        allowed = tl.load(plant_allowed_ptr + unit_offset, mask=execute, other=0).to(tl.int1)
        plant = owned & (op == 15) & allowed & (tile == 0) & valid_crop_arg & (seed_count > 0)
        tl.store(seeds_ptr + seed_offset, seed_count - 1, mask=plant)
        tl.store(tile_ptr + board_offset, 3 + crop_arg_safe, mask=plant)
        tl.store(planted_ptr + board_offset, day, mask=plant)
        tl.store(watered_ptr + board_offset, 0, mask=plant)
        tl.store(consecutive_ptr + board_offset, 1, mask=plant)
        planted_ongoing = (crop_arg_safe == 2) | (crop_arg_safe == 3)
        tl.store(yield_ptr + board_offset, tl.where(planted_ongoing, 0, 1), mask=plant)
        planted_max_day = tl.where(
            crop_arg_safe == 0,
            4,
            tl.where(crop_arg_safe == 1, 3, tl.where(crop_arg_safe == 2, 8, tl.where(crop_arg_safe == 3, 10, 12))),
        )
        max_life = tl.where(planted_ongoing, -1, (day + planted_max_day + 1) * 24)
        tl.store(max_life_ptr + board_offset, max_life, mask=plant)
        tl.store(fert_until_ptr + board_offset, -1, mask=plant)


    @triton.jit
    def _inventory_interaction_kernel(
        op_ptr,
        arg_ptr,
        quantity_ptr,
        active_ptr,
        position_ptr,
        tile_ptr,
        placed_ptr,
        fed_ptr,
        cared_ptr,
        consecutive_ptr,
        yield_ptr,
        fert_available_ptr,
        care_bonus_ptr,
        inventory_ptr,
        shed_ptr,
        unit,
        day,
        pair_count,
        MAX_UNITS: tl.constexpr,
        ITEM_COUNT: tl.constexpr,
        SHED_CAPACITY: tl.constexpr,
        OP_KIND: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid_pair = pair < pair_count
        unit_offset = pair * MAX_UNITS + unit
        active = tl.load(active_ptr + unit_offset, mask=valid_pair, other=0).to(tl.int1)
        op = tl.load(op_ptr + unit_offset, mask=valid_pair, other=0)
        execute = valid_pair & active & (op == OP_KIND)
        arg = tl.load(arg_ptr + unit_offset, mask=execute, other=0).to(tl.int32)
        quantity = tl.maximum(tl.load(quantity_ptr + unit_offset, mask=execute, other=1).to(tl.int32), 1)

        position_offset = unit_offset * 2
        x = tl.load(position_ptr + position_offset, mask=execute, other=0).to(tl.int32)
        y = tl.load(position_ptr + position_offset + 1, mask=execute, other=0).to(tl.int32)
        shed_adjacent = ((x == 4) | (x == 5)) & ((y == 4) | (y == 5))
        inventory_base = unit_offset * ITEM_COUNT
        shed_base = pair * ITEM_COUNT
        if OP_KIND == 5:
            shed_total = tl.zeros((BLOCK,), tl.int32)
            for item_index in tl.static_range(0, ITEM_COUNT):
                shed_total += tl.load(shed_ptr + shed_base + item_index, mask=execute, other=0).to(tl.int32)
            room = tl.maximum(SHED_CAPACITY - shed_total, 0)
            drop = execute & shed_adjacent
            for item_index in tl.static_range(0, ITEM_COUNT):
                source_offset = inventory_base + item_index
                destination_offset = shed_base + item_index
                source = tl.load(inventory_ptr + source_offset, mask=execute, other=0).to(tl.int32)
                destination = tl.load(shed_ptr + destination_offset, mask=execute, other=0).to(tl.int32)
                amount = tl.where(drop, tl.minimum(source, room), 0)
                tl.store(inventory_ptr + source_offset, source - amount, mask=drop)
                tl.store(shed_ptr + destination_offset, destination + amount, mask=drop)
                room -= amount
        else:
            valid_item = (arg >= 0) & (arg < ITEM_COUNT)
            item = tl.maximum(0, tl.minimum(arg, ITEM_COUNT - 1))
            inventory_offset = inventory_base + item
            shed_offset = shed_base + item
            shed_item = tl.load(shed_ptr + shed_offset, mask=execute, other=0).to(tl.int32)
            carried = tl.load(inventory_ptr + inventory_offset, mask=execute, other=0).to(tl.int32)

            if OP_KIND == 16:
                pickup = execute & shed_adjacent & valid_item
                take = tl.where(pickup, tl.minimum(quantity, shed_item), 0)
                tl.store(shed_ptr + shed_offset, shed_item - take, mask=pickup)
                tl.store(inventory_ptr + inventory_offset, carried + take, mask=pickup)
            else:
                board_offset = pair * 100 + y * 10 + x
                tile = tl.load(tile_ptr + board_offset, mask=execute, other=1).to(tl.int32)
                shed_total = tl.zeros((BLOCK,), tl.int32)
                for item_index in tl.static_range(0, ITEM_COUNT):
                    shed_total += tl.load(shed_ptr + shed_base + item_index, mask=execute, other=0).to(tl.int32)
                room = tl.maximum(SHED_CAPACITY - shed_total, 0)
                animal_arg = (arg >= 9) & (arg < 12)
                animal_index = tl.maximum(0, tl.minimum(arg - 9, 2))
                required_structure = tl.where(animal_index == 0, 8, 9)
                structure_match = execute & animal_arg & (tile == required_structure)
                place_animal = structure_match & (carried > 0)
                tl.store(inventory_ptr + inventory_offset, carried - 1, mask=place_animal)
                tl.store(tile_ptr + board_offset, 10 + animal_index, mask=place_animal)
                tl.store(placed_ptr + board_offset, day, mask=place_animal)
                tl.store(yield_ptr + board_offset, 0, mask=place_animal)
                tl.store(consecutive_ptr + board_offset, 0, mask=place_animal)
                tl.store(care_bonus_ptr + board_offset, 0, mask=place_animal)
                tl.store(fed_ptr + board_offset, 0, mask=place_animal)
                tl.store(cared_ptr + board_offset, 0, mask=place_animal)
                tl.store(fert_available_ptr + board_offset, 0, mask=place_animal)
                place_shed = execute & shed_adjacent & ~structure_match & valid_item
                place_amount = tl.where(place_shed, tl.minimum(quantity, tl.minimum(carried, room)), 0)
                tl.store(inventory_ptr + inventory_offset, carried - place_amount, mask=place_shed)
                tl.store(shed_ptr + shed_offset, shed_item + place_amount, mask=place_shed)


def run_common_interactions(state, unit_ops, unit_args, plant_allowed, unit: int, day: int, max_units: int) -> None:
    """Run supported operations for one unit slot on the current CUDA stream."""
    if not TRITON_AVAILABLE:  # pragma: no cover - guarded by caller
        raise RuntimeError("Triton is not available")
    pair_count = unit_ops.shape[0] * unit_ops.shape[1]
    block = 256
    _common_interaction_kernel[(triton.cdiv(pair_count, block),)](
        unit_ops,
        unit_args,
        state.unit_active,
        plant_allowed,
        state.positions,
        state.tile_type,
        state.planted_day,
        state.placed_day,
        state.watered,
        state.fed,
        state.cared,
        state.consecutive,
        state.yield_units,
        state.max_lifespan_step,
        state.fertilized_until_day,
        state.fertilizer_available,
        state.pending_care_bonus,
        state.unit_inventory,
        state.seeds,
        unit,
        day,
        pair_count,
        MAX_UNITS=max_units,
        ITEM_COUNT=12,
        BLOCK=block,
    )


def run_inventory_interactions(
    state,
    unit_ops,
    unit_args,
    unit_quantities,
    unit: int,
    day: int,
    max_units: int,
    shed_capacity: int,
    op_kind: int,
) -> None:
    """Run DROP/PICKUP/PLACE for one unit slot on the current CUDA stream."""
    if not TRITON_AVAILABLE:  # pragma: no cover - guarded by caller
        raise RuntimeError("Triton is not available")
    pair_count = unit_ops.shape[0] * unit_ops.shape[1]
    block = 256
    _inventory_interaction_kernel[(triton.cdiv(pair_count, block),)](
        unit_ops,
        unit_args,
        unit_quantities,
        state.unit_active,
        state.positions,
        state.tile_type,
        state.placed_day,
        state.fed,
        state.cared,
        state.consecutive,
        state.yield_units,
        state.fertilizer_available,
        state.pending_care_bonus,
        state.unit_inventory,
        state.shed,
        unit,
        day,
        pair_count,
        MAX_UNITS=max_units,
        ITEM_COUNT=12,
        SHED_CAPACITY=shed_capacity,
        OP_KIND=op_kind,
        BLOCK=block,
    )
