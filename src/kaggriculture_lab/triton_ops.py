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
    def _move_units_kernel(
        op_ptr,
        active_ptr,
        position_ptr,
        unit_count,
        BLOCK: tl.constexpr,
    ):
        unit = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = unit < unit_count
        op = tl.load(op_ptr + unit, mask=valid, other=0).to(tl.int32)
        active = tl.load(active_ptr + unit, mask=valid, other=0).to(tl.int1)
        moving = active & (op >= 1) & (op <= 4)
        position_offset = unit * 2
        x = tl.load(position_ptr + position_offset, mask=valid, other=0).to(tl.int32)
        y = tl.load(position_ptr + position_offset + 1, mask=valid, other=0).to(tl.int32)
        dx = tl.where(op == 3, 1, 0) - tl.where(op == 4, 1, 0)
        dy = tl.where(op == 2, 1, 0) - tl.where(op == 1, 1, 0)
        nx = x + dx
        ny = y + dy
        update = moving & (nx >= 0) & (nx < 10) & (ny >= 0) & (ny < 10)
        tl.store(position_ptr + position_offset, nx, mask=valid & update)
        tl.store(position_ptr + position_offset + 1, ny, mask=valid & update)

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


    @triton.jit
    def _fixed_market_kernel(
        op_ptr,
        arg_ptr,
        quantity_ptr,
        money_ptr,
        position_ptr,
        active_ptr,
        hands_count_ptr,
        hires_today_ptr,
        unit_inventory_ptr,
        unlocked_count_ptr,
        tile_ptr,
        shed_ptr,
        seeds_ptr,
        hire_cost_ptr,
        land_price_ptr,
        seed_cost_ptr,
        animal_cost_ptr,
        pair_count,
        ORDER: tl.constexpr,
        OP_KIND: tl.constexpr,
        MARKET_ORDERS: tl.constexpr,
        MAX_QUANTITY: tl.constexpr,
        MAX_UNITS: tl.constexpr,
        ITEM_COUNT: tl.constexpr,
        CROP_COUNT: tl.constexpr,
        ANIMAL_COUNT: tl.constexpr,
        SHED_CAPACITY: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid_pair = pair < pair_count
        action_offset = pair * MARKET_ORDERS + ORDER
        op = tl.load(op_ptr + action_offset, mask=valid_pair, other=0).to(tl.int32)
        execute = valid_pair & (op == OP_KIND)
        money = tl.load(money_ptr + pair, mask=execute, other=0.0).to(tl.float32)

        if OP_KIND == 1:  # HIRE
            hands = tl.load(hands_count_ptr + pair, mask=execute, other=MAX_UNITS).to(tl.int32)
            hires = tl.load(hires_today_ptr + pair, mask=execute, other=MAX_UNITS).to(tl.int32)
            hire_index = tl.maximum(0, tl.minimum(hires, MAX_UNITS - 1))
            cost = tl.load(hire_cost_ptr + hire_index, mask=execute, other=0.0).to(tl.float32)
            can_hire = execute & (hands < MAX_UNITS - 1) & (money >= cost)

            occupancy0 = tl.zeros((BLOCK,), tl.int32)
            occupancy1 = tl.zeros((BLOCK,), tl.int32)
            occupancy2 = tl.zeros((BLOCK,), tl.int32)
            occupancy3 = tl.zeros((BLOCK,), tl.int32)
            for unit in tl.static_range(0, MAX_UNITS):
                unit_offset = pair * MAX_UNITS + unit
                active = tl.load(active_ptr + unit_offset, mask=can_hire, other=0).to(tl.int1)
                position_offset = unit_offset * 2
                x = tl.load(position_ptr + position_offset, mask=can_hire, other=0).to(tl.int32)
                y = tl.load(position_ptr + position_offset + 1, mask=can_hire, other=0).to(tl.int32)
                occupancy0 += active & (x == 4) & (y == 4)
                occupancy1 += active & (x == 5) & (y == 4)
                occupancy2 += active & (x == 4) & (y == 5)
                occupancy3 += active & (x == 5) & (y == 5)

            choose0 = (occupancy0 <= occupancy1) & (occupancy0 <= occupancy2) & (occupancy0 <= occupancy3)
            choose1 = ~choose0 & (occupancy1 <= occupancy2) & (occupancy1 <= occupancy3)
            choose2 = ~choose0 & ~choose1 & (occupancy2 <= occupancy3)
            x = tl.where(choose0 | choose2, 4, 5)
            y = tl.where(choose0 | choose1, 4, 5)
            slot = tl.minimum(hands + 1, MAX_UNITS - 1)
            unit_offset = pair * MAX_UNITS + slot
            position_offset = unit_offset * 2
            tl.store(position_ptr + position_offset, x, mask=can_hire)
            tl.store(position_ptr + position_offset + 1, y, mask=can_hire)
            tl.store(active_ptr + unit_offset, 1, mask=can_hire)
            for item in tl.static_range(0, ITEM_COUNT):
                tl.store(unit_inventory_ptr + unit_offset * ITEM_COUNT + item, 0, mask=can_hire)
            tl.store(hands_count_ptr + pair, hands + 1, mask=can_hire)
            tl.store(hires_today_ptr + pair, hires + 1, mask=can_hire)
            tl.store(money_ptr + pair, money - cost, mask=can_hire)

        elif OP_KIND == 2:  # BUY_LAND
            unlocked = tl.load(unlocked_count_ptr + pair, mask=execute, other=4).to(tl.int32)
            stage = tl.maximum(0, tl.minimum(unlocked - 1, 3))
            cost = tl.load(land_price_ptr + stage, mask=execute, other=0.0).to(tl.float32)
            can_buy = execute & (unlocked < 4) & (money >= cost)
            y_base = tl.where(stage == 0, 0, 5)
            x_base = tl.where(stage == 1, 0, 5)
            for cell in tl.static_range(0, 25):
                y = y_base + cell // 5
                x = x_base + cell % 5
                board_offset = pair * 100 + y * 10 + x
                tile = tl.load(tile_ptr + board_offset, mask=can_buy, other=0).to(tl.int32)
                tl.store(tile_ptr + board_offset, 0, mask=can_buy & (tile == 1))
            tl.store(unlocked_count_ptr + pair, unlocked + 1, mask=can_buy)
            tl.store(money_ptr + pair, money - cost, mask=can_buy)

        elif OP_KIND == 3:  # BUY_SEED
            arg = tl.load(arg_ptr + action_offset, mask=execute, other=-1).to(tl.int32)
            quantity = tl.load(quantity_ptr + action_offset, mask=execute, other=0).to(tl.int32)
            quantity = tl.maximum(0, tl.minimum(quantity, MAX_QUANTITY))
            valid_crop = execute & (arg >= 0) & (arg < CROP_COUNT)
            crop = tl.maximum(0, tl.minimum(arg, CROP_COUNT - 1))
            cost = tl.load(seed_cost_ptr + crop, mask=valid_crop, other=1.0).to(tl.float32)
            affordable = tl.floor(money / cost).to(tl.int32)
            units = tl.minimum(quantity, affordable)
            seed_offset = pair * CROP_COUNT + crop
            seeds = tl.load(seeds_ptr + seed_offset, mask=valid_crop, other=0).to(tl.int32)
            tl.store(seeds_ptr + seed_offset, seeds + units, mask=valid_crop)
            tl.store(money_ptr + pair, money - units.to(tl.float32) * cost, mask=valid_crop)

        elif OP_KIND == 5:  # BUY_ANIMAL
            arg = tl.load(arg_ptr + action_offset, mask=execute, other=-1).to(tl.int32)
            quantity = tl.load(quantity_ptr + action_offset, mask=execute, other=0).to(tl.int32)
            quantity = tl.maximum(0, tl.minimum(quantity, MAX_QUANTITY))
            valid_animal = execute & (arg >= 0) & (arg < ANIMAL_COUNT)
            animal = tl.maximum(0, tl.minimum(arg, ANIMAL_COUNT - 1))
            cost = tl.load(animal_cost_ptr + animal, mask=valid_animal, other=1.0).to(tl.float32)
            shed_total = tl.zeros((BLOCK,), tl.int32)
            for item in tl.static_range(0, ITEM_COUNT):
                shed_total += tl.load(shed_ptr + pair * ITEM_COUNT + item, mask=valid_animal, other=0).to(tl.int32)
            room = tl.maximum(SHED_CAPACITY - shed_total, 0)
            affordable = tl.floor(money / cost).to(tl.int32)
            units = tl.minimum(quantity, tl.minimum(affordable, room))
            shed_offset = pair * ITEM_COUNT + 9 + animal
            held = tl.load(shed_ptr + shed_offset, mask=valid_animal, other=0).to(tl.int32)
            tl.store(shed_ptr + shed_offset, held + units, mask=valid_animal)
            tl.store(money_ptr + pair, money - units.to(tl.float32) * cost, mask=valid_animal)


    @triton.jit
    def _market_shape(code, x):
        x = tl.maximum(x, 0.0)
        return tl.where(code == 0, x, tl.where(code == 1, x * x, tl.where(code == 2, tl.sqrt(x), tl.log(x + 1.0))))


    @triton.jit
    def _market_quote(
        item,
        inventory,
        base_ptr,
        scale_ptr,
        below_func_ptr,
        below_target_ptr,
        above_func_ptr,
        above_target_ptr,
    ):
        below = inventory < 10000
        base = tl.load(base_ptr + item).to(tl.float32)
        scale = tl.load(scale_ptr + item).to(tl.float32)
        func = tl.where(below, tl.load(below_func_ptr + item), tl.load(above_func_ptr + item))
        target = tl.where(below, tl.load(below_target_ptr + item), tl.load(above_target_ptr + item)).to(tl.float32)
        distance = tl.abs(inventory.to(tl.float32) - 10000.0)
        amplitude = target * base / _market_shape(func, scale)
        price = tl.where(below, base + amplitude * _market_shape(func, distance), base - amplitude * _market_shape(func, distance))
        return tl.maximum(tl.floor(price + 0.5), 1.0)


    @triton.jit
    def _dynamic_market_player(
        env,
        valid_env,
        player: tl.constexpr,
        order: tl.constexpr,
        quote,
        remaining,
        money,
        shed_total,
        op_ptr,
        arg_ptr,
        market_inventory_ptr,
        shed_ptr,
        base_ptr,
        scale_ptr,
        below_func_ptr,
        below_target_ptr,
        above_func_ptr,
        above_target_ptr,
        MARKET_ORDERS: tl.constexpr,
        ITEM_COUNT: tl.constexpr,
        SHED_CAPACITY: tl.constexpr,
    ):
        action_offset = (env * 2 + player) * MARKET_ORDERS + order
        op = tl.load(op_ptr + action_offset, mask=valid_env, other=0)
        arg = tl.load(arg_ptr + action_offset, mask=valid_env, other=0).to(tl.int32)
        item = tl.maximum(0, tl.minimum(arg, 8))
        inventory_offset = env * 9 + item
        inventory = tl.load(market_inventory_ptr + inventory_offset, mask=valid_env, other=10000).to(tl.int32)
        shed_offset = (env * 2 + player) * ITEM_COUNT + item
        held = tl.load(shed_ptr + shed_offset, mask=valid_env, other=0).to(tl.int32)
        active = valid_env & (remaining > 0)

        sell = active & (op == 6) & (arg >= 0) & (arg < 9)
        sell_success = sell & (held > 0)
        tl.store(shed_ptr + shed_offset, held - 1, mask=sell_success)
        inventory_after_sell = inventory + tl.where(sell_success & (quote > 1), 1, 0)
        tl.store(market_inventory_ptr + inventory_offset, inventory_after_sell, mask=sell_success & (quote > 1))
        money += tl.where(sell_success, quote, 0.0)
        shed_total -= tl.where(sell_success, 1, 0)
        remaining = tl.where(sell, tl.where(sell_success, remaining - 1, 0), remaining)

        inventory = tl.where(sell_success & (quote > 1), inventory_after_sell, inventory)
        buy = active & (op == 4) & ((arg == 0) | (arg == 8))
        buy_success = buy & (shed_total < SHED_CAPACITY) & (money >= quote)
        held_after_sell = held - tl.where(sell_success, 1, 0)
        tl.store(shed_ptr + shed_offset, held_after_sell + 1, mask=buy_success)
        tl.store(market_inventory_ptr + inventory_offset, inventory - 1, mask=buy_success)
        money -= tl.where(buy_success, quote, 0.0)
        shed_total += tl.where(buy_success, 1, 0)
        remaining = tl.where(buy, tl.where(buy_success, remaining - 1, 0), remaining)
        malformed = active & ~sell & ~buy
        remaining = tl.where(malformed, 0, remaining)
        return remaining, money, shed_total


    @triton.jit
    def _dynamic_market_kernel(
        op_ptr,
        arg_ptr,
        quantity_ptr,
        money_ptr,
        market_inventory_ptr,
        shed_ptr,
        base_ptr,
        scale_ptr,
        below_func_ptr,
        below_target_ptr,
        above_func_ptr,
        above_target_ptr,
        env_count,
        ORDER: tl.constexpr,
        ROUNDS: tl.constexpr,
        MARKET_ORDERS: tl.constexpr,
        ITEM_COUNT: tl.constexpr,
        SHED_CAPACITY: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        env = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid_env = env < env_count
        offset0 = (env * 2) * MARKET_ORDERS + ORDER
        offset1 = (env * 2 + 1) * MARKET_ORDERS + ORDER
        op0 = tl.load(op_ptr + offset0, mask=valid_env, other=0)
        op1 = tl.load(op_ptr + offset1, mask=valid_env, other=0)
        arg0 = tl.load(arg_ptr + offset0, mask=valid_env, other=0).to(tl.int32)
        arg1 = tl.load(arg_ptr + offset1, mask=valid_env, other=0).to(tl.int32)
        item0 = tl.maximum(0, tl.minimum(arg0, 8))
        item1 = tl.maximum(0, tl.minimum(arg1, 8))
        quantity0 = tl.load(quantity_ptr + offset0, mask=valid_env, other=0).to(tl.int32)
        quantity1 = tl.load(quantity_ptr + offset1, mask=valid_env, other=0).to(tl.int32)
        remaining0 = tl.where((op0 == 4) | (op0 == 6), tl.maximum(0, tl.minimum(quantity0, ROUNDS)), 0)
        remaining1 = tl.where((op1 == 4) | (op1 == 6), tl.maximum(0, tl.minimum(quantity1, ROUNDS)), 0)
        money0 = tl.load(money_ptr + env * 2, mask=valid_env, other=0.0).to(tl.float32)
        money1 = tl.load(money_ptr + env * 2 + 1, mask=valid_env, other=0.0).to(tl.float32)
        shed_total0 = tl.zeros((BLOCK,), tl.int32)
        shed_total1 = tl.zeros((BLOCK,), tl.int32)
        for item_index in tl.static_range(0, ITEM_COUNT):
            shed_total0 += tl.load(shed_ptr + (env * 2) * ITEM_COUNT + item_index, mask=valid_env, other=0).to(tl.int32)
            shed_total1 += tl.load(shed_ptr + (env * 2 + 1) * ITEM_COUNT + item_index, mask=valid_env, other=0).to(tl.int32)

        for _ in tl.static_range(0, ROUNDS):
            snapshot0 = tl.load(market_inventory_ptr + env * 9 + item0, mask=valid_env, other=10000).to(tl.int32)
            snapshot1 = tl.load(market_inventory_ptr + env * 9 + item1, mask=valid_env, other=10000).to(tl.int32)
            quote0 = _market_quote(
                item0,
                snapshot0 - tl.where(op0 == 4, 1, 0),
                base_ptr,
                scale_ptr,
                below_func_ptr,
                below_target_ptr,
                above_func_ptr,
                above_target_ptr,
            )
            quote1 = _market_quote(
                item1,
                snapshot1 - tl.where(op1 == 4, 1, 0),
                base_ptr,
                scale_ptr,
                below_func_ptr,
                below_target_ptr,
                above_func_ptr,
                above_target_ptr,
            )
            remaining0, money0, shed_total0 = _dynamic_market_player(
                env,
                valid_env,
                0,
                ORDER,
                quote0,
                remaining0,
                money0,
                shed_total0,
                op_ptr,
                arg_ptr,
                market_inventory_ptr,
                shed_ptr,
                base_ptr,
                scale_ptr,
                below_func_ptr,
                below_target_ptr,
                above_func_ptr,
                above_target_ptr,
                MARKET_ORDERS,
                ITEM_COUNT,
                SHED_CAPACITY,
            )
            remaining1, money1, shed_total1 = _dynamic_market_player(
                env,
                valid_env,
                1,
                ORDER,
                quote1,
                remaining1,
                money1,
                shed_total1,
                op_ptr,
                arg_ptr,
                market_inventory_ptr,
                shed_ptr,
                base_ptr,
                scale_ptr,
                below_func_ptr,
                below_target_ptr,
                above_func_ptr,
                above_target_ptr,
                MARKET_ORDERS,
                ITEM_COUNT,
                SHED_CAPACITY,
            )
        tl.store(money_ptr + env * 2, money0, mask=valid_env)
        tl.store(money_ptr + env * 2 + 1, money1, mask=valid_env)


    @triton.jit
    def _all_market_fixed_player(
        env,
        valid_env,
        player: tl.constexpr,
        op,
        arg,
        quantity,
        money,
        shed_total,
        position_ptr,
        active_ptr,
        hands_count_ptr,
        hires_today_ptr,
        unit_inventory_ptr,
        unlocked_count_ptr,
        tile_ptr,
        shed_ptr,
        seeds_ptr,
        hire_cost_ptr,
        land_price_ptr,
        seed_cost_ptr,
        animal_cost_ptr,
        MAX_UNITS: tl.constexpr,
        ITEM_COUNT: tl.constexpr,
        CROP_COUNT: tl.constexpr,
        ANIMAL_COUNT: tl.constexpr,
        SHED_CAPACITY: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pair = env * 2 + player

        hire = valid_env & (op == 1)
        hands = tl.load(hands_count_ptr + pair, mask=hire, other=MAX_UNITS).to(tl.int32)
        hires = tl.load(hires_today_ptr + pair, mask=hire, other=MAX_UNITS).to(tl.int32)
        hire_index = tl.maximum(0, tl.minimum(hires, MAX_UNITS - 1))
        hire_cost = tl.load(hire_cost_ptr + hire_index, mask=hire, other=0.0).to(tl.float32)
        can_hire = hire & (hands < MAX_UNITS - 1) & (money >= hire_cost)
        occupancy0 = tl.zeros((BLOCK,), tl.int32)
        occupancy1 = tl.zeros((BLOCK,), tl.int32)
        occupancy2 = tl.zeros((BLOCK,), tl.int32)
        occupancy3 = tl.zeros((BLOCK,), tl.int32)
        for unit in tl.static_range(0, MAX_UNITS):
            unit_offset = pair * MAX_UNITS + unit
            active = tl.load(active_ptr + unit_offset, mask=can_hire, other=0).to(tl.int1)
            position_offset = unit_offset * 2
            x = tl.load(position_ptr + position_offset, mask=can_hire, other=0).to(tl.int32)
            y = tl.load(position_ptr + position_offset + 1, mask=can_hire, other=0).to(tl.int32)
            occupancy0 += active & (x == 4) & (y == 4)
            occupancy1 += active & (x == 5) & (y == 4)
            occupancy2 += active & (x == 4) & (y == 5)
            occupancy3 += active & (x == 5) & (y == 5)
        choose0 = (occupancy0 <= occupancy1) & (occupancy0 <= occupancy2) & (occupancy0 <= occupancy3)
        choose1 = ~choose0 & (occupancy1 <= occupancy2) & (occupancy1 <= occupancy3)
        choose2 = ~choose0 & ~choose1 & (occupancy2 <= occupancy3)
        hire_x = tl.where(choose0 | choose2, 4, 5)
        hire_y = tl.where(choose0 | choose1, 4, 5)
        slot = tl.minimum(hands + 1, MAX_UNITS - 1)
        unit_offset = pair * MAX_UNITS + slot
        position_offset = unit_offset * 2
        tl.store(position_ptr + position_offset, hire_x, mask=can_hire)
        tl.store(position_ptr + position_offset + 1, hire_y, mask=can_hire)
        tl.store(active_ptr + unit_offset, 1, mask=can_hire)
        for item in tl.static_range(0, ITEM_COUNT):
            tl.store(unit_inventory_ptr + unit_offset * ITEM_COUNT + item, 0, mask=can_hire)
        tl.store(hands_count_ptr + pair, hands + 1, mask=can_hire)
        tl.store(hires_today_ptr + pair, hires + 1, mask=can_hire)
        money -= tl.where(can_hire, hire_cost, 0.0)

        buy_land = valid_env & (op == 2)
        unlocked = tl.load(unlocked_count_ptr + pair, mask=buy_land, other=4).to(tl.int32)
        stage = tl.maximum(0, tl.minimum(unlocked - 1, 3))
        land_cost = tl.load(land_price_ptr + stage, mask=buy_land, other=0.0).to(tl.float32)
        can_buy_land = buy_land & (unlocked < 4) & (money >= land_cost)
        y_base = tl.where(stage == 0, 0, 5)
        x_base = tl.where(stage == 1, 0, 5)
        for cell in tl.static_range(0, 25):
            y = y_base + cell // 5
            x = x_base + cell % 5
            board_offset = pair * 100 + y * 10 + x
            tile = tl.load(tile_ptr + board_offset, mask=can_buy_land, other=0).to(tl.int32)
            tl.store(tile_ptr + board_offset, 0, mask=can_buy_land & (tile == 1))
        tl.store(unlocked_count_ptr + pair, unlocked + 1, mask=can_buy_land)
        money -= tl.where(can_buy_land, land_cost, 0.0)

        buy_seed = valid_env & (op == 3) & (arg >= 0) & (arg < CROP_COUNT)
        crop = tl.maximum(0, tl.minimum(arg, CROP_COUNT - 1))
        seed_cost = tl.load(seed_cost_ptr + crop, mask=buy_seed, other=1.0).to(tl.float32)
        seed_units = tl.minimum(quantity, tl.floor(money / seed_cost).to(tl.int32))
        seed_offset = pair * CROP_COUNT + crop
        seeds = tl.load(seeds_ptr + seed_offset, mask=buy_seed, other=0).to(tl.int32)
        tl.store(seeds_ptr + seed_offset, seeds + seed_units, mask=buy_seed)
        money -= tl.where(buy_seed, seed_units.to(tl.float32) * seed_cost, 0.0)

        buy_animal = valid_env & (op == 5) & (arg >= 0) & (arg < ANIMAL_COUNT)
        animal = tl.maximum(0, tl.minimum(arg, ANIMAL_COUNT - 1))
        animal_cost = tl.load(animal_cost_ptr + animal, mask=buy_animal, other=1.0).to(tl.float32)
        room = tl.maximum(SHED_CAPACITY - shed_total, 0)
        animal_units = tl.minimum(quantity, tl.minimum(tl.floor(money / animal_cost).to(tl.int32), room))
        shed_offset = pair * ITEM_COUNT + 9 + animal
        held = tl.load(shed_ptr + shed_offset, mask=buy_animal, other=0).to(tl.int32)
        tl.store(shed_ptr + shed_offset, held + animal_units, mask=buy_animal)
        money -= tl.where(buy_animal, animal_units.to(tl.float32) * animal_cost, 0.0)
        shed_total += tl.where(buy_animal, animal_units, 0)
        return money, shed_total


    @triton.jit
    def _all_market_dynamic_player(
        env,
        valid_env,
        player: tl.constexpr,
        op,
        arg,
        item,
        quote,
        remaining,
        money,
        shed_total,
        market_inventory_ptr,
        shed_ptr,
        ITEM_COUNT: tl.constexpr,
        SHED_CAPACITY: tl.constexpr,
    ):
        pair = env * 2 + player
        inventory_offset = env * 9 + item
        inventory = tl.load(market_inventory_ptr + inventory_offset, mask=valid_env, other=10000).to(tl.int32)
        shed_offset = pair * ITEM_COUNT + item
        held = tl.load(shed_ptr + shed_offset, mask=valid_env, other=0).to(tl.int32)
        active = valid_env & (remaining > 0)

        sell = active & (op == 6) & (arg >= 0) & (arg < 9)
        sell_success = sell & (held > 0)
        tl.store(shed_ptr + shed_offset, held - 1, mask=sell_success)
        inventory_after_sell = inventory + tl.where(sell_success & (quote > 1), 1, 0)
        tl.store(market_inventory_ptr + inventory_offset, inventory_after_sell, mask=sell_success & (quote > 1))
        money += tl.where(sell_success, quote, 0.0)
        shed_total -= tl.where(sell_success, 1, 0)
        remaining = tl.where(sell, tl.where(sell_success, remaining - 1, 0), remaining)

        inventory = tl.where(sell_success & (quote > 1), inventory_after_sell, inventory)
        buy = active & (op == 4) & ((arg == 0) | (arg == 8))
        buy_success = buy & (shed_total < SHED_CAPACITY) & (money >= quote)
        held_after_sell = held - tl.where(sell_success, 1, 0)
        tl.store(shed_ptr + shed_offset, held_after_sell + 1, mask=buy_success)
        tl.store(market_inventory_ptr + inventory_offset, inventory - 1, mask=buy_success)
        money -= tl.where(buy_success, quote, 0.0)
        shed_total += tl.where(buy_success, 1, 0)
        remaining = tl.where(buy, tl.where(buy_success, remaining - 1, 0), remaining)
        malformed = active & ~sell & ~buy
        remaining = tl.where(malformed, 0, remaining)
        return remaining, money, shed_total


    @triton.jit
    def _all_market_orders_kernel(
        op_ptr,
        arg_ptr,
        quantity_ptr,
        money_ptr,
        position_ptr,
        active_ptr,
        hands_count_ptr,
        hires_today_ptr,
        unit_inventory_ptr,
        unlocked_count_ptr,
        tile_ptr,
        shed_ptr,
        seeds_ptr,
        market_inventory_ptr,
        hire_cost_ptr,
        land_price_ptr,
        seed_cost_ptr,
        animal_cost_ptr,
        base_ptr,
        scale_ptr,
        below_func_ptr,
        below_target_ptr,
        above_func_ptr,
        above_target_ptr,
        env_count,
        MARKET_ORDERS: tl.constexpr,
        MAX_QUANTITY: tl.constexpr,
        MAX_UNITS: tl.constexpr,
        ITEM_COUNT: tl.constexpr,
        CROP_COUNT: tl.constexpr,
        ANIMAL_COUNT: tl.constexpr,
        SHED_CAPACITY: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        env = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid_env = env < env_count
        money0 = tl.load(money_ptr + env * 2, mask=valid_env, other=0.0).to(tl.float32)
        money1 = tl.load(money_ptr + env * 2 + 1, mask=valid_env, other=0.0).to(tl.float32)
        shed_total0 = tl.zeros((BLOCK,), tl.int32)
        shed_total1 = tl.zeros((BLOCK,), tl.int32)
        for item_index in tl.static_range(0, ITEM_COUNT):
            shed_total0 += tl.load(shed_ptr + (env * 2) * ITEM_COUNT + item_index, mask=valid_env, other=0).to(tl.int32)
            shed_total1 += tl.load(shed_ptr + (env * 2 + 1) * ITEM_COUNT + item_index, mask=valid_env, other=0).to(tl.int32)

        for order in tl.range(0, MARKET_ORDERS):
            offset0 = (env * 2) * MARKET_ORDERS + order
            offset1 = (env * 2 + 1) * MARKET_ORDERS + order
            op0 = tl.load(op_ptr + offset0, mask=valid_env, other=0).to(tl.int32)
            op1 = tl.load(op_ptr + offset1, mask=valid_env, other=0).to(tl.int32)
            arg0 = tl.load(arg_ptr + offset0, mask=valid_env, other=-1).to(tl.int32)
            arg1 = tl.load(arg_ptr + offset1, mask=valid_env, other=-1).to(tl.int32)
            quantity0 = tl.load(quantity_ptr + offset0, mask=valid_env, other=0).to(tl.int32)
            quantity1 = tl.load(quantity_ptr + offset1, mask=valid_env, other=0).to(tl.int32)
            quantity0 = tl.maximum(0, tl.minimum(quantity0, MAX_QUANTITY))
            quantity1 = tl.maximum(0, tl.minimum(quantity1, MAX_QUANTITY))
            money0, shed_total0 = _all_market_fixed_player(
                env, valid_env, 0, op0, arg0, quantity0, money0, shed_total0,
                position_ptr, active_ptr, hands_count_ptr, hires_today_ptr, unit_inventory_ptr,
                unlocked_count_ptr, tile_ptr, shed_ptr, seeds_ptr, hire_cost_ptr, land_price_ptr,
                seed_cost_ptr, animal_cost_ptr, MAX_UNITS, ITEM_COUNT, CROP_COUNT, ANIMAL_COUNT,
                SHED_CAPACITY, BLOCK,
            )
            money1, shed_total1 = _all_market_fixed_player(
                env, valid_env, 1, op1, arg1, quantity1, money1, shed_total1,
                position_ptr, active_ptr, hands_count_ptr, hires_today_ptr, unit_inventory_ptr,
                unlocked_count_ptr, tile_ptr, shed_ptr, seeds_ptr, hire_cost_ptr, land_price_ptr,
                seed_cost_ptr, animal_cost_ptr, MAX_UNITS, ITEM_COUNT, CROP_COUNT, ANIMAL_COUNT,
                SHED_CAPACITY, BLOCK,
            )

            item0 = tl.maximum(0, tl.minimum(arg0, 8))
            item1 = tl.maximum(0, tl.minimum(arg1, 8))
            remaining0 = tl.where((op0 == 4) | (op0 == 6), quantity0, 0)
            remaining1 = tl.where((op1 == 4) | (op1 == 6), quantity1, 0)
            for _ in tl.range(0, MAX_QUANTITY):
                snapshot0 = tl.load(market_inventory_ptr + env * 9 + item0, mask=valid_env, other=10000).to(tl.int32)
                snapshot1 = tl.load(market_inventory_ptr + env * 9 + item1, mask=valid_env, other=10000).to(tl.int32)
                quote0 = _market_quote(
                    item0, snapshot0 - tl.where(op0 == 4, 1, 0), base_ptr, scale_ptr,
                    below_func_ptr, below_target_ptr, above_func_ptr, above_target_ptr,
                )
                quote1 = _market_quote(
                    item1, snapshot1 - tl.where(op1 == 4, 1, 0), base_ptr, scale_ptr,
                    below_func_ptr, below_target_ptr, above_func_ptr, above_target_ptr,
                )
                remaining0, money0, shed_total0 = _all_market_dynamic_player(
                    env, valid_env, 0, op0, arg0, item0, quote0, remaining0, money0,
                    shed_total0, market_inventory_ptr, shed_ptr, ITEM_COUNT, SHED_CAPACITY,
                )
                remaining1, money1, shed_total1 = _all_market_dynamic_player(
                    env, valid_env, 1, op1, arg1, item1, quote1, remaining1, money1,
                    shed_total1, market_inventory_ptr, shed_ptr, ITEM_COUNT, SHED_CAPACITY,
                )
        tl.store(money_ptr + env * 2, money0, mask=valid_env)
        tl.store(money_ptr + env * 2 + 1, money1, mask=valid_env)


    @triton.jit
    def _town_consume_kernel(
        market_inventory_ptr,
        market_prices_ptr,
        shop_counts_ptr,
        shop_demand_ptr,
        base_ptr,
        scale_ptr,
        below_func_ptr,
        below_target_ptr,
        above_func_ptr,
        above_target_ptr,
        env_count,
        SHOP_ACTIVE: tl.constexpr,
        CENTER_ACTIVE: tl.constexpr,
        ITEM_COUNT: tl.constexpr,
        SHOP_COUNT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = lane < env_count * ITEM_COUNT
        env = lane // ITEM_COUNT
        item = lane - env * ITEM_COUNT
        inventory = tl.load(market_inventory_ptr + lane, mask=valid, other=10000).to(tl.int32)

        if SHOP_ACTIVE:
            demand = tl.zeros((BLOCK,), tl.int32)
            for shop in tl.static_range(0, SHOP_COUNT):
                count = tl.load(shop_counts_ptr + env * SHOP_COUNT + shop, mask=valid, other=0).to(tl.int32)
                units = tl.load(shop_demand_ptr + shop * ITEM_COUNT + item, mask=valid, other=0).to(tl.int32)
                demand += count * units
            inventory -= demand
        if CENTER_ACTIVE:
            inventory -= 1
            inventory += tl.where(item == 8, 1, 0)
        tl.store(market_inventory_ptr + lane, inventory, mask=valid)

        price = _market_quote(
            item,
            inventory,
            base_ptr,
            scale_ptr,
            below_func_ptr,
            below_target_ptr,
            above_func_ptr,
            above_target_ptr,
        )
        tl.store(market_prices_ptr + lane, price, mask=valid)


    @triton.jit
    def _decay_plants_kernel(
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
        cell_count,
        step,
        BLOCK: tl.constexpr,
    ):
        cell = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = cell < cell_count
        tile = tl.load(tile_ptr + cell, mask=valid, other=0).to(tl.int32)
        max_life = tl.load(max_life_ptr + cell, mask=valid, other=-1).to(tl.int32)
        age = step - max_life
        decay = valid & (tile >= 3) & (tile < 8) & (max_life >= 0) & (age >= 0) & ((age & 1) == 0)
        old_yield = tl.load(yield_ptr + cell, mask=decay, other=0).to(tl.int32)
        new_yield = old_yield - 1
        weed = decay & (new_yield <= 0)

        tl.store(yield_ptr + cell, tl.where(weed, 0, new_yield), mask=decay)
        tl.store(tile_ptr + cell, 2, mask=weed)
        tl.store(planted_ptr + cell, -1, mask=weed)
        tl.store(placed_ptr + cell, -1, mask=weed)
        tl.store(max_life_ptr + cell, -1, mask=weed)
        tl.store(fert_until_ptr + cell, -1, mask=weed)
        tl.store(watered_ptr + cell, 0, mask=weed)
        tl.store(fed_ptr + cell, 0, mask=weed)
        tl.store(cared_ptr + cell, 0, mask=weed)
        tl.store(fert_available_ptr + cell, 0, mask=weed)
        tl.store(consecutive_ptr + cell, 0, mask=weed)
        tl.store(care_bonus_ptr + cell, 0, mask=weed)


    @triton.jit
    def _end_of_day_board_kernel(
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
        episode_seed_ptr,
        cell_count,
        day,
        next_day,
        weed_chance,
        TURNS_PER_DAY: tl.constexpr,
        WEED_ENABLED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        cell = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = cell < cell_count
        tile = tl.load(tile_ptr + cell, mask=valid, other=0).to(tl.int32)
        planted_day = tl.load(planted_ptr + cell, mask=valid, other=-1).to(tl.int32)
        placed_day = tl.load(placed_ptr + cell, mask=valid, other=-1).to(tl.int32)
        watered = tl.load(watered_ptr + cell, mask=valid, other=0).to(tl.int1)
        fed = tl.load(fed_ptr + cell, mask=valid, other=0).to(tl.int1)
        cared = tl.load(cared_ptr + cell, mask=valid, other=0).to(tl.int1)
        consecutive = tl.load(consecutive_ptr + cell, mask=valid, other=0).to(tl.int32)
        held_yield = tl.load(yield_ptr + cell, mask=valid, other=0).to(tl.int32)
        fert_until = tl.load(fert_until_ptr + cell, mask=valid, other=-1).to(tl.int32)
        care_bonus = tl.load(care_bonus_ptr + cell, mask=valid, other=0).to(tl.int32)

        plant = valid & (tile >= 3) & (tile < 8)
        plant_consecutive = tl.where(watered, 0, consecutive + 1)
        dead = plant & (plant_consecutive >= 2)
        live_plant = plant & ~dead
        tl.store(consecutive_ptr + cell, plant_consecutive, mask=plant)
        tl.store(watered_ptr + cell, 0, mask=plant)

        crop = tl.maximum(0, tl.minimum(tile - 3, 4))
        crop_first = tl.where(crop == 0, 2, tl.where(crop == 1, 2, tl.where(crop == 2, 8, 10)))
        crop_interval = tl.where(crop == 2, 1, tl.where(crop == 3, 2, 1))
        crop_max_yield = tl.where(crop == 0, 6, tl.where((crop >= 1) & (crop <= 3), 4, 6))
        ongoing = (crop == 2) | (crop == 3)
        plant_age = next_day - planted_day - crop_first
        production_count = plant_age // crop_interval + 1
        produce_plant = (
            live_plant
            & ongoing
            & (plant_age >= 0)
            & ((plant_age % crop_interval) == 0)
            & (production_count <= crop_max_yield)
        )
        fertilized = watered & (fert_until >= day)
        plant_yield = tl.minimum(held_yield + tl.where(fertilized, 2, 1), crop_max_yield)
        tl.store(yield_ptr + cell, plant_yield, mask=produce_plant)
        last_crop = produce_plant & (production_count == crop_max_yield)
        tl.store(max_life_ptr + cell, (next_day + 1) * TURNS_PER_DAY, mask=last_crop)

        animal = valid & (tile >= 10) & (tile < 13)
        animal_index = tl.maximum(0, tl.minimum(tile - 10, 2))
        animal_consecutive = tl.where(fed, 0, consecutive + 1)
        escaped = animal & (animal_consecutive >= 2)
        live_animal = animal & ~escaped
        tl.store(consecutive_ptr + cell, animal_consecutive, mask=animal)

        animal_first = tl.where(animal_index == 0, 4, tl.where(animal_index == 1, 8, 6))
        animal_interval = tl.where(animal_index == 0, 1, tl.where(animal_index == 1, 2, 3))
        animal_max_held = tl.where(animal_index == 0, 4, 6)
        animal_age = next_day - placed_day - animal_first
        produce_animal = live_animal & (animal_age >= 0) & ((animal_age % animal_interval) == 0)
        produced_yield = tl.minimum(held_yield + 1 + tl.where(fed, care_bonus, 0), animal_max_held)
        tl.store(yield_ptr + cell, produced_yield, mask=produce_animal)
        bank = live_animal & cared & fed
        next_bonus = tl.where(produce_animal, 0, care_bonus) + tl.where(bank, 1, 0)
        tl.store(care_bonus_ptr + cell, next_bonus, mask=live_animal & (produce_animal | bank))
        tl.store(fert_available_ptr + cell, 1, mask=live_animal)
        tl.store(fed_ptr + cell, 0, mask=live_animal)
        tl.store(cared_ptr + cell, 0, mask=live_animal)

        clear = dead | escaped
        escaped_structure = tl.where(animal_index == 0, 8, 9)
        tl.store(tile_ptr + cell, tl.where(dead, 2, escaped_structure), mask=clear)
        tl.store(planted_ptr + cell, -1, mask=clear)
        tl.store(placed_ptr + cell, -1, mask=clear)
        tl.store(max_life_ptr + cell, -1, mask=clear)
        tl.store(fert_until_ptr + cell, -1, mask=clear)
        tl.store(watered_ptr + cell, 0, mask=clear)
        tl.store(fed_ptr + cell, 0, mask=clear)
        tl.store(cared_ptr + cell, 0, mask=clear)
        tl.store(fert_available_ptr + cell, 0, mask=clear)
        tl.store(consecutive_ptr + cell, 0, mask=clear)
        tl.store(yield_ptr + cell, 0, mask=clear)
        tl.store(care_bonus_ptr + cell, 0, mask=clear)

        if WEED_ENABLED:
            env = cell // 200
            board_index = cell - env * 200
            seed = tl.load(episode_seed_ptr + env, mask=valid, other=0)
            hashed = (seed * 48271 + board_index * 69621 + day * 1000003 + 12345) % 2147483647
            uniform = hashed.to(tl.float32) / 2147483647.0
            spawn = valid & (tile == 0) & (uniform < weed_chance)
            tl.store(tile_ptr + cell, 2, mask=spawn)


    @triton.jit
    def _end_of_day_reset_kernel(
        unit_inventory_ptr,
        shed_ptr,
        position_ptr,
        unit_active_ptr,
        hands_count_ptr,
        hires_today_ptr,
        shop_counts_ptr,
        episode_seed_ptr,
        pair_count,
        day,
        SHED_CAPACITY: tl.constexpr,
        MAX_UNITS: tl.constexpr,
        ITEM_COUNT: tl.constexpr,
        SHOP_COUNT: tl.constexpr,
        SHOP_UNLOCK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = pair < pair_count
        shed_total = tl.zeros((BLOCK,), tl.int32)
        for item in tl.static_range(0, ITEM_COUNT):
            shed_total += tl.load(shed_ptr + pair * ITEM_COUNT + item, mask=valid, other=0).to(tl.int32)

        for item in tl.static_range(0, ITEM_COUNT):
            carried = tl.zeros((BLOCK,), tl.int32)
            for unit in tl.static_range(0, MAX_UNITS):
                inventory_offset = (pair * MAX_UNITS + unit) * ITEM_COUNT + item
                carried += tl.load(unit_inventory_ptr + inventory_offset, mask=valid, other=0).to(tl.int32)
                tl.store(unit_inventory_ptr + inventory_offset, 0, mask=valid)
            room = tl.maximum(SHED_CAPACITY - shed_total, 0)
            take = tl.minimum(carried, room)
            shed_offset = pair * ITEM_COUNT + item
            old_shed = tl.load(shed_ptr + shed_offset, mask=valid, other=0).to(tl.int32)
            tl.store(shed_ptr + shed_offset, old_shed + take, mask=valid)
            shed_total += take

        for unit in tl.static_range(0, MAX_UNITS):
            unit_offset = pair * MAX_UNITS + unit
            position_offset = unit_offset * 2
            farmer = unit == 0
            tl.store(position_ptr + position_offset, tl.where(farmer, 4, 0), mask=valid)
            tl.store(position_ptr + position_offset + 1, tl.where(farmer, 4, 0), mask=valid)
            tl.store(unit_active_ptr + unit_offset, farmer, mask=valid)
        tl.store(hands_count_ptr + pair, 0, mask=valid)
        tl.store(hires_today_ptr + pair, 0, mask=valid)

        if SHOP_UNLOCK:
            player = pair & 1
            env = pair // 2
            shop_mask = valid & (player == 0)
            current = tl.zeros((BLOCK,), tl.int32)
            for shop in tl.static_range(0, SHOP_COUNT):
                current += tl.load(shop_counts_ptr + env * SHOP_COUNT + shop, mask=shop_mask, other=0).to(tl.int32)
            seed = tl.load(episode_seed_ptr + env, mask=shop_mask, other=0)
            hashed = (seed * 48271 + day * 1000003 + 97531) % 2147483647
            choice = hashed % SHOP_COUNT
            offset = env * SHOP_COUNT + choice
            old_count = tl.load(shop_counts_ptr + offset, mask=shop_mask, other=0).to(tl.int32)
            tl.store(shop_counts_ptr + offset, old_count + 1, mask=shop_mask & (current < SHOP_COUNT))


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


def run_move_units(state, actions, unit_limit: int) -> None:
    """Advance all active moving units in one launch."""
    if not TRITON_AVAILABLE:  # pragma: no cover - guarded by caller
        raise RuntimeError("Triton is not available")
    block = 256
    unit_count = actions.unit_ops.shape[0] * 2 * unit_limit
    _move_units_kernel[(triton.cdiv(unit_count, block),)](
        actions.unit_ops,
        state.unit_active,
        state.positions,
        unit_count,
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


def run_dynamic_market(state, actions, engine, order: int, rounds: int) -> None:
    """Run one dynamically priced product order in exact player/unit order."""
    if not TRITON_AVAILABLE:  # pragma: no cover - guarded by caller
        raise RuntimeError("Triton is not available")
    block = 128
    _dynamic_market_kernel[(triton.cdiv(engine.num_envs, block),)](
        actions.market_ops,
        actions.market_args,
        actions.market_quantities,
        state.money,
        state.market_inventory,
        state.shed,
        engine.market_base,
        engine.market_t,
        engine.market_below_func,
        engine.market_below_target,
        engine.market_above_func,
        engine.market_above_target,
        engine.num_envs,
        ORDER=order,
        ROUNDS=rounds,
        MARKET_ORDERS=engine.config.max_market_orders,
        ITEM_COUNT=12,
        SHED_CAPACITY=engine.config.shed_capacity,
        BLOCK=block,
    )


def run_fixed_market(state, actions, engine, order: int, op_kind: int) -> None:
    """Run one fixed-price/structural market operation in a single launch."""
    if not TRITON_AVAILABLE:  # pragma: no cover - guarded by caller
        raise RuntimeError("Triton is not available")
    block = 256
    pair_count = engine.num_envs * 2
    _fixed_market_kernel[(triton.cdiv(pair_count, block),)](
        actions.market_ops,
        actions.market_args,
        actions.market_quantities,
        state.money,
        state.positions,
        state.unit_active,
        state.hands_count,
        state.hires_today,
        state.unit_inventory,
        state.unlocked_count,
        state.tile_type,
        state.shed,
        state.seeds,
        engine.hire_cost,
        engine.land_prices,
        engine.crop_seed_cost,
        engine.animal_cost,
        pair_count,
        ORDER=order,
        OP_KIND=op_kind,
        MARKET_ORDERS=engine.config.max_market_orders,
        MAX_QUANTITY=engine.config.max_market_quantity,
        MAX_UNITS=engine.max_units,
        ITEM_COUNT=12,
        CROP_COUNT=5,
        ANIMAL_COUNT=3,
        SHED_CAPACITY=engine.config.shed_capacity,
        BLOCK=block,
    )


def run_market_orders(state, actions, engine) -> None:
    """Process every market order in one device-resident sequential interpreter."""
    if not TRITON_AVAILABLE:  # pragma: no cover - guarded by caller
        raise RuntimeError("Triton is not available")
    block = 128
    _all_market_orders_kernel[(triton.cdiv(engine.num_envs, block),)](
        actions.market_ops,
        actions.market_args,
        actions.market_quantities,
        state.money,
        state.positions,
        state.unit_active,
        state.hands_count,
        state.hires_today,
        state.unit_inventory,
        state.unlocked_count,
        state.tile_type,
        state.shed,
        state.seeds,
        state.market_inventory,
        engine.hire_cost,
        engine.land_prices,
        engine.crop_seed_cost,
        engine.animal_cost,
        engine.market_base,
        engine.market_t,
        engine.market_below_func,
        engine.market_below_target,
        engine.market_above_func,
        engine.market_above_target,
        engine.num_envs,
        MARKET_ORDERS=engine.config.max_market_orders,
        MAX_QUANTITY=engine.config.max_market_quantity,
        MAX_UNITS=engine.max_units,
        ITEM_COUNT=12,
        CROP_COUNT=5,
        ANIMAL_COUNT=3,
        SHED_CAPACITY=engine.config.shed_capacity,
        BLOCK=block,
    )


def run_town_consume(state, engine, *, shop_active: bool, center_active: bool) -> None:
    """Apply town demand and refresh all product prices in one launch."""
    if not TRITON_AVAILABLE:  # pragma: no cover - guarded by caller
        raise RuntimeError("Triton is not available")
    block = 256
    item_count = 9
    _town_consume_kernel[(triton.cdiv(engine.num_envs * item_count, block),)](
        state.market_inventory,
        state.market_prices,
        state.shop_counts,
        engine.shop_demand,
        engine.market_base,
        engine.market_t,
        engine.market_below_func,
        engine.market_below_target,
        engine.market_above_func,
        engine.market_above_target,
        engine.num_envs,
        SHOP_ACTIVE=shop_active,
        CENTER_ACTIVE=center_active,
        ITEM_COUNT=item_count,
        SHOP_COUNT=8,
        BLOCK=block,
    )


def run_decay_plants(state, step: int) -> None:
    """Fuse the full-board decay scan and weed-field reset into one launch."""
    if not TRITON_AVAILABLE:  # pragma: no cover - guarded by caller
        raise RuntimeError("Triton is not available")
    block = 256
    cell_count = state.tile_type.numel()
    _decay_plants_kernel[(triton.cdiv(cell_count, block),)](
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
        cell_count,
        step,
        BLOCK=block,
    )


def run_end_of_day(state, engine, day: int) -> None:
    """Fuse daily board settlement and farm-hand reset into two launches."""
    if not TRITON_AVAILABLE:  # pragma: no cover - guarded by caller
        raise RuntimeError("Triton is not available")
    board_block = 256
    cell_count = state.tile_type.numel()
    next_day = day + 1
    _end_of_day_board_kernel[(triton.cdiv(cell_count, board_block),)](
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
        engine.episode_seeds,
        cell_count,
        day,
        next_day,
        engine.config.weed_spawn_chance,
        TURNS_PER_DAY=engine.config.turns_per_day,
        WEED_ENABLED=engine.config.weed_spawn_chance > 0,
        BLOCK=board_block,
    )

    reset_block = 128
    pair_count = engine.num_envs * 2
    _end_of_day_reset_kernel[(triton.cdiv(pair_count, reset_block),)](
        state.unit_inventory,
        state.shed,
        state.positions,
        state.unit_active,
        state.hands_count,
        state.hires_today,
        state.shop_counts,
        engine.episode_seeds,
        pair_count,
        day,
        SHED_CAPACITY=engine.config.shed_capacity,
        MAX_UNITS=engine.max_units,
        ITEM_COUNT=12,
        SHOP_COUNT=8,
        SHOP_UNLOCK=next_day > 0 and next_day % engine.config.town_shop_unlock_interval == 0,
        BLOCK=reset_block,
    )
