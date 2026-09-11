"""Pure fixed-shape JAX implementation of the Kaggriculture interpreter.

The hot path intentionally uses only integer/bool tensors.  Python's floating
market arithmetic and MT19937 event stream are represented by frozen exact
lookup tables generated from the immutable 1.32.7 reference implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .constants import (
    ANIMAL_COST,
    ANIMAL_FIRST_YIELD_DAY,
    ANIMAL_INTERVAL,
    ANIMAL_MAX_HELD,
    ANIMAL_PRODUCT,
    ANIMAL_STRUCTURE,
    BOARD_SIZE,
    CROP_FIRST_YIELD_DAY,
    CROP_INTERVAL,
    CROP_MAX_YIELD,
    CROP_MAX_YIELD_DAY,
    CROP_ONGOING,
    CROP_SEED_COST,
    DEFAULT_SPAWN,
    EPISODE_STEPS,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_WATERED,
    HIRE_COST,
    LAND_PRICES,
    MARKET_LUT_SIZE,
    MARKET_MIN_INVENTORY,
    MAX_HANDS,
    MAX_MARKET_ORDERS,
    MAX_SHOPS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PLAYERS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    NUM_TILES,
    SHED_ACCESS,
    SHED_CAPACITY,
    TURNS_PER_DAY,
    MarketOp,
    TileKind,
    UnitOp,
)
from .types import Action, Events, State, StaticTables


_CROP_SEED_COST = jnp.asarray(CROP_SEED_COST, dtype=jnp.int32)
_CROP_FIRST = jnp.asarray(CROP_FIRST_YIELD_DAY, dtype=jnp.int16)
_CROP_MAX_DAY = jnp.asarray(CROP_MAX_YIELD_DAY, dtype=jnp.int16)
_CROP_INTERVAL = jnp.asarray(CROP_INTERVAL, dtype=jnp.int16)
_CROP_MAX = jnp.asarray(CROP_MAX_YIELD, dtype=jnp.int16)
_CROP_ONGOING = jnp.asarray(CROP_ONGOING, dtype=jnp.bool_)
_ANIMAL_COST = jnp.asarray(ANIMAL_COST, dtype=jnp.int32)
_ANIMAL_STRUCTURE = jnp.asarray(ANIMAL_STRUCTURE, dtype=jnp.int8)
_ANIMAL_FIRST = jnp.asarray(ANIMAL_FIRST_YIELD_DAY, dtype=jnp.int16)
_ANIMAL_INTERVAL = jnp.asarray(ANIMAL_INTERVAL, dtype=jnp.int16)
_ANIMAL_MAX = jnp.asarray(ANIMAL_MAX_HELD, dtype=jnp.int16)
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int8)
_LAND_PRICES = jnp.asarray(LAND_PRICES, dtype=jnp.int32)
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int8)

# fib(0)..fib(31), using the official 1,1,2,... indexing.
_HIRE_COST = jnp.asarray(HIRE_COST, dtype=jnp.int32)

# Rows follow constants.SHOP_NAMES (sorted official names); columns are PRODUCTS.
_SHOP_DEMAND = jnp.asarray(
    (
        (1, 0, 0, 0, 0, 1, 0, 0, 0),  # BAKERY
        (1, 0, 0, 1, 0, 1, 0, 0, 0),  # BRUNCH_SPOT
        (1, 1, 1, 1, 0, 0, 0, 0, 0),  # FARMERS_MARKET
        (1, 0, 0, 1, 0, 0, 1, 0, 0),  # ICE_CREAM_SHOP
        (0, 2, 0, 0, 0, 0, 0, 0, 0),  # PET_CAFE
        (1, 0, 1, 0, 0, 0, 1, 0, 0),  # PIZZA_SHOP
        (0, 0, 0, 1, 0, 0, 1, 0, 0),  # SMOOTHIE_SHOP
        (0, 0, 0, 0, 0, 0, 0, 2, 0),  # YARN_STORE
    ),
    dtype=jnp.int32,
)


def _flag(flags: jax.Array, bit: int) -> jax.Array:
    return (flags & jnp.asarray(bit, dtype=jnp.uint8)) != 0


def _set_flag(flags: jax.Array, bit: int, value: jax.Array) -> jax.Array:
    mask = jnp.asarray(bit, dtype=jnp.uint8)
    return jnp.where(value, flags | mask, flags & jnp.bitwise_not(mask))


def _is_shed_adjacent(pos: jax.Array) -> jax.Array:
    return jnp.any(jnp.all(_SHED_ACCESS == pos[None, :], axis=1))


def _clear_tile(state: State, player: jax.Array, y: jax.Array, x: jax.Array, kind: int) -> State:
    state = state._replace(
        tile_kind=state.tile_kind.at[player, y, x].set(jnp.int8(kind)),
        tile_crop=state.tile_crop.at[player, y, x].set(jnp.int8(-1)),
        tile_animal=state.tile_animal.at[player, y, x].set(jnp.int8(-1)),
        tile_origin_day=state.tile_origin_day.at[player, y, x].set(jnp.int8(-1)),
        tile_yield=state.tile_yield.at[player, y, x].set(jnp.int16(0)),
        tile_neglect=state.tile_neglect.at[player, y, x].set(jnp.int8(0)),
        tile_max_lifespan=state.tile_max_lifespan.at[player, y, x].set(jnp.int16(-1)),
        tile_fertilized_until=state.tile_fertilized_until.at[player, y, x].set(jnp.int8(-1)),
        tile_pending_care=state.tile_pending_care.at[player, y, x].set(jnp.int8(0)),
        tile_flags=state.tile_flags.at[player, y, x].set(jnp.uint8(0)),
    )
    return state


def _inventory_add(
    state: State,
    player: jax.Array,
    unit: jax.Array,
    item: jax.Array,
    amount: jax.Array,
) -> State:
    safe_item = jnp.clip(item, 0, NUM_SHED_ITEMS - 1)
    valid = (item >= 0) & (item < NUM_SHED_ITEMS) & (amount > 0)

    def add(s: State) -> State:
        old = s.unit_inventory[player, unit, safe_item]
        is_new = old == 0
        order = jnp.where(
            is_new,
            s.unit_inventory_next_order[player, unit],
            s.unit_inventory_order[player, unit, safe_item],
        )
        next_order = s.unit_inventory_next_order[player, unit] + is_new.astype(jnp.int8)
        return s._replace(
            unit_inventory=s.unit_inventory.at[player, unit, safe_item].set(
                old + amount.astype(jnp.int16)
            ),
            unit_inventory_order=s.unit_inventory_order.at[
                player, unit, safe_item
            ].set(order),
            unit_inventory_next_order=s.unit_inventory_next_order.at[
                player, unit
            ].set(next_order),
        )

    return lax.cond(valid, add, lambda s: s, state)


def _inventory_take(
    state: State,
    player: jax.Array,
    unit: jax.Array,
    item: jax.Array,
    amount: jax.Array,
) -> tuple[State, jax.Array]:
    safe_item = jnp.clip(item, 0, NUM_SHED_ITEMS - 1)
    valid_item = (item >= 0) & (item < NUM_SHED_ITEMS) & (amount > 0)
    old = state.unit_inventory[player, unit, safe_item].astype(jnp.int32)
    ok = valid_item & (old >= amount)
    new = old - jnp.where(ok, amount, 0)
    order = jnp.where(
        ok & (new == 0),
        jnp.int8(-1),
        state.unit_inventory_order[player, unit, safe_item],
    )
    state = state._replace(
        unit_inventory=state.unit_inventory.at[player, unit, safe_item].set(
            new.astype(jnp.int16)
        ),
        unit_inventory_order=state.unit_inventory_order.at[
            player, unit, safe_item
        ].set(order),
    )
    return state, ok


def _drop_inventory(state: State, player: jax.Array, unit: jax.Array) -> State:
    counts = state.unit_inventory[player, unit]
    orders = state.unit_inventory_order[player, unit]
    sort_key = jnp.where(orders >= 0, orders.astype(jnp.int16), jnp.int16(32767))
    item_order = jnp.argsort(sort_key, stable=True)

    def body(i: int, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        shed, room = carry
        item = item_order[i]
        amount = counts[item].astype(jnp.int32)
        take = jnp.minimum(jnp.maximum(amount, 0), room)
        shed = shed.at[item].add(take.astype(jnp.int16))
        return shed, room - take

    shed = state.shed[player]
    room = jnp.maximum(0, SHED_CAPACITY - jnp.sum(shed.astype(jnp.int32)))
    shed, _ = lax.fori_loop(0, NUM_SHED_ITEMS, body, (shed, room))
    return state._replace(
        shed=state.shed.at[player].set(shed),
        unit_inventory=state.unit_inventory.at[player, unit].set(
            jnp.zeros((NUM_SHED_ITEMS,), dtype=jnp.int16)
        ),
        unit_inventory_order=state.unit_inventory_order.at[player, unit].set(
            jnp.full((NUM_SHED_ITEMS,), -1, dtype=jnp.int8)
        ),
    )


def _move(state: State, player: jax.Array, unit: jax.Array, dx: int, dy: int) -> State:
    pos = state.unit_pos[player, unit].astype(jnp.int16)
    target = pos + jnp.asarray((dx, dy), dtype=jnp.int16)
    valid = jnp.all((target >= 0) & (target < BOARD_SIZE))
    new_pos = jnp.where(valid, target, pos).astype(jnp.int8)
    return state._replace(unit_pos=state.unit_pos.at[player, unit].set(new_pos))


def _apply_unit_action(
    state: State,
    op: jax.Array,
    item: jax.Array,
    amount: jax.Array,
    player: jax.Array,
    unit: jax.Array,
    day: jax.Array,
) -> State:
    pos = state.unit_pos[player, unit]
    x = pos[0].astype(jnp.int32)
    y = pos[1].astype(jnp.int32)
    safe_item = jnp.clip(item, 0, NUM_SHED_ITEMS - 1)

    def pass_op(s: State) -> State:
        return s

    def north(s: State) -> State:
        return _move(s, player, unit, 0, -1)

    def south(s: State) -> State:
        return _move(s, player, unit, 0, 1)

    def east(s: State) -> State:
        return _move(s, player, unit, 1, 0)

    def west(s: State) -> State:
        return _move(s, player, unit, -1, 0)

    def drop(s: State) -> State:
        return lax.cond(
            _is_shed_adjacent(pos),
            lambda z: _drop_inventory(z, player, unit),
            lambda z: z,
            s,
        )

    def pickup(s: State) -> State:
        valid = (
            _is_shed_adjacent(pos)
            & (item >= 0)
            & (item < NUM_SHED_ITEMS)
            & (amount > 0)
        )
        available = s.shed[player, safe_item].astype(jnp.int32)
        take = jnp.where(valid, jnp.minimum(amount, available), 0)
        s = s._replace(
            shed=s.shed.at[player, safe_item].add(-take.astype(jnp.int16))
        )
        return _inventory_add(s, player, unit, safe_item, take)

    def place(s: State) -> State:
        kind = s.tile_kind[player, y, x]
        animal = item - NUM_PRODUCTS
        safe_animal = jnp.clip(animal, 0, NUM_ANIMALS - 1)
        animal_match = (
            (animal >= 0)
            & (animal < NUM_ANIMALS)
            & (kind == _ANIMAL_STRUCTURE[safe_animal])
            & (s.tile_animal[player, y, x] < 0)
        )

        def place_animal(z: State) -> State:
            z, ok = _inventory_take(z, player, unit, item, jnp.int32(1))

            def commit(q: State) -> State:
                q = q._replace(
                    tile_animal=q.tile_animal.at[player, y, x].set(
                        safe_animal.astype(jnp.int8)
                    ),
                    tile_origin_day=q.tile_origin_day.at[player, y, x].set(
                        day.astype(jnp.int8)
                    ),
                    tile_yield=q.tile_yield.at[player, y, x].set(jnp.int16(0)),
                    tile_neglect=q.tile_neglect.at[player, y, x].set(jnp.int8(0)),
                    tile_pending_care=q.tile_pending_care.at[player, y, x].set(
                        jnp.int8(0)
                    ),
                    tile_flags=q.tile_flags.at[player, y, x].set(jnp.uint8(0)),
                )
                return q

            return lax.cond(ok, commit, lambda q: q, z)

        def place_shed(z: State) -> State:
            valid = (
                _is_shed_adjacent(pos)
                & (item >= 0)
                & (item < NUM_SHED_ITEMS)
                & (amount > 0)
            )
            available = z.unit_inventory[player, unit, safe_item].astype(jnp.int32)
            room = jnp.maximum(
                0, SHED_CAPACITY - jnp.sum(z.shed[player].astype(jnp.int32))
            )
            take = jnp.where(valid, jnp.minimum(jnp.minimum(amount, available), room), 0)
            new_count = available - take
            new_order = jnp.where(
                (take > 0) & (new_count == 0),
                jnp.int8(-1),
                z.unit_inventory_order[player, unit, safe_item],
            )
            return z._replace(
                unit_inventory=z.unit_inventory.at[player, unit, safe_item].set(
                    new_count.astype(jnp.int16)
                ),
                unit_inventory_order=z.unit_inventory_order.at[
                    player, unit, safe_item
                ].set(new_order),
                shed=z.shed.at[player, safe_item].add(take.astype(jnp.int16)),
            )

        return lax.cond(animal_match, place_animal, place_shed, s)

    def plant(s: State) -> State:
        crop = item
        safe_crop = jnp.clip(crop, 0, NUM_CROPS - 1)
        valid = (
            (s.tile_kind[player, y, x] == TileKind.EMPTY)
            & (crop >= 0)
            & (crop < NUM_CROPS)
            & (s.seeds[player, safe_crop] > 0)
        )

        def commit(z: State) -> State:
            ongoing = _CROP_ONGOING[safe_crop]
            max_life = jnp.where(
                ongoing,
                -1,
                (day.astype(jnp.int16) + _CROP_MAX_DAY[safe_crop] + 1)
                * TURNS_PER_DAY,
            )
            return z._replace(
                seeds=z.seeds.at[player, safe_crop].add(jnp.int16(-1)),
                tile_kind=z.tile_kind.at[player, y, x].set(jnp.int8(TileKind.PLANT)),
                tile_crop=z.tile_crop.at[player, y, x].set(safe_crop.astype(jnp.int8)),
                tile_animal=z.tile_animal.at[player, y, x].set(jnp.int8(-1)),
                tile_origin_day=z.tile_origin_day.at[player, y, x].set(day.astype(jnp.int8)),
                tile_yield=z.tile_yield.at[player, y, x].set(
                    jnp.where(ongoing, 0, 1).astype(jnp.int16)
                ),
                tile_neglect=z.tile_neglect.at[player, y, x].set(jnp.int8(1)),
                tile_max_lifespan=z.tile_max_lifespan.at[player, y, x].set(
                    max_life.astype(jnp.int16)
                ),
                tile_fertilized_until=z.tile_fertilized_until.at[player, y, x].set(
                    jnp.int8(-1)
                ),
                tile_pending_care=z.tile_pending_care.at[player, y, x].set(jnp.int8(0)),
                tile_flags=z.tile_flags.at[player, y, x].set(jnp.uint8(0)),
            )

        return lax.cond(valid, commit, lambda z: z, s)

    def water(s: State) -> State:
        flags = s.tile_flags[player, y, x]
        crop = s.tile_crop[player, y, x]
        safe_crop = jnp.clip(crop, 0, NUM_CROPS - 1)
        valid = (
            (s.tile_kind[player, y, x] == TileKind.PLANT)
            & (~_flag(flags, FLAG_WATERED))
        )

        def commit(z: State) -> State:
            age = day.astype(jnp.int16) - z.tile_origin_day[player, y, x].astype(
                jnp.int16
            )
            window_start = (_CROP_MAX_DAY[safe_crop] + 1) // 2
            in_window = (
                (~_CROP_ONGOING[safe_crop])
                & (age >= window_start)
                & (age <= _CROP_MAX_DAY[safe_crop])
            )
            bonus = jnp.where(
                z.tile_fertilized_until[player, y, x] >= day, 2, 1
            ).astype(jnp.int16)
            new_yield = jnp.where(
                in_window,
                jnp.minimum(
                    _CROP_MAX[safe_crop], z.tile_yield[player, y, x] + bonus
                ),
                z.tile_yield[player, y, x],
            )
            return z._replace(
                tile_flags=z.tile_flags.at[player, y, x].set(
                    _set_flag(flags, FLAG_WATERED, jnp.asarray(True))
                ),
                tile_yield=z.tile_yield.at[player, y, x].set(new_yield),
            )

        return lax.cond(valid, commit, lambda z: z, s)

    def harvest(s: State) -> State:
        kind = s.tile_kind[player, y, x]
        units = s.tile_yield[player, y, x].astype(jnp.int32)
        crop = s.tile_crop[player, y, x]
        safe_crop = jnp.clip(crop, 0, NUM_CROPS - 1)
        is_plant = kind == TileKind.PLANT
        mature = (
            day.astype(jnp.int16)
            - s.tile_origin_day[player, y, x].astype(jnp.int16)
            >= _CROP_FIRST[safe_crop]
        )
        animal = s.tile_animal[player, y, x]
        safe_animal = jnp.clip(animal, 0, NUM_ANIMALS - 1)
        is_animal = animal >= 0
        valid_plant = (units > 0) & is_plant & mature
        valid_animal = (units > 0) & is_animal

        def plant_commit(z: State) -> State:
            z = _inventory_add(z, player, unit, safe_crop, units)
            z = z._replace(
                tile_yield=z.tile_yield.at[player, y, x].set(jnp.int16(0))
            )
            return lax.cond(
                ~_CROP_ONGOING[safe_crop],
                lambda q: _clear_tile(q, player, y, x, TileKind.EMPTY),
                lambda q: q,
                z,
            )

        def maybe_animal(z: State) -> State:
            def animal_commit(q: State) -> State:
                q = _inventory_add(
                    q, player, unit, _ANIMAL_PRODUCT[safe_animal], units
                )
                return q._replace(
                    tile_yield=q.tile_yield.at[player, y, x].set(jnp.int16(0))
                )

            return lax.cond(valid_animal, animal_commit, lambda q: q, z)

        return lax.cond(valid_plant, plant_commit, maybe_animal, s)

    def fertilize(s: State) -> State:
        valid = s.tile_kind[player, y, x] == TileKind.PLANT

        def attempt(z: State) -> State:
            z, ok = _inventory_take(z, player, unit, jnp.int8(8), jnp.int32(1))
            until = jnp.maximum(
                z.tile_fertilized_until[player, y, x],
                (day + 2).astype(jnp.int8),
            )
            return z._replace(
                tile_fertilized_until=z.tile_fertilized_until.at[
                    player, y, x
                ].set(jnp.where(ok, until, z.tile_fertilized_until[player, y, x]))
            )

        return lax.cond(valid, attempt, lambda z: z, s)

    def dig(s: State) -> State:
        kind = s.tile_kind[player, y, x]
        valid = (
            (kind != TileKind.EMPTY)
            & (kind != TileKind.LOCKED)
            & (s.tile_animal[player, y, x] < 0)
        )
        return lax.cond(
            valid,
            lambda z: _clear_tile(z, player, y, x, TileKind.EMPTY),
            lambda z: z,
            s,
        )

    def build_coop(s: State) -> State:
        valid = s.tile_kind[player, y, x] == TileKind.EMPTY
        return s._replace(
            tile_kind=s.tile_kind.at[player, y, x].set(
                jnp.where(valid, TileKind.COOP, s.tile_kind[player, y, x]).astype(
                    jnp.int8
                )
            )
        )

    def build_pasture(s: State) -> State:
        valid = s.tile_kind[player, y, x] == TileKind.EMPTY
        return s._replace(
            tile_kind=s.tile_kind.at[player, y, x].set(
                jnp.where(
                    valid, TileKind.PASTURE, s.tile_kind[player, y, x]
                ).astype(jnp.int8)
            )
        )

    def feed(s: State) -> State:
        flags = s.tile_flags[player, y, x]
        valid = (s.tile_animal[player, y, x] >= 0) & (~_flag(flags, FLAG_FED))

        def attempt(z: State) -> State:
            z, ok = _inventory_take(z, player, unit, jnp.int8(0), jnp.int32(1))
            return z._replace(
                tile_flags=z.tile_flags.at[player, y, x].set(
                    _set_flag(z.tile_flags[player, y, x], FLAG_FED, ok)
                )
            )

        return lax.cond(valid, attempt, lambda z: z, s)

    def collect_fertilizer(s: State) -> State:
        flags = s.tile_flags[player, y, x]
        valid = (s.tile_animal[player, y, x] >= 0) & _flag(
            flags, FLAG_FERTILIZER_AVAILABLE
        )

        def commit(z: State) -> State:
            z = z._replace(
                tile_flags=z.tile_flags.at[player, y, x].set(
                    _set_flag(flags, FLAG_FERTILIZER_AVAILABLE, jnp.asarray(False))
                )
            )
            return _inventory_add(z, player, unit, jnp.int8(8), jnp.int32(1))

        return lax.cond(valid, commit, lambda z: z, s)

    def care(s: State) -> State:
        flags = s.tile_flags[player, y, x]
        valid = (s.tile_animal[player, y, x] >= 0) & (~_flag(flags, FLAG_CARED))
        return s._replace(
            tile_flags=s.tile_flags.at[player, y, x].set(
                jnp.where(
                    valid,
                    _set_flag(flags, FLAG_CARED, jnp.asarray(True)),
                    flags,
                ).astype(jnp.uint8)
            )
        )

    branches = (
        pass_op,
        north,
        south,
        east,
        west,
        drop,
        pickup,
        place,
        plant,
        water,
        harvest,
        fertilize,
        dig,
        build_coop,
        build_pasture,
        feed,
        collect_fertilizer,
        care,
    )
    valid_op = (op >= 0) & (op < len(branches))
    safe_op = jnp.clip(op, 0, len(branches) - 1)
    return lax.cond(
        valid_op,
        lambda s: lax.switch(safe_op, branches, s),
        lambda s: s,
        state,
    )


def _process_units(state: State, action: Action, day: jax.Array) -> State:
    crop_ids = jnp.arange(NUM_CROPS, dtype=jnp.int8)
    provided = jnp.arange(MAX_UNITS)[None, :] < action.unit_count[:, None]
    plant = action.unit_op == UnitOp.PLANT
    demand = jnp.sum(
        provided[:, :, None]
        & plant[:, :, None]
        & (action.unit_item[:, :, None] == crop_ids[None, None, :]),
        axis=1,
    )
    blocked = demand > state.seeds

    def player_body(player: int, s: State) -> State:
        max_units = jnp.clip(action.unit_count[player].astype(jnp.int32), 0, MAX_UNITS)

        def unit_cond(carry: tuple[jax.Array, State]) -> jax.Array:
            unit, _ = carry
            return unit < max_units

        def unit_body(carry: tuple[jax.Array, State]) -> tuple[jax.Array, State]:
            unit, z = carry
            within = unit < action.unit_count[player]
            active = z.unit_active[player, unit]
            op = action.unit_op[player, unit]
            item = action.unit_item[player, unit]
            safe_crop = jnp.clip(item, 0, NUM_CROPS - 1)
            plant_blocked = (
                (op == UnitOp.PLANT)
                & (item >= 0)
                & (item < NUM_CROPS)
                & blocked[player, safe_crop]
            )
            effective_op = jnp.where(plant_blocked, UnitOp.PASS, op)
            z = lax.cond(
                within & active,
                lambda q: _apply_unit_action(
                    q,
                    effective_op,
                    item,
                    action.unit_amount[player, unit],
                    jnp.asarray(player),
                    jnp.asarray(unit),
                    day,
                ),
                lambda q: q,
                z,
            )
            return unit + 1, z

        _, s = lax.while_loop(
            unit_cond,
            unit_body,
            (jnp.asarray(0, dtype=jnp.int32), s),
        )
        return s

    return lax.fori_loop(0, NUM_PLAYERS, player_body, state)


def _lookup_price(
    state: State, tables: StaticTables, item: jax.Array, inventory: jax.Array
) -> tuple[jax.Array, State]:
    safe_item = jnp.clip(item, 0, NUM_PRODUCTS - 1)
    index = inventory.astype(jnp.int32) - MARKET_MIN_INVENTORY
    oob = (index < 0) | (index >= MARKET_LUT_SIZE)
    safe_index = jnp.clip(index, 0, MARKET_LUT_SIZE - 1)
    price = tables.market_price[safe_item, safe_index].astype(jnp.int32)
    return price, state._replace(price_lut_oob=state.price_lut_oob + oob.astype(jnp.int32))


def _refresh_prices(state: State, tables: StaticTables) -> State:
    indices = state.market_inventory - MARKET_MIN_INVENTORY
    oob = (indices < 0) | (indices >= MARKET_LUT_SIZE)
    safe = jnp.clip(indices, 0, MARKET_LUT_SIZE - 1)
    prices = tables.market_price[jnp.arange(NUM_PRODUCTS), safe]
    return state._replace(
        market_price=prices.astype(jnp.int32),
        price_lut_oob=state.price_lut_oob + jnp.sum(oob.astype(jnp.int32)),
    )


def _spawn_hand(state: State, player: jax.Array, unit_index: jax.Array) -> State:
    positions = state.unit_pos[player]
    active = state.unit_active[player]
    occupancy = jnp.sum(
        active[:, None]
        & jnp.all(positions[:, None, :] == _SHED_ACCESS[None, :, :], axis=2),
        axis=0,
    )
    spawn = _SHED_ACCESS[jnp.argmin(occupancy)]
    return state._replace(
        unit_pos=state.unit_pos.at[player, unit_index].set(spawn),
        unit_active=state.unit_active.at[player, unit_index].set(True),
        unit_inventory=state.unit_inventory.at[player, unit_index].set(
            jnp.zeros((NUM_SHED_ITEMS,), dtype=jnp.int16)
        ),
        unit_inventory_order=state.unit_inventory_order.at[player, unit_index].set(
            jnp.full((NUM_SHED_ITEMS,), -1, dtype=jnp.int8)
        ),
        unit_inventory_next_order=state.unit_inventory_next_order.at[
            player, unit_index
        ].set(jnp.int8(0)),
    )


def _hire(state: State, player: jax.Array) -> State:
    n = state.hires_today[player].astype(jnp.int32)
    has_slot = n < MAX_HANDS
    safe_n = jnp.clip(n, 0, MAX_HANDS - 1)
    cost = _HIRE_COST[safe_n]
    can_pay = state.money[player] >= cost
    success = has_slot & can_pay
    cap_hit = (~has_slot) & can_pay
    unit_index = jnp.clip(n + 1, 1, MAX_UNITS - 1)

    def commit(s: State) -> State:
        s = s._replace(
            money=s.money.at[player].add(-cost),
            hires_today=s.hires_today.at[player].add(jnp.int8(1)),
        )
        return _spawn_hand(s, player, unit_index)

    state = lax.cond(success, commit, lambda s: s, state)
    return state._replace(
        hand_cap_hits=state.hand_cap_hits.at[player].add(cap_hit.astype(jnp.int32))
    )


def _buy_land(state: State, player: jax.Array) -> State:
    extra = state.unlocked_count[player].astype(jnp.int32) - 1
    safe = jnp.clip(extra, 0, 2)
    cost = _LAND_PRICES[safe]
    valid = (extra >= 0) & (extra < 3) & (state.money[player] >= cost)

    def commit(s: State) -> State:
        y, x = jnp.indices((BOARD_SIZE, BOARD_SIZE))
        quadrant = jnp.where(
            extra == 0,
            (x >= BOARD_SIZE // 2) & (y < BOARD_SIZE // 2),
            jnp.where(
                extra == 1,
                (x < BOARD_SIZE // 2) & (y >= BOARD_SIZE // 2),
                (x >= BOARD_SIZE // 2) & (y >= BOARD_SIZE // 2),
            ),
        )
        current = s.tile_kind[player]
        unlocked = quadrant & (current == TileKind.LOCKED)
        return s._replace(
            money=s.money.at[player].add(-cost),
            unlocked_count=s.unlocked_count.at[player].add(jnp.int8(1)),
            tile_kind=s.tile_kind.at[player].set(
                jnp.where(unlocked, TileKind.EMPTY, current).astype(jnp.int8)
            ),
        )

    return lax.cond(valid, commit, lambda s: s, state)


def _commit_market_unit(
    state: State,
    player: jax.Array,
    op: jax.Array,
    item: jax.Array,
    price: jax.Array,
) -> tuple[State, jax.Array]:
    safe_product = jnp.clip(item, 0, NUM_PRODUCTS - 1)
    animal = item - NUM_PRODUCTS
    safe_animal = jnp.clip(animal, 0, NUM_ANIMALS - 1)
    shed_total = jnp.sum(state.shed[player].astype(jnp.int32))

    def sell(s: State) -> tuple[State, jax.Array]:
        valid_item = (item >= 0) & (item < NUM_PRODUCTS)
        ok = valid_item & (s.shed[player, safe_product] > 0)
        supply_delta = jnp.where(ok & (price > 1), 1, 0).astype(jnp.int32)
        return (
            s._replace(
                shed=s.shed.at[player, safe_product].add(-ok.astype(jnp.int16)),
                money=s.money.at[player].add(jnp.where(ok, price, 0)),
                market_inventory=s.market_inventory.at[safe_product].add(supply_delta),
            ),
            ok,
        )

    def buy_product(s: State) -> tuple[State, jax.Array]:
        valid_item = (item == 0) | (item == 8)
        ok = valid_item & (s.money[player] >= price) & (shed_total < SHED_CAPACITY)
        return (
            s._replace(
                money=s.money.at[player].add(-jnp.where(ok, price, 0)),
                shed=s.shed.at[player, safe_product].add(ok.astype(jnp.int16)),
                market_inventory=s.market_inventory.at[safe_product].add(
                    -ok.astype(jnp.int32)
                ),
            ),
            ok,
        )

    def buy_seed(s: State) -> tuple[State, jax.Array]:
        safe_crop = jnp.clip(item, 0, NUM_CROPS - 1)
        valid_item = (item >= 0) & (item < NUM_CROPS)
        ok = valid_item & (s.money[player] >= price)
        return (
            s._replace(
                money=s.money.at[player].add(-jnp.where(ok, price, 0)),
                seeds=s.seeds.at[player, safe_crop].add(ok.astype(jnp.int16)),
            ),
            ok,
        )

    def buy_animal(s: State) -> tuple[State, jax.Array]:
        valid_item = (animal >= 0) & (animal < NUM_ANIMALS)
        ok = valid_item & (s.money[player] >= price) & (shed_total < SHED_CAPACITY)
        shed_item = NUM_PRODUCTS + safe_animal
        return (
            s._replace(
                money=s.money.at[player].add(-jnp.where(ok, price, 0)),
                shed=s.shed.at[player, shed_item].add(ok.astype(jnp.int16)),
            ),
            ok,
        )

    branches = (sell, buy_product, buy_seed, buy_animal)
    branch_index = jnp.where(
        op == MarketOp.SELL,
        0,
        jnp.where(
            op == MarketOp.BUY_PRODUCT,
            1,
            jnp.where(op == MarketOp.BUY_SEED, 2, 3),
        ),
    )
    valid_op = (
        (op == MarketOp.SELL)
        | (op == MarketOp.BUY_PRODUCT)
        | (op == MarketOp.BUY_SEED)
        | (op == MarketOp.BUY_ANIMAL)
    )
    return lax.cond(
        valid_op,
        lambda s: lax.switch(branch_index, branches, s),
        lambda s: (s, jnp.asarray(False)),
        state,
    )


def _process_market(state: State, action: Action, tables: StaticTables) -> State:
    max_len = jnp.maximum(action.market_count[0], action.market_count[1])

    def order_cond(carry: tuple[jax.Array, State]) -> jax.Array:
        order_index, _ = carry
        return order_index < max_len

    def order_body(carry: tuple[jax.Array, State]) -> tuple[jax.Array, State]:
        order_index, z = carry
        within = order_index < action.market_count
        ops = jnp.where(within, action.market_op[:, order_index], MarketOp.NONE)
        items = action.market_item[:, order_index]
        remaining = action.market_amount[:, order_index]

        def atomic_body(
            player: int, q: tuple[State, jax.Array]
        ) -> tuple[State, jax.Array]:
            current, active = q
            op = ops[player]
            current = lax.cond(
                op == MarketOp.HIRE,
                lambda a: _hire(a, jnp.asarray(player)),
                lambda a: a,
                current,
            )
            current = lax.cond(
                op == MarketOp.BUY_LAND,
                lambda a: _buy_land(a, jnp.asarray(player)),
                lambda a: a,
                current,
            )
            atomic = (op == MarketOp.HIRE) | (op == MarketOp.BUY_LAND)
            active = active.at[player].set(
                (~atomic) & (op != MarketOp.NONE) & (remaining[player] > 0)
            )
            return current, active

        z, active = lax.fori_loop(
            0,
            NUM_PLAYERS,
            atomic_body,
            (z, jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)),
        )

        def cond(carry: tuple[State, jax.Array, jax.Array, jax.Array]) -> jax.Array:
            _, rem, act, iteration = carry
            return jnp.any(act & (rem > 0)) & (iteration < 99_999)

        def body(
            carry: tuple[State, jax.Array, jax.Array, jax.Array]
        ) -> tuple[State, jax.Array, jax.Array, jax.Array]:
            current, rem, act, iteration = carry
            prices = jnp.zeros((NUM_PLAYERS,), dtype=jnp.int32)
            quoted = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)

            def quote_body(
                player: int,
                quote_carry: tuple[State, jax.Array, jax.Array],
            ) -> tuple[State, jax.Array, jax.Array]:
                qstate, qprices, qvalid = quote_carry
                op = ops[player]
                item = items[player]
                safe_product = jnp.clip(item, 0, NUM_PRODUCTS - 1)
                valid_sell = (op == MarketOp.SELL) & (item >= 0) & (
                    item < NUM_PRODUCTS
                )
                valid_product = (op == MarketOp.BUY_PRODUCT) & (
                    (item == 0) | (item == 8)
                )
                valid_seed = (op == MarketOp.BUY_SEED) & (item >= 0) & (
                    item < NUM_CROPS
                )
                animal = item - NUM_PRODUCTS
                valid_animal = (op == MarketOp.BUY_ANIMAL) & (animal >= 0) & (
                    animal < NUM_ANIMALS
                )
                dynamic_price = valid_sell | valid_product
                inventory = qstate.market_inventory[safe_product] - valid_product.astype(
                    jnp.int32
                )

                def dynamic(args: tuple[State, jax.Array]) -> tuple[jax.Array, State]:
                    st, inv = args
                    return _lookup_price(st, tables, item, inv)

                dynamic_value, qstate = lax.cond(
                    dynamic_price,
                    dynamic,
                    lambda args: (jnp.int32(0), args[0]),
                    (qstate, inventory),
                )
                safe_crop = jnp.clip(item, 0, NUM_CROPS - 1)
                safe_animal = jnp.clip(animal, 0, NUM_ANIMALS - 1)
                fixed = jnp.where(
                    valid_seed,
                    _CROP_SEED_COST[safe_crop],
                    jnp.where(valid_animal, _ANIMAL_COST[safe_animal], 0),
                )
                is_valid = act[player] & (
                    valid_sell | valid_product | valid_seed | valid_animal
                )
                qprices = qprices.at[player].set(
                    jnp.where(dynamic_price, dynamic_value, fixed)
                )
                qvalid = qvalid.at[player].set(is_valid)
                return qstate, qprices, qvalid

            current, prices, quoted = lax.fori_loop(
                0, NUM_PLAYERS, quote_body, (current, prices, quoted)
            )
            act = act & quoted

            def commit_body(
                player: int,
                commit_carry: tuple[State, jax.Array, jax.Array],
            ) -> tuple[State, jax.Array, jax.Array]:
                qstate, qrem, qactive = commit_carry

                def attempt(st: State) -> tuple[State, jax.Array]:
                    return _commit_market_unit(
                        st,
                        jnp.asarray(player),
                        ops[player],
                        items[player],
                        prices[player],
                    )

                qstate, ok = lax.cond(
                    qactive[player],
                    attempt,
                    lambda st: (st, jnp.asarray(False)),
                    qstate,
                )
                qrem = qrem.at[player].add(-ok.astype(jnp.int32))
                qactive = qactive.at[player].set(qactive[player] & ok)
                return qstate, qrem, qactive

            current, rem, act = lax.fori_loop(
                0, NUM_PLAYERS, commit_body, (current, rem, act)
            )
            return current, rem, act, iteration + 1

        z, remaining_after, active_after, iterations = lax.while_loop(
            cond,
            body,
            (z, remaining, active, jnp.asarray(0, dtype=jnp.int32)),
        )
        del remaining_after
        hit_cap = (iterations >= 99_999) & jnp.any(active_after)
        z = z._replace(
            market_loop_cap_hits=z.market_loop_cap_hits + hit_cap.astype(jnp.int32)
        )
        z = _refresh_prices(z, tables)
        return order_index + 1, z

    _, state = lax.while_loop(
        order_cond,
        order_body,
        (jnp.asarray(0, dtype=jnp.int32), state),
    )
    return state


def _refresh_prices_batch(states: State, tables: StaticTables) -> State:
    indices = states.market_inventory - MARKET_MIN_INVENTORY
    oob = (indices < 0) | (indices >= MARKET_LUT_SIZE)
    safe = jnp.clip(indices, 0, MARKET_LUT_SIZE - 1)
    prices = tables.market_price[jnp.arange(NUM_PRODUCTS)[None, :], safe]
    return states._replace(
        market_price=prices.astype(jnp.int32),
        price_lut_oob=states.price_lut_oob + jnp.sum(oob, axis=1).astype(jnp.int32),
    )


def _batch_hire_player(
    states: State, player: int, requested: jax.Array
) -> State:
    batch_size = states.money.shape[0]
    batch = jnp.arange(batch_size)
    n = states.hires_today[:, player].astype(jnp.int32)
    safe_n = jnp.clip(n, 0, MAX_HANDS - 1)
    cost = _HIRE_COST[safe_n]
    has_slot = n < MAX_HANDS
    can_pay = states.money[:, player] >= cost
    success = requested & has_slot & can_pay
    cap_hit = requested & (~has_slot) & can_pay
    unit_index = jnp.clip(n + 1, 1, MAX_UNITS - 1)

    positions = states.unit_pos[:, player]
    active = states.unit_active[:, player]
    occupancy = jnp.sum(
        active[:, :, None]
        & jnp.all(
            positions[:, :, None, :] == _SHED_ACCESS[None, None, :, :], axis=3
        ),
        axis=1,
    )
    spawn = _SHED_ACCESS[jnp.argmin(occupancy, axis=1)]
    old_pos = states.unit_pos[batch, player, unit_index]
    old_active = states.unit_active[batch, player, unit_index]
    old_inventory = states.unit_inventory[batch, player, unit_index]
    old_order = states.unit_inventory_order[batch, player, unit_index]
    old_next_order = states.unit_inventory_next_order[batch, player, unit_index]
    return states._replace(
        money=states.money.at[:, player].add(-jnp.where(success, cost, 0)),
        hires_today=states.hires_today.at[:, player].add(success.astype(jnp.int8)),
        unit_pos=states.unit_pos.at[batch, player, unit_index].set(
            jnp.where(success[:, None], spawn, old_pos).astype(jnp.int8)
        ),
        unit_active=states.unit_active.at[batch, player, unit_index].set(
            jnp.where(success, True, old_active)
        ),
        unit_inventory=states.unit_inventory.at[batch, player, unit_index].set(
            jnp.where(
                success[:, None], jnp.zeros_like(old_inventory), old_inventory
            )
        ),
        unit_inventory_order=states.unit_inventory_order.at[
            batch, player, unit_index
        ].set(
            jnp.where(success[:, None], jnp.full_like(old_order, -1), old_order)
        ),
        unit_inventory_next_order=states.unit_inventory_next_order.at[
            batch, player, unit_index
        ].set(jnp.where(success, 0, old_next_order).astype(jnp.int8)),
        hand_cap_hits=states.hand_cap_hits.at[:, player].add(
            cap_hit.astype(jnp.int32)
        ),
    )


def _batch_buy_land_player(
    states: State, player: int, requested: jax.Array
) -> State:
    extra = states.unlocked_count[:, player].astype(jnp.int32) - 1
    safe_extra = jnp.clip(extra, 0, 2)
    cost = _LAND_PRICES[safe_extra]
    success = requested & (extra >= 0) & (extra < 3) & (
        states.money[:, player] >= cost
    )
    y, x = jnp.indices((BOARD_SIZE, BOARD_SIZE))
    quadrant = jnp.where(
        (extra == 0)[:, None, None],
        (x >= BOARD_SIZE // 2) & (y < BOARD_SIZE // 2),
        jnp.where(
            (extra == 1)[:, None, None],
            (x < BOARD_SIZE // 2) & (y >= BOARD_SIZE // 2),
            (x >= BOARD_SIZE // 2) & (y >= BOARD_SIZE // 2),
        ),
    )
    current = states.tile_kind[:, player]
    unlock = success[:, None, None] & quadrant & (current == TileKind.LOCKED)
    return states._replace(
        money=states.money.at[:, player].add(-jnp.where(success, cost, 0)),
        unlocked_count=states.unlocked_count.at[:, player].add(
            success.astype(jnp.int8)
        ),
        tile_kind=states.tile_kind.at[:, player].set(
            jnp.where(unlock, TileKind.EMPTY, current).astype(jnp.int8)
        ),
    )


def _batch_market_quote(
    states: State,
    ops: jax.Array,
    items: jax.Array,
    active: jax.Array,
    tables: StaticTables,
) -> tuple[State, jax.Array, jax.Array]:
    batch_size = states.money.shape[0]
    batch = jnp.arange(batch_size)[:, None]
    safe_product = jnp.clip(items, 0, NUM_PRODUCTS - 1)
    valid_sell = (ops == MarketOp.SELL) & (items >= 0) & (items < NUM_PRODUCTS)
    valid_product = (ops == MarketOp.BUY_PRODUCT) & (
        (items == 0) | (items == 8)
    )
    valid_seed = (ops == MarketOp.BUY_SEED) & (items >= 0) & (items < NUM_CROPS)
    animal = items - NUM_PRODUCTS
    valid_animal = (ops == MarketOp.BUY_ANIMAL) & (animal >= 0) & (
        animal < NUM_ANIMALS
    )
    dynamic = valid_sell | valid_product
    inventory = states.market_inventory[batch, safe_product] - valid_product.astype(
        jnp.int32
    )
    lut_index = inventory - MARKET_MIN_INVENTORY
    oob = (lut_index < 0) | (lut_index >= MARKET_LUT_SIZE)
    safe_lut = jnp.clip(lut_index, 0, MARKET_LUT_SIZE - 1)
    dynamic_price = tables.market_price[safe_product, safe_lut].astype(jnp.int32)
    safe_crop = jnp.clip(items, 0, NUM_CROPS - 1)
    safe_animal = jnp.clip(animal, 0, NUM_ANIMALS - 1)
    fixed_price = jnp.where(
        valid_seed,
        _CROP_SEED_COST[safe_crop],
        jnp.where(valid_animal, _ANIMAL_COST[safe_animal], 0),
    )
    prices = jnp.where(dynamic, dynamic_price, fixed_price)
    quoted = active & (valid_sell | valid_product | valid_seed | valid_animal)
    states = states._replace(
        price_lut_oob=states.price_lut_oob
        + jnp.sum(oob & dynamic & active, axis=1).astype(jnp.int32)
    )
    return states, prices, quoted


def _batch_commit_market_player(
    states: State,
    player: int,
    op: jax.Array,
    item: jax.Array,
    price: jax.Array,
    active: jax.Array,
) -> tuple[State, jax.Array]:
    batch_size = states.money.shape[0]
    batch = jnp.arange(batch_size)
    safe_product = jnp.clip(item, 0, NUM_PRODUCTS - 1)
    animal = item - NUM_PRODUCTS
    safe_animal = jnp.clip(animal, 0, NUM_ANIMALS - 1)
    animal_item = NUM_PRODUCTS + safe_animal
    product_count = states.shed[batch, player, safe_product]
    shed_total = jnp.sum(states.shed[:, player].astype(jnp.int32), axis=1)

    sell = (
        active
        & (op == MarketOp.SELL)
        & (item >= 0)
        & (item < NUM_PRODUCTS)
        & (product_count > 0)
    )
    buy_product = (
        active
        & (op == MarketOp.BUY_PRODUCT)
        & ((item == 0) | (item == 8))
        & (states.money[:, player] >= price)
        & (shed_total < SHED_CAPACITY)
    )
    buy_seed = (
        active
        & (op == MarketOp.BUY_SEED)
        & (item >= 0)
        & (item < NUM_CROPS)
        & (states.money[:, player] >= price)
    )
    buy_animal = (
        active
        & (op == MarketOp.BUY_ANIMAL)
        & (animal >= 0)
        & (animal < NUM_ANIMALS)
        & (states.money[:, player] >= price)
        & (shed_total < SHED_CAPACITY)
    )
    success = sell | buy_product | buy_seed | buy_animal
    money_delta = (
        sell.astype(jnp.int32) * price
        - buy_product.astype(jnp.int32) * price
        - buy_seed.astype(jnp.int32) * price
        - buy_animal.astype(jnp.int32) * price
    )
    product_shed_delta = buy_product.astype(jnp.int16) - sell.astype(jnp.int16)
    market_delta = (
        (sell & (price > 1)).astype(jnp.int32) - buy_product.astype(jnp.int32)
    )
    safe_crop = jnp.clip(item, 0, NUM_CROPS - 1)
    states = states._replace(
        money=states.money.at[:, player].add(money_delta),
        shed=states.shed.at[batch, player, safe_product].add(product_shed_delta),
        market_inventory=states.market_inventory.at[batch, safe_product].add(
            market_delta
        ),
        seeds=states.seeds.at[batch, player, safe_crop].add(
            buy_seed.astype(jnp.int16)
        ),
    )
    states = states._replace(
        shed=states.shed.at[batch, player, animal_item].add(
            buy_animal.astype(jnp.int16)
        )
    )
    return states, success


def _process_market_batch_sync(
    states: State, actions: Action, tables: StaticTables
) -> State:
    """Native batched two-player market with scalar order iteration."""

    max_count = jnp.max(actions.market_count).astype(jnp.int32)

    def order_cond(carry: tuple[jax.Array, State]) -> jax.Array:
        order_index, _ = carry
        return order_index < max_count

    def order_body(carry: tuple[jax.Array, State]) -> tuple[jax.Array, State]:
        order_index, batch_states = carry
        within = order_index < actions.market_count
        ops = jnp.where(
            within, actions.market_op[:, :, order_index], MarketOp.NONE
        ).astype(jnp.int8)
        items = actions.market_item[:, :, order_index]
        remaining = actions.market_amount[:, :, order_index]

        for player in range(NUM_PLAYERS):
            batch_states = _batch_hire_player(
                batch_states, player, ops[:, player] == MarketOp.HIRE
            )
            batch_states = _batch_buy_land_player(
                batch_states, player, ops[:, player] == MarketOp.BUY_LAND
            )
        atomic = (ops == MarketOp.HIRE) | (ops == MarketOp.BUY_LAND)
        active = (~atomic) & (ops != MarketOp.NONE) & (remaining > 0)

        def unit_cond(
            unit_carry: tuple[State, jax.Array, jax.Array, jax.Array]
        ) -> jax.Array:
            _, rem, act, iteration = unit_carry
            return jnp.any(act & (rem > 0)) & (iteration < 99_999)

        def unit_body(
            unit_carry: tuple[State, jax.Array, jax.Array, jax.Array]
        ) -> tuple[State, jax.Array, jax.Array, jax.Array]:
            current, rem, act, iteration = unit_carry
            # The global batched loop may continue for another environment or
            # player.  Finished orders must stop committing immediately.
            act = act & (rem > 0)
            current, prices, quoted = _batch_market_quote(
                current, ops, items, act, tables
            )
            act = act & quoted
            successes = []
            for player in range(NUM_PLAYERS):
                current, ok = _batch_commit_market_player(
                    current,
                    player,
                    ops[:, player],
                    items[:, player],
                    prices[:, player],
                    act[:, player],
                )
                successes.append(ok)
            success = jnp.stack(successes, axis=1)
            rem = rem - success.astype(jnp.int32)
            act = act & success
            return current, rem, act, iteration + 1

        batch_states, remaining_after, active_after, iterations = lax.while_loop(
            unit_cond,
            unit_body,
            (
                batch_states,
                remaining,
                active,
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )
        del remaining_after
        hit_cap = (iterations >= 99_999) & jnp.any(active_after, axis=1)
        batch_states = batch_states._replace(
            market_loop_cap_hits=batch_states.market_loop_cap_hits
            + hit_cap.astype(jnp.int32)
        )
        batch_states = _refresh_prices_batch(batch_states, tables)
        return order_index + 1, batch_states

    _, states = lax.while_loop(
        order_cond,
        order_body,
        (jnp.asarray(0, dtype=jnp.int32), states),
    )
    return states


def _town_consume(state: State, tables: StaticTables) -> State:
    step = state.step.astype(jnp.int32)
    shop_tick = step % 4 == 0
    center_tick = step % 24 == 0
    safe_shops = jnp.clip(state.town_shops, 0, MAX_SHOPS - 1)
    active = jnp.arange(MAX_SHOPS) < state.town_count
    shop_demand = jnp.sum(_SHOP_DEMAND[safe_shops] * active[:, None], axis=0)
    center_demand = jnp.asarray(
        (1, 1, 1, 1, 1, 1, 1, 1, 0), dtype=jnp.int32
    )
    demand = jnp.where(shop_tick, shop_demand, 0) + jnp.where(
        center_tick, center_demand, 0
    )
    state = state._replace(market_inventory=state.market_inventory - demand)
    return _refresh_prices(state, tables)


def _decay_plants(state: State) -> State:
    step = state.step.astype(jnp.int16)
    mls = state.tile_max_lifespan
    decay = (
        (state.tile_kind == TileKind.PLANT)
        & (mls >= 0)
        & (step >= mls)
        & (((step - mls) % 2) == 0)
    )
    new_yield = state.tile_yield - decay.astype(jnp.int16)
    to_weed = decay & (new_yield <= 0)
    return state._replace(
        tile_kind=jnp.where(to_weed, TileKind.WEED, state.tile_kind).astype(jnp.int8),
        tile_crop=jnp.where(to_weed, -1, state.tile_crop).astype(jnp.int8),
        tile_animal=jnp.where(to_weed, -1, state.tile_animal).astype(jnp.int8),
        tile_origin_day=jnp.where(to_weed, -1, state.tile_origin_day).astype(jnp.int8),
        tile_yield=jnp.where(to_weed, 0, new_yield).astype(jnp.int16),
        tile_neglect=jnp.where(to_weed, 0, state.tile_neglect).astype(jnp.int8),
        tile_max_lifespan=jnp.where(to_weed, -1, state.tile_max_lifespan).astype(
            jnp.int16
        ),
        tile_fertilized_until=jnp.where(
            to_weed, -1, state.tile_fertilized_until
        ).astype(jnp.int8),
        tile_pending_care=jnp.where(to_weed, 0, state.tile_pending_care).astype(
            jnp.int8
        ),
        tile_flags=jnp.where(to_weed, 0, state.tile_flags).astype(jnp.uint8),
    )


def _daily_refresh_plants(state: State, day: jax.Array) -> State:
    plant = state.tile_kind == TileKind.PLANT
    flags = state.tile_flags
    watered = _flag(flags, FLAG_WATERED)
    neglect = jnp.where(watered, 0, state.tile_neglect + 1).astype(jnp.int8)
    dies = plant & (neglect >= 2)
    flags = _set_flag(flags, FLAG_WATERED, jnp.zeros_like(watered))

    safe_crop = jnp.clip(state.tile_crop, 0, NUM_CROPS - 1)
    next_day = day.astype(jnp.int16) + 1
    days_since_first = (
        next_day - state.tile_origin_day.astype(jnp.int16) - _CROP_FIRST[safe_crop]
    )
    interval = jnp.maximum(_CROP_INTERVAL[safe_crop], 1)
    production_count = days_since_first // interval + 1
    due = (
        plant
        & (~dies)
        & _CROP_ONGOING[safe_crop]
        & (days_since_first >= 0)
        & ((days_since_first % interval) == 0)
        & (production_count <= _CROP_MAX[safe_crop])
    )
    fertilized = watered & (state.tile_fertilized_until >= day)
    produced = jnp.where(fertilized, 2, 1).astype(jnp.int16)
    yield_units = jnp.where(
        due,
        jnp.minimum(_CROP_MAX[safe_crop], state.tile_yield + produced),
        state.tile_yield,
    )
    last_production = due & (production_count == _CROP_MAX[safe_crop])
    max_life = jnp.where(
        last_production, (next_day + 1) * TURNS_PER_DAY, state.tile_max_lifespan
    ).astype(jnp.int16)
    return state._replace(
        tile_kind=jnp.where(dies, TileKind.WEED, state.tile_kind).astype(jnp.int8),
        tile_crop=jnp.where(dies, -1, state.tile_crop).astype(jnp.int8),
        tile_origin_day=jnp.where(dies, -1, state.tile_origin_day).astype(jnp.int8),
        tile_yield=jnp.where(dies, 0, yield_units).astype(jnp.int16),
        tile_neglect=jnp.where(dies, 0, jnp.where(plant, neglect, state.tile_neglect)).astype(
            jnp.int8
        ),
        tile_max_lifespan=jnp.where(dies, -1, max_life).astype(jnp.int16),
        tile_fertilized_until=jnp.where(
            dies, -1, state.tile_fertilized_until
        ).astype(jnp.int8),
        tile_flags=jnp.where(dies, 0, flags).astype(jnp.uint8),
    )


def _daily_refresh_animals(state: State, day: jax.Array) -> State:
    animal_present = state.tile_animal >= 0
    safe_animal = jnp.clip(state.tile_animal, 0, NUM_ANIMALS - 1)
    fed = _flag(state.tile_flags, FLAG_FED)
    cared = _flag(state.tile_flags, FLAG_CARED)
    neglect = jnp.where(fed, 0, state.tile_neglect + 1).astype(jnp.int8)
    escapes = animal_present & (neglect >= 2)

    next_day = day.astype(jnp.int16) + 1
    days_since_first = (
        next_day
        - state.tile_origin_day.astype(jnp.int16)
        - _ANIMAL_FIRST[safe_animal]
    )
    interval = _ANIMAL_INTERVAL[safe_animal]
    due = (
        animal_present
        & (~escapes)
        & (days_since_first >= 0)
        & ((days_since_first % interval) == 0)
    )
    bonus = jnp.where(fed, state.tile_pending_care, 0).astype(jnp.int16)
    yield_units = jnp.where(
        due,
        jnp.minimum(_ANIMAL_MAX[safe_animal], state.tile_yield + 1 + bonus),
        state.tile_yield,
    )
    pending = jnp.where(due, 0, state.tile_pending_care).astype(jnp.int8)
    pending = pending + (animal_present & (~escapes) & cared & fed).astype(jnp.int8)
    flags = _set_flag(state.tile_flags, FLAG_FERTILIZER_AVAILABLE, animal_present & (~escapes))
    flags = _set_flag(flags, FLAG_FED, jnp.zeros_like(fed))
    flags = _set_flag(flags, FLAG_CARED, jnp.zeros_like(cared))

    return state._replace(
        tile_animal=jnp.where(escapes, -1, state.tile_animal).astype(jnp.int8),
        tile_origin_day=jnp.where(escapes, -1, state.tile_origin_day).astype(jnp.int8),
        tile_yield=jnp.where(escapes, 0, yield_units).astype(jnp.int16),
        tile_neglect=jnp.where(
            escapes,
            0,
            jnp.where(animal_present, neglect, state.tile_neglect),
        ).astype(jnp.int8),
        tile_pending_care=jnp.where(escapes, 0, pending).astype(jnp.int8),
        tile_flags=jnp.where(escapes, 0, flags).astype(jnp.uint8),
    )


def _spawn_weeds_and_shop(state: State, events: Events, day: jax.Array) -> State:
    flat_kind = state.tile_kind.reshape((NUM_PLAYERS * NUM_TILES,))
    empty = flat_kind == TileKind.EMPTY
    draw_index = jnp.cumsum(empty.astype(jnp.int16)) - 1
    safe_index = jnp.clip(draw_index, 0, NUM_PLAYERS * NUM_TILES - 1)
    spawn = empty & events.weed_spawn[day, safe_index]
    flat_kind = jnp.where(spawn, TileKind.WEED, flat_kind).astype(jnp.int8)
    empty_count = jnp.sum(empty.astype(jnp.int16))
    state = state._replace(tile_kind=flat_kind.reshape(state.tile_kind.shape))

    next_day = day + 1
    unlock = (
        (next_day > 0)
        & ((next_day % 3) == 0)
        & (state.town_count < MAX_SHOPS)
    )
    slot = jnp.clip(state.town_count, 0, MAX_SHOPS - 1)
    shop = events.shop_choice[day, empty_count]
    return state._replace(
        town_shops=state.town_shops.at[slot].set(
            jnp.where(unlock, shop, state.town_shops[slot]).astype(jnp.int8)
        ),
        town_count=state.town_count + unlock.astype(jnp.int8),
    )


def _drop_all_and_reset_units(state: State) -> State:
    def player_body(player: int, s: State) -> State:
        unit_count = 1 + s.hires_today[player].astype(jnp.int32)

        def unit_cond(carry: tuple[jax.Array, State]) -> jax.Array:
            unit, _ = carry
            return unit < unit_count

        def unit_body(carry: tuple[jax.Array, State]) -> tuple[jax.Array, State]:
            unit, z = carry
            z = _drop_inventory(z, jnp.asarray(player), unit)
            return unit + 1, z

        _, s = lax.while_loop(
            unit_cond,
            unit_body,
            (jnp.asarray(0, dtype=jnp.int32), s),
        )
        return s

    state = lax.fori_loop(0, NUM_PLAYERS, player_body, state)
    unit_pos = jnp.zeros_like(state.unit_pos)
    unit_pos = unit_pos.at[:, 0].set(jnp.asarray(DEFAULT_SPAWN, dtype=jnp.int8))
    unit_active = jnp.zeros_like(state.unit_active).at[:, 0].set(True)
    return state._replace(
        unit_pos=unit_pos,
        unit_active=unit_active,
        unit_inventory=jnp.zeros_like(state.unit_inventory),
        unit_inventory_order=jnp.full_like(state.unit_inventory_order, -1),
        unit_inventory_next_order=jnp.zeros_like(state.unit_inventory_next_order),
        hires_today=jnp.zeros_like(state.hires_today),
    )


def _end_of_day(state: State, events: Events, day: jax.Array) -> State:
    state = _daily_refresh_plants(state, day)
    state = _daily_refresh_animals(state, day)
    state = _spawn_weeds_and_shop(state, events, day)
    return _drop_all_and_reset_units(state)


def project_unit_phase_env(state: State, action: Action) -> State:
    """Apply only the official unit phase, without advancing the environment."""

    day = (state.step // TURNS_PER_DAY).astype(jnp.int8)
    return _process_units(state, action, day)


def project_action_phases_env(
    state: State, action: Action, tables: StaticTables
) -> State:
    """Apply official unit + market phases, excluding town/end-of-step effects.

    Strategic compilers use this public projection boundary for parity tests.
    Keeping it in the simulator prevents a planner from inventing a second
    interpretation of unit-before-market and lockstep market settlement.
    """

    state = project_unit_phase_env(state, action)
    return _process_market(state, action, tables)


batched_project_unit_phase = jax.vmap(project_unit_phase_env, in_axes=(0, 0))
batched_project_action_phases = jax.vmap(
    project_action_phases_env, in_axes=(0, 0, None)
)


def batched_project_action_phases_sync(
    states: State, actions: Action, tables: StaticTables
) -> State:
    """Fast synchronized-batch projection of the official unit+market phases."""

    day = (states.step[0] // TURNS_PER_DAY).astype(jnp.int8)
    states = jax.vmap(_process_units, in_axes=(0, 0, None))(states, actions, day)
    return _process_market_batch_sync(states, actions, tables)


def step_env(
    state: State, action: Action, events: Events, tables: StaticTables
) -> State:
    """Advance one official environment transition for both players."""

    def advance(s: State) -> State:
        day = (s.step // TURNS_PER_DAY).astype(jnp.int8)
        s = _process_units(s, action, day)
        s = _process_market(s, action, tables)
        s = _town_consume(s, tables)
        s = _decay_plants(s)
        s = lax.cond(
            ((s.step.astype(jnp.int32) + 1) % TURNS_PER_DAY) == 0,
            lambda z: _end_of_day(z, events, day),
            lambda z: z,
            s,
        )
        terminal = s.step.astype(jnp.int32) >= EPISODE_STEPS - 2
        reward = jnp.where(terminal, s.money, s.reward)
        return s._replace(
            step=(s.step + 1).astype(jnp.int16),
            reward=reward.astype(jnp.int32),
            done=terminal,
        )

    return lax.cond(state.done, lambda s: s, advance, state)


batched_step = jax.vmap(step_env, in_axes=(0, 0, 0, None))


def batched_step_sync(
    states: State, actions: Action, events: Events, tables: StaticTables
) -> State:
    """Fast batch path for the synchronized environments used by self-play.

    All environments must have the same ``step`` and ``done`` values.  Keeping
    episode/end-of-day control flow outside ``vmap`` avoids materializing both
    full-state branches for every environment, which is critical at batch 4096.
    The general independent-state path remains available as ``batched_step``.
    """

    step0 = states.step[0]
    done0 = states.done[0]

    def advance(batch: State) -> State:
        day = (step0 // TURNS_PER_DAY).astype(jnp.int8)
        batch = jax.vmap(_process_units, in_axes=(0, 0, None))(
            batch, actions, day
        )
        batch = _process_market_batch_sync(batch, actions, tables)
        batch = jax.vmap(_town_consume, in_axes=(0, None))(batch, tables)
        batch = jax.vmap(_decay_plants)(batch)
        batch = lax.cond(
            ((step0.astype(jnp.int32) + 1) % TURNS_PER_DAY) == 0,
            lambda value: jax.vmap(_end_of_day, in_axes=(0, 0, None))(
                value, events, day
            ),
            lambda value: value,
            batch,
        )
        terminal = step0.astype(jnp.int32) >= EPISODE_STEPS - 2
        reward = jnp.where(terminal, batch.money, batch.reward)
        return batch._replace(
            step=(batch.step + 1).astype(jnp.int16),
            reward=reward.astype(jnp.int32),
            done=jnp.full_like(batch.done, terminal),
        )

    return lax.cond(done0, lambda value: value, advance, states)


def batched_step_from_projected_unit_phase_sync(
    projected: State, actions: Action, events: Events, tables: StaticTables
) -> State:
    """Finish a synchronized step whose official unit phase is already applied.

    This is exactly the suffix of :func:`batched_step_sync`: shared-market
    lockstep, town consumption, crop decay, optional day-end processing, then
    step/reward/done bookkeeping.  It permits a planner to reuse the one joint
    unit projection that it needed for ordered market planning instead of
    simulating every unit action a second time.

    All environments must have synchronized ``step``/``done`` values, matching
    the contract of ``batched_step_sync``.
    """

    step0 = projected.step[0]
    done0 = projected.done[0]

    def advance(batch: State) -> State:
        day = (step0 // TURNS_PER_DAY).astype(jnp.int8)
        batch = _process_market_batch_sync(batch, actions, tables)
        batch = jax.vmap(_town_consume, in_axes=(0, None))(batch, tables)
        batch = jax.vmap(_decay_plants)(batch)
        batch = lax.cond(
            ((step0.astype(jnp.int32) + 1) % TURNS_PER_DAY) == 0,
            lambda value: jax.vmap(_end_of_day, in_axes=(0, 0, None))(
                value, events, day
            ),
            lambda value: value,
            batch,
        )
        terminal = step0.astype(jnp.int32) >= EPISODE_STEPS - 2
        reward = jnp.where(terminal, batch.money, batch.reward)
        return batch._replace(
            step=(batch.step + 1).astype(jnp.int16),
            reward=reward.astype(jnp.int32),
            done=jnp.full_like(batch.done, terminal),
        )

    return lax.cond(done0, lambda value: value, advance, projected)
