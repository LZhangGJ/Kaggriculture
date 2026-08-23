"""JAX pytree state, action and frozen-event containers."""

from __future__ import annotations

from typing import NamedTuple

import jax


Array = jax.Array


class State(NamedTuple):
    step: Array
    episode_seed: Array
    money: Array
    tile_kind: Array
    tile_crop: Array
    tile_animal: Array
    tile_origin_day: Array
    tile_yield: Array
    tile_neglect: Array
    tile_max_lifespan: Array
    tile_fertilized_until: Array
    tile_pending_care: Array
    tile_flags: Array
    unit_pos: Array
    unit_active: Array
    unit_inventory: Array
    unit_inventory_order: Array
    unit_inventory_next_order: Array
    shed: Array
    seeds: Array
    hires_today: Array
    unlocked_count: Array
    market_inventory: Array
    market_price: Array
    town_shops: Array
    town_count: Array
    reward: Array
    done: Array
    hand_cap_hits: Array
    market_loop_cap_hits: Array
    price_lut_oob: Array


class Action(NamedTuple):
    unit_op: Array
    unit_item: Array
    unit_amount: Array
    unit_count: Array
    market_op: Array
    market_item: Array
    market_amount: Array
    market_count: Array


class Events(NamedTuple):
    weed_spawn: Array
    shop_choice: Array


class StaticTables(NamedTuple):
    market_price: Array
    market_price_prefix: Array
    market_first_floor_index: Array
