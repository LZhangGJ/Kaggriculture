"""Construction helpers for fixed-shape simulator pytrees."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .constants import (
    BOARD_SIZE,
    DEFAULT_SPAWN,
    MARKET_BASE_PRICES,
    MARKET_INITIAL_INVENTORY,
    MAX_MARKET_ORDERS,
    MAX_SHOPS,
    MAX_UNITS,
    NUM_CROPS,
    NUM_DAYS,
    NUM_PLAYERS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    NUM_TILES,
    TileKind,
    UnitOp,
)
from .types import Action, Events, State, StaticTables


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_TABLES_PATH = PROJECT / "reference" / "static_tables_v1.npz"


def reset(seed: int | jnp.ndarray = 0) -> State:
    y, x = jnp.indices((BOARD_SIZE, BOARD_SIZE))
    unlocked = (x < BOARD_SIZE // 2) & (y < BOARD_SIZE // 2)
    initial_kind = jnp.where(unlocked, TileKind.EMPTY, TileKind.LOCKED).astype(
        jnp.int8
    )
    tile_shape = (NUM_PLAYERS, BOARD_SIZE, BOARD_SIZE)
    unit_pos = jnp.zeros((NUM_PLAYERS, MAX_UNITS, 2), dtype=jnp.int8)
    unit_pos = unit_pos.at[:, 0, :].set(jnp.asarray(DEFAULT_SPAWN, dtype=jnp.int8))
    unit_active = jnp.zeros((NUM_PLAYERS, MAX_UNITS), dtype=jnp.bool_)
    unit_active = unit_active.at[:, 0].set(True)
    return State(
        step=jnp.asarray(0, dtype=jnp.int16),
        episode_seed=jnp.asarray(seed, dtype=jnp.int32),
        money=jnp.full((NUM_PLAYERS,), 3000, dtype=jnp.int32),
        tile_kind=jnp.broadcast_to(initial_kind, tile_shape),
        tile_crop=jnp.full(tile_shape, -1, dtype=jnp.int8),
        tile_animal=jnp.full(tile_shape, -1, dtype=jnp.int8),
        tile_origin_day=jnp.full(tile_shape, -1, dtype=jnp.int8),
        tile_yield=jnp.zeros(tile_shape, dtype=jnp.int16),
        tile_neglect=jnp.zeros(tile_shape, dtype=jnp.int8),
        tile_max_lifespan=jnp.full(tile_shape, -1, dtype=jnp.int16),
        tile_fertilized_until=jnp.full(tile_shape, -1, dtype=jnp.int8),
        tile_pending_care=jnp.zeros(tile_shape, dtype=jnp.int8),
        tile_flags=jnp.zeros(tile_shape, dtype=jnp.uint8),
        unit_pos=unit_pos,
        unit_active=unit_active,
        unit_inventory=jnp.zeros(
            (NUM_PLAYERS, MAX_UNITS, NUM_SHED_ITEMS), dtype=jnp.int16
        ),
        unit_inventory_order=jnp.full(
            (NUM_PLAYERS, MAX_UNITS, NUM_SHED_ITEMS), -1, dtype=jnp.int8
        ),
        unit_inventory_next_order=jnp.zeros(
            (NUM_PLAYERS, MAX_UNITS), dtype=jnp.int8
        ),
        shed=jnp.zeros((NUM_PLAYERS, NUM_SHED_ITEMS), dtype=jnp.int16),
        seeds=jnp.zeros((NUM_PLAYERS, NUM_CROPS), dtype=jnp.int16),
        hires_today=jnp.zeros((NUM_PLAYERS,), dtype=jnp.int8),
        unlocked_count=jnp.ones((NUM_PLAYERS,), dtype=jnp.int8),
        market_inventory=jnp.full(
            (NUM_PRODUCTS,), MARKET_INITIAL_INVENTORY, dtype=jnp.int32
        ),
        market_price=jnp.asarray(MARKET_BASE_PRICES, dtype=jnp.int16),
        town_shops=jnp.full((MAX_SHOPS,), -1, dtype=jnp.int8),
        town_count=jnp.asarray(0, dtype=jnp.int8),
        reward=jnp.zeros((NUM_PLAYERS,), dtype=jnp.int32),
        done=jnp.asarray(False),
        hand_cap_hits=jnp.zeros((NUM_PLAYERS,), dtype=jnp.int32),
        market_loop_cap_hits=jnp.asarray(0, dtype=jnp.int32),
        price_lut_oob=jnp.asarray(0, dtype=jnp.int32),
    )


def empty_action() -> Action:
    return Action(
        unit_op=jnp.full(
            (NUM_PLAYERS, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8
        ),
        unit_item=jnp.full((NUM_PLAYERS, MAX_UNITS), -1, dtype=jnp.int8),
        unit_amount=jnp.ones((NUM_PLAYERS, MAX_UNITS), dtype=jnp.int32),
        unit_count=jnp.ones((NUM_PLAYERS,), dtype=jnp.int8),
        market_op=jnp.zeros(
            (NUM_PLAYERS, MAX_MARKET_ORDERS), dtype=jnp.int8
        ),
        market_item=jnp.full(
            (NUM_PLAYERS, MAX_MARKET_ORDERS), -1, dtype=jnp.int8
        ),
        market_amount=jnp.zeros(
            (NUM_PLAYERS, MAX_MARKET_ORDERS), dtype=jnp.int32
        ),
        market_count=jnp.zeros((NUM_PLAYERS,), dtype=jnp.int8),
    )


def load_tables(path: Path = DEFAULT_TABLES_PATH) -> StaticTables:
    with np.load(path, allow_pickle=False) as data:
        market_price = np.asarray(data["market_price"], dtype=np.int16)
    return StaticTables(market_price=jnp.asarray(market_price))


def load_event_bank(
    path: Path = DEFAULT_TABLES_PATH,
) -> tuple[np.ndarray, Events]:
    with np.load(path, allow_pickle=False) as data:
        seeds = np.asarray(data["event_seeds"], dtype=np.int64)
        weed = np.asarray(data["weed_spawn"], dtype=np.bool_)
        choice = np.asarray(data["shop_choice"], dtype=np.int8)
    return seeds, Events(weed_spawn=jnp.asarray(weed), shop_choice=jnp.asarray(choice))


def events_for_seed(seed: int, seeds: np.ndarray, bank: Events) -> Events:
    matches = np.flatnonzero(seeds == seed)
    if len(matches) != 1:
        raise KeyError(f"Seed {seed} is not unique in the frozen event bank")
    index = int(matches[0])
    return Events(
        weed_spawn=bank.weed_spawn[index], shop_choice=bank.shop_choice[index]
    )
