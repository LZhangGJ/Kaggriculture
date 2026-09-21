"""Fixed IDs and tensor constants for the official default configuration."""

from __future__ import annotations

from enum import IntEnum


NUM_PLAYERS = 2
BOARD_SIZE = 10
NUM_TILES = BOARD_SIZE * BOARD_SIZE
TURNS_PER_DAY = 24
NUM_DAYS = 30
EPISODE_STEPS = 720
MAX_HANDS = 32
MAX_UNITS = 1 + MAX_HANDS
MAX_MARKET_ORDERS = 10
MAX_SHOPS = 8
SHED_CAPACITY = 100

PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)
CROPS = PRODUCTS[:5]
ANIMALS = ("GOOSE", "COW", "SHEEP")
SHED_ITEMS = PRODUCTS + ANIMALS
NUM_PRODUCTS = len(PRODUCTS)
NUM_CROPS = len(CROPS)
NUM_ANIMALS = len(ANIMALS)
NUM_SHED_ITEMS = len(SHED_ITEMS)

SHOP_NAMES = tuple(
    sorted(
        (
            "BAKERY",
            "PIZZA_SHOP",
            "BRUNCH_SPOT",
            "YARN_STORE",
            "ICE_CREAM_SHOP",
            "PET_CAFE",
            "SMOOTHIE_SHOP",
            "FARMERS_MARKET",
        )
    )
)

MARKET_MIN_INVENTORY = -32768
MARKET_MAX_INVENTORY = 65535
MARKET_LUT_SIZE = MARKET_MAX_INVENTORY - MARKET_MIN_INVENTORY + 1
MARKET_INITIAL_INVENTORY = 10000
MARKET_BASE_PRICES = (25, 35, 60, 120, 250, 50, 160, 200, 100)

CROP_SEED_COST = (10, 20, 50, 100, 80)
CROP_FIRST_YIELD_DAY = (2, 2, 8, 10, 10)
CROP_MAX_YIELD_DAY = (4, 3, 8, 10, 12)
CROP_INTERVAL = (0, 0, 1, 2, 0)
CROP_MAX_YIELD = (6, 4, 4, 4, 6)
CROP_ONGOING = (False, False, True, True, False)

ANIMAL_COST = (300, 400, 500)
ANIMAL_STRUCTURE = (4, 5, 5)  # TileKind.COOP/PASTURE
ANIMAL_FIRST_YIELD_DAY = (4, 8, 6)
ANIMAL_INTERVAL = (1, 2, 3)
ANIMAL_MAX_HELD = (4, 6, 6)
ANIMAL_PRODUCT = (5, 6, 7)

LAND_PRICES = (1000, 2000, 4000)
DEFAULT_SPAWN = (4, 4)
SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))


class TileKind(IntEnum):
    EMPTY = 0
    LOCKED = 1
    WEED = 2
    PLANT = 3
    COOP = 4
    PASTURE = 5


class UnitOp(IntEnum):
    PASS = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4
    DROP = 5
    PICKUP = 6
    PLACE = 7
    PLANT = 8
    WATER = 9
    HARVEST = 10
    FERTILIZE = 11
    DIG = 12
    BUILD_COOP = 13
    BUILD_PASTURE = 14
    FEED = 15
    COLLECT_FERTILIZER = 16
    CARE = 17


class MarketOp(IntEnum):
    NONE = 0
    HIRE = 1
    BUY_LAND = 2
    BUY_SEED = 3
    BUY_PRODUCT = 4
    BUY_ANIMAL = 5
    SELL = 6


FLAG_WATERED = 1
FLAG_FED = 2
FLAG_CARED = 4
FLAG_FERTILIZER_AVAILABLE = 8
