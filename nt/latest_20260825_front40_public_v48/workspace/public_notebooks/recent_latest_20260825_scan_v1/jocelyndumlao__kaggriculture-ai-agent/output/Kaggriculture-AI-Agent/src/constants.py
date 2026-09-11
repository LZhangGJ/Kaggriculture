"""
constants.py

Global constants and enumerations for the Kaggriculture AI Agent.

This module contains only static values and enumerations used
throughout the project. No game logic should be implemented here.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from enum import Enum


# ==========================================================
# Game Configuration
# ==========================================================

BOARD_SIZE = 10
QUADRANT_SIZE = 5

TURNS_PER_DAY = 24
TOTAL_DAYS = 30
TOTAL_TURNS = TURNS_PER_DAY * TOTAL_DAYS

STARTING_MONEY = 3000

SHED_CAPACITY = 100

MAX_MARKET_ORDERS = 10

# ==========================================================
# Directions
# ==========================================================


class Direction(str, Enum):
    """Movement directions."""

    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"


# Convenient aliases
NORTH = Direction.NORTH
SOUTH = Direction.SOUTH
EAST = Direction.EAST
WEST = Direction.WEST


# Movement vectors
DIRECTION_VECTOR = {
    Direction.NORTH: (0, -1),
    Direction.SOUTH: (0, 1),
    Direction.EAST: (1, 0),
    Direction.WEST: (-1, 0),
}


REVERSE_DIRECTION = {
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.EAST: Direction.WEST,
    Direction.WEST: Direction.EAST,
}

# ==========================================================
# Farmer Actions
# ==========================================================


class Action(str, Enum):
    """Actions the farmer can perform."""

    PASS = "PASS"

    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"

    PLANT = "PLANT"
    WATER = "WATER"
    HARVEST = "HARVEST"
    DIG = "DIG"

    FERTILIZE = "FERTILIZE"

    FEED = "FEED"
    CARE = "CARE"

    COLLECT_FERTILIZER = "COLLECT_FERTILIZER"

    PICKUP = "PICKUP"
    PLACE = "PLACE"
    DROP = "DROP"

    BUILD_COOP = "BUILD_COOP"
    BUILD_PASTURE = "BUILD_PASTURE"


MOVE_ACTIONS = (
    Action.NORTH,
    Action.SOUTH,
    Action.EAST,
    Action.WEST,
)

# ==========================================================
# Market Actions
# ==========================================================


class MarketAction(str, Enum):
    """Market actions."""

    BUY_SEED = "BUY_SEED"
    BUY_PRODUCT = "BUY_PRODUCT"
    BUY_ANIMAL = "BUY_ANIMAL"

    SELL = "SELL"

    BUY_LAND = "BUY_LAND"

    HIRE = "HIRE"

# ==========================================================
# Crops
# ==========================================================


class Crop(str, Enum):

    WHEAT = "WHEAT"
    CARROT = "CARROT"
    TOMATO = "TOMATO"
    STRAWBERRY = "STRAWBERRY"
    MELON = "MELON"

# ==========================================================
# Animals
# ==========================================================


class Animal(str, Enum):

    GOOSE = "GOOSE"
    COW = "COW"
    SHEEP = "SHEEP"

# ==========================================================
# Buildings
# ==========================================================


class Building(str, Enum):

    COOP = "COOP"
    PASTURE = "PASTURE"

# ==========================================================
# Products
# ==========================================================


class Product(str, Enum):

    WHEAT = "WHEAT"
    CARROT = "CARROT"
    TOMATO = "TOMATO"
    STRAWBERRY = "STRAWBERRY"
    MELON = "MELON"

    EGG = "EGG"
    MILK = "MILK"
    WOOL = "WOOL"

    FERTILIZER = "FERTILIZER"

# ==========================================================
# Tile Types
# ==========================================================


class TileKind(str, Enum):

    EMPTY = "EMPTY"
    LOCKED = "LOCKED"

    PLANT = "PLANT"
    WEED = "WEED"

    COOP = "COOP"
    PASTURE = "PASTURE"

# ==========================================================
# Farm Quadrants
# ==========================================================


class Quadrant(str, Enum):

    NW = "NW"
    NE = "NE"
    SW = "SW"
    SE = "SE"

# ==========================================================
# Town Shops
# ==========================================================


class Shop(str, Enum):

    BAKERY = "BAKERY"
    PIZZA_SHOP = "PIZZA_SHOP"
    BRUNCH_SPOT = "BRUNCH_SPOT"
    YARN_STORE = "YARN_STORE"
    ICE_CREAM_SHOP = "ICE_CREAM_SHOP"
    PET_CAFE = "PET_CAFE"
    SMOOTHIE_SHOP = "SMOOTHIE_SHOP"
    FARMERS_MARKET = "FARMERS_MARKET"

# ==========================================================
# Resource Collections
# ==========================================================

ALL_CROPS = tuple(Crop)

ALL_ANIMALS = tuple(Animal)

ALL_PRODUCTS = tuple(Product)

ALL_BUILDINGS = tuple(Building)

ALL_SHOPS = tuple(Shop)

ALL_DIRECTIONS = tuple(Direction)

ALL_ACTIONS = tuple(Action)

ALL_MARKET_ACTIONS = tuple(MarketAction)

# ==========================================================
# Crop Categories
# ==========================================================

ONE_TIME_CROPS = {
    Crop.WHEAT,
    Crop.CARROT,
    Crop.MELON,
}

ONGOING_CROPS = {
    Crop.TOMATO,
    Crop.STRAWBERRY,
}

# ==========================================================
# Animal Products
# ==========================================================

ANIMAL_PRODUCTS = {
    Animal.GOOSE: Product.EGG,
    Animal.COW: Product.MILK,
    Animal.SHEEP: Product.WOOL,
}

# ==========================================================
# Animal Buildings
# ==========================================================

BUILDING_FOR_ANIMAL = {
    Animal.GOOSE: Building.COOP,
    Animal.COW: Building.PASTURE,
    Animal.SHEEP: Building.PASTURE,
}

# ==========================================================
# Helper Lookup Sets
# ==========================================================

MOVE_DIRECTIONS = {
    Direction.NORTH.value,
    Direction.SOUTH.value,
    Direction.EAST.value,
    Direction.WEST.value,
}

MOVE_ACTION_VALUES = {
    Action.NORTH.value,
    Action.SOUTH.value,
    Action.EAST.value,
    Action.WEST.value,
}