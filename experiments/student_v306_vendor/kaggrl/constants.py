"""Stable vocabularies for Kaggle's official Kaggriculture environment.

The values in this file mirror ``kaggle_environments.envs.kaggriculture``.
Index zero is reserved for padding in every categorical vocabulary.
"""

from __future__ import annotations


CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")

# ``PAD`` is never emitted as an environment action.  Keeping it at zero lets
# inactive action branches use a valid categorical index without leaking it to
# Kaggle's action schema.
ITEMS = ("PAD", *PRODUCTS, *ANIMALS)
ITEM_TO_ID = {item: index for index, item in enumerate(ITEMS)}

UNIT_OPS = (
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PICKUP",
    "DROP",
    "PLACE",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
)
UNIT_OP_TO_ID = {op: index for index, op in enumerate(UNIT_OPS)}
UNIT_ITEM_OPS = frozenset({"PICKUP", "PLACE", "PLANT"})
UNIT_QTY_OPS = frozenset({"PICKUP", "PLACE"})

# ``NONE`` is end-of-sequence. ``PASS`` is an official slot-consuming no-op
# and is therefore distinct: public agents often emit an unfillable order in
# one slot followed by effectful later orders.
MARKET_OPS = (
    "NONE", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND", "PASS"
)
MARKET_OP_TO_ID = {op: index for index, op in enumerate(MARKET_OPS)}
MARKET_ITEM_QTY_OPS = frozenset({"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"})

# Kaggle's default shed capacity is 100, which is also an upper bound on any
# useful per-order quantity.  Larger custom capacities are conservatively
# clipped by masks rather than producing an out-of-range action index.
MAX_UNIT_QTY = 100
MAX_MARKET_QTY = 100
MAX_MARKET_ORDERS = 10

# Entity-token vocabulary sizes.  The tokenizer deliberately leaves room for
# custom board sizes and new environment categories while preserving index 0 as
# padding/unknown.
CONT_DIM = 24
NUM_TOKEN_TYPES = 7  # pad, global, farm, unit, tile, market-item, town-shop
NUM_GENERIC_CATEGORIES = 32
NUM_POSITION_IDS = 64
NUM_OWNER_IDS = 4  # pad, self, opponent, shared

TOKEN_PAD = 0
TOKEN_GLOBAL = 1
TOKEN_FARM = 2
TOKEN_UNIT = 3
TOKEN_TILE = 4
TOKEN_MARKET = 5
TOKEN_TOWN = 6

OWNER_PAD = 0
OWNER_SELF = 1
OWNER_OPPONENT = 2
OWNER_SHARED = 3
