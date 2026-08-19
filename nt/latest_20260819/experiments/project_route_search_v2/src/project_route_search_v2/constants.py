"""Frozen M0 IDs and fixed tensor limits for Project Route Search V2."""

from __future__ import annotations

from enum import IntEnum

from kaggriculture_jax.constants import (
    ANIMALS,
    BOARD_SIZE,
    CROPS,
    EPISODE_STEPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_DAYS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    NUM_TILES,
    PRODUCTS,
    SHED_CAPACITY,
)


CONTRACT_VERSION = "E0_PROJECT_ROUTE_SEARCH_V1"
SCHEMA_VERSION_V2 = 2
OFFICIAL_PACKAGE_VERSION = "1.32.7"
TRUTH_BOUNDARY = "BEST_OBSERVED_WITHIN_FROZEN_DOMAIN"

MAX_PROJECTS_V2 = 24
MAX_PHASES_V2 = 6
MAX_OBLIGATIONS_V2 = 128
MAX_LAYOUT_CANDIDATES_V2 = 16
MAX_PROJECT_INTENTS_V2 = 24
MAX_ROUTE_STOPS = 8
# Route cards are a new executor-side contract.  Keep the frozen M0
# ``MAX_ROUTE_STOPS`` untouched for old UnitPlan fixtures, while allowing the
# observed gold maximum of ten WATER stops plus two slots of headroom.
MAX_ROUTE_CARD_STOPS_V3 = 12
K_EXACT_TASKS = 8
U_EXACT_UNITS = 12
STATIC_SPECIES_CAP = NUM_TILES

ROUTE_FAMILY_NAMES = (
    "R1_FAST_CROP",
    "R2_TOMATO",
    "R3_STRAWBERRY",
    "R4_MELON",
    "R5_EGG",
    "R6_MILK",
    "R7_WOOL",
    "R8_HYBRID",
)

NORMAL_LAYOUT_NAMES = (
    "CENTER_COMPACT",
    "EDGE_COMPACT",
    "QUADRANT_ZONED",
    "TOUR_OPTIMIZED",
)
CONTROL_LAYOUT_NAME = "DISPERSED_CONTROL"


class DomainModeV2(IntEnum):
    NATIVE_DOMAIN = 0
    CAUSAL_HYBRID = 1


class RouteFamilyV2(IntEnum):
    R1_FAST_CROP = 0
    R2_TOMATO = 1
    R3_STRAWBERRY = 2
    R4_MELON = 3
    R5_EGG = 4
    R6_MILK = 5
    R7_WOOL = 6
    R8_HYBRID = 7


class ProjectTypeV2(IntEnum):
    NONE = 0
    CROP_LOT = 1
    ANIMAL_LOT = 2
    FEED_SUPPLY = 3
    LAND_EXPANSION = 4
    WORKFORCE = 5
    INVENTORY_POLICY = 6
    LIQUIDATION = 7


class ProjectStatusV2(IntEnum):
    UNUSED = 0
    PLANNED = 1
    ACTIVE = 2
    COMPLETE = 3
    STOPPED = 4
    FAILED = 5


class ObligationPriorityV2(IntEnum):
    HARD = 0
    URGENT = 1
    OPTIONAL = 2


class LayoutPolicyV2(IntEnum):
    CENTER_COMPACT = 0
    EDGE_COMPACT = 1
    QUADRANT_ZONED = 2
    TOUR_OPTIMIZED = 3
    DISPERSED_CONTROL = 4


class M26CropLayoutPolicyV2(IntEnum):
    """Coarse crop layouts frozen by the M2.6 contract."""

    CENTER_COMPACT = 0
    CLUSTER_EXPANSION = 1
    QUADRANT_ZONED = 2
    ROUTE_STRIP = 3


class M26FertilizerPolicyV2(IntEnum):
    OFF = 0
    ALWAYS_WHEN_AVAILABLE = 1
    HIGH_VALUE_ONLY = 2


class M26WeedRecoveryPolicyV2(IntEnum):
    ABANDON_TILE = 0
    DIG_AND_REPLANT_SAME_CROP = 1
    DIG_AND_REPLAN_TO_PHASE_DEFICIT = 2


class M26CropAbandonPolicyV2(IntEnum):
    COMPLETE_EXISTING_PROJECTS = 0
    ABANDON_ONLY_IF_NOT_BANKABLE = 1
    ABANDON_IF_TARGET_SHRINKS_AND_REPLACEMENT_IS_BETTER = 2


class CropProjectPhaseV2(IntEnum):
    NONE = 0
    ACQUIRING_SEEDS = 1
    ESTABLISHING = 2
    ACTIVE = 3
    HARVESTING = 4
    CLOSING = 5
    DONE = 6
    FAILED = 7


class CropObligationTypeV2(IntEnum):
    NONE = 0
    BUY_SEED = 1
    PLANT_AND_WATER = 2
    WATER_CROP = 3
    HARVEST_AND_DEPOSIT = 4
    DEPOSIT_INVENTORY = 5
    SELL_SHED = 6
    LIQUIDATE_SELL = 7


__all__ = [
    "ANIMALS",
    "BOARD_SIZE",
    "CONTRACT_VERSION",
    "CONTROL_LAYOUT_NAME",
    "CROPS",
    "CropObligationTypeV2",
    "CropProjectPhaseV2",
    "DomainModeV2",
    "EPISODE_STEPS",
    "K_EXACT_TASKS",
    "LayoutPolicyV2",
    "M26CropAbandonPolicyV2",
    "M26CropLayoutPolicyV2",
    "M26FertilizerPolicyV2",
    "M26WeedRecoveryPolicyV2",
    "MAX_LAYOUT_CANDIDATES_V2",
    "MAX_MARKET_ORDERS",
    "MAX_OBLIGATIONS_V2",
    "MAX_PHASES_V2",
    "MAX_PROJECTS_V2",
    "MAX_PROJECT_INTENTS_V2",
    "MAX_ROUTE_STOPS",
    "MAX_ROUTE_CARD_STOPS_V3",
    "MAX_UNITS",
    "NORMAL_LAYOUT_NAMES",
    "NUM_DAYS",
    "NUM_PRODUCTS",
    "NUM_SHED_ITEMS",
    "NUM_TILES",
    "OFFICIAL_PACKAGE_VERSION",
    "ObligationPriorityV2",
    "PRODUCTS",
    "ProjectStatusV2",
    "ProjectTypeV2",
    "ROUTE_FAMILY_NAMES",
    "RouteFamilyV2",
    "SCHEMA_VERSION_V2",
    "SHED_CAPACITY",
    "STATIC_SPECIES_CAP",
    "TRUTH_BOUNDARY",
    "U_EXACT_UNITS",
]
