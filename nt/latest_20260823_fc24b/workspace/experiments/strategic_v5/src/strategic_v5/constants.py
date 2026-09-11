"""Stable IDs for the V5 full-farm strategic controller."""

from __future__ import annotations

from enum import IntEnum


SCHEMA_VERSION_V1 = 1
MAX_CANDIDATES_V1 = 96
MAX_SELECTIONS_V1 = 8


class TaskTypeV1(IntEnum):
    """The complete high-level task family frozen by V5 section 6.1."""

    NONE = 0
    IDLE_OR_PASS = 1
    SAFE_RECOVERY = 2
    CROP_PRODUCTION = 3
    WATER_CROP = 4
    CLEAR_OR_REMOVE_TILE = 5
    BUILD_ANIMAL_STRUCTURE = 6
    ANIMAL_PURCHASE = 7
    ANIMAL_PLACE = 8
    ANIMAL_FEED = 9
    ANIMAL_CARE = 10
    ANIMAL_COLLECT_PRODUCT = 11
    ANIMAL_COLLECT_FERTILIZER = 12
    BUY_LAND = 13
    BUY_PRODUCT = 14
    APPLY_FERTILIZER = 15
    HIRE_WORKER = 16
    SHED_PICKUP = 17
    SHED_DEPOSIT = 18
    SELL_INVENTORY = 19
    TERMINAL_LIQUIDATION = 20


class TaskPhaseV1(IntEnum):
    NONE = 0
    MOVE_TO_SHED = 1
    PICKUP = 2
    MOVE_TO_TARGET = 3
    OPERATE = 4
    MOVE_TO_DEPOT = 5
    DEPOSIT = 6
    MARKET = 7


class TaskStatusV1(IntEnum):
    EMPTY = 0
    ACTIVE = 1
    DONE = 2
    FAILED = 3
    CANCELLED = 4


class FailureCodeV1(IntEnum):
    NONE = 0
    OWNER_INACTIVE = 1
    TARGET_INVALID = 2
    DEADLINE_MISSED = 3
    EFFECT_MISMATCH = 4
    RESOURCE_UNAVAILABLE = 5
    MARKET_SLOT_UNAVAILABLE = 6
    TERMINAL_UNBANKABLE = 7
    UNSUPPORTED_STATE = 8


class CandidateSourceV1(IntEnum):
    NONE = 0
    LAND = 1
    FERTILIZER_BUY = 2
    FERTILIZER_APPLY = 3
    CROP = 4
    ANIMAL = 5
    MAINTENANCE = 6
    INVENTORY = 7
    TERMINAL = 8
