"""M3-local animal genome enums.

These identifiers intentionally live outside the M0 frozen constants module so
new milestone work cannot invalidate the M0 contract receipt.
"""

from __future__ import annotations

from enum import IntEnum


class M3FeedSourcePolicyV2(IntEnum):
    """Only BUY_ONLY is active before M3.5."""

    BUY_ONLY = 0
    GROW_ONLY = 1
    HYBRID = 2


class M3AnimalCarePolicyV2(IntEnum):
    OFF = 0
    FIRST_CYCLE_ONLY = 1
    CAPACITY_AWARE_EVERY_CYCLE = 2


class M3AnimalFertilizerPolicyV2(IntEnum):
    IGNORE = 0
    COLLECT_WHEN_VISITING_AND_SELL = 1
    ACTIVE_COLLECT_AND_SELL = 2
    RESERVE_FOR_CROPS = 3


class M3AnimalLayoutPolicyV2(IntEnum):
    CENTER_COMPACT = 0
    CLUSTER_EXPANSION = 1
    QUADRANT_ZONED = 2
    ROUTE_STRIP = 3


__all__ = [
    "M3AnimalCarePolicyV2",
    "M3AnimalFertilizerPolicyV2",
    "M3AnimalLayoutPolicyV2",
    "M3FeedSourcePolicyV2",
]
