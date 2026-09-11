"""Fixed-shape M3.6 calendar, transaction and decision contracts."""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax


Array = jax.Array
ROUTE_DAYS_V3 = 30
MAX_MARKET_INTENTS_V3 = 10
NUM_STRUCTURE_KINDS_V3 = 2  # coop, pasture


class M36MarketTemplateV3(IntEnum):
    HIRE_ANIMAL_SEED_FEED = 0
    SELL_HIRE_ANIMAL_SEED_FEED = 1
    CRITICAL_INPUT_HIRE_ANIMAL_SELL = 2


class M36BundleStatusV3(IntEnum):
    EMPTY = 0
    PLANNED = 1
    PARTIAL = 2
    FILLED = 3
    FAILED = 4


class M36IntentStatusV3(IntEnum):
    EMPTY = 0
    REQUESTED = 1
    PARTIAL = 2
    FILLED = 3
    REJECTED = 4


class M36IntentFailureV3(IntEnum):
    NONE = 0
    INSUFFICIENT_CASH = 1
    SHED_CAPACITY = 2
    MARKET_SLOT_OVERFLOW = 3
    INVALID_ITEM = 4
    MARKET_CHANGED_OR_UNKNOWN = 5


class M36ReleasePolicyV3(IntEnum):
    LOWEST_FUTURE_NET_VALUE = 0
    FARTHEST_FROM_SERVICE_ROUTE = 1
    LOWEST_PENDING_VALUE = 2


class RouteCalendarV3(NamedTuple):
    """Complete executor-facing 30-day plan.

    Search genomes may remain low dimensional, but they must compile into this
    complete calendar without truncation before execution.
    """

    candidate_id: Array
    hand_target_by_day: Array
    crop_target_by_day: Array
    animal_purchase_additions_by_day: Array
    animal_service_target_by_day: Array
    animal_care_policy_by_day: Array
    land_additions_by_day: Array
    feed_stock_target_by_day: Array
    fertilizer_safety_stock_by_day: Array
    market_template_by_day: Array
    planned_release_policy_by_day: Array
    event_overflow_count: Array


class CommitmentBundleV3(NamedTuple):
    """One same-turn commitment plus its future placement obligations."""

    bundle_id: Array
    start_step: Array
    unit_unlock_requested: Array
    projected_new_structures: Array
    market_op: Array
    market_item: Array
    requested_quantity: Array
    filled_quantity: Array
    post_step_quantity: Array
    intent_status: Array
    failure_code: Array
    project_id: Array
    priority: Array
    market_count: Array
    reserved_cash: Array
    reserved_shed_capacity: Array
    reserved_market_slots: Array
    pending_animals: Array
    pending_structures: Array
    pending_feed: Array
    future_place_plan: Array
    status: Array
    overflow_count: Array


class M36TransactionDiagnosticsV3(NamedTuple):
    invalid_unit_unlock_count: Array
    market_slot_overflow_count: Array
    missing_future_place_plan_count: Array
    unclassified_partial_fill_count: Array
    hard_error_count: Array


class M36PlayerDecisionV3(NamedTuple):
    unit_op: Array
    unit_item: Array
    unit_amount: Array
    unit_count: Array
    market_op: Array
    market_item: Array
    market_amount: Array
    market_count: Array
    bundle: CommitmentBundleV3
    diagnostics: M36TransactionDiagnosticsV3


class M36SchedulerDiagnosticsV3(NamedTuple):
    sticky_task_count: Array
    crop_candidate_count: Array
    animal_candidate_count: Array
    hard_candidate_count: Array
    selected_crop_count: Array
    selected_animal_count: Array
    deadline_preemption_count: Array
    duplicate_reservation_prevented: Array
    local_swap_count: Array
    planned_release_tile_count: Array


class M36FertilizerDiagnosticsV3(NamedTuple):
    held_units: Array
    reserved_units: Array
    surplus_units: Array
    sell_units: Array
    market_slot_overflow_count: Array


__all__ = [
    "CommitmentBundleV3",
    "MAX_MARKET_INTENTS_V3",
    "M36BundleStatusV3",
    "M36IntentFailureV3",
    "M36IntentStatusV3",
    "M36MarketTemplateV3",
    "M36PlayerDecisionV3",
    "M36FertilizerDiagnosticsV3",
    "M36ReleasePolicyV3",
    "M36SchedulerDiagnosticsV3",
    "M36TransactionDiagnosticsV3",
    "NUM_STRUCTURE_KINDS_V3",
    "ROUTE_DAYS_V3",
    "RouteCalendarV3",
]
