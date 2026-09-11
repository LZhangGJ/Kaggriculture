"""M3.6 complete daily-calendar constructors and host validation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import NUM_ANIMALS, NUM_CROPS

from .m36_schema import (
    M36MarketTemplateV3,
    M36ReleasePolicyV3,
    ROUTE_DAYS_V3,
    RouteCalendarV3,
)


def empty_route_calendar_v3(batch_size: int) -> RouteCalendarV3:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return RouteCalendarV3(
        candidate_id=jnp.arange(batch_size, dtype=jnp.int32),
        hand_target_by_day=jnp.zeros((batch_size, ROUTE_DAYS_V3), dtype=jnp.int8),
        crop_target_by_day=jnp.zeros(
            (batch_size, ROUTE_DAYS_V3, NUM_CROPS), dtype=jnp.int16
        ),
        animal_purchase_additions_by_day=jnp.zeros(
            (batch_size, ROUTE_DAYS_V3, NUM_ANIMALS), dtype=jnp.int16
        ),
        animal_service_target_by_day=jnp.zeros(
            (batch_size, ROUTE_DAYS_V3, NUM_ANIMALS), dtype=jnp.int16
        ),
        animal_care_policy_by_day=jnp.zeros(
            (batch_size, ROUTE_DAYS_V3, NUM_ANIMALS), dtype=jnp.int8
        ),
        land_additions_by_day=jnp.zeros(
            (batch_size, ROUTE_DAYS_V3), dtype=jnp.int8
        ),
        feed_stock_target_by_day=jnp.zeros(
            (batch_size, ROUTE_DAYS_V3), dtype=jnp.int16
        ),
        fertilizer_safety_stock_by_day=jnp.zeros(
            (batch_size, ROUTE_DAYS_V3), dtype=jnp.int16
        ),
        market_template_by_day=jnp.full(
            (batch_size, ROUTE_DAYS_V3),
            M36MarketTemplateV3.HIRE_ANIMAL_SEED_FEED,
            dtype=jnp.int8,
        ),
        planned_release_policy_by_day=jnp.full(
            (batch_size, ROUTE_DAYS_V3),
            M36ReleasePolicyV3.LOWEST_FUTURE_NET_VALUE,
            dtype=jnp.int8,
        ),
        event_overflow_count=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


def kawashigi_opening_calendar_v3(batch_size: int = 1) -> RouteCalendarV3:
    """High-level day-0 plan that naturally generates the gold opening.

    This contains no raw Replay action.  It describes business targets: five
    hands, seven wheat plots, twelve melon plots, two cows, two sheep and a
    six-unit feed request.
    """

    calendar = empty_route_calendar_v3(batch_size)
    hand = calendar.hand_target_by_day.at[:, 0].set(jnp.int8(5))
    crop = calendar.crop_target_by_day.at[:, 0].set(
        jnp.asarray((7, 0, 0, 0, 12), dtype=jnp.int16)
    )
    animal_buy = calendar.animal_purchase_additions_by_day.at[:, 0].set(
        jnp.asarray((0, 2, 2), dtype=jnp.int16)
    )
    animal_service = calendar.animal_service_target_by_day.at[:, 0].set(
        jnp.asarray((0, 2, 2), dtype=jnp.int16)
    )
    feed = calendar.feed_stock_target_by_day.at[:, 0].set(jnp.int16(6))
    return calendar._replace(
        hand_target_by_day=hand,
        crop_target_by_day=crop,
        animal_purchase_additions_by_day=animal_buy,
        animal_service_target_by_day=animal_service,
        feed_stock_target_by_day=feed,
    )


def validate_route_calendar_v3(calendar: RouteCalendarV3) -> list[str]:
    errors: list[str] = []
    candidate = np.asarray(calendar.candidate_id)
    if candidate.ndim != 1 or candidate.shape[0] == 0:
        return ["shape:candidate_id"]
    batch_size = candidate.shape[0]
    expected = {
        "hand_target_by_day": (batch_size, ROUTE_DAYS_V3),
        "crop_target_by_day": (batch_size, ROUTE_DAYS_V3, NUM_CROPS),
        "animal_purchase_additions_by_day": (
            batch_size,
            ROUTE_DAYS_V3,
            NUM_ANIMALS,
        ),
        "animal_service_target_by_day": (
            batch_size,
            ROUTE_DAYS_V3,
            NUM_ANIMALS,
        ),
        "animal_care_policy_by_day": (
            batch_size,
            ROUTE_DAYS_V3,
            NUM_ANIMALS,
        ),
        "land_additions_by_day": (batch_size, ROUTE_DAYS_V3),
        "feed_stock_target_by_day": (batch_size, ROUTE_DAYS_V3),
        "fertilizer_safety_stock_by_day": (batch_size, ROUTE_DAYS_V3),
        "market_template_by_day": (batch_size, ROUTE_DAYS_V3),
        "planned_release_policy_by_day": (batch_size, ROUTE_DAYS_V3),
        "event_overflow_count": (batch_size,),
    }
    for name, shape in expected.items():
        if np.asarray(getattr(calendar, name)).shape != shape:
            errors.append(f"shape:{name}")
    if errors:
        return errors
    if np.any((np.asarray(calendar.hand_target_by_day) < 0) | (np.asarray(calendar.hand_target_by_day) > 32)):
        errors.append("range:hand_target_by_day")
    if np.any(np.asarray(calendar.crop_target_by_day) < 0):
        errors.append("range:crop_target_by_day")
    additions = np.asarray(calendar.animal_purchase_additions_by_day)
    service = np.asarray(calendar.animal_service_target_by_day)
    if np.any(additions < 0):
        errors.append("range:animal_purchase_additions_by_day")
    if np.any(service < 0):
        errors.append("range:animal_service_target_by_day")
    care = np.asarray(calendar.animal_care_policy_by_day)
    if np.any((care < 0) | (care > 2)):
        errors.append("range:animal_care_policy_by_day")
    cumulative = np.cumsum(additions, axis=1)
    if np.any(service > cumulative):
        errors.append("semantic:service_exceeds_purchased_commitment")
    if np.any((np.asarray(calendar.land_additions_by_day) < 0) | (np.asarray(calendar.land_additions_by_day) > 3)):
        errors.append("range:land_additions_by_day")
    if np.any(np.sum(np.asarray(calendar.land_additions_by_day), axis=1) > 3):
        errors.append("semantic:land_additions_exceed_three")
    if np.any(np.asarray(calendar.feed_stock_target_by_day) < 0):
        errors.append("range:feed_stock_target_by_day")
    if np.any(np.asarray(calendar.fertilizer_safety_stock_by_day) < 0):
        errors.append("range:fertilizer_safety_stock_by_day")
    if np.any(np.asarray(calendar.event_overflow_count) != 0):
        errors.append("event_overflow_count:nonzero")
    return errors


__all__ = [
    "empty_route_calendar_v3",
    "kawashigi_opening_calendar_v3",
    "validate_route_calendar_v3",
]
