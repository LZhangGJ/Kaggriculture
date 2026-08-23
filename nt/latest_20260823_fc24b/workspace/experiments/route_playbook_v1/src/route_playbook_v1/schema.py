"""Fixed-shape per-environment route schedules for GPU search."""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_DAYS,
    NUM_PRODUCTS,
)


class RouteFamilyV1(IntEnum):
    FAST_CROP = 0
    TOMATO = 1
    STRAWBERRY = 2
    MELON = 3
    EGG = 4
    MILK = 5
    WOOL = 6
    HYBRID = 7


class FertilizerPolicyV1(IntEnum):
    SELL = 0
    APPLY = 1
    MIXED = 2


class RouteScheduleV1(NamedTuple):
    """A dense 30-day program; the leading dimension is simulation batch."""

    route_id: jax.Array
    family_id: jax.Array
    enabled: jax.Array
    crop_target_by_day: jax.Array
    animal_target_by_day: jax.Array
    land_target_by_day: jax.Array
    hire_target_by_day: jax.Array
    hire_batch_max: jax.Array
    seed_batch: jax.Array
    feed_reserve_days: jax.Array
    feed_sell_reserve_days: jax.Array
    fertilizer_policy: jax.Array
    parallel_plant_lanes: jax.Array
    parallel_plant_min_maintenance_code: jax.Array
    harvest_dispatch_lanes: jax.Array
    harvest_dispatch_start_step: jax.Array
    deposit_batch_units: jax.Array
    crop_harvest_batch_units: jax.Array
    chain_care_after_collection: jax.Array
    chain_animal_service_after_action: jax.Array
    chain_crop_harvest_after_action: jax.Array
    financing_cash_floor: jax.Array
    sell_interval: jax.Array
    sell_phase: jax.Array
    sell_price_floor: jax.Array
    investment_stop_step: jax.Array
    liquidation_start_step: jax.Array


def empty_route_schedule_v1(batch_size: int) -> RouteScheduleV1:
    """Return shape-stable disabled schedules for tests and null controls."""

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return RouteScheduleV1(
        route_id=jnp.arange(batch_size, dtype=jnp.int32),
        family_id=jnp.zeros((batch_size,), dtype=jnp.int8),
        enabled=jnp.zeros((batch_size,), dtype=jnp.bool_),
        crop_target_by_day=jnp.zeros(
            (batch_size, NUM_DAYS, NUM_CROPS), dtype=jnp.int8
        ),
        animal_target_by_day=jnp.zeros(
            (batch_size, NUM_DAYS, NUM_ANIMALS), dtype=jnp.int8
        ),
        land_target_by_day=jnp.ones((batch_size, NUM_DAYS), dtype=jnp.int8),
        hire_target_by_day=jnp.zeros((batch_size, NUM_DAYS), dtype=jnp.int8),
        hire_batch_max=jnp.ones((batch_size,), dtype=jnp.int8),
        seed_batch=jnp.ones((batch_size, NUM_CROPS), dtype=jnp.int8),
        feed_reserve_days=jnp.ones((batch_size,), dtype=jnp.int8),
        feed_sell_reserve_days=jnp.zeros((batch_size,), dtype=jnp.int8),
        fertilizer_policy=jnp.full(
            (batch_size,), FertilizerPolicyV1.SELL, dtype=jnp.int8
        ),
        parallel_plant_lanes=jnp.zeros((batch_size,), dtype=jnp.int8),
        parallel_plant_min_maintenance_code=jnp.full(
            (batch_size,), 5, dtype=jnp.int8
        ),
        harvest_dispatch_lanes=jnp.zeros((batch_size,), dtype=jnp.int8),
        harvest_dispatch_start_step=jnp.zeros((batch_size,), dtype=jnp.int16),
        deposit_batch_units=jnp.ones((batch_size,), dtype=jnp.int16),
        crop_harvest_batch_units=jnp.ones((batch_size,), dtype=jnp.int16),
        chain_care_after_collection=jnp.zeros(
            (batch_size,), dtype=jnp.bool_
        ),
        chain_animal_service_after_action=jnp.zeros(
            (batch_size,), dtype=jnp.bool_
        ),
        chain_crop_harvest_after_action=jnp.zeros(
            (batch_size,), dtype=jnp.bool_
        ),
        financing_cash_floor=jnp.zeros((batch_size,), dtype=jnp.int32),
        sell_interval=jnp.full(
            (batch_size, NUM_PRODUCTS), 24, dtype=jnp.int16
        ),
        sell_phase=jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int16),
        sell_price_floor=jnp.ones(
            (batch_size, NUM_PRODUCTS), dtype=jnp.int16
        ),
        investment_stop_step=jnp.full((batch_size,), 624, dtype=jnp.int16),
        liquidation_start_step=jnp.full((batch_size,), 680, dtype=jnp.int16),
    )


def validate_route_schedule_v1(schedule: RouteScheduleV1) -> int:
    """Validate host-visible shapes and return the common batch size."""

    batch_size = int(schedule.route_id.shape[0])
    expected = (
        ("route_id", schedule.route_id, (batch_size,)),
        ("family_id", schedule.family_id, (batch_size,)),
        ("enabled", schedule.enabled, (batch_size,)),
        (
            "crop_target_by_day",
            schedule.crop_target_by_day,
            (batch_size, NUM_DAYS, NUM_CROPS),
        ),
        (
            "animal_target_by_day",
            schedule.animal_target_by_day,
            (batch_size, NUM_DAYS, NUM_ANIMALS),
        ),
        ("land_target_by_day", schedule.land_target_by_day, (batch_size, NUM_DAYS)),
        ("hire_target_by_day", schedule.hire_target_by_day, (batch_size, NUM_DAYS)),
        ("hire_batch_max", schedule.hire_batch_max, (batch_size,)),
        ("seed_batch", schedule.seed_batch, (batch_size, NUM_CROPS)),
        ("feed_reserve_days", schedule.feed_reserve_days, (batch_size,)),
        (
            "feed_sell_reserve_days",
            schedule.feed_sell_reserve_days,
            (batch_size,),
        ),
        ("fertilizer_policy", schedule.fertilizer_policy, (batch_size,)),
        ("parallel_plant_lanes", schedule.parallel_plant_lanes, (batch_size,)),
        (
            "parallel_plant_min_maintenance_code",
            schedule.parallel_plant_min_maintenance_code,
            (batch_size,),
        ),
        ("harvest_dispatch_lanes", schedule.harvest_dispatch_lanes, (batch_size,)),
        (
            "harvest_dispatch_start_step",
            schedule.harvest_dispatch_start_step,
            (batch_size,),
        ),
        ("deposit_batch_units", schedule.deposit_batch_units, (batch_size,)),
        (
            "crop_harvest_batch_units",
            schedule.crop_harvest_batch_units,
            (batch_size,),
        ),
        (
            "chain_care_after_collection",
            schedule.chain_care_after_collection,
            (batch_size,),
        ),
        (
            "chain_animal_service_after_action",
            schedule.chain_animal_service_after_action,
            (batch_size,),
        ),
        (
            "chain_crop_harvest_after_action",
            schedule.chain_crop_harvest_after_action,
            (batch_size,),
        ),
        ("financing_cash_floor", schedule.financing_cash_floor, (batch_size,)),
        ("sell_interval", schedule.sell_interval, (batch_size, NUM_PRODUCTS)),
        ("sell_phase", schedule.sell_phase, (batch_size, NUM_PRODUCTS)),
        ("sell_price_floor", schedule.sell_price_floor, (batch_size, NUM_PRODUCTS)),
        ("investment_stop_step", schedule.investment_stop_step, (batch_size,)),
        ("liquidation_start_step", schedule.liquidation_start_step, (batch_size,)),
    )
    for name, value, wanted in expected:
        if tuple(value.shape) != tuple(wanted):
            raise ValueError(f"{name} shape {tuple(value.shape)} != {tuple(wanted)}")
    if batch_size <= 0:
        raise ValueError("route schedule batch must be positive")
    return batch_size
