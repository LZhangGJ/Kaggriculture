"""Frozen M3A/M3B animal genome constructors and host validation."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_MAX_HELD,
    EPISODE_STEPS,
    NUM_ANIMALS,
    NUM_PRODUCTS,
)

from .constants import MAX_PHASES_V2
from .m3_constants import M3FeedSourcePolicyV2
from .m3_schema import M3AnimalGenomeV2


FIRST_CYCLE_CARE_MAX = np.asarray((3, 5, 5), dtype=np.int8)
STEADY_CYCLE_CARE_MAX = np.asarray((1, 2, 3), dtype=np.int8)
ANIMAL_HARVEST_MAX = np.asarray(ANIMAL_MAX_HELD, dtype=np.int8)


def default_m3_animal_genome_v2(batch_size: int) -> M3AnimalGenomeV2:
    """Return a conservative three-stage BUY_ONLY mixed-animal route."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    def vector(value, dtype):
        return jnp.full((batch_size,), value, dtype=dtype)

    def rows(values, dtype):
        value = jnp.asarray(values, dtype=dtype)
        return jnp.broadcast_to(value, (batch_size,) + value.shape)

    targets = np.asarray(
        (
            (2, 0, 0),
            (4, 2, 0),
            (4, 2, 2),
            (4, 2, 2),
            (4, 2, 2),
            (4, 2, 2),
        ),
        dtype=np.int16,
    )
    return M3AnimalGenomeV2(
        candidate_id=jnp.arange(batch_size, dtype=jnp.int32),
        phase_count=vector(3, jnp.int8),
        phase_start_step=rows((0, 4 * 24, 10 * 24, 720, 720, 720), jnp.int16),
        animal_target=rows(targets, jnp.int16),
        land_target=rows((1, 1, 2, 2, 2, 2), jnp.int8),
        hand_target=rows((4, 7, 9, 0, 0, 0), jnp.int8),
        animal_investment_stop_step=rows((24 * 24, 19 * 24, 21 * 24), jnp.int16),
        animal_layout_policy=rows((0, 2, 3), jnp.int8),
        animal_place_wave_size=rows((2, 2, 2), jnp.int8),
        animal_project_cash_cap=rows((12_000, 12_000, 12_000), jnp.int32),
        feed_source_policy=vector(M3FeedSourcePolicyV2.BUY_ONLY, jnp.int8),
        feed_stock_horizon_days=vector(2, jnp.int8),
        care_policy=rows((2, 2, 2), jnp.int8),
        first_cycle_care_bonus_target=rows((3, 5, 5), jnp.int8),
        steady_cycle_care_bonus_target=rows((1, 2, 3), jnp.int8),
        animal_harvest_trigger_units=rows((3, 5, 5), jnp.int8),
        animal_fertilizer_policy=vector(1, jnp.int8),
        maintenance_utilization_cap=vector(0.80, jnp.float32),
        enforce_productive_cap=vector(True, jnp.bool_),
        deposit_min_value=vector(1, jnp.int32),
        sell_interval=rows((24,) * NUM_PRODUCTS, jnp.int16),
        sell_phase=rows((0,) * NUM_PRODUCTS, jnp.int16),
        sell_price_floor_ratio=rows((0.0,) * NUM_PRODUCTS, jnp.float32),
        sell_fraction=rows((1.0,) * NUM_PRODUCTS, jnp.float32),
        shed_pressure_trigger=vector(80, jnp.int16),
        cash_floor=vector(300, jnp.int32),
        liquidation_start_step=vector(28 * 24, jnp.int16),
    )


def validate_m3_animal_genome_v2(genome: M3AnimalGenomeV2) -> list[str]:
    """Return deterministic M3 contract violations without mutating candidates."""

    errors: list[str] = []
    batch_size = int(np.asarray(genome.candidate_id).shape[0])
    expected = {
        "phase_count": (batch_size,),
        "phase_start_step": (batch_size, MAX_PHASES_V2),
        "animal_target": (batch_size, MAX_PHASES_V2, NUM_ANIMALS),
        "land_target": (batch_size, MAX_PHASES_V2),
        "hand_target": (batch_size, MAX_PHASES_V2),
        "animal_investment_stop_step": (batch_size, NUM_ANIMALS),
        "animal_layout_policy": (batch_size, NUM_ANIMALS),
        "animal_place_wave_size": (batch_size, NUM_ANIMALS),
        "animal_project_cash_cap": (batch_size, NUM_ANIMALS),
        "care_policy": (batch_size, NUM_ANIMALS),
        "first_cycle_care_bonus_target": (batch_size, NUM_ANIMALS),
        "steady_cycle_care_bonus_target": (batch_size, NUM_ANIMALS),
        "animal_harvest_trigger_units": (batch_size, NUM_ANIMALS),
        "sell_interval": (batch_size, NUM_PRODUCTS),
        "sell_phase": (batch_size, NUM_PRODUCTS),
        "sell_price_floor_ratio": (batch_size, NUM_PRODUCTS),
        "sell_fraction": (batch_size, NUM_PRODUCTS),
        "enforce_productive_cap": (batch_size,),
    }
    for field, shape in expected.items():
        if np.asarray(getattr(genome, field)).shape != shape:
            errors.append(f"shape:{field}")
    if errors:
        return errors

    phase_count = np.asarray(genome.phase_count)
    phase_start = np.asarray(genome.phase_start_step)
    active = np.arange(MAX_PHASES_V2)[None] < phase_count[:, None]
    active_pairs = np.arange(MAX_PHASES_V2 - 1)[None] < (phase_count[:, None] - 1)
    if np.any((phase_count < 1) | (phase_count > MAX_PHASES_V2)):
        errors.append("range:phase_count")
    if np.any(phase_start[:, 0] != 0):
        errors.append("phase:first_step_not_zero")
    if np.any((phase_start % 24 != 0) & active):
        errors.append("phase:not_day_boundary")
    if np.any((np.diff(phase_start, axis=1) <= 0) & active_pairs):
        errors.append("phase:not_strictly_increasing")

    target = np.asarray(genome.animal_target)
    if np.any((target < 0) | (target > 100)):
        errors.append("range:animal_target")
    if np.any((np.diff(target, axis=1) < 0) & active_pairs[..., None]):
        errors.append("animal_target:not_monotonic")
    if np.any(np.sum(target, axis=-1) > 100):
        errors.append("range:animal_target_total")

    land = np.asarray(genome.land_target)
    if np.any((land < 1) | (land > 4)):
        errors.append("range:land_target")
    if np.any((np.diff(land, axis=1) < 0) & active_pairs):
        errors.append("land_target:not_monotonic")
    hand = np.asarray(genome.hand_target)
    if np.any((hand < 0) | (hand > 32)):
        errors.append("range:hand_target")
    if np.any((np.diff(hand, axis=1) < 0) & active_pairs):
        errors.append("hand_target:not_monotonic")
    stop = np.asarray(genome.animal_investment_stop_step)
    if np.any((stop < 0) | (stop >= EPISODE_STEPS)):
        errors.append("range:animal_investment_stop_step")
    layout = np.asarray(genome.animal_layout_policy)
    if np.any((layout < 0) | (layout > 3)):
        errors.append("range:animal_layout_policy")
    wave = np.asarray(genome.animal_place_wave_size)
    if np.any((wave < 1) | (wave > 8)):
        errors.append("range:animal_place_wave_size")
    if np.any(np.asarray(genome.animal_project_cash_cap) < 0):
        errors.append("range:animal_project_cash_cap")
    if np.any(np.asarray(genome.feed_source_policy) != M3FeedSourcePolicyV2.BUY_ONLY):
        errors.append("milestone:feed_source_policy_not_buy_only")
    horizon = np.asarray(genome.feed_stock_horizon_days)
    if np.any((horizon < 1) | (horizon > 4)):
        errors.append("range:feed_stock_horizon_days")
    care = np.asarray(genome.care_policy)
    if np.any((care < 0) | (care > 2)):
        errors.append("range:care_policy")
    first = np.asarray(genome.first_cycle_care_bonus_target)
    if np.any((first < 0) | (first > FIRST_CYCLE_CARE_MAX[None])):
        errors.append("range:first_cycle_care_bonus_target")
    steady = np.asarray(genome.steady_cycle_care_bonus_target)
    if np.any((steady < 0) | (steady > STEADY_CYCLE_CARE_MAX[None])):
        errors.append("range:steady_cycle_care_bonus_target")
    harvest = np.asarray(genome.animal_harvest_trigger_units)
    if np.any((harvest < 1) | (harvest > ANIMAL_HARVEST_MAX[None])):
        errors.append("range:animal_harvest_trigger_units")
    fertilizer = np.asarray(genome.animal_fertilizer_policy)
    if np.any((fertilizer < 0) | (fertilizer > 2)):
        errors.append("milestone:animal_fertilizer_policy")
    utilization = np.asarray(genome.maintenance_utilization_cap)
    if np.any((utilization <= 0.0) | (utilization > 1.0)):
        errors.append("range:maintenance_utilization_cap")
    enforce_cap = np.asarray(genome.enforce_productive_cap)
    if enforce_cap.dtype.kind != "b":
        errors.append("dtype:enforce_productive_cap")
    if np.any(np.asarray(genome.deposit_min_value) < 0):
        errors.append("range:deposit_min_value")
    if np.any(np.asarray(genome.sell_interval) < 1):
        errors.append("range:sell_interval")
    if np.any(np.asarray(genome.sell_price_floor_ratio) < 0.0):
        errors.append("range:sell_price_floor_ratio")
    fraction = np.asarray(genome.sell_fraction)
    if np.any((fraction <= 0.0) | (fraction > 1.0)):
        errors.append("range:sell_fraction")
    pressure = np.asarray(genome.shed_pressure_trigger)
    if np.any((pressure < 1) | (pressure > 100)):
        errors.append("range:shed_pressure_trigger")
    if np.any(np.asarray(genome.cash_floor) < 0):
        errors.append("range:cash_floor")
    liquidation = np.asarray(genome.liquidation_start_step)
    if np.any((liquidation < 0) | (liquidation >= EPISODE_STEPS)):
        errors.append("range:liquidation_start_step")
    return errors


__all__ = [
    "ANIMAL_HARVEST_MAX",
    "FIRST_CYCLE_CARE_MAX",
    "STEADY_CYCLE_CARE_MAX",
    "default_m3_animal_genome_v2",
    "validate_m3_animal_genome_v2",
]
