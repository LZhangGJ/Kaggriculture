"""Frozen M2.6 five-crop genome constructors and host-side validation."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from kaggriculture_jax.constants import EPISODE_STEPS, NUM_CROPS, NUM_PRODUCTS

from .constants import MAX_PHASES_V2
from .schema import M26CropGenomeV2


HARVEST_MIN_AGE_LOWER = np.asarray((2, 2, 8, 10, 10), dtype=np.int16)
HARVEST_MIN_AGE_UPPER = np.asarray((4, 3, 11, 16, 12), dtype=np.int16)


def default_m26_crop_genome_v2(batch_size: int) -> M26CropGenomeV2:
    """Return a generalized candidate-101-like crop genome.

    The returned object contains no primary/support crop aliases.  WHEAT and
    TOMATO targets are represented in the same ``[batch, phase, crop]`` tensor
    used by every other crop.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    def vector(value, dtype):
        return jnp.full((batch_size,), value, dtype=dtype)

    def rows(values, dtype):
        values = jnp.asarray(values, dtype=dtype)
        return jnp.broadcast_to(values, (batch_size,) + values.shape)

    crop_target = np.zeros((MAX_PHASES_V2, NUM_CROPS), dtype=np.int16)
    crop_target[0] = (0, 0, 12, 0, 0)
    crop_target[1] = (8, 0, 18, 0, 0)
    crop_target[2] = (6, 0, 28, 0, 0)

    return M26CropGenomeV2(
        candidate_id=jnp.arange(batch_size, dtype=jnp.int32),
        phase_count=vector(3, jnp.int8),
        phase_start_step=rows((0, 5 * 24, 14 * 24, 720, 720, 720), jnp.int16),
        crop_target=rows(crop_target, jnp.int16),
        land_target=rows((1, 2, 2, 2, 2, 2), jnp.int8),
        land_start_step=rows((0, 6 * 24, 14 * 24, 720, 720, 720), jnp.int16),
        hand_target=rows((9, 11, 7, 0, 0, 0), jnp.int8),
        crop_last_plant_step=rows((24 * 24, 25 * 24, 19 * 24, 15 * 24, 17 * 24), jnp.int16),
        seed_buy_batch=rows((8, 8, 16, 8, 8), jnp.int16),
        plant_wave_size=rows((4, 4, 4, 4, 4), jnp.int8),
        harvest_min_age_days=rows(HARVEST_MIN_AGE_LOWER, jnp.int8),
        harvest_trigger_units=rows((1, 1, 2, 2, 1), jnp.int8),
        fertilizer_policy=rows((0, 0, 0, 0, 0), jnp.int8),
        crop_layout_policy=rows((0, 1, 0, 2, 3), jnp.int8),
        crop_cash_cap=rows((10_000, 10_000, 25_000, 25_000, 25_000), jnp.int32),
        weed_recovery_policy=rows((2, 2, 2, 2, 2), jnp.int8),
        crop_abandon_policy=rows((0, 0, 0, 0, 0), jnp.int8),
        maintenance_utilization_cap=vector(0.80, jnp.float32),
        deposit_min_value=vector(500, jnp.int32),
        sell_interval=rows((48,) * NUM_PRODUCTS, jnp.int16),
        sell_phase=rows((0,) * NUM_PRODUCTS, jnp.int16),
        sell_price_floor_ratio=rows((0.0,) * NUM_PRODUCTS, jnp.float32),
        sell_fraction=rows((1.0,) * NUM_PRODUCTS, jnp.float32),
        shed_pressure_trigger=vector(80, jnp.int16),
        cash_floor=vector(500, jnp.int32),
        liquidation_start_step=vector(28 * 24, jnp.int16),
    )


def validate_m26_crop_genome_v2(genome: M26CropGenomeV2) -> list[str]:
    """Return deterministic host-side contract violations.

    This validator is intentionally outside JIT.  It protects candidate
    generation and acceptance tools; the controller still uses legal-action
    masks and admission control at runtime.
    """

    errors: list[str] = []
    batch_size = int(np.asarray(genome.candidate_id).shape[0])

    expected = {
        "phase_count": (batch_size,),
        "phase_start_step": (batch_size, MAX_PHASES_V2),
        "crop_target": (batch_size, MAX_PHASES_V2, NUM_CROPS),
        "land_target": (batch_size, MAX_PHASES_V2),
        "land_start_step": (batch_size, MAX_PHASES_V2),
        "hand_target": (batch_size, MAX_PHASES_V2),
        "crop_last_plant_step": (batch_size, NUM_CROPS),
        "seed_buy_batch": (batch_size, NUM_CROPS),
        "plant_wave_size": (batch_size, NUM_CROPS),
        "harvest_min_age_days": (batch_size, NUM_CROPS),
        "harvest_trigger_units": (batch_size, NUM_CROPS),
        "fertilizer_policy": (batch_size, NUM_CROPS),
        "crop_layout_policy": (batch_size, NUM_CROPS),
        "crop_cash_cap": (batch_size, NUM_CROPS),
        "weed_recovery_policy": (batch_size, NUM_CROPS),
        "crop_abandon_policy": (batch_size, NUM_CROPS),
        "sell_interval": (batch_size, NUM_PRODUCTS),
        "sell_phase": (batch_size, NUM_PRODUCTS),
        "sell_price_floor_ratio": (batch_size, NUM_PRODUCTS),
        "sell_fraction": (batch_size, NUM_PRODUCTS),
    }
    for field, shape in expected.items():
        if np.asarray(getattr(genome, field)).shape != shape:
            errors.append(f"shape:{field}")

    if errors:
        return errors

    phase_count = np.asarray(genome.phase_count)
    phase_start = np.asarray(genome.phase_start_step)
    active_phase = np.arange(MAX_PHASES_V2)[None, :] < phase_count[:, None]
    if np.any((phase_count < 1) | (phase_count > MAX_PHASES_V2)):
        errors.append("range:phase_count")
    if np.any(phase_start[:, 0] != 0):
        errors.append("phase:first_step_not_zero")
    if np.any((phase_start % 24 != 0) & active_phase):
        errors.append("phase:not_day_boundary")
    active_pairs = np.arange(MAX_PHASES_V2 - 1)[None, :] < (phase_count[:, None] - 1)
    if np.any((np.diff(phase_start, axis=1) <= 0) & active_pairs):
        errors.append("phase:not_strictly_increasing")
    if np.any(np.asarray(genome.crop_target) < 0):
        errors.append("range:crop_target")
    if np.any((np.asarray(genome.land_target) < 1) | (np.asarray(genome.land_target) > 4)):
        errors.append("range:land_target")
    if np.any((np.asarray(genome.hand_target) < 0) | (np.asarray(genome.hand_target) > 32)):
        errors.append("range:hand_target")
    if np.any(np.asarray(genome.crop_last_plant_step) >= EPISODE_STEPS):
        errors.append("range:crop_last_plant_step")
    harvest_age = np.asarray(genome.harvest_min_age_days)
    if np.any(harvest_age < HARVEST_MIN_AGE_LOWER[None, :]) or np.any(
        harvest_age > HARVEST_MIN_AGE_UPPER[None, :]
    ):
        errors.append("range:harvest_min_age_days")
    if np.any(np.asarray(genome.seed_buy_batch) < 1):
        errors.append("range:seed_buy_batch")
    if np.any(np.asarray(genome.plant_wave_size) < 1):
        errors.append("range:plant_wave_size")
    if np.any(np.asarray(genome.harvest_trigger_units) < 1):
        errors.append("range:harvest_trigger_units")
    if np.any((np.asarray(genome.fertilizer_policy) < 0) | (np.asarray(genome.fertilizer_policy) > 2)):
        errors.append("range:fertilizer_policy")
    if np.any((np.asarray(genome.crop_layout_policy) < 0) | (np.asarray(genome.crop_layout_policy) > 3)):
        errors.append("range:crop_layout_policy")
    if np.any((np.asarray(genome.weed_recovery_policy) < 0) | (np.asarray(genome.weed_recovery_policy) > 2)):
        errors.append("range:weed_recovery_policy")
    if np.any((np.asarray(genome.crop_abandon_policy) < 0) | (np.asarray(genome.crop_abandon_policy) > 2)):
        errors.append("range:crop_abandon_policy")
    if np.any((np.asarray(genome.maintenance_utilization_cap) <= 0.0) | (np.asarray(genome.maintenance_utilization_cap) > 1.0)):
        errors.append("range:maintenance_utilization_cap")
    if np.any(np.asarray(genome.deposit_min_value) < 0):
        errors.append("range:deposit_min_value")
    if np.any(np.asarray(genome.sell_interval) < 1):
        errors.append("range:sell_interval")
    if np.any(np.asarray(genome.sell_price_floor_ratio) < 0.0):
        errors.append("range:sell_price_floor_ratio")
    sell_fraction = np.asarray(genome.sell_fraction)
    if np.any((sell_fraction <= 0.0) | (sell_fraction > 1.0)):
        errors.append("range:sell_fraction")
    if np.any((np.asarray(genome.shed_pressure_trigger) < 1) | (np.asarray(genome.shed_pressure_trigger) > 100)):
        errors.append("range:shed_pressure_trigger")
    if np.any(np.asarray(genome.cash_floor) < 0):
        errors.append("range:cash_floor")
    if np.any((np.asarray(genome.liquidation_start_step) < 0) | (np.asarray(genome.liquidation_start_step) >= EPISODE_STEPS)):
        errors.append("range:liquidation_start_step")
    return errors


__all__ = [
    "HARVEST_MIN_AGE_LOWER",
    "HARVEST_MIN_AGE_UPPER",
    "default_m26_crop_genome_v2",
    "validate_m26_crop_genome_v2",
]
