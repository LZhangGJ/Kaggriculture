"""Deterministic host-side sampler for M2.6 crop-only search candidates."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import NUM_CROPS, NUM_PRODUCTS

from .m26_genome import (
    HARVEST_MIN_AGE_LOWER,
    HARVEST_MIN_AGE_UPPER,
    default_m26_crop_genome_v2,
)


def sample_m26_crop_genomes_v2(candidate_count: int, seed: int) -> object:
    """Sample legal, deliberately broad crop genomes for search/smoke use.

    This is not a profitability prior.  It spans phase, crop, layout,
    maintenance, recovery and market branches while leaving primitive legality
    to Full-core.
    """

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    rng = np.random.default_rng(seed)
    phase_count = rng.integers(1, 7, size=candidate_count, dtype=np.int8)
    # Guarantee phase-count branch coverage when the panel is large enough.
    phase_count[: min(candidate_count, 6)] = np.arange(1, min(candidate_count, 6) + 1)
    phase_start = np.full((candidate_count, 6), 720, dtype=np.int16)
    phase_start[:, 0] = 0
    for row, count in enumerate(phase_count.tolist()):
        if count > 1:
            days = np.sort(rng.choice(np.arange(2, 24), size=count - 1, replace=False))
            phase_start[row, 1:count] = days * 24

    target = np.zeros((candidate_count, 6, NUM_CROPS), dtype=np.int16)
    for row, count in enumerate(phase_count.tolist()):
        active_crops = rng.choice(NUM_CROPS, size=int(rng.integers(1, 4)), replace=False)
        current = np.zeros(NUM_CROPS, dtype=np.int16)
        current[active_crops] = rng.integers(2, 11, size=len(active_crops), dtype=np.int16)
        for phase in range(count):
            if phase > 0:
                delta = rng.integers(-5, 7, size=NUM_CROPS, dtype=np.int16)
                current = np.clip(current + delta, 0, 18).astype(np.int16)
                current[rng.random(NUM_CROPS) < 0.18] = 0
                if not np.any(current):
                    current[int(rng.integers(0, NUM_CROPS))] = int(rng.integers(2, 9))
            target[row, phase] = current
    # Force every crop to occur in the first few candidates.
    for crop in range(min(candidate_count, NUM_CROPS)):
        target[crop, 0] = 0
        target[crop, 0, crop] = 4 + crop

    land_target = np.ones((candidate_count, 6), dtype=np.int8)
    hand_target = np.zeros((candidate_count, 6), dtype=np.int8)
    land_start = np.full((candidate_count, 6), 720, dtype=np.int16)
    for row, count in enumerate(phase_count.tolist()):
        land = 1
        for phase in range(count):
            if phase > 0 and rng.random() < 0.35:
                land = min(3, land + 1)
            land_target[row, phase:] = land
            land_start[row, phase] = phase_start[row, phase] + int(rng.integers(0, 2)) * 24
            hand_target[row, phase] = int(rng.integers(3, 13))

    last_plant_days = np.stack(
        [rng.integers(14, 28, size=candidate_count) for _ in range(NUM_CROPS)], axis=1
    ).astype(np.int16)
    last_plant = last_plant_days * 24
    seed_batch = rng.choice(np.asarray((2, 4, 8, 12, 16), dtype=np.int16), size=(candidate_count, NUM_CROPS))
    wave = rng.choice(np.asarray((1, 2, 4, 6, 8), dtype=np.int8), size=(candidate_count, NUM_CROPS))
    harvest_age = np.stack(
        [
            rng.integers(
                int(HARVEST_MIN_AGE_LOWER[crop]),
                int(HARVEST_MIN_AGE_UPPER[crop]) + 1,
                size=candidate_count,
            )
            for crop in range(NUM_CROPS)
        ],
        axis=1,
    ).astype(np.int8)
    harvest_trigger = rng.integers(1, 5, size=(candidate_count, NUM_CROPS), dtype=np.int8)
    fertilizer = rng.integers(0, 3, size=(candidate_count, NUM_CROPS), dtype=np.int8)
    layout = rng.integers(0, 4, size=(candidate_count, NUM_CROPS), dtype=np.int8)
    cash_cap = rng.choice(
        np.asarray((5_000, 10_000, 20_000, 30_000, 45_000), dtype=np.int32),
        size=(candidate_count, NUM_CROPS),
    )
    weed = rng.integers(0, 3, size=(candidate_count, NUM_CROPS), dtype=np.int8)
    abandon = rng.integers(0, 3, size=(candidate_count, NUM_CROPS), dtype=np.int8)
    utilization = rng.choice(
        np.asarray((0.50, 0.65, 0.80, 0.95), dtype=np.float32), size=candidate_count
    )
    deposit = rng.choice(np.asarray((0, 250, 500, 1_000, 2_000), dtype=np.int32), size=candidate_count)
    sell_interval = rng.choice(
        np.asarray((24, 48, 72, 96), dtype=np.int16), size=(candidate_count, NUM_PRODUCTS)
    )
    sell_phase = np.zeros((candidate_count, NUM_PRODUCTS), dtype=np.int16)
    for row in range(candidate_count):
        sell_phase[row] = np.asarray(
            [rng.integers(0, int(interval)) for interval in sell_interval[row]], dtype=np.int16
        )
    price_floor = rng.choice(
        np.asarray((0.0, 0.5, 0.75, 1.0), dtype=np.float32),
        size=(candidate_count, NUM_PRODUCTS),
    )
    sell_fraction = rng.choice(
        np.asarray((0.25, 0.5, 0.75, 1.0), dtype=np.float32),
        size=(candidate_count, NUM_PRODUCTS),
    )

    base = default_m26_crop_genome_v2(candidate_count)
    return base._replace(
        candidate_id=jnp.arange(candidate_count, dtype=jnp.int32),
        phase_count=jnp.asarray(phase_count), phase_start_step=jnp.asarray(phase_start),
        crop_target=jnp.asarray(target), land_target=jnp.asarray(land_target),
        land_start_step=jnp.asarray(land_start), hand_target=jnp.asarray(hand_target),
        crop_last_plant_step=jnp.asarray(last_plant), seed_buy_batch=jnp.asarray(seed_batch),
        plant_wave_size=jnp.asarray(wave), harvest_min_age_days=jnp.asarray(harvest_age),
        harvest_trigger_units=jnp.asarray(harvest_trigger), fertilizer_policy=jnp.asarray(fertilizer),
        crop_layout_policy=jnp.asarray(layout), crop_cash_cap=jnp.asarray(cash_cap),
        weed_recovery_policy=jnp.asarray(weed), crop_abandon_policy=jnp.asarray(abandon),
        maintenance_utilization_cap=jnp.asarray(utilization), deposit_min_value=jnp.asarray(deposit),
        sell_interval=jnp.asarray(sell_interval), sell_phase=jnp.asarray(sell_phase),
        sell_price_floor_ratio=jnp.asarray(price_floor), sell_fraction=jnp.asarray(sell_fraction),
        shed_pressure_trigger=jnp.asarray(
            rng.choice((40, 60, 80, 95), size=candidate_count), dtype=jnp.int16
        ),
        cash_floor=jnp.asarray(
            rng.choice((0, 500, 1_500, 3_000), size=candidate_count), dtype=jnp.int32
        ),
        liquidation_start_step=jnp.asarray(
            rng.choice((24, 25, 26, 27, 28, 29), size=candidate_count) * 24,
            dtype=jnp.int16,
        ),
    )


__all__ = ["sample_m26_crop_genomes_v2"]
