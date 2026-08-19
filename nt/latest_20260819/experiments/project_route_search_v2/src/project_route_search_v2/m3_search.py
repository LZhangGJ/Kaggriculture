"""Deterministic structured sampling for the active M3 BUY_ONLY domain."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from kaggriculture_jax.constants import NUM_ANIMALS, NUM_PRODUCTS

from .m3_genome import default_m3_animal_genome_v2
from .m3_schema import M3AnimalGenomeV2


def sample_m3_animal_genomes_v2(count: int, *, seed: int = 0) -> M3AnimalGenomeV2:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    genome = default_m3_animal_genome_v2(count)
    phase_count = rng.integers(1, 7, size=count, dtype=np.int8)
    phase_start = np.full((count, 6), 720, dtype=np.int16)
    targets = np.zeros((count, 6, NUM_ANIMALS), dtype=np.int16)
    land = np.ones((count, 6), dtype=np.int8)
    hands = np.zeros((count, 6), dtype=np.int8)
    for lane in range(count):
        phases = int(phase_count[lane])
        if phases == 1:
            starts = np.asarray((0,), dtype=np.int16)
        else:
            days = np.sort(rng.choice(np.arange(2, 23), size=phases - 1, replace=False))
            starts = np.concatenate((np.asarray((0,)), days * 24)).astype(np.int16)
        phase_start[lane, :phases] = starts
        increments = rng.integers(0, 5, size=(phases, NUM_ANIMALS), dtype=np.int16)
        cumulative = np.minimum(np.cumsum(increments, axis=0), 100)
        targets[lane, :phases] = cumulative
        targets[lane, phases:] = cumulative[-1]
        total_target = targets[lane, :phases].sum(axis=1)
        land[lane, :phases] = np.maximum.accumulate(np.clip((total_target + 24) // 25, 1, 4)).astype(np.int8)
        land[lane, phases:] = land[lane, phases - 1]
        hands[lane, :phases] = np.clip(2 + (total_target + 1) // 2, 2, 14).astype(np.int8)
        hands[lane, phases:] = hands[lane, phases - 1]

    care = rng.integers(0, 3, size=(count, NUM_ANIMALS), dtype=np.int8)
    first_max = np.asarray((3, 5, 5), dtype=np.int8)
    steady_max = np.asarray((1, 2, 3), dtype=np.int8)
    first = np.stack(
        [rng.integers(0, int(m) + 1, size=count, dtype=np.int8) for m in first_max], axis=1
    )
    steady = np.stack(
        [rng.integers(0, int(m) + 1, size=count, dtype=np.int8) for m in steady_max], axis=1
    )
    first = np.where(care == 0, 0, first).astype(np.int8)
    steady = np.where(care == 2, steady, 0).astype(np.int8)
    harvest_max = np.asarray((4, 6, 6), dtype=np.int8)
    harvest = np.stack(
        [rng.integers(1, int(m) + 1, size=count, dtype=np.int8) for m in harvest_max], axis=1
    )
    sell_interval = rng.choice(np.asarray((12, 24, 48, 72), dtype=np.int16), size=(count, NUM_PRODUCTS))
    return genome._replace(
        candidate_id=jnp.arange(count, dtype=jnp.int32),
        phase_count=jnp.asarray(phase_count),
        phase_start_step=jnp.asarray(phase_start),
        animal_target=jnp.asarray(targets),
        land_target=jnp.asarray(land),
        hand_target=jnp.asarray(hands),
        animal_investment_stop_step=jnp.asarray(
            rng.integers(17 * 24, 25 * 24, size=(count, NUM_ANIMALS), dtype=np.int16)
        ),
        animal_layout_policy=jnp.asarray(
            rng.integers(0, 4, size=(count, NUM_ANIMALS), dtype=np.int8)
        ),
        animal_place_wave_size=jnp.asarray(
            rng.integers(1, 9, size=(count, NUM_ANIMALS), dtype=np.int8)
        ),
        feed_stock_horizon_days=jnp.asarray(rng.integers(1, 5, size=count, dtype=np.int8)),
        care_policy=jnp.asarray(care),
        first_cycle_care_bonus_target=jnp.asarray(first),
        steady_cycle_care_bonus_target=jnp.asarray(steady),
        animal_harvest_trigger_units=jnp.asarray(harvest),
        animal_fertilizer_policy=jnp.asarray(rng.integers(0, 3, size=count, dtype=np.int8)),
        maintenance_utilization_cap=jnp.asarray(
            rng.uniform(0.55, 0.95, size=count), dtype=jnp.float32
        ),
        sell_interval=jnp.asarray(sell_interval),
        sell_phase=jnp.asarray(rng.integers(0, 12, size=(count, NUM_PRODUCTS), dtype=np.int16)),
        sell_price_floor_ratio=jnp.asarray(rng.uniform(0.0, 1.1, size=(count, NUM_PRODUCTS)), dtype=jnp.float32),
        sell_fraction=jnp.asarray(rng.uniform(0.25, 1.0, size=(count, NUM_PRODUCTS)), dtype=jnp.float32),
        shed_pressure_trigger=jnp.asarray(rng.integers(55, 96, size=count, dtype=np.int16)),
        cash_floor=jnp.asarray(rng.integers(0, 1001, size=count, dtype=np.int32)),
        liquidation_start_step=jnp.asarray(rng.integers(25 * 24, 29 * 24, size=count, dtype=np.int16)),
    )


__all__ = ["sample_m3_animal_genomes_v2"]
