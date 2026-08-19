"""Structured M3.5 farm candidates for random full-season safety tests."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .constants import M26FertilizerPolicyV2
from .m3_constants import M3AnimalFertilizerPolicyV2, M3FeedSourcePolicyV2
from .m35_genome import default_m35_farm_genome_v2


def sample_m35_farm_genomes_v2(count: int, *, seed: int) -> object:
    """Sample conservative but behaviorally varied crop/animal commitments."""

    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    genome = default_m35_farm_genome_v2(count)

    crop_target = np.zeros((count, 6, 5), dtype=np.int16)
    wheat0 = rng.integers(6, 9, size=count, dtype=np.int16)
    tomato0 = rng.integers(3, 6, size=count, dtype=np.int16)
    wheat1 = wheat0 + rng.integers(0, 2, size=count, dtype=np.int16)
    tomato1 = tomato0 + rng.integers(0, 2, size=count, dtype=np.int16)
    wheat2 = wheat1 + rng.integers(0, 2, size=count, dtype=np.int16)
    tomato2 = tomato1 + rng.integers(0, 2, size=count, dtype=np.int16)
    crop_target[:, 0, 0] = wheat0
    crop_target[:, 0, 2] = tomato0
    crop_target[:, 1, 0] = wheat1
    crop_target[:, 1, 2] = tomato1
    crop_target[:, 2, 0] = wheat2
    crop_target[:, 2, 2] = tomato2
    crop_target[:, 3:, 0] = wheat2[:, None]
    crop_target[:, 3:, 2] = tomato2[:, None]

    animal_target = np.zeros((count, 6, 3), dtype=np.int16)
    goose1 = rng.integers(1, 3, size=count, dtype=np.int16)
    animal_target[:, 1, 0] = goose1
    animal_target[:, 2:, 0] = goose1[:, None] + rng.integers(
        0, 2, size=(count, 1), dtype=np.int16
    )
    animal_target[:, 2:, 1] = rng.integers(
        0, 2, size=(count, 1), dtype=np.int16
    )
    animal_target[:, 2:, 2] = rng.integers(
        0, 2, size=(count, 1), dtype=np.int16
    )

    feed = rng.integers(
        M3FeedSourcePolicyV2.BUY_ONLY,
        M3FeedSourcePolicyV2.HYBRID + 1,
        size=count,
        dtype=np.int8,
    )
    recycle = rng.integers(0, 2, size=count, dtype=np.int8).astype(bool)
    fertilizer_policy = np.asarray(genome.crop.fertilizer_policy).copy()
    fertilizer_policy[:, 2] = np.where(
        recycle,
        M26FertilizerPolicyV2.ALWAYS_WHEN_AVAILABLE,
        M26FertilizerPolicyV2.OFF,
    )
    animal_fertilizer = np.where(
        recycle,
        M3AnimalFertilizerPolicyV2.RESERVE_FOR_CROPS,
        M3AnimalFertilizerPolicyV2.COLLECT_WHEN_VISITING_AND_SELL,
    ).astype(np.int8)
    hand_target = np.broadcast_to(
        np.asarray((5, 8, 9, 9, 9, 9), dtype=np.int8), (count, 6)
    ).copy()
    liquidation = rng.choice(
        np.asarray((25 * 24, 25 * 24 + 12, 26 * 24), dtype=np.int16),
        size=count,
    )

    crop = genome.crop._replace(
        candidate_id=jnp.arange(count, dtype=jnp.int32),
        crop_target=jnp.asarray(crop_target),
        hand_target=jnp.asarray(hand_target),
        fertilizer_policy=jnp.asarray(fertilizer_policy),
        liquidation_start_step=jnp.asarray(liquidation),
    )
    animal = genome.animal._replace(
        candidate_id=jnp.arange(count, dtype=jnp.int32),
        animal_target=jnp.asarray(animal_target),
        hand_target=jnp.asarray(hand_target),
        feed_source_policy=jnp.asarray(feed),
        feed_stock_horizon_days=jnp.asarray(
            rng.integers(2, 5, size=count, dtype=np.int8)
        ),
        animal_fertilizer_policy=jnp.asarray(animal_fertilizer),
        liquidation_start_step=jnp.asarray(liquidation),
    )
    return genome._replace(
        candidate_id=jnp.arange(count, dtype=jnp.int32),
        crop=crop,
        animal=animal,
        crop_unit_share=jnp.asarray(
            rng.choice(np.asarray((0.40, 0.50, 0.60), dtype=np.float32), size=count)
        ),
    )


__all__ = ["sample_m35_farm_genomes_v2"]
