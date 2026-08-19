"""M3.5 synchronized crop/animal genome constructors and validation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .constants import M26FertilizerPolicyV2
from .m26_genome import default_m26_crop_genome_v2, validate_m26_crop_genome_v2
from .m3_constants import M3AnimalFertilizerPolicyV2, M3FeedSourcePolicyV2
from .m3_genome import default_m3_animal_genome_v2, validate_m3_animal_genome_v2
from .m35_schema import M35FarmGenomeV2


def default_m35_farm_genome_v2(batch_size: int) -> M35FarmGenomeV2:
    """Return a conservative HYBRID route with real fertilizer recycling."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    crop = default_m26_crop_genome_v2(batch_size)
    animal = default_m3_animal_genome_v2(batch_size)
    phase_start = jnp.broadcast_to(
        jnp.asarray((0, 6 * 24, 14 * 24, 720, 720, 720), dtype=jnp.int16),
        crop.phase_start_step.shape,
    )
    crop_target = jnp.zeros_like(crop.crop_target)
    crop_target = crop_target.at[:, 0].set(
        jnp.asarray((6, 0, 4, 0, 0), dtype=jnp.int16)
    )
    crop_target = crop_target.at[:, 1].set(
        jnp.asarray((8, 0, 6, 0, 0), dtype=jnp.int16)
    )
    crop_target = crop_target.at[:, 2].set(
        jnp.asarray((8, 0, 8, 0, 0), dtype=jnp.int16)
    )
    shared_land = jnp.broadcast_to(
        jnp.asarray((1, 1, 1, 1, 1, 1), dtype=jnp.int8), crop.land_target.shape
    )
    shared_hands = jnp.broadcast_to(
        jnp.asarray((4, 7, 9, 9, 9, 9), dtype=jnp.int8), crop.hand_target.shape
    )
    crop = crop._replace(
        candidate_id=jnp.arange(batch_size, dtype=jnp.int32),
        phase_count=jnp.full((batch_size,), 3, dtype=jnp.int8),
        phase_start_step=phase_start,
        crop_target=crop_target,
        land_target=shared_land,
        land_start_step=phase_start,
        hand_target=shared_hands,
        fertilizer_policy=crop.fertilizer_policy.at[:, 2].set(
            jnp.int8(M26FertilizerPolicyV2.ALWAYS_WHEN_AVAILABLE)
        ),
        crop_cash_cap=jnp.broadcast_to(
            jnp.asarray((12_000, 8_000, 20_000, 20_000, 20_000), dtype=jnp.int32),
            crop.crop_cash_cap.shape,
        ),
        cash_floor=jnp.full((batch_size,), 500, dtype=jnp.int32),
        liquidation_start_step=jnp.full(
            (batch_size,), 25 * 24, dtype=jnp.int16
        ),
    )

    animal_target = jnp.zeros_like(animal.animal_target)
    animal_target = animal_target.at[:, 1].set(
        jnp.asarray((2, 0, 0), dtype=jnp.int16)
    )
    animal_target = animal_target.at[:, 2:].set(
        jnp.asarray((2, 1, 1), dtype=jnp.int16)
    )
    animal = animal._replace(
        candidate_id=jnp.arange(batch_size, dtype=jnp.int32),
        phase_count=jnp.full((batch_size,), 3, dtype=jnp.int8),
        phase_start_step=phase_start,
        animal_target=animal_target,
        land_target=shared_land,
        hand_target=shared_hands,
        feed_source_policy=jnp.full(
            (batch_size,), M3FeedSourcePolicyV2.HYBRID, dtype=jnp.int8
        ),
        animal_fertilizer_policy=jnp.full(
            (batch_size,), M3AnimalFertilizerPolicyV2.RESERVE_FOR_CROPS, dtype=jnp.int8
        ),
        care_policy=jnp.zeros_like(animal.care_policy),
        first_cycle_care_bonus_target=jnp.zeros_like(
            animal.first_cycle_care_bonus_target
        ),
        steady_cycle_care_bonus_target=jnp.zeros_like(
            animal.steady_cycle_care_bonus_target
        ),
        animal_harvest_trigger_units=jnp.broadcast_to(
            jnp.asarray((1, 2, 2), dtype=jnp.int8),
            animal.animal_harvest_trigger_units.shape,
        ),
        cash_floor=jnp.full((batch_size,), 500, dtype=jnp.int32),
        liquidation_start_step=jnp.full(
            (batch_size,), 25 * 24, dtype=jnp.int16
        ),
    )
    return M35FarmGenomeV2(
        candidate_id=jnp.arange(batch_size, dtype=jnp.int32),
        crop=crop,
        animal=animal,
        crop_unit_share=jnp.full((batch_size,), 0.5, dtype=jnp.float32),
    )


def validate_m35_farm_genome_v2(genome: M35FarmGenomeV2) -> list[str]:
    """Validate the joint milestone without weakening frozen M2.6/M3 gates."""

    errors: list[str] = []
    candidate = np.asarray(genome.candidate_id)
    if candidate.ndim != 1 or candidate.shape[0] == 0:
        return ["shape:candidate_id"]
    batch_size = candidate.shape[0]
    if np.asarray(genome.crop_unit_share).shape != (batch_size,):
        errors.append("shape:crop_unit_share")
    share = np.asarray(genome.crop_unit_share)
    if np.any((share <= 0.0) | (share >= 1.0)):
        errors.append("range:crop_unit_share")

    errors.extend(f"crop:{value}" for value in validate_m26_crop_genome_v2(genome.crop))
    # Reuse every frozen M3 validation except the two modes intentionally
    # activated at M3.5.
    proxy_animal = genome.animal._replace(
        feed_source_policy=jnp.full_like(
            genome.animal.feed_source_policy, M3FeedSourcePolicyV2.BUY_ONLY
        ),
        animal_fertilizer_policy=jnp.minimum(
            genome.animal.animal_fertilizer_policy,
            jnp.int8(M3AnimalFertilizerPolicyV2.ACTIVE_COLLECT_AND_SELL),
        ),
    )
    errors.extend(
        f"animal:{value}" for value in validate_m3_animal_genome_v2(proxy_animal)
    )
    feed = np.asarray(genome.animal.feed_source_policy)
    if np.any((feed < M3FeedSourcePolicyV2.BUY_ONLY) | (feed > M3FeedSourcePolicyV2.HYBRID)):
        errors.append("range:feed_source_policy")
    fertilizer = np.asarray(genome.animal.animal_fertilizer_policy)
    if np.any(
        (fertilizer < M3AnimalFertilizerPolicyV2.IGNORE)
        | (fertilizer > M3AnimalFertilizerPolicyV2.RESERVE_FOR_CROPS)
    ):
        errors.append("range:animal_fertilizer_policy")

    # M3.5 has one business calendar and one shared land/workforce budget.
    for name in ("phase_count", "phase_start_step", "land_target", "hand_target"):
        if not np.array_equal(
            np.asarray(getattr(genome.crop, name)),
            np.asarray(getattr(genome.animal, name)),
        ):
            errors.append(f"joint:not_synchronized:{name}")
    if not np.array_equal(
        np.asarray(genome.crop.liquidation_start_step),
        np.asarray(genome.animal.liquidation_start_step),
    ):
        errors.append("joint:not_synchronized:liquidation_start_step")
    if not np.array_equal(np.asarray(genome.crop.candidate_id), candidate):
        errors.append("joint:candidate_id_crop")
    if not np.array_equal(np.asarray(genome.animal.candidate_id), candidate):
        errors.append("joint:candidate_id_animal")

    grow = feed == M3FeedSourcePolicyV2.GROW_ONLY
    wheat_target = np.asarray(genome.crop.crop_target)[..., 0]
    animal_target = np.asarray(genome.animal.animal_target)
    if np.any(grow[:, None] & (np.max(animal_target, axis=-1) > 0) & (wheat_target <= 0)):
        errors.append("joint:grow_only_without_wheat_project")
    reserve = fertilizer == M3AnimalFertilizerPolicyV2.RESERVE_FOR_CROPS
    crop_fertilizer = np.asarray(genome.crop.fertilizer_policy)
    if np.any(reserve & (~np.any(crop_fertilizer != M26FertilizerPolicyV2.OFF, axis=-1))):
        errors.append("joint:reserved_fertilizer_without_crop_consumer")
    return errors


__all__ = ["default_m35_farm_genome_v2", "validate_m35_farm_genome_v2"]
