"""Deterministic M3.5 branch and cross-flow acceptance candidates."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import M26FertilizerPolicyV2
from .m3_constants import M3AnimalFertilizerPolicyV2, M3FeedSourcePolicyV2
from .m35_genome import default_m35_farm_genome_v2
from .m35_schema import M35FarmGenomeV2


def _one(
    candidate_id: int,
    *,
    feed: int,
    recycle: bool,
    share: float = 0.5,
) -> M35FarmGenomeV2:
    genome = default_m35_farm_genome_v2(1)
    crop = genome.crop
    animal = genome.animal
    if feed == M3FeedSourcePolicyV2.GROW_ONLY:
        crop_target = crop.crop_target.at[:, 0, 0].set(8).at[:, 1:, 0].set(10)
        crop = crop._replace(crop_target=crop_target)
    crop_fertilizer = crop.fertilizer_policy.at[:, 2].set(
        jnp.int8(
            M26FertilizerPolicyV2.ALWAYS_WHEN_AVAILABLE
            if recycle
            else M26FertilizerPolicyV2.OFF
        )
    )
    crop = crop._replace(
        candidate_id=jnp.asarray((candidate_id,), dtype=jnp.int32),
        fertilizer_policy=crop_fertilizer,
    )
    animal = animal._replace(
        candidate_id=jnp.asarray((candidate_id,), dtype=jnp.int32),
        feed_source_policy=jnp.asarray((feed,), dtype=jnp.int8),
        animal_fertilizer_policy=jnp.asarray(
            (
                M3AnimalFertilizerPolicyV2.RESERVE_FOR_CROPS
                if recycle
                else M3AnimalFertilizerPolicyV2.COLLECT_WHEN_VISITING_AND_SELL,
            ),
            dtype=jnp.int8,
        ),
    )
    return genome._replace(
        candidate_id=jnp.asarray((candidate_id,), dtype=jnp.int32),
        crop=crop,
        animal=animal,
        crop_unit_share=jnp.asarray((share,), dtype=jnp.float32),
    )


def _stack(candidates: list[M35FarmGenomeV2]) -> M35FarmGenomeV2:
    return jax.tree.map(lambda *values: jnp.concatenate(values, axis=0), *candidates)


def m35_branch_coverage_panel_v2() -> tuple[M35FarmGenomeV2, tuple[str, ...]]:
    names = (
        "BUY_ONLY_NO_RECYCLE",
        "GROW_ONLY_NO_RECYCLE",
        "HYBRID_NO_RECYCLE",
        "BUY_ONLY_FERTILIZER_RECYCLE",
        "GROW_ONLY_FERTILIZER_RECYCLE",
        "HYBRID_FERTILIZER_RECYCLE",
    )
    candidates = [
        _one(0, feed=M3FeedSourcePolicyV2.BUY_ONLY, recycle=False),
        _one(1, feed=M3FeedSourcePolicyV2.GROW_ONLY, recycle=False, share=0.50),
        _one(2, feed=M3FeedSourcePolicyV2.HYBRID, recycle=False),
        _one(3, feed=M3FeedSourcePolicyV2.BUY_ONLY, recycle=True),
        _one(4, feed=M3FeedSourcePolicyV2.GROW_ONLY, recycle=True, share=0.50),
        _one(5, feed=M3FeedSourcePolicyV2.HYBRID, recycle=True),
    ]
    return _stack(candidates), names


__all__ = ["m35_branch_coverage_panel_v2"]
