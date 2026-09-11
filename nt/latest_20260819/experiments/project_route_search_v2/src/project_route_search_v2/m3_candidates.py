"""Deterministic M3A/M3B branch-coverage candidates."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import NUM_ANIMALS, NUM_PRODUCTS

from .m3_genome import default_m3_animal_genome_v2
from .m3_schema import M3AnimalGenomeV2


def _empty_candidate() -> M3AnimalGenomeV2:
    genome = default_m3_animal_genome_v2(1)
    return genome._replace(
        phase_count=jnp.asarray((1,), dtype=jnp.int8),
        phase_start_step=jnp.asarray(((0, 720, 720, 720, 720, 720),), dtype=jnp.int16),
        animal_target=jnp.zeros_like(genome.animal_target),
        land_target=jnp.ones_like(genome.land_target),
        hand_target=jnp.full_like(genome.hand_target, 5),
        care_policy=jnp.zeros_like(genome.care_policy),
        first_cycle_care_bonus_target=jnp.zeros_like(genome.first_cycle_care_bonus_target),
        steady_cycle_care_bonus_target=jnp.zeros_like(genome.steady_cycle_care_bonus_target),
        animal_fertilizer_policy=jnp.asarray((0,), dtype=jnp.int8),
        sell_interval=jnp.full((1, NUM_PRODUCTS), 24, dtype=jnp.int16),
        sell_phase=jnp.zeros((1, NUM_PRODUCTS), dtype=jnp.int16),
        sell_price_floor_ratio=jnp.zeros((1, NUM_PRODUCTS), dtype=jnp.float32),
        sell_fraction=jnp.ones((1, NUM_PRODUCTS), dtype=jnp.float32),
        liquidation_start_step=jnp.asarray((26 * 24,), dtype=jnp.int16),
    )


def _stack(candidates: list[M3AnimalGenomeV2]) -> M3AnimalGenomeV2:
    result = jax.tree.map(lambda *values: jnp.concatenate(values, axis=0), *candidates)
    return result._replace(candidate_id=jnp.arange(len(candidates), dtype=jnp.int32))


def m3_branch_coverage_panel_v2() -> tuple[M3AnimalGenomeV2, tuple[str, ...]]:
    candidates: list[M3AnimalGenomeV2] = []
    names: list[str] = []
    species = ("GOOSE", "COW", "SHEEP")

    for animal, name in enumerate(species):
        for scale in (1, 4, 8):
            genome = _empty_candidate()._replace(
                animal_target=jnp.zeros((1, 6, NUM_ANIMALS), dtype=jnp.int16).at[:, 0, animal].set(scale),
                hand_target=jnp.full((1, 6), 4 if scale == 1 else 7, dtype=jnp.int8),
                land_target=jnp.full((1, 6), 1 if scale <= 4 else 2, dtype=jnp.int8),
                animal_layout_policy=jnp.zeros((1, NUM_ANIMALS), dtype=jnp.int8).at[:, animal].set(animal % 4),
            )
            candidates.append(genome)
            names.append(f"PURE_{name}_{scale}")

    for name, values in (
        ("MIX_GOOSE_COW", (3, 3, 0)),
        ("MIX_GOOSE_SHEEP", (3, 0, 3)),
        ("MIX_COW_SHEEP", (0, 3, 3)),
        ("MIX_ALL", (3, 2, 2)),
    ):
        genome = _empty_candidate()._replace(
            animal_target=jnp.broadcast_to(
                jnp.asarray(values, dtype=jnp.int16), (1, 6, NUM_ANIMALS)
            ),
            hand_target=jnp.full((1, 6), 8, dtype=jnp.int8),
            land_target=jnp.full((1, 6), 2, dtype=jnp.int8),
        )
        candidates.append(genome)
        names.append(name)

    for policy, label in ((1, "FIRST_CYCLE"), (2, "EVERY_CYCLE")):
        genome = _empty_candidate()._replace(
            animal_target=jnp.zeros((1, 6, NUM_ANIMALS), dtype=jnp.int16).at[:, :, 1].set(3),
            care_policy=jnp.zeros((1, NUM_ANIMALS), dtype=jnp.int8).at[:, 1].set(policy),
            first_cycle_care_bonus_target=jnp.zeros((1, NUM_ANIMALS), dtype=jnp.int8).at[:, 1].set(5),
            steady_cycle_care_bonus_target=jnp.zeros((1, NUM_ANIMALS), dtype=jnp.int8).at[:, 1].set(2),
            hand_target=jnp.full((1, 6), 7, dtype=jnp.int8),
        )
        candidates.append(genome)
        names.append(f"COW_CARE_{label}")

    for policy, label in ((1, "PASSIVE"), (2, "ACTIVE")):
        genome = _empty_candidate()._replace(
            animal_target=jnp.zeros((1, 6, NUM_ANIMALS), dtype=jnp.int16).at[:, :, 0].set(4),
            animal_fertilizer_policy=jnp.asarray((policy,), dtype=jnp.int8),
            hand_target=jnp.full((1, 6), 6, dtype=jnp.int8),
        )
        candidates.append(genome)
        names.append(f"GOOSE_FERTILIZER_{label}")

    staged = _empty_candidate()._replace(
        phase_count=jnp.asarray((3,), dtype=jnp.int8),
        phase_start_step=jnp.asarray(((0, 5 * 24, 12 * 24, 720, 720, 720),), dtype=jnp.int16),
        animal_target=jnp.asarray(
            (((2, 0, 0), (4, 2, 0), (5, 3, 3), (5, 3, 3), (5, 3, 3), (5, 3, 3)),),
            dtype=jnp.int16,
        ),
        hand_target=jnp.asarray(((4, 7, 10, 10, 10, 10),), dtype=jnp.int8),
        land_target=jnp.asarray(((1, 1, 2, 2, 2, 2),), dtype=jnp.int8),
        care_policy=jnp.full((1, NUM_ANIMALS), 2, dtype=jnp.int8),
        first_cycle_care_bonus_target=jnp.asarray(((3, 5, 5),), dtype=jnp.int8),
        steady_cycle_care_bonus_target=jnp.asarray(((1, 2, 3),), dtype=jnp.int8),
        animal_fertilizer_policy=jnp.asarray((1,), dtype=jnp.int8),
    )
    candidates.append(staged)
    names.append("STAGED_ALL_SPECIES")
    return _stack(candidates), tuple(names)


__all__ = ["m3_branch_coverage_panel_v2"]
