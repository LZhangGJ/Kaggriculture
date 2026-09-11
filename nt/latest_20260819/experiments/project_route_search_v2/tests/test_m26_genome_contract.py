from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from project_route_search_v2.m26_genome import (
    default_m26_crop_genome_v2,
    validate_m26_crop_genome_v2,
)
from project_route_search_v2.schema import M26CropGenomeV2


def test_default_m26_genome_is_valid_and_a_jax_pytree() -> None:
    genome = default_m26_crop_genome_v2(3)
    assert validate_m26_crop_genome_v2(genome) == []
    leaves, tree = jax.tree.flatten(genome)
    assert len(leaves) == len(M26CropGenomeV2._fields)
    assert isinstance(jax.tree.unflatten(tree, leaves), M26CropGenomeV2)


def test_m26_crop_targets_use_one_uniform_five_crop_tensor() -> None:
    genome = default_m26_crop_genome_v2(2)
    assert genome.crop_target.shape == (2, 6, 5)
    np.testing.assert_array_equal(genome.crop_target[:, 0, :], [[0, 0, 12, 0, 0]] * 2)
    np.testing.assert_array_equal(genome.crop_target[:, 1, :], [[8, 0, 18, 0, 0]] * 2)
    assert not any("primary" in field or "support" in field for field in genome._fields)


def test_m26_validator_rejects_phase_and_harvest_boundary_errors() -> None:
    genome = default_m26_crop_genome_v2(1)
    bad_phase = genome._replace(
        phase_start_step=genome.phase_start_step.at[0, 1].set(jnp.int16(121))
    )
    assert "phase:not_day_boundary" in validate_m26_crop_genome_v2(bad_phase)

    bad_age = genome._replace(
        harvest_min_age_days=genome.harvest_min_age_days.at[0, 2].set(jnp.int8(7))
    )
    assert "range:harvest_min_age_days" in validate_m26_crop_genome_v2(bad_age)
