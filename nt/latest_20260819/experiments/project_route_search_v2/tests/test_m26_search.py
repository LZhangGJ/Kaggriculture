from __future__ import annotations

import jax
import numpy as np

from project_route_search_v2.m26_genome import validate_m26_crop_genome_v2
from project_route_search_v2.m26_search import sample_m26_crop_genomes_v2


def test_m26_sampler_is_deterministic_and_contract_valid() -> None:
    left = sample_m26_crop_genomes_v2(16, 2601)
    right = sample_m26_crop_genomes_v2(16, 2601)
    assert validate_m26_crop_genome_v2(left) == []
    for left_value, right_value in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(left_value, right_value)
    assert set(np.asarray(left.phase_count).tolist()) == {1, 2, 3, 4, 5, 6}
    assert np.all(np.sum(np.asarray(left.crop_target), axis=(0, 1)) > 0)
