from __future__ import annotations

import jax
import jax.numpy as jnp

from project_route_search_v2.m3_candidates import m3_branch_coverage_panel_v2
from project_route_search_v2.m3_genome import (
    default_m3_animal_genome_v2,
    validate_m3_animal_genome_v2,
)
from project_route_search_v2.m3_search import sample_m3_animal_genomes_v2
from project_route_search_v2.m3_schema import M3AnimalGenomeV2


def test_default_m3_genome_is_valid_monotone_and_a_jax_pytree() -> None:
    genome = default_m3_animal_genome_v2(3)
    assert validate_m3_animal_genome_v2(genome) == []
    assert jnp.all(jnp.diff(genome.animal_target, axis=1) >= 0)
    leaves, tree = jax.tree.flatten(genome)
    assert len(leaves) == len(M3AnimalGenomeV2._fields)
    assert isinstance(jax.tree.unflatten(tree, leaves), M3AnimalGenomeV2)


def test_m3_validator_rejects_target_shrink_and_future_feed_mode() -> None:
    genome = default_m3_animal_genome_v2(1)
    shrinking = genome._replace(
        animal_target=genome.animal_target.at[0, 1, 0].set(jnp.int16(1))
    )
    assert "animal_target:not_monotonic" in validate_m3_animal_genome_v2(shrinking)

    grow_only = genome._replace(feed_source_policy=jnp.asarray((1,), dtype=jnp.int8))
    assert "milestone:feed_source_policy_not_buy_only" in validate_m3_animal_genome_v2(
        grow_only
    )


def test_m3_branch_panel_and_structured_sampler_only_emit_valid_genomes() -> None:
    panel, names = m3_branch_coverage_panel_v2()
    assert len(names) == 18
    assert validate_m3_animal_genome_v2(panel) == []
    assert validate_m3_animal_genome_v2(
        sample_m3_animal_genomes_v2(256, seed=1701)
    ) == []
