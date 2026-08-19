"""Deterministic M2.6 candidate panels used before any formal route search."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .m26_genome import default_m26_crop_genome_v2
from .schema import M26CropGenomeV2


def _empty_candidate() -> M26CropGenomeV2:
    genome = default_m26_crop_genome_v2(1)
    return genome._replace(
        phase_count=jnp.asarray((1,), dtype=jnp.int8),
        phase_start_step=jnp.asarray(((0, 720, 720, 720, 720, 720),), dtype=jnp.int16),
        crop_target=jnp.zeros_like(genome.crop_target),
        land_target=jnp.ones_like(genome.land_target),
        land_start_step=jnp.zeros_like(genome.land_start_step),
        hand_target=jnp.zeros_like(genome.hand_target).at[:, 0].set(5),
        fertilizer_policy=jnp.zeros_like(genome.fertilizer_policy),
        weed_recovery_policy=jnp.full_like(genome.weed_recovery_policy, 2),
        crop_abandon_policy=jnp.zeros_like(genome.crop_abandon_policy),
        sell_interval=jnp.full_like(genome.sell_interval, 48),
        sell_phase=jnp.zeros_like(genome.sell_phase),
        sell_price_floor_ratio=jnp.zeros_like(genome.sell_price_floor_ratio),
        sell_fraction=jnp.ones_like(genome.sell_fraction),
        shed_pressure_trigger=jnp.asarray((80,), dtype=jnp.int16),
    )


def _stack(candidates: list[M26CropGenomeV2]) -> M26CropGenomeV2:
    stacked = jax.tree.map(lambda *values: jnp.concatenate(values, axis=0), *candidates)
    return stacked._replace(candidate_id=jnp.arange(len(candidates), dtype=jnp.int32))


def m26_branch_coverage_panel_v2() -> tuple[M26CropGenomeV2, tuple[str, ...]]:
    """Return complete-season candidates covering the major M2.6 branches."""

    candidates: list[M26CropGenomeV2] = []
    names: list[str] = []

    for crop_id, crop_name in enumerate(("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")):
        genome = _empty_candidate()
        target = 10 if crop_id < 2 else 8
        genome = genome._replace(
            crop_target=genome.crop_target.at[:, 0, crop_id].set(target),
            crop_layout_policy=genome.crop_layout_policy.at[:, crop_id].set(crop_id % 4),
            hand_target=genome.hand_target.at[:, 0].set(5 if crop_id < 2 else 7),
        )
        candidates.append(genome)
        names.append(f"PURE_{crop_name}_L{crop_id % 4}")

    mixed_specs = (
        ("MIX_WHEAT_TOMATO", (5, 0, 8, 0, 0)),
        ("MIX_CARROT_STRAWBERRY", (0, 5, 0, 6, 0)),
        ("MIX_TOMATO_MELON", (0, 0, 6, 0, 5)),
    )
    for name, values in mixed_specs:
        genome = _empty_candidate()._replace(
            crop_target=jnp.zeros((1, 6, 5), dtype=jnp.int16).at[:, 0, :].set(
                jnp.asarray(values, dtype=jnp.int16)
            ),
            hand_target=jnp.zeros((1, 6), dtype=jnp.int8).at[:, 0].set(8),
            land_target=jnp.full((1, 6), 2, dtype=jnp.int8),
        )
        candidates.append(genome)
        names.append(name)

    lifecycle_targets = jnp.asarray(
        (
            (0, 0, 2, 0, 0),
            (0, 0, 6, 0, 0),
            (3, 0, 3, 0, 0),
            (5, 0, 0, 0, 0),
            (2, 0, 0, 0, 0),
            (2, 0, 4, 0, 0),
        ),
        dtype=jnp.int16,
    )[None]
    lifecycle = _empty_candidate()._replace(
        phase_count=jnp.asarray((6,), dtype=jnp.int8),
        phase_start_step=jnp.asarray(((0, 2 * 24, 5 * 24, 10 * 24, 15 * 24, 18 * 24),), dtype=jnp.int16),
        crop_target=lifecycle_targets,
        hand_target=jnp.asarray(((5, 7, 7, 6, 5, 7),), dtype=jnp.int8),
        land_target=jnp.asarray(((1, 1, 2, 2, 2, 2),), dtype=jnp.int8),
        land_start_step=jnp.asarray(((0, 0, 5 * 24, 10 * 24, 15 * 24, 18 * 24),), dtype=jnp.int16),
        crop_abandon_policy=jnp.zeros((1, 5), dtype=jnp.int8).at[:, 2].set(2),
        crop_last_plant_step=jnp.asarray(((24 * 24, 25 * 24, 19 * 24, 15 * 24, 17 * 24),), dtype=jnp.int16),
    )
    candidates.append(lifecycle)
    names.append("SIX_PHASE_EXPAND_SHRINK_STOP_RESTART")

    for policy, name, crop_id in (
        (1, "FERTILIZER_ALWAYS_STRAWBERRY", 3),
        (2, "FERTILIZER_HIGH_VALUE_MELON", 4),
    ):
        genome = _empty_candidate()._replace(
            crop_target=jnp.zeros((1, 6, 5), dtype=jnp.int16).at[:, 0, crop_id].set(6),
            fertilizer_policy=jnp.zeros((1, 5), dtype=jnp.int8).at[:, crop_id].set(policy),
            hand_target=jnp.zeros((1, 6), dtype=jnp.int8).at[:, 0].set(7),
            crop_layout_policy=jnp.zeros((1, 5), dtype=jnp.int8).at[:, crop_id].set(3),
        )
        candidates.append(genome)
        names.append(name)

    staged = _empty_candidate()._replace(
        phase_count=jnp.asarray((3,), dtype=jnp.int8),
        phase_start_step=jnp.asarray(((0, 7 * 24, 16 * 24, 720, 720, 720),), dtype=jnp.int16),
        crop_target=jnp.asarray(
            (((4, 0, 4, 0, 0), (6, 0, 8, 0, 0), (3, 0, 5, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),),
            dtype=jnp.int16,
        ),
        hand_target=jnp.asarray(((5, 8, 6, 0, 0, 0),), dtype=jnp.int8),
        land_target=jnp.asarray(((1, 2, 2, 2, 2, 2),), dtype=jnp.int8),
        land_start_step=jnp.asarray(((0, 7 * 24, 16 * 24, 720, 720, 720),), dtype=jnp.int16),
        sell_interval=jnp.full((1, 9), 24, dtype=jnp.int16),
        sell_fraction=jnp.full((1, 9), 0.5, dtype=jnp.float32),
        sell_price_floor_ratio=jnp.full((1, 9), 0.75, dtype=jnp.float32),
    )
    candidates.append(staged)
    names.append("THREE_PHASE_MARKET_AWARE_HALF_SELL")

    return _stack(candidates), tuple(names)


__all__ = ["m26_branch_coverage_panel_v2"]
