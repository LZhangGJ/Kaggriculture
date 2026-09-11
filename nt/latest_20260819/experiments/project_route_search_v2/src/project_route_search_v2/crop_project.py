"""Minimal M2 crop-project creation and state refresh."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    CROP_FIRST_YIELD_DAY,
    CROP_SEED_COST,
    NUM_CROPS,
    TileKind,
)
from kaggriculture_jax.types import State

from .constants import (
    CropProjectPhaseV2,
    ProjectStatusV2,
    ProjectTypeV2,
)
from .schema import CropProjectConfigV2, ProjectControllerStateV2


_CROP_FIRST = jnp.asarray(CROP_FIRST_YIELD_DAY, dtype=jnp.int8)
_CROP_SEED_COST = jnp.asarray(CROP_SEED_COST, dtype=jnp.int32)


def default_crop_project_config_v2(
    batch_size: int,
    *,
    crop_id: int = 0,
    target_tiles: int = 4,
    seed_batch_size: int = 4,
    cash_floor: int = 500,
    investment_stop_step: int = 600,
    liquidation_start_step: int = 648,
    harvest_age_days: int | None = None,
) -> CropProjectConfigV2:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if crop_id < 0 or crop_id >= NUM_CROPS:
        raise ValueError("crop_id is outside the official crop range")
    if target_tiles <= 0:
        raise ValueError("target_tiles must be positive")
    if harvest_age_days is None:
        harvest_age_days = int(CROP_FIRST_YIELD_DAY[crop_id])
    return CropProjectConfigV2(
        crop_id=jnp.full((batch_size,), crop_id, dtype=jnp.int8),
        target_tiles=jnp.full((batch_size,), target_tiles, dtype=jnp.int16),
        seed_batch_size=jnp.full(
            (batch_size,), seed_batch_size, dtype=jnp.int16
        ),
        cash_floor=jnp.full((batch_size,), cash_floor, dtype=jnp.int32),
        investment_stop_step=jnp.full(
            (batch_size,), investment_stop_step, dtype=jnp.int16
        ),
        liquidation_start_step=jnp.full(
            (batch_size,), liquidation_start_step, dtype=jnp.int16
        ),
        harvest_age_days=jnp.full(
            (batch_size,), harvest_age_days, dtype=jnp.int8
        ),
    )


def ensure_crop_project_v2(
    states: State,
    controller: ProjectControllerStateV2,
    config: CropProjectConfigV2,
) -> ProjectControllerStateV2:
    """Materialize exactly one crop project in slot zero when it is empty."""

    projects = controller.projects
    create = projects.status[:, 0] == ProjectStatusV2.UNUSED
    projects = projects._replace(
        project_id=projects.project_id.at[:, 0].set(
            jnp.where(create, 0, projects.project_id[:, 0]).astype(jnp.int16)
        ),
        project_type=projects.project_type.at[:, 0].set(
            jnp.where(create, ProjectTypeV2.CROP_LOT, projects.project_type[:, 0]).astype(
                jnp.int8
            )
        ),
        item_id=projects.item_id.at[:, 0].set(
            jnp.where(create, config.crop_id, projects.item_id[:, 0]).astype(jnp.int8)
        ),
        status=projects.status.at[:, 0].set(
            jnp.where(create, ProjectStatusV2.ACTIVE, projects.status[:, 0]).astype(
                jnp.int8
            )
        ),
        phase=projects.phase.at[:, 0].set(
            jnp.where(
                create,
                CropProjectPhaseV2.ACQUIRING_SEEDS,
                projects.phase[:, 0],
            ).astype(jnp.int8)
        ),
        target_count=projects.target_count.at[:, 0].set(
            jnp.where(create, config.target_tiles, projects.target_count[:, 0]).astype(
                jnp.int16
            )
        ),
        start_step=projects.start_step.at[:, 0].set(
            jnp.where(create, states.step, projects.start_step[:, 0]).astype(jnp.int16)
        ),
        stop_step=projects.stop_step.at[:, 0].set(
            jnp.where(
                create, config.investment_stop_step, projects.stop_step[:, 0]
            ).astype(jnp.int16)
        ),
        latest_bank_step=projects.latest_bank_step.at[:, 0].set(
            jnp.where(
                create, config.liquidation_start_step, projects.latest_bank_step[:, 0]
            ).astype(jnp.int16)
        ),
    )
    return controller._replace(projects=projects)


def refresh_crop_project_v2(
    states: State,
    controller: ProjectControllerStateV2,
    config: CropProjectConfigV2,
    player: int,
    *,
    harvested_delta: jax.Array | None = None,
    purchased_seed_units: jax.Array | None = None,
) -> ProjectControllerStateV2:
    """Refresh counts and coarse project phase from exact environment state."""

    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    crop_id = config.crop_id.astype(jnp.int32)
    kind = states.tile_kind[:, player]
    crop = states.tile_crop[:, player]
    origin = states.tile_origin_day[:, player].astype(jnp.int16)
    yields = states.tile_yield[:, player]
    current_day = (states.step // 24).astype(jnp.int16)
    owned_crop = (kind == TileKind.PLANT) & (crop == crop_id[:, None, None])
    active_count = jnp.sum(owned_crop, axis=(1, 2), dtype=jnp.int16)
    mature = owned_crop & (
        current_day[:, None, None] - origin
        >= config.harvest_age_days.astype(jnp.int16)[:, None, None]
    ) & (yields > 0)
    harvestable = jnp.any(mature, axis=(1, 2))
    seed_count = states.seeds[batch, player, crop_id].astype(jnp.int16)
    closing = states.step >= config.liquidation_start_step
    phase = jnp.where(
        closing,
        CropProjectPhaseV2.CLOSING,
        jnp.where(
            harvestable,
            CropProjectPhaseV2.HARVESTING,
            jnp.where(
                active_count < config.target_tiles,
                jnp.where(
                    seed_count > 0,
                    CropProjectPhaseV2.ESTABLISHING,
                    CropProjectPhaseV2.ACQUIRING_SEEDS,
                ),
                CropProjectPhaseV2.ACTIVE,
            ),
        ),
    ).astype(jnp.int8)
    phase = jnp.where(states.done, CropProjectPhaseV2.DONE, phase).astype(jnp.int8)
    status = jnp.where(
        states.done, ProjectStatusV2.COMPLETE, ProjectStatusV2.ACTIVE
    ).astype(jnp.int8)
    if harvested_delta is None:
        harvested_delta = jnp.zeros((batch_size,), dtype=jnp.int32)
    if purchased_seed_units is None:
        purchased_seed_units = jnp.zeros((batch_size,), dtype=jnp.int32)

    projects = controller.projects
    seed_cost = _CROP_SEED_COST[crop_id]
    projects = projects._replace(
        status=projects.status.at[:, 0].set(status),
        phase=projects.phase.at[:, 0].set(phase),
        active_count=projects.active_count.at[:, 0].set(active_count),
        committed_count=projects.committed_count.at[:, 0].set(
            jnp.minimum(active_count + seed_count, config.target_tiles).astype(jnp.int16)
        ),
        completed_count=projects.completed_count.at[:, 0].add(
            harvested_delta.astype(jnp.int16)
        ),
        last_progress_step=projects.last_progress_step.at[:, 0].set(
            jnp.where(
                (harvested_delta > 0) | (purchased_seed_units > 0),
                states.step,
                projects.last_progress_step[:, 0],
            ).astype(jnp.int16)
        ),
        cash_spent=projects.cash_spent.at[:, 0].add(
            (purchased_seed_units.astype(jnp.int32) * seed_cost).astype(jnp.int32)
        ),
    )

    tile_project_id = jnp.where(
        owned_crop,
        jnp.int16(0),
        jnp.where(controller.tile_project_id == 0, -1, controller.tile_project_id),
    ).astype(jnp.int16)
    return controller._replace(
        projects=projects,
        tile_project_id=tile_project_id,
        route_phase=phase,
        liquidation_mode=closing,
    )


__all__ = [
    "default_crop_project_config_v2",
    "ensure_crop_project_v2",
    "refresh_crop_project_v2",
]

