"""M2.6 unified five-crop controller.

This module keeps the M2.5 controller intact as a historical benchmark while
removing primary/support crop aliases from all new route-search decisions.
The controller owns project targets, coarse layout, admission, recovery and
market policy; primitive legality remains owned by the parity-tested Strategic
V5 Full-core compiler.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    BOARD_SIZE,
    CROP_FIRST_YIELD_DAY,
    CROP_SEED_COST,
    EPISODE_STEPS,
    FLAG_WATERED,
    HIRE_COST,
    LAND_PRICES,
    MARKET_BASE_PRICES,
    MARKET_INITIAL_INVENTORY,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_CROPS,
    NUM_PRODUCTS,
    SHED_ACCESS,
    TURNS_PER_DAY,
    TileKind,
)
from kaggriculture_jax.state import load_tables
from kaggriculture_jax.types import State, StaticTables
from strategic_v5.constants import (
    FailureCodeV1,
    TaskPhaseV1,
    TaskStatusV1,
    TaskTypeV1,
)
from strategic_v5.e4_executor import (
    E4PlayerActionV1,
    FullCoreEffectDiagnosticsV1,
    compile_full_core_player_action_v1,
    update_full_core_controller_from_effects_v1,
)
from strategic_v5.e5_econ import exact_market_quote_v1

from .constants import (
    M26CropAbandonPolicyV2,
    M26CropLayoutPolicyV2,
    M26FertilizerPolicyV2,
    M26WeedRecoveryPolicyV2,
    ProjectStatusV2,
    ProjectTypeV2,
)
from .crop_executor import clear_invalidated_crop_tasks_v2
from .lifecycle import empty_market_tasks_v2, reconcile_project_controller_v2
from .schema import M26CropGenomeV2, ProjectControllerStateV2


_TILE_X = jnp.tile(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
_TILE_Y = jnp.repeat(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
_TILE_ID = jnp.arange(BOARD_SIZE * BOARD_SIZE, dtype=jnp.int32)
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)
_SEED_COST = jnp.asarray(CROP_SEED_COST, dtype=jnp.int32)
_LAND_COST = jnp.asarray(LAND_PRICES, dtype=jnp.int32)
_HIRE_COST = jnp.asarray(HIRE_COST, dtype=jnp.int32)
_BASE_PRICE = jnp.asarray(MARKET_BASE_PRICES, dtype=jnp.int32)
_FERTILIZER_ITEM = 8
_INVALID_SCORE = jnp.int32(1_000_000_000)
_DEFAULT_TABLES = load_tables()

# Rows follow the frozen sorted SHOP_NAMES contract; columns are PRODUCTS.
_SHOP_DEMAND = jnp.asarray(
    (
        (1, 0, 0, 0, 0, 1, 0, 0, 0),
        (1, 0, 0, 1, 0, 1, 0, 0, 0),
        (1, 1, 1, 1, 0, 0, 0, 0, 0),
        (1, 0, 0, 1, 0, 0, 1, 0, 0),
        (0, 2, 0, 0, 0, 0, 0, 0, 0),
        (1, 0, 1, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 2, 0),
    ),
    dtype=jnp.int16,
)


def m26_phase_v2(states: State, genome: M26CropGenomeV2) -> jax.Array:
    phase_slot = jnp.arange(genome.phase_start_step.shape[1], dtype=jnp.int8)[None]
    active_slot = phase_slot < genome.phase_count[:, None]
    crossed = active_slot & (states.step[:, None] >= genome.phase_start_step)
    return jnp.maximum(jnp.sum(crossed, axis=-1) - 1, 0).astype(jnp.int8)


def _phase_value(values: jax.Array, phase: jax.Array) -> jax.Array:
    batch = jnp.arange(phase.shape[0], dtype=jnp.int32)
    return values[batch, phase.astype(jnp.int32)]


def m26_phase_targets_v2(
    states: State, genome: M26CropGenomeV2
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    phase = m26_phase_v2(states, genome)
    return (
        phase,
        _phase_value(genome.crop_target, phase).astype(jnp.int16),
        _phase_value(genome.land_target, phase).astype(jnp.int8),
        _phase_value(genome.hand_target, phase).astype(jnp.int8),
    )


def _active_crop_counts(states: State, player: int) -> jax.Array:
    kind = states.tile_kind[:, player].reshape(states.step.shape[0], -1)
    crop = states.tile_crop[:, player].reshape(states.step.shape[0], -1)
    safe_crop = jnp.clip(crop.astype(jnp.int32), 0, NUM_CROPS - 1)
    one_hot = jax.nn.one_hot(safe_crop, NUM_CROPS, dtype=jnp.int16)
    return jnp.sum(
        one_hot * (kind == TileKind.PLANT)[..., None], axis=1, dtype=jnp.int16
    )


def ensure_m26_projects_v2(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M26CropGenomeV2,
    player: int,
) -> ProjectControllerStateV2:
    """Materialize five crop projects plus land/workforce without aliases."""

    phase, targets, land_target, hand_target = m26_phase_targets_v2(states, genome)
    batch_size = states.step.shape[0]
    counts = _active_crop_counts(states, player)
    projects = controller.projects

    for crop_id in range(NUM_CROPS):
        slot = crop_id
        target = targets[:, crop_id]
        status = jnp.where(
            (target > 0) | (counts[:, crop_id] > 0),
            ProjectStatusV2.ACTIVE,
            ProjectStatusV2.PLANNED,
        ).astype(jnp.int8)
        projects = projects._replace(
            project_id=projects.project_id.at[:, slot].set(jnp.int16(slot)),
            project_type=projects.project_type.at[:, slot].set(jnp.int8(ProjectTypeV2.CROP_LOT)),
            item_id=projects.item_id.at[:, slot].set(jnp.int8(crop_id)),
            status=projects.status.at[:, slot].set(status),
            phase=projects.phase.at[:, slot].set(phase),
            target_count=projects.target_count.at[:, slot].set(target),
            active_count=projects.active_count.at[:, slot].set(counts[:, crop_id]),
            start_step=projects.start_step.at[:, slot].set(
                jnp.where(
                    (projects.start_step[:, slot] < 0) & (target > 0),
                    states.step,
                    projects.start_step[:, slot],
                )
            ),
            stop_step=projects.stop_step.at[:, slot].set(genome.crop_last_plant_step[:, crop_id]),
            latest_bank_step=projects.latest_bank_step.at[:, slot].set(genome.liquidation_start_step),
            cash_budget=projects.cash_budget.at[:, slot].set(genome.crop_cash_cap[:, crop_id]),
            layout_policy_id=projects.layout_policy_id.at[:, slot].set(
                genome.crop_layout_policy[:, crop_id]
            ),
        )

    for slot, project_type, target, active_count in (
        (NUM_CROPS, ProjectTypeV2.LAND_EXPANSION, land_target, states.unlocked_count[:, player]),
        (NUM_CROPS + 1, ProjectTypeV2.WORKFORCE, hand_target, states.hires_today[:, player]),
    ):
        projects = projects._replace(
            project_id=projects.project_id.at[:, slot].set(jnp.int16(slot)),
            project_type=projects.project_type.at[:, slot].set(jnp.int8(project_type)),
            item_id=projects.item_id.at[:, slot].set(jnp.int8(-1)),
            status=projects.status.at[:, slot].set(jnp.int8(ProjectStatusV2.ACTIVE)),
            phase=projects.phase.at[:, slot].set(phase),
            target_count=projects.target_count.at[:, slot].set(target.astype(jnp.int16)),
            active_count=projects.active_count.at[:, slot].set(active_count.astype(jnp.int16)),
        )

    kind = states.tile_kind[:, player]
    crop = states.tile_crop[:, player].astype(jnp.int16)
    old = controller.tile_project_id
    remembered = jnp.where(
        (kind == TileKind.PLANT) & (crop >= 0),
        crop,
        jnp.where(
            ((kind == TileKind.WEED) | (kind == TileKind.EMPTY)) & (old >= 0) & (old < NUM_CROPS),
            old,
            jnp.int16(-1),
        ),
    )
    return controller._replace(
        projects=projects,
        tile_project_id=remembered,
        route_phase=phase,
        liquidation_mode=states.step >= genome.liquidation_start_step,
    )


def _nearest_shed(position: jax.Array) -> tuple[jax.Array, jax.Array]:
    distance = jnp.sum(
        jnp.abs(position[:, None, :].astype(jnp.int16) - _SHED_ACCESS[None]), axis=-1
    )
    index = jnp.argmin(distance, axis=-1)
    return _SHED_ACCESS[index], jnp.min(distance, axis=-1).astype(jnp.int16)


def _crop_centroids(states: State, player: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    kind = states.tile_kind[:, player].reshape(states.step.shape[0], -1)
    crop = states.tile_crop[:, player].reshape(states.step.shape[0], -1)
    safe_crop = jnp.clip(crop.astype(jnp.int32), 0, NUM_CROPS - 1)
    one_hot = jax.nn.one_hot(safe_crop, NUM_CROPS, dtype=jnp.int32)
    present = one_hot * (kind == TileKind.PLANT)[..., None]
    count = jnp.sum(present, axis=1, dtype=jnp.int32)
    sum_x = jnp.sum(present * _TILE_X[None, :, None], axis=1, dtype=jnp.int32)
    sum_y = jnp.sum(present * _TILE_Y[None, :, None], axis=1, dtype=jnp.int32)
    centroid_x = jnp.where(count > 0, sum_x // jnp.maximum(count, 1), 4).astype(jnp.int16)
    centroid_y = jnp.where(count > 0, sum_y // jnp.maximum(count, 1), 4).astype(jnp.int16)
    return centroid_x, centroid_y, count.astype(jnp.int16)


def _choose_m26_tile(
    mask: jax.Array,
    position: jax.Array,
    crop_id: jax.Array,
    layout_policy: jax.Array,
    centroid_x: jax.Array,
    centroid_y: jax.Array,
    crop_count: jax.Array,
    remembered_project: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch = jnp.arange(position.shape[0], dtype=jnp.int32)
    ux = position[:, 0, None].astype(jnp.int16)
    uy = position[:, 1, None].astype(jnp.int16)
    travel = jnp.abs(_TILE_X[None] - ux) + jnp.abs(_TILE_Y[None] - uy)
    shed_distance = jnp.min(
        jnp.abs(_TILE_X[:, None] - _SHED_ACCESS[None, :, 0])
        + jnp.abs(_TILE_Y[:, None] - _SHED_ACCESS[None, :, 1]),
        axis=-1,
    )[None]
    safe_crop = jnp.clip(crop_id.astype(jnp.int32), 0, NUM_CROPS - 1)
    cx = centroid_x[batch, safe_crop][:, None]
    cy = centroid_y[batch, safe_crop][:, None]
    cluster = jnp.abs(_TILE_X[None] - cx) + jnp.abs(_TILE_Y[None] - cy)

    qx = jnp.where(safe_crop == 4, 5, jnp.where((safe_crop % 2) == 0, 2, 7))[:, None]
    qy = jnp.where(safe_crop == 4, 5, jnp.where(((safe_crop // 2) % 2) == 0, 2, 7))[:, None]
    quadrant = jnp.abs(_TILE_X[None] - qx) + jnp.abs(_TILE_Y[None] - qy)

    serpentine = _TILE_Y * BOARD_SIZE + jnp.where((_TILE_Y % 2) == 0, _TILE_X, 9 - _TILE_X)
    desired_rank = (safe_crop * 19 + crop_count[batch, safe_crop].astype(jnp.int32)) % 100
    strip = jnp.abs(serpentine[None] - desired_rank[:, None])

    layout_cost = jnp.where(
        layout_policy[:, None] == M26CropLayoutPolicyV2.CENTER_COMPACT,
        shed_distance,
        jnp.where(
            layout_policy[:, None] == M26CropLayoutPolicyV2.CLUSTER_EXPANSION,
            cluster,
            jnp.where(
                layout_policy[:, None] == M26CropLayoutPolicyV2.QUADRANT_ZONED,
                quadrant,
                strip,
            ),
        ),
    ).astype(jnp.int32)
    same_crop_replant = remembered_project == safe_crop[:, None]
    score = (
        travel.astype(jnp.int32) * 100
        # Coarse layout must be strong enough to alter actual tile choices;
        # travel remains a tie-breaker inside the selected business zone.
        + layout_cost * 150
        + _TILE_ID[None]
        - same_crop_replant.astype(jnp.int32) * 75
    )
    score = jnp.where(mask, score, _INVALID_SCORE)
    target = jnp.argmin(score, axis=-1)
    valid = jnp.min(score, axis=-1) < _INVALID_SCORE
    distance = jnp.take_along_axis(travel, target[:, None], axis=-1)[:, 0]
    return target.astype(jnp.int32), valid, distance.astype(jnp.int16)


def _choose_nearest(
    mask: jax.Array, position: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    travel = (
        jnp.abs(_TILE_X[None] - position[:, 0, None].astype(jnp.int16))
        + jnp.abs(_TILE_Y[None] - position[:, 1, None].astype(jnp.int16))
    )
    score = jnp.where(mask, travel.astype(jnp.int32) * 100 + _TILE_ID[None], _INVALID_SCORE)
    target = jnp.argmin(score, axis=-1)
    valid = jnp.min(score, axis=-1) < _INVALID_SCORE
    distance = jnp.take_along_axis(travel, target[:, None], axis=-1)[:, 0]
    return target.astype(jnp.int32), valid, distance.astype(jnp.int16)


def _set_unit_task(
    tasks,
    unit: int,
    assign: jax.Array,
    *,
    task_type: jax.Array,
    target_id: jax.Array,
    target_x: jax.Array,
    target_y: jax.Array,
    item_id: jax.Array,
    phase: jax.Array,
    states: State,
    distance: jax.Array,
    deadline: jax.Array,
):
    def set_field(field, value):
        return field.at[:, unit].set(jnp.where(assign, value, field[:, unit]))

    return tasks._replace(
        task_type=set_field(tasks.task_type, task_type.astype(jnp.int8)),
        target_id=set_field(tasks.target_id, target_id.astype(jnp.int16)),
        target_x=set_field(tasks.target_x, target_x.astype(jnp.int8)),
        target_y=set_field(tasks.target_y, target_y.astype(jnp.int8)),
        item_id=set_field(tasks.item_id, item_id.astype(jnp.int8)),
        quantity=set_field(tasks.quantity, jnp.ones_like(states.step, dtype=jnp.int16)),
        phase=set_field(tasks.phase, phase.astype(jnp.int8)),
        start_step=set_field(tasks.start_step, states.step.astype(jnp.int16)),
        last_progress_step=set_field(tasks.last_progress_step, states.step.astype(jnp.int16)),
        expected_finish_step=set_field(
            tasks.expected_finish_step, (states.step + distance + 1).astype(jnp.int16)
        ),
        deadline_step=set_field(tasks.deadline_step, deadline.astype(jnp.int16)),
        status=set_field(tasks.status, jnp.full_like(states.step, TaskStatusV1.ACTIVE, dtype=jnp.int8)),
        failure_code=set_field(tasks.failure_code, jnp.full_like(states.step, FailureCodeV1.NONE, dtype=jnp.int8)),
    )


def _inventory_value(states: State, player: int, unit: int) -> jax.Array:
    return jnp.sum(
        states.unit_inventory[:, player, unit, :NUM_PRODUCTS].astype(jnp.int32)
        * states.market_price.astype(jnp.int32),
        axis=-1,
        dtype=jnp.int32,
    )


def materialize_m26_unit_tasks_v2(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M26CropGenomeV2,
    player: int,
    eligible_units: jax.Array | None = None,
    *,
    planning_target: jax.Array | None = None,
    externally_reserved_tiles: jax.Array | None = None,
) -> ProjectControllerStateV2:
    """Deterministic M2.6 baseline scheduler with five-crop decisions.

    ``planning_target`` may be lower than the current-day operating target when
    a higher-level calendar knows that a crop family contracts next day.  It
    only changes replacement/plant admission and excess accounting; existing
    crops are still watered, harvested and banked by the same executor.  The
    standalone M2.6 behaviour is unchanged when the override is omitted.
    """

    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    if eligible_units is None:
        eligible_units = jnp.ones((batch_size, MAX_UNITS), dtype=jnp.bool_)
    _, target, _, hand_target = m26_phase_targets_v2(states, genome)
    if planning_target is not None:
        target = planning_target.astype(jnp.int16)
    tasks = controller.unit_tasks
    kind = states.tile_kind[:, player].reshape(batch_size, -1)
    crop = states.tile_crop[:, player].reshape(batch_size, -1)
    flags = states.tile_flags[:, player].reshape(batch_size, -1)
    origin = states.tile_origin_day[:, player].reshape(batch_size, -1).astype(jnp.int16)
    tile_yield = states.tile_yield[:, player].reshape(batch_size, -1).astype(jnp.int16)
    fertilized_until = states.tile_fertilized_until[:, player].reshape(batch_size, -1).astype(jnp.int16)
    remembered = controller.tile_project_id.reshape(batch_size, -1)
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)
    current = _active_crop_counts(states, player)
    centroid_x, centroid_y, crop_count = _crop_centroids(states, player)

    active_task = tasks.status == TaskStatusV1.ACTIVE
    active_target_kind = states.tile_kind[
        batch[:, None],
        player,
        jnp.clip(tasks.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1),
        jnp.clip(tasks.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1),
    ]
    pending_plant = (
        active_task
        & (tasks.task_type == TaskTypeV1.CROP_PRODUCTION)
        & (active_target_kind == TileKind.EMPTY)
    )
    pending_crop_id = jnp.clip(tasks.item_id.astype(jnp.int32), 0, NUM_CROPS - 1)
    pending_by_crop = jnp.sum(
        jax.nn.one_hot(pending_crop_id, NUM_CROPS, dtype=jnp.int16)
        * pending_plant[..., None],
        axis=1,
        dtype=jnp.int16,
    )
    deficit = jnp.maximum(target - current - pending_by_crop, 0)
    excess = jnp.maximum(current - target, 0)

    reserved = (
        jnp.zeros((batch_size, BOARD_SIZE * BOARD_SIZE), dtype=jnp.bool_)
        if externally_reserved_tiles is None
        else jnp.asarray(externally_reserved_tiles, dtype=jnp.bool_)
    )
    reserve_task = active_task & (
        (tasks.task_type == TaskTypeV1.CROP_PRODUCTION)
        | (tasks.task_type == TaskTypeV1.WATER_CROP)
        | (tasks.task_type == TaskTypeV1.APPLY_FERTILIZER)
        | (tasks.task_type == TaskTypeV1.CLEAR_OR_REMOVE_TILE)
    )
    task_target = jnp.clip(tasks.target_id.astype(jnp.int32), 0, BOARD_SIZE * BOARD_SIZE - 1)
    reserved = reserved.at[batch[:, None], task_target].max(reserve_task)

    assigned_plant = jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int16)
    assigned_clear = jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int16)
    seed_inventory = states.seeds[:, player].astype(jnp.int16)
    day_end = (((states.step // TURNS_PER_DAY) + 1) * TURNS_PER_DAY - 1).astype(jnp.int16)
    turns_remaining = (TURNS_PER_DAY - (states.step % TURNS_PER_DAY)).astype(jnp.int16)
    closing = states.step >= genome.liquidation_start_step
    expected_units = 1 + hand_target.astype(jnp.int16)
    maintenance_capacity = jnp.floor(
        expected_units.astype(jnp.float32)
        * TURNS_PER_DAY
        * genome.maintenance_utilization_cap
    ).astype(jnp.int16)
    maintenance_admitted = jnp.sum(target, axis=-1, dtype=jnp.int16) <= maintenance_capacity
    shed_used = jnp.sum(states.shed[:, player].astype(jnp.int32), axis=-1)
    shed_space = shed_used < 100
    unit_has_fertilizer = (
        states.unit_inventory[:, player, :, _FERTILIZER_ITEM].astype(jnp.int16) > 0
    )
    active_fertilizer_task = active_task & (
        tasks.task_type == TaskTypeV1.APPLY_FERTILIZER
    )
    reserved_shed_fertilizer = jnp.sum(
        active_fertilizer_task & (~unit_has_fertilizer), axis=-1, dtype=jnp.int16
    )
    shed_fertilizer = states.shed[:, player, _FERTILIZER_ITEM].astype(jnp.int16)
    assigned_fertilizer_pickups = jnp.zeros((batch_size,), dtype=jnp.int16)

    for unit in range(MAX_UNITS):
        position = states.unit_pos[:, player, unit].astype(jnp.int16)
        free = (
            states.unit_active[:, player, unit]
            & eligible_units[:, unit]
            & (tasks.status[:, unit] != TaskStatusV1.ACTIVE)
        )
        inventory_value = _inventory_value(states, player, unit)
        carrying = jnp.sum(
            states.unit_inventory[:, player, unit, :NUM_PRODUCTS].astype(jnp.int32), axis=-1
        ) > 0
        depot, depot_distance = _nearest_shed(position)
        at_shed = depot_distance == 0
        deposit_override = (
            closing
            | at_shed
            | (turns_remaining <= depot_distance + 1)
        )
        deposit = (
            free
            & carrying
            & shed_space
            & (deposit_override | (inventory_value >= genome.deposit_min_value))
        )

        unwatered = (
            (kind == TileKind.PLANT)
            & ((flags & jnp.uint8(FLAG_WATERED)) == 0)
            & (~reserved)
        )
        water_target, water_valid, water_distance = _choose_nearest(unwatered, position)
        safe_water_target = jnp.clip(
            water_target, 0, BOARD_SIZE * BOARD_SIZE - 1
        )
        water_target_yield = tile_yield[batch, safe_water_target]
        water_position = jnp.stack(
            (_TILE_X[safe_water_target], _TILE_Y[safe_water_target]), axis=-1
        )
        _, water_to_shed = _nearest_shed(water_position)
        # In liquidation, watering is useful only if the resulting crop can
        # still be harvested, returned, deposited and sold before step 719.
        water_bank_steps = (
            water_distance + 1 + 1 + water_to_shed + 1 + 1
        ).astype(jnp.int16)
        closing_water_bankable = (
            states.step.astype(jnp.int32)
            + water_bank_steps.astype(jnp.int32)
            <= EPISODE_STEPS - 1
        )
        water = (
            free
            & (~deposit)
            & water_valid
            & ((~closing) | (water_target_yield <= 0))
            & ((~closing) | closing_water_bankable)
            & (water_distance + 1 <= turns_remaining)
        )

        safe_crop = jnp.clip(crop.astype(jnp.int32), 0, NUM_CROPS - 1)
        min_age = genome.harvest_min_age_days[batch[:, None], safe_crop]
        trigger = genome.harvest_trigger_units[batch[:, None], safe_crop]
        age = current_day[:, None] - origin
        mature = (
            (kind == TileKind.PLANT)
            & (tile_yield > 0)
            & (age >= min_age)
            & ((tile_yield >= trigger) | closing[:, None])
            & (~reserved)
        )
        harvest_target, harvest_valid, harvest_distance = _choose_nearest(mature, position)
        safe_harvest_target = jnp.clip(harvest_target, 0, BOARD_SIZE * BOARD_SIZE - 1)
        harvest_position = jnp.stack(
            (_TILE_X[safe_harvest_target], _TILE_Y[safe_harvest_target]), axis=-1
        )
        _, harvest_to_shed = _nearest_shed(harvest_position)
        # A newly harvested item cannot be sold by the market phase of the same
        # step because the sell order was compiled from the pre-step shed.  It
        # therefore needs travel-to-crop, harvest, travel-to-shed, deposit and
        # one following market step before the final executable action (718).
        harvest_bank_steps = (
            harvest_distance + 1 + harvest_to_shed + 1 + 1
        ).astype(jnp.int16)
        harvest_bankable = (
            states.step.astype(jnp.int32) + harvest_bank_steps.astype(jnp.int32)
            <= EPISODE_STEPS - 1
        )
        harvest = free & (~deposit) & (~water) & harvest_valid & harvest_bankable

        selected_harvest_crop = safe_crop[batch, jnp.clip(harvest_target, 0, 99)]
        fertilizer_mode = genome.fertilizer_policy[batch[:, None], safe_crop]
        fertilizer_enabled = (fertilizer_mode == M26FertilizerPolicyV2.ALWAYS_WHEN_AVAILABLE) | (
            (fertilizer_mode == M26FertilizerPolicyV2.HIGH_VALUE_ONLY) & (safe_crop >= 2)
        )
        fertilizer_resource = unit_has_fertilizer[:, unit] | (
            shed_fertilizer
            > reserved_shed_fertilizer + assigned_fertilizer_pickups
        )
        fertilize_mask = (
            (kind == TileKind.PLANT)
            & fertilizer_enabled
            & (fertilized_until <= current_day[:, None])
            & (~reserved)
        )
        fertilize_target, fertilize_valid, fertilize_distance = _choose_nearest(
            fertilize_mask, position
        )
        safe_fertilize_target = jnp.clip(
            fertilize_target, 0, BOARD_SIZE * BOARD_SIZE - 1
        )
        fertilize_position = jnp.stack(
            (_TILE_X[safe_fertilize_target], _TILE_Y[safe_fertilize_target]), axis=-1
        )
        depot_to_fertilize = jnp.sum(
            jnp.abs(depot.astype(jnp.int16) - fertilize_position.astype(jnp.int16)),
            axis=-1,
        ).astype(jnp.int16)
        fertilize_total_steps = jnp.where(
            unit_has_fertilizer[:, unit],
            fertilize_distance + 1,
            depot_distance + 1 + depot_to_fertilize + 1,
        ).astype(jnp.int16)
        fertilize = (
            free
            & (~deposit)
            & (~water)
            & (~harvest)
            & (~closing)
            & fertilizer_resource
            & fertilize_valid
            & (fertilize_total_steps <= turns_remaining)
        )

        remembered_crop = jnp.clip(remembered, 0, NUM_CROPS - 1)
        weed_policy = genome.weed_recovery_policy[batch[:, None], remembered_crop]
        recover_weed = (
            (kind == TileKind.WEED)
            & (remembered >= 0)
            & (remembered < NUM_CROPS)
            & (weed_policy != M26WeedRecoveryPolicyV2.ABANDON_TILE)
            & (~reserved)
        )
        weed_target, weed_valid, weed_distance = _choose_nearest(recover_weed, position)
        clear_weed = (
            free
            & (~deposit)
            & (~water)
            & (~harvest)
            & (~fertilize)
            & weed_valid
        )

        tile_policy = genome.crop_abandon_policy[batch[:, None], safe_crop]
        remaining_excess = jnp.maximum(excess - assigned_clear, 0)
        tile_excess = remaining_excess[batch[:, None], safe_crop] > 0
        any_replacement = jnp.any((deficit - assigned_plant) > 0, axis=-1)
        days_needed = jnp.maximum(
            genome.harvest_min_age_days[batch[:, None], safe_crop] - age, 0
        )
        not_bankable = states.step[:, None] + days_needed * TURNS_PER_DAY >= genome.liquidation_start_step[:, None]
        abandon_mask = (
            (kind == TileKind.PLANT)
            & tile_excess
            & any_replacement[:, None]
            & (
                (tile_policy == M26CropAbandonPolicyV2.ABANDON_IF_TARGET_SHRINKS_AND_REPLACEMENT_IS_BETTER)
                | (
                    (tile_policy == M26CropAbandonPolicyV2.ABANDON_ONLY_IF_NOT_BANKABLE)
                    & not_bankable
                )
            )
            & (~reserved)
        )
        abandon_target, abandon_valid, abandon_distance = _choose_nearest(abandon_mask, position)
        abandon = (
            free
            & (~deposit)
            & (~water)
            & (~harvest)
            & (~fertilize)
            & (~clear_weed)
            & abandon_valid
        )

        remaining = jnp.maximum(deficit - assigned_plant, 0)
        crop_slot = jnp.arange(NUM_CROPS, dtype=jnp.int16)[None]
        within_wave = pending_by_crop + assigned_plant < genome.plant_wave_size.astype(jnp.int16)
        within_cash_cap = (
            (current + pending_by_crop + assigned_plant + 1) * _SEED_COST[None]
            <= genome.crop_cash_cap
        )
        plantable_crop = (
            (remaining > 0)
            & (seed_inventory - assigned_plant > 0)
            & within_wave
            & within_cash_cap
            & (states.step[:, None] < genome.crop_last_plant_step)
            & maintenance_admitted[:, None]
        )
        crop_score = jnp.where(plantable_crop, remaining.astype(jnp.int32) * 100 - crop_slot, -1)
        chosen_crop = jnp.argmax(crop_score, axis=-1).astype(jnp.int32)
        crop_available = jnp.max(crop_score, axis=-1) >= 0
        layout = genome.crop_layout_policy[batch, chosen_crop]
        # In M3.5 an empty tile may already be reserved by an animal project.
        # Standalone M2.6 never produces project ids >= NUM_CROPS, so this is a
        # no-op for the frozen crop milestone and prevents cross-project theft
        # in the joint controller.
        crop_available_tile = (remembered < 0) | (remembered < NUM_CROPS)
        empty = (kind == TileKind.EMPTY) & crop_available_tile & (~reserved)
        plant_target, plant_valid, plant_distance = _choose_m26_tile(
            empty,
            position,
            chosen_crop,
            layout,
            centroid_x,
            centroid_y,
            crop_count,
            remembered,
        )
        plant = (
            free
            & (~deposit)
            & (~water)
            & (~harvest)
            & (~fertilize)
            & (~clear_weed)
            & (~abandon)
            & (~closing)
            & crop_available
            & plant_valid
            & (plant_distance + 2 <= turns_remaining)
        )

        selected_target = jnp.where(
            water,
            water_target,
            jnp.where(
                harvest,
                harvest_target,
                jnp.where(
                    fertilize,
                    fertilize_target,
                    jnp.where(clear_weed, weed_target, jnp.where(abandon, abandon_target, plant_target)),
                ),
            ),
        ).astype(jnp.int32)
        safe_target = jnp.clip(selected_target, 0, BOARD_SIZE * BOARD_SIZE - 1)
        selected_x = _TILE_X[safe_target]
        selected_y = _TILE_Y[safe_target]
        selected_tile_crop = safe_crop[batch, safe_target]
        selected_weed_crop = remembered_crop[batch, jnp.clip(weed_target, 0, 99)]
        selected_item = jnp.where(
            plant,
            chosen_crop,
            jnp.where(
                fertilize,
                _FERTILIZER_ITEM,
                jnp.where(clear_weed, selected_weed_crop, selected_tile_crop),
            ),
        ).astype(jnp.int8)
        task_type = jnp.where(
            deposit,
            TaskTypeV1.SHED_DEPOSIT,
            jnp.where(
                water,
                TaskTypeV1.WATER_CROP,
                jnp.where(
                    fertilize,
                    TaskTypeV1.APPLY_FERTILIZER,
                    jnp.where(
                        clear_weed | abandon,
                        TaskTypeV1.CLEAR_OR_REMOVE_TILE,
                        TaskTypeV1.CROP_PRODUCTION,
                    ),
                ),
            ),
        ).astype(jnp.int8)
        task_phase = jnp.where(
            deposit, TaskPhaseV1.MOVE_TO_DEPOT, TaskPhaseV1.MOVE_TO_TARGET
        ).astype(jnp.int8)
        final_target = jnp.where(
            deposit,
            depot[:, 1].astype(jnp.int32) * BOARD_SIZE + depot[:, 0].astype(jnp.int32),
            selected_target,
        )
        final_x = jnp.where(deposit, depot[:, 0], selected_x)
        final_y = jnp.where(deposit, depot[:, 1], selected_y)
        distance = jnp.where(
            deposit,
            depot_distance,
            jnp.where(
                water,
                water_distance,
                jnp.where(
                    harvest,
                    harvest_distance,
                    jnp.where(
                        fertilize,
                        jnp.maximum(fertilize_total_steps - 1, 0),
                        jnp.where(clear_weed, weed_distance, jnp.where(abandon, abandon_distance, plant_distance)),
                    ),
                ),
            ),
        ).astype(jnp.int16)
        deadline = jnp.where(
            water | plant,
            day_end,
            jnp.where(closing, jnp.int16(EPISODE_STEPS - 2), genome.liquidation_start_step),
        ).astype(jnp.int16)
        map_task = water | harvest | fertilize | clear_weed | abandon | plant
        assign = deposit | map_task
        tasks = _set_unit_task(
            tasks,
            unit,
            assign,
            task_type=task_type,
            target_id=final_target,
            target_x=final_x,
            target_y=final_y,
            item_id=selected_item,
            phase=task_phase,
            states=states,
            distance=distance,
            deadline=deadline,
        )
        reserved = reserved.at[batch, safe_target].set(reserved[batch, safe_target] | map_task)
        assigned_plant = assigned_plant.at[batch, chosen_crop].add(plant.astype(jnp.int16))
        assigned_fertilizer_pickups = assigned_fertilizer_pickups + (
            fertilize & (~unit_has_fertilizer[:, unit])
        ).astype(jnp.int16)
        cleared_crop = safe_crop[batch, jnp.clip(abandon_target, 0, 99)]
        assigned_clear = assigned_clear.at[batch, cleared_crop].add(abandon.astype(jnp.int16))

    return controller._replace(unit_tasks=tasks)


def _hire_prefix_cost(current_hires: jax.Array) -> jax.Array:
    offsets = jnp.arange(10, dtype=jnp.int32)[None]
    indices = jnp.clip(current_hires[:, None].astype(jnp.int32) + offsets, 0, len(HIRE_COST) - 1)
    return jnp.cumsum(_HIRE_COST[indices], axis=-1, dtype=jnp.int32)


def _visible_town_demand(states: State) -> jax.Array:
    safe = jnp.clip(states.town_shops.astype(jnp.int32), 0, _SHOP_DEMAND.shape[0] - 1)
    slots = jnp.arange(states.town_shops.shape[1], dtype=jnp.int8)[None]
    active = slots < states.town_count[:, None]
    return jnp.sum(_SHOP_DEMAND[safe] * active[..., None], axis=1, dtype=jnp.int16)


def _allocate_units_for_pressure(available: jax.Array, required: jax.Array) -> jax.Array:
    selected = jnp.zeros_like(available, dtype=jnp.int16)
    remaining = required.astype(jnp.int32)
    for product in range(NUM_PRODUCTS):
        take = jnp.minimum(available[:, product].astype(jnp.int32), remaining)
        selected = selected.at[:, product].set(take.astype(jnp.int16))
        remaining = jnp.maximum(remaining - take, 0)
    return selected


def _allocate_units_for_cash(
    available: jax.Array, prices: jax.Array, required_value: jax.Array
) -> jax.Array:
    selected = jnp.zeros_like(available, dtype=jnp.int16)
    remaining = required_value.astype(jnp.int32)
    for product in range(NUM_PRODUCTS):
        price = jnp.maximum(prices[:, product].astype(jnp.int32), 1)
        need = (remaining + price - 1) // price
        take = jnp.minimum(available[:, product].astype(jnp.int32), need)
        selected = selected.at[:, product].set(take.astype(jnp.int16))
        remaining = jnp.maximum(remaining - take * price, 0)
    return selected


def materialize_m26_market_tasks_v2(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M26CropGenomeV2,
    player: int,
    tables: StaticTables | None = None,
    *,
    allow_closing_hires: bool = False,
) -> ProjectControllerStateV2:
    """Create product-aware sells and cash-safe land/seed/hire intents."""

    tables = _DEFAULT_TABLES if tables is None else tables
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    phase, target, desired_land, hand_target = m26_phase_targets_v2(states, genome)
    closing = states.step >= genome.liquidation_start_step
    current = _active_crop_counts(states, player)
    seeds = states.seeds[:, player].astype(jnp.int16)
    expected_units = 1 + hand_target.astype(jnp.int16)
    capacity = jnp.floor(
        expected_units.astype(jnp.float32) * TURNS_PER_DAY * genome.maintenance_utilization_cap
    ).astype(jnp.int16)
    admitted = jnp.sum(target, axis=-1, dtype=jnp.int16) <= capacity

    unlocked = states.unlocked_count[:, player].astype(jnp.int32)
    phase_land_start = _phase_value(genome.land_start_step, phase)
    land_needed = unlocked < desired_land.astype(jnp.int32)
    land_cost = _LAND_COST[jnp.clip(unlocked - 1, 0, len(LAND_PRICES) - 1)]
    land_window = states.step >= phase_land_start

    raw_deficit = jnp.maximum(target - current - seeds, 0)
    within_stop = states.step[:, None] < genome.crop_last_plant_step
    cap_remaining_units = jnp.maximum(
        genome.crop_cash_cap // _SEED_COST[None] - current.astype(jnp.int32) - seeds.astype(jnp.int32),
        0,
    ).astype(jnp.int16)
    seed_demand = jnp.minimum(raw_deficit, cap_remaining_units)
    seed_demand = jnp.where(within_stop & admitted[:, None], seed_demand, 0)
    seed_plan = jnp.minimum(seed_demand, genome.seed_buy_batch).astype(jnp.int16)

    immediate_seed_cost = jnp.sum(seed_plan.astype(jnp.int32) * _SEED_COST[None], axis=-1)
    immediate_land_cost = jnp.where(land_needed & land_window, land_cost, 0)
    committed_cost = immediate_seed_cost + immediate_land_cost

    shed = states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int16)
    fertilizer_needed = jnp.any(
        (target > 0)
        & (genome.fertilizer_policy != M26FertilizerPolicyV2.OFF),
        axis=-1,
    )
    fertilizer_stock = (
        states.shed[:, player, _FERTILIZER_ITEM].astype(jnp.int16)
        + jnp.sum(states.unit_inventory[:, player, :, _FERTILIZER_ITEM].astype(jnp.int16), axis=-1)
    )
    reserved_fertilizer = fertilizer_needed & (~closing)
    sellable = shed.at[:, _FERTILIZER_ITEM].set(
        jnp.maximum(shed[:, _FERTILIZER_ITEM] - reserved_fertilizer.astype(jnp.int16), 0)
    )

    due = (
        jnp.mod(states.step[:, None] - genome.sell_phase, jnp.maximum(genome.sell_interval, 1))
        == 0
    )
    town_demand = _visible_town_demand(states)
    supply_glut = states.market_inventory > MARKET_INITIAL_INVENTORY
    effective_floor = genome.sell_price_floor_ratio + 0.05 * (town_demand > 0) + 0.05 * supply_glut
    price_ok = states.market_price.astype(jnp.float32) >= _BASE_PRICE[None] * effective_floor
    normal_quantity = jnp.ceil(sellable.astype(jnp.float32) * genome.sell_fraction).astype(jnp.int16)
    normal_quantity = jnp.where(due & price_ok, normal_quantity, 0)

    shed_used = jnp.sum(states.shed[:, player].astype(jnp.int32), axis=-1)
    pressure_units = jnp.maximum(shed_used - genome.shed_pressure_trigger.astype(jnp.int32) + 1, 0)
    pressure_quantity = _allocate_units_for_pressure(sellable, pressure_units)
    cash_need = jnp.maximum(genome.cash_floor + committed_cost - states.money[:, player], 0)
    cash_quantity = _allocate_units_for_cash(sellable, states.market_price, cash_need)
    sell_quantity = jnp.maximum(normal_quantity, jnp.maximum(pressure_quantity, cash_quantity))
    sell_quantity = jnp.where(closing[:, None], shed, sell_quantity)

    product_ids = jnp.broadcast_to(
        jnp.arange(NUM_PRODUCTS, dtype=jnp.int8)[None], sell_quantity.shape
    )
    estimated_sell_value = jnp.sum(
        exact_market_quote_v1(
            states,
            tables,
            product_ids,
            sell_quantity,
            buy=False,
            player=player,
        ),
        axis=-1,
        dtype=jnp.int32,
    )
    available = jnp.maximum(
        states.money[:, player] + estimated_sell_value - genome.cash_floor, 0
    ).astype(jnp.int32)
    buy_land = (
        (~closing)
        & land_window
        & land_needed
        & (available >= land_cost)
    )
    available = available - jnp.where(buy_land, land_cost, 0)

    seed_buy = jnp.zeros_like(seed_plan, dtype=jnp.int16)
    for crop_id in range(NUM_CROPS):
        affordable = available // jnp.maximum(_SEED_COST[crop_id], 1)
        quantity = jnp.minimum(seed_plan[:, crop_id].astype(jnp.int32), affordable).astype(jnp.int16)
        quantity = jnp.where(~closing, quantity, 0)
        seed_buy = seed_buy.at[:, crop_id].set(quantity)
        available = available - quantity.astype(jnp.int32) * _SEED_COST[crop_id]

    fertilizer_buy = (
        (~closing)
        & fertilizer_needed
        & (fertilizer_stock <= 0)
        & (available >= states.market_price[:, _FERTILIZER_ITEM])
    )
    available = available - jnp.where(
        fertilizer_buy, states.market_price[:, _FERTILIZER_ITEM], 0
    )

    current_hires = states.hires_today[:, player].astype(jnp.int16)
    desired_hires = jnp.clip(hand_target.astype(jnp.int16) - current_hires, 0, 10)
    hire_prefix = _hire_prefix_cost(current_hires)
    hire_slot = jnp.arange(10, dtype=jnp.int16)[None]
    last_turn = ((states.step + 1) % TURNS_PER_DAY) == 0
    hire_present = (
        ((~closing) | allow_closing_hires)[:, None]
        & (~last_turn)[:, None]
        & (hire_slot < desired_hires[:, None])
        & (hire_prefix <= available[:, None])
    )

    candidate_count = NUM_PRODUCTS + 1 + NUM_CROPS + 1 + 10
    present = jnp.zeros((batch_size, candidate_count), dtype=jnp.bool_)
    task_type = jnp.zeros((batch_size, candidate_count), dtype=jnp.int8)
    item = jnp.full((batch_size, candidate_count), -1, dtype=jnp.int8)
    quantity = jnp.zeros((batch_size, candidate_count), dtype=jnp.int16)

    present = present.at[:, :NUM_PRODUCTS].set(sell_quantity > 0)
    task_type = task_type.at[:, :NUM_PRODUCTS].set(
        jnp.where(
            closing[:, None],
            TaskTypeV1.TERMINAL_LIQUIDATION,
            TaskTypeV1.SELL_INVENTORY,
        ).astype(jnp.int8)
    )
    item = item.at[:, :NUM_PRODUCTS].set(
        jnp.broadcast_to(jnp.arange(NUM_PRODUCTS, dtype=jnp.int8), (batch_size, NUM_PRODUCTS))
    )
    quantity = quantity.at[:, :NUM_PRODUCTS].set(sell_quantity)

    land_slot = NUM_PRODUCTS
    present = present.at[:, land_slot].set(buy_land)
    task_type = task_type.at[:, land_slot].set(jnp.int8(TaskTypeV1.BUY_LAND))
    quantity = quantity.at[:, land_slot].set(jnp.int16(1))

    seed_start = land_slot + 1
    present = present.at[:, seed_start : seed_start + NUM_CROPS].set(seed_buy > 0)
    task_type = task_type.at[:, seed_start : seed_start + NUM_CROPS].set(
        jnp.int8(TaskTypeV1.CROP_PRODUCTION)
    )
    item = item.at[:, seed_start : seed_start + NUM_CROPS].set(
        jnp.broadcast_to(jnp.arange(NUM_CROPS, dtype=jnp.int8), (batch_size, NUM_CROPS))
    )
    quantity = quantity.at[:, seed_start : seed_start + NUM_CROPS].set(seed_buy)

    fertilizer_slot = seed_start + NUM_CROPS
    present = present.at[:, fertilizer_slot].set(fertilizer_buy)
    task_type = task_type.at[:, fertilizer_slot].set(jnp.int8(TaskTypeV1.BUY_PRODUCT))
    item = item.at[:, fertilizer_slot].set(jnp.int8(_FERTILIZER_ITEM))
    quantity = quantity.at[:, fertilizer_slot].set(jnp.int16(1))

    hire_start = fertilizer_slot + 1
    present = present.at[:, hire_start : hire_start + 10].set(hire_present)
    task_type = task_type.at[:, hire_start : hire_start + 10].set(jnp.int8(TaskTypeV1.HIRE_WORKER))
    quantity = quantity.at[:, hire_start : hire_start + 10].set(jnp.int16(1))

    def pack_lane(p, t, i, q):
        indices = jnp.nonzero(p, size=MAX_MARKET_ORDERS, fill_value=-1)[0]
        valid = indices >= 0
        safe = jnp.clip(indices, 0, candidate_count - 1)
        return (
            valid,
            jnp.where(valid, t[safe], 0).astype(jnp.int8),
            jnp.where(valid, i[safe], -1).astype(jnp.int8),
            jnp.where(valid, q[safe], 0).astype(jnp.int16),
        )

    packed_valid, packed_type, packed_item, packed_quantity = jax.vmap(pack_lane)(
        present, task_type, item, quantity
    )
    desired = empty_market_tasks_v2(batch_size)._replace(
        task_type=packed_type,
        item_id=packed_item,
        quantity=packed_quantity,
        start_step=jnp.where(packed_valid, states.step[:, None], -1).astype(jnp.int16),
        deadline_step=jnp.where(
            packed_valid, jnp.minimum(states.step[:, None] + 1, EPISODE_STEPS - 2), -1
        ).astype(jnp.int16),
        status=jnp.where(packed_valid, TaskStatusV1.ACTIVE, TaskStatusV1.EMPTY).astype(jnp.int8),
        failure_code=jnp.zeros((batch_size, MAX_MARKET_ORDERS), dtype=jnp.int8),
    )
    active = controller.market_tasks.status == TaskStatusV1.ACTIVE
    market = jax.tree.map(
        lambda old, new: jnp.where(active, old, new), controller.market_tasks, desired
    )
    return controller._replace(market_tasks=market)


def m26_policy_step_v2(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M26CropGenomeV2,
    player: int,
    tables: StaticTables | None = None,
) -> tuple[E4PlayerActionV1, ProjectControllerStateV2]:
    controller, _ = reconcile_project_controller_v2(states, controller, player)
    controller = clear_invalidated_crop_tasks_v2(states, controller, player)
    controller = ensure_m26_projects_v2(states, controller, genome, player)
    controller = materialize_m26_unit_tasks_v2(states, controller, genome, player)
    controller = materialize_m26_market_tasks_v2(
        states, controller, genome, player, tables
    )
    return compile_full_core_player_action_v1(states, controller, player), controller


def update_m26_controller_from_effects_v2(
    states: State,
    next_states: State,
    controller: ProjectControllerStateV2,
    action: E4PlayerActionV1,
    genome: M26CropGenomeV2,
    player: int,
) -> tuple[ProjectControllerStateV2, FullCoreEffectDiagnosticsV1]:
    controller, diagnostics = update_full_core_controller_from_effects_v1(
        states, next_states, controller, action, player
    )
    controller = ensure_m26_projects_v2(next_states, controller, genome, player)
    controller = controller._replace(
        unexplained_effect_failures=(
            controller.unexplained_effect_failures
            + diagnostics.effect_mismatch_count
            + diagnostics.owner_inactive_count
            + diagnostics.deadline_missed_count
            + diagnostics.resource_unavailable_count
        ).astype(jnp.int32)
    )
    return controller, diagnostics


def m26_player_action_dict_v2(action: E4PlayerActionV1) -> dict:
    return {
        "unit_op": action.unit_op,
        "unit_item": action.unit_item,
        "unit_amount": action.unit_amount,
        "unit_count": action.unit_count,
        "market_op": action.market_op,
        "market_item": action.market_item,
        "market_amount": action.market_amount,
        "market_count": action.market_count,
    }


__all__ = [
    "ensure_m26_projects_v2",
    "m26_phase_targets_v2",
    "m26_phase_v2",
    "m26_player_action_dict_v2",
    "m26_policy_step_v2",
    "materialize_m26_market_tasks_v2",
    "materialize_m26_unit_tasks_v2",
    "update_m26_controller_from_effects_v2",
]
