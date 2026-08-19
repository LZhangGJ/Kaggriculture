"""M2.5 staged crop expansion, market planning, and multi-unit scheduling.

The module deliberately reuses the parity-tested Strategic V5 Full-core
compiler.  It owns only the higher-level decisions: staged capacity targets,
cash-safe market orders, and collision-free unit-to-task assignment.
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
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_CROPS,
    NUM_PRODUCTS,
    SHED_ACCESS,
    TURNS_PER_DAY,
    TileKind,
)
from kaggriculture_jax.types import State
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

from .constants import ProjectStatusV2, ProjectTypeV2
from .crop_executor import clear_invalidated_crop_tasks_v2
from .lifecycle import (
    empty_market_tasks_v2,
    reconcile_project_controller_v2,
)
from .schema import M25CropExpansionConfigV2, ProjectControllerStateV2


_TILE_X = jnp.tile(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
_TILE_Y = jnp.repeat(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
_TILE_ID = jnp.arange(BOARD_SIZE * BOARD_SIZE, dtype=jnp.int32)
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)
_FIRST_YIELD = jnp.asarray(CROP_FIRST_YIELD_DAY, dtype=jnp.int16)
_SEED_COST = jnp.asarray(CROP_SEED_COST, dtype=jnp.int32)
_LAND_COST = jnp.asarray(LAND_PRICES, dtype=jnp.int32)
_HIRE_COST = jnp.asarray(HIRE_COST, dtype=jnp.int32)
_INVALID_SCORE = jnp.int32(1_000_000)


def default_r2_tomato_m25_config_v2(batch_size: int) -> M25CropExpansionConfigV2:
    """Return the old candidate-101 R2 staged specification in JAX form."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    def one(value, dtype):
        return jnp.full((batch_size,), value, dtype=dtype)

    def stages(values, dtype):
        return jnp.broadcast_to(jnp.asarray(values, dtype=dtype), (batch_size, 3))

    return M25CropExpansionConfigV2(
        primary_crop_id=one(2, jnp.int8),  # TOMATO
        support_crop_id=one(0, jnp.int8),  # WHEAT
        phase_start_step=stages((0, 5 * 24, 14 * 24), jnp.int16),
        primary_target=stages((12, 18, 28), jnp.int16),
        support_target=stages((0, 8, 6), jnp.int16),
        hand_target=stages((9, 11, 7), jnp.int16),
        land_target=stages((1, 2, 2), jnp.int8),
        land_start_step=stages((0, 6 * 24, 6 * 24), jnp.int16),
        primary_seed_batch=one(16, jnp.int16),
        support_seed_batch=one(1, jnp.int16),
        hire_batch_max=one(10, jnp.int8),
        parallel_plant_lanes=one(4, jnp.int8),
        harvest_dispatch_lanes=one(2, jnp.int8),
        deposit_lanes=one(2, jnp.int8),
        cash_floor=one(500, jnp.int32),
        sell_interval=one(48, jnp.int16),
        sell_phase=one(0, jnp.int16),
        investment_stop_step=one(28 * 24, jnp.int16),
        liquidation_start_step=one(28 * 24, jnp.int16),
    )


def m25_phase_v2(states: State, config: M25CropExpansionConfigV2) -> jax.Array:
    crossed = states.step[:, None] >= config.phase_start_step
    return jnp.clip(jnp.sum(crossed, axis=-1) - 1, 0, 2).astype(jnp.int8)


def _phase_value(values: jax.Array, phase: jax.Array) -> jax.Array:
    batch = jnp.arange(phase.shape[0], dtype=jnp.int32)
    return values[batch, phase.astype(jnp.int32)]


def _desired_land_count(
    states: State, config: M25CropExpansionConfigV2
) -> jax.Array:
    phase = m25_phase_v2(states, config)
    return jnp.clip(_phase_value(config.land_target, phase), 1, 4).astype(jnp.int8)


def _land_purchase_window_open(
    states: State, config: M25CropExpansionConfigV2
) -> jax.Array:
    phase = m25_phase_v2(states, config)
    return states.step >= _phase_value(config.land_start_step, phase)


def ensure_m25_projects_v2(
    states: State,
    controller: ProjectControllerStateV2,
    config: M25CropExpansionConfigV2,
    player: int,
) -> ProjectControllerStateV2:
    """Maintain primary crop, support crop, land, and workforce project slots."""

    phase = m25_phase_v2(states, config)
    primary_target = _phase_value(config.primary_target, phase)
    support_target = _phase_value(config.support_target, phase)
    hand_target = _phase_value(config.hand_target, phase)
    land_target = _desired_land_count(states, config)
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    primary = config.primary_crop_id.astype(jnp.int32)
    support = config.support_crop_id.astype(jnp.int32)
    kind = states.tile_kind[:, player].reshape(batch_size, -1)
    crop = states.tile_crop[:, player].reshape(batch_size, -1)
    primary_active = jnp.sum(
        (kind == TileKind.PLANT) & (crop == primary[:, None]), axis=-1, dtype=jnp.int16
    )
    support_active = jnp.sum(
        (kind == TileKind.PLANT) & (crop == support[:, None]), axis=-1, dtype=jnp.int16
    )
    projects = controller.projects
    slot_ids = (0, 1, 2, 3)
    types = (
        ProjectTypeV2.CROP_LOT,
        ProjectTypeV2.CROP_LOT,
        ProjectTypeV2.LAND_EXPANSION,
        ProjectTypeV2.WORKFORCE,
    )
    items = (config.primary_crop_id, config.support_crop_id, -1, -1)
    targets = (primary_target, support_target, land_target, hand_target)
    active_counts = (
        primary_active,
        support_active,
        states.unlocked_count[:, player].astype(jnp.int16),
        states.hires_today[:, player].astype(jnp.int16),
    )
    for slot, project_type, item, target, active_count in zip(
        slot_ids, types, items, targets, active_counts, strict=True
    ):
        projects = projects._replace(
            project_id=projects.project_id.at[:, slot].set(jnp.int16(slot)),
            project_type=projects.project_type.at[:, slot].set(jnp.int8(project_type)),
            item_id=projects.item_id.at[:, slot].set(
                jnp.asarray(item, dtype=jnp.int8)
                if hasattr(item, "shape")
                else jnp.full((batch_size,), item, dtype=jnp.int8)
            ),
            status=projects.status.at[:, slot].set(jnp.int8(ProjectStatusV2.ACTIVE)),
            phase=projects.phase.at[:, slot].set(phase),
            target_count=projects.target_count.at[:, slot].set(target.astype(jnp.int16)),
            active_count=projects.active_count.at[:, slot].set(active_count.astype(jnp.int16)),
            start_step=projects.start_step.at[:, slot].set(
                jnp.where(projects.start_step[:, slot] < 0, states.step, projects.start_step[:, slot])
            ),
            stop_step=projects.stop_step.at[:, slot].set(config.investment_stop_step),
            latest_bank_step=projects.latest_bank_step.at[:, slot].set(
                config.liquidation_start_step
            ),
        )
    tile_project_id = jnp.where(
        (states.tile_kind[:, player] == TileKind.PLANT)
        & (states.tile_crop[:, player] == config.primary_crop_id[:, None, None]),
        jnp.int16(0),
        jnp.where(
            (states.tile_kind[:, player] == TileKind.PLANT)
            & (states.tile_crop[:, player] == config.support_crop_id[:, None, None]),
            jnp.int16(1),
            jnp.int16(-1),
        ),
    )
    return controller._replace(
        projects=projects,
        tile_project_id=tile_project_id,
        route_phase=phase,
        liquidation_mode=states.step >= config.liquidation_start_step,
    )


def _nearest_shed(position: jax.Array) -> tuple[jax.Array, jax.Array]:
    distance = jnp.sum(
        jnp.abs(position[:, None, :].astype(jnp.int16) - _SHED_ACCESS[None]), axis=-1
    )
    index = jnp.argmin(distance, axis=-1)
    return _SHED_ACCESS[index], jnp.min(distance, axis=-1).astype(jnp.int16)


def _choose_tile(
    mask: jax.Array,
    position: jax.Array,
    *,
    logistics_weight: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    ux = position[:, 0, None].astype(jnp.int16)
    uy = position[:, 1, None].astype(jnp.int16)
    travel = jnp.abs(_TILE_X[None] - ux) + jnp.abs(_TILE_Y[None] - uy)
    shed_distance = jnp.min(
        jnp.abs(_TILE_X[:, None] - _SHED_ACCESS[None, :, 0])
        + jnp.abs(_TILE_Y[:, None] - _SHED_ACCESS[None, :, 1]),
        axis=-1,
    )
    score = (
        travel.astype(jnp.int32) * 100
        + logistics_weight * shed_distance[None].astype(jnp.int32) * 10
        + _TILE_ID[None]
    )
    score = jnp.where(mask, score, _INVALID_SCORE)
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
        status=set_field(
            tasks.status,
            jnp.full_like(states.step, TaskStatusV1.ACTIVE, dtype=jnp.int8),
        ),
        failure_code=set_field(
            tasks.failure_code,
            jnp.full_like(states.step, FailureCodeV1.NONE, dtype=jnp.int8),
        ),
    )


def materialize_m25_unit_tasks_v2(
    states: State,
    controller: ProjectControllerStateV2,
    config: M25CropExpansionConfigV2,
    player: int,
) -> ProjectControllerStateV2:
    """Assign distinct urgent/production tasks to every free active unit."""

    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    tasks = controller.unit_tasks
    phase_index = m25_phase_v2(states, config)
    primary_target = _phase_value(config.primary_target, phase_index).astype(jnp.int16)
    support_target = _phase_value(config.support_target, phase_index).astype(jnp.int16)
    primary_id = config.primary_crop_id.astype(jnp.int32)
    support_id = config.support_crop_id.astype(jnp.int32)
    kind = states.tile_kind[:, player].reshape(batch_size, -1)
    crop = states.tile_crop[:, player].reshape(batch_size, -1)
    flags = states.tile_flags[:, player].reshape(batch_size, -1)
    origin = states.tile_origin_day[:, player].reshape(batch_size, -1).astype(jnp.int16)
    tile_yield = states.tile_yield[:, player].reshape(batch_size, -1)
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)
    current_primary = jnp.sum(
        (kind == TileKind.PLANT) & (crop == primary_id[:, None]), axis=-1, dtype=jnp.int16
    )
    current_support = jnp.sum(
        (kind == TileKind.PLANT) & (crop == support_id[:, None]), axis=-1, dtype=jnp.int16
    )
    active_task = tasks.status == TaskStatusV1.ACTIVE
    active_plant = active_task & (tasks.task_type == TaskTypeV1.CROP_PRODUCTION)
    active_target_kind = states.tile_kind[
        batch[:, None],
        player,
        jnp.clip(tasks.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1),
        jnp.clip(tasks.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1),
    ]
    pending_plant = active_plant & (active_target_kind == TileKind.EMPTY)
    pending_primary = jnp.sum(
        pending_plant & (tasks.item_id == primary_id[:, None]), axis=-1, dtype=jnp.int16
    )
    pending_support = jnp.sum(
        pending_plant & (tasks.item_id == support_id[:, None]), axis=-1, dtype=jnp.int16
    )
    primary_deficit = jnp.maximum(primary_target - current_primary - pending_primary, 0)
    support_deficit = jnp.maximum(support_target - current_support - pending_support, 0)
    primary_seed = states.seeds[batch, player, primary_id].astype(jnp.int16)
    support_seed = states.seeds[batch, player, support_id].astype(jnp.int16)

    reserved = jnp.zeros((batch_size, BOARD_SIZE * BOARD_SIZE), dtype=jnp.bool_)
    reserve_task = active_task & (
        (tasks.task_type == TaskTypeV1.CROP_PRODUCTION)
        | (tasks.task_type == TaskTypeV1.WATER_CROP)
    )
    target = jnp.clip(tasks.target_id.astype(jnp.int32), 0, BOARD_SIZE * BOARD_SIZE - 1)
    reserved = reserved.at[batch[:, None], target].max(reserve_task)

    plant_lanes = jnp.sum(pending_plant, axis=-1, dtype=jnp.int16)
    harvest_lanes = jnp.sum(
        active_plant & (active_target_kind == TileKind.PLANT), axis=-1, dtype=jnp.int16
    )
    deposit_lanes = jnp.sum(
        active_task & (tasks.task_type == TaskTypeV1.SHED_DEPOSIT),
        axis=-1,
        dtype=jnp.int16,
    )
    assigned_primary = jnp.zeros((batch_size,), dtype=jnp.int16)
    assigned_support = jnp.zeros((batch_size,), dtype=jnp.int16)
    day_end = (((states.step // TURNS_PER_DAY) + 1) * TURNS_PER_DAY - 1).astype(jnp.int16)
    turns_remaining = (TURNS_PER_DAY - (states.step % TURNS_PER_DAY)).astype(jnp.int16)
    closing = states.step >= config.liquidation_start_step
    investing = states.step < config.investment_stop_step

    for unit in range(MAX_UNITS):
        position = states.unit_pos[:, player, unit].astype(jnp.int16)
        free = states.unit_active[:, player, unit] & (
            tasks.status[:, unit] != TaskStatusV1.ACTIVE
        )
        inventory_total = jnp.sum(
            states.unit_inventory[:, player, unit, :NUM_PRODUCTS].astype(jnp.int32), axis=-1
        )
        carrying = inventory_total > 0
        depot, depot_distance = _nearest_shed(position)
        deposit = free & carrying & (deposit_lanes < config.deposit_lanes.astype(jnp.int16))

        unwatered = (
            (kind == TileKind.PLANT)
            & ((flags & jnp.uint8(FLAG_WATERED)) == 0)
            & (~reserved)
        )
        water_target, water_valid, water_distance = _choose_tile(unwatered, position)
        water = (
            free
            & (~deposit)
            & (~closing)
            & water_valid
            & (water_distance + 1 <= turns_remaining)
        )

        safe_crop = jnp.clip(crop, 0, NUM_CROPS - 1)
        first_yield = _FIRST_YIELD[safe_crop]
        mature = (
            (kind == TileKind.PLANT)
            & (tile_yield > 0)
            & ((current_day[:, None] - origin) >= first_yield)
            & (~reserved)
        )
        harvest_target, harvest_valid, harvest_distance = _choose_tile(mature, position)
        harvest = (
            free
            & (~deposit)
            & (~water)
            & harvest_valid
            & (harvest_lanes < config.harvest_dispatch_lanes.astype(jnp.int16))
        )

        empty = (kind == TileKind.EMPTY) & (~reserved)
        plant_target, plant_valid, plant_distance = _choose_tile(
            empty, position, logistics_weight=2
        )
        primary_available = (
            (primary_deficit - assigned_primary > 0)
            & (primary_seed - assigned_primary > 0)
        )
        support_available = (
            (support_deficit - assigned_support > 0)
            & (support_seed - assigned_support > 0)
        )
        choose_primary = primary_available
        plant_item = jnp.where(choose_primary, primary_id, support_id).astype(jnp.int8)
        plant = (
            free
            & (~deposit)
            & (~water)
            & (~harvest)
            & (~closing)
            & investing
            & plant_valid
            & (plant_distance + 2 <= turns_remaining)
            & (plant_lanes < config.parallel_plant_lanes.astype(jnp.int16))
            & (primary_available | support_available)
        )

        any_map = water | harvest | plant
        selected_target = jnp.where(
            water,
            water_target,
            jnp.where(harvest, harvest_target, plant_target),
        ).astype(jnp.int32)
        selected_x = _TILE_X[jnp.clip(selected_target, 0, BOARD_SIZE * BOARD_SIZE - 1)]
        selected_y = _TILE_Y[jnp.clip(selected_target, 0, BOARD_SIZE * BOARD_SIZE - 1)]
        selected_crop = crop[batch, jnp.clip(selected_target, 0, BOARD_SIZE * BOARD_SIZE - 1)]
        selected_item = jnp.where(
            plant,
            plant_item,
            jnp.where(any_map, selected_crop, -1),
        ).astype(jnp.int8)
        task_type = jnp.where(
            deposit,
            TaskTypeV1.SHED_DEPOSIT,
            jnp.where(water, TaskTypeV1.WATER_CROP, TaskTypeV1.CROP_PRODUCTION),
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
            jnp.where(water, water_distance, jnp.where(harvest, harvest_distance, plant_distance)),
        ).astype(jnp.int16)
        deadline = jnp.where(
            water | plant, day_end, jnp.int16(EPISODE_STEPS - 2)
        )
        assign = deposit | any_map
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
        safe_target = jnp.clip(selected_target, 0, BOARD_SIZE * BOARD_SIZE - 1)
        reserved = reserved.at[batch, safe_target].set(
            reserved[batch, safe_target] | any_map
        )
        deposit_lanes = deposit_lanes + deposit.astype(jnp.int16)
        harvest_lanes = harvest_lanes + harvest.astype(jnp.int16)
        plant_lanes = plant_lanes + plant.astype(jnp.int16)
        assigned_primary = assigned_primary + (plant & choose_primary).astype(jnp.int16)
        assigned_support = assigned_support + (plant & (~choose_primary)).astype(jnp.int16)

    return controller._replace(unit_tasks=tasks)


def _hire_prefix_cost(current_hires: jax.Array) -> jax.Array:
    offsets = jnp.arange(10, dtype=jnp.int32)[None, :]
    indices = jnp.clip(current_hires[:, None].astype(jnp.int32) + offsets, 0, len(HIRE_COST) - 1)
    return jnp.cumsum(_HIRE_COST[indices], axis=-1, dtype=jnp.int32)


def materialize_m25_market_tasks_v2(
    states: State,
    controller: ProjectControllerStateV2,
    config: M25CropExpansionConfigV2,
    player: int,
) -> ProjectControllerStateV2:
    """Pack cash-safe sell/land/seed/hire intents into the ten market slots."""

    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    phase_index = m25_phase_v2(states, config)
    primary_id = config.primary_crop_id.astype(jnp.int32)
    support_id = config.support_crop_id.astype(jnp.int32)
    primary_target = _phase_value(config.primary_target, phase_index).astype(jnp.int16)
    support_target = _phase_value(config.support_target, phase_index).astype(jnp.int16)
    hand_target = _phase_value(config.hand_target, phase_index).astype(jnp.int16)
    closing = states.step >= config.liquidation_start_step
    investing = states.step < config.investment_stop_step
    due = closing | (
        jnp.mod(states.step - config.sell_phase, jnp.maximum(config.sell_interval, 1)) == 0
    )
    primary_shed = states.shed[batch, player, primary_id].astype(jnp.int16)
    support_shed = states.shed[batch, player, support_id].astype(jnp.int16)
    sell_primary = due & (primary_shed > 0)
    sell_support = due & (support_shed > 0) & (support_id != primary_id)

    desired_land = _desired_land_count(states, config)
    unlocked = states.unlocked_count[:, player].astype(jnp.int32)
    land_needed = unlocked < desired_land.astype(jnp.int32)
    land_cost = _LAND_COST[jnp.clip(unlocked - 1, 0, len(LAND_PRICES) - 1)]
    available = jnp.maximum(states.money[:, player] - config.cash_floor, 0).astype(jnp.int32)
    buy_land = (
        investing
        & _land_purchase_window_open(states, config)
        & land_needed
        & (available >= land_cost)
    )
    available = available - jnp.where(buy_land, land_cost, 0)

    kind = states.tile_kind[:, player].reshape(batch_size, -1)
    crop = states.tile_crop[:, player].reshape(batch_size, -1)
    primary_active = jnp.sum(
        (kind == TileKind.PLANT) & (crop == primary_id[:, None]), axis=-1, dtype=jnp.int16
    )
    support_active = jnp.sum(
        (kind == TileKind.PLANT) & (crop == support_id[:, None]), axis=-1, dtype=jnp.int16
    )
    seed_primary = states.seeds[batch, player, primary_id].astype(jnp.int16)
    seed_support = states.seeds[batch, player, support_id].astype(jnp.int16)
    primary_deficit = jnp.maximum(primary_target - primary_active - seed_primary, 0)
    support_deficit = jnp.maximum(support_target - support_active - seed_support, 0)
    primary_cost = _SEED_COST[primary_id]
    support_cost = _SEED_COST[support_id]
    primary_quantity = jnp.minimum(primary_deficit, config.primary_seed_batch)
    primary_quantity = jnp.minimum(
        primary_quantity.astype(jnp.int32), available // jnp.maximum(primary_cost, 1)
    ).astype(jnp.int16)
    buy_primary = investing & (primary_quantity > 0)
    available = available - jnp.where(
        buy_primary, primary_quantity.astype(jnp.int32) * primary_cost, 0
    )
    support_quantity = jnp.minimum(support_deficit, config.support_seed_batch)
    support_quantity = jnp.minimum(
        support_quantity.astype(jnp.int32), available // jnp.maximum(support_cost, 1)
    ).astype(jnp.int16)
    buy_support = investing & (support_quantity > 0) & (support_id != primary_id)
    available = available - jnp.where(
        buy_support, support_quantity.astype(jnp.int32) * support_cost, 0
    )

    current_hires = states.hires_today[:, player].astype(jnp.int16)
    desired_hires = jnp.clip(
        hand_target - current_hires,
        0,
        config.hire_batch_max.astype(jnp.int16),
    )
    hire_prefix = _hire_prefix_cost(current_hires)
    hire_index = jnp.arange(10, dtype=jnp.int16)[None, :]
    hire_present = (
        investing[:, None]
        & (hire_index < desired_hires[:, None])
        & (hire_prefix <= available[:, None])
    )

    candidate_count = 15
    present = jnp.zeros((batch_size, candidate_count), dtype=jnp.bool_)
    task_type = jnp.zeros((batch_size, candidate_count), dtype=jnp.int8)
    item = jnp.full((batch_size, candidate_count), -1, dtype=jnp.int8)
    quantity = jnp.zeros((batch_size, candidate_count), dtype=jnp.int16)

    def write(slot, mask, kind_value, item_value, quantity_value):
        return (
            present.at[:, slot].set(mask),
            task_type.at[:, slot].set(jnp.asarray(kind_value, dtype=jnp.int8)),
            item.at[:, slot].set(jnp.asarray(item_value, dtype=jnp.int8)),
            quantity.at[:, slot].set(jnp.asarray(quantity_value, dtype=jnp.int16)),
        )

    present, task_type, item, quantity = write(
        0,
        sell_primary,
        jnp.where(closing, TaskTypeV1.TERMINAL_LIQUIDATION, TaskTypeV1.SELL_INVENTORY),
        config.primary_crop_id,
        primary_shed,
    )
    present, task_type, item, quantity = write(
        1,
        sell_support,
        jnp.where(closing, TaskTypeV1.TERMINAL_LIQUIDATION, TaskTypeV1.SELL_INVENTORY),
        config.support_crop_id,
        support_shed,
    )
    present, task_type, item, quantity = write(
        2, buy_land, TaskTypeV1.BUY_LAND, -1, 1
    )
    present, task_type, item, quantity = write(
        3, buy_primary, TaskTypeV1.CROP_PRODUCTION, config.primary_crop_id, primary_quantity
    )
    present, task_type, item, quantity = write(
        4, buy_support, TaskTypeV1.CROP_PRODUCTION, config.support_crop_id, support_quantity
    )
    present = present.at[:, 5:15].set(hire_present)
    task_type = task_type.at[:, 5:15].set(jnp.int8(TaskTypeV1.HIRE_WORKER))
    quantity = quantity.at[:, 5:15].set(jnp.int16(1))

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


def m25_policy_step_v2(
    states: State,
    controller: ProjectControllerStateV2,
    config: M25CropExpansionConfigV2,
    player: int,
) -> tuple[E4PlayerActionV1, ProjectControllerStateV2]:
    controller, _ = reconcile_project_controller_v2(states, controller, player)
    controller = clear_invalidated_crop_tasks_v2(states, controller, player)
    controller = ensure_m25_projects_v2(states, controller, config, player)
    controller = materialize_m25_unit_tasks_v2(states, controller, config, player)
    controller = materialize_m25_market_tasks_v2(states, controller, config, player)
    action = compile_full_core_player_action_v1(states, controller, player)
    return action, controller


def update_m25_controller_from_effects_v2(
    states: State,
    next_states: State,
    controller: ProjectControllerStateV2,
    action: E4PlayerActionV1,
    config: M25CropExpansionConfigV2,
    player: int,
) -> tuple[ProjectControllerStateV2, FullCoreEffectDiagnosticsV1]:
    controller, diagnostics = update_full_core_controller_from_effects_v1(
        states, next_states, controller, action, player
    )
    controller = ensure_m25_projects_v2(next_states, controller, config, player)
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


def m25_player_action_dict_v2(action: E4PlayerActionV1) -> dict:
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
    "default_r2_tomato_m25_config_v2",
    "ensure_m25_projects_v2",
    "m25_phase_v2",
    "m25_player_action_dict_v2",
    "m25_policy_step_v2",
    "materialize_m25_market_tasks_v2",
    "materialize_m25_unit_tasks_v2",
    "update_m25_controller_from_effects_v2",
]
