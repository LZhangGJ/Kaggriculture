"""M2 crop policy adapter over the previously parity-tested V5 primitive compiler."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    BOARD_SIZE,
    FLAG_WATERED,
    NUM_PRODUCTS,
    MarketOp,
    TileKind,
)
from kaggriculture_jax.types import State
from strategic_v5.constants import TaskPhaseV1, TaskStatusV1, TaskTypeV1
from strategic_v5.e4_executor import (
    CropEffectDiagnosticsV1,
    CropPlayerActionV1,
    compile_crop_inventory_player_action_v1,
    update_crop_inventory_controller_from_effects_v1,
)

from .crop_project import ensure_crop_project_v2, refresh_crop_project_v2
from .lifecycle import empty_unit_tasks_v2, reconcile_project_controller_v2
from .obligations import (
    build_crop_obligations_v2,
    materialize_crop_obligations_v2,
)


def clear_invalidated_crop_tasks_v2(
    states: State,
    controller: ProjectControllerStateV2,
    player: int,
) -> ProjectControllerStateV2:
    """Drop persistent M2 tasks whose target/resource no longer exists.

    This matters for ongoing crops: a tomato or strawberry plot can expire into
    a weed while the farmer is still travelling to a previously generated
    WATER/HARVEST target.  Keeping that stale task would repeatedly compile a
    legal enum whose environment effect is a silent no-op.
    """

    tasks = controller.unit_tasks
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    unit = jnp.arange(tasks.status.shape[1], dtype=jnp.int32)[None, :]
    x = jnp.clip(tasks.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(tasks.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    kind = states.tile_kind[batch, player, y, x]
    crop = states.tile_crop[batch, player, y, x]
    flags = states.tile_flags[batch, player, y, x]
    lifespan = states.tile_max_lifespan[batch, player, y, x].astype(jnp.int16)
    item = tasks.item_id.astype(jnp.int8)
    carrying = (
        jnp.sum(
            states.unit_inventory[:, player, :, :NUM_PRODUCTS].astype(jnp.int32),
            axis=-1,
        )
        > 0
    )
    active = tasks.status == TaskStatusV1.ACTIVE
    crop_task = tasks.task_type == TaskTypeV1.CROP_PRODUCTION
    water_task = tasks.task_type == TaskTypeV1.WATER_CROP
    fertilizer_task = tasks.task_type == TaskTypeV1.APPLY_FERTILIZER
    deposit_task = tasks.task_type == TaskTypeV1.SHED_DEPOSIT
    returning = crop_task & (tasks.phase == TaskPhaseV1.MOVE_TO_DEPOT)
    target_is_empty = kind == TileKind.EMPTY
    target_is_owned_crop = (kind == TileKind.PLANT) & (crop == item)
    target_can_be_watered = (
        target_is_owned_crop
        & ((flags & jnp.uint8(FLAG_WATERED)) == 0)
        & ((lifespan < 0) | (states.step[:, None] < lifespan))
    )
    crop_valid = jnp.where(
        returning,
        carrying,
        target_is_empty | target_is_owned_crop,
    )
    supported = crop_task | water_task | fertilizer_task | deposit_task
    valid = jnp.where(
        crop_task,
        crop_valid,
        jnp.where(
            water_task,
            target_can_be_watered,
            jnp.where(fertilizer_task, kind == TileKind.PLANT, carrying),
        ),
    )
    invalid = active & supported & (~valid)
    empty = empty_unit_tasks_v2(batch_size)
    cleared = jax.tree.map(
        lambda value, replacement: jnp.where(invalid, replacement, value),
        tasks,
        empty,
    )
    return controller._replace(unit_tasks=cleared)
from .schema import (
    CropProjectConfigV2,
    ObligationV2,
    ProjectControllerStateV2,
)


def crop_policy_step_v2(
    states: State,
    controller: ProjectControllerStateV2,
    config: CropProjectConfigV2,
    player: int,
) -> tuple[CropPlayerActionV1, ProjectControllerStateV2, ObligationV2]:
    """Reconcile, generate obligations, attach tasks, and compile legal actions."""

    controller, _ = reconcile_project_controller_v2(states, controller, player)
    controller = clear_invalidated_crop_tasks_v2(states, controller, player)
    controller = ensure_crop_project_v2(states, controller, config)
    controller = refresh_crop_project_v2(states, controller, config, player)
    obligations = build_crop_obligations_v2(states, controller, config, player)
    controller = materialize_crop_obligations_v2(
        states, controller, obligations
    )
    action = compile_crop_inventory_player_action_v1(
        states, controller, player
    )
    return action, controller, obligations


def update_crop_controller_from_effects_v2(
    states: State,
    next_states: State,
    controller: ProjectControllerStateV2,
    player_action: CropPlayerActionV1,
    config: CropProjectConfigV2,
    player: int,
) -> tuple[ProjectControllerStateV2, CropEffectDiagnosticsV1]:
    controller, diagnostics = update_crop_inventory_controller_from_effects_v1(
        states,
        next_states,
        controller,
        player_action,
        player,
    )
    purchased_units = jnp.sum(
        jnp.where(
            player_action.market_op == MarketOp.BUY_SEED,
            player_action.market_amount,
            0,
        ),
        axis=-1,
        dtype=jnp.int32,
    ) * (diagnostics.seed_purchase_success_count > 0).astype(jnp.int32)
    controller = refresh_crop_project_v2(
        next_states,
        controller,
        config,
        player,
        harvested_delta=diagnostics.harvest_success_count,
        purchased_seed_units=purchased_units,
    )
    controller = controller._replace(
        unexplained_effect_failures=(
            controller.unexplained_effect_failures
            + diagnostics.effect_mismatch_count
        ).astype(controller.unexplained_effect_failures.dtype)
    )
    return controller, diagnostics


def crop_player_action_dict_v2(action: CropPlayerActionV1) -> dict:
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
    "clear_invalidated_crop_tasks_v2",
    "crop_player_action_dict_v2",
    "crop_policy_step_v2",
    "update_crop_controller_from_effects_v2",
]
