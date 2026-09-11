"""Episode-safe lifecycle helpers for the V5 controller."""

from __future__ import annotations

import jax.numpy as jnp

from kaggriculture_jax.constants import (
    BOARD_SIZE,
    FLAG_WATERED,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    TileKind,
)
from kaggriculture_jax.types import State

from .constants import SCHEMA_VERSION_V1, TaskStatusV1, TaskTypeV1
from .schema import ControllerStateV1, MarketTaskStateV1, UnitTaskStateV1


def empty_unit_tasks_v1() -> UnitTaskStateV1:
    i8 = lambda value=0: jnp.full((MAX_UNITS,), value, dtype=jnp.int8)
    i16 = lambda value=0: jnp.full((MAX_UNITS,), value, dtype=jnp.int16)
    return UnitTaskStateV1(
        task_type=i8(),
        owner_unit=jnp.arange(MAX_UNITS, dtype=jnp.int8),
        target_id=i16(-1),
        target_x=i8(-1),
        target_y=i8(-1),
        item_id=i8(-1),
        quantity=i16(),
        phase=i8(),
        start_step=i16(-1),
        last_progress_step=i16(-1),
        expected_finish_step=i16(-1),
        deadline_step=i16(-1),
        status=i8(TaskStatusV1.EMPTY),
        failure_code=i8(),
    )


def empty_market_tasks_v1() -> MarketTaskStateV1:
    i8 = lambda value=0: jnp.full((MAX_MARKET_ORDERS,), value, dtype=jnp.int8)
    i16 = lambda value=0: jnp.full((MAX_MARKET_ORDERS,), value, dtype=jnp.int16)
    return MarketTaskStateV1(
        task_type=i8(),
        item_id=i8(-1),
        quantity=i16(),
        start_step=i16(-1),
        deadline_step=i16(-1),
        status=i8(TaskStatusV1.EMPTY),
        failure_code=i8(),
    )


def reset_controller_state_v1() -> ControllerStateV1:
    return ControllerStateV1(
        schema_version=jnp.asarray(SCHEMA_VERSION_V1, dtype=jnp.int16),
        unit_tasks=empty_unit_tasks_v1(),
        market_tasks=empty_market_tasks_v1(),
    )


def clear_finished_tasks_v1(controller: ControllerStateV1) -> ControllerStateV1:
    """Clear terminal task records before generating a new decision."""

    unit_finished = controller.unit_tasks.status >= TaskStatusV1.DONE
    market_finished = controller.market_tasks.status >= TaskStatusV1.DONE
    empty_units = empty_unit_tasks_v1()
    empty_market = empty_market_tasks_v1()
    units = type(controller.unit_tasks)(
        *(jnp.where(unit_finished, replacement, value) for value, replacement in zip(
            controller.unit_tasks, empty_units, strict=True
        ))
    )
    market = type(controller.market_tasks)(
        *(jnp.where(market_finished, replacement, value) for value, replacement in zip(
            controller.market_tasks, empty_market, strict=True
        ))
    )
    return controller._replace(unit_tasks=units, market_tasks=market)


def clear_invalidated_full_core_tasks_v1(
    states: State, controller: ControllerStateV1, player: int
) -> ControllerStateV1:
    """Clear completed records and active water routes whose crop disappeared.

    Weed spawning and crop neglect can invalidate a persistent water route while
    its owner is still travelling.  Keeping that route active would eventually
    compile WATER against a non-plant tile and create a false expected effect.
    """

    controller = clear_finished_tasks_v1(controller)
    tasks = controller.unit_tasks
    active_water = (
        (tasks.status == TaskStatusV1.ACTIVE)
        & (tasks.task_type == TaskTypeV1.WATER_CROP)
    )
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    x = jnp.clip(tasks.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(tasks.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    kind = states.tile_kind[batch, player, y, x]
    crop = states.tile_crop[batch, player, y, x]
    flags = states.tile_flags[batch, player, y, x]
    water_still_needed = (
        (kind == TileKind.PLANT)
        & (crop == tasks.item_id)
        & ((flags & jnp.uint8(FLAG_WATERED)) == 0)
    )
    invalidated = active_water & (~water_still_needed)
    empty = empty_unit_tasks_v1()
    unit_tasks = type(tasks)(
        *(
            jnp.where(invalidated, replacement, value)
            for value, replacement in zip(tasks, empty, strict=True)
        )
    )
    return controller._replace(unit_tasks=unit_tasks)
