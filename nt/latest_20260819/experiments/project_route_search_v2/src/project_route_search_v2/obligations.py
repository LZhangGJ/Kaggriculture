"""Fixed-shape M2 crop obligations and direct single-farmer materialization."""

from __future__ import annotations

import jax.numpy as jnp

from kaggriculture_jax.constants import (
    BOARD_SIZE,
    CROP_SEED_COST,
    EPISODE_STEPS,
    FLAG_WATERED,
    MAX_MARKET_ORDERS,
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

from .constants import (
    CropObligationTypeV2,
    MAX_OBLIGATIONS_V2,
    ObligationPriorityV2,
)
from .schema import CropProjectConfigV2, ObligationV2, ProjectControllerStateV2


_TILE_X = jnp.tile(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
_TILE_Y = jnp.repeat(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
_TILE_ID = jnp.arange(BOARD_SIZE * BOARD_SIZE, dtype=jnp.int32)
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)
_SEED_COST = jnp.asarray(CROP_SEED_COST, dtype=jnp.int32)
_INVALID_KEY = jnp.int32(1_000_000)

UNIT_OBLIGATION_SLOT = 0
SELL_OBLIGATION_SLOT = 1
SEED_OBLIGATION_SLOT = 2


def empty_obligations_v2(batch_size: int) -> ObligationV2:
    shape = (batch_size, MAX_OBLIGATIONS_V2)
    i8 = lambda value=0: jnp.full(shape, value, dtype=jnp.int8)
    i16 = lambda value=0: jnp.full(shape, value, dtype=jnp.int16)
    i32 = lambda value=0: jnp.full(shape, value, dtype=jnp.int32)
    return ObligationV2(
        obligation_type=i8(),
        project_id=i16(-1),
        target_id=i16(-1),
        target_x=i8(-1),
        target_y=i8(-1),
        item_id=i8(-1),
        quantity=i16(),
        priority_level=i8(ObligationPriorityV2.OPTIONAL),
        earliest_step=i16(-1),
        deadline_step=i16(-1),
        expected_finish_step=i16(-1),
        slack=i16(),
        required_unit_item=i8(-1),
        required_unit_quantity=i16(),
        required_shed_item=i8(-1),
        required_shed_quantity=i16(),
        cash_required=i32(),
        market_slots_required=i8(),
        expected_bank_delta=i32(),
        value_at_risk=i32(),
        future_maintenance_actions=i16(),
        route_insertion_cost=i16(),
        present=jnp.zeros(shape, dtype=jnp.bool_),
        hard_mask=jnp.zeros(shape, dtype=jnp.bool_),
        failure_code=i8(),
    )


def _nearest(mask: jnp.ndarray, position: jnp.ndarray):
    distance = jnp.abs(_TILE_X[None, :] - position[:, 0, None]) + jnp.abs(
        _TILE_Y[None, :] - position[:, 1, None]
    )
    key = jnp.where(mask, distance.astype(jnp.int32) * 100 + _TILE_ID, _INVALID_KEY)
    target = jnp.argmin(key, axis=-1)
    valid = jnp.min(key, axis=-1) < _INVALID_KEY
    return target, valid, jnp.take_along_axis(distance, target[:, None], axis=-1)[:, 0]


def _write_obligation(
    obligations: ObligationV2,
    slot: int,
    *,
    obligation_type,
    present,
    target_id,
    target_x,
    target_y,
    item_id,
    quantity,
    priority,
    step,
    deadline,
    expected_finish,
    cash_required,
    market_slots,
    expected_bank_delta,
    hard,
):
    return obligations._replace(
        obligation_type=obligations.obligation_type.at[:, slot].set(
            jnp.where(present, obligation_type, 0).astype(jnp.int8)
        ),
        project_id=obligations.project_id.at[:, slot].set(
            jnp.where(present, 0, -1).astype(jnp.int16)
        ),
        target_id=obligations.target_id.at[:, slot].set(
            jnp.where(present, target_id, -1).astype(jnp.int16)
        ),
        target_x=obligations.target_x.at[:, slot].set(
            jnp.where(present, target_x, -1).astype(jnp.int8)
        ),
        target_y=obligations.target_y.at[:, slot].set(
            jnp.where(present, target_y, -1).astype(jnp.int8)
        ),
        item_id=obligations.item_id.at[:, slot].set(
            jnp.where(present, item_id, -1).astype(jnp.int8)
        ),
        quantity=obligations.quantity.at[:, slot].set(
            jnp.where(present, quantity, 0).astype(jnp.int16)
        ),
        priority_level=obligations.priority_level.at[:, slot].set(
            jnp.where(present, priority, ObligationPriorityV2.OPTIONAL).astype(jnp.int8)
        ),
        earliest_step=obligations.earliest_step.at[:, slot].set(
            jnp.where(present, step, -1).astype(jnp.int16)
        ),
        deadline_step=obligations.deadline_step.at[:, slot].set(
            jnp.where(present, deadline, -1).astype(jnp.int16)
        ),
        expected_finish_step=obligations.expected_finish_step.at[:, slot].set(
            jnp.where(present, expected_finish, -1).astype(jnp.int16)
        ),
        slack=obligations.slack.at[:, slot].set(
            jnp.where(present, deadline - expected_finish, 0).astype(jnp.int16)
        ),
        cash_required=obligations.cash_required.at[:, slot].set(
            jnp.where(present, cash_required, 0).astype(jnp.int32)
        ),
        market_slots_required=obligations.market_slots_required.at[:, slot].set(
            jnp.where(present, market_slots, 0).astype(jnp.int8)
        ),
        expected_bank_delta=obligations.expected_bank_delta.at[:, slot].set(
            jnp.where(present, expected_bank_delta, 0).astype(jnp.int32)
        ),
        present=obligations.present.at[:, slot].set(present),
        hard_mask=obligations.hard_mask.at[:, slot].set(present & hard),
    )


def build_crop_obligations_v2(
    states: State,
    controller: ProjectControllerStateV2,
    config: CropProjectConfigV2,
    player: int,
) -> ObligationV2:
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    obligations = empty_obligations_v2(batch_size)
    crop_id = config.crop_id.astype(jnp.int32)
    kind = states.tile_kind[:, player].reshape(batch_size, -1)
    crop = states.tile_crop[:, player].reshape(batch_size, -1)
    flags = states.tile_flags[:, player].reshape(batch_size, -1)
    origin = states.tile_origin_day[:, player].reshape(batch_size, -1).astype(jnp.int16)
    yields = states.tile_yield[:, player].reshape(batch_size, -1)
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)
    target_crop = (kind == TileKind.PLANT) & (crop == crop_id[:, None])
    unwatered = target_crop & ((flags & jnp.uint8(FLAG_WATERED)) == 0)
    mature = target_crop & (
        current_day[:, None] - origin
        >= config.harvest_age_days.astype(jnp.int16)[:, None]
    ) & (yields > 0)
    empty = kind == TileKind.EMPTY
    farmer_pos = states.unit_pos[:, player, 0].astype(jnp.int16)
    water_target, water_valid, water_distance = _nearest(unwatered, farmer_pos)
    harvest_target, harvest_valid, harvest_distance = _nearest(mature, farmer_pos)
    plant_target, plant_valid, plant_distance = _nearest(empty, farmer_pos)

    inventory = states.unit_inventory[:, player, 0, :NUM_PRODUCTS]
    carrying_units = jnp.sum(inventory.astype(jnp.int32), axis=-1)
    carrying = carrying_units > 0
    depot_distance = jnp.sum(
        jnp.abs(farmer_pos[:, None, :] - _SHED_ACCESS[None, :, :]), axis=-1
    )
    depot_index = jnp.argmin(depot_distance, axis=-1)
    depot = _SHED_ACCESS[depot_index]
    at_depot = jnp.min(depot_distance, axis=-1) == 0
    unit_busy = controller.unit_tasks.status[:, 0] == TaskStatusV1.ACTIVE
    closing = states.step >= config.liquidation_start_step
    day_end = ((states.step // TURNS_PER_DAY) + 1) * TURNS_PER_DAY - 1
    turns_remaining_today = TURNS_PER_DAY - (states.step % TURNS_PER_DAY)
    can_plant_and_water = plant_distance + 2 <= turns_remaining_today
    active_count = jnp.sum(target_crop, axis=-1, dtype=jnp.int16)
    seed_count = states.seeds[batch, player, crop_id].astype(jnp.int16)

    deposit_present = (~unit_busy) & carrying
    water_present = (~unit_busy) & (~deposit_present) & (~closing) & water_valid
    harvest_present = (
        (~unit_busy)
        & (~deposit_present)
        & (~water_present)
        & harvest_valid
    )
    need_more = active_count < config.target_tiles
    plant_present = (
        (~unit_busy)
        & (~deposit_present)
        & (~water_present)
        & (~harvest_present)
        & (~closing)
        & (states.step < config.investment_stop_step)
        & need_more
        & (seed_count > 0)
        & plant_valid
        & can_plant_and_water
    )
    unit_type = jnp.where(
        deposit_present,
        CropObligationTypeV2.DEPOSIT_INVENTORY,
        jnp.where(
            water_present,
            CropObligationTypeV2.WATER_CROP,
            jnp.where(
                harvest_present,
                CropObligationTypeV2.HARVEST_AND_DEPOSIT,
                CropObligationTypeV2.PLANT_AND_WATER,
            ),
        ),
    ).astype(jnp.int8)
    unit_present = deposit_present | water_present | harvest_present | plant_present
    unit_target = jnp.where(
        deposit_present,
        depot[:, 1].astype(jnp.int32) * BOARD_SIZE + depot[:, 0].astype(jnp.int32),
        jnp.where(
            water_present,
            water_target,
            jnp.where(harvest_present, harvest_target, plant_target),
        ),
    )
    unit_x = _TILE_X[jnp.clip(unit_target, 0, BOARD_SIZE * BOARD_SIZE - 1)]
    unit_y = _TILE_Y[jnp.clip(unit_target, 0, BOARD_SIZE * BOARD_SIZE - 1)]
    unit_distance = jnp.where(
        deposit_present,
        jnp.min(depot_distance, axis=-1),
        jnp.where(
            water_present,
            water_distance,
            jnp.where(harvest_present, harvest_distance, plant_distance),
        ),
    ).astype(jnp.int16)
    unit_deadline = jnp.where(
        water_present | plant_present, day_end, EPISODE_STEPS - 2
    ).astype(jnp.int16)
    unit_priority = jnp.where(
        deposit_present | water_present,
        ObligationPriorityV2.HARD,
        jnp.where(harvest_present, ObligationPriorityV2.URGENT, ObligationPriorityV2.OPTIONAL),
    ).astype(jnp.int8)
    obligations = _write_obligation(
        obligations,
        UNIT_OBLIGATION_SLOT,
        obligation_type=unit_type,
        present=unit_present,
        target_id=unit_target,
        target_x=unit_x,
        target_y=unit_y,
        item_id=config.crop_id,
        quantity=jnp.ones((batch_size,), dtype=jnp.int16),
        priority=unit_priority,
        step=states.step,
        deadline=unit_deadline,
        expected_finish=(states.step + unit_distance + 1).astype(jnp.int16),
        cash_required=jnp.zeros((batch_size,), dtype=jnp.int32),
        market_slots=jnp.zeros((batch_size,), dtype=jnp.int8),
        expected_bank_delta=jnp.zeros((batch_size,), dtype=jnp.int32),
        hard=deposit_present | water_present,
    )

    shed_units = states.shed[batch, player, crop_id].astype(jnp.int16)
    carried_crop_units = inventory[batch, crop_id].astype(jnp.int16)
    sell_quantity = shed_units + jnp.where(at_depot, carried_crop_units, 0).astype(jnp.int16)
    sell_present = sell_quantity > 0
    sell_type = jnp.where(
        closing,
        CropObligationTypeV2.LIQUIDATE_SELL,
        CropObligationTypeV2.SELL_SHED,
    ).astype(jnp.int8)
    sell_value = sell_quantity.astype(jnp.int32) * states.market_price[batch, crop_id]
    obligations = _write_obligation(
        obligations,
        SELL_OBLIGATION_SLOT,
        obligation_type=sell_type,
        present=sell_present,
        target_id=jnp.full((batch_size,), -1, dtype=jnp.int16),
        target_x=jnp.full((batch_size,), -1, dtype=jnp.int8),
        target_y=jnp.full((batch_size,), -1, dtype=jnp.int8),
        item_id=config.crop_id,
        quantity=sell_quantity,
        priority=jnp.where(
            closing, ObligationPriorityV2.HARD, ObligationPriorityV2.URGENT
        ),
        step=states.step,
        deadline=jnp.full((batch_size,), EPISODE_STEPS - 2, dtype=jnp.int16),
        expected_finish=states.step,
        cash_required=jnp.zeros((batch_size,), dtype=jnp.int32),
        market_slots=jnp.ones((batch_size,), dtype=jnp.int8),
        expected_bank_delta=sell_value,
        hard=closing,
    )

    deficit = jnp.maximum(config.target_tiles - active_count - seed_count, 0)
    buy_quantity = jnp.minimum(deficit, config.seed_batch_size).astype(jnp.int16)
    buy_cost = buy_quantity.astype(jnp.int32) * _SEED_COST[crop_id]
    buy_present = (
        (buy_quantity > 0)
        & (~closing)
        & (states.step < config.investment_stop_step)
        & ((states.money[:, player] - buy_cost) >= config.cash_floor)
    )
    obligations = _write_obligation(
        obligations,
        SEED_OBLIGATION_SLOT,
        obligation_type=jnp.full(
            (batch_size,), CropObligationTypeV2.BUY_SEED, dtype=jnp.int8
        ),
        present=buy_present,
        target_id=jnp.full((batch_size,), -1, dtype=jnp.int16),
        target_x=jnp.full((batch_size,), -1, dtype=jnp.int8),
        target_y=jnp.full((batch_size,), -1, dtype=jnp.int8),
        item_id=config.crop_id,
        quantity=buy_quantity,
        priority=jnp.full(
            (batch_size,), ObligationPriorityV2.OPTIONAL, dtype=jnp.int8
        ),
        step=states.step,
        deadline=jnp.minimum(states.step + 1, EPISODE_STEPS - 2).astype(jnp.int16),
        expected_finish=states.step,
        cash_required=buy_cost,
        market_slots=jnp.ones((batch_size,), dtype=jnp.int8),
        expected_bank_delta=-buy_cost,
        hard=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )
    return obligations


def materialize_crop_obligations_v2(
    states: State,
    controller: ProjectControllerStateV2,
    obligations: ObligationV2,
) -> ProjectControllerStateV2:
    """Attach the M2 unit obligation and ordered SELL-before-BUY market tasks."""

    unit = obligations.present[:, UNIT_OBLIGATION_SLOT] & (
        controller.unit_tasks.status[:, 0] != TaskStatusV1.ACTIVE
    )
    unit_type = obligations.obligation_type[:, UNIT_OBLIGATION_SLOT]
    task_type = jnp.where(
        unit_type == CropObligationTypeV2.WATER_CROP,
        TaskTypeV1.WATER_CROP,
        jnp.where(
            unit_type == CropObligationTypeV2.DEPOSIT_INVENTORY,
            TaskTypeV1.SHED_DEPOSIT,
            TaskTypeV1.CROP_PRODUCTION,
        ),
    ).astype(jnp.int8)
    phase = jnp.where(
        unit_type == CropObligationTypeV2.DEPOSIT_INVENTORY,
        TaskPhaseV1.MOVE_TO_DEPOT,
        TaskPhaseV1.MOVE_TO_TARGET,
    ).astype(jnp.int8)
    tasks = controller.unit_tasks
    tasks = tasks._replace(
        task_type=tasks.task_type.at[:, 0].set(jnp.where(unit, task_type, tasks.task_type[:, 0])),
        target_id=tasks.target_id.at[:, 0].set(
            jnp.where(unit, obligations.target_id[:, UNIT_OBLIGATION_SLOT], tasks.target_id[:, 0])
        ),
        target_x=tasks.target_x.at[:, 0].set(
            jnp.where(unit, obligations.target_x[:, UNIT_OBLIGATION_SLOT], tasks.target_x[:, 0])
        ),
        target_y=tasks.target_y.at[:, 0].set(
            jnp.where(unit, obligations.target_y[:, UNIT_OBLIGATION_SLOT], tasks.target_y[:, 0])
        ),
        item_id=tasks.item_id.at[:, 0].set(
            jnp.where(unit, obligations.item_id[:, UNIT_OBLIGATION_SLOT], tasks.item_id[:, 0])
        ),
        quantity=tasks.quantity.at[:, 0].set(
            jnp.where(unit, obligations.quantity[:, UNIT_OBLIGATION_SLOT], tasks.quantity[:, 0])
        ),
        phase=tasks.phase.at[:, 0].set(jnp.where(unit, phase, tasks.phase[:, 0])),
        start_step=tasks.start_step.at[:, 0].set(jnp.where(unit, states.step, tasks.start_step[:, 0])),
        last_progress_step=tasks.last_progress_step.at[:, 0].set(
            jnp.where(unit, states.step, tasks.last_progress_step[:, 0])
        ),
        expected_finish_step=tasks.expected_finish_step.at[:, 0].set(
            jnp.where(
                unit,
                obligations.expected_finish_step[:, UNIT_OBLIGATION_SLOT],
                tasks.expected_finish_step[:, 0],
            )
        ),
        deadline_step=tasks.deadline_step.at[:, 0].set(
            jnp.where(unit, obligations.deadline_step[:, UNIT_OBLIGATION_SLOT], tasks.deadline_step[:, 0])
        ),
        status=tasks.status.at[:, 0].set(
            jnp.where(unit, TaskStatusV1.ACTIVE, tasks.status[:, 0]).astype(jnp.int8)
        ),
        failure_code=tasks.failure_code.at[:, 0].set(
            jnp.where(unit, FailureCodeV1.NONE, tasks.failure_code[:, 0]).astype(jnp.int8)
        ),
    )

    market = controller.market_tasks
    sell = obligations.present[:, SELL_OBLIGATION_SLOT] & (
        market.status[:, 0] != TaskStatusV1.ACTIVE
    )
    sell_terminal = (
        obligations.obligation_type[:, SELL_OBLIGATION_SLOT]
        == CropObligationTypeV2.LIQUIDATE_SELL
    )
    seed = obligations.present[:, SEED_OBLIGATION_SLOT] & (
        market.status[:, 1] != TaskStatusV1.ACTIVE
    )
    market = market._replace(
        task_type=market.task_type.at[:, 0].set(
            jnp.where(
                sell,
                jnp.where(
                    sell_terminal,
                    TaskTypeV1.TERMINAL_LIQUIDATION,
                    TaskTypeV1.SELL_INVENTORY,
                ),
                market.task_type[:, 0],
            ).astype(jnp.int8)
        ),
        item_id=market.item_id.at[:, 0].set(
            jnp.where(sell, obligations.item_id[:, SELL_OBLIGATION_SLOT], market.item_id[:, 0])
        ),
        quantity=market.quantity.at[:, 0].set(
            jnp.where(sell, obligations.quantity[:, SELL_OBLIGATION_SLOT], market.quantity[:, 0])
        ),
        start_step=market.start_step.at[:, 0].set(jnp.where(sell, states.step, market.start_step[:, 0])),
        deadline_step=market.deadline_step.at[:, 0].set(
            jnp.where(sell, obligations.deadline_step[:, SELL_OBLIGATION_SLOT], market.deadline_step[:, 0])
        ),
        status=market.status.at[:, 0].set(
            jnp.where(sell, TaskStatusV1.ACTIVE, market.status[:, 0]).astype(jnp.int8)
        ),
        failure_code=market.failure_code.at[:, 0].set(
            jnp.where(sell, FailureCodeV1.NONE, market.failure_code[:, 0]).astype(jnp.int8)
        ),
    )
    market = market._replace(
        task_type=market.task_type.at[:, 1].set(
            jnp.where(seed, TaskTypeV1.CROP_PRODUCTION, market.task_type[:, 1]).astype(jnp.int8)
        ),
        item_id=market.item_id.at[:, 1].set(
            jnp.where(seed, obligations.item_id[:, SEED_OBLIGATION_SLOT], market.item_id[:, 1])
        ),
        quantity=market.quantity.at[:, 1].set(
            jnp.where(seed, obligations.quantity[:, SEED_OBLIGATION_SLOT], market.quantity[:, 1])
        ),
        start_step=market.start_step.at[:, 1].set(jnp.where(seed, states.step, market.start_step[:, 1])),
        deadline_step=market.deadline_step.at[:, 1].set(
            jnp.where(seed, obligations.deadline_step[:, SEED_OBLIGATION_SLOT], market.deadline_step[:, 1])
        ),
        status=market.status.at[:, 1].set(
            jnp.where(seed, TaskStatusV1.ACTIVE, market.status[:, 1]).astype(jnp.int8)
        ),
        failure_code=market.failure_code.at[:, 1].set(
            jnp.where(seed, FailureCodeV1.NONE, market.failure_code[:, 1]).astype(jnp.int8)
        ),
    )
    return controller._replace(unit_tasks=tasks, market_tasks=market)


__all__ = [
    "build_crop_obligations_v2",
    "empty_obligations_v2",
    "materialize_crop_obligations_v2",
]
