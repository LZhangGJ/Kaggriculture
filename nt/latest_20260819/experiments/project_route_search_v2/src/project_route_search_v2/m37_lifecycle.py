"""Low-overhead batched M3.7 lifecycle aggregates.

The detailed per-commitment timeline intentionally stays in the host-side
single-seed analyzer.  This module carries only fixed-size daily debt, blocker
and latest-stage counters through GPU batch rollouts.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_COST,
    CROP_SEED_COST,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    SHED_CAPACITY,
    TURNS_PER_DAY,
    TileKind,
)
from kaggriculture_jax.types import Events, StaticTables
from strategic_v5.constants import TaskStatusV1, TaskTypeV1

from .m35_schema import M35FarmGenomeV2
from .m36_rollout import M36CRolloutCarryV3
from .m36_schema import RouteCalendarV3
from .m36_split_rollout import (
    initialize_m36c_split_v3,
    make_m36c_light_chunk_v3,
    m36c_daily_plan_carry_v3,
)


_ANIMAL_COST = jnp.asarray(ANIMAL_COST, dtype=jnp.int32)
_SEED_COST = jnp.asarray(CROP_SEED_COST, dtype=jnp.int32)


class M37BlockerV1(IntEnum):
    CASH = 0
    MARKET_SLOT = 1
    CAPACITY = 2
    FACILITY = 3
    UNIT = 4
    PREEMPTION = 5
    PLACEMENT = 6
    FEED = 7
    PARTIAL_FILL = 8
    TARGET_NOT_EMITTED = 9


NUM_M37_BLOCKERS_V1 = len(M37BlockerV1)


class M37LifecycleAggregatesV1(NamedTuple):
    """Fixed-size aggregate only; no step-by-step or per-commitment arrays."""

    last_observed_day: jax.Array
    crop_target_debt_days: jax.Array
    animal_purchase_debt_days: jax.Array
    animal_activation_debt_days: jax.Array
    crop_debt_age_days: jax.Array
    animal_debt_age_days: jax.Array
    max_crop_debt_age_days: jax.Array
    max_animal_debt_age_days: jax.Array
    cash_idle_while_admissible_backlog_count: jax.Array
    blocker_day_counts: jax.Array
    latest_crop_target: jax.Array
    latest_crop_active: jax.Array
    latest_crop_pending_seed: jax.Array
    latest_crop_pending_plant: jax.Array
    latest_crop_rejection: jax.Array
    latest_animal_purchase_target: jax.Array
    latest_animal_service_target: jax.Array
    latest_animal_committed: jax.Array
    latest_animal_active: jax.Array
    latest_animal_unplaced: jax.Array


class M37RolloutCarryV1(NamedTuple):
    base: M36CRolloutCarryV3
    lifecycle: M37LifecycleAggregatesV1


def empty_m37_lifecycle_aggregates_v1(
    batch_size: int,
) -> M37LifecycleAggregatesV1:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    crop = jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int32)
    animal = jnp.zeros((batch_size, NUM_ANIMALS), dtype=jnp.int32)
    return M37LifecycleAggregatesV1(
        last_observed_day=jnp.full((batch_size,), -1, dtype=jnp.int16),
        crop_target_debt_days=crop,
        animal_purchase_debt_days=animal,
        animal_activation_debt_days=animal,
        crop_debt_age_days=crop,
        animal_debt_age_days=animal,
        max_crop_debt_age_days=crop,
        max_animal_debt_age_days=animal,
        cash_idle_while_admissible_backlog_count=jnp.zeros(
            (batch_size,), dtype=jnp.int32
        ),
        blocker_day_counts=jnp.zeros(
            (batch_size, NUM_M37_BLOCKERS_V1), dtype=jnp.int32
        ),
        latest_crop_target=crop,
        latest_crop_active=crop,
        latest_crop_pending_seed=crop,
        latest_crop_pending_plant=crop,
        latest_crop_rejection=crop,
        latest_animal_purchase_target=animal,
        latest_animal_service_target=animal,
        latest_animal_committed=animal,
        latest_animal_active=animal,
        latest_animal_unplaced=animal,
    )


def _counts(values: jax.Array, size: int) -> jax.Array:
    return jnp.stack(
        tuple(
            jnp.sum(values == item, axis=(1, 2), dtype=jnp.int32)
            for item in range(size)
        ),
        axis=-1,
    )


def _task_counts(controller, task_type: int, size: int, *, item_offset: int = 0):
    tasks = controller.unit_tasks
    active = (tasks.status == TaskStatusV1.ACTIVE) & (
        tasks.task_type == task_type
    )
    item = jnp.clip(tasks.item_id.astype(jnp.int32) - item_offset, 0, size - 1)
    return jnp.sum(
        jax.nn.one_hot(item, size, dtype=jnp.int32) * active[..., None],
        axis=1,
        dtype=jnp.int32,
    )


def update_m37_daily_lifecycle_v1(
    carry: M36CRolloutCarryV3,
    lifecycle: M37LifecycleAggregatesV1,
    calendar: RouteCalendarV3,
    *,
    player: int,
) -> M37LifecycleAggregatesV1:
    """Update one daily aggregate row; repeated same-day calls are idempotent."""

    farm = carry.farm
    states = farm.environment_state
    controller = farm.controller
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    day = jnp.clip(states.step.astype(jnp.int32) // TURNS_PER_DAY, 0, 29)
    fresh_day = day.astype(jnp.int16) != lifecycle.last_observed_day
    fresh_i32 = fresh_day.astype(jnp.int32)

    crop_target = calendar.crop_target_by_day[batch, day].astype(jnp.int32)
    crop_active = _counts(states.tile_crop[:, player], NUM_CROPS)
    crop_debt = jnp.maximum(crop_target - crop_active, 0)
    seed_held = states.seeds[:, player].astype(jnp.int32)
    unit_tasks = controller.unit_tasks
    task_x = jnp.clip(unit_tasks.target_x.astype(jnp.int32), 0, 9)
    task_y = jnp.clip(unit_tasks.target_y.astype(jnp.int32), 0, 9)
    target_kind = states.tile_kind[
        batch[:, None], player, task_y, task_x
    ]
    true_pending_plant = (
        (unit_tasks.status == TaskStatusV1.ACTIVE)
        & (unit_tasks.task_type == TaskTypeV1.CROP_PRODUCTION)
        & (target_kind == TileKind.EMPTY)
    )
    crop_task_item = jnp.clip(
        unit_tasks.item_id.astype(jnp.int32), 0, NUM_CROPS - 1
    )
    crop_tasks = jnp.sum(
        jax.nn.one_hot(crop_task_item, NUM_CROPS, dtype=jnp.int32)
        * true_pending_plant[..., None],
        axis=1,
        dtype=jnp.int32,
    )
    unassigned_crop_debt = jnp.maximum(crop_debt - crop_tasks, 0)
    pending_plant = jnp.minimum(unassigned_crop_debt, seed_held)
    pending_seed = jnp.maximum(unassigned_crop_debt - seed_held, 0)
    crop_rejection = pending_plant

    day_axis = jnp.arange(30, dtype=jnp.int32)[None, :, None]
    purchase_target = jnp.sum(
        jnp.where(
            day_axis <= day[:, None, None],
            calendar.animal_purchase_additions_by_day.astype(jnp.int32),
            0,
        ),
        axis=1,
        dtype=jnp.int32,
    )
    service_target = calendar.animal_service_target_by_day[batch, day].astype(
        jnp.int32
    )
    animal_active = _counts(states.tile_animal[:, player], NUM_ANIMALS)
    in_shed = states.shed[
        :, player, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
    ].astype(jnp.int32)
    carried = jnp.sum(
        states.unit_inventory[
            :, player, :, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
        ].astype(jnp.int32),
        axis=1,
        dtype=jnp.int32,
    )
    unplaced = in_shed + carried
    animal_committed = animal_active + unplaced
    purchase_debt = jnp.maximum(purchase_target - animal_committed, 0)
    activation_debt = jnp.maximum(service_target - animal_active, 0)
    place_tasks = _task_counts(
        controller,
        int(TaskTypeV1.ANIMAL_PLACE),
        NUM_ANIMALS,
        item_offset=NUM_PRODUCTS,
    )
    build_tasks = _task_counts(
        controller, int(TaskTypeV1.BUILD_ANIMAL_STRUCTURE), NUM_ANIMALS
    )

    crop_age = jnp.where(
        crop_debt > 0,
        lifecycle.crop_debt_age_days + fresh_i32[:, None],
        0,
    )
    animal_age = jnp.where(
        (purchase_debt + activation_debt) > 0,
        lifecycle.animal_debt_age_days + fresh_i32[:, None],
        0,
    )

    shed_used = jnp.sum(states.shed[:, player].astype(jnp.int32), axis=-1)
    market_active = controller.market_tasks.status == TaskStatusV1.ACTIVE
    market_full = jnp.sum(market_active, axis=-1, dtype=jnp.int32) >= 10
    unit_task_active = controller.unit_tasks.status == TaskStatusV1.ACTIVE
    active_units = jnp.sum(
        states.unit_active[:, player], axis=-1, dtype=jnp.int32
    )
    assigned_units = jnp.sum(unit_task_active, axis=-1, dtype=jnp.int32)
    no_free_unit = assigned_units >= active_units

    kind = states.tile_kind[:, player]
    animal_tile = states.tile_animal[:, player]
    coop_free = jnp.sum(
        (kind == TileKind.COOP) & (animal_tile < 0),
        axis=(1, 2),
        dtype=jnp.int32,
    )
    pasture_free = jnp.sum(
        (kind == TileKind.PASTURE) & (animal_tile < 0),
        axis=(1, 2),
        dtype=jnp.int32,
    )
    structure_free = jnp.stack((coop_free, pasture_free, pasture_free), axis=-1)
    facility_missing = (purchase_debt > 0) & (structure_free <= 0) & (
        build_tasks <= 0
    )

    crop_needed_cost = jnp.min(
        jnp.where(crop_debt > 0, _SEED_COST[None], 1_000_000), axis=-1
    )
    animal_needed_cost = jnp.min(
        jnp.where(purchase_debt > 0, _ANIMAL_COST[None], 1_000_000), axis=-1
    )
    minimum_needed_cost = jnp.minimum(crop_needed_cost, animal_needed_cost)
    backlog = jnp.any(crop_debt > 0, axis=-1) | jnp.any(
        (purchase_debt + activation_debt) > 0, axis=-1
    )
    cash_block = backlog & (states.money[:, player] < minimum_needed_cost)
    capacity_block = jnp.any(purchase_debt > 0, axis=-1) & (
        shed_used >= SHED_CAPACITY
    )
    facility_block = jnp.any(facility_missing, axis=-1)
    unit_block = jnp.any((unplaced > 0) & (place_tasks <= 0), axis=-1)
    preemption_block = unit_block & no_free_unit
    placement_block = jnp.any((carried > 0) & (place_tasks <= 0), axis=-1)
    wheat_held = (
        states.shed[:, player, 0].astype(jnp.int32)
        + jnp.sum(
            states.unit_inventory[:, player, :, 0].astype(jnp.int32),
            axis=-1,
            dtype=jnp.int32,
        )
    )
    feed_block = (jnp.sum(animal_active, axis=-1) > 0) & (
        wheat_held < jnp.sum(animal_active, axis=-1)
    )
    target_emitted = (
        jnp.any(crop_tasks > 0, axis=-1)
        | jnp.any(place_tasks > 0, axis=-1)
        | jnp.any(
            market_active
            & (
                (controller.market_tasks.task_type == TaskTypeV1.ANIMAL_PURCHASE)
                | (controller.market_tasks.task_type == TaskTypeV1.CROP_PRODUCTION)
            ),
            axis=-1,
        )
    )
    target_not_emitted = backlog & (~target_emitted)
    market_slot_block = jnp.any(purchase_debt > 0, axis=-1) & market_full
    partial_fill = jnp.zeros((batch_size,), dtype=jnp.bool_)

    animal_admissible = jnp.any(
        (purchase_debt > 0)
        & (structure_free > 0)
        & (states.money[:, player, None] >= _ANIMAL_COST[None]),
        axis=-1,
    ) & (shed_used < SHED_CAPACITY)
    crop_admissible = jnp.any(
        (crop_debt > 0)
        & (
            (seed_held > 0)
            | (states.money[:, player, None] >= _SEED_COST[None])
        ),
        axis=-1,
    )
    cash_idle = (
        (animal_admissible | crop_admissible)
        & backlog
        & (~target_emitted)
        & (~market_full)
    )
    blocker_flags = jnp.stack(
        (
            cash_block,
            market_slot_block,
            capacity_block,
            facility_block,
            unit_block,
            preemption_block,
            placement_block,
            feed_block,
            partial_fill,
            target_not_emitted,
        ),
        axis=-1,
    ).astype(jnp.int32)

    return lifecycle._replace(
        last_observed_day=jnp.where(
            fresh_day, day.astype(jnp.int16), lifecycle.last_observed_day
        ),
        crop_target_debt_days=(
            lifecycle.crop_target_debt_days
            + (crop_debt > 0).astype(jnp.int32) * fresh_i32[:, None]
        ),
        animal_purchase_debt_days=(
            lifecycle.animal_purchase_debt_days
            + (purchase_debt > 0).astype(jnp.int32) * fresh_i32[:, None]
        ),
        animal_activation_debt_days=(
            lifecycle.animal_activation_debt_days
            + (activation_debt > 0).astype(jnp.int32) * fresh_i32[:, None]
        ),
        crop_debt_age_days=crop_age,
        animal_debt_age_days=animal_age,
        max_crop_debt_age_days=jnp.maximum(
            lifecycle.max_crop_debt_age_days, crop_age
        ),
        max_animal_debt_age_days=jnp.maximum(
            lifecycle.max_animal_debt_age_days, animal_age
        ),
        cash_idle_while_admissible_backlog_count=(
            lifecycle.cash_idle_while_admissible_backlog_count
            + cash_idle.astype(jnp.int32) * fresh_i32
        ),
        blocker_day_counts=(
            lifecycle.blocker_day_counts + blocker_flags * fresh_i32[:, None]
        ),
        latest_crop_target=crop_target,
        latest_crop_active=crop_active,
        latest_crop_pending_seed=pending_seed,
        latest_crop_pending_plant=pending_plant,
        latest_crop_rejection=crop_rejection,
        latest_animal_purchase_target=purchase_target,
        latest_animal_service_target=service_target,
        latest_animal_committed=animal_committed,
        latest_animal_active=animal_active,
        latest_animal_unplaced=unplaced,
    )


def initialize_m37_split_v1(
    seeds: jax.Array,
    calendar: RouteCalendarV3,
    events: Events,
    tables: StaticTables,
    template: M35FarmGenomeV2,
    *,
    player: int = 0,
) -> M37RolloutCarryV1:
    base = initialize_m36c_split_v3(
        seeds, calendar, events, tables, template, player=player
    )
    lifecycle = update_m37_daily_lifecycle_v1(
        base,
        empty_m37_lifecycle_aggregates_v1(seeds.shape[0]),
        calendar,
        player=player,
    )
    return M37RolloutCarryV1(base, lifecycle)


def m37_daily_plan_carry_v1(
    carry: M37RolloutCarryV1,
    calendar: RouteCalendarV3,
    tables: StaticTables,
    template: M35FarmGenomeV2,
    *,
    player: int = 0,
) -> M37RolloutCarryV1:
    base = m36c_daily_plan_carry_v3(
        carry.base,
        calendar,
        tables,
        template,
        player=player,
        enable_unlock_committed_capital=True,
    )
    lifecycle = update_m37_daily_lifecycle_v1(
        base, carry.lifecycle, calendar, player=player
    )
    return M37RolloutCarryV1(base, lifecycle)


def make_m37_light_chunk_v1(*, chunk_steps: int, player: int = 0):
    base_chunk = make_m36c_light_chunk_v3(
        chunk_steps=chunk_steps,
        player=player,
        enable_unlock_committed_capital=True,
    )

    def chunk(
        carry: M37RolloutCarryV1,
        calendar: RouteCalendarV3,
        events: Events,
        tables: StaticTables,
        template: M35FarmGenomeV2,
        active_steps: jax.Array | int | None = None,
    ) -> M37RolloutCarryV1:
        base = base_chunk(
            carry.base, calendar, events, tables, template, active_steps
        )
        return M37RolloutCarryV1(base, carry.lifecycle)

    return chunk


__all__ = [
    "M37BlockerV1",
    "M37LifecycleAggregatesV1",
    "M37RolloutCarryV1",
    "NUM_M37_BLOCKERS_V1",
    "empty_m37_lifecycle_aggregates_v1",
    "initialize_m37_split_v1",
    "m37_daily_plan_carry_v1",
    "make_m37_light_chunk_v1",
    "update_m37_daily_lifecycle_v1",
]
