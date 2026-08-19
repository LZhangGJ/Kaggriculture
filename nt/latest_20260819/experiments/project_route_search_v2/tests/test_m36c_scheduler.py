from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import FLAG_FERTILIZER_AVAILABLE, TileKind
from kaggriculture_jax.state import reset
from strategic_v5.constants import TaskPhaseV1, TaskStatusV1, TaskTypeV1

from project_route_search_v2.lifecycle import (
    empty_unit_tasks_v2,
    initialize_project_controller_v2,
)
from project_route_search_v2.m36_calendar import empty_route_calendar_v3
from project_route_search_v2.m36_genome_adapter import calendar_step_genome_v3
from project_route_search_v2.m36_scheduler import (
    _choose_proposals_v3,
    mark_m37_daily_unlock_gate_v1,
    materialize_m36_unified_unit_tasks_v3,
    planned_service_tile_mask_v3,
)
from project_route_search_v2.m35_genome import default_m35_farm_genome_v2
from project_route_search_v2.m3_controller import m3_phase_targets_v2


def _states_two_units():
    states = jax.vmap(reset)(jnp.asarray((17,), dtype=jnp.int32))
    active = states.unit_active.at[0, 0, 0].set(True).at[0, 0, 1].set(True)
    pos = states.unit_pos.at[0, 0, 0].set(jnp.asarray((0, 0), dtype=jnp.int8))
    pos = pos.at[0, 0, 1].set(jnp.asarray((9, 9), dtype=jnp.int8))
    return states._replace(unit_active=active, unit_pos=pos)


def _set_task(tasks, unit, task_type, x, y, *, phase=TaskPhaseV1.MOVE_TO_TARGET):
    target = y * 10 + x
    return tasks._replace(
        task_type=tasks.task_type.at[0, unit].set(jnp.int8(task_type)),
        target_id=tasks.target_id.at[0, unit].set(jnp.int16(target)),
        target_x=tasks.target_x.at[0, unit].set(jnp.int8(x)),
        target_y=tasks.target_y.at[0, unit].set(jnp.int8(y)),
        phase=tasks.phase.at[0, unit].set(jnp.int8(phase)),
        start_step=tasks.start_step.at[0, unit].set(jnp.int16(0)),
        deadline_step=tasks.deadline_step.at[0, unit].set(jnp.int16(23)),
        status=tasks.status.at[0, unit].set(jnp.int8(TaskStatusV1.ACTIVE)),
    )


def test_hard_animal_feed_preempts_new_crop_but_not_sticky_task():
    states = _states_two_units()
    # Make target (1, 0) an animal already neglected once, hence hard feed.
    animal = states.tile_animal.at[0, 0, 0, 1].set(jnp.int8(1))
    neglect = states.tile_neglect.at[0, 0, 0, 1].set(jnp.int8(1))
    states = states._replace(tile_animal=animal, tile_neglect=neglect)
    existing = empty_unit_tasks_v2(1)
    crop = _set_task(empty_unit_tasks_v2(1), 0, TaskTypeV1.CROP_PRODUCTION, 0, 1)
    animal_tasks = _set_task(
        empty_unit_tasks_v2(1), 0, TaskTypeV1.ANIMAL_FEED, 1, 0
    )
    allowed = jnp.ones((1, 10, 10), dtype=jnp.bool_)
    selected, diag = _choose_proposals_v3(
        states, existing, crop, animal_tasks, allowed, 0
    )
    assert int(selected.task_type[0, 0]) == TaskTypeV1.ANIMAL_FEED
    assert int(diag.deadline_preemption_count[0]) == 1

    sticky = _set_task(
        empty_unit_tasks_v2(1), 0, TaskTypeV1.CROP_PRODUCTION, 0, 1
    )
    selected, diag = _choose_proposals_v3(
        states, sticky, crop, animal_tasks, allowed, 0
    )
    assert int(selected.task_type[0, 0]) == TaskTypeV1.CROP_PRODUCTION
    assert int(diag.sticky_task_count[0]) == 1


def test_adjacent_local_swap_strictly_reduces_travel():
    states = _states_two_units()
    existing = empty_unit_tasks_v2(1)
    crop = empty_unit_tasks_v2(1)
    crop = _set_task(crop, 0, TaskTypeV1.CROP_PRODUCTION, 9, 9)
    crop = _set_task(crop, 1, TaskTypeV1.CROP_PRODUCTION, 0, 0)
    selected, diag = _choose_proposals_v3(
        states,
        existing,
        crop,
        empty_unit_tasks_v2(1),
        jnp.ones((1, 10, 10), dtype=jnp.bool_),
        0,
    )
    assert int(diag.local_swap_count[0]) == 1
    assert (int(selected.target_x[0, 0]), int(selected.target_y[0, 0])) == (0, 0)
    assert (int(selected.target_x[0, 1]), int(selected.target_y[0, 1])) == (9, 9)


def test_planned_release_keeps_only_daily_service_target():
    states = jax.vmap(reset)(jnp.asarray((29,), dtype=jnp.int32))
    kinds = states.tile_kind.at[0, 0, 0, 0].set(jnp.int8(TileKind.PASTURE))
    kinds = kinds.at[0, 0, 4, 4].set(jnp.int8(TileKind.PASTURE))
    animals = states.tile_animal.at[0, 0, 0, 0].set(jnp.int8(2))
    animals = animals.at[0, 0, 4, 4].set(jnp.int8(2))
    states = states._replace(tile_kind=kinds, tile_animal=animals)
    calendar = empty_route_calendar_v3(1)
    service = calendar.animal_service_target_by_day.at[0, 0, 2].set(jnp.int16(1))
    calendar = calendar._replace(animal_service_target_by_day=service)
    allowed, release_count = planned_service_tile_mask_v3(states, calendar, 0)
    assert int(np.asarray(allowed).sum()) == 1
    assert int(release_count[0]) == 1


def test_daily_calendar_adapter_uses_current_day_without_six_phase_truncation():
    states = jax.vmap(reset)(jnp.asarray((31,), dtype=jnp.int32))
    states = states._replace(step=jnp.asarray((6 * 24,), dtype=jnp.int16))
    calendar = empty_route_calendar_v3(1)
    crop = calendar.crop_target_by_day.at[0, 6].set(
        jnp.asarray((3, 0, 0, 6, 12), dtype=jnp.int16)
    )
    animal = calendar.animal_service_target_by_day.at[0, 6].set(
        jnp.asarray((0, 1, 1), dtype=jnp.int16)
    )
    purchases = calendar.animal_purchase_additions_by_day.at[0, 0].set(
        jnp.asarray((0, 2, 2), dtype=jnp.int16)
    )
    purchases = purchases.at[0, 5].add(
        jnp.asarray((0, 2, 0), dtype=jnp.int16)
    )
    land = calendar.land_additions_by_day.at[0, 2].set(jnp.int8(1))
    land = land.at[0, 6].set(jnp.int8(1))
    hands = calendar.hand_target_by_day.at[0, 6].set(jnp.int8(9))
    calendar = calendar._replace(
        crop_target_by_day=crop,
        animal_purchase_additions_by_day=purchases,
        animal_service_target_by_day=animal,
        land_additions_by_day=land,
        hand_target_by_day=hands,
    )
    genome = calendar_step_genome_v3(states, calendar)
    np.testing.assert_array_equal(
        np.asarray(genome.crop.crop_target[0, 0]), np.asarray((3, 0, 0, 6, 12))
    )
    np.testing.assert_array_equal(
        np.asarray(genome.animal.animal_target[0, 0]), np.asarray((0, 4, 2))
    )
    assert int(genome.crop.land_target[0, 0]) == 3
    assert int(genome.crop.hand_target[0, 0]) == 9
    assert int(genome.crop.harvest_trigger_units[0, 3]) == 1


def test_calendar_adapter_preserves_explicit_large_animal_commitment_and_care():
    states = jax.vmap(reset)(jnp.asarray((61,), dtype=jnp.int32))
    calendar = empty_route_calendar_v3(1)
    purchases = calendar.animal_purchase_additions_by_day.at[0, 0].set(
        jnp.asarray((0, 6, 12), dtype=jnp.int16)
    )
    service = calendar.animal_service_target_by_day.at[0, 0].set(
        jnp.asarray((0, 6, 12), dtype=jnp.int16)
    )
    care = calendar.animal_care_policy_by_day.at[0, 0].set(
        jnp.asarray((0, 2, 2), dtype=jnp.int8)
    )
    hands = calendar.hand_target_by_day.at[0, 0].set(jnp.int8(5))
    calendar = calendar._replace(
        animal_purchase_additions_by_day=purchases,
        animal_service_target_by_day=service,
        animal_care_policy_by_day=care,
        hand_target_by_day=hands,
    )
    genome = calendar_step_genome_v3(states, calendar)
    _, effective, _, _ = m3_phase_targets_v2(states, genome.animal)
    np.testing.assert_array_equal(
        np.asarray(effective[0]), np.asarray((0, 6, 12))
    )
    np.testing.assert_array_equal(
        np.asarray(genome.animal.care_policy[0]), np.asarray((0, 2, 2))
    )


def test_same_day_crop_operate_continuation_is_not_preempted():
    states = jax.vmap(reset)(jnp.asarray((59,), dtype=jnp.int32))
    states = states._replace(step=jnp.asarray((22,), dtype=jnp.int16))
    active = states.unit_active.at[0, 0, 0].set(True)
    kind = states.tile_kind.at[0, 0, 0, 0].set(jnp.int8(TileKind.PLANT))
    crop = states.tile_crop.at[0, 0, 0, 0].set(jnp.int8(0))
    origin = states.tile_origin_day.at[0, 0, 0, 0].set(
        jnp.asarray(0, dtype=states.tile_origin_day.dtype)
    )
    states = states._replace(
        unit_active=active,
        tile_kind=kind,
        tile_crop=crop,
        tile_origin_day=origin,
    )
    task = _set_task(
        empty_unit_tasks_v2(1),
        0,
        TaskTypeV1.CROP_PRODUCTION,
        0,
        0,
        phase=TaskPhaseV1.OPERATE,
    )
    controller = initialize_project_controller_v2(states, 0)._replace(unit_tasks=task)
    planned, _ = materialize_m36_unified_unit_tasks_v3(
        states,
        controller,
        default_m35_farm_genome_v2(1),
        empty_route_calendar_v3(1),
        0,
    )
    assert int(planned.unit_tasks.task_type[0, 0]) == TaskTypeV1.CROP_PRODUCTION
    assert int(planned.unit_tasks.phase[0, 0]) == TaskPhaseV1.OPERATE


def test_worker_carrying_wheat_continues_feed_chain_before_nearby_crop() -> None:
    states = jax.vmap(reset)(jnp.asarray((65,), dtype=jnp.int32))
    states = states._replace(
        unit_active=states.unit_active.at[0, 0, 0].set(True),
        unit_pos=states.unit_pos.at[0, 0, 0].set(
            jnp.asarray((0, 0), dtype=jnp.int8)
        ),
        unit_inventory=states.unit_inventory.at[0, 0, 0, 0].set(
            jnp.int16(1)
        ),
        tile_kind=states.tile_kind.at[0, 0, 0, 1]
        .set(jnp.int8(TileKind.PLANT))
        .at[0, 0, 9, 9]
        .set(jnp.int8(TileKind.PASTURE)),
        tile_crop=states.tile_crop.at[0, 0, 0, 1].set(jnp.int8(0)),
        tile_animal=states.tile_animal.at[0, 0, 9, 9].set(jnp.int8(1)),
    )
    calendar = empty_route_calendar_v3(1)
    calendar = calendar._replace(
        crop_target_by_day=calendar.crop_target_by_day.at[0, 0, 0].set(
            jnp.int16(1)
        ),
        animal_purchase_additions_by_day=(
            calendar.animal_purchase_additions_by_day.at[0, 0, 1].set(
                jnp.int16(1)
            )
        ),
        animal_service_target_by_day=(
            calendar.animal_service_target_by_day.at[0, 0, 1].set(
                jnp.int16(1)
            )
        ),
        animal_care_policy_by_day=(
            calendar.animal_care_policy_by_day.at[0, 0, 1].set(jnp.int8(2))
        ),
    )
    genome = calendar_step_genome_v3(states, calendar)
    controller = initialize_project_controller_v2(states, 0)
    planned, _ = materialize_m36_unified_unit_tasks_v3(
        states,
        controller,
        genome,
        calendar,
        0,
        enable_unlock_committed_capital=True,
    )
    assert int(planned.unit_tasks.task_type[0, 0]) == TaskTypeV1.ANIMAL_FEED
    assert int(planned.unit_tasks.target_id[0, 0]) == 99


def test_one_urgent_feed_does_not_clear_every_sticky_crop_route() -> None:
    states = jax.vmap(reset)(jnp.asarray((67,), dtype=jnp.int32))
    states = states._replace(
        unit_active=states.unit_active.at[0, 0, :3].set(True),
        tile_kind=states.tile_kind.at[0, 0, 0, 1].set(
            jnp.int8(TileKind.PASTURE)
        ),
        tile_animal=states.tile_animal.at[0, 0, 0, 1].set(jnp.int8(1)),
        tile_neglect=states.tile_neglect.at[0, 0, 0, 1].set(jnp.int8(1)),
        shed=states.shed.at[0, 0, 0].set(jnp.int16(1)),
    )
    calendar = empty_route_calendar_v3(1)
    calendar = calendar._replace(
        animal_purchase_additions_by_day=(
            calendar.animal_purchase_additions_by_day.at[0, 0, 1].set(
                jnp.int16(1)
            )
        ),
        animal_service_target_by_day=(
            calendar.animal_service_target_by_day.at[0, :, 1].set(
                jnp.int16(1)
            )
        ),
        hand_target_by_day=calendar.hand_target_by_day.at[0, :].set(
            jnp.int8(2)
        ),
    )
    tasks = empty_unit_tasks_v2(1)
    tasks = _set_task(tasks, 0, TaskTypeV1.CROP_PRODUCTION, 2, 2)
    tasks = _set_task(tasks, 1, TaskTypeV1.CROP_PRODUCTION, 3, 3)
    tasks = _set_task(tasks, 2, TaskTypeV1.CROP_PRODUCTION, 4, 4)
    controller = initialize_project_controller_v2(states, 0)._replace(
        unit_tasks=tasks
    )
    planned, _ = materialize_m36_unified_unit_tasks_v3(
        states,
        controller,
        calendar_step_genome_v3(states, calendar),
        calendar,
        0,
        enable_unlock_committed_capital=True,
    )
    original_targets = jnp.asarray((22, 33, 44), dtype=jnp.int16)
    preserved_crop = (
        (planned.unit_tasks.status[0] == TaskStatusV1.ACTIVE)
        & (planned.unit_tasks.task_type[0] == TaskTypeV1.CROP_PRODUCTION)
        & jnp.any(
            planned.unit_tasks.target_id[0, :, None]
            == original_targets[None, :],
            axis=-1,
        )
    )
    urgent_feed = (
        (planned.unit_tasks.status[0] == TaskStatusV1.ACTIVE)
        & (planned.unit_tasks.task_type[0] == TaskTypeV1.ANIMAL_FEED)
    )
    assert int(jnp.sum(preserved_crop)) == 2
    assert int(jnp.sum(urgent_feed)) == 1


def test_optional_feed_does_not_hide_a_different_survival_feed_deadline() -> None:
    states = jax.vmap(reset)(jnp.asarray((71,), dtype=jnp.int32))
    states = states._replace(
        unit_active=states.unit_active.at[0, 0, :3].set(True),
        tile_kind=states.tile_kind.at[0, 0, 0, 1].set(
            jnp.int8(TileKind.PASTURE)
        ).at[0, 0, 0, 2].set(jnp.int8(TileKind.PASTURE)),
        tile_animal=states.tile_animal.at[0, 0, 0, 1].set(
            jnp.int8(1)
        ).at[0, 0, 0, 2].set(jnp.int8(1)),
        tile_neglect=states.tile_neglect.at[0, 0, 0, 1].set(jnp.int8(1)),
        shed=states.shed.at[0, 0, 0].set(jnp.int16(2)),
    )
    calendar = empty_route_calendar_v3(1)
    calendar = calendar._replace(
        animal_purchase_additions_by_day=(
            calendar.animal_purchase_additions_by_day.at[0, 0, 1].set(
                jnp.int16(2)
            )
        ),
        animal_service_target_by_day=(
            calendar.animal_service_target_by_day.at[0, :, 1].set(
                jnp.int16(2)
            )
        ),
        hand_target_by_day=calendar.hand_target_by_day.at[0, :].set(
            jnp.int8(2)
        ),
    )
    tasks = empty_unit_tasks_v2(1)
    tasks = _set_task(tasks, 0, TaskTypeV1.ANIMAL_FEED, 2, 0)
    tasks = _set_task(tasks, 1, TaskTypeV1.CROP_PRODUCTION, 3, 3)
    tasks = _set_task(tasks, 2, TaskTypeV1.CROP_PRODUCTION, 4, 4)
    controller = initialize_project_controller_v2(states, 0)._replace(
        unit_tasks=tasks
    )
    planned, _ = materialize_m36_unified_unit_tasks_v3(
        states,
        controller,
        calendar_step_genome_v3(states, calendar),
        calendar,
        0,
        enable_unlock_committed_capital=True,
    )
    preserved_crop = (
        (planned.unit_tasks.status[0] == TaskStatusV1.ACTIVE)
        & (planned.unit_tasks.task_type[0] == TaskTypeV1.CROP_PRODUCTION)
    )
    survival_feed = (
        (planned.unit_tasks.status[0] == TaskStatusV1.ACTIVE)
        & (planned.unit_tasks.task_type[0] == TaskTypeV1.ANIMAL_FEED)
        & (planned.unit_tasks.target_id[0] == 1)
    )
    assert int(jnp.sum(preserved_crop)) == 2
    assert int(jnp.sum(survival_feed)) == 1


def test_optional_task_on_survival_tile_is_preempted_before_other_work() -> None:
    states = jax.vmap(reset)(jnp.asarray((79,), dtype=jnp.int32))
    states = states._replace(
        unit_active=states.unit_active.at[0, 0, :3].set(True),
        tile_kind=states.tile_kind.at[0, 0, 0, 1].set(
            jnp.int8(TileKind.PASTURE)
        ),
        tile_animal=states.tile_animal.at[0, 0, 0, 1].set(jnp.int8(1)),
        tile_neglect=states.tile_neglect.at[0, 0, 0, 1].set(jnp.int8(1)),
        tile_flags=states.tile_flags.at[0, 0, 0, 1].set(
            jnp.uint8(FLAG_FERTILIZER_AVAILABLE)
        ),
        shed=states.shed.at[0, 0, 0].set(jnp.int16(1)),
    )
    calendar = empty_route_calendar_v3(1)
    calendar = calendar._replace(
        animal_purchase_additions_by_day=(
            calendar.animal_purchase_additions_by_day.at[0, 0, 1].set(
                jnp.int16(1)
            )
        ),
        animal_service_target_by_day=(
            calendar.animal_service_target_by_day.at[0, :, 1].set(
                jnp.int16(1)
            )
        ),
        hand_target_by_day=calendar.hand_target_by_day.at[0, :].set(
            jnp.int8(2)
        ),
    )
    tasks = empty_unit_tasks_v2(1)
    tasks = _set_task(
        tasks, 0, TaskTypeV1.ANIMAL_COLLECT_FERTILIZER, 1, 0
    )
    tasks = _set_task(tasks, 1, TaskTypeV1.CROP_PRODUCTION, 3, 3)
    tasks = _set_task(tasks, 2, TaskTypeV1.CROP_PRODUCTION, 4, 4)
    controller = initialize_project_controller_v2(states, 0)._replace(
        unit_tasks=tasks
    )
    planned, _ = materialize_m36_unified_unit_tasks_v3(
        states,
        controller,
        calendar_step_genome_v3(states, calendar),
        calendar,
        0,
        enable_unlock_committed_capital=True,
    )
    preserved_crop = (
        (planned.unit_tasks.status[0] == TaskStatusV1.ACTIVE)
        & (planned.unit_tasks.task_type[0] == TaskTypeV1.CROP_PRODUCTION)
    )
    survival_feed = (
        (planned.unit_tasks.status[0] == TaskStatusV1.ACTIVE)
        & (planned.unit_tasks.task_type[0] == TaskTypeV1.ANIMAL_FEED)
        & (planned.unit_tasks.target_id[0] == 1)
    )
    assert int(jnp.sum(preserved_crop)) == 2
    assert int(jnp.sum(survival_feed)) == 1


def test_mixed_new_animal_wave_gets_one_unlock_lane_per_species() -> None:
    states = jax.vmap(reset)(jnp.asarray((83,), dtype=jnp.int32))
    states = states._replace(
        step=jnp.asarray((1,), dtype=jnp.int16),
        unit_active=states.unit_active.at[0, 0, :6].set(True),
        shed=(
            states.shed.at[0, 0, 10].set(jnp.int16(2)).at[0, 0, 11].set(
                jnp.int16(2)
            )
        ),
    )
    calendar = empty_route_calendar_v3(1)
    calendar = calendar._replace(
        crop_target_by_day=calendar.crop_target_by_day.at[0, 0].set(
            jnp.asarray((7, 0, 0, 0, 12), dtype=jnp.int16)
        ),
        animal_purchase_additions_by_day=(
            calendar.animal_purchase_additions_by_day.at[0, 0].set(
                jnp.asarray((0, 2, 2), dtype=jnp.int16)
            )
        ),
        animal_service_target_by_day=(
            calendar.animal_service_target_by_day.at[0, 0].set(
                jnp.asarray((0, 2, 2), dtype=jnp.int16)
            )
        ),
        hand_target_by_day=calendar.hand_target_by_day.at[0, 0].set(
            jnp.int8(5)
        ),
    )
    genome = calendar_step_genome_v3(states, calendar)
    controller = initialize_project_controller_v2(states, 0)
    controller = mark_m37_daily_unlock_gate_v1(
        states, controller, calendar, genome, 0
    )
    planned, _ = materialize_m36_unified_unit_tasks_v3(
        states,
        controller,
        genome,
        calendar,
        0,
        enable_unlock_committed_capital=True,
    )
    animal_activation = (
        (planned.unit_tasks.status[0] == TaskStatusV1.ACTIVE)
        & (
            (planned.unit_tasks.task_type[0] == TaskTypeV1.ANIMAL_PLACE)
            | (
                planned.unit_tasks.task_type[0]
                == TaskTypeV1.BUILD_ANIMAL_STRUCTURE
            )
        )
    )
    assert int(jnp.sum(animal_activation)) >= 2


def test_animal_partition_without_executable_task_falls_back_to_crop() -> None:
    states = jax.vmap(reset)(jnp.asarray((89,), dtype=jnp.int32))
    states = states._replace(
        step=jnp.asarray((2 * 24 + 16,), dtype=jnp.int16),
        unit_active=states.unit_active.at[0, 0, 0].set(True),
        unit_pos=states.unit_pos.at[0, 0, 0].set(
            jnp.asarray((0, 0), dtype=jnp.int8)
        ),
        tile_kind=(
            states.tile_kind.at[0, 0, 0, 0]
            .set(jnp.int8(TileKind.PASTURE))
            .at[0, 0, 0, 1]
            .set(jnp.int8(TileKind.PLANT))
        ),
        tile_animal=states.tile_animal.at[0, 0, 0, 0].set(jnp.int8(1)),
        tile_neglect=states.tile_neglect.at[0, 0, 0, 0].set(jnp.int8(1)),
        tile_crop=states.tile_crop.at[0, 0, 0, 1].set(jnp.int8(0)),
        tile_origin_day=states.tile_origin_day.at[0, 0, 0, 1].set(
            jnp.asarray(2, dtype=states.tile_origin_day.dtype)
        ),
    )
    calendar = empty_route_calendar_v3(1)
    calendar = calendar._replace(
        crop_target_by_day=calendar.crop_target_by_day.at[0, 2, 0].set(
            jnp.int16(1)
        ),
        animal_purchase_additions_by_day=(
            calendar.animal_purchase_additions_by_day.at[0, 0, 1].set(
                jnp.int16(1)
            )
        ),
        animal_service_target_by_day=(
            calendar.animal_service_target_by_day.at[0, 2, 1].set(jnp.int16(1))
        ),
    )
    genome = calendar_step_genome_v3(states, calendar)
    controller = initialize_project_controller_v2(states, 0)
    planned, diagnostics = materialize_m36_unified_unit_tasks_v3(
        states,
        controller,
        genome,
        calendar,
        0,
        enable_unlock_committed_capital=True,
    )
    # The cheap partition sees a survival FEED at distance zero, but the full
    # animal planner cannot execute it because neither the unit nor shed has
    # wheat.  The only worker must therefore be returned to the adjacent crop.
    assert int(planned.unit_tasks.task_type[0, 0]) == TaskTypeV1.WATER_CROP
    assert int(planned.unit_tasks.target_id[0, 0]) == 1
    assert int(diagnostics.selected_crop_count[0]) == 1


def test_next_day_crop_contraction_prevents_myopic_replant() -> None:
    states = jax.vmap(reset)(jnp.asarray((97,), dtype=jnp.int32))
    kinds = states.tile_kind
    crops = states.tile_crop
    flags = states.tile_flags
    origins = states.tile_origin_day
    for x in range(3):
        kinds = kinds.at[0, 0, 0, x].set(jnp.int8(TileKind.PLANT))
        crops = crops.at[0, 0, 0, x].set(jnp.int8(0))
        # Existing crops are already watered, so the only possible crop work
        # would be an unnecessary replacement toward today's larger target.
        flags = flags.at[0, 0, 0, x].set(jnp.uint8(1))
        origins = origins.at[0, 0, 0, x].set(jnp.int8(4))
    states = states._replace(
        step=jnp.asarray((4 * 24,), dtype=jnp.int16),
        unit_active=states.unit_active.at[0, 0, :4].set(True),
        seeds=states.seeds.at[0, 0, 0].set(jnp.int16(4)),
        tile_kind=kinds,
        tile_crop=crops,
        tile_flags=flags,
        tile_origin_day=origins,
    )
    calendar = empty_route_calendar_v3(1)
    crop_targets = calendar.crop_target_by_day
    crop_targets = crop_targets.at[0, 4, 0].set(jnp.int16(7))
    crop_targets = crop_targets.at[0, 5, 0].set(jnp.int16(3))
    calendar = calendar._replace(crop_target_by_day=crop_targets)
    genome = calendar_step_genome_v3(states, calendar)
    planned, _ = materialize_m36_unified_unit_tasks_v3(
        states,
        initialize_project_controller_v2(states, 0),
        genome,
        calendar,
        0,
    )
    unnecessary_replant = (
        (planned.unit_tasks.status[0] == TaskStatusV1.ACTIVE)
        & (planned.unit_tasks.task_type[0] == TaskTypeV1.CROP_PRODUCTION)
        & (planned.unit_tasks.item_id[0] == 0)
    )
    assert int(jnp.sum(unnecessary_replant)) == 0


def test_survival_feed_reserves_one_lane_per_uncovered_animal():
    states = jax.vmap(reset)(jnp.asarray((61,), dtype=jnp.int32))
    states = states._replace(step=jnp.asarray((1,), dtype=jnp.int16))
    active = states.unit_active.at[0, 0, :8].set(True)
    wheat = states.shed.at[0, 0, 0].set(jnp.int16(10))
    kind = states.tile_kind
    animal = states.tile_animal
    neglect = states.tile_neglect
    for x, y in ((0, 0), (1, 0), (2, 0), (3, 0)):
        kind = kind.at[0, 0, y, x].set(jnp.int8(TileKind.PASTURE))
        animal = animal.at[0, 0, y, x].set(jnp.int8(1))
        neglect = neglect.at[0, 0, y, x].set(jnp.int8(1))
    crop = states.tile_crop
    origin = states.tile_origin_day
    for x, y in ((0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)):
        kind = kind.at[0, 0, y, x].set(jnp.int8(TileKind.PLANT))
        crop = crop.at[0, 0, y, x].set(jnp.int8(0))
        origin = origin.at[0, 0, y, x].set(
            jnp.asarray(0, dtype=origin.dtype)
        )
    states = states._replace(
        unit_active=active,
        shed=wheat,
        tile_kind=kind,
        tile_animal=animal,
        tile_neglect=neglect,
        tile_crop=crop,
        tile_origin_day=origin,
    )
    existing = _set_task(
        empty_unit_tasks_v2(1),
        0,
        TaskTypeV1.ANIMAL_FEED,
        0,
        0,
    )
    controller = initialize_project_controller_v2(states, 0)._replace(
        unit_tasks=existing
    )
    calendar = empty_route_calendar_v3(1)
    service = calendar.animal_service_target_by_day.at[0, 0, 1].set(
        jnp.int16(4)
    )
    calendar = calendar._replace(animal_service_target_by_day=service)
    planned, diagnostics = materialize_m36_unified_unit_tasks_v3(
        states,
        controller,
        default_m35_farm_genome_v2(1),
        calendar,
        0,
    )
    feed_count = jnp.sum(
        (planned.unit_tasks.status[0] == TaskStatusV1.ACTIVE)
        & (planned.unit_tasks.task_type[0] == TaskTypeV1.ANIMAL_FEED)
    )
    assert int(feed_count) == 4
    assert int(diagnostics.selected_animal_count[0]) >= 3


def test_fertilizer_ready_animal_reserves_a_banking_lane_alongside_crop_work():
    states = jax.vmap(reset)(jnp.asarray((67,), dtype=jnp.int32))
    states = states._replace(step=jnp.asarray((1,), dtype=jnp.int16))
    active = states.unit_active.at[0, 0, :2].set(True)
    positions = states.unit_pos.at[0, 0, 0].set(
        jnp.asarray((0, 0), dtype=jnp.int8)
    )
    kind = states.tile_kind.at[0, 0, 0, 0].set(jnp.int8(TileKind.PASTURE))
    kind = kind.at[0, 0, 2, 0].set(jnp.int8(TileKind.PLANT))
    animal = states.tile_animal.at[0, 0, 0, 0].set(jnp.int8(1))
    crop = states.tile_crop.at[0, 0, 2, 0].set(jnp.int8(0))
    origin = states.tile_origin_day.at[0, 0, 2, 0].set(jnp.int8(0))
    flags = states.tile_flags.at[0, 0, 0, 0].set(
        jnp.uint8(FLAG_FERTILIZER_AVAILABLE)
    )
    states = states._replace(
        unit_active=active,
        unit_pos=positions,
        tile_kind=kind,
        tile_animal=animal,
        tile_crop=crop,
        tile_origin_day=origin,
        tile_flags=flags,
    )
    calendar = empty_route_calendar_v3(1)
    calendar = calendar._replace(
        crop_target_by_day=calendar.crop_target_by_day.at[0, 0, 0].set(
            jnp.int16(1)
        ),
        animal_purchase_additions_by_day=(
            calendar.animal_purchase_additions_by_day.at[0, 0, 1].set(
                jnp.int16(1)
            )
        ),
        animal_service_target_by_day=(
            calendar.animal_service_target_by_day.at[0, 0, 1].set(jnp.int16(1))
        ),
    )
    genome = calendar_step_genome_v3(states, calendar)
    controller = initialize_project_controller_v2(states, 0)
    planned, diagnostics = materialize_m36_unified_unit_tasks_v3(
        states, controller, genome, calendar, 0
    )
    task = planned.unit_tasks
    fertilizer_count = jnp.sum(
        (task.status[0] == TaskStatusV1.ACTIVE)
        & (task.task_type[0] == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
    )
    assert int(fertilizer_count) == 1
    assert int(diagnostics.selected_animal_count[0]) >= 1


def test_cross_day_crop_debt_allocates_multiple_planting_lanes() -> None:
    states = jax.vmap(reset)(jnp.asarray((73,), dtype=jnp.int32))
    states = states._replace(
        step=jnp.asarray((5 * 24,), dtype=jnp.int16),
        unit_active=states.unit_active.at[0, 0, :6].set(True),
        seeds=states.seeds.at[0, 0, 0].set(jnp.int16(10)),
        tile_kind=states.tile_kind.at[0, 0, 0, 0].set(
            jnp.int8(TileKind.PASTURE)
        ),
        tile_animal=states.tile_animal.at[0, 0, 0, 0].set(jnp.int8(1)),
        tile_flags=states.tile_flags.at[0, 0, 0, 0].set(
            jnp.uint8(FLAG_FERTILIZER_AVAILABLE)
        ),
    )
    calendar = empty_route_calendar_v3(1)
    crop_targets = calendar.crop_target_by_day.at[0, 2:, 0].set(
        jnp.int16(6)
    )
    animal_purchases = (
        calendar.animal_purchase_additions_by_day.at[0, 0, 1].set(
            jnp.int16(1)
        )
    )
    animal_service = calendar.animal_service_target_by_day.at[0, :, 1].set(
        jnp.int16(1)
    )
    calendar = calendar._replace(
        crop_target_by_day=crop_targets,
        animal_purchase_additions_by_day=animal_purchases,
        animal_service_target_by_day=animal_service,
        hand_target_by_day=calendar.hand_target_by_day.at[0, :,].set(
            jnp.int8(5)
        ),
    )
    genome = calendar_step_genome_v3(states, calendar)
    controller = initialize_project_controller_v2(states, 0)
    controller = mark_m37_daily_unlock_gate_v1(
        states, controller, calendar, genome, 0
    )
    assert int(controller.projects.priority_class[0, 0]) == 4

    planned, _ = materialize_m36_unified_unit_tasks_v3(
        states,
        controller,
        genome,
        calendar,
        0,
        enable_unlock_committed_capital=True,
        enable_crop_debt_lane_repair=True,
    )
    crop_tasks = (
        (planned.unit_tasks.status[0] == TaskStatusV1.ACTIVE)
        & (planned.unit_tasks.task_type[0] == TaskTypeV1.CROP_PRODUCTION)
        & (planned.unit_tasks.item_id[0] == 0)
    )
    assert int(jnp.sum(crop_tasks)) >= 2
