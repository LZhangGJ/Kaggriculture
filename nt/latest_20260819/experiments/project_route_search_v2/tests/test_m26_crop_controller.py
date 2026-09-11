from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import (
    FLAG_WATERED,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.state import reset
from strategic_v5.constants import TaskStatusV1, TaskTypeV1

from project_route_search_v2.lifecycle import initialize_project_controller_v2
from project_route_search_v2.m26_controller import (
    ensure_m26_projects_v2,
    m26_phase_v2,
    m26_policy_step_v2,
    materialize_m26_market_tasks_v2,
    materialize_m26_unit_tasks_v2,
)
from project_route_search_v2.crop_executor import clear_invalidated_crop_tasks_v2
from project_route_search_v2.m26_genome import default_m26_crop_genome_v2


def _states(seeds=(1701, 1702)):
    return jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))


def _zero_targets(genome):
    return genome._replace(crop_target=jnp.zeros_like(genome.crop_target))


def test_m26_phase_uses_active_phase_count_and_supports_six_phases() -> None:
    states = _states()
    genome = default_m26_crop_genome_v2(2)._replace(
        phase_count=jnp.full((2,), 6, dtype=jnp.int8),
        phase_start_step=jnp.broadcast_to(
            jnp.asarray((0, 24, 48, 72, 96, 120), dtype=jnp.int16), (2, 6)
        ),
    )
    np.testing.assert_array_equal(m26_phase_v2(states, genome), 0)
    states = states._replace(step=jnp.full((2,), 120, dtype=jnp.int16))
    np.testing.assert_array_equal(m26_phase_v2(states, genome), 5)


def test_m26_land_target_changes_buy_land_while_window_only_gates() -> None:
    states = _states()._replace(
        step=jnp.full((2,), 6 * 24, dtype=jnp.int16),
        money=jnp.full((2, 2), 20_000, dtype=jnp.int32),
    )
    genome = _zero_targets(default_m26_crop_genome_v2(2))._replace(
        hand_target=jnp.zeros((2, 6), dtype=jnp.int8)
    )
    controller = initialize_project_controller_v2(states, 0)
    controller = ensure_m26_projects_v2(states, controller, genome, 0)
    market = materialize_m26_market_tasks_v2(states, controller, genome, 0).market_tasks
    assert np.all(np.any(np.asarray(market.task_type) == TaskTypeV1.BUY_LAND, axis=-1))

    no_land = genome._replace(land_target=genome.land_target.at[:, 1].set(jnp.int8(1)))
    market = materialize_m26_market_tasks_v2(states, controller, no_land, 0).market_tasks
    assert not np.any(np.asarray(market.task_type) == TaskTypeV1.BUY_LAND)

    delayed = genome._replace(land_start_step=genome.land_start_step.at[:, 1].set(7 * 24))
    market = materialize_m26_market_tasks_v2(states, controller, delayed, 0).market_tasks
    assert not np.any(np.asarray(market.task_type) == TaskTypeV1.BUY_LAND)


def test_all_five_crops_share_the_same_seed_market_path() -> None:
    states = _states()._replace(money=jnp.full((2, 2), 100_000, dtype=jnp.int32))
    genome = _zero_targets(default_m26_crop_genome_v2(2))
    genome = genome._replace(crop_target=genome.crop_target.at[:, 0, :].set(1))
    controller = initialize_project_controller_v2(states, 0)
    controller = ensure_m26_projects_v2(states, controller, genome, 0)
    market = materialize_m26_market_tasks_v2(states, controller, genome, 0).market_tasks
    active = np.asarray(market.status) == TaskStatusV1.ACTIVE
    crop_seed_items = np.where(
        active & (np.asarray(market.task_type) == TaskTypeV1.CROP_PRODUCTION),
        np.asarray(market.item_id),
        -1,
    )
    assert all(set(row[row >= 0].tolist()) == {0, 1, 2, 3, 4} for row in crop_seed_items)


def test_harvest_age_and_trigger_control_harvest_action() -> None:
    base = _states()
    states = base._replace(
        step=jnp.full((2,), 8 * 24, dtype=jnp.int16),
        tile_kind=base.tile_kind.at[:, 0, 4, 4].set(TileKind.PLANT),
        tile_crop=base.tile_crop.at[:, 0, 4, 4].set(2),
        tile_origin_day=base.tile_origin_day.at[:, 0, 4, 4].set(0),
        tile_yield=base.tile_yield.at[:, 0, 4, 4].set(2),
        tile_flags=base.tile_flags.at[:, 0, 4, 4].set(jnp.uint8(FLAG_WATERED)),
    )
    genome = _zero_targets(default_m26_crop_genome_v2(2))._replace(
        hand_target=jnp.zeros((2, 6), dtype=jnp.int8)
    )
    controller = initialize_project_controller_v2(states, 0)
    action, _ = m26_policy_step_v2(states, controller, genome, 0)
    np.testing.assert_array_equal(action.unit_op[:, 0], UnitOp.HARVEST)

    delayed = genome._replace(
        harvest_min_age_days=genome.harvest_min_age_days.at[:, 2].set(jnp.int8(9))
    )
    action, _ = m26_policy_step_v2(states, controller, delayed, 0)
    assert not np.any(np.asarray(action.unit_op[:, 0]) == UnitOp.HARVEST)

    high_trigger = genome._replace(
        harvest_trigger_units=genome.harvest_trigger_units.at[:, 2].set(jnp.int8(3))
    )
    action, _ = m26_policy_step_v2(states, controller, high_trigger, 0)
    assert not np.any(np.asarray(action.unit_op[:, 0]) == UnitOp.HARVEST)


def test_deposit_min_value_changes_return_to_shed_task() -> None:
    base = _states()
    states = base._replace(
        unit_pos=base.unit_pos.at[:, 0, 0].set(jnp.asarray((0, 0), dtype=jnp.int8)),
        unit_inventory=base.unit_inventory.at[:, 0, 0, 2].set(1),
    )
    genome = _zero_targets(default_m26_crop_genome_v2(2))._replace(
        hand_target=jnp.zeros((2, 6), dtype=jnp.int8),
        deposit_min_value=jnp.full((2,), 50, dtype=jnp.int32),
    )
    controller = initialize_project_controller_v2(states, 0)
    controller = ensure_m26_projects_v2(states, controller, genome, 0)
    controller = materialize_m26_unit_tasks_v2(states, controller, genome, 0)
    np.testing.assert_array_equal(controller.unit_tasks.task_type[:, 0], TaskTypeV1.SHED_DEPOSIT)

    hold = genome._replace(deposit_min_value=jnp.full((2,), 100, dtype=jnp.int32))
    controller = initialize_project_controller_v2(states, 0)
    controller = ensure_m26_projects_v2(states, controller, hold, 0)
    controller = materialize_m26_unit_tasks_v2(states, controller, hold, 0)
    np.testing.assert_array_equal(controller.unit_tasks.status[:, 0], TaskStatusV1.EMPTY)


def test_sell_fraction_changes_market_quantity() -> None:
    base = _states()
    states = base._replace(
        shed=base.shed.at[:, 0, 2].set(10),
        money=jnp.full((2, 2), 10_000, dtype=jnp.int32),
    )
    genome = _zero_targets(default_m26_crop_genome_v2(2))._replace(
        hand_target=jnp.zeros((2, 6), dtype=jnp.int8),
        sell_fraction=jnp.full((2, 9), 0.5, dtype=jnp.float32),
    )
    controller = initialize_project_controller_v2(states, 0)
    market = materialize_m26_market_tasks_v2(states, controller, genome, 0).market_tasks
    sell = (np.asarray(market.task_type) == TaskTypeV1.SELL_INVENTORY) & (
        np.asarray(market.item_id) == 2
    )
    assert np.all(np.asarray(market.quantity)[sell] == 5)

    full = genome._replace(sell_fraction=jnp.ones((2, 9), dtype=jnp.float32))
    market = materialize_m26_market_tasks_v2(states, controller, full, 0).market_tasks
    sell = (np.asarray(market.task_type) == TaskTypeV1.SELL_INVENTORY) & (
        np.asarray(market.item_id) == 2
    )
    assert np.all(np.asarray(market.quantity)[sell] == 10)


def test_weed_recovery_policy_changes_dig_task() -> None:
    base = _states()
    states = base._replace(tile_kind=base.tile_kind.at[:, 0, 0, 0].set(TileKind.WEED))
    genome = _zero_targets(default_m26_crop_genome_v2(2))._replace(
        hand_target=jnp.zeros((2, 6), dtype=jnp.int8)
    )
    controller = initialize_project_controller_v2(states, 0)._replace(
        tile_project_id=jnp.full((2, 10, 10), -1, dtype=jnp.int16).at[:, 0, 0].set(2)
    )
    controller = ensure_m26_projects_v2(states, controller, genome, 0)
    controller = materialize_m26_unit_tasks_v2(states, controller, genome, 0)
    np.testing.assert_array_equal(
        controller.unit_tasks.task_type[:, 0], TaskTypeV1.CLEAR_OR_REMOVE_TILE
    )

    abandon = genome._replace(
        weed_recovery_policy=jnp.zeros((2, 5), dtype=jnp.int8)
    )
    controller = initialize_project_controller_v2(states, 0)._replace(
        tile_project_id=jnp.full((2, 10, 10), -1, dtype=jnp.int16).at[:, 0, 0].set(2)
    )
    controller = ensure_m26_projects_v2(states, controller, abandon, 0)
    controller = materialize_m26_unit_tasks_v2(states, controller, abandon, 0)
    np.testing.assert_array_equal(controller.unit_tasks.status[:, 0], TaskStatusV1.EMPTY)


def test_fertilizer_policy_activates_buy_and_apply_paths() -> None:
    base = _states()
    states = base._replace(
        tile_kind=base.tile_kind.at[:, 0, 4, 4].set(TileKind.PLANT),
        tile_crop=base.tile_crop.at[:, 0, 4, 4].set(2),
        tile_flags=base.tile_flags.at[:, 0, 4, 4].set(jnp.uint8(FLAG_WATERED)),
        money=jnp.full((2, 2), 10_000, dtype=jnp.int32),
    )
    genome = _zero_targets(default_m26_crop_genome_v2(2))._replace(
        crop_target=jnp.zeros((2, 6, 5), dtype=jnp.int16).at[:, 0, 2].set(1),
        hand_target=jnp.zeros((2, 6), dtype=jnp.int8),
        fertilizer_policy=jnp.zeros((2, 5), dtype=jnp.int8).at[:, 2].set(1),
    )
    controller = initialize_project_controller_v2(states, 0)
    market = materialize_m26_market_tasks_v2(states, controller, genome, 0).market_tasks
    assert np.all(np.any(np.asarray(market.task_type) == TaskTypeV1.BUY_PRODUCT, axis=-1))

    with_fertilizer = states._replace(shed=states.shed.at[:, 0, 8].set(1))
    controller = initialize_project_controller_v2(with_fertilizer, 0)
    controller = ensure_m26_projects_v2(with_fertilizer, controller, genome, 0)
    controller = materialize_m26_unit_tasks_v2(with_fertilizer, controller, genome, 0)
    np.testing.assert_array_equal(
        controller.unit_tasks.task_type[:, 0], TaskTypeV1.APPLY_FERTILIZER
    )

    too_late = with_fertilizer._replace(
        step=jnp.full((2,), 23, dtype=jnp.int16),
        unit_pos=with_fertilizer.unit_pos.at[:, 0, 0].set(
            jnp.asarray((0, 0), dtype=jnp.int8)
        ),
    )
    controller = initialize_project_controller_v2(too_late, 0)
    controller = ensure_m26_projects_v2(too_late, controller, genome, 0)
    controller = materialize_m26_unit_tasks_v2(too_late, controller, genome, 0)
    np.testing.assert_array_equal(
        controller.unit_tasks.status[:, 0], TaskStatusV1.EMPTY
    )


def test_m26_does_not_hire_workers_on_the_last_turn_of_a_day() -> None:
    states = _states()._replace(
        step=jnp.full((2,), 23, dtype=jnp.int16),
        money=jnp.full((2, 2), 100_000, dtype=jnp.int32),
    )
    genome = _zero_targets(default_m26_crop_genome_v2(2))._replace(
        hand_target=jnp.full((2, 6), 10, dtype=jnp.int8)
    )
    controller = initialize_project_controller_v2(states, 0)
    market = materialize_m26_market_tasks_v2(states, controller, genome, 0).market_tasks
    assert not np.any(np.asarray(market.task_type) == TaskTypeV1.HIRE_WORKER)


def test_stale_fertilizer_task_is_cleared_when_target_is_no_longer_a_plant() -> None:
    states = _states()._replace(
        tile_kind=_states().tile_kind.at[:, 0, 2, 3].set(TileKind.WEED)
    )
    controller = initialize_project_controller_v2(states, 0)
    tasks = controller.unit_tasks._replace(
        task_type=controller.unit_tasks.task_type.at[:, 0].set(
            TaskTypeV1.APPLY_FERTILIZER
        ),
        target_x=controller.unit_tasks.target_x.at[:, 0].set(3),
        target_y=controller.unit_tasks.target_y.at[:, 0].set(2),
        target_id=controller.unit_tasks.target_id.at[:, 0].set(23),
        status=controller.unit_tasks.status.at[:, 0].set(TaskStatusV1.ACTIVE),
    )
    controller = controller._replace(unit_tasks=tasks)
    cleared = clear_invalidated_crop_tasks_v2(states, controller, 0)
    np.testing.assert_array_equal(
        cleared.unit_tasks.status[:, 0], TaskStatusV1.EMPTY
    )


def test_all_four_coarse_layouts_change_real_tile_selection_when_applicable() -> None:
    base_state = _states((1901,))
    base_genome = _zero_targets(default_m26_crop_genome_v2(1))._replace(
        crop_target=jnp.zeros((1, 6, 5), dtype=jnp.int16).at[:, 0, 2].set(1),
        hand_target=jnp.zeros((1, 6), dtype=jnp.int8),
    )

    def chosen_target(states, genome):
        controller = initialize_project_controller_v2(states, 0)
        controller = ensure_m26_projects_v2(states, controller, genome, 0)
        controller = materialize_m26_unit_tasks_v2(states, controller, genome, 0)
        return int(np.asarray(controller.unit_tasks.target_id)[0, 0])

    simple = base_state._replace(
        step=jnp.asarray((1,), dtype=jnp.int16),
        seeds=base_state.seeds.at[:, 0, 2].set(2),
    )
    center = base_genome._replace(
        crop_layout_policy=base_genome.crop_layout_policy.at[:, 2].set(0)
    )
    center_target = chosen_target(simple, center)

    quadrant = center._replace(crop_layout_policy=center.crop_layout_policy.at[:, 2].set(2))
    strip = center._replace(crop_layout_policy=center.crop_layout_policy.at[:, 2].set(3))
    assert chosen_target(simple, quadrant) != center_target
    assert chosen_target(simple, strip) != center_target

    clustered_state = simple._replace(
        tile_kind=simple.tile_kind.at[:, 0, 0, 0].set(TileKind.PLANT),
        tile_crop=simple.tile_crop.at[:, 0, 0, 0].set(2),
        tile_flags=simple.tile_flags.at[:, 0, 0, 0].set(jnp.uint8(FLAG_WATERED)),
    )
    clustered_genome = center._replace(
        crop_target=center.crop_target.at[:, 0, 2].set(2),
        crop_layout_policy=center.crop_layout_policy.at[:, 2].set(1),
    )
    assert chosen_target(clustered_state, clustered_genome) != chosen_target(
        clustered_state,
        center._replace(crop_target=center.crop_target.at[:, 0, 2].set(2)),
    )
