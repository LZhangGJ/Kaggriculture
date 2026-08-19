from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import FLAG_WATERED, MarketOp, TileKind
from kaggriculture_jax.state import reset
from strategic_v5.constants import TaskStatusV1, TaskTypeV1

from project_route_search_v2.lifecycle import initialize_project_controller_v2
from project_route_search_v2.m25_controller import (
    default_r2_tomato_m25_config_v2,
    m25_phase_v2,
    m25_policy_step_v2,
    materialize_m25_unit_tasks_v2,
)


def _states(seeds=(901, 902)):
    return jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))


def test_candidate_101_staged_targets_are_frozen() -> None:
    config = default_r2_tomato_m25_config_v2(2)
    np.testing.assert_array_equal(config.primary_target, [[12, 18, 28], [12, 18, 28]])
    np.testing.assert_array_equal(config.support_target, [[0, 8, 6], [0, 8, 6]])
    np.testing.assert_array_equal(config.hand_target, [[9, 11, 7], [9, 11, 7]])
    np.testing.assert_array_equal(config.land_target, [[1, 2, 2], [1, 2, 2]])
    np.testing.assert_array_equal(config.land_start_step[:, 1], 6 * 24)

    states = _states()
    np.testing.assert_array_equal(m25_phase_v2(states, config), 0)
    states = states._replace(step=jnp.full((2,), 5 * 24, dtype=jnp.int16))
    np.testing.assert_array_equal(m25_phase_v2(states, config), 1)
    states = states._replace(step=jnp.full((2,), 14 * 24, dtype=jnp.int16))
    np.testing.assert_array_equal(m25_phase_v2(states, config), 2)


def test_opening_market_plan_buys_tomato_seed_then_nine_hands() -> None:
    states = _states()
    config = default_r2_tomato_m25_config_v2(2)
    controller = initialize_project_controller_v2(states, 0)
    action, _ = m25_policy_step_v2(states, controller, config, 0)
    np.testing.assert_array_equal(action.market_op[:, 0], MarketOp.BUY_SEED)
    np.testing.assert_array_equal(action.market_item[:, 0], 2)
    np.testing.assert_array_equal(action.market_amount[:, 0], 12)
    np.testing.assert_array_equal(action.market_op[:, 1:10], MarketOp.HIRE)


def test_day_six_market_plan_contains_land_purchase() -> None:
    states = _states()._replace(
        step=jnp.full((2,), 6 * 24, dtype=jnp.int16),
        money=jnp.asarray([[10_000, 3_000], [10_000, 3_000]], dtype=jnp.int32),
    )
    config = default_r2_tomato_m25_config_v2(2)
    controller = initialize_project_controller_v2(states, 0)
    action, _ = m25_policy_step_v2(states, controller, config, 0)
    assert np.all(np.any(np.asarray(action.market_op) == MarketOp.BUY_LAND, axis=-1))


def test_land_target_controls_buy_land_and_start_step_only_gates_it() -> None:
    states = _states()._replace(
        step=jnp.full((2,), 6 * 24, dtype=jnp.int16),
        money=jnp.asarray([[10_000, 3_000], [10_000, 3_000]], dtype=jnp.int32),
    )
    baseline = default_r2_tomato_m25_config_v2(2)
    controller = initialize_project_controller_v2(states, 0)
    action, _ = m25_policy_step_v2(states, controller, baseline, 0)
    assert np.all(np.any(np.asarray(action.market_op) == MarketOp.BUY_LAND, axis=-1))

    no_expansion = baseline._replace(
        land_target=baseline.land_target.at[:, 1].set(jnp.int8(1))
    )
    action, _ = m25_policy_step_v2(states, controller, no_expansion, 0)
    assert not np.any(np.asarray(action.market_op) == MarketOp.BUY_LAND)

    delayed = baseline._replace(
        land_start_step=baseline.land_start_step.at[:, 1].set(jnp.int16(7 * 24))
    )
    action, _ = m25_policy_step_v2(states, controller, delayed, 0)
    assert not np.any(np.asarray(action.market_op) == MarketOp.BUY_LAND)


def test_four_free_units_receive_distinct_plant_targets() -> None:
    states = _states()._replace(
        step=jnp.full((2,), 1, dtype=jnp.int16),
        unit_active=_states().unit_active.at[:, 0, :4].set(True),
        seeds=_states().seeds.at[:, 0, 2].set(12),
    )
    config = default_r2_tomato_m25_config_v2(2)
    controller = initialize_project_controller_v2(states, 0)
    controller = materialize_m25_unit_tasks_v2(states, controller, config, 0)
    active = np.asarray(controller.unit_tasks.status[:, :4]) == TaskStatusV1.ACTIVE
    np.testing.assert_array_equal(active, True)
    np.testing.assert_array_equal(
        controller.unit_tasks.task_type[:, :4], TaskTypeV1.CROP_PRODUCTION
    )
    targets = np.asarray(controller.unit_tasks.target_id[:, :4])
    assert all(len(set(row.tolist())) == 4 for row in targets)


def test_unreachable_last_turn_water_is_not_scheduled() -> None:
    states = _states()._replace(
        step=jnp.full((2,), 23, dtype=jnp.int16),
        unit_pos=_states().unit_pos.at[:, 0, 0].set(jnp.asarray([4, 4], dtype=jnp.int8)),
        tile_kind=_states().tile_kind.at[:, 0, 0, 0].set(TileKind.PLANT),
        tile_crop=_states().tile_crop.at[:, 0, 0, 0].set(2),
        tile_flags=_states().tile_flags.at[:, 0, 0, 0].set(jnp.uint8(0 & FLAG_WATERED)),
    )
    config = default_r2_tomato_m25_config_v2(2)
    controller = initialize_project_controller_v2(states, 0)
    controller = materialize_m25_unit_tasks_v2(states, controller, config, 0)
    np.testing.assert_array_equal(controller.unit_tasks.status[:, 0], TaskStatusV1.EMPTY)
