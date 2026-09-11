from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import TileKind
from kaggriculture_jax.state import load_event_bank, load_tables, reset
from kaggriculture_jax.types import Events
from strategic_v5.constants import TaskStatusV1, TaskTypeV1

from project_route_search_v2.constants import CropObligationTypeV2
from project_route_search_v2.crop_executor import clear_invalidated_crop_tasks_v2
from project_route_search_v2.crop_project import (
    default_crop_project_config_v2,
    ensure_crop_project_v2,
    refresh_crop_project_v2,
)
from project_route_search_v2.crop_rollout import (
    initialize_crop_rollout_carry_v2,
    make_m2_crop_rollout_v2,
    summarize_m2_crop_rollout_v2,
)
from project_route_search_v2.lifecycle import initialize_project_controller_v2
from project_route_search_v2.obligations import (
    SEED_OBLIGATION_SLOT,
    UNIT_OBLIGATION_SLOT,
    build_crop_obligations_v2,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _controller_and_config(states):
    config = default_crop_project_config_v2(states.step.shape[0])
    controller = initialize_project_controller_v2(states, player=0)
    controller = ensure_crop_project_v2(states, controller, config)
    controller = refresh_crop_project_v2(states, controller, config, player=0)
    return controller, config


def test_seed_buy_is_generated_but_new_seed_is_not_used_in_same_step() -> None:
    states = jax.vmap(reset)(jnp.asarray([101, 102], dtype=jnp.int32))
    controller, config = _controller_and_config(states)
    obligations = build_crop_obligations_v2(states, controller, config, player=0)
    np.testing.assert_array_equal(obligations.present[:, SEED_OBLIGATION_SLOT], True)
    np.testing.assert_array_equal(
        obligations.obligation_type[:, SEED_OBLIGATION_SLOT],
        CropObligationTypeV2.BUY_SEED,
    )
    np.testing.assert_array_equal(obligations.present[:, UNIT_OBLIGATION_SLOT], False)


def test_plant_admission_requires_time_for_plant_and_same_day_water() -> None:
    initial = jax.vmap(reset)(jnp.asarray([111, 112], dtype=jnp.int32))
    initial = initial._replace(seeds=initial.seeds.at[:, 0, 0].set(1))

    step22 = initial._replace(step=jnp.full((2,), 22, dtype=jnp.int16))
    controller22, config22 = _controller_and_config(step22)
    obligations22 = build_crop_obligations_v2(step22, controller22, config22, 0)
    np.testing.assert_array_equal(obligations22.present[:, UNIT_OBLIGATION_SLOT], True)
    np.testing.assert_array_equal(
        obligations22.obligation_type[:, UNIT_OBLIGATION_SLOT],
        CropObligationTypeV2.PLANT_AND_WATER,
    )

    step23 = initial._replace(step=jnp.full((2,), 23, dtype=jnp.int16))
    controller23, config23 = _controller_and_config(step23)
    obligations23 = build_crop_obligations_v2(step23, controller23, config23, 0)
    np.testing.assert_array_equal(obligations23.present[:, UNIT_OBLIGATION_SLOT], False)


def test_stale_tomato_water_task_is_cleared_after_target_becomes_weed() -> None:
    states = jax.vmap(reset)(jnp.asarray([121], dtype=jnp.int32))
    states = states._replace(
        tile_kind=states.tile_kind.at[0, 0, 4, 4].set(TileKind.WEED),
        tile_crop=states.tile_crop.at[0, 0, 4, 4].set(-1),
    )
    controller = initialize_project_controller_v2(states, player=0)
    tasks = controller.unit_tasks._replace(
        task_type=controller.unit_tasks.task_type.at[0, 0].set(
            TaskTypeV1.WATER_CROP
        ),
        target_id=controller.unit_tasks.target_id.at[0, 0].set(44),
        target_x=controller.unit_tasks.target_x.at[0, 0].set(4),
        target_y=controller.unit_tasks.target_y.at[0, 0].set(4),
        item_id=controller.unit_tasks.item_id.at[0, 0].set(2),
        status=controller.unit_tasks.status.at[0, 0].set(TaskStatusV1.ACTIVE),
    )
    cleared = clear_invalidated_crop_tasks_v2(
        states, controller._replace(unit_tasks=tasks), player=0
    )
    assert int(cleared.unit_tasks.status[0, 0]) == TaskStatusV1.EMPTY
    assert int(cleared.unit_tasks.target_id[0, 0]) == -1


def test_two_full_seasons_close_crop_cash_cycle_without_hard_errors() -> None:
    bank_path = PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz"
    event_seeds, event_bank = load_event_bank(bank_path)
    selected = np.asarray([84101, 84102], dtype=np.int64)
    indices = np.asarray(
        [int(np.flatnonzero(event_seeds == seed)[0]) for seed in selected],
        dtype=np.int32,
    )
    events = Events(
        weed_spawn=event_bank.weed_spawn[indices],
        shop_choice=event_bank.shop_choice[indices],
    )
    seeds = jnp.asarray(selected, dtype=jnp.int32)
    config = default_crop_project_config_v2(2)
    carry = initialize_crop_rollout_carry_v2(seeds, config, player=0)
    rollout = jax.jit(make_m2_crop_rollout_v2())
    final_carry = rollout(carry, events, load_tables(), config)
    summary = summarize_m2_crop_rollout_v2(final_carry, config, player=0)
    np.testing.assert_array_equal(summary.done, True)
    np.testing.assert_array_equal(summary.terminal_sellable_shed_value, 0)
    np.testing.assert_array_equal(summary.terminal_unit_inventory_value, 0)
    np.testing.assert_array_equal(summary.avoidable_liquidation_loss, 0)
    np.testing.assert_array_equal(summary.plant_without_same_day_water, 0)
    np.testing.assert_array_equal(summary.unexplained_failure_count, 0)
    assert np.all(np.asarray(summary.final_bank) > 3000)
    assert np.all(np.asarray(summary.plant_success_count) > 0)
    assert np.all(np.asarray(summary.harvest_success_count) > 0)
    assert np.all(np.asarray(summary.deposit_success_count) > 0)
    assert np.all(np.asarray(summary.sold_product_units) > 0)
