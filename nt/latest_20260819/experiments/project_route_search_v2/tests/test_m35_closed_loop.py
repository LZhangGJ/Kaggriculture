from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import MarketOp, NUM_PRODUCTS, TileKind
from kaggriculture_jax.state import load_tables, reset
from strategic_v5.constants import TaskStatusV1, TaskTypeV1

from project_route_search_v2.lifecycle import initialize_project_controller_v2
from project_route_search_v2.m35_candidates import m35_branch_coverage_panel_v2
from project_route_search_v2.m35_controller import (
    ensure_m35_projects_v2,
    m35_unit_masks_v2,
    materialize_m35_market_tasks_v2,
    materialize_m35_unit_tasks_v2,
)
from project_route_search_v2.m35_genome import (
    default_m35_farm_genome_v2,
    validate_m35_farm_genome_v2,
)


def _lane(tree, index: int):
    return jax.tree.map(lambda value: value[index : index + 1], tree)


def _state(seed: int = 3501):
    return jax.vmap(reset)(jnp.asarray((seed,), dtype=jnp.int32))


def _four_animals(base, *, step: int, wheat: int = 10, product: int = 0):
    positions = ((0, 0, 0), (0, 1, 0), (1, 0, 1), (2, 0, 2))
    state = base._replace(
        step=jnp.asarray((step,), dtype=jnp.int16),
        money=base.money.at[0, 0].set(100_000),
        shed=base.shed.at[0, 0, 0].set(wheat),
    )
    for y, x, species in positions:
        kind = TileKind.COOP if species == 0 else TileKind.PASTURE
        state = state._replace(
            tile_kind=state.tile_kind.at[0, 0, y, x].set(kind),
            tile_animal=state.tile_animal.at[0, 0, y, x].set(species),
            tile_yield=state.tile_yield.at[0, 0, y, x].set(product),
        )
    return state


def test_default_and_branch_m35_genomes_are_valid() -> None:
    assert validate_m35_farm_genome_v2(default_m35_farm_genome_v2(3)) == []
    panel, names = m35_branch_coverage_panel_v2()
    assert len(names) == 6
    assert validate_m35_farm_genome_v2(panel) == []


def test_empty_animal_structure_reservation_survives_crop_refresh() -> None:
    states = _state()
    states = states._replace(
        tile_kind=states.tile_kind.at[0, 0, 4, 4].set(TileKind.COOP)
    )
    genome = default_m35_farm_genome_v2(1)
    controller = initialize_project_controller_v2(states, 0)
    controller = controller._replace(
        tile_project_id=controller.tile_project_id.at[0, 4, 4].set(5)
    )
    refreshed = ensure_m35_projects_v2(states, controller, genome, 0)
    assert int(refreshed.tile_project_id[0, 4, 4]) == 5


def test_terminal_feed_emergency_assigns_every_active_unit_to_animals() -> None:
    base = _state()
    states = _four_animals(base, step=25 * 24)
    states = states._replace(
        tile_neglect=states.tile_neglect.at[0, 0, 0, 0].set(1),
        unit_active=states.unit_active.at[0, 0, :3].set(True),
    )
    genome = default_m35_farm_genome_v2(1)
    controller = initialize_project_controller_v2(states, 0)
    crop_mask, animal_mask = m35_unit_masks_v2(states, controller, genome, 0)
    assert not np.any(np.asarray(crop_mask)[0, :3])
    assert np.all(np.asarray(animal_mask)[0, :3])


def test_grow_only_reserves_full_terminal_feed_and_never_buys_wheat() -> None:
    panel, _ = m35_branch_coverage_panel_v2()
    genome = _lane(panel, 1)
    states = _four_animals(_state(), step=25 * 24, wheat=10)
    controller = initialize_project_controller_v2(states, 0)
    market = materialize_m35_market_tasks_v2(
        states, controller, genome, 0, load_tables()
    ).market_tasks
    active = np.asarray(market.status)[0] == TaskStatusV1.ACTIVE
    task_type = np.asarray(market.task_type)[0]
    item = np.asarray(market.item_id)[0]
    quantity = np.asarray(market.quantity)[0]
    wheat_buy = active & (task_type == TaskTypeV1.BUY_PRODUCT) & (item == 0)
    wheat_sell = active & np.isin(
        task_type,
        (TaskTypeV1.SELL_INVENTORY, TaskTypeV1.TERMINAL_LIQUIDATION),
    ) & (item == 0)
    assert not np.any(wheat_buy)
    assert int(np.sum(quantity[wheat_sell])) <= 2


def test_final_day_animal_product_permits_terminal_hires() -> None:
    states = _four_animals(_state(), step=29 * 24, wheat=0, product=1)
    genome = default_m35_farm_genome_v2(1)
    controller = initialize_project_controller_v2(states, 0)
    market = materialize_m35_market_tasks_v2(
        states, controller, genome, 0, load_tables()
    ).market_tasks
    active = np.asarray(market.status)[0] == TaskStatusV1.ACTIVE
    assert np.any(
        active & (np.asarray(market.task_type)[0] == TaskTypeV1.HIRE_WORKER)
    )


def test_reserved_animal_fertilizer_is_not_sold_before_closing() -> None:
    panel, _ = m35_branch_coverage_panel_v2()
    genome = _lane(panel, 3)
    base = _state()
    states = base._replace(
        step=jnp.asarray((20 * 24,), dtype=jnp.int16),
        shed=base.shed.at[0, 0, 8].set(5),
    )
    controller = initialize_project_controller_v2(states, 0)
    market = materialize_m35_market_tasks_v2(
        states, controller, genome, 0, load_tables()
    ).market_tasks
    slots = np.asarray(market.status)[0] == TaskStatusV1.ACTIVE
    sold_fertilizer = slots & (
        np.asarray(market.task_type)[0] == TaskTypeV1.SELL_INVENTORY
    ) & (np.asarray(market.item_id)[0] == 8)
    assert not np.any(sold_fertilizer)


def test_no_feed_task_is_created_after_last_real_day_end() -> None:
    states = _four_animals(_state(), step=696, wheat=10, product=0)
    states = states._replace(
        tile_neglect=states.tile_neglect.at[0, 0, 0, 0].set(1),
        unit_active=states.unit_active.at[0, 0, :4].set(True),
    )
    genome = default_m35_farm_genome_v2(1)
    controller = ensure_m35_projects_v2(
        states, initialize_project_controller_v2(states, 0), genome, 0
    )
    controller = materialize_m35_unit_tasks_v2(states, controller, genome, 0)
    active = np.asarray(controller.unit_tasks.status)[0] == TaskStatusV1.ACTIVE
    feed = np.asarray(controller.unit_tasks.task_type)[0] == TaskTypeV1.ANIMAL_FEED
    assert not np.any(active & feed)


def test_liquidation_cancels_obsolete_crop_fertilizer_task() -> None:
    base = _state()
    states = base._replace(
        step=jnp.asarray((25 * 24,), dtype=jnp.int16),
        unit_active=base.unit_active.at[0, 0, :3].set(True),
    )
    genome = default_m35_farm_genome_v2(1)
    controller = ensure_m35_projects_v2(
        states, initialize_project_controller_v2(states, 0), genome, 0
    )
    tasks = controller.unit_tasks._replace(
        task_type=controller.unit_tasks.task_type.at[0, 1].set(
            TaskTypeV1.APPLY_FERTILIZER
        ),
        status=controller.unit_tasks.status.at[0, 1].set(TaskStatusV1.ACTIVE),
        item_id=controller.unit_tasks.item_id.at[0, 1].set(8),
        target_x=controller.unit_tasks.target_x.at[0, 1].set(0),
        target_y=controller.unit_tasks.target_y.at[0, 1].set(0),
    )
    controller = controller._replace(unit_tasks=tasks)
    controller = materialize_m35_unit_tasks_v2(states, controller, genome, 0)
    active = np.asarray(controller.unit_tasks.status)[0] == TaskStatusV1.ACTIVE
    fertilizer = (
        np.asarray(controller.unit_tasks.task_type)[0]
        == TaskTypeV1.APPLY_FERTILIZER
    )
    assert not np.any(active & fertilizer)
