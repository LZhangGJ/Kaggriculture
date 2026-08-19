"""M3.5 deterministic joint crop/animal controller.

M3.5 deliberately stops short of the M4 top-K assignment solver.  It provides
one project view, deterministic worker bands and one market/cash ledger so the
already accepted M2.6 and M3 task executors can form a real farm loop without
double spending shared resources.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_COST,
    BOARD_SIZE,
    CROP_SEED_COST,
    EPISODE_STEPS,
    FLAG_FED,
    HIRE_COST,
    LAND_PRICES,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    SHED_CAPACITY,
    TURNS_PER_DAY,
)
from kaggriculture_jax.state import load_tables
from kaggriculture_jax.types import State, StaticTables
from strategic_v5.constants import TaskStatusV1, TaskTypeV1
from strategic_v5.e4_executor import (
    E4PlayerActionV1,
    FullCoreEffectDiagnosticsV1,
    compile_full_core_player_action_v1,
    update_full_core_controller_from_effects_v1,
)
from strategic_v5.e5_econ import exact_market_quote_v1

from .crop_executor import clear_invalidated_crop_tasks_v2
from .lifecycle import (
    empty_market_tasks_v2,
    empty_unit_tasks_v2,
    reconcile_project_controller_v2,
)
from .m26_controller import (
    ensure_m26_projects_v2,
    m26_phase_targets_v2,
    materialize_m26_market_tasks_v2,
    materialize_m26_unit_tasks_v2,
)
from .m3_constants import M3AnimalFertilizerPolicyV2
from .m3_controller import (
    clear_invalidated_m3_tasks_v2,
    ensure_m3_projects_v2,
    m3_phase_targets_v2,
    materialize_m3_market_tasks_v2,
    materialize_m3_unit_tasks_v2,
)
from .m35_schema import M35FarmGenomeV2
from .schema import ProjectControllerStateV2


_DEFAULT_TABLES = load_tables()
_SEED_COST = jnp.asarray(CROP_SEED_COST, dtype=jnp.int32)
_ANIMAL_COST = jnp.asarray(ANIMAL_COST, dtype=jnp.int32)
_LAND_COST = jnp.asarray(LAND_PRICES, dtype=jnp.int32)
_HIRE_COST = jnp.asarray(HIRE_COST, dtype=jnp.int32)


def _closing_animal_obligations_v2(
    states: State, genome: M35FarmGenomeV2, player: int
) -> tuple[jax.Array, jax.Array]:
    animal = states.tile_animal[:, player]
    fed = (states.tile_flags[:, player] & jnp.uint8(FLAG_FED)) != 0
    hard_feed_due = jnp.any(
        (animal >= 0) & (states.tile_neglect[:, player] >= 1) & (~fed),
        axis=(1, 2),
    )
    hard_feed_due = hard_feed_due & (
        states.step <= jnp.int16(EPISODE_STEPS - TURNS_PER_DAY - 1)
    )
    harvestable_count = jnp.sum(
        (animal >= 0) & (states.tile_yield[:, player] > 0),
        axis=(1, 2),
        dtype=jnp.int16,
    )
    closing = states.step >= genome.animal.liquidation_start_step
    return closing & hard_feed_due, jnp.where(
        closing, harvestable_count, 0
    )


def ensure_m35_projects_v2(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M35FarmGenomeV2,
    player: int,
) -> ProjectControllerStateV2:
    """Merge crop slots 0..4, animal slots 5..7 and shared slots 8..9."""

    previous_animal_tile = jnp.where(
        controller.tile_project_id >= NUM_CROPS,
        controller.tile_project_id,
        -1,
    ).astype(jnp.int16)
    crop_controller = ensure_m26_projects_v2(states, controller, genome.crop, player)
    crop_tile_project = crop_controller.tile_project_id
    # M2.6 intentionally owns only crop ids and clears non-crop tiles.  Restore
    # the previous animal reservation before refreshing M3; otherwise every
    # newly built empty coop/pasture appears unreserved on the next step and is
    # built again indefinitely.
    animal_input_tile = jnp.where(
        previous_animal_tile >= NUM_CROPS,
        previous_animal_tile,
        crop_tile_project,
    ).astype(jnp.int16)
    combined = ensure_m3_projects_v2(
        states,
        crop_controller._replace(tile_project_id=animal_input_tile),
        genome.animal,
        player,
    )
    animal_tile_project = combined.tile_project_id
    tile_project = jnp.where(
        animal_tile_project >= NUM_CROPS,
        animal_tile_project,
        crop_tile_project,
    ).astype(jnp.int16)
    return combined._replace(tile_project_id=tile_project)


def m35_unit_masks_v2(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M35FarmGenomeV2,
    player: int,
) -> tuple[jax.Array, jax.Array]:
    """Assign one stable crop/animal band; M4 will replace this with matching."""

    _, crop_target, _, _ = m26_phase_targets_v2(states, genome.crop)
    _, animal_target, _, _ = m3_phase_targets_v2(states, genome.animal)
    crop_physical = jnp.any(states.tile_crop[:, player] >= 0, axis=(1, 2))
    animal_physical = jnp.any(states.tile_animal[:, player] >= 0, axis=(1, 2))
    has_crop = jnp.any(crop_target > 0, axis=-1) | crop_physical
    has_animal = jnp.any(animal_target > 0, axis=-1) | animal_physical

    active = states.unit_active[:, player]
    active_count = jnp.sum(active, axis=-1, dtype=jnp.int16)
    rank = jnp.cumsum(active.astype(jnp.int16), axis=-1) - 1
    raw_crop = jnp.rint(
        active_count.astype(jnp.float32) * genome.crop_unit_share
    ).astype(jnp.int16)
    both = has_crop & has_animal
    closing_feed_emergency, harvestable_count = _closing_animal_obligations_v2(
        states, genome, player
    )
    closing = states.step >= genome.animal.liquidation_start_step
    base_crop_count = jnp.where(
        both & (active_count >= 2),
        jnp.clip(raw_crop, 1, active_count - 1),
        jnp.where(has_crop, active_count, 0),
    ).astype(jnp.int16)
    closing_animal_emergency = closing_feed_emergency | (harvestable_count > 0)
    crop_count = jnp.where(
        closing_animal_emergency,
        0,
        jnp.where(closing & crop_physical, active_count, base_crop_count),
    ).astype(jnp.int16)
    crop_mask = active & has_crop[:, None] & (rank < crop_count[:, None])
    animal_mask = active & has_animal[:, None] & (~crop_mask)
    return crop_mask, animal_mask


def materialize_m35_unit_tasks_v2(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M35FarmGenomeV2,
    player: int,
) -> ProjectControllerStateV2:
    closing = states.step >= genome.animal.liquidation_start_step
    obsolete_fertilizer = (
        closing[:, None]
        & (controller.unit_tasks.status == TaskStatusV1.ACTIVE)
        & (controller.unit_tasks.task_type == TaskTypeV1.APPLY_FERTILIZER)
    )
    empty = empty_unit_tasks_v2(states.step.shape[0])
    controller = controller._replace(
        unit_tasks=jax.tree.map(
            lambda old, blank: jnp.where(obsolete_fertilizer, blank, old),
            controller.unit_tasks,
            empty,
        )
    )
    crop_mask, animal_mask = m35_unit_masks_v2(states, controller, genome, player)
    # During terminal feed emergencies, an already sticky crop harvest/deposit
    # chain may otherwise keep a worker until after the animal's day-end hard
    # deadline.  Only crop-specific tasks are preempted, and only in closing;
    # same-day planting/watering in the normal operating window is untouched.
    hard_feed, harvestable_count = _closing_animal_obligations_v2(
        states, genome, player
    )
    emergency = hard_feed | (harvestable_count > 0)
    crop_task = (
        (controller.unit_tasks.task_type == TaskTypeV1.CROP_PRODUCTION)
        | (controller.unit_tasks.task_type == TaskTypeV1.WATER_CROP)
        | (controller.unit_tasks.task_type == TaskTypeV1.CLEAR_OR_REMOVE_TILE)
        | (controller.unit_tasks.task_type == TaskTypeV1.APPLY_FERTILIZER)
    )
    preempt = emergency[:, None] & animal_mask & crop_task
    controller = controller._replace(
        unit_tasks=jax.tree.map(
            lambda old, blank: jnp.where(preempt, blank, old),
            controller.unit_tasks,
            empty,
        )
    )
    # Animal hard obligations own their band first; crop same-day water owns the
    # crop band.  Persistent tasks remain sticky inside both executors.
    controller = materialize_m3_unit_tasks_v2(
        states, controller, genome.animal, player, animal_mask
    )
    controller = materialize_m26_unit_tasks_v2(
        states, controller, genome.crop, player, crop_mask
    )
    return controller


def _market_plan(
    planner,
    states: State,
    controller: ProjectControllerStateV2,
    subgenome,
    player: int,
    tables: StaticTables,
    **planner_kwargs,
):
    empty = controller._replace(market_tasks=empty_market_tasks_v2(states.step.shape[0]))
    return planner(
        states, empty, subgenome, player, tables, **planner_kwargs
    ).market_tasks


def materialize_m35_market_tasks_v2(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M35FarmGenomeV2,
    player: int,
    tables: StaticTables | None = None,
    market_template: jax.Array | None = None,
    *,
    reserve_future_hires: bool = True,
    rolling_replan: bool = False,
) -> ProjectControllerStateV2:
    """Merge both sub-plans through one sequential cash/capacity ledger.

    ``market_template`` is the M3.6 daily template (0/1/2).  It must affect
    ordering before the ten-slot truncation: sorting an already truncated task
    list cannot recover an animal/seed/land candidate that was discarded by a
    legacy fixed priority.
    """

    tables = _DEFAULT_TABLES if tables is None else tables
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    animal = _market_plan(
        materialize_m3_market_tasks_v2,
        states,
        controller,
        genome.animal,
        player,
        tables,
        # Open-loop callers retain the conservative multi-day workforce
        # reserve.  A complete RouteCalendar caller may disable only that
        # duplicate reserve because it replans after every realized day.  The
        # animal planner still protects current-day hires, exact feed purchase,
        # pre-revenue feed runway and one replacement feed cycle.
        reserve_future_hires=reserve_future_hires,
        rolling_replan=rolling_replan,
        market_template=market_template,
    )
    crop = _market_plan(
        materialize_m26_market_tasks_v2,
        states,
        controller,
        genome.crop,
        player,
        tables,
        allow_closing_hires=True,
    )

    # The crop planner owns the synchronized workforce target in every phase,
    # including liquidation.  Animal hires are intentionally removed; keeping
    # both plans would create two independent copies of the same HIRE sequence.
    animal_hire = animal.task_type == TaskTypeV1.HIRE_WORKER
    closing = states.step >= genome.animal.liquidation_start_step
    animal_active = (animal.status == TaskStatusV1.ACTIVE) & (~animal_hire)
    crop_type = crop.task_type
    crop_active = (crop.status == TaskStatusV1.ACTIVE) & (
        (crop_type == TaskTypeV1.CROP_PRODUCTION)
        | (crop_type == TaskTypeV1.HIRE_WORKER)
        | (
            (crop_type == TaskTypeV1.BUY_PRODUCT)
            & (crop.item_id == 8)
            & (
                genome.animal.animal_fertilizer_policy
                != M3AnimalFertilizerPolicyV2.RESERVE_FOR_CROPS
            )[:, None]
        )
    )
    valid = jnp.concatenate((animal_active, crop_active), axis=-1)
    task_type = jnp.concatenate((animal.task_type, crop.task_type), axis=-1)
    item = jnp.concatenate((animal.item_id, crop.item_id), axis=-1)
    quantity = jnp.concatenate((animal.quantity, crop.quantity), axis=-1)
    candidate_count = valid.shape[1]
    ordinal = jnp.arange(candidate_count, dtype=jnp.int32)[None]

    is_sell = (task_type == TaskTypeV1.SELL_INVENTORY) | (
        task_type == TaskTypeV1.TERMINAL_LIQUIDATION
    )
    is_product_buy = task_type == TaskTypeV1.BUY_PRODUCT
    is_wheat_buy = is_product_buy & (item == 0)
    is_seed = task_type == TaskTypeV1.CROP_PRODUCTION
    is_wheat_seed = is_seed & (item == 0)
    is_hire = task_type == TaskTypeV1.HIRE_WORKER
    is_land = task_type == TaskTypeV1.BUY_LAND
    is_animal = task_type == TaskTypeV1.ANIMAL_PURCHASE
    legacy_priority = jnp.where(
        is_sell,
        0,
        jnp.where(
            is_wheat_buy,
            1,
            jnp.where(
                is_wheat_seed,
                2,
                jnp.where(
                    is_hire,
                    3,
                    jnp.where(
                        is_land,
                        4,
                        jnp.where(is_seed, 5, jnp.where(is_animal, 6, 7)),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    if market_template is None:
        priority = legacy_priority
    else:
        template = jnp.asarray(market_template, dtype=jnp.int8)[:, None]
        is_input = is_product_buy
        # 0 HIRE_ANIMAL_SEED_FEED: existing cash finances commitments first.
        hire_first = jnp.where(
            is_hire,
            0,
            jnp.where(
                is_animal,
                1,
                jnp.where(
                    is_seed,
                    2,
                    jnp.where(
                        is_input,
                        3,
                        jnp.where(is_land, 4, jnp.where(is_sell, 5, 6)),
                    ),
                ),
            ),
        )
        # 1 SELL_HIRE_ANIMAL_SEED_FEED: realize inventory before spending it.
        sell_first = jnp.where(
            is_sell,
            0,
            jnp.where(
                is_hire,
                1,
                jnp.where(
                    is_animal,
                    2,
                    jnp.where(
                        is_seed,
                        3,
                        jnp.where(is_input, 4, jnp.where(is_land, 5, 6)),
                    ),
                ),
            ),
        )
        # 2 CRITICAL_INPUT_HIRE_ANIMAL_SELL: survival inputs cannot be crowded
        # out by optional sales or expansion.
        input_first = jnp.where(
            is_input,
            0,
            jnp.where(
                is_hire,
                1,
                jnp.where(
                    is_animal,
                    2,
                    jnp.where(
                        is_seed,
                        3,
                        jnp.where(is_sell, 4, jnp.where(is_land, 5, 6)),
                    ),
                ),
            ),
        )
        priority = jnp.where(
            template == 1,
            sell_first,
            jnp.where(template == 2, input_first, hire_first),
        ).astype(jnp.int32)
        # A sub-planner may reserve sale proceeds for mandatory wheat, but that
        # reserve is lost if the merged ledger executes hires/animals/seeds
        # before the BUY_PRODUCT.  Whenever a critical input order is present,
        # realize safe sales first, then buy the input, and only then admit
        # optional spending.  The hour-1 retry recovers commitments that do not
        # fit beside those safety orders in the first ten-slot bundle.
        critical_input_present = jnp.any(valid & is_input, axis=-1)[:, None]
        critical_priority = jnp.where(
            is_sell,
            0,
            jnp.where(
                is_input,
                1,
                jnp.where(
                    is_hire,
                    2,
                    jnp.where(
                        is_animal,
                        3,
                        jnp.where(is_seed, 4, jnp.where(is_land, 5, 6)),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        priority = jnp.where(
            critical_input_present, critical_priority, priority
        ).astype(jnp.int32)
    key = jnp.where(valid, priority * 100 + ordinal, 100_000 + ordinal)
    order = jnp.argsort(key, axis=-1)
    ordered_valid = jnp.take_along_axis(valid, order, axis=-1)
    ordered_type = jnp.take_along_axis(task_type, order, axis=-1)
    ordered_item = jnp.take_along_axis(item, order, axis=-1)
    ordered_quantity = jnp.take_along_axis(quantity, order, axis=-1)
    # Quote the complete candidate matrix once.  Calling the official
    # sequential quote scan inside the 20-candidate selection loop replicated
    # the quote graph forty times and made first compilation impractical.
    safe_product_all = jnp.clip(item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    buy_quotes_all = exact_market_quote_v1(
        states,
        tables,
        safe_product_all.astype(jnp.int8),
        quantity.astype(jnp.int16),
        buy=True,
        player=player,
        enforce_resources=False,
    ).astype(jnp.int32)
    sell_quotes_all = exact_market_quote_v1(
        states,
        tables,
        safe_product_all.astype(jnp.int8),
        quantity.astype(jnp.int16),
        buy=False,
        player=player,
        enforce_resources=False,
    ).astype(jnp.int32)
    ordered_buy_quote = jnp.take_along_axis(buy_quotes_all, order, axis=-1)
    ordered_sell_quote = jnp.take_along_axis(sell_quotes_all, order, axis=-1)

    desired = empty_market_tasks_v2(batch_size)
    money = states.money[:, player].astype(jnp.int32)
    shed_used = jnp.sum(states.shed[:, player].astype(jnp.int32), axis=-1)
    accepted_count = jnp.zeros((batch_size,), dtype=jnp.int16)
    accepted_hires = jnp.zeros((batch_size,), dtype=jnp.int16)
    cash_floor = jnp.maximum(genome.crop.cash_floor, genome.animal.cash_floor)
    unlocked = states.unlocked_count[:, player].astype(jnp.int32)
    current_hires = states.hires_today[:, player].astype(jnp.int32)

    for index in range(candidate_count):
        present = ordered_valid[:, index]
        typ = ordered_type[:, index]
        itm = ordered_item[:, index]
        qty = ordered_quantity[:, index].astype(jnp.int32)
        sell = (typ == TaskTypeV1.SELL_INVENTORY) | (
            typ == TaskTypeV1.TERMINAL_LIQUIDATION
        )
        product_buy = typ == TaskTypeV1.BUY_PRODUCT
        seed_buy = typ == TaskTypeV1.CROP_PRODUCTION
        animal_buy = typ == TaskTypeV1.ANIMAL_PURCHASE
        land_buy = typ == TaskTypeV1.BUY_LAND
        hire = typ == TaskTypeV1.HIRE_WORKER

        product_quote = ordered_buy_quote[:, index]
        sell_quote = ordered_sell_quote[:, index]
        seed_cost = _SEED_COST[jnp.clip(itm.astype(jnp.int32), 0, NUM_CROPS - 1)] * qty
        animal_id = jnp.clip(
            itm.astype(jnp.int32) - NUM_PRODUCTS, 0, NUM_ANIMALS - 1
        )
        animal_cost = _ANIMAL_COST[animal_id] * qty
        land_cost = _LAND_COST[jnp.clip(unlocked - 1, 0, len(LAND_PRICES) - 1)]
        hire_index = jnp.clip(
            current_hires + accepted_hires, 0, len(HIRE_COST) - 1
        )
        hire_cost = _HIRE_COST[hire_index]
        cost = jnp.where(
            product_buy,
            product_quote,
            jnp.where(
                seed_buy,
                seed_cost,
                jnp.where(
                    animal_buy,
                    animal_cost,
                    jnp.where(land_buy, land_cost, jnp.where(hire, hire_cost, 0)),
                ),
            ),
        ).astype(jnp.int32)
        shed_delta = jnp.where(
            sell,
            -qty,
            jnp.where(product_buy | animal_buy, qty, 0),
        ).astype(jnp.int32)
        room_ok = (shed_used + shed_delta >= 0) & (
            shed_used + shed_delta <= SHED_CAPACITY
        )
        hard_spend = product_buy & (itm == 0) | hire
        affordable = jnp.where(
            sell,
            True,
            (money >= cost) & (hard_spend | (money - cost >= cash_floor)),
        )
        accept = (
            present
            & (accepted_count < MAX_MARKET_ORDERS)
            & room_ok
            & affordable
        )
        slot = jnp.clip(accepted_count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1)

        def set_field(field, value):
            old = field[batch, slot]
            return field.at[batch, slot].set(jnp.where(accept, value, old))

        desired = desired._replace(
            task_type=set_field(desired.task_type, typ.astype(jnp.int8)),
            item_id=set_field(desired.item_id, itm.astype(jnp.int8)),
            quantity=set_field(desired.quantity, qty.astype(jnp.int16)),
            start_step=set_field(desired.start_step, states.step.astype(jnp.int16)),
            deadline_step=set_field(
                desired.deadline_step,
                jnp.minimum(states.step + 1, EPISODE_STEPS - 2).astype(jnp.int16),
            ),
            status=set_field(
                desired.status,
                jnp.full_like(states.step, TaskStatusV1.ACTIVE, dtype=jnp.int8),
            ),
        )
        money = money + jnp.where(accept & sell, sell_quote, 0)
        money = money - jnp.where(accept & (~sell), cost, 0)
        shed_used = shed_used + jnp.where(accept, shed_delta, 0)
        accepted_hires = accepted_hires + (accept & hire).astype(jnp.int16)
        accepted_count = accepted_count + accept.astype(jnp.int16)

    old_active = controller.market_tasks.status == TaskStatusV1.ACTIVE
    market = jax.tree.map(
        lambda old, new: jnp.where(old_active, old, new),
        controller.market_tasks,
        desired,
    )
    return controller._replace(market_tasks=market)


def m35_policy_step_v2(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M35FarmGenomeV2,
    player: int,
    tables: StaticTables | None = None,
) -> tuple[E4PlayerActionV1, ProjectControllerStateV2]:
    controller, _ = reconcile_project_controller_v2(states, controller, player)
    controller = clear_invalidated_crop_tasks_v2(states, controller, player)
    controller = clear_invalidated_m3_tasks_v2(
        states, controller, genome.animal, player
    )
    controller = ensure_m35_projects_v2(states, controller, genome, player)
    controller = materialize_m35_unit_tasks_v2(states, controller, genome, player)
    controller = materialize_m35_market_tasks_v2(
        states, controller, genome, player, tables
    )
    return compile_full_core_player_action_v1(states, controller, player), controller


def update_m35_controller_from_effects_v2(
    states: State,
    next_states: State,
    controller: ProjectControllerStateV2,
    action: E4PlayerActionV1,
    genome: M35FarmGenomeV2,
    player: int,
) -> tuple[ProjectControllerStateV2, FullCoreEffectDiagnosticsV1]:
    controller, diagnostics = update_full_core_controller_from_effects_v1(
        states, next_states, controller, action, player
    )
    controller = ensure_m35_projects_v2(next_states, controller, genome, player)
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


def m35_player_action_dict_v2(action: E4PlayerActionV1) -> dict:
    return {
        "unit_op": action.unit_op,
        "unit_item": action.unit_item,
        "unit_amount": action.unit_amount,
        "unit_count": action.unit_count,
        "market_count": action.market_count,
        "market_op": action.market_op,
        "market_item": action.market_item,
        "market_amount": action.market_amount,
    }


__all__ = [
    "ensure_m35_projects_v2",
    "m35_player_action_dict_v2",
    "m35_policy_step_v2",
    "m35_unit_masks_v2",
    "materialize_m35_market_tasks_v2",
    "materialize_m35_unit_tasks_v2",
    "update_m35_controller_from_effects_v2",
]
