"""M3.6C calendar-driven full-farm controller."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    MAX_MARKET_ORDERS,
    NUM_PRODUCTS,
    SHED_CAPACITY,
    TURNS_PER_DAY,
    UnitOp,
)
from kaggriculture_jax.types import State, StaticTables
from strategic_v5.constants import TaskStatusV1, TaskTypeV1
from strategic_v5.e4_executor import E4PlayerActionV1, compile_full_core_player_action_v1

from .crop_executor import clear_invalidated_crop_tasks_v2
from .lifecycle import (
    empty_market_tasks_v2,
    empty_unit_tasks_v2,
    reconcile_project_controller_v2,
)
from .m3_controller import (
    clear_invalidated_m3_tasks_v2,
    derive_m3_commitment_ledger_v2,
)
from .m35_controller import (
    ensure_m35_projects_v2,
    materialize_m35_market_tasks_v2,
    update_m35_controller_from_effects_v2,
)
from .m35_schema import M35FarmGenomeV2
from .m36_genome_adapter import calendar_step_genome_v3
from .m36_scheduler import (
    mark_m37_daily_unlock_gate_v1,
    materialize_m36_unified_unit_tasks_v3,
    planned_service_tile_mask_v3,
)
from .m36_schema import (
    M36FertilizerDiagnosticsV3,
    M36MarketTemplateV3,
    M36SchedulerDiagnosticsV3,
    RouteCalendarV3,
)
from .schema import ProjectControllerStateV2
from .m38_route_cards import (
    RouteCardModeV4,
    clear_expired_route_cards_v3,
    materialize_route_card_unit_tasks_v3,
    materialize_route_cards_v3,
    update_route_cards_from_effects_v3,
)


FERTILIZER_PRODUCT_ID = 8


def continue_m36_fertilizer_batch_after_effects_v3(
    states: State,
    next_states: State,
    planned: ProjectControllerStateV2,
    updated: ProjectControllerStateV2,
    action: E4PlayerActionV1,
    player: int,
) -> ProjectControllerStateV2:
    """Release a successful fertilizer collection to collect another tile.

    The generic E3 lifecycle changes every successful fertilizer collection
    directly into a return-to-shed leg.  Gold routes instead collect several
    nearby fertilizer units with one worker and deposit once.  M3.6 keeps the
    verified primitive effect accounting, then releases only that successful
    task.  On the next step the live scheduler chooses another feasible
    collection or emits SHED_DEPOSIT.  No Replay action or precomputed path is
    stored.
    """

    before = planned.unit_tasks
    pre_inventory = states.unit_inventory[
        :, player, :, FERTILIZER_PRODUCT_ID
    ]
    post_inventory = next_states.unit_inventory[
        :, player, :, FERTILIZER_PRODUCT_ID
    ]
    fertilizer_collected = (
        (before.status == TaskStatusV1.ACTIVE)
        & (before.task_type == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
        & (action.unit_op == UnitOp.COLLECT_FERTILIZER)
        & (post_inventory > pre_inventory)
    )
    blank = empty_unit_tasks_v2(states.step.shape[0])
    unit_tasks = jax.tree.map(
        lambda value, empty_value: jnp.where(
            fertilizer_collected, empty_value, value
        ),
        updated.unit_tasks,
        blank,
    )
    return updated._replace(unit_tasks=unit_tasks)


def order_m36_market_tasks_v3(
    states: State,
    controller: ProjectControllerStateV2,
    calendar: RouteCalendarV3,
) -> ProjectControllerStateV2:
    """Order the ten legal slots by the current day's transaction template."""

    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    day = jnp.clip(states.step.astype(jnp.int32) // 24, 0, 29)
    template = calendar.market_template_by_day[batch, day]
    tasks = controller.market_tasks
    active = tasks.status == TaskStatusV1.ACTIVE
    task = tasks.task_type
    item = tasks.item_id
    is_sell = (task == TaskTypeV1.SELL_INVENTORY) | (
        task == TaskTypeV1.TERMINAL_LIQUIDATION
    )
    is_input = (task == TaskTypeV1.BUY_PRODUCT) & (
        (item == 0) | (item == FERTILIZER_PRODUCT_ID)
    )
    is_hire = task == TaskTypeV1.HIRE_WORKER
    is_animal = task == TaskTypeV1.ANIMAL_PURCHASE
    is_seed = (task == TaskTypeV1.CROP_PRODUCTION) & (item >= 0)
    is_land = task == TaskTypeV1.BUY_LAND

    def priority(sell, hire, animal, seed, product_input, land):
        return jnp.where(
            sell,
            0,
            jnp.where(
                hire,
                1,
                jnp.where(
                    animal,
                    2,
                    jnp.where(
                        seed,
                        3,
                        jnp.where(product_input, 4, jnp.where(land, 5, 6)),
                    ),
                ),
            ),
        )

    sell_first = priority(is_sell, is_hire, is_animal, is_seed, is_input, is_land)
    default = priority(
        jnp.zeros_like(is_sell), is_hire, is_animal, is_seed, is_input, is_land
    )
    default = jnp.where(is_sell, 5, default)
    input_first = priority(
        is_input,
        is_hire,
        is_animal,
        is_seed,
        jnp.zeros_like(is_input),
        is_land,
    )
    input_first = jnp.where(is_sell, 5, input_first)
    selected = jnp.where(
        template[:, None] == M36MarketTemplateV3.SELL_HIRE_ANIMAL_SEED_FEED,
        sell_first,
        jnp.where(
            template[:, None]
            == M36MarketTemplateV3.CRITICAL_INPUT_HIRE_ANIMAL_SELL,
            input_first,
            default,
        ),
    ).astype(jnp.int32)
    ordinal = jnp.arange(MAX_MARKET_ORDERS, dtype=jnp.int32)[None]
    order = jnp.argsort(
        jnp.where(active, selected * 100 + ordinal, 100_000 + ordinal), axis=-1
    )
    ordered = jax.tree.map(
        lambda value: jnp.take_along_axis(value, order, axis=1), tasks
    )
    return controller._replace(market_tasks=ordered)


def materialize_m36_fertilizer_ledger_v3(
    states: State,
    controller: ProjectControllerStateV2,
    calendar: RouteCalendarV3,
    player: int,
) -> tuple[ProjectControllerStateV2, M36FertilizerDiagnosticsV3]:
    """Reserve exactly the declared safety/application stock and sell surplus."""

    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    day = jnp.clip(states.step.astype(jnp.int32) // 24, 0, 29)
    safety = calendar.fertilizer_safety_stock_by_day[batch, day].astype(jnp.int16)
    shed_held = states.shed[:, player, FERTILIZER_PRODUCT_ID].astype(jnp.int16)
    carried = jnp.sum(
        states.unit_inventory[:, player, :, FERTILIZER_PRODUCT_ID].astype(
            jnp.int16
        ),
        axis=-1,
        dtype=jnp.int16,
    )
    held = shed_held + carried
    apply_reserved = jnp.sum(
        (controller.unit_tasks.status == TaskStatusV1.ACTIVE)
        & (controller.unit_tasks.task_type == TaskTypeV1.APPLY_FERTILIZER),
        axis=-1,
        dtype=jnp.int16,
    )
    reserved = jnp.minimum(held, safety + apply_reserved).astype(jnp.int16)
    surplus = jnp.maximum(held - reserved, 0).astype(jnp.int16)
    # Market SELL can consume only shed stock.  A mid-day planning pass may see
    # fertilizer carried by a worker, but emitting an order for that stock is a
    # deterministic no-op until the worker deposits it.  Reserve carried stock
    # first, then expose only the physically sellable shed remainder.
    reserve_from_shed = jnp.maximum(reserved - carried, 0).astype(jnp.int16)
    sellable_surplus = jnp.maximum(shed_held - reserve_from_shed, 0).astype(
        jnp.int16
    )
    market = controller.market_tasks
    active = market.status == TaskStatusV1.ACTIVE
    fertilizer_sell = (
        active
        & (
            (market.task_type == TaskTypeV1.SELL_INVENTORY)
            | (market.task_type == TaskTypeV1.TERMINAL_LIQUIDATION)
        )
        & (market.item_id == FERTILIZER_PRODUCT_ID)
    )
    has_sell = jnp.any(fertilizer_sell, axis=-1)
    first_sell = jnp.argmax(fertilizer_sell, axis=-1).astype(jnp.int32)
    existing_sell = market.quantity[batch, first_sell].astype(jnp.int16)
    sell_amount = jnp.where(
        has_sell, jnp.minimum(existing_sell, sellable_surplus), 0
    ).astype(jnp.int16)

    # Clear every old fertilizer sell slot, then install one exact surplus
    # order.  This prevents two independent subprojects from selling the same
    # reserved fertilizer.
    blank = empty_market_tasks_v2(batch_size)
    market = jax.tree.map(
        lambda value, empty_value: jnp.where(fertilizer_sell, empty_value, value),
        market,
        blank,
    )
    active_after = market.status == TaskStatusV1.ACTIVE
    empty_slot = ~active_after
    first_empty = jnp.argmax(empty_slot, axis=-1).astype(jnp.int32)
    has_empty = jnp.any(empty_slot, axis=-1)
    # Do not sell on the day-end refresh turn: newly generated fertilizer can
    # exactly replace the sold quantity and make a real sale look like an
    # effect mismatch in net-inventory accounting.  On ordinary turns, a
    # profitable fertilizer sale must be part of the ten-slot transaction,
    # not an eleventh order.  If all slots are occupied, replace the final
    # (lowest retained priority) order; the hour-1 daily replanning pass can
    # issue that displaced commitment after the sale has funded it.
    day_end_turn = ((states.step + 1) % 24) == 0
    needs_sell = (sellable_surplus > 0) & (~day_end_turn)
    install = needs_sell
    slot = jnp.where(
        has_empty,
        first_empty,
        jnp.int32(MAX_MARKET_ORDERS - 1),
    )

    def set_field(field, value):
        old = field[batch, slot]
        return field.at[batch, slot].set(jnp.where(install, value, old))

    market = market._replace(
        task_type=set_field(
            market.task_type,
            jnp.full(
                (batch_size,), TaskTypeV1.SELL_INVENTORY, dtype=jnp.int8
            ),
        ),
        item_id=set_field(
            market.item_id,
            jnp.full((batch_size,), FERTILIZER_PRODUCT_ID, dtype=jnp.int8),
        ),
        quantity=set_field(market.quantity, sellable_surplus),
        start_step=set_field(market.start_step, states.step.astype(jnp.int16)),
        deadline_step=set_field(
            market.deadline_step,
            jnp.minimum(states.step + 1, 718).astype(jnp.int16),
        ),
        status=set_field(
            market.status,
            jnp.full((batch_size,), TaskStatusV1.ACTIVE, dtype=jnp.int8),
        ),
    )
    diagnostics = M36FertilizerDiagnosticsV3(
        held_units=held,
        reserved_units=reserved,
        surplus_units=surplus,
        sell_units=jnp.where(install, sellable_surplus, 0).astype(jnp.int16),
        # The replacement above makes the transaction representable within
        # the official ten-slot cap; no order is emitted out of bounds.
        market_slot_overflow_count=jnp.zeros(
            (batch_size,), dtype=jnp.int32
        ),
    )
    return controller._replace(market_tasks=market), diagnostics


def materialize_m36_reactive_sales_v3(
    states: State,
    controller: ProjectControllerStateV2,
    calendar: RouteCalendarV3,
    genome: M35FarmGenomeV2,
    player: int,
) -> ProjectControllerStateV2:
    """Cheap per-step sales repair; investment planning remains once per day."""

    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    day = jnp.clip(states.step.astype(jnp.int32) // 24, 0, 29)
    reserve = jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int16)
    animal_ledger = derive_m3_commitment_ledger_v2(
        states, controller, genome.animal, player
    )
    # ``feed_stock_target_by_day`` records explicit Replay purchase intent, not
    # the complete amount that the current physical animal commitments still
    # require.  Treating a zero Replay purchase as a zero reserve caused the
    # light per-step repair to sell wheat already reserved by the animal
    # planner, followed by avoidable feed failures and escapes.  Keep whichever
    # reserve is larger: the calendar's explicit target or the live commitment
    # ledger's rolling feed horizon.
    committed_feed = jnp.sum(
        animal_ledger.planned_feed_horizon, axis=-1, dtype=jnp.int16
    )
    reserve = reserve.at[:, 0].set(
        jnp.maximum(
            calendar.feed_stock_target_by_day[batch, day].astype(jnp.int16),
            committed_feed,
        )
    )
    apply_reserved = jnp.sum(
        (controller.unit_tasks.status == TaskStatusV1.ACTIVE)
        & (controller.unit_tasks.task_type == TaskTypeV1.APPLY_FERTILIZER),
        axis=-1,
        dtype=jnp.int16,
    )
    reserve = reserve.at[:, FERTILIZER_PRODUCT_ID].set(
        calendar.fertilizer_safety_stock_by_day[batch, day].astype(jnp.int16)
        + apply_reserved
    )
    market = controller.market_tasks
    existing_sell = (
        (market.status == TaskStatusV1.ACTIVE)
        & (
            (market.task_type == TaskTypeV1.SELL_INVENTORY)
            | (market.task_type == TaskTypeV1.TERMINAL_LIQUIDATION)
        )
    )
    existing_item = jnp.clip(
        market.item_id.astype(jnp.int32), 0, NUM_PRODUCTS - 1
    )
    # A fertilizer sale on the final turn of a day is economically valid, but
    # animal refresh can create fertilizer in the same transition.  That makes
    # the shed's net delta ambiguous to the strict effect checker (sell one,
    # generate one => no observed decrease).  Defer only that sale by one turn;
    # every other product and every non-refresh fertilizer sale is unchanged.
    # Clear a fertilizer sell that was installed on the preceding light step as
    # well as preventing a new one below.
    day_end_turn = ((states.step + 1) % 24) == 0
    defer_fertilizer_sell = (
        day_end_turn[:, None]
        & existing_sell
        & (existing_item == FERTILIZER_PRODUCT_ID)
    )
    blank_market = empty_market_tasks_v2(batch_size)
    market = jax.tree.map(
        lambda value, empty_value: jnp.where(
            defer_fertilizer_sell, empty_value, value
        ),
        market,
        blank_market,
    )
    existing_sell = existing_sell & (~defer_fertilizer_sell)
    reserved_sell = jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int16)
    reserved_sell = reserved_sell.at[batch[:, None], existing_item].add(
        jnp.where(existing_sell, market.quantity, 0).astype(jnp.int16)
    )
    available = jnp.maximum(
        states.shed[:, player, :NUM_PRODUCTS].astype(jnp.int16)
        - reserve
        - reserved_sell,
        0,
    )
    for product in range(NUM_PRODUCTS):
        quantity = available[:, product]
        empty_slot = market.status != TaskStatusV1.ACTIVE
        has_empty = jnp.any(empty_slot, axis=-1)
        slot = jnp.argmax(empty_slot, axis=-1).astype(jnp.int32)
        install = (quantity > 0) & has_empty
        if product == FERTILIZER_PRODUCT_ID:
            install = install & (~day_end_turn)

        def set_field(field, value):
            old = field[batch, slot]
            return field.at[batch, slot].set(jnp.where(install, value, old))

        market = market._replace(
            task_type=set_field(
                market.task_type,
                jnp.full(
                    (batch_size,), TaskTypeV1.SELL_INVENTORY, dtype=jnp.int8
                ),
            ),
            item_id=set_field(
                market.item_id,
                jnp.full((batch_size,), product, dtype=jnp.int8),
            ),
            quantity=set_field(market.quantity, quantity.astype(jnp.int16)),
            start_step=set_field(market.start_step, states.step.astype(jnp.int16)),
            deadline_step=set_field(
                market.deadline_step,
                jnp.minimum(states.step + 1, 718).astype(jnp.int16),
            ),
            status=set_field(
                market.status,
                jnp.full((batch_size,), TaskStatusV1.ACTIVE, dtype=jnp.int8),
            ),
        )
    return controller._replace(market_tasks=market)


def materialize_m37_emergency_feed_buy_v1(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M35FarmGenomeV2,
    player: int,
) -> ProjectControllerStateV2:
    """Install one immediately affordable wheat unit for survival debt.

    Full economic planning remains low-frequency.  This narrow repair reacts
    only after realized cash or shed state changes make an existing, hard
    animal commitment serviceable.  Buying one unit avoids speculative bulk
    quotes in the light graph and is sufficient for the scheduler to launch a
    feed route on the next state.  Survival may displace an optional future
    investment, but never a sale or another already planned wheat order.
    """

    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    ledger = derive_m3_commitment_ledger_v2(
        states, controller, genome.animal, player
    )
    survival_due = jnp.sum(
        ledger.survival_feed_due_today, axis=-1, dtype=jnp.int16
    )
    wheat_held = (
        states.shed[:, player, 0].astype(jnp.int16)
        + jnp.sum(
            states.unit_inventory[:, player, :, 0].astype(jnp.int16),
            axis=-1,
            dtype=jnp.int16,
        )
    )
    deficit = jnp.maximum(survival_due - wheat_held, 0)
    shed_used = jnp.sum(
        states.shed[:, player].astype(jnp.int32), axis=-1, dtype=jnp.int32
    )

    market = controller.market_tasks
    active = market.status == TaskStatusV1.ACTIVE
    existing_wheat_buy = jnp.any(
        active
        & (market.task_type == TaskTypeV1.BUY_PRODUCT)
        & (market.item_id == 0),
        axis=-1,
    )
    empty = ~active
    optional_investment = active & (
        (market.task_type == TaskTypeV1.HIRE_WORKER)
        | (market.task_type == TaskTypeV1.BUY_LAND)
        | (market.task_type == TaskTypeV1.ANIMAL_PURCHASE)
        | (
            (market.task_type == TaskTypeV1.BUY_PRODUCT)
            & (market.item_id != 0)
        )
    )
    has_empty = jnp.any(empty, axis=-1)
    has_optional = jnp.any(optional_investment, axis=-1)
    empty_slot = jnp.argmax(empty, axis=-1).astype(jnp.int32)
    optional_slot = jnp.argmax(optional_investment, axis=-1).astype(jnp.int32)
    slot = jnp.where(has_empty, empty_slot, optional_slot)

    wheat_unit_price = states.market_price[:, 0].astype(jnp.int32)
    last_service_day_end = jnp.int32(719 - TURNS_PER_DAY)
    install = (
        (deficit > 0)
        & (~existing_wheat_buy)
        & (states.money[:, player].astype(jnp.int32) >= wheat_unit_price)
        & (states.market_inventory[:, 0] > 0)
        & (shed_used < SHED_CAPACITY)
        & (states.step.astype(jnp.int32) < last_service_day_end)
        & (has_empty | has_optional)
    )
    day_end = (
        ((states.step.astype(jnp.int32) // TURNS_PER_DAY) + 1)
        * TURNS_PER_DAY
        - 1
    )

    def set_field(field, value):
        old = field[batch, slot]
        return field.at[batch, slot].set(jnp.where(install, value, old))

    market = market._replace(
        task_type=set_field(
            market.task_type,
            jnp.full((batch_size,), TaskTypeV1.BUY_PRODUCT, dtype=jnp.int8),
        ),
        item_id=set_field(
            market.item_id, jnp.zeros((batch_size,), dtype=jnp.int8)
        ),
        quantity=set_field(
            market.quantity, jnp.ones((batch_size,), dtype=jnp.int16)
        ),
        start_step=set_field(
            market.start_step, states.step.astype(jnp.int16)
        ),
        deadline_step=set_field(
            market.deadline_step, day_end.astype(jnp.int16)
        ),
        status=set_field(
            market.status,
            jnp.full((batch_size,), TaskStatusV1.ACTIVE, dtype=jnp.int8),
        ),
    )
    return controller._replace(market_tasks=market)


def m36c_daily_plan_v3(
    states: State,
    controller: ProjectControllerStateV2,
    calendar: RouteCalendarV3,
    template: M35FarmGenomeV2,
    player: int,
    tables: StaticTables,
    *,
    enable_unlock_committed_capital: bool = False,
) -> tuple[ProjectControllerStateV2, M36FertilizerDiagnosticsV3]:
    """Full project/market planning kernel, called once at each day boundary."""

    genome = calendar_step_genome_v3(states, calendar, template)
    controller, _ = reconcile_project_controller_v2(states, controller, player)
    controller = clear_invalidated_crop_tasks_v2(states, controller, player)
    controller = clear_invalidated_m3_tasks_v2(
        states, controller, genome.animal, player
    )
    controller = ensure_m35_projects_v2(states, controller, genome, player)
    if enable_unlock_committed_capital:
        controller = mark_m37_daily_unlock_gate_v1(
            states, controller, calendar, genome, player
        )
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)
    day = jnp.clip(states.step.astype(jnp.int32) // 24, 0, 29)
    controller = materialize_m35_market_tasks_v2(
        states,
        controller,
        genome,
        player,
        tables,
        market_template=calendar.market_template_by_day[batch, day],
        reserve_future_hires=False,
    )
    controller, fertilizer = materialize_m36_fertilizer_ledger_v3(
        states, controller, calendar, player
    )
    controller = order_m36_market_tasks_v3(states, controller, calendar)
    return controller, fertilizer


def m36c_light_policy_step_v3(
    states: State,
    controller: ProjectControllerStateV2,
    calendar: RouteCalendarV3,
    template: M35FarmGenomeV2,
    player: int,
    *,
    enable_unlock_committed_capital: bool = False,
    enable_route_cards: bool = False,
    route_card_mode: int = RouteCardModeV4.LEGACY_M38,
) -> tuple[
    E4PlayerActionV1,
    ProjectControllerStateV2,
    M35FarmGenomeV2,
    M36SchedulerDiagnosticsV3,
]:
    """Per-step kernel: reconcile, urgent scheduling, reactive sales, execute."""

    genome = calendar_step_genome_v3(states, calendar, template)
    controller, _ = reconcile_project_controller_v2(states, controller, player)
    controller = clear_invalidated_crop_tasks_v2(states, controller, player)
    controller = clear_invalidated_m3_tasks_v2(
        states, controller, genome.animal, player
    )
    controller = ensure_m35_projects_v2(states, controller, genome, player)
    allowed_service = None
    if enable_route_cards:
        controller = clear_expired_route_cards_v3(
            states, controller, player
        )
        allowed_service, _ = planned_service_tile_mask_v3(
            states, calendar, player
        )
        # Full route admission is more expensive than advancing an existing
        # card.  Gold-style commitments are created at three deterministic
        # planning windows per day; the primitive scheduler remains live on
        # every step and handles single-target emergencies between windows.
        admission = jnp.any(
            (states.step % TURNS_PER_DAY == 0)
            | (states.step % TURNS_PER_DAY == 1)
            | (states.step % TURNS_PER_DAY == 12)
        )
        controller = jax.lax.cond(
            admission,
            lambda value: materialize_route_cards_v3(
                states,
                value,
                genome,
                calendar,
                allowed_service,
                player,
                route_card_mode,
            )[0],
            lambda value: value,
            controller,
        )
        controller = materialize_route_card_unit_tasks_v3(
            states, controller, player
        )
    controller, scheduler = materialize_m36_unified_unit_tasks_v3(
        states,
        controller,
        genome,
        calendar,
        player,
        enable_unlock_committed_capital=enable_unlock_committed_capital,
        allowed_service_tiles=allowed_service,
    )
    controller = materialize_m36_reactive_sales_v3(
        states, controller, calendar, genome, player
    )
    if enable_unlock_committed_capital:
        controller = materialize_m37_emergency_feed_buy_v1(
            states, controller, genome, player
        )
    controller = order_m36_market_tasks_v3(states, controller, calendar)
    action = compile_full_core_player_action_v1(states, controller, player)
    return action, controller, genome, scheduler


def m36c_policy_step_v3(
    states: State,
    controller: ProjectControllerStateV2,
    calendar: RouteCalendarV3,
    template: M35FarmGenomeV2,
    player: int,
    tables: StaticTables,
    *,
    enable_route_cards: bool = False,
    route_card_mode: int = RouteCardModeV4.LEGACY_M38,
) -> tuple[
    E4PlayerActionV1,
    ProjectControllerStateV2,
    M35FarmGenomeV2,
    M36SchedulerDiagnosticsV3,
    M36FertilizerDiagnosticsV3,
]:
    genome = calendar_step_genome_v3(states, calendar, template)
    controller, _ = reconcile_project_controller_v2(states, controller, player)
    controller = clear_invalidated_crop_tasks_v2(states, controller, player)
    controller = clear_invalidated_m3_tasks_v2(
        states, controller, genome.animal, player
    )
    controller = ensure_m35_projects_v2(states, controller, genome, player)
    allowed_service = None
    if enable_route_cards:
        controller = clear_expired_route_cards_v3(
            states, controller, player
        )
        allowed_service, _ = planned_service_tile_mask_v3(
            states, calendar, player
        )
        admission = jnp.any(
            (states.step % TURNS_PER_DAY == 0)
            | (states.step % TURNS_PER_DAY == 1)
            | (states.step % TURNS_PER_DAY == 12)
        )
        controller = jax.lax.cond(
            admission,
            lambda value: materialize_route_cards_v3(
                states,
                value,
                genome,
                calendar,
                allowed_service,
                player,
                route_card_mode,
            )[0],
            lambda value: value,
            controller,
        )
        controller = materialize_route_card_unit_tasks_v3(
            states, controller, player
        )
    controller, scheduler = materialize_m36_unified_unit_tasks_v3(
        states,
        controller,
        genome,
        calendar,
        player,
        allowed_service_tiles=allowed_service,
    )
    controller = materialize_m35_market_tasks_v2(
        states,
        controller,
        genome,
        player,
        tables,
        market_template=calendar.market_template_by_day[
            jnp.arange(states.step.shape[0], dtype=jnp.int32),
            jnp.clip(states.step.astype(jnp.int32) // 24, 0, 29),
        ],
        reserve_future_hires=False,
    )
    controller, fertilizer = materialize_m36_fertilizer_ledger_v3(
        states, controller, calendar, player
    )
    controller = materialize_m36_reactive_sales_v3(
        states, controller, calendar, genome, player
    )
    controller = order_m36_market_tasks_v3(states, controller, calendar)
    action = compile_full_core_player_action_v1(states, controller, player)
    return action, controller, genome, scheduler, fertilizer


__all__ = [
    "continue_m36_fertilizer_batch_after_effects_v3",
    "m36c_daily_plan_v3",
    "m36c_light_policy_step_v3",
    "m36c_policy_step_v3",
    "materialize_m36_fertilizer_ledger_v3",
    "materialize_m36_reactive_sales_v3",
    "materialize_m37_emergency_feed_buy_v1",
    "order_m36_market_tasks_v3",
    "update_route_cards_from_effects_v3",
    "update_m35_controller_from_effects_v2",
]
