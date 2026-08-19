"""M3.6C unified crop/animal task selection with sticky local scheduling."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_COST,
    ANIMAL_FIRST_YIELD_DAY,
    ANIMAL_INTERVAL,
    ANIMAL_PRODUCT,
    BOARD_SIZE,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_WATERED,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    SHED_ACCESS,
    TURNS_PER_DAY,
    TileKind,
)
from kaggriculture_jax.types import State
from strategic_v5.constants import TaskPhaseV1, TaskStatusV1, TaskTypeV1
from strategic_v5.schema import UnitTaskStateV1

from .lifecycle import empty_unit_tasks_v2
from .m26_controller import materialize_m26_unit_tasks_v2
from .m26_controller import m26_phase_targets_v2
from .m3_controller import (
    m3_care_obligation_mask_v2,
    m3_feed_obligation_masks_v2,
    materialize_m3_unit_tasks_v2,
    m3_phase_targets_v2,
)
from .m3_constants import M3AnimalFertilizerPolicyV2
from .m35_schema import M35FarmGenomeV2
from .m36_schema import (
    M36ReleasePolicyV3,
    M36SchedulerDiagnosticsV3,
    RouteCalendarV3,
)
from .m38_route_cards import active_route_target_mask_v3
from .schema import ProjectControllerStateV2


_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int16)
_ANIMAL_COST = jnp.asarray(ANIMAL_COST, dtype=jnp.int32)
_ANIMAL_FIRST = jnp.asarray(ANIMAL_FIRST_YIELD_DAY, dtype=jnp.int16)
_ANIMAL_INTERVAL = jnp.asarray(ANIMAL_INTERVAL, dtype=jnp.int16)
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int32)
_DISTANCE_TILE_ORDER = jnp.asarray(
    tuple(
        sorted(
            range(BOARD_SIZE * BOARD_SIZE),
            key=lambda tile: (
                min(
                    abs((tile % BOARD_SIZE) - int(access[0]))
                    + abs((tile // BOARD_SIZE) - int(access[1]))
                    for access in SHED_ACCESS
                ),
                tile,
            ),
        )
    ),
    dtype=jnp.int32,
)
_DISTANCE_TILE_INVERSE = jnp.argsort(_DISTANCE_TILE_ORDER)
_HARD_TYPES = (
    TaskTypeV1.WATER_CROP,
    TaskTypeV1.ANIMAL_FEED,
    TaskTypeV1.ANIMAL_COLLECT_PRODUCT,
    TaskTypeV1.SHED_DEPOSIT,
)
_ANIMAL_SERVICE_TYPES = (
    TaskTypeV1.ANIMAL_FEED,
    TaskTypeV1.ANIMAL_CARE,
)


def _task_present(tasks: UnitTaskStateV1) -> jax.Array:
    return tasks.status == TaskStatusV1.ACTIVE


def _task_target_distance(states: State, tasks: UnitTaskStateV1, player: int) -> jax.Array:
    positions = states.unit_pos[:, player].astype(jnp.int16)
    target = jnp.stack((tasks.target_x, tasks.target_y), axis=-1).astype(jnp.int16)
    tile_distance = jnp.sum(jnp.abs(positions - target), axis=-1, dtype=jnp.int16)
    depot_distance = jnp.min(
        jnp.sum(
            jnp.abs(positions[:, :, None] - _SHED_ACCESS[None, None]),
            axis=-1,
            dtype=jnp.int16,
        ),
        axis=-1,
    )
    returning = (tasks.phase == TaskPhaseV1.MOVE_TO_DEPOT) | (
        tasks.phase == TaskPhaseV1.DEPOSIT
    )
    return jnp.where(returning, depot_distance, tile_distance).astype(jnp.int16)


def _task_priority(
    states: State, tasks: UnitTaskStateV1, player: int
) -> tuple[jax.Array, jax.Array]:
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)[:, None]
    x = jnp.clip(tasks.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(tasks.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    neglect = states.tile_neglect[batch, player, y, x]
    task = tasks.task_type
    returning = (tasks.phase == TaskPhaseV1.MOVE_TO_DEPOT) | (
        tasks.phase == TaskPhaseV1.DEPOSIT
    )
    hard_feed = (task == TaskTypeV1.ANIMAL_FEED) & (neglect >= 1)
    hard = hard_feed | (task == TaskTypeV1.WATER_CROP) | returning
    base = jnp.where(
        hard,
        0,
        jnp.where(
            task == TaskTypeV1.ANIMAL_COLLECT_PRODUCT,
            1,
            jnp.where(
                task == TaskTypeV1.ANIMAL_FEED,
                2,
                jnp.where(
                    task == TaskTypeV1.ANIMAL_CARE,
                    3,
                    jnp.where(
                        task == TaskTypeV1.ANIMAL_PLACE,
                        4,
                        jnp.where(
                            task == TaskTypeV1.BUILD_ANIMAL_STRUCTURE,
                            5,
                            jnp.where(
                                task == TaskTypeV1.CROP_PRODUCTION,
                                6,
                                jnp.where(
                                    task == TaskTypeV1.CLEAR_OR_REMOVE_TILE,
                                    7,
                                    8,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int16)
    slack = tasks.deadline_step.astype(jnp.int32) - states.step[:, None].astype(
        jnp.int32
    )
    deadline_urgent = (tasks.deadline_step >= 0) & (slack <= 3)
    base = jnp.where(deadline_urgent, jnp.minimum(base, 1), base).astype(jnp.int16)
    return base, hard | deadline_urgent


def planned_service_tile_mask_v3(
    states: State, calendar: RouteCalendarV3, player: int
) -> tuple[jax.Array, jax.Array]:
    """Choose the deterministic subset that remains under daily maintenance."""

    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    day = jnp.clip(states.step.astype(jnp.int32) // TURNS_PER_DAY, 0, 29)
    target = calendar.animal_service_target_by_day[batch, day].astype(jnp.int16)
    policy = calendar.planned_release_policy_by_day[batch, day]
    animal = states.tile_animal[:, player].reshape(batch_size, -1)
    held = states.tile_yield[:, player].reshape(batch_size, -1).astype(jnp.int16)
    pending = states.tile_pending_care[:, player].reshape(batch_size, -1).astype(
        jnp.int16
    )
    allowed = jnp.zeros_like(animal, dtype=jnp.bool_)

    def prefix_keep(present: jax.Array, count: jax.Array) -> jax.Array:
        ordered = present[:, _DISTANCE_TILE_ORDER]
        rank = jnp.cumsum(ordered.astype(jnp.int16), axis=-1)
        keep_ordered = ordered & (rank <= count[:, None])
        # ``_DISTANCE_TILE_ORDER`` is a static permutation, so its inverse is
        # a gather.  The previous equivalent scatter was harmless at runtime
        # but triggered a pathological XLA GPU optimisation pass when this
        # selector was inlined twice into the 719-step scan body.
        return keep_ordered[:, _DISTANCE_TILE_INVERSE]

    for species in range(NUM_ANIMALS):
        present = animal == species
        closest = prefix_keep(present, target[:, species])
        pending_value = (held > 0) | (pending > 0)
        product_value = held > 0
        valuable_future = present & pending_value
        valuable_pending = present & product_value
        keep_future_first = prefix_keep(valuable_future, target[:, species])
        future_room = jnp.maximum(
            target[:, species]
            - jnp.sum(keep_future_first, axis=-1, dtype=jnp.int16),
            0,
        )
        keep_future = keep_future_first | prefix_keep(
            present & (~keep_future_first), future_room
        )
        keep_pending_first = prefix_keep(valuable_pending, target[:, species])
        pending_room = jnp.maximum(
            target[:, species]
            - jnp.sum(keep_pending_first, axis=-1, dtype=jnp.int16),
            0,
        )
        keep_pending = keep_pending_first | prefix_keep(
            present & (~keep_pending_first), pending_room
        )
        keep = jnp.where(
            policy[:, None] == M36ReleasePolicyV3.FARTHEST_FROM_SERVICE_ROUTE,
            closest,
            jnp.where(
                policy[:, None] == M36ReleasePolicyV3.LOWEST_PENDING_VALUE,
                keep_pending,
                keep_future,
            ),
        )
        allowed = allowed | keep
    release_count = jnp.sum(
        (animal >= 0) & (~allowed), axis=-1, dtype=jnp.int32
    )
    return allowed.reshape(batch_size, BOARD_SIZE, BOARD_SIZE), release_count


def _service_allowed_for_tasks(
    tasks: UnitTaskStateV1, allowed_tiles: jax.Array
) -> jax.Array:
    batch = jnp.arange(tasks.task_type.shape[0], dtype=jnp.int32)[:, None]
    x = jnp.clip(tasks.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    y = jnp.clip(tasks.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    target_allowed = allowed_tiles[batch, y, x]
    service = (tasks.task_type == TaskTypeV1.ANIMAL_FEED) | (
        tasks.task_type == TaskTypeV1.ANIMAL_CARE
    )
    return (~service) | target_allowed


def _choose_proposals_v3(
    states: State,
    existing: UnitTaskStateV1,
    crop: UnitTaskStateV1,
    animal: UnitTaskStateV1,
    allowed_tiles: jax.Array,
    player: int,
) -> tuple[UnitTaskStateV1, M36SchedulerDiagnosticsV3]:
    batch_size = states.step.shape[0]
    active_unit = states.unit_active[:, player]
    sticky = _task_present(existing) & active_unit
    crop_present = _task_present(crop) & (~sticky) & active_unit
    animal_present = (
        _task_present(animal)
        & (~sticky)
        & active_unit
        & _service_allowed_for_tasks(animal, allowed_tiles)
    )
    crop_priority, crop_hard = _task_priority(states, crop, player)
    animal_priority, animal_hard = _task_priority(states, animal, player)
    crop_score = crop_priority.astype(jnp.int32) * 1_000 + _task_target_distance(
        states, crop, player
    ).astype(jnp.int32)
    animal_score = animal_priority.astype(jnp.int32) * 1_000 + _task_target_distance(
        states, animal, player
    ).astype(jnp.int32)
    choose_animal = animal_present & (
        (~crop_present) | (animal_score < crop_score)
    )
    choose_crop = crop_present & (~choose_animal)
    selected = jax.tree.map(
        lambda old, crop_value, animal_value: jnp.where(
            sticky,
            old,
            jnp.where(choose_animal, animal_value, crop_value),
        ),
        existing,
        crop,
        animal,
    )
    empty = empty_unit_tasks_v2(batch_size)
    any_choice = sticky | choose_crop | choose_animal
    selected = jax.tree.map(
        lambda value, blank: jnp.where(any_choice, value, blank), selected, empty
    )

    # Cross-project duplicate guard.  It operates after selection so the raw
    # crop and animal planners remain independently testable.
    mutating = any_choice & (
        (selected.task_type == TaskTypeV1.CROP_PRODUCTION)
        | (selected.task_type == TaskTypeV1.WATER_CROP)
        | (selected.task_type == TaskTypeV1.CLEAR_OR_REMOVE_TILE)
        | (selected.task_type == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
        | (selected.task_type == TaskTypeV1.ANIMAL_PLACE)
        | (selected.task_type == TaskTypeV1.ANIMAL_FEED)
        | (selected.task_type == TaskTypeV1.ANIMAL_CARE)
        | (selected.task_type == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
    )
    same = selected.target_id[:, :, None] == selected.target_id[:, None, :]
    earlier = jnp.tril(jnp.ones((MAX_UNITS, MAX_UNITS), dtype=jnp.bool_), k=-1)
    duplicate = (~sticky) & mutating & jnp.any(
        same & mutating[:, None, :] & earlier[None], axis=-1
    )
    selected = jax.tree.map(
        lambda value, blank: jnp.where(duplicate, blank, value), selected, empty
    )
    choose_crop = choose_crop & (~duplicate)
    choose_animal = choose_animal & (~duplicate)

    # Cheap local improvement: disjoint adjacent free-unit pairs may exchange
    # fresh MOVE_TO_TARGET tasks when that strictly shortens total travel.
    swap_count = jnp.zeros((batch_size,), dtype=jnp.int32)
    positions = states.unit_pos[:, player].astype(jnp.int16)
    inventory_empty = jnp.sum(
        states.unit_inventory[:, player].astype(jnp.int32), axis=-1
    ) == 0
    for left in range(0, MAX_UNITS - 1, 2):
        right = left + 1
        fresh = (
            (~sticky[:, left])
            & (~sticky[:, right])
            & (selected.status[:, left] == TaskStatusV1.ACTIVE)
            & (selected.status[:, right] == TaskStatusV1.ACTIVE)
            & (selected.phase[:, left] == TaskPhaseV1.MOVE_TO_TARGET)
            & (selected.phase[:, right] == TaskPhaseV1.MOVE_TO_TARGET)
            & inventory_empty[:, left]
            & inventory_empty[:, right]
        )
        left_pos = positions[:, left]
        right_pos = positions[:, right]
        left_target = jnp.stack(
            (selected.target_x[:, left], selected.target_y[:, left]), axis=-1
        ).astype(jnp.int16)
        right_target = jnp.stack(
            (selected.target_x[:, right], selected.target_y[:, right]), axis=-1
        ).astype(jnp.int16)
        old_cost = jnp.sum(jnp.abs(left_pos - left_target), axis=-1) + jnp.sum(
            jnp.abs(right_pos - right_target), axis=-1
        )
        new_cost = jnp.sum(jnp.abs(left_pos - right_target), axis=-1) + jnp.sum(
            jnp.abs(right_pos - left_target), axis=-1
        )
        swap = fresh & (new_cost < old_cost)
        fields = {}
        for name in UnitTaskStateV1._fields:
            value = getattr(selected, name)
            left_value = value[:, left]
            right_value = value[:, right]
            value = value.at[:, left].set(jnp.where(swap, right_value, left_value))
            value = value.at[:, right].set(jnp.where(swap, left_value, right_value))
            fields[name] = value
        selected = UnitTaskStateV1(**fields)
        swap_count += swap.astype(jnp.int32)

    owner = jnp.broadcast_to(
        jnp.arange(MAX_UNITS, dtype=jnp.int8)[None], selected.owner_unit.shape
    )
    selected = selected._replace(owner_unit=owner)
    deadline_preempt = choose_animal & animal_hard & crop_present & (~crop_hard)
    diagnostics = M36SchedulerDiagnosticsV3(
        sticky_task_count=jnp.sum(sticky, axis=-1, dtype=jnp.int32),
        crop_candidate_count=jnp.sum(crop_present, axis=-1, dtype=jnp.int32),
        animal_candidate_count=jnp.sum(animal_present, axis=-1, dtype=jnp.int32),
        hard_candidate_count=jnp.sum(
            (crop_present & crop_hard) | (animal_present & animal_hard),
            axis=-1,
            dtype=jnp.int32,
        ),
        selected_crop_count=jnp.sum(choose_crop, axis=-1, dtype=jnp.int32),
        selected_animal_count=jnp.sum(choose_animal, axis=-1, dtype=jnp.int32),
        deadline_preemption_count=jnp.sum(
            deadline_preempt, axis=-1, dtype=jnp.int32
        ),
        duplicate_reservation_prevented=jnp.sum(
            duplicate, axis=-1, dtype=jnp.int32
        ),
        local_swap_count=swap_count,
        planned_release_tile_count=jnp.zeros((batch_size,), dtype=jnp.int32),
    )
    return selected, diagnostics


def m37_unlock_commitment_signals_v1(
    states: State,
    calendar: RouteCalendarV3,
    genome: M35FarmGenomeV2,
    player: int,
    *,
    include_planned_target_gap: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Value-gate already purchased animals before scheduler preemption.

    Returns ``(viable_species, unplaced_count, debt_age_days,
    remaining_future_net_value, locked_capital)``.  The estimate is purposely
    conservative and only decides whether optional work may be displaced; it
    never weakens survival FEED, same-day WATER, legality, or effect checks.
    """

    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    day = jnp.clip(states.step.astype(jnp.int32) // TURNS_PER_DAY, 0, 29)

    tile_animal = states.tile_animal[:, player]
    active = jnp.stack(
        tuple(
            jnp.sum(tile_animal == species, axis=(1, 2), dtype=jnp.int16)
            for species in range(NUM_ANIMALS)
        ),
        axis=-1,
    )
    in_shed = states.shed[
        :, player, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
    ].astype(jnp.int16)
    carried = jnp.sum(
        states.unit_inventory[
            :, player, :, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
        ].astype(jnp.int16),
        axis=1,
        dtype=jnp.int16,
    )
    unplaced = in_shed + carried
    committed = active + unplaced
    _, phase_target, _, _ = m3_phase_targets_v2(states, genome.animal)
    gated_units = jnp.where(
        include_planned_target_gap,
        jnp.maximum(unplaced, phase_target - active),
        unplaced,
    ).astype(jnp.int16)

    # The detailed target-debt ledger is updated by the low-frequency M3.7
    # diagnostics.  The per-step scheduler only needs a cheap age proxy for
    # capital that is physically present but not active: age since the first
    # planned purchase day for that species.  Avoiding a dynamic 30-day prefix
    # scan in the 719-step graph materially reduces XLA compile cost.
    days = jnp.arange(30, dtype=jnp.int32)[None, :, None]
    planned_purchase_day = (
        (days <= day[:, None, None])
        & (calendar.animal_purchase_additions_by_day > 0)
    )
    first_debt_day = jnp.min(
        jnp.where(planned_purchase_day, days, 30), axis=1
    )
    debt_age = jnp.where(
        (first_debt_day < 30) & (gated_units > 0),
        day[:, None] - first_debt_day + 1,
        0,
    ).astype(jnp.int16)

    liquidation = genome.animal.liquidation_start_step.astype(jnp.int32)
    remaining_days = jnp.maximum(
        (liquidation - states.step.astype(jnp.int32)) // TURNS_PER_DAY,
        0,
    )[:, None]
    first_yield = _ANIMAL_FIRST[None].astype(jnp.int32)
    interval = _ANIMAL_INTERVAL[None].astype(jnp.int32)
    cycles = jnp.maximum(
        (remaining_days - first_yield) // jnp.maximum(interval, 1) + 1,
        0,
    )
    product_price = states.market_price[
        batch[:, None], _ANIMAL_PRODUCT[None]
    ].astype(jnp.int32)
    # One wheat-equivalent per maintained day is a conservative cost proxy.
    # Travel/collection opportunity cost is represented by a small fixed daily
    # charge, avoiding a false positive near liquidation.
    feed_price = states.market_price[:, 0].astype(jnp.int32)[:, None]
    future_revenue = cycles * product_price
    future_maintenance = remaining_days * (feed_price + 8)
    future_net = future_revenue - future_maintenance
    locked_capital = unplaced.astype(jnp.int32) * _ANIMAL_COST[None]

    # Include a bounded pickup/place/collection allowance.  This deadline gate
    # prevents sunk capital from receiving unconditional priority when no bank
    # event can occur before liquidation.
    earliest_bank_step = (
        states.step.astype(jnp.int32)[:, None]
        + jnp.int32(12)
        + first_yield * TURNS_PER_DAY
        + jnp.int32(12)
    )
    deadline_feasible = earliest_bank_step < liquidation[:, None]
    viable = (
        (gated_units > 0)
        & (future_net > 0)
        & deadline_feasible
    )
    viable_count = jnp.sum(
        jnp.where(viable, gated_units, 0), axis=-1, dtype=jnp.int32
    )
    return viable, viable_count, debt_age, future_net, locked_capital


def mark_m37_daily_unlock_gate_v1(
    states: State,
    controller: ProjectControllerStateV2,
    calendar: RouteCalendarV3,
    genome: M35FarmGenomeV2,
    player: int,
) -> ProjectControllerStateV2:
    """Store low-frequency animal viability and crop target-debt ages.

    The fixed project slots provide three bytes for animal unlock age and five
    bytes for crop debt age without expanding the frozen calendar or rollout
    schema.  Crop age is reconstructed from the complete 30-day calendar and
    the currently active/pending count, so repeated h0/h1/h20 planning in one
    day is idempotent.
    """

    viable, _, debt_age, _, _ = m37_unlock_commitment_signals_v1(
        states,
        calendar,
        genome,
        player,
        include_planned_target_gap=True,
    )
    encoded_age = jnp.where(
        viable,
        jnp.clip(jnp.maximum(debt_age, 1), 1, 127),
        0,
    ).astype(jnp.int8)
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    day = jnp.clip(
        states.step.astype(jnp.int32) // TURNS_PER_DAY, 0, 29
    )
    kind = states.tile_kind[:, player]
    crop = states.tile_crop[:, player]
    active_crop = jnp.stack(
        tuple(
            jnp.sum(
                (kind == TileKind.PLANT) & (crop == crop_id),
                axis=(1, 2),
                dtype=jnp.int16,
            )
            for crop_id in range(NUM_CROPS)
        ),
        axis=-1,
    )
    tasks = controller.unit_tasks
    task_active = tasks.status == TaskStatusV1.ACTIVE
    tx = jnp.clip(tasks.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    ty = jnp.clip(tasks.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    target_kind = states.tile_kind[batch[:, None], player, ty, tx]
    pending_plant = (
        task_active
        & (tasks.task_type == TaskTypeV1.CROP_PRODUCTION)
        & (target_kind == TileKind.EMPTY)
    )
    pending_crop_id = jnp.clip(
        tasks.item_id.astype(jnp.int32), 0, NUM_CROPS - 1
    )
    pending_crop = jnp.sum(
        jax.nn.one_hot(
            pending_crop_id, NUM_CROPS, dtype=jnp.int16
        )
        * pending_plant[..., None],
        axis=1,
        dtype=jnp.int16,
    )
    committed_crop = active_crop + pending_crop
    current_crop_target = calendar.crop_target_by_day[batch, day]
    historical_required = jnp.minimum(
        calendar.crop_target_by_day,
        current_crop_target[:, None, :],
    )
    day_axis = jnp.arange(30, dtype=jnp.int32)[None, :, None]
    historical_debt = (
        (day_axis <= day[:, None, None])
        & (historical_required > committed_crop[:, None, :])
    )
    first_crop_debt_day = jnp.min(
        jnp.where(historical_debt, day_axis, 30), axis=1
    )
    crop_debt = current_crop_target > committed_crop
    crop_debt_age = jnp.where(
        crop_debt & (first_crop_debt_day < 30),
        day[:, None] - first_crop_debt_day + 1,
        0,
    )
    encoded_crop_age = jnp.clip(crop_debt_age, 0, 127).astype(jnp.int8)

    priority = controller.projects.priority_class
    priority = priority.at[:, :NUM_CROPS].set(encoded_crop_age)
    priority = priority.at[
        :, NUM_CROPS : NUM_CROPS + NUM_ANIMALS
    ].set(encoded_age)
    projects = controller.projects._replace(priority_class=priority)
    return controller._replace(projects=projects)


def materialize_m36_unified_unit_tasks_v3(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M35FarmGenomeV2,
    calendar: RouteCalendarV3,
    player: int,
    *,
    enable_unlock_committed_capital: bool = False,
    enable_crop_debt_lane_repair: bool = False,
    allowed_service_tiles: jax.Array | None = None,
) -> tuple[ProjectControllerStateV2, M36SchedulerDiagnosticsV3]:
    """Integrated low-graph scheduler used by full-season rollouts.

    It scores crop and animal obligations once, partitions units dynamically,
    and then invokes each accepted sub-executor only for its assigned units.
    The proposal-level selector above remains a focused semantic test fixture;
    running both full sub-planners for every unit doubled the XLA graph without
    improving the executable task set.
    """

    if allowed_service_tiles is None:
        allowed, release_count = planned_service_tile_mask_v3(
            states, calendar, player
        )
    else:
        allowed = jnp.asarray(allowed_service_tiles, dtype=jnp.bool_)
        release_count = jnp.sum(
            (states.tile_animal[:, player] >= 0) & (~allowed),
            axis=(1, 2),
            dtype=jnp.int32,
        )
    unlock_mask = None
    unlock_count = jnp.zeros_like(states.step, dtype=jnp.int32)
    if enable_unlock_committed_capital:
        # Animal project slots are fixed at NUM_CROPS..NUM_CROPS+2 (5..7).
        # Daily planning stores a positive debt age only for value/deadline
        # viable species; the light kernel reads three bytes and counts only
        # physical unplaced animals.
        unlock_mask = controller.projects.priority_class[
            :, NUM_CROPS : NUM_CROPS + NUM_ANIMALS
        ] > 0
        in_shed = states.shed[
            :, player, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
        ].astype(jnp.int16)
        carried = jnp.sum(
            states.unit_inventory[
                :, player, :, NUM_PRODUCTS : NUM_PRODUCTS + NUM_ANIMALS
            ].astype(jnp.int16),
            axis=1,
            dtype=jnp.int16,
        )
        unplaced = in_shed + carried
        physical_unlock_count = jnp.sum(
            jnp.where(unlock_mask, unplaced, 0),
            axis=-1,
            dtype=jnp.int32,
        )
        max_debt_age = jnp.max(
            controller.projects.priority_class[
                :, NUM_CROPS : NUM_CROPS + NUM_ANIMALS
            ].astype(jnp.int32),
            axis=-1,
        )
        # A newly purchased mixed wave needs one independent placement lane
        # per species.  The old day-1 cap of one lane serialized cow and sheep
        # activation: one species reached pasture while the other remained in
        # the shed, delaying fertilizer cash and the next day's workforce.
        # This is project-level concurrency, not a Replay special case.  A
        # single older species may use a second catch-up lane; three species
        # are the official maximum, so the bound stays static and cheap.
        unplaced_species = jnp.sum(
            (unplaced > 0) & unlock_mask, axis=-1, dtype=jnp.int32
        )
        age_budget = jnp.where(max_debt_age <= 1, 1, 2).astype(jnp.int32)
        lane_budget = jnp.clip(
            jnp.maximum(unplaced_species, age_budget), 1, NUM_ANIMALS
        ).astype(jnp.int32)
        unlock_count = jnp.minimum(physical_unlock_count, lane_budget)
    tasks = controller.unit_tasks
    service_allowed = _service_allowed_for_tasks(tasks, allowed)
    clear_release = _task_present(tasks) & (~service_allowed)
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)
    raw_crop = states.tile_crop[:, player]
    raw_animal = states.tile_animal[:, player]
    raw_flags = states.tile_flags[:, player]
    same_day_water_due = (
        (raw_crop >= 0)
        & ((raw_flags & jnp.uint8(FLAG_WATERED)) == 0)
        & (states.tile_origin_day[:, player] == current_day[:, None, None])
    )
    survival_feed_raw, bonus_feed_raw, care_feed_raw = m3_feed_obligation_masks_v2(
        states, genome.animal, player
    )
    care_action_raw = m3_care_obligation_mask_v2(
        states, genome.animal, player
    ) & ((raw_flags & jnp.uint8(FLAG_CARED)) == 0)
    survival_feed_due = survival_feed_raw & allowed
    hard_due = jnp.any(
        same_day_water_due | survival_feed_due, axis=(1, 2)
    )
    returning = (tasks.phase == TaskPhaseV1.MOVE_TO_DEPOT) | (
        tasks.phase == TaskPhaseV1.DEPOSIT
    )
    carrying_leg = tasks.phase == TaskPhaseV1.MOVE_TO_TARGET
    operation_leg = tasks.phase == TaskPhaseV1.OPERATE
    unit_carries_resource = jnp.sum(
        states.unit_inventory[:, player].astype(jnp.int32), axis=-1
    ) > 0
    task_batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)[:, None]
    task_x = jnp.clip(tasks.target_x.astype(jnp.int32), 0, BOARD_SIZE - 1)
    task_y = jnp.clip(tasks.target_y.astype(jnp.int32), 0, BOARD_SIZE - 1)
    task_targets_survival_feed = survival_feed_due[
        task_batch, task_y, task_x
    ]
    existing_hard = (
        (tasks.task_type == TaskTypeV1.WATER_CROP)
        # A previously scheduled bonus/CARE feed is not evidence that the
        # *currently neglected* animal has a protected lane.  Treat only a
        # FEED whose target is in the live survival mask as hard; otherwise an
        # optional sticky feed can hide a real escape deadline.
        | (
            (tasks.task_type == TaskTypeV1.ANIMAL_FEED)
            & task_targets_survival_feed
        )
        | returning
        # MOVE_TO_TARGET follows a pickup for seed, feed, animal and product
        # routes.  Protecting that phase is a cheap exact proxy for “do not
        # abandon carried resources” and avoids rescanning the full inventory.
        | (
            carrying_leg
            & (
                unit_carries_resource
                if enable_unlock_committed_capital
                else jnp.ones_like(unit_carries_resource)
            )
        )
        # CROP_PRODUCTION changes to OPERATE immediately after PLANT.  Clearing
        # it here discards the mandatory same-day WATER continuation, which can
        # create an unwatered crop on the final turn.  Other operations are also
        # already at their irreversible execution point and must remain sticky.
        | operation_leg
        # A route card has already reserved its future targets and inventory.
        # Treat its current atomic stop as one sticky commitment; otherwise a
        # one-step preemption can strand the rest of the card.
        | (controller.route_cards.status != 0)
    )
    # Stickiness prevents ordinary churn, but it is not allowed to strand a
    # same-day water or survival-feed deadline.  Only non-hard work is cleared
    # and the normal nearest-task planners immediately reassign those lanes.
    active_hard = _task_present(tasks) & existing_hard
    need_one_preempt = hard_due & (~jnp.any(active_hard, axis=-1))
    preemptable = _task_present(tasks) & (~existing_hard)
    # Release exactly one least-valuable lane when the scheduler only needs one
    # foothold for an urgent task.  The previous vector mask cleared *every*
    # non-hard task in the batch row, repeatedly destroying otherwise feasible
    # crop routes.  Prefer optional animal service, then generic work, then a
    # crop production route, and preserve an already admitted BUILD/PLACE
    # pipeline until no cheaper lane remains.  Subsequent urgent obligations
    # can request another lane after the first one completes; hard work itself
    # remains sticky throughout.
    optional_service = preemptable & (
        (tasks.task_type == TaskTypeV1.ANIMAL_CARE)
        | (tasks.task_type == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
        | (
            (tasks.task_type == TaskTypeV1.ANIMAL_FEED)
            & (~task_targets_survival_feed)
        )
    )
    task_targets_same_day_water = same_day_water_due[
        task_batch, task_y, task_x
    ]
    hard_target_conflict = preemptable & (
        task_targets_survival_feed | task_targets_same_day_water
    )
    crop_route = preemptable & (
        tasks.task_type == TaskTypeV1.CROP_PRODUCTION
    )
    committed_pipeline = preemptable & (
        (tasks.task_type == TaskTypeV1.ANIMAL_PLACE)
        | (tasks.task_type == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
    )
    preempt_class = jnp.where(
        hard_target_conflict,
        0,
        jnp.where(
            optional_service,
            1,
            jnp.where(crop_route, 3, jnp.where(committed_pipeline, 4, 2)),
        ),
    ).astype(jnp.int32)
    progress_age = jnp.maximum(
        tasks.last_progress_step.astype(jnp.int32)
        - tasks.start_step.astype(jnp.int32),
        0,
    )
    unit_ordinal = jnp.arange(MAX_UNITS, dtype=jnp.int32)[None]
    preempt_score = jnp.where(
        preemptable,
        preempt_class * 1_000_000 + progress_age * MAX_UNITS + unit_ordinal,
        jnp.int32(2_000_000_000),
    )
    preempt_slot = jnp.argmin(preempt_score, axis=-1)
    deadline_preempt_task = (
        jax.nn.one_hot(preempt_slot, MAX_UNITS, dtype=jnp.bool_)
        & need_one_preempt[:, None]
        & jnp.any(preemptable, axis=-1)[:, None]
    )
    unlock_preempt_task = jnp.zeros_like(deadline_preempt_task)
    if enable_unlock_committed_capital:
        committed_pipeline = _task_present(tasks) & (
            (tasks.task_type == TaskTypeV1.ANIMAL_PLACE)
            | (tasks.task_type == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
        )
        unlock_preemptable = (
            _task_present(tasks)
            & (~existing_hard)
            & (~committed_pipeline)
            # Capital unlocking may defer optional animal service, but it must
            # never discard a crop production/harvest route or a product return
            # route.  Those tasks carry multi-step state that cannot be rebuilt
            # losslessly from the current frame alone.
            & (
                (tasks.task_type == TaskTypeV1.ANIMAL_CARE)
                | (tasks.task_type == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
            )
        )
        unlock_preempt_rank = jnp.cumsum(
            unlock_preemptable.astype(jnp.int16), axis=-1
        )
        already_unlocking = jnp.sum(
            committed_pipeline, axis=-1, dtype=jnp.int32
        )
        need_unlock_preempt = jnp.maximum(
            unlock_count - already_unlocking, 0
        )
        unlock_preempt_task = unlock_preemptable & (
            unlock_preempt_rank <= need_unlock_preempt[:, None]
        )
    empty = empty_unit_tasks_v2(states.step.shape[0])
    existing = jax.tree.map(
        lambda value, blank: jnp.where(
            clear_release | deadline_preempt_task | unlock_preempt_task, blank, value
        ),
        tasks,
        empty,
    )
    base = controller._replace(unit_tasks=existing)
    batch_size = states.step.shape[0]
    active_unit = states.unit_active[:, player]
    sticky = _task_present(existing) & active_unit
    animal_task = (
        (existing.task_type == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
        | (existing.task_type == TaskTypeV1.ANIMAL_PLACE)
        | (existing.task_type == TaskTypeV1.ANIMAL_FEED)
        | (existing.task_type == TaskTypeV1.ANIMAL_CARE)
        | (existing.task_type == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
        | (existing.task_type == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
    )
    sticky_animal = sticky & animal_task
    sticky_crop = sticky & (~animal_task)
    free = active_unit & (~sticky)

    positions = states.unit_pos[:, player].astype(jnp.int16)
    tile_x = jnp.tile(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
    tile_y = jnp.repeat(jnp.arange(BOARD_SIZE, dtype=jnp.int16), BOARD_SIZE)
    distance = (
        jnp.abs(positions[:, :, None, 0] - tile_x[None, None])
        + jnp.abs(positions[:, :, None, 1] - tile_y[None, None])
    ).astype(jnp.int16)
    crop = states.tile_crop[:, player].reshape(batch_size, -1)
    animal = states.tile_animal[:, player].reshape(batch_size, -1)
    flags = states.tile_flags[:, player].reshape(batch_size, -1)
    tile_yield = states.tile_yield[:, player].reshape(batch_size, -1)
    allowed_flat = allowed.reshape(batch_size, -1)
    crop_water = (crop >= 0) & ((flags & jnp.uint8(FLAG_WATERED)) == 0)
    crop_harvest = (crop >= 0) & (tile_yield > 0)
    survival_feed_flat = survival_feed_raw.reshape(batch_size, -1) & allowed_flat
    care_action = care_action_raw.reshape(batch_size, -1) & allowed_flat
    animal_feed = (
        survival_feed_flat
        | (bonus_feed_raw.reshape(batch_size, -1) & allowed_flat)
        | (care_feed_raw.reshape(batch_size, -1) & allowed_flat)
    )
    animal_collect = (animal >= 0) & (tile_yield > 0)
    animal_fertilizer = (
        (animal >= 0)
        & ((flags & jnp.uint8(FLAG_FERTILIZER_AVAILABLE)) != 0)
        & allowed_flat
        & (
            genome.animal.animal_fertilizer_policy[:, None]
            != M3AnimalFertilizerPolicyV2.IGNORE
        )
    )
    crop_tile_score = jnp.where(
        crop_water[:, None],
        distance,
        jnp.where(crop_harvest[:, None], distance + 100, 1_000_000),
    )
    animal_tile_score = jnp.where(
        animal_feed[:, None],
        distance,
        jnp.where(
            care_action[:, None],
            distance + 50,
            jnp.where(
                animal_collect[:, None],
                distance + 100,
                jnp.where(
                    # Start a fertilizer batch only when the worker is already
                    # standing on the animal tile after FEED/CARE.  Once one
                    # unit is carried, the explicit carry rule below keeps the
                    # same worker in the batching/deposit lane.
                    animal_fertilizer[:, None] & (distance == 0),
                    distance,
                    1_000_000,
                ),
            ),
        ),
    )
    crop_score = jnp.min(crop_tile_score, axis=-1)
    animal_score = jnp.min(animal_tile_score, axis=-1)
    _, crop_target, _, _ = m26_phase_targets_v2(states, genome.crop)
    # Daily Replay calendars record the intended operating envelope for each
    # day.  When tomorrow contracts a crop family, replacing today's harvested
    # plants back to the larger current target is pure churn: it consumes seed,
    # land and next-day watering capacity immediately before the planned
    # switch.  Anticipate contractions only; expansions remain scheduled for
    # their own day.  Existing crops are still fully serviced by M2.6.
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    calendar_day = jnp.clip(
        states.step.astype(jnp.int32) // TURNS_PER_DAY,
        0,
        calendar.crop_target_by_day.shape[1] - 1,
    )
    next_calendar_day = jnp.minimum(
        calendar_day + 1, calendar.crop_target_by_day.shape[1] - 1
    )
    next_crop_target = calendar.crop_target_by_day[
        batch, next_calendar_day
    ].astype(jnp.int16)
    crop_planning_target = jnp.minimum(crop_target, next_crop_target)
    _, animal_target, _, _ = m3_phase_targets_v2(states, genome.animal)
    has_crop = jnp.any(crop_target > 0, axis=-1) | jnp.any(crop >= 0, axis=-1)
    has_animal = jnp.any(animal_target > 0, axis=-1) | jnp.any(animal >= 0, axis=-1)
    crop_score = jnp.where(
        (crop_score >= 1_000_000) & has_crop[:, None], 500, crop_score
    )
    animal_score = jnp.where(
        (animal_score >= 1_000_000) & has_animal[:, None], 500, animal_score
    )
    choose_animal = free & has_animal[:, None] & (
        (~has_crop[:, None]) | (animal_score < crop_score)
    )
    choose_crop = free & has_crop[:, None] & (~choose_animal)
    carrying_fertilizer = (
        states.unit_inventory[:, player, :, NUM_PRODUCTS - 1] > 0
    )
    continue_fertilizer_batch = free & carrying_fertilizer
    choose_animal = choose_animal | continue_fertilizer_batch
    choose_crop = choose_crop & (~continue_fertilizer_batch)
    # Distance-only partitioning can under-allocate animal lanes on a busy day:
    # nearby crops win most workers even though every planned FEED must finish
    # before day end.  Survival receives one lane per animal.  Ordinary
    # production-bonus FEED and CARE are packed according to the remaining
    # per-lane service capacity, which prevents both late misses and the earlier
    # mistake of assigning every unit to animals all day.  The animal planner's
    # reservation maps those lanes to distinct tasks; other work continues to
    # use the cheaper distance comparison above.
    active_animal_service = sticky & (
        (existing.task_type == TaskTypeV1.ANIMAL_FEED)
        | (existing.task_type == TaskTypeV1.ANIMAL_CARE)
    )
    survival_count = jnp.sum(
        survival_feed_flat, axis=-1, dtype=jnp.int32
    )
    nonhard_service_count = jnp.sum(
        (animal_feed & (~survival_feed_flat)).astype(jnp.int32)
        + care_action.astype(jnp.int32),
        axis=-1,
        dtype=jnp.int32,
    )
    day_end = ((states.step // TURNS_PER_DAY) + 1) * TURNS_PER_DAY - 1
    remaining_turns = jnp.maximum(day_end - states.step + 1, 1).astype(jnp.int32)
    # A clustered route can normally complete up to six FEED/CARE operations
    # per lane over a full day.  Reduce that capacity as the deadline nears;
    # the last four turns therefore reserve one lane per remaining operation.
    service_capacity_per_lane = jnp.clip(
        remaining_turns // 4, 1, 6
    ).astype(jnp.int32)
    nonhard_lanes = (
        nonhard_service_count + service_capacity_per_lane - 1
    ) // service_capacity_per_lane
    required_service_lanes = survival_count + nonhard_lanes
    uncovered_feed = jnp.maximum(
        required_service_lanes
        - jnp.sum(active_animal_service, axis=-1, dtype=jnp.int32),
        0,
    )
    uncovered_feed = jnp.minimum(
        uncovered_feed,
        jnp.sum(free, axis=-1, dtype=jnp.int32),
    )
    unit_id = jnp.arange(MAX_UNITS, dtype=jnp.int32)[None]
    survival_lane_score = jnp.where(
        free,
        animal_score.astype(jnp.int32) * jnp.int32(MAX_UNITS) + unit_id,
        jnp.int32(2_000_000_000),
    )
    survival_lane_rank = jnp.sum(
        survival_lane_score[:, :, None] > survival_lane_score[:, None, :],
        axis=-1,
        dtype=jnp.int32,
    )
    force_survival_lane = free & (
        survival_lane_rank < uncovered_feed[:, None]
    )
    survival_lane_repair = force_survival_lane & (~choose_animal)
    choose_animal = choose_animal | force_survival_lane
    choose_crop = choose_crop & (~force_survival_lane)

    force_unlock_lane = jnp.zeros_like(free)
    if enable_unlock_committed_capital:
        sticky_unlock = sticky & (
            (existing.task_type == TaskTypeV1.ANIMAL_PLACE)
            | (existing.task_type == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
        )
        uncovered_unlock = jnp.maximum(
            unlock_count
            - jnp.sum(sticky_unlock, axis=-1, dtype=jnp.int32),
            0,
        )
        uncovered_unlock = jnp.minimum(
            uncovered_unlock,
            jnp.sum(free, axis=-1, dtype=jnp.int32),
        )
        # A static prefix keeps the light graph cheap.  Its dynamic length is
        # bounded by the three official animal species; exact target selection
        # still uses the existing distance-aware task planner.
        unlock_lane_rank = jnp.cumsum(
            free.astype(jnp.int16), axis=-1
        ).astype(jnp.int32) - 1
        force_unlock_lane = free & (
            unlock_lane_rank < uncovered_unlock[:, None]
        )
        choose_animal = choose_animal | force_unlock_lane
        choose_crop = choose_crop & (~force_unlock_lane)

        # Target debt is a real cross-day production commitment, not merely a
        # diagnostic.  Allocate enough non-sticky lanes to work it down while
        # preserving already reserved survival/service and capital-unlock
        # lanes.  A lane can usually complete up to three clustered plant
        # routes over a full day; as the deadline approaches this estimate
        # shrinks automatically and requests more parallel lanes.
        crop_active = jnp.stack(
            tuple(
                jnp.sum(crop == crop_id, axis=-1, dtype=jnp.int16)
                for crop_id in range(NUM_CROPS)
            ),
            axis=-1,
        )
        crop_debt_by_species = jnp.maximum(
            crop_target.astype(jnp.int16) - crop_active, 0
        )
        crop_debt_count = jnp.sum(
            crop_debt_by_species, axis=-1, dtype=jnp.int32
        )
        crop_debt_age = jnp.max(
            controller.projects.priority_class[:, :NUM_CROPS].astype(
                jnp.int32
            ),
            axis=-1,
        )
        plant_routes_per_lane = jnp.clip(
            remaining_turns // 8, 1, 3
        ).astype(jnp.int32)
        requested_crop_lanes = (
            crop_debt_count + plant_routes_per_lane - 1
        ) // plant_routes_per_lane
        protected_fresh = force_survival_lane | force_unlock_lane
        crop_lane_capacity = (
            jnp.sum(sticky_crop, axis=-1, dtype=jnp.int32)
            + jnp.sum(
                free & (~protected_fresh), axis=-1, dtype=jnp.int32
            )
        )
        desired_crop_lanes = jnp.minimum(
            requested_crop_lanes, crop_lane_capacity
        )
        current_crop_lanes = jnp.sum(
            sticky_crop | choose_crop, axis=-1, dtype=jnp.int32
        )
        uncovered_crop_lanes = jnp.maximum(
            desired_crop_lanes - current_crop_lanes, 0
        )
        debt_candidate = (
            free
            & (~protected_fresh)
            & (~choose_crop)
            & (crop_debt_age > 0)[:, None]
            # V8 showed that forcing a count of nominal crop lanes can starve
            # animal activation without completing more seed->plant->water
            # chains.  Keep that rejected experiment opt-in while M3.7C is
            # rebuilt around complete feasible route capacity.
            & jnp.asarray(enable_crop_debt_lane_repair, dtype=jnp.bool_)
        )
        debt_lane_score = jnp.where(
            debt_candidate,
            crop_score.astype(jnp.int32) * jnp.int32(MAX_UNITS)
            + unit_id,
            jnp.int32(2_000_000_000),
        )
        debt_lane_rank = jnp.sum(
            debt_lane_score[:, :, None] > debt_lane_score[:, None, :],
            axis=-1,
            dtype=jnp.int32,
        )
        force_crop_debt_lane = debt_candidate & (
            debt_lane_rank < uncovered_crop_lanes[:, None]
        )
        choose_crop = choose_crop | force_crop_debt_lane
        choose_animal = choose_animal & (~force_crop_debt_lane)

    # If both businesses are live and at least two units exist, reserve one
    # best-positioned lane for each side.  Animal is repaired first because an
    # equal score previously assigned every free unit to crop and left no lane
    # from which the animal repair could choose.  Both repairs may steal only
    # a fresh assignment; sticky work remains untouched.
    both = has_crop & has_animal
    need_animal = both & (~jnp.any(sticky_animal | choose_animal, axis=-1))
    free_animal_score = jnp.where(free, animal_score, 1_000_000)
    force_animal_slot = jnp.argmin(free_animal_score, axis=-1)
    force_animal = (
        jax.nn.one_hot(force_animal_slot, MAX_UNITS, dtype=jnp.bool_)
        & need_animal[:, None]
        & jnp.any(free, axis=-1)[:, None]
    )
    choose_animal = (choose_animal | force_animal) & (~sticky_crop)
    choose_crop = choose_crop & (~force_animal)
    need_crop = both & (~jnp.any(sticky_crop | choose_crop, axis=-1))
    free_after_animal = free & (~choose_animal)
    free_crop_score = jnp.where(free_after_animal, crop_score, 1_000_000)
    force_crop_slot = jnp.argmin(free_crop_score, axis=-1)
    force_crop = (
        jax.nn.one_hot(force_crop_slot, MAX_UNITS, dtype=jnp.bool_)
        & need_crop[:, None]
        & jnp.any(free_after_animal, axis=-1)[:, None]
    )
    choose_crop = (choose_crop | force_crop) & (~sticky_animal)
    choose_animal = choose_animal & (~force_crop)
    animal_mask = sticky_animal | choose_animal
    crop_mask = sticky_crop | choose_crop
    route_reserved_tiles = active_route_target_mask_v3(
        controller.route_cards
    )

    planned = materialize_m3_unit_tasks_v2(
        states,
        base,
        genome.animal,
        player,
        animal_mask,
        unlock_commitment_mask=unlock_mask,
        externally_reserved_tiles=route_reserved_tiles,
    )
    animal_tasks = planned.unit_tasks
    service_allowed = _service_allowed_for_tasks(animal_tasks, allowed)
    rejected_release = _task_present(animal_tasks) & (~service_allowed)
    after_animal = jax.tree.map(
        lambda value, blank: jnp.where(rejected_release, blank, value),
        animal_tasks,
        empty,
    )
    # The lightweight partition deliberately uses cheap upper-bound masks.  A
    # lane can therefore be offered to the animal planner even though the full
    # planner later rejects every candidate (for example a survival FEED with
    # no wheat, or sub-trigger product yield).  Do not strand that lane: once
    # the animal planner has emitted no task, offer the free worker to the crop
    # planner in the same step.  This is a generic sub-planner fallback rather
    # than another task-specific threshold copied into the partitioner.
    unused_animal_lane = choose_animal & (~_task_present(after_animal))
    # A rejected release task likewise frees that worker for crop work.
    crop_mask = crop_mask | rejected_release | unused_animal_lane
    planned = materialize_m26_unit_tasks_v2(
        states,
        planned._replace(unit_tasks=after_animal),
        genome.crop,
        player,
        crop_mask,
        planning_target=crop_planning_target,
        externally_reserved_tiles=route_reserved_tiles,
    )
    final_tasks = planned.unit_tasks
    final_active = _task_present(final_tasks)
    final_animal = final_active & (
        (final_tasks.task_type == TaskTypeV1.BUILD_ANIMAL_STRUCTURE)
        | (final_tasks.task_type == TaskTypeV1.ANIMAL_PLACE)
        | (final_tasks.task_type == TaskTypeV1.ANIMAL_FEED)
        | (final_tasks.task_type == TaskTypeV1.ANIMAL_CARE)
        | (final_tasks.task_type == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
        | (final_tasks.task_type == TaskTypeV1.ANIMAL_COLLECT_FERTILIZER)
    )
    hard_candidate = jnp.any(crop_water, axis=-1).astype(jnp.int32) + jnp.any(
        survival_feed_flat, axis=-1
    ).astype(jnp.int32)
    diagnostics = M36SchedulerDiagnosticsV3(
        sticky_task_count=jnp.sum(sticky, axis=-1, dtype=jnp.int32),
        crop_candidate_count=jnp.sum(
            free & has_crop[:, None], axis=-1, dtype=jnp.int32
        ),
        animal_candidate_count=jnp.sum(
            free & has_animal[:, None], axis=-1, dtype=jnp.int32
        ),
        hard_candidate_count=hard_candidate,
        selected_crop_count=jnp.sum(
            final_active & (~final_animal) & (~sticky), axis=-1, dtype=jnp.int32
        ),
        selected_animal_count=jnp.sum(
            final_animal & (~sticky), axis=-1, dtype=jnp.int32
        ),
        deadline_preemption_count=jnp.sum(
            choose_animal
            & (animal_score < 100)
            & (crop_score >= 100),
            axis=-1,
            dtype=jnp.int32,
        )
        + jnp.sum(survival_lane_repair, axis=-1, dtype=jnp.int32)
        + jnp.sum(deadline_preempt_task, axis=-1, dtype=jnp.int32),
        duplicate_reservation_prevented=jnp.sum(
            rejected_release, axis=-1, dtype=jnp.int32
        ),
        local_swap_count=jnp.sum(
            force_crop | force_animal, axis=-1, dtype=jnp.int32
        ),
        planned_release_tile_count=release_count,
    )
    return planned, diagnostics


__all__ = [
    "materialize_m36_unified_unit_tasks_v3",
    "planned_service_tile_mask_v3",
]
