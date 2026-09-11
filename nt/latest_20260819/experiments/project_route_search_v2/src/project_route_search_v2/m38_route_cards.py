"""Static-shape multi-stop route cards for the M3.7 farm controller.

The existing V5 primitive executor remains the only component that emits raw
official actions.  This module plans a bounded route, exposes exactly one
atomic ``UnitTaskStateV1`` stop to that executor, and advances the route only
after the primitive effect has been observed in the next official/JAX state.

The first accepted route families are deliberately narrow:

* ``FEED_TOUR``: one exact wheat pickup followed by several animal stops;
* ``CROP_HARVEST_TOUR``: several harvest stops, optionally renewing one-shot
  crops in place, followed by one return or the official day-end auto bank.

No Replay action or coordinate sequence is embedded here.  Targets are derived
from the current state, calendar and genome on every route admission.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_FIRST_YIELD_DAY,
    ANIMAL_INTERVAL,
    ANIMAL_MAX_HELD,
    ANIMAL_PRODUCT,
    BOARD_SIZE,
    CROP_ONGOING,
    EPISODE_STEPS,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_WATERED,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    TURNS_PER_DAY,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.types import State
from strategic_v5.constants import (
    FailureCodeV1,
    TaskPhaseV1,
    TaskStatusV1,
    TaskTypeV1,
)
from strategic_v5.e4_executor import E4PlayerActionV1
from strategic_v5.geometry import nearest_shed_access_v1

from .constants import MAX_ROUTE_CARD_STOPS_V3, M26FertilizerPolicyV2
from .lifecycle import empty_route_cards_v3, empty_unit_tasks_v2
from .m3_constants import M3AnimalFertilizerPolicyV2
from .m3_controller import (
    m3_care_obligation_mask_v2,
    m3_feed_obligation_masks_v2,
)
from .m26_controller import m26_phase_targets_v2
from .m35_schema import M35FarmGenomeV2
from .m36_schema import RouteCalendarV3
from .schema import ProjectControllerStateV2, RouteCardStateV3


class RouteCardTypeV3(IntEnum):
    NONE = 0
    FEED_TOUR = 1
    CROP_HARVEST_TOUR = 2
    ANIMAL_HARVEST_TOUR = 3
    WATER_TOUR = 4
    FERTILIZER_TOUR = 5
    PLACE_TOUR = 6
    CROP_FIELD_TOUR = 7
    MIXED_FARM_TOUR = 8


class RouteCardModeV4(IntEnum):
    """Static admission profiles used for honest route-card ablations."""

    LEGACY_M38 = 0
    ANIMAL_ONLY = 1
    CROP_ONLY = 2
    SEPARATE_CROP_ANIMAL = 3
    MIXED_WITH_FALLBACK = 4


class RouteCardStatusV3(IntEnum):
    EMPTY = 0
    ACTIVE = 1
    RETURNING = 2
    WAIT_AUTO_BANK = 3
    DONE = 4
    FAILED = 5


class RouteReturnModeV3(IntEnum):
    NONE = 0
    RETURN_AND_DROP = 1
    AUTO_BANK_AT_DAY_END = 2


class RouteActionV3(IntEnum):
    FEED = 1
    CARE = 2
    COLLECT_FERTILIZER = 4
    HARVEST = 8
    REPLANT_AND_WATER = 16
    PLANT_AND_WATER = 16
    WATER = 32
    FERTILIZE = 64
    ANIMAL_HARVEST = 128


class RouteCardDiagnosticsV3(NamedTuple):
    active_card_count: jax.Array
    created_feed_card_count: jax.Array
    created_harvest_card_count: jax.Array
    created_crop_field_card_count: jax.Array
    created_mixed_card_count: jax.Array
    created_stop_count: jax.Array
    reserved_target_count: jax.Array


_TILE_ID = jnp.arange(BOARD_SIZE * BOARD_SIZE, dtype=jnp.int32)
_TILE_X = (_TILE_ID % BOARD_SIZE).astype(jnp.int16)
_TILE_Y = (_TILE_ID // BOARD_SIZE).astype(jnp.int16)
_CROP_ONGOING = jnp.asarray(CROP_ONGOING, dtype=jnp.bool_)
_ANIMAL_FIRST = jnp.asarray(ANIMAL_FIRST_YIELD_DAY, dtype=jnp.int16)
_ANIMAL_INTERVAL = jnp.asarray(ANIMAL_INTERVAL, dtype=jnp.int16)
_ANIMAL_MAX = jnp.asarray(ANIMAL_MAX_HELD, dtype=jnp.int16)
_ANIMAL_PRODUCT = jnp.asarray(ANIMAL_PRODUCT, dtype=jnp.int8)
_INVALID_SCORE = jnp.int32(2_000_000_000)
_FERTILIZER_ITEM = NUM_PRODUCTS - 1


def _live_route(status: jax.Array) -> jax.Array:
    return (
        (status == RouteCardStatusV3.ACTIVE)
        | (status == RouteCardStatusV3.RETURNING)
        | (status == RouteCardStatusV3.WAIT_AUTO_BANK)
    )


def active_route_target_mask_v3(cards: RouteCardStateV3) -> jax.Array:
    """Return ``[batch, 100]`` reservation mask without a large one-hot."""

    batch_size = cards.status.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    unit = jnp.arange(MAX_UNITS, dtype=jnp.int32)[None]
    mask = jnp.zeros((batch_size, BOARD_SIZE * BOARD_SIZE), dtype=jnp.bool_)
    live = _live_route(cards.status)
    for stop in range(MAX_ROUTE_CARD_STOPS_V3):
        target = cards.target_ids[:, :, stop].astype(jnp.int32)
        valid = (
            live
            & (stop < cards.route_length)
            & (target >= 0)
            & (target < BOARD_SIZE * BOARD_SIZE)
        )
        safe = jnp.clip(target, 0, BOARD_SIZE * BOARD_SIZE - 1)
        mask = mask.at[batch, safe].max(valid)
    return mask


def _atomic_action_count(mask: jax.Array) -> jax.Array:
    total = jnp.zeros_like(mask, dtype=jnp.int16)
    for bit in (
        RouteActionV3.FEED,
        RouteActionV3.CARE,
        RouteActionV3.COLLECT_FERTILIZER,
        RouteActionV3.HARVEST,
        RouteActionV3.REPLANT_AND_WATER,
        RouteActionV3.WATER,
        RouteActionV3.FERTILIZE,
        RouteActionV3.ANIMAL_HARVEST,
    ):
        weight = 2 if bit == RouteActionV3.REPLANT_AND_WATER else 1
        total = total + jnp.where((mask & int(bit)) != 0, weight, 0).astype(
            jnp.int16
        )
    return total


def _ordered_route_prefix(
    candidate: jax.Array,
    candidate_items: jax.Array,
    candidate_actions: jax.Array,
    candidate_priority: jax.Array,
    start_position: jax.Array,
    base_cost: jax.Array,
    stop_budget: jax.Array,
    action_budget: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Nearest-neighbour route with an exact static prefix feasibility gate."""

    batch_size = candidate.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    targets = jnp.full(
        (batch_size, MAX_ROUTE_CARD_STOPS_V3), -1, dtype=jnp.int16
    )
    items = jnp.full(
        (batch_size, MAX_ROUTE_CARD_STOPS_V3), -1, dtype=jnp.int8
    )
    actions = jnp.zeros(
        (batch_size, MAX_ROUTE_CARD_STOPS_V3), dtype=jnp.uint8
    )
    available = candidate.astype(jnp.bool_)
    position = start_position.astype(jnp.int16)
    total_cost = base_cost.astype(jnp.int16)
    length = jnp.zeros((batch_size,), dtype=jnp.int8)

    for stop in range(MAX_ROUTE_CARD_STOPS_V3):
        distance = (
            jnp.abs(_TILE_X[None] - position[:, 0, None])
            + jnp.abs(_TILE_Y[None] - position[:, 1, None])
        ).astype(jnp.int16)
        score = jnp.where(
            available,
            candidate_priority.astype(jnp.int32) * 1_000_000
            + distance.astype(jnp.int32) * 100
            + _TILE_ID[None],
            _INVALID_SCORE,
        )
        target = jnp.argmin(score, axis=-1).astype(jnp.int32)
        has_target = jnp.min(score, axis=-1) < _INVALID_SCORE
        action_mask = candidate_actions[batch, target]
        increment = distance[batch, target] + _atomic_action_count(action_mask)
        take = (
            has_target
            & (stop < stop_budget)
            & (total_cost + increment <= action_budget)
        )
        targets = targets.at[:, stop].set(
            jnp.where(take, target, -1).astype(jnp.int16)
        )
        items = items.at[:, stop].set(
            jnp.where(take, candidate_items[batch, target], -1).astype(jnp.int8)
        )
        actions = actions.at[:, stop].set(
            jnp.where(take, action_mask, 0).astype(jnp.uint8)
        )
        total_cost = jnp.where(take, total_cost + increment, total_cost)
        length = length + take.astype(jnp.int8)
        position = jnp.where(
            take[:, None],
            jnp.stack((_TILE_X[target], _TILE_Y[target]), axis=-1),
            position,
        ).astype(jnp.int16)
        current = available[batch, target]
        available = available.at[batch, target].set(
            jnp.where(take, False, current)
        )
    return targets, items, actions, length, total_cost, position


def _limit_mask_by_distance(
    candidate: jax.Array,
    priority: jax.Array,
    position: jax.Array,
    limit: jax.Array,
) -> jax.Array:
    """Keep a deterministic nearest/priority prefix under a per-row limit."""

    distance = (
        jnp.abs(_TILE_X[None] - position[:, 0, None])
        + jnp.abs(_TILE_Y[None] - position[:, 1, None])
    ).astype(jnp.int32)
    score = jnp.where(
        candidate,
        priority.astype(jnp.int32) * 1_000_000
        + distance * 100
        + _TILE_ID[None],
        _INVALID_SCORE,
    )
    order = jnp.argsort(score, axis=-1)
    rank = jnp.argsort(order, axis=-1)
    return candidate & (rank < limit[:, None])


def _active_route_plant_reservations(
    cards: RouteCardStateV3,
) -> jax.Array:
    """Count remaining route-level seed commitments by crop."""

    batch_size = cards.status.shape[0]
    reserved = jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int16)
    live = _live_route(cards.status)
    for stop in range(MAX_ROUTE_CARD_STOPS_V3):
        item = cards.target_items[:, :, stop].astype(jnp.int32)
        action = cards.action_masks[:, :, stop]
        valid = (
            live
            & (stop >= cards.route_cursor)
            & (stop < cards.route_length)
            & ((action & int(RouteActionV3.PLANT_AND_WATER)) != 0)
            & (item >= 0)
            & (item < NUM_CROPS)
        )
        safe = jnp.clip(item, 0, NUM_CROPS - 1)
        for crop_id in range(NUM_CROPS):
            reserved = reserved.at[:, crop_id].add(
                jnp.sum(
                    valid & (safe == crop_id), axis=-1, dtype=jnp.int16
                )
            )
    return reserved


def _selected_action_count(
    actions: jax.Array,
    length: jax.Array,
    action: int,
) -> jax.Array:
    valid = (
        jnp.arange(MAX_ROUTE_CARD_STOPS_V3, dtype=jnp.int8)[None]
        < length[:, None]
    )
    return jnp.sum(
        valid & ((actions & int(action)) != 0), axis=-1, dtype=jnp.int16
    )


def _route_owner(
    states: State,
    controller: ProjectControllerStateV2,
    candidate: jax.Array,
    player: int,
    *,
    via_shed: bool,
) -> tuple[jax.Array, jax.Array]:
    active_unit = states.unit_active[:, player]
    free = (
        active_unit
        & (controller.unit_tasks.status != TaskStatusV1.ACTIVE)
        & (controller.route_cards.status == RouteCardStatusV3.EMPTY)
    )
    positions = states.unit_pos[:, player].astype(jnp.int16)
    distance = (
        jnp.abs(positions[:, :, None, 0] - _TILE_X[None, None])
        + jnp.abs(positions[:, :, None, 1] - _TILE_Y[None, None])
    ).astype(jnp.int16)
    nearest_candidate = jnp.min(
        jnp.where(candidate[:, None], distance, jnp.int16(30_000)), axis=-1
    )
    if via_shed:
        _, to_shed = nearest_shed_access_v1(positions)
        score = to_shed.astype(jnp.int32) * 1_000 + nearest_candidate
    else:
        score = nearest_candidate.astype(jnp.int32)
    ordinal = jnp.arange(MAX_UNITS, dtype=jnp.int32)[None]
    score = jnp.where(free, score * MAX_UNITS + ordinal, _INVALID_SCORE)
    owner = jnp.argmin(score, axis=-1).astype(jnp.int32)
    valid = (jnp.min(score, axis=-1) < _INVALID_SCORE) & jnp.any(candidate, axis=-1)
    return owner, valid


def _install_card(
    cards: RouteCardStateV3,
    create: jax.Array,
    owner: jax.Array,
    *,
    card_type: int,
    targets: jax.Array,
    items: jax.Array,
    actions: jax.Array,
    length: jax.Array,
    pickup_item: jax.Array,
    pickup_quantity: jax.Array,
    return_mode: jax.Array,
    states: State,
    deadline: jax.Array,
) -> RouteCardStateV3:
    batch_size = cards.status.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)

    def set_unit(field, value):
        old = field[batch, owner]
        mask = create.reshape((batch_size,) + (1,) * (old.ndim - 1))
        return field.at[batch, owner].set(jnp.where(mask, value, old))

    return cards._replace(
        card_type=set_unit(
            cards.card_type,
            jnp.full((batch_size,), card_type, dtype=jnp.int8),
        ),
        status=set_unit(
            cards.status,
            jnp.full(
                (batch_size,), RouteCardStatusV3.ACTIVE, dtype=jnp.int8
            ),
        ),
        target_ids=set_unit(cards.target_ids, targets),
        target_items=set_unit(cards.target_items, items),
        action_masks=set_unit(cards.action_masks, actions),
        route_length=set_unit(cards.route_length, length.astype(jnp.int8)),
        route_cursor=set_unit(
            cards.route_cursor, jnp.zeros((batch_size,), dtype=jnp.int8)
        ),
        pickup_item=set_unit(cards.pickup_item, pickup_item.astype(jnp.int8)),
        pickup_quantity=set_unit(
            cards.pickup_quantity, pickup_quantity.astype(jnp.int16)
        ),
        return_mode=set_unit(cards.return_mode, return_mode.astype(jnp.int8)),
        start_step=set_unit(cards.start_step, states.step.astype(jnp.int16)),
        deadline_step=set_unit(cards.deadline_step, deadline.astype(jnp.int16)),
        last_progress_step=set_unit(
            cards.last_progress_step, states.step.astype(jnp.int16)
        ),
        failure_code=set_unit(
            cards.failure_code, jnp.zeros((batch_size,), dtype=jnp.int8)
        ),
    )


def _clear_card_owners(
    cards: RouteCardStateV3,
    clear: jax.Array,
) -> RouteCardStateV3:
    """Clear selected ``[batch, unit]`` slots before replacing two routes."""

    blank = empty_route_cards_v3(cards.status.shape[0])
    return jax.tree.map(
        lambda value, blank_value: jnp.where(
            clear.reshape(clear.shape + (1,) * (value.ndim - clear.ndim)),
            blank_value,
            value,
        ),
        cards,
        blank,
    )


def _active_task_target_mask(
    controller: ProjectControllerStateV2,
) -> jax.Array:
    tasks = controller.unit_tasks
    batch_size = tasks.status.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    target = tasks.target_id.astype(jnp.int32)
    valid = (
        (tasks.status == TaskStatusV1.ACTIVE)
        & (target >= 0)
        & (target < BOARD_SIZE * BOARD_SIZE)
    )
    mask = jnp.zeros((batch_size, BOARD_SIZE * BOARD_SIZE), dtype=jnp.bool_)
    return mask.at[batch, jnp.clip(target, 0, BOARD_SIZE * BOARD_SIZE - 1)].max(
        valid
    )


def materialize_route_cards_v3(
    states: State,
    controller: ProjectControllerStateV2,
    genome: M35FarmGenomeV2,
    calendar: RouteCalendarV3,
    allowed_service_tiles: jax.Array,
    player: int,
    route_card_mode: int = RouteCardModeV4.LEGACY_M38,
) -> tuple[ProjectControllerStateV2, RouteCardDiagnosticsV3]:
    """Admit legacy, crop, separate, or mixed route-card profiles."""

    del calendar  # Calendar semantics are already compiled into ``genome``.
    mode = int(route_card_mode)
    enable_feed = mode in {
        int(RouteCardModeV4.LEGACY_M38),
        int(RouteCardModeV4.ANIMAL_ONLY),
        int(RouteCardModeV4.SEPARATE_CROP_ANIMAL),
        int(RouteCardModeV4.MIXED_WITH_FALLBACK),
    }
    enable_new_crop = mode in {
        int(RouteCardModeV4.CROP_ONLY),
        int(RouteCardModeV4.SEPARATE_CROP_ANIMAL),
        int(RouteCardModeV4.MIXED_WITH_FALLBACK),
    }
    enable_legacy_crop = mode == int(RouteCardModeV4.LEGACY_M38)
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    cards = controller.route_cards
    reserved = active_route_target_mask_v3(cards) | _active_task_target_mask(
        controller
    )
    flags = states.tile_flags[:, player].reshape(batch_size, -1)
    animal = states.tile_animal[:, player].reshape(batch_size, -1)
    survival, bonus, care_feed = m3_feed_obligation_masks_v2(
        states, genome.animal, player
    )
    feed_candidate = (
        (survival | bonus | care_feed).reshape(batch_size, -1)
        & allowed_service_tiles.reshape(batch_size, -1)
        & ((flags & jnp.uint8(FLAG_FED)) == 0)
        & (~reserved)
    )
    care_candidate = (
        m3_care_obligation_mask_v2(states, genome.animal, player).reshape(
            batch_size, -1
        )
        & ((flags & jnp.uint8(FLAG_CARED)) == 0)
    )
    fertilizer_policy = genome.animal.animal_fertilizer_policy[:, None]
    # Only policies whose semantics explicitly say "collect while visiting"
    # or "actively collect" are fused into a FEED route.  RESERVE_FOR_CROPS
    # needs the global crop/economic reserve calculation and therefore stays
    # with the existing joint scheduler instead of becoming an unconditional
    # same-tile action on every animal every day.
    collect_candidate = (
        ((flags & jnp.uint8(FLAG_FERTILIZER_AVAILABLE)) != 0)
        & (
            (fertilizer_policy == M3AnimalFertilizerPolicyV2.COLLECT_WHEN_VISITING_AND_SELL)
            | (fertilizer_policy == M3AnimalFertilizerPolicyV2.ACTIVE_COLLECT_AND_SELL)
        )
        & (
            states.step[:, None]
            < genome.animal.liquidation_start_step[:, None]
        )
    )
    feed_actions = jnp.full(
        feed_candidate.shape, int(RouteActionV3.FEED), dtype=jnp.uint8
    )
    safe_animal = jnp.clip(animal.astype(jnp.int32), 0, NUM_ANIMALS - 1)
    held_product = states.tile_yield[:, player].reshape(
        batch_size, -1
    ).astype(jnp.int16)
    pending_care = states.tile_pending_care[:, player].reshape(
        batch_size, -1
    ).astype(jnp.int16)
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)
    origin_day = states.tile_origin_day[:, player].reshape(
        batch_size, -1
    ).astype(jnp.int16)
    days_since_first = (
        current_day[:, None]
        + 1
        - origin_day
        - _ANIMAL_FIRST[safe_animal]
    )
    production_today = (
        (animal >= 0)
        & (days_since_first >= 0)
        & (
            jnp.mod(days_since_first, _ANIMAL_INTERVAL[safe_animal]) == 0
        )
    )
    already_fed = (flags & jnp.uint8(FLAG_FED)) != 0
    feed_required = feed_candidate
    expected_incoming = jnp.where(
        production_today,
        1 + jnp.where(already_fed | feed_required, pending_care, 0),
        0,
    ).astype(jnp.int16)
    capacity_loss = jnp.maximum(
        held_product + expected_incoming - _ANIMAL_MAX[safe_animal], 0
    )
    harvest_threshold = genome.animal.animal_harvest_trigger_units[
        batch[:, None], safe_animal
    ].astype(jnp.int16)
    terminal_harvest = (
        states.step[:, None]
        >= genome.animal.liquidation_start_step[:, None]
    ) & (held_product > 0)
    product_harvest_candidate = (
        feed_candidate
        & (held_product > 0)
        & (
            (capacity_loss > 0)
            | (held_product >= harvest_threshold)
            | terminal_harvest
        )
    )
    feed_actions = feed_actions | jnp.where(
        product_harvest_candidate,
        int(RouteActionV3.ANIMAL_HARVEST),
        0,
    ).astype(jnp.uint8)
    feed_actions = feed_actions | jnp.where(
        care_candidate, int(RouteActionV3.CARE), 0
    ).astype(jnp.uint8)
    feed_actions = feed_actions | jnp.where(
        collect_candidate, int(RouteActionV3.COLLECT_FERTILIZER), 0
    ).astype(jnp.uint8)
    feed_items = jnp.clip(animal, 0, 2).astype(jnp.int8)
    survival_flat = survival.reshape(batch_size, -1)
    bonus_flat = bonus.reshape(batch_size, -1)
    # Survival obligations dominate route compactness.  Without this tier a
    # nearest-neighbour tour could spend the whole day on nearby optional CARE
    # bonuses and leave one distant neglect==1 animal to escape at settlement.
    feed_priority = jnp.where(
        survival_flat,
        0,
        jnp.where(bonus_flat, 1, 2),
    ).astype(jnp.int8)

    feed_owner, feed_owner_valid = _route_owner(
        states, controller, feed_candidate, player, via_shed=True
    )
    owner_position = states.unit_pos[batch, player, feed_owner].astype(jnp.int16)
    carried_wheat = states.unit_inventory[
        batch, player, feed_owner, 0
    ].astype(jnp.int16)
    depot, depot_distance = nearest_shed_access_v1(owner_position)
    needs_pickup = carried_wheat <= 0
    start_position = jnp.where(needs_pickup[:, None], depot, owner_position)

    tasks = controller.unit_tasks
    unit_wheat = states.unit_inventory[:, player, :, 0].astype(jnp.int16)
    active_feed = (
        (tasks.status == TaskStatusV1.ACTIVE)
        & (tasks.task_type == TaskTypeV1.ANIMAL_FEED)
        & (unit_wheat <= 0)
    )
    task_pickup_reserved = jnp.sum(
        jnp.where(active_feed, jnp.maximum(tasks.quantity, 1), 0),
        axis=-1,
        dtype=jnp.int16,
    )
    live_route = _live_route(cards.status)
    route_pickup_pending = (
        live_route
        & (
            (cards.card_type == RouteCardTypeV3.FEED_TOUR)
            | (cards.card_type == RouteCardTypeV3.MIXED_FARM_TOUR)
        )
        & (unit_wheat <= 0)
        & (cards.route_cursor == 0)
    )
    route_pickup_reserved = jnp.sum(
        jnp.where(route_pickup_pending, cards.pickup_quantity, 0),
        axis=-1,
        dtype=jnp.int16,
    )
    shed_wheat = jnp.maximum(
        states.shed[:, player, 0].astype(jnp.int16)
        - task_pickup_reserved
        - route_pickup_reserved,
        0,
    )
    # A card has one explicit pickup phase at its beginning.  If the owner is
    # already carrying wheat, do not promise more feed stops than that carried
    # stock; otherwise later stops would silently run out after cursor 0.
    feed_stop_budget = jnp.where(
        needs_pickup,
        jnp.minimum(shed_wheat, jnp.int16(8)),
        jnp.minimum(carried_wheat, jnp.int16(8)),
    ).astype(jnp.int16)
    day_end = (
        ((states.step.astype(jnp.int32) // TURNS_PER_DAY) + 1)
        * TURNS_PER_DAY
        - 1
    ).astype(jnp.int16)
    # The primitive compiler suppresses maintenance operations on the final
    # transition of a day.  Reserve that transition for official day-end
    # settlement, exactly as the harvest route does below.  Counting it as an
    # executable slot admitted routes whose final COLLECT/CARE became PASS and
    # was then (correctly) diagnosed as a deadline miss.
    feed_action_budget = jnp.maximum(
        day_end - states.step, 0
    ).astype(jnp.int16)
    feed_base_cost = jnp.where(needs_pickup, depot_distance + 1, 0).astype(
        jnp.int16
    )
    (
        feed_targets,
        feed_route_items,
        feed_route_actions,
        feed_length,
        _,
        _,
    ) = _ordered_route_prefix(
        feed_candidate,
        feed_items,
        feed_actions,
        feed_priority,
        start_position,
        feed_base_cost,
        feed_stop_budget,
        feed_action_budget,
    )
    create_feed = enable_feed & feed_owner_valid & (feed_length >= 2)
    feed_pickup_quantity = jnp.where(
        needs_pickup, feed_length, 0
    ).astype(jnp.int16)
    cards = _install_card(
        cards,
        create_feed,
        feed_owner,
        card_type=RouteCardTypeV3.FEED_TOUR,
        targets=feed_targets,
        items=feed_route_items,
        actions=feed_route_actions,
        length=feed_length,
        pickup_item=jnp.where(create_feed, 0, -1),
        pickup_quantity=feed_pickup_quantity,
        return_mode=jnp.full(
            (batch_size,), RouteReturnModeV3.NONE, dtype=jnp.int8
        ),
        states=states,
        deadline=day_end,
    )
    controller = controller._replace(route_cards=cards)
    reserved = active_route_target_mask_v3(cards) | _active_task_target_mask(
        controller
    )

    # Crop routes use the same maturity/trigger contract as M2.6.  The legacy
    # M3.8 profile retains its old all-watered admission gate for historical
    # reproducibility.  New profiles instead encode WATER before HARVEST on
    # each stop, matching the current-gold Replay order.
    kind = states.tile_kind[:, player].reshape(batch_size, -1)
    crop = states.tile_crop[:, player].reshape(batch_size, -1)
    tile_yield = states.tile_yield[:, player].reshape(batch_size, -1)
    origin = states.tile_origin_day[:, player].reshape(batch_size, -1).astype(
        jnp.int16
    )
    safe_crop = jnp.clip(crop.astype(jnp.int32), 0, NUM_CROPS - 1)
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)
    min_age = genome.crop.harvest_min_age_days[batch[:, None], safe_crop]
    trigger = genome.crop.harvest_trigger_units[batch[:, None], safe_crop]
    age = current_day[:, None] - origin
    closing = states.step >= genome.crop.liquidation_start_step
    unwatered = (
        (kind == TileKind.PLANT)
        & ((flags & jnp.uint8(FLAG_WATERED)) == 0)
    )
    available_unwatered = unwatered & (~reserved)
    mature_candidate_base = (
        (kind == TileKind.PLANT)
        & (tile_yield > 0)
        & (age >= min_age)
        & ((tile_yield >= trigger) | closing[:, None])
        & (~reserved)
    )
    legacy_harvest_candidate = (
        mature_candidate_base
        & (~jnp.any(unwatered, axis=-1)[:, None])
    )
    current_crop_count = jnp.stack(
        tuple(
            jnp.sum(crop == crop_id, axis=-1, dtype=jnp.int16)
            for crop_id in range(NUM_CROPS)
        ),
        axis=-1,
    )
    _, crop_target, _, _ = m26_phase_targets_v2(states, genome.crop)
    renew_candidate = (
        (~_CROP_ONGOING[safe_crop])
        & (~closing[:, None])
        & (
            states.step[:, None]
            < genome.crop.crop_last_plant_step[batch[:, None], safe_crop]
        )
        & (
            crop_target[batch[:, None], safe_crop]
            >= current_crop_count[batch[:, None], safe_crop]
        )
    )

    if enable_new_crop:
        route_seed_reserved = _active_route_plant_reservations(cards)
        route_seed_available = jnp.maximum(
            states.seeds[:, player].astype(jnp.int16) - route_seed_reserved,
            0,
        )
        renew_by_crop = jnp.stack(
            tuple(
                jnp.sum(
                    mature_candidate_base
                    & renew_candidate
                    & (safe_crop == crop_id),
                    axis=-1,
                    dtype=jnp.int16,
                )
                for crop_id in range(NUM_CROPS)
            ),
            axis=-1,
        )
        new_seed_available = jnp.maximum(
            route_seed_available - renew_by_crop, 0
        )
        deficit = jnp.maximum(
            crop_target.astype(jnp.int16)
            - current_crop_count.astype(jnp.int16)
            - route_seed_reserved,
            0,
        )
        crop_slot = jnp.arange(NUM_CROPS, dtype=jnp.int32)[None]
        plant_quantity = jnp.minimum(deficit, new_seed_available)
        plant_score = jnp.where(
            plant_quantity > 0,
            plant_quantity.astype(jnp.int32) * 100 - crop_slot,
            -1,
        )
        chosen_plant_crop = jnp.argmax(plant_score, axis=-1).astype(jnp.int32)
        chosen_plant_quantity = plant_quantity[batch, chosen_plant_crop]
        remembered = controller.tile_project_id.reshape(batch_size, -1)
        crop_available_tile = (remembered < 0) | (remembered < NUM_CROPS)
        raw_plant_candidate = (
            (kind == TileKind.EMPTY)
            & crop_available_tile
            & (~reserved)
            & (jnp.max(plant_score, axis=-1) >= 0)[:, None]
            & (~closing[:, None])
            & (
                states.step[:, None]
                < genome.crop.crop_last_plant_step[
                    batch, chosen_plant_crop
                ][:, None]
            )
        )
        crop_candidate = (
            available_unwatered
            | mature_candidate_base
            | raw_plant_candidate
        )
        crop_owner, crop_owner_valid = _route_owner(
            states,
            controller._replace(route_cards=cards),
            crop_candidate,
            player,
            via_shed=False,
        )
        crop_position = states.unit_pos[
            batch, player, crop_owner
        ].astype(jnp.int16)
        plant_priority = jnp.full_like(kind, 3, dtype=jnp.int8)
        limited_plant_candidate = _limit_mask_by_distance(
            raw_plant_candidate,
            plant_priority,
            crop_position,
            chosen_plant_quantity.astype(jnp.int16),
        )
        mature_candidate = mature_candidate_base
        crop_candidate = (
            available_unwatered
            | mature_candidate
            | limited_plant_candidate
        )
        crop_items = jnp.where(
            kind == TileKind.PLANT,
            crop,
            chosen_plant_crop[:, None],
        ).astype(jnp.int8)
        crop_actions = jnp.zeros_like(kind, dtype=jnp.uint8)
        crop_actions = crop_actions | jnp.where(
            available_unwatered, int(RouteActionV3.WATER), 0
        ).astype(jnp.uint8)
        crop_actions = crop_actions | jnp.where(
            mature_candidate, int(RouteActionV3.HARVEST), 0
        ).astype(jnp.uint8)
        crop_actions = crop_actions | jnp.where(
            mature_candidate & renew_candidate,
            int(RouteActionV3.REPLANT_AND_WATER),
            0,
        ).astype(jnp.uint8)
        crop_actions = crop_actions | jnp.where(
            limited_plant_candidate,
            int(RouteActionV3.PLANT_AND_WATER),
            0,
        ).astype(jnp.uint8)

        fertilizer_mode = genome.crop.fertilizer_policy[
            batch[:, None], safe_crop
        ]
        fertilizer_enabled = (
            fertilizer_mode == M26FertilizerPolicyV2.ALWAYS_WHEN_AVAILABLE
        ) | (
            (fertilizer_mode == M26FertilizerPolicyV2.HIGH_VALUE_ONLY)
            & (safe_crop >= 2)
        )
        fertilized_until = states.tile_fertilized_until[:, player].reshape(
            batch_size, -1
        ).astype(jnp.int16)
        unit_fertilizer = states.unit_inventory[
            :, player, :, _FERTILIZER_ITEM
        ].astype(jnp.int16)
        active_apply = (
            (controller.unit_tasks.status == TaskStatusV1.ACTIVE)
            & (controller.unit_tasks.task_type == TaskTypeV1.APPLY_FERTILIZER)
        )
        reserved_apply = jnp.sum(
            active_apply & (unit_fertilizer <= 0), axis=-1, dtype=jnp.int16
        )
        fertilizer_available = jnp.maximum(
            states.shed[:, player, _FERTILIZER_ITEM].astype(jnp.int16)
            + jnp.sum(unit_fertilizer, axis=-1, dtype=jnp.int16)
            - reserved_apply,
            0,
        )
        # Gold crop chains apply fertilizer immediately before watering.  Do
        # not create fertilizer-only stops; this also keeps resource pruning
        # from producing an empty action mask.
        fertilize_desired = (
            available_unwatered
            & fertilizer_enabled
            & (fertilized_until <= current_day[:, None])
        )
        fertilize_candidate = fertilize_desired & (
            fertilizer_available > 0
        )[:, None]
        crop_actions = crop_actions | jnp.where(
            fertilize_candidate, int(RouteActionV3.FERTILIZE), 0
        ).astype(jnp.uint8)
        crop_priority = jnp.where(
            available_unwatered,
            0,
            jnp.where(mature_candidate, 1, 2),
        ).astype(jnp.int8)
        crop_budget = jnp.full((batch_size,), 12, dtype=jnp.int16)
    else:
        crop_candidate = legacy_harvest_candidate
        crop_items = crop.astype(jnp.int8)
        crop_actions = jnp.full(
            crop_candidate.shape,
            int(RouteActionV3.HARVEST),
            dtype=jnp.uint8,
        ) | jnp.where(
            renew_candidate,
            int(RouteActionV3.REPLANT_AND_WATER),
            0,
        ).astype(jnp.uint8)
        crop_priority = jnp.zeros_like(crop, dtype=jnp.int8)
        crop_owner, crop_owner_valid = _route_owner(
            states,
            controller._replace(route_cards=cards),
            crop_candidate,
            player,
            via_shed=False,
        )
        crop_position = states.unit_pos[
            batch, player, crop_owner
        ].astype(jnp.int16)
        crop_budget = jnp.full((batch_size,), 8, dtype=jnp.int16)

    if not (enable_new_crop or enable_legacy_crop):
        crop_candidate = jnp.zeros_like(kind, dtype=jnp.bool_)
        crop_owner_valid = jnp.zeros((batch_size,), dtype=jnp.bool_)

    crop_owner, crop_owner_valid_now = _route_owner(
        states,
        controller._replace(route_cards=cards),
        crop_candidate,
        player,
        via_shed=False,
    )
    crop_owner_valid = crop_owner_valid & crop_owner_valid_now
    crop_position = states.unit_pos[
        batch, player, crop_owner
    ].astype(jnp.int16)
    crop_action_budget = jnp.maximum(
        day_end - states.step, 0
    ).astype(jnp.int16)
    (
        crop_targets,
        crop_route_items,
        crop_route_actions,
        crop_length,
        crop_cost,
        crop_end_position,
    ) = _ordered_route_prefix(
        crop_candidate,
        crop_items,
        crop_actions,
        crop_priority,
        crop_position,
        jnp.zeros((batch_size,), dtype=jnp.int16),
        crop_budget,
        crop_action_budget,
    )
    # Do not promise more renewal seeds than physically exist.  The route may
    # still harvest all selected stops; only the optional renewal bit is
    # removed when its per-crop prefix exceeds current stock.
    stop_valid = (
        jnp.arange(MAX_ROUTE_CARD_STOPS_V3, dtype=jnp.int8)[None]
        < crop_length[:, None]
    )
    for crop_id in range(NUM_CROPS):
        wants = (
            stop_valid
            & (crop_route_items == crop_id)
            & (
                (crop_route_actions & int(RouteActionV3.REPLANT_AND_WATER))
                != 0
            )
        )
        rank = jnp.cumsum(wants.astype(jnp.int16), axis=-1)
        allowed = rank <= jnp.maximum(
            states.seeds[:, player, crop_id].astype(jnp.int16)
            - route_seed_reserved[:, crop_id]
            if enable_new_crop
            else states.seeds[:, player, crop_id].astype(jnp.int16),
            0,
        )[:, None]
        remove = wants & (~allowed)
        crop_route_actions = jnp.where(
            remove,
            crop_route_actions
            & jnp.uint8(255 - int(RouteActionV3.REPLANT_AND_WATER)),
            crop_route_actions,
        ).astype(jnp.uint8)

    if enable_new_crop:
        wants_fertilizer = stop_valid & (
            (crop_route_actions & int(RouteActionV3.FERTILIZE)) != 0
        )
        fertilizer_rank = jnp.cumsum(
            wants_fertilizer.astype(jnp.int16), axis=-1
        )
        crop_route_actions = jnp.where(
            wants_fertilizer
            & (fertilizer_rank > fertilizer_available[:, None]),
            crop_route_actions
            & jnp.uint8(255 - int(RouteActionV3.FERTILIZE)),
            crop_route_actions,
        ).astype(jnp.uint8)

    _, return_distance = nearest_shed_access_v1(crop_end_position)
    can_return = (
        crop_cost + return_distance + 1 <= crop_action_budget
    )
    route_has_harvest = _selected_action_count(
        crop_route_actions, crop_length, RouteActionV3.HARVEST
    ) > 0
    return_mode = jnp.where(
        route_has_harvest,
        jnp.where(
            can_return,
            RouteReturnModeV3.RETURN_AND_DROP,
            RouteReturnModeV3.AUTO_BANK_AT_DAY_END,
        ),
        RouteReturnModeV3.NONE,
    ).astype(jnp.int8)
    create_harvest = crop_owner_valid & (crop_length >= 2)
    cards = _install_card(
        cards,
        create_harvest,
        crop_owner,
        card_type=(
            RouteCardTypeV3.CROP_FIELD_TOUR
            if enable_new_crop
            else RouteCardTypeV3.CROP_HARVEST_TOUR
        ),
        targets=crop_targets,
        items=crop_route_items,
        actions=crop_route_actions,
        length=crop_length,
        pickup_item=jnp.full((batch_size,), -1, dtype=jnp.int8),
        pickup_quantity=jnp.zeros((batch_size,), dtype=jnp.int16),
        return_mode=return_mode,
        states=states,
        deadline=day_end,
    )

    create_mixed = jnp.zeros((batch_size,), dtype=jnp.bool_)
    mixed_length = jnp.zeros((batch_size,), dtype=jnp.int8)
    if mode == int(RouteCardModeV4.MIXED_WITH_FALLBACK):
        # Re-plan the selected animal and crop prefixes as one bounded route.
        # This is not concatenation: nearest-neighbour ordering and the shared
        # day budget are recomputed for the actual owner, so the merged card
        # cannot inherit two independently feasible but jointly impossible
        # schedules.
        mixed_candidate = jnp.zeros(
            (batch_size, BOARD_SIZE * BOARD_SIZE), dtype=jnp.bool_
        )
        mixed_items = jnp.full(
            (batch_size, BOARD_SIZE * BOARD_SIZE), -1, dtype=jnp.int8
        )
        mixed_actions = jnp.zeros(
            (batch_size, BOARD_SIZE * BOARD_SIZE), dtype=jnp.uint8
        )
        mixed_priority = jnp.full(
            (batch_size, BOARD_SIZE * BOARD_SIZE), 7, dtype=jnp.int8
        )
        # Keep the animal policy's own fertilizer collection distinct from
        # opportunistic collection added only to fund crop fertilization.  The
        # latter must be pruned if no selected crop stop can consume it;
        # otherwise a mixed card silently spends unit actions carrying unused
        # fertilizer to the end of the route.
        mixed_policy_collect_required = jnp.zeros(
            (batch_size, BOARD_SIZE * BOARD_SIZE), dtype=jnp.bool_
        )
        has_crop_fertilizer_demand = jnp.any(
            fertilize_desired & crop_candidate, axis=-1
        )

        for stop in range(MAX_ROUTE_CARD_STOPS_V3):
            target = feed_targets[:, stop].astype(jnp.int32)
            safe = jnp.clip(target, 0, BOARD_SIZE * BOARD_SIZE - 1)
            valid = (stop < feed_length) & (target >= 0)
            old_candidate = mixed_candidate[batch, safe]
            old_item = mixed_items[batch, safe]
            old_action = mixed_actions[batch, safe]
            old_priority = mixed_priority[batch, safe]
            action = feed_route_actions[:, stop]
            policy_collect_required = valid & (
                (action & int(RouteActionV3.COLLECT_FERTILIZER)) != 0
            )
            old_policy_collect_required = mixed_policy_collect_required[
                batch, safe
            ]
            can_collect_for_crop = (
                has_crop_fertilizer_demand
                & ((flags[batch, safe] & jnp.uint8(FLAG_FERTILIZER_AVAILABLE)) != 0)
                & (
                    states.step
                    < genome.animal.liquidation_start_step
                )
            )
            action = action | jnp.where(
                can_collect_for_crop,
                int(RouteActionV3.COLLECT_FERTILIZER),
                0,
            ).astype(jnp.uint8)
            mixed_candidate = mixed_candidate.at[batch, safe].set(
                old_candidate | valid
            )
            mixed_items = mixed_items.at[batch, safe].set(
                jnp.where(valid, feed_route_items[:, stop], old_item)
            )
            mixed_actions = mixed_actions.at[batch, safe].set(
                jnp.where(valid, action, old_action)
            )
            mixed_priority = mixed_priority.at[batch, safe].set(
                jnp.where(
                    valid,
                    feed_priority[batch, safe],
                    old_priority,
                )
            )
            mixed_policy_collect_required = (
                mixed_policy_collect_required.at[batch, safe].set(
                    old_policy_collect_required | policy_collect_required
                )
            )

        for stop in range(MAX_ROUTE_CARD_STOPS_V3):
            target = crop_targets[:, stop].astype(jnp.int32)
            safe = jnp.clip(target, 0, BOARD_SIZE * BOARD_SIZE - 1)
            valid = (stop < crop_length) & (target >= 0)
            old_candidate = mixed_candidate[batch, safe]
            old_item = mixed_items[batch, safe]
            old_action = mixed_actions[batch, safe]
            old_priority = mixed_priority[batch, safe]
            action = crop_route_actions[:, stop]
            action = action | jnp.where(
                fertilize_desired[batch, safe],
                int(RouteActionV3.FERTILIZE),
                0,
            ).astype(jnp.uint8)
            mixed_candidate = mixed_candidate.at[batch, safe].set(
                old_candidate | valid
            )
            mixed_items = mixed_items.at[batch, safe].set(
                jnp.where(valid, crop_route_items[:, stop], old_item)
            )
            mixed_actions = mixed_actions.at[batch, safe].set(
                jnp.where(valid, action, old_action)
            )
            mixed_priority = mixed_priority.at[batch, safe].set(
                jnp.where(
                    valid,
                    crop_priority[batch, safe] + jnp.int8(3),
                    old_priority,
                )
            )

        (
            mixed_targets,
            mixed_route_items,
            mixed_route_actions,
            mixed_length,
            mixed_cost,
            mixed_end_position,
        ) = _ordered_route_prefix(
            mixed_candidate,
            mixed_items,
            mixed_actions,
            mixed_priority,
            start_position,
            feed_base_cost,
            jnp.full((batch_size,), MAX_ROUTE_CARD_STOPS_V3, dtype=jnp.int16),
            feed_action_budget,
        )

        # Fertilizer is a prefix resource.  The mixed owner may use fertilizer
        # already carried, then gains one unit after each confirmed animal
        # collection.  A crop FERTILIZE bit is removed if that prefix has not
        # earned enough stock yet; no future collection is borrowed backward.
        initial_fertilizer_balance = states.unit_inventory[
            batch, player, feed_owner, _FERTILIZER_ITEM
        ].astype(jnp.int16)
        fertilizer_balance = initial_fertilizer_balance
        mixed_stop_valid = (
            jnp.arange(MAX_ROUTE_CARD_STOPS_V3, dtype=jnp.int8)[None]
            < mixed_length[:, None]
        )
        for stop in range(MAX_ROUTE_CARD_STOPS_V3):
            action = mixed_route_actions[:, stop]
            collected = mixed_stop_valid[:, stop] & (
                (action & int(RouteActionV3.COLLECT_FERTILIZER)) != 0
            )
            wants_fertilize = mixed_stop_valid[:, stop] & (
                (action & int(RouteActionV3.FERTILIZE)) != 0
            )
            can_fertilize = wants_fertilize & (
                fertilizer_balance + collected.astype(jnp.int16) > 0
            )
            remove = wants_fertilize & (~can_fertilize)
            mixed_route_actions = mixed_route_actions.at[:, stop].set(
                jnp.where(
                    remove,
                    action & jnp.uint8(255 - int(RouteActionV3.FERTILIZE)),
                    action,
                ).astype(jnp.uint8)
            )
            fertilizer_balance = (
                fertilizer_balance
                + collected.astype(jnp.int16)
                - can_fertilize.astype(jnp.int16)
            )

        selected_policy_collect_required = jnp.zeros(
            (batch_size, MAX_ROUTE_CARD_STOPS_V3), dtype=jnp.bool_
        )
        for stop in range(MAX_ROUTE_CARD_STOPS_V3):
            target = mixed_targets[:, stop].astype(jnp.int32)
            safe = jnp.clip(target, 0, BOARD_SIZE * BOARD_SIZE - 1)
            selected_policy_collect_required = (
                selected_policy_collect_required.at[:, stop].set(
                    mixed_stop_valid[:, stop]
                    & (target >= 0)
                    & mixed_policy_collect_required[batch, safe]
                )
            )

        # Animal candidates have priorities 0..2 while crop candidates have
        # priorities 3..5, so every selected collection precedes every crop
        # fertilization.  This lets one prefix rank retain the exact number of
        # optional collections required, without a quadratic stop-by-stop
        # look-ahead in the compiled GPU graph.
        selected_collect = mixed_stop_valid & (
            (mixed_route_actions & int(RouteActionV3.COLLECT_FERTILIZER)) != 0
        )
        selected_fertilize_count = jnp.sum(
            mixed_stop_valid
            & ((mixed_route_actions & int(RouteActionV3.FERTILIZE)) != 0),
            axis=-1,
            dtype=jnp.int16,
        )
        selected_policy_collect_count = jnp.sum(
            selected_collect & selected_policy_collect_required,
            axis=-1,
            dtype=jnp.int16,
        )
        optional_collect = selected_collect & (
            ~selected_policy_collect_required
        )
        optional_collect_needed = jnp.maximum(
            selected_fertilize_count
            - initial_fertilizer_balance
            - selected_policy_collect_count,
            0,
        )
        optional_collect_rank = jnp.cumsum(
            optional_collect.astype(jnp.int16), axis=-1
        )
        remove_optional = optional_collect & (
            optional_collect_rank > optional_collect_needed[:, None]
        )
        mixed_route_actions = jnp.where(
            remove_optional,
            mixed_route_actions
            & jnp.uint8(255 - int(RouteActionV3.COLLECT_FERTILIZER)),
            mixed_route_actions,
        ).astype(jnp.uint8)

        mixed_animal_count = _selected_action_count(
            mixed_route_actions, mixed_length, RouteActionV3.FEED
        )
        mixed_crop_count = (
            _selected_action_count(
                mixed_route_actions, mixed_length, RouteActionV3.WATER
            )
            + _selected_action_count(
                mixed_route_actions, mixed_length, RouteActionV3.HARVEST
            )
            + _selected_action_count(
                mixed_route_actions,
                mixed_length,
                RouteActionV3.REPLANT_AND_WATER,
            )
        )
        create_mixed = (
            feed_owner_valid
            & crop_owner_valid
            & (mixed_animal_count > 0)
            & (mixed_crop_count > 0)
        )
        _, mixed_return_distance = nearest_shed_access_v1(mixed_end_position)
        mixed_can_return = (
            mixed_cost + mixed_return_distance + 1 <= feed_action_budget
        )
        mixed_has_harvest = (
            _selected_action_count(
                mixed_route_actions, mixed_length, RouteActionV3.HARVEST
            )
            + _selected_action_count(
                mixed_route_actions,
                mixed_length,
                RouteActionV3.ANIMAL_HARVEST,
            )
        ) > 0
        mixed_return_mode = jnp.where(
            mixed_has_harvest,
            jnp.where(
                mixed_can_return,
                RouteReturnModeV3.RETURN_AND_DROP,
                RouteReturnModeV3.AUTO_BANK_AT_DAY_END,
            ),
            RouteReturnModeV3.NONE,
        ).astype(jnp.int8)
        units = jnp.arange(MAX_UNITS, dtype=jnp.int32)[None]
        clear = create_mixed[:, None] & (
            (units == feed_owner[:, None]) | (units == crop_owner[:, None])
        )
        cards = _clear_card_owners(cards, clear)
        mixed_feed_count = _selected_action_count(
            mixed_route_actions, mixed_length, RouteActionV3.FEED
        )
        cards = _install_card(
            cards,
            create_mixed,
            feed_owner,
            card_type=RouteCardTypeV3.MIXED_FARM_TOUR,
            targets=mixed_targets,
            items=mixed_route_items,
            actions=mixed_route_actions,
            length=mixed_length,
            pickup_item=jnp.where(create_mixed & needs_pickup, 0, -1),
            pickup_quantity=jnp.where(
                create_mixed & needs_pickup, mixed_feed_count, 0
            ).astype(jnp.int16),
            return_mode=mixed_return_mode,
            states=states,
            deadline=day_end,
        )

    controller = controller._replace(route_cards=cards)
    active = _live_route(cards.status)
    separate_feed = create_feed & (~create_mixed)
    separate_crop = create_harvest & (~create_mixed)
    diagnostics = RouteCardDiagnosticsV3(
        active_card_count=jnp.sum(active, axis=-1, dtype=jnp.int32),
        created_feed_card_count=separate_feed.astype(jnp.int32),
        created_harvest_card_count=(
            separate_crop & enable_legacy_crop
        ).astype(jnp.int32),
        created_crop_field_card_count=(
            separate_crop & enable_new_crop
        ).astype(jnp.int32),
        created_mixed_card_count=create_mixed.astype(jnp.int32),
        created_stop_count=(
            jnp.where(separate_feed, feed_length, 0).astype(jnp.int32)
            + jnp.where(separate_crop, crop_length, 0).astype(jnp.int32)
            + jnp.where(create_mixed, mixed_length, 0).astype(jnp.int32)
        ),
        reserved_target_count=jnp.sum(
            active_route_target_mask_v3(cards), axis=-1, dtype=jnp.int32
        ),
    )
    return controller, diagnostics


def materialize_route_card_unit_tasks_v3(
    states: State,
    controller: ProjectControllerStateV2,
    player: int,
) -> ProjectControllerStateV2:
    """Expose one current route operation per owner to the primitive compiler."""

    cards = controller.route_cards
    tasks = controller.unit_tasks
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    unit = jnp.arange(MAX_UNITS, dtype=jnp.int32)[None]
    cursor = jnp.clip(
        cards.route_cursor.astype(jnp.int32), 0, MAX_ROUTE_CARD_STOPS_V3 - 1
    )
    target = jnp.take_along_axis(
        cards.target_ids, cursor[..., None], axis=-1
    )[..., 0]
    item = jnp.take_along_axis(
        cards.target_items, cursor[..., None], axis=-1
    )[..., 0]
    action_mask = jnp.take_along_axis(
        cards.action_masks, cursor[..., None], axis=-1
    )[..., 0]
    x = (jnp.clip(target, 0, BOARD_SIZE * BOARD_SIZE - 1) % BOARD_SIZE).astype(
        jnp.int8
    )
    y = (jnp.clip(target, 0, BOARD_SIZE * BOARD_SIZE - 1) // BOARD_SIZE).astype(
        jnp.int8
    )
    active = cards.status == RouteCardStatusV3.ACTIVE
    returning = cards.status == RouteCardStatusV3.RETURNING
    waiting = cards.status == RouteCardStatusV3.WAIT_AUTO_BANK

    wants_feed = active & ((action_mask & int(RouteActionV3.FEED)) != 0)
    wants_animal_harvest = (
        active
        & (~wants_feed)
        & ((action_mask & int(RouteActionV3.ANIMAL_HARVEST)) != 0)
    )
    wants_care = (
        active
        & (~wants_feed)
        & (~wants_animal_harvest)
        & ((action_mask & int(RouteActionV3.CARE)) != 0)
    )
    wants_collect = (
        active
        & (~wants_feed)
        & (~wants_animal_harvest)
        & (~wants_care)
        & ((action_mask & int(RouteActionV3.COLLECT_FERTILIZER)) != 0)
    )
    wants_fertilize = (
        active
        & (~wants_feed)
        & (~wants_animal_harvest)
        & (~wants_care)
        & (~wants_collect)
        & ((action_mask & int(RouteActionV3.FERTILIZE)) != 0)
    )
    wants_water = (
        active
        & (~wants_feed)
        & (~wants_animal_harvest)
        & (~wants_care)
        & (~wants_collect)
        & (~wants_fertilize)
        & ((action_mask & int(RouteActionV3.WATER)) != 0)
    )
    wants_harvest = (
        active
        & (~wants_feed)
        & (~wants_animal_harvest)
        & (~wants_care)
        & (~wants_collect)
        & (~wants_fertilize)
        & (~wants_water)
        & ((action_mask & int(RouteActionV3.HARVEST)) != 0)
    )
    wants_renew = (
        active
        & (~wants_feed)
        & (~wants_animal_harvest)
        & (~wants_care)
        & (~wants_collect)
        & (~wants_fertilize)
        & (~wants_water)
        & (~wants_harvest)
        & ((action_mask & int(RouteActionV3.REPLANT_AND_WATER)) != 0)
    )
    task_type = jnp.where(
        wants_feed,
        TaskTypeV1.ANIMAL_FEED,
        jnp.where(
            wants_animal_harvest,
            TaskTypeV1.ANIMAL_COLLECT_PRODUCT,
            jnp.where(
                wants_care,
                TaskTypeV1.ANIMAL_CARE,
                jnp.where(
                wants_collect,
                TaskTypeV1.ANIMAL_COLLECT_FERTILIZER,
                jnp.where(
                    wants_fertilize,
                    TaskTypeV1.APPLY_FERTILIZER,
                    jnp.where(
                        wants_water,
                        TaskTypeV1.WATER_CROP,
                        jnp.where(
                            wants_harvest | wants_renew,
                            TaskTypeV1.CROP_PRODUCTION,
                            jnp.where(
                                returning,
                                TaskTypeV1.SHED_DEPOSIT,
                                TaskTypeV1.IDLE_OR_PASS,
                            ),
                        ),
                    ),
                ),
                ),
            ),
        ),
    ).astype(jnp.int8)
    task_item = jnp.where(
        wants_feed,
        0,
        jnp.where(
            wants_animal_harvest,
            _ANIMAL_PRODUCT[jnp.clip(item.astype(jnp.int32), 0, NUM_ANIMALS - 1)],
            jnp.where(
            wants_collect,
            _FERTILIZER_ITEM,
            jnp.where(wants_fertilize, _FERTILIZER_ITEM, item),
            ),
        ),
    ).astype(jnp.int8)
    position = states.unit_pos[:, player].astype(jnp.int8)
    depot, depot_distance = nearest_shed_access_v1(position)
    task_target_x = jnp.where(returning, depot[..., 0], x)
    task_target_y = jnp.where(returning, depot[..., 1], y)
    task_target_id = jnp.where(
        returning,
        depot[..., 1].astype(jnp.int16) * BOARD_SIZE
        + depot[..., 0].astype(jnp.int16),
        target,
    ).astype(jnp.int16)
    unit_wheat = states.unit_inventory[:, player, :, 0]
    pickup_pending = (
        wants_feed
        & (cards.route_cursor == 0)
        & (cards.pickup_quantity > 0)
        & (unit_wheat <= 0)
    )
    phase = jnp.where(
        returning,
        TaskPhaseV1.MOVE_TO_DEPOT,
        jnp.where(
            pickup_pending,
            TaskPhaseV1.MOVE_TO_SHED,
            TaskPhaseV1.MOVE_TO_TARGET,
        ),
    ).astype(jnp.int8)
    quantity = jnp.where(
        pickup_pending, cards.pickup_quantity, 1
    ).astype(jnp.int16)
    route_owned = active | returning | waiting
    distance = jnp.where(
        returning,
        depot_distance,
        jnp.abs(position[..., 0].astype(jnp.int16) - task_target_x)
        + jnp.abs(position[..., 1].astype(jnp.int16) - task_target_y),
    ).astype(jnp.int16)
    desired = empty_unit_tasks_v2(batch_size)._replace(
        task_type=task_type,
        owner_unit=jnp.broadcast_to(
            jnp.arange(MAX_UNITS, dtype=jnp.int8)[None], task_type.shape
        ),
        target_id=task_target_id,
        target_x=task_target_x.astype(jnp.int8),
        target_y=task_target_y.astype(jnp.int8),
        item_id=task_item,
        quantity=quantity,
        phase=phase,
        start_step=jnp.broadcast_to(
            states.step.astype(jnp.int16)[:, None], task_type.shape
        ),
        last_progress_step=jnp.broadcast_to(
            states.step.astype(jnp.int16)[:, None], task_type.shape
        ),
        expected_finish_step=(states.step[:, None] + distance + 1).astype(
            jnp.int16
        ),
        deadline_step=cards.deadline_step,
        status=jnp.where(
            route_owned, TaskStatusV1.ACTIVE, TaskStatusV1.EMPTY
        ).astype(jnp.int8),
        failure_code=jnp.zeros_like(task_type, dtype=jnp.int8),
    )
    same_atomic = (
        route_owned
        & (tasks.status == TaskStatusV1.ACTIVE)
        & (tasks.task_type == desired.task_type)
        & (tasks.target_id == desired.target_id)
    )
    unit_tasks = jax.tree.map(
        lambda old, new: jnp.where(
            route_owned,
            jnp.where(same_atomic, old, new),
            old,
        ),
        tasks,
        desired,
    )
    return controller._replace(unit_tasks=unit_tasks)


def update_route_cards_from_effects_v3(
    states: State,
    next_states: State,
    planned: ProjectControllerStateV2,
    updated: ProjectControllerStateV2,
    action: E4PlayerActionV1,
    player: int,
) -> ProjectControllerStateV2:
    """Advance a card only after its current primitive effect is confirmed."""

    cards = planned.route_cards
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    unit = jnp.arange(MAX_UNITS, dtype=jnp.int32)[None]
    cursor = jnp.clip(
        cards.route_cursor.astype(jnp.int32), 0, MAX_ROUTE_CARD_STOPS_V3 - 1
    )
    target = jnp.take_along_axis(
        cards.target_ids, cursor[..., None], axis=-1
    )[..., 0]
    safe_target = jnp.clip(target.astype(jnp.int32), 0, BOARD_SIZE * BOARD_SIZE - 1)
    x = safe_target % BOARD_SIZE
    y = safe_target // BOARD_SIZE
    action_mask = jnp.take_along_axis(
        cards.action_masks, cursor[..., None], axis=-1
    )[..., 0]
    active = cards.status == RouteCardStatusV3.ACTIVE
    returning = cards.status == RouteCardStatusV3.RETURNING
    waiting = cards.status == RouteCardStatusV3.WAIT_AUTO_BANK
    op = action.unit_op
    before_task = planned.unit_tasks
    after_task = updated.unit_tasks
    atomic_done = after_task.status == TaskStatusV1.DONE
    atomic_failed = after_task.status == TaskStatusV1.FAILED

    feed_done = (
        active
        & ((action_mask & int(RouteActionV3.FEED)) != 0)
        & (before_task.task_type == TaskTypeV1.ANIMAL_FEED)
        & atomic_done
    )
    care_done = (
        active
        & ((action_mask & int(RouteActionV3.CARE)) != 0)
        & (before_task.task_type == TaskTypeV1.ANIMAL_CARE)
        & atomic_done
    )
    pre_yield = states.tile_yield[batch, player, y, x]
    post_yield = next_states.tile_yield[batch, player, y, x]
    animal_harvest_done = (
        active
        & ((action_mask & int(RouteActionV3.ANIMAL_HARVEST)) != 0)
        & (before_task.task_type == TaskTypeV1.ANIMAL_COLLECT_PRODUCT)
        & (op == UnitOp.HARVEST)
        & (pre_yield > 0)
        & (post_yield < pre_yield)
    )
    pre_flags = states.tile_flags[batch, player, y, x]
    post_flags = next_states.tile_flags[batch, player, y, x]
    collect_done = (
        active
        & ((action_mask & int(RouteActionV3.COLLECT_FERTILIZER)) != 0)
        & (op == UnitOp.COLLECT_FERTILIZER)
        & ((pre_flags & jnp.uint8(FLAG_FERTILIZER_AVAILABLE)) != 0)
        & ((post_flags & jnp.uint8(FLAG_FERTILIZER_AVAILABLE)) == 0)
    )
    fertilize_done = (
        active
        & ((action_mask & int(RouteActionV3.FERTILIZE)) != 0)
        & (before_task.task_type == TaskTypeV1.APPLY_FERTILIZER)
        & (op == UnitOp.FERTILIZE)
        & atomic_done
    )
    water_done = (
        active
        & ((action_mask & int(RouteActionV3.WATER)) != 0)
        & (before_task.task_type == TaskTypeV1.WATER_CROP)
        & (op == UnitOp.WATER)
        & atomic_done
    )
    pre_kind = states.tile_kind[batch, player, y, x]
    post_kind = next_states.tile_kind[batch, player, y, x]
    harvest_done = (
        active
        & ((action_mask & int(RouteActionV3.HARVEST)) != 0)
        & (op == UnitOp.HARVEST)
        & (pre_yield > 0)
        & ((post_yield < pre_yield) | (post_kind != pre_kind))
    )
    renew_done = (
        active
        & ((action_mask & int(RouteActionV3.REPLANT_AND_WATER)) != 0)
        & (op == UnitOp.WATER)
        & atomic_done
    )
    completed = (
        feed_done
        | animal_harvest_done
        | care_done
        | collect_done
        | fertilize_done
        | water_done
        | harvest_done
        | renew_done
    )
    completed_bit = jnp.where(
        feed_done,
        int(RouteActionV3.FEED),
        jnp.where(
            animal_harvest_done,
            int(RouteActionV3.ANIMAL_HARVEST),
            jnp.where(
                care_done,
                int(RouteActionV3.CARE),
                jnp.where(
                collect_done,
                int(RouteActionV3.COLLECT_FERTILIZER),
                jnp.where(
                    fertilize_done,
                    int(RouteActionV3.FERTILIZE),
                    jnp.where(
                        water_done,
                        int(RouteActionV3.WATER),
                        jnp.where(
                            harvest_done,
                            int(RouteActionV3.HARVEST),
                            jnp.where(
                                renew_done,
                                int(RouteActionV3.REPLANT_AND_WATER),
                                0,
                            ),
                        ),
                    ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.uint8)
    remaining_mask = action_mask & jnp.bitwise_not(completed_bit)
    action_masks = cards.action_masks.at[batch, unit, cursor].set(
        jnp.where(completed, remaining_mask, action_mask).astype(jnp.uint8)
    )
    stop_complete = completed & (remaining_mask == 0)
    next_cursor = cards.route_cursor + stop_complete.astype(jnp.int8)
    route_complete = stop_complete & (next_cursor >= cards.route_length)
    auto_mode = cards.return_mode == RouteReturnModeV3.AUTO_BANK_AT_DAY_END
    no_return_mode = cards.return_mode == RouteReturnModeV3.NONE
    explicit_return_mode = (
        cards.return_mode == RouteReturnModeV3.RETURN_AND_DROP
    )
    next_status = cards.status
    next_status = jnp.where(
        route_complete & no_return_mode,
        RouteCardStatusV3.DONE,
        next_status,
    )
    next_status = jnp.where(
        route_complete & auto_mode,
        RouteCardStatusV3.WAIT_AUTO_BANK,
        next_status,
    )
    next_status = jnp.where(
        route_complete & explicit_return_mode,
        RouteCardStatusV3.RETURNING,
        next_status,
    )
    inventory_total = jnp.sum(
        next_states.unit_inventory[:, player].astype(jnp.int32), axis=-1
    )
    day_boundary = (next_states.step % TURNS_PER_DAY) == 0
    auto_banked = waiting & day_boundary[:, None] & (inventory_total == 0)
    returned = returning & atomic_done & (op == UnitOp.DROP)
    next_status = jnp.where(
        auto_banked | returned, RouteCardStatusV3.DONE, next_status
    )
    route_failed = active & atomic_failed
    next_status = jnp.where(
        route_failed, RouteCardStatusV3.FAILED, next_status
    ).astype(jnp.int8)
    failure = jnp.where(
        route_failed,
        jnp.where(
            after_task.failure_code != FailureCodeV1.NONE,
            after_task.failure_code,
            FailureCodeV1.EFFECT_MISMATCH,
        ),
        cards.failure_code,
    ).astype(jnp.int8)
    cards = cards._replace(
        status=next_status,
        action_masks=action_masks,
        route_cursor=jnp.where(
            stop_complete, next_cursor, cards.route_cursor
        ).astype(jnp.int8),
        last_progress_step=jnp.where(
            completed | returned | auto_banked,
            next_states.step[:, None],
            cards.last_progress_step,
        ).astype(jnp.int16),
        failure_code=failure,
    )
    blank_tasks = empty_unit_tasks_v2(batch_size)
    release_atomic = completed | returned | route_failed | auto_banked
    unit_tasks = jax.tree.map(
        lambda value, blank: jnp.where(release_atomic, blank, value),
        updated.unit_tasks,
        blank_tasks,
    )
    return updated._replace(route_cards=cards, unit_tasks=unit_tasks)


def clear_expired_route_cards_v3(
    states: State,
    controller: ProjectControllerStateV2,
    player: int,
) -> ProjectControllerStateV2:
    """Clear stale optional cards before they can emit a late no-op."""

    cards = controller.route_cards
    live = _live_route(cards.status)
    active = cards.status == RouteCardStatusV3.ACTIVE
    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    cursor = jnp.clip(
        cards.route_cursor.astype(jnp.int32), 0, MAX_ROUTE_CARD_STOPS_V3 - 1
    )
    target = jnp.take_along_axis(
        cards.target_ids, cursor[..., None], axis=-1
    )[..., 0]
    item = jnp.take_along_axis(
        cards.target_items, cursor[..., None], axis=-1
    )[..., 0]
    action_mask = jnp.take_along_axis(
        cards.action_masks, cursor[..., None], axis=-1
    )[..., 0]
    safe_target = jnp.clip(
        target.astype(jnp.int32), 0, BOARD_SIZE * BOARD_SIZE - 1
    )
    x = safe_target % BOARD_SIZE
    y = safe_target // BOARD_SIZE
    kind = states.tile_kind[batch, player, y, x]
    crop = states.tile_crop[batch, player, y, x]
    tile_yield = states.tile_yield[batch, player, y, x]
    flags = states.tile_flags[batch, player, y, x]
    fertilized_until = states.tile_fertilized_until[batch, player, y, x]
    current_day = (states.step // TURNS_PER_DAY).astype(jnp.int16)[:, None]
    crop_matches = (
        (kind == TileKind.PLANT)
        & (crop == item)
        & (target >= 0)
    )
    wants_water = (action_mask & int(RouteActionV3.WATER)) != 0
    wants_fertilize = (action_mask & int(RouteActionV3.FERTILIZE)) != 0
    wants_harvest = (action_mask & int(RouteActionV3.HARVEST)) != 0
    wants_renew = (
        action_mask & int(RouteActionV3.REPLANT_AND_WATER)
    ) != 0
    invalid_water = wants_water & (
        (~crop_matches) | ((flags & jnp.uint8(FLAG_WATERED)) != 0)
    )
    invalid_fertilize = wants_fertilize & (
        (~crop_matches) | (fertilized_until > current_day)
    )
    invalid_harvest = wants_harvest & (
        (~crop_matches) | (tile_yield <= 0)
    )
    valid_renew_state = (
        (kind == TileKind.EMPTY)
        | (
            crop_matches
            & ((flags & jnp.uint8(FLAG_WATERED)) == 0)
        )
        | (wants_harvest & crop_matches & (tile_yield > 0))
    )
    invalid_renew = wants_renew & (~valid_renew_state)
    # Weeds and other asynchronous board changes can invalidate a later stop
    # after the route was admitted.  Cancel the optional card before compiling
    # an impossible primitive action; the live scheduler can re-plan the
    # remaining obligations from the observed state on this same step.
    invalid_current_stop = active & (
        invalid_water | invalid_fertilize | invalid_harvest | invalid_renew
    )
    inactive = ~states.unit_active[:, player]
    expired = live & (states.step[:, None] > cards.deadline_step)
    clear = inactive & live
    clear = (
        clear
        | expired
        | invalid_current_stop
        | (cards.status >= RouteCardStatusV3.DONE)
    )
    empty_cards = empty_route_cards_v3(states.step.shape[0])
    cards = jax.tree.map(
        lambda value, blank: jnp.where(
            clear.reshape(clear.shape + (1,) * (value.ndim - clear.ndim)),
            blank,
            value,
        ),
        cards,
        empty_cards,
    )
    empty_tasks = empty_unit_tasks_v2(states.step.shape[0])
    unit_tasks = jax.tree.map(
        lambda value, blank: jnp.where(
            clear.reshape(clear.shape + (1,) * (value.ndim - clear.ndim)),
            blank,
            value,
        ),
        controller.unit_tasks,
        empty_tasks,
    )
    return controller._replace(route_cards=cards, unit_tasks=unit_tasks)


__all__ = [
    "RouteActionV3",
    "RouteCardDiagnosticsV3",
    "RouteCardModeV4",
    "RouteCardStatusV3",
    "RouteCardTypeV3",
    "RouteReturnModeV3",
    "active_route_target_mask_v3",
    "clear_expired_route_cards_v3",
    "materialize_route_card_unit_tasks_v3",
    "materialize_route_cards_v3",
    "update_route_cards_from_effects_v3",
]
