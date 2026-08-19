"""M3.6A same-turn unit projection and ordered commitment compiler."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    ANIMAL_COST,
    CROP_SEED_COST,
    HIRE_COST,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    SHED_CAPACITY,
    TileKind,
    UnitOp,
    MarketOp,
)
from kaggriculture_jax.policy import combine_player_actions
from kaggriculture_jax.simulator import batched_project_unit_phase
from kaggriculture_jax.types import State

from .m36_schema import (
    CommitmentBundleV3,
    MAX_MARKET_INTENTS_V3,
    M36BundleStatusV3,
    M36IntentFailureV3,
    M36IntentStatusV3,
    M36PlayerDecisionV3,
    M36TransactionDiagnosticsV3,
    NUM_STRUCTURE_KINDS_V3,
    RouteCalendarV3,
)
from .null_opponent import jax_null_player_action


_HIRE_COST = jnp.asarray(HIRE_COST, dtype=jnp.int32)
_SEED_COST = jnp.asarray(CROP_SEED_COST, dtype=jnp.int32)
_ANIMAL_COST = jnp.asarray(ANIMAL_COST, dtype=jnp.int32)


def empty_commitment_bundle_v3(batch_size: int) -> CommitmentBundleV3:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    zeros = jnp.zeros((batch_size,), dtype=jnp.int32)
    market_shape = (batch_size, MAX_MARKET_INTENTS_V3)
    return CommitmentBundleV3(
        bundle_id=zeros,
        start_step=jnp.full((batch_size,), -1, dtype=jnp.int16),
        unit_unlock_requested=jnp.zeros(
            (batch_size, NUM_STRUCTURE_KINDS_V3), dtype=jnp.int16
        ),
        projected_new_structures=jnp.zeros(
            (batch_size, NUM_STRUCTURE_KINDS_V3), dtype=jnp.int16
        ),
        market_op=jnp.zeros(market_shape, dtype=jnp.int8),
        market_item=jnp.full(market_shape, -1, dtype=jnp.int8),
        requested_quantity=jnp.zeros(market_shape, dtype=jnp.int16),
        filled_quantity=jnp.zeros(market_shape, dtype=jnp.int16),
        post_step_quantity=jnp.zeros(market_shape, dtype=jnp.int16),
        intent_status=jnp.zeros(market_shape, dtype=jnp.int8),
        failure_code=jnp.zeros(market_shape, dtype=jnp.int8),
        project_id=jnp.full(market_shape, -1, dtype=jnp.int16),
        priority=jnp.zeros(market_shape, dtype=jnp.int16),
        market_count=jnp.zeros((batch_size,), dtype=jnp.int8),
        reserved_cash=zeros,
        reserved_shed_capacity=jnp.zeros((batch_size,), dtype=jnp.int16),
        reserved_market_slots=jnp.zeros((batch_size,), dtype=jnp.int8),
        pending_animals=jnp.zeros((batch_size, NUM_ANIMALS), dtype=jnp.int16),
        pending_structures=jnp.zeros(
            (batch_size, NUM_STRUCTURE_KINDS_V3), dtype=jnp.int16
        ),
        pending_feed=jnp.zeros((batch_size,), dtype=jnp.int16),
        future_place_plan=jnp.zeros((batch_size, NUM_ANIMALS), dtype=jnp.int16),
        status=jnp.zeros((batch_size,), dtype=jnp.int8),
        overflow_count=zeros,
    )


def _day_values(states: State, calendar: RouteCalendarV3):
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)
    day = jnp.clip(states.step.astype(jnp.int32) // 24, 0, 29)
    return (
        calendar.hand_target_by_day[batch, day],
        calendar.crop_target_by_day[batch, day],
        calendar.animal_purchase_additions_by_day[batch, day],
        calendar.land_additions_by_day[batch, day],
        calendar.feed_stock_target_by_day[batch, day],
        calendar.market_template_by_day[batch, day],
    )


def _structure_counts(states: State, player: int) -> jax.Array:
    kinds = states.tile_kind[:, player]
    return jnp.stack(
        (
            jnp.sum(kinds == TileKind.COOP, axis=(1, 2), dtype=jnp.int16),
            jnp.sum(kinds == TileKind.PASTURE, axis=(1, 2), dtype=jnp.int16),
        ),
        axis=-1,
    )


def _plan_unit_unlocks_v3(
    states: State, calendar: RouteCalendarV3, player: int
) -> tuple[jax.Array, jax.Array]:
    """Plan only structures that unlock the current commitment bundle."""

    _, _, animal_additions, _, _, _ = _day_values(states, calendar)
    day_start = (states.step % 24) == 0
    animal_additions = jnp.where(day_start[:, None], animal_additions, 0)
    structures = _structure_counts(states, player)
    required = jnp.stack(
        (
            animal_additions[:, 0],
            animal_additions[:, 1] + animal_additions[:, 2],
        ),
        axis=-1,
    ).astype(jnp.int16)
    missing = jnp.maximum(required - structures, 0).astype(jnp.int16)
    positions = states.unit_pos[:, player, 0].astype(jnp.int32)
    batch = jnp.arange(states.step.shape[0], dtype=jnp.int32)
    x = jnp.clip(positions[:, 0], 0, 9)
    y = jnp.clip(positions[:, 1], 0, 9)
    farmer_ready = states.unit_active[:, player, 0] & (
        states.tile_kind[batch, player, y, x] == TileKind.EMPTY
    )
    # Cow/sheep commitments win the first-step tie because a shared pasture
    # unlocks both species.  A later unified scheduler will select among all
    # unlock tasks, but the choice is already based on business commitments.
    build_pasture = farmer_ready & (missing[:, 1] > 0)
    build_coop = farmer_ready & (~build_pasture) & (missing[:, 0] > 0)
    requested = jnp.stack((build_coop, build_pasture), axis=-1).astype(jnp.int16)
    op = jnp.where(
        build_pasture,
        UnitOp.BUILD_PASTURE,
        jnp.where(build_coop, UnitOp.BUILD_COOP, UnitOp.PASS),
    ).astype(jnp.int8)
    return requested, op


def _unit_action_dict(states: State, player: int, farmer_op: jax.Array) -> dict:
    batch_size = states.step.shape[0]
    unit_op = jnp.full((batch_size, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8)
    unit_op = unit_op.at[:, 0].set(farmer_op)
    return {
        "unit_op": unit_op,
        "unit_item": jnp.full((batch_size, MAX_UNITS), -1, dtype=jnp.int8),
        "unit_amount": jnp.ones((batch_size, MAX_UNITS), dtype=jnp.int32),
        "unit_count": jnp.sum(
            states.unit_active[:, player], axis=-1, dtype=jnp.int8
        ),
        "market_op": jnp.zeros(
            (batch_size, MAX_MARKET_ORDERS), dtype=jnp.int8
        ),
        "market_item": jnp.full(
            (batch_size, MAX_MARKET_ORDERS), -1, dtype=jnp.int8
        ),
        "market_amount": jnp.zeros(
            (batch_size, MAX_MARKET_ORDERS), dtype=jnp.int32
        ),
        "market_count": jnp.zeros((batch_size,), dtype=jnp.int8),
    }


def project_m36_unit_phase_v3(
    states: State, player: int, player_unit_action: dict
) -> State:
    null = jax_null_player_action(states.step.shape[0])
    joint = (
        combine_player_actions(player_unit_action, null)
        if player == 0
        else combine_player_actions(null, player_unit_action)
    )
    return batched_project_unit_phase(states, joint)


def _append_intent(
    bundle: CommitmentBundleV3,
    present: jax.Array,
    op: int,
    item: jax.Array,
    quantity: jax.Array,
    project_id: jax.Array,
    priority: int,
) -> CommitmentBundleV3:
    batch_size = bundle.market_count.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    room = bundle.market_count < MAX_MARKET_INTENTS_V3
    accept = present & room & (quantity > 0)
    overflow = present & (~room) & (quantity > 0)
    slot = jnp.clip(bundle.market_count.astype(jnp.int32), 0, MAX_MARKET_INTENTS_V3 - 1)

    def set_field(field, value):
        old = field[batch, slot]
        return field.at[batch, slot].set(jnp.where(accept, value, old))

    return bundle._replace(
        market_op=set_field(
            bundle.market_op,
            jnp.full((batch_size,), op, dtype=jnp.int8),
        ),
        market_item=set_field(bundle.market_item, item.astype(jnp.int8)),
        requested_quantity=set_field(
            bundle.requested_quantity, quantity.astype(jnp.int16)
        ),
        intent_status=set_field(
            bundle.intent_status,
            jnp.full(
                (batch_size,), M36IntentStatusV3.REQUESTED, dtype=jnp.int8
            ),
        ),
        project_id=set_field(bundle.project_id, project_id.astype(jnp.int16)),
        priority=set_field(
            bundle.priority,
            jnp.full((batch_size,), priority, dtype=jnp.int16),
        ),
        market_count=bundle.market_count + accept.astype(jnp.int8),
        overflow_count=bundle.overflow_count + overflow.astype(jnp.int32),
    )


def _active_crop_counts(states: State, player: int) -> jax.Array:
    crop = states.tile_crop[:, player]
    kind = states.tile_kind[:, player]
    return jnp.stack(
        [
            jnp.sum(
                (kind == TileKind.PLANT) & (crop == crop_id),
                axis=(1, 2),
                dtype=jnp.int16,
            )
            for crop_id in range(NUM_CROPS)
        ],
        axis=-1,
    )


def materialize_commitment_bundle_v3(
    states: State,
    projected_unit_states: State,
    calendar: RouteCalendarV3,
    unit_unlock_requested: jax.Array,
    player: int,
) -> CommitmentBundleV3:
    """Compile one ordered market transaction from high-level daily targets."""

    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    hand_target, crop_target, animal_additions, land_additions, feed_target, _ = (
        _day_values(states, calendar)
    )
    day_start = (states.step % 24) == 0
    animal_additions = jnp.where(day_start[:, None], animal_additions, 0)
    land_additions = jnp.where(day_start, land_additions, 0)
    active_crop = _active_crop_counts(projected_unit_states, player)
    seed_held = projected_unit_states.seeds[:, player].astype(jnp.int16)
    seed_need = jnp.maximum(crop_target - active_crop - seed_held, 0).astype(
        jnp.int16
    )
    hires_needed = jnp.clip(
        hand_target.astype(jnp.int16)
        - projected_unit_states.hires_today[:, player].astype(jnp.int16),
        0,
        10,
    )
    wheat_held = (
        projected_unit_states.shed[:, player, 0].astype(jnp.int16)
        + jnp.sum(
            projected_unit_states.unit_inventory[:, player, :, 0].astype(
                jnp.int16
            ),
            axis=-1,
            dtype=jnp.int16,
        )
    )
    feed_need = jnp.maximum(feed_target - wheat_held, 0).astype(jnp.int16)
    before_structures = _structure_counts(states, player)
    after_structures = _structure_counts(projected_unit_states, player)
    projected_new = jnp.maximum(after_structures - before_structures, 0).astype(
        jnp.int16
    )
    structure_need = jnp.stack(
        (
            animal_additions[:, 0],
            animal_additions[:, 1] + animal_additions[:, 2],
        ),
        axis=-1,
    ).astype(jnp.int16)
    pending_structures = jnp.maximum(structure_need - after_structures, 0).astype(
        jnp.int16
    )
    bundle = empty_commitment_bundle_v3(batch_size)._replace(
        bundle_id=calendar.candidate_id.astype(jnp.int32),
        start_step=states.step.astype(jnp.int16),
        unit_unlock_requested=unit_unlock_requested.astype(jnp.int16),
        projected_new_structures=projected_new,
        pending_animals=animal_additions.astype(jnp.int16),
        pending_structures=pending_structures,
        pending_feed=feed_need,
        future_place_plan=animal_additions.astype(jnp.int16),
        status=jnp.full(
            (batch_size,), M36BundleStatusV3.PLANNED, dtype=jnp.int8
        ),
    )

    # Explicit template T0: HIRE -> BUY_ANIMAL -> BUY_SEED -> BUY_FEED.
    # HIRE is represented as one atomic order per worker, matching official
    # semantics and preserving the exact 10-slot gold opening.
    for hire_index in range(10):
        present = hire_index < hires_needed
        bundle = _append_intent(
            bundle,
            present,
            MarketOp.HIRE,
            jnp.full((batch_size,), -1, dtype=jnp.int8),
            jnp.ones((batch_size,), dtype=jnp.int16),
            jnp.full((batch_size,), 8, dtype=jnp.int16),
            priority=10 + hire_index,
        )
    for animal_id in range(NUM_ANIMALS):
        quantity = animal_additions[:, animal_id]
        bundle = _append_intent(
            bundle,
            quantity > 0,
            MarketOp.BUY_ANIMAL,
            jnp.full(
                (batch_size,), NUM_PRODUCTS + animal_id, dtype=jnp.int8
            ),
            quantity,
            jnp.full(
                (batch_size,), NUM_CROPS + animal_id, dtype=jnp.int16
            ),
            priority=30 + animal_id,
        )
    for crop_id in range(NUM_CROPS):
        quantity = seed_need[:, crop_id]
        bundle = _append_intent(
            bundle,
            quantity > 0,
            MarketOp.BUY_SEED,
            jnp.full((batch_size,), crop_id, dtype=jnp.int8),
            quantity,
            jnp.full((batch_size,), crop_id, dtype=jnp.int16),
            priority=40 + crop_id,
        )
    bundle = _append_intent(
        bundle,
        feed_need > 0,
        MarketOp.BUY_PRODUCT,
        jnp.zeros((batch_size,), dtype=jnp.int8),
        feed_need,
        jnp.full((batch_size,), NUM_CROPS + 1, dtype=jnp.int16),
        priority=50,
    )
    # Land is not used in the KAWASHIGI opening.  Keep it after critical
    # inputs in T0 so a future day cannot consume the cash needed by an
    # already admitted animal commitment.
    for land_index in range(3):
        present = land_index < land_additions
        bundle = _append_intent(
            bundle,
            present,
            MarketOp.BUY_LAND,
            jnp.full((batch_size,), -1, dtype=jnp.int8),
            jnp.ones((batch_size,), dtype=jnp.int16),
            jnp.full((batch_size,), 9, dtype=jnp.int16),
            priority=60 + land_index,
        )

    current_hires = projected_unit_states.hires_today[:, player].astype(jnp.int32)
    hire_offsets = jnp.arange(10, dtype=jnp.int32)[None]
    hire_indices = jnp.clip(
        current_hires[:, None] + hire_offsets, 0, len(HIRE_COST) - 1
    )
    hire_cost = jnp.sum(
        jnp.where(hire_offsets < hires_needed[:, None], _HIRE_COST[hire_indices], 0),
        axis=-1,
        dtype=jnp.int32,
    )
    seed_cost = jnp.sum(seed_need.astype(jnp.int32) * _SEED_COST[None], axis=-1)
    animal_cost = jnp.sum(
        animal_additions.astype(jnp.int32) * _ANIMAL_COST[None], axis=-1
    )
    # Dynamic product pricing is reconciled from official effects.  This
    # reserve is deliberately conservative and does not clip the requested
    # quantity; partial official fills remain observable.
    feed_reserve = feed_need.astype(jnp.int32) * projected_unit_states.market_price[
        :, 0
    ].astype(jnp.int32)
    shed_reserve = jnp.sum(animal_additions, axis=-1, dtype=jnp.int16) + feed_need
    return bundle._replace(
        reserved_cash=hire_cost + seed_cost + animal_cost + feed_reserve,
        reserved_shed_capacity=shed_reserve.astype(jnp.int16),
        reserved_market_slots=bundle.market_count,
    )


def _compile_market_action(bundle: CommitmentBundleV3) -> tuple[jax.Array, ...]:
    active = (
        jnp.arange(MAX_MARKET_INTENTS_V3, dtype=jnp.int8)[None]
        < bundle.market_count[:, None]
    )
    market_op = jnp.where(active, bundle.market_op, MarketOp.NONE).astype(jnp.int8)
    market_item = jnp.where(active, bundle.market_item, -1).astype(jnp.int8)
    atomic = (market_op == MarketOp.HIRE) | (market_op == MarketOp.BUY_LAND)
    market_amount = jnp.where(
        active & (~atomic), bundle.requested_quantity, 0
    ).astype(jnp.int32)
    return market_op, market_item, market_amount, bundle.market_count


def m36a_policy_step_v3(
    states: State, calendar: RouteCalendarV3, player: int
) -> tuple[M36PlayerDecisionV3, State]:
    """Generate one full action through unit projection and commitments."""

    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    batch_size = states.step.shape[0]
    unlock_requested, farmer_op = _plan_unit_unlocks_v3(states, calendar, player)
    unit_action = _unit_action_dict(states, player, farmer_op)
    projected = project_m36_unit_phase_v3(states, player, unit_action)
    bundle = materialize_commitment_bundle_v3(
        states, projected, calendar, unlock_requested, player
    )
    market_op, market_item, market_amount, market_count = _compile_market_action(
        bundle
    )
    positions = states.unit_pos[:, player, 0].astype(jnp.int32)
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    x = jnp.clip(positions[:, 0], 0, 9)
    y = jnp.clip(positions[:, 1], 0, 9)
    unlock_present = jnp.any(unlock_requested > 0, axis=-1)
    invalid_unlock = unlock_present & (
        (~states.unit_active[:, player, 0])
        | (states.tile_kind[batch, player, y, x] != TileKind.EMPTY)
    )
    missing_place = jnp.sum(
        jnp.maximum(bundle.pending_animals - bundle.future_place_plan, 0),
        axis=-1,
        dtype=jnp.int32,
    )
    hard = (
        invalid_unlock.astype(jnp.int32)
        + bundle.overflow_count
        + missing_place
    )
    diagnostics = M36TransactionDiagnosticsV3(
        invalid_unit_unlock_count=invalid_unlock.astype(jnp.int32),
        market_slot_overflow_count=bundle.overflow_count,
        missing_future_place_plan_count=missing_place,
        unclassified_partial_fill_count=jnp.zeros(
            (batch_size,), dtype=jnp.int32
        ),
        hard_error_count=hard,
    )
    return (
        M36PlayerDecisionV3(
            unit_op=unit_action["unit_op"],
            unit_item=unit_action["unit_item"],
            unit_amount=unit_action["unit_amount"],
            unit_count=unit_action["unit_count"],
            market_op=market_op,
            market_item=market_item,
            market_amount=market_amount,
            market_count=market_count,
            bundle=bundle,
            diagnostics=diagnostics,
        ),
        projected,
    )


def m36_player_action_dict_v3(decision: M36PlayerDecisionV3) -> dict:
    return {
        "unit_op": decision.unit_op,
        "unit_item": decision.unit_item,
        "unit_amount": decision.unit_amount,
        "unit_count": decision.unit_count,
        "market_op": decision.market_op,
        "market_item": decision.market_item,
        "market_amount": decision.market_amount,
        "market_count": decision.market_count,
    }


def reconcile_commitment_bundle_v3(
    projected_unit_states: State,
    post_step_states: State,
    bundle: CommitmentBundleV3,
    player: int,
) -> tuple[CommitmentBundleV3, M36TransactionDiagnosticsV3]:
    """Recover requested, filled and post-step quantities from official effects."""

    batch_size = projected_unit_states.step.shape[0]
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    active = (
        jnp.arange(MAX_MARKET_INTENTS_V3, dtype=jnp.int8)[None]
        < bundle.market_count[:, None]
    )
    hire_total = jnp.maximum(
        post_step_states.hires_today[:, player].astype(jnp.int16)
        - projected_unit_states.hires_today[:, player].astype(jnp.int16),
        0,
    )
    land_total = jnp.maximum(
        post_step_states.unlocked_count[:, player].astype(jnp.int16)
        - projected_unit_states.unlocked_count[:, player].astype(jnp.int16),
        0,
    )
    seed_delta = jnp.maximum(
        post_step_states.seeds[:, player].astype(jnp.int16)
        - projected_unit_states.seeds[:, player].astype(jnp.int16),
        0,
    )
    shed_delta = (
        post_step_states.shed[:, player].astype(jnp.int16)
        - projected_unit_states.shed[:, player].astype(jnp.int16)
    )
    shed_buy_total = jnp.maximum(shed_delta, 0)
    shed_sell_total = jnp.maximum(-shed_delta[:, :NUM_PRODUCTS], 0)
    used_hire = jnp.zeros((batch_size,), dtype=jnp.int16)
    used_land = jnp.zeros((batch_size,), dtype=jnp.int16)
    used_seed = jnp.zeros((batch_size, NUM_CROPS), dtype=jnp.int16)
    used_buy = jnp.zeros((batch_size, NUM_SHED_ITEMS), dtype=jnp.int16)
    used_sell = jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int16)
    filled = jnp.zeros_like(bundle.filled_quantity)
    post_quantity = jnp.zeros_like(bundle.post_step_quantity)

    for slot in range(MAX_MARKET_INTENTS_V3):
        present = active[:, slot]
        op = bundle.market_op[:, slot]
        item = bundle.market_item[:, slot].astype(jnp.int32)
        requested = bundle.requested_quantity[:, slot]
        safe_crop = jnp.clip(item, 0, NUM_CROPS - 1)
        safe_shed = jnp.clip(item, 0, NUM_SHED_ITEMS - 1)
        safe_product = jnp.clip(item, 0, NUM_PRODUCTS - 1)
        seed_available = seed_delta[batch, safe_crop] - used_seed[batch, safe_crop]
        buy_available = shed_buy_total[batch, safe_shed] - used_buy[batch, safe_shed]
        sell_available = shed_sell_total[batch, safe_product] - used_sell[batch, safe_product]
        available = jnp.where(
            op == MarketOp.HIRE,
            hire_total - used_hire,
            jnp.where(
                op == MarketOp.BUY_LAND,
                land_total - used_land,
                jnp.where(
                    op == MarketOp.BUY_SEED,
                    seed_available,
                    jnp.where(
                        (op == MarketOp.BUY_PRODUCT)
                        | (op == MarketOp.BUY_ANIMAL),
                        buy_available,
                        jnp.where(op == MarketOp.SELL, sell_available, 0),
                    ),
                ),
            ),
        ).astype(jnp.int16)
        take = jnp.where(
            present, jnp.minimum(requested, jnp.maximum(available, 0)), 0
        ).astype(jnp.int16)
        filled = filled.at[:, slot].set(take)
        used_hire = used_hire + jnp.where(op == MarketOp.HIRE, take, 0)
        used_land = used_land + jnp.where(op == MarketOp.BUY_LAND, take, 0)
        used_seed = used_seed.at[batch, safe_crop].add(
            jnp.where(op == MarketOp.BUY_SEED, take, 0)
        )
        used_buy = used_buy.at[batch, safe_shed].add(
            jnp.where(
                (op == MarketOp.BUY_PRODUCT) | (op == MarketOp.BUY_ANIMAL),
                take,
                0,
            )
        )
        used_sell = used_sell.at[batch, safe_product].add(
            jnp.where(op == MarketOp.SELL, take, 0)
        )
        post = jnp.where(
            op == MarketOp.HIRE,
            post_step_states.hires_today[:, player].astype(jnp.int16),
            jnp.where(
                op == MarketOp.BUY_LAND,
                post_step_states.unlocked_count[:, player].astype(jnp.int16),
                jnp.where(
                    op == MarketOp.BUY_SEED,
                    post_step_states.seeds[batch, player, safe_crop],
                    post_step_states.shed[batch, player, safe_shed],
                ),
            ),
        ).astype(jnp.int16)
        post_quantity = post_quantity.at[:, slot].set(jnp.where(present, post, 0))

    incomplete = active & (filled < bundle.requested_quantity)
    any_fill = filled > 0
    shed_full = (
        jnp.sum(post_step_states.shed[:, player].astype(jnp.int32), axis=-1)
        >= SHED_CAPACITY
    )[:, None]
    safe_item = jnp.clip(bundle.market_item.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    next_dynamic_price = post_step_states.market_price[batch[:, None], safe_item]
    fixed_seed = _SEED_COST[
        jnp.clip(bundle.market_item.astype(jnp.int32), 0, NUM_CROPS - 1)
    ]
    animal_id = jnp.clip(
        bundle.market_item.astype(jnp.int32) - NUM_PRODUCTS,
        0,
        NUM_ANIMALS - 1,
    )
    fixed_animal = _ANIMAL_COST[animal_id]
    next_price = jnp.where(
        bundle.market_op == MarketOp.BUY_PRODUCT,
        next_dynamic_price,
        jnp.where(
            bundle.market_op == MarketOp.BUY_SEED,
            fixed_seed,
            jnp.where(bundle.market_op == MarketOp.BUY_ANIMAL, fixed_animal, 0),
        ),
    )
    cash_short = post_step_states.money[:, player, None] < next_price
    capacity_op = (bundle.market_op == MarketOp.BUY_PRODUCT) | (
        bundle.market_op == MarketOp.BUY_ANIMAL
    )
    failure = jnp.where(
        incomplete & capacity_op & shed_full,
        M36IntentFailureV3.SHED_CAPACITY,
        jnp.where(
            incomplete & cash_short,
            M36IntentFailureV3.INSUFFICIENT_CASH,
            jnp.where(
                incomplete,
                M36IntentFailureV3.MARKET_CHANGED_OR_UNKNOWN,
                M36IntentFailureV3.NONE,
            ),
        ),
    ).astype(jnp.int8)
    intent_status = jnp.where(
        ~active,
        M36IntentStatusV3.EMPTY,
        jnp.where(
            ~incomplete,
            M36IntentStatusV3.FILLED,
            jnp.where(any_fill, M36IntentStatusV3.PARTIAL, M36IntentStatusV3.REJECTED),
        ),
    ).astype(jnp.int8)
    unknown = incomplete & (
        failure == M36IntentFailureV3.MARKET_CHANGED_OR_UNKNOWN
    )
    unknown_count = jnp.sum(unknown, axis=-1, dtype=jnp.int32)
    any_incomplete = jnp.any(incomplete, axis=-1)
    any_success = jnp.any(active & any_fill, axis=-1)
    bundle_status = jnp.where(
        ~jnp.any(active, axis=-1),
        M36BundleStatusV3.EMPTY,
        jnp.where(
            ~any_incomplete,
            M36BundleStatusV3.FILLED,
            jnp.where(any_success, M36BundleStatusV3.PARTIAL, M36BundleStatusV3.FAILED),
        ),
    ).astype(jnp.int8)
    filled_animals = jnp.stack(
        [
            jnp.sum(
                jnp.where(
                    active
                    & (bundle.market_op == MarketOp.BUY_ANIMAL)
                    & (bundle.market_item == NUM_PRODUCTS + animal_id),
                    filled,
                    0,
                ),
                axis=-1,
                dtype=jnp.int16,
            )
            for animal_id in range(NUM_ANIMALS)
        ],
        axis=-1,
    )
    filled_feed = jnp.sum(
        jnp.where(
            active
            & (bundle.market_op == MarketOp.BUY_PRODUCT)
            & (bundle.market_item == 0),
            filled,
            0,
        ),
        axis=-1,
        dtype=jnp.int16,
    )
    reconciled = bundle._replace(
        filled_quantity=filled,
        post_step_quantity=post_quantity,
        intent_status=intent_status,
        failure_code=failure,
        pending_animals=filled_animals,
        pending_feed=jnp.maximum(bundle.pending_feed - filled_feed, 0).astype(
            jnp.int16
        ),
        status=bundle_status,
    )
    hard = (
        bundle.overflow_count
        + unknown_count
        + jnp.sum(
            jnp.maximum(filled_animals - bundle.future_place_plan, 0),
            axis=-1,
            dtype=jnp.int32,
        )
    )
    diagnostics = M36TransactionDiagnosticsV3(
        invalid_unit_unlock_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        market_slot_overflow_count=bundle.overflow_count,
        missing_future_place_plan_count=jnp.sum(
            jnp.maximum(filled_animals - bundle.future_place_plan, 0),
            axis=-1,
            dtype=jnp.int32,
        ),
        unclassified_partial_fill_count=unknown_count,
        hard_error_count=hard,
    )
    return reconciled, diagnostics


__all__ = [
    "empty_commitment_bundle_v3",
    "m36_player_action_dict_v3",
    "m36a_policy_step_v3",
    "materialize_commitment_bundle_v3",
    "project_m36_unit_phase_v3",
    "reconcile_commitment_bundle_v3",
]
