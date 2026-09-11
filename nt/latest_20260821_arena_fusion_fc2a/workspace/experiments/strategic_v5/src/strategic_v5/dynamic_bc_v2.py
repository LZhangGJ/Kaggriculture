"""Ordered Replay-BC inputs for the deployable Dynamic Full-core V2 policy.

The actor sees only the frozen actor-visible observation.  Expert unit actions
and earlier market orders are teacher-forced only after the current market
features and legality mask have been built.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    EPISODE_STEPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_CROPS,
    NUM_PLAYERS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    MarketOp,
    UnitOp,
)
from kaggriculture_jax.types import Action, State, StaticTables

from .dynamic_policy_v2 import (
    MARKET_CANDIDATE_COUNT_V2,
    MARKET_STOP_INDEX_V2,
    DynamicMarketTraceV2,
    build_lightweight_dynamic_market_features_v2,
)
from .e4_core import build_full_core_candidates_v1, evaluate_full_core_feasibility_v1
from .e5_econ import build_full_econ_features_v1, full_econ_score_v1
from .lifecycle import reset_controller_state_v1
from .opponent_v2 import build_candidate_features_v2, build_global_features_v2
from .replay_bc_v2 import broad_replay_bc_candidate_program_v2
from .turn_ledger_v2 import (
    apply_turn_market_order_v2,
    close_unit_phase_v2,
    ordered_market_label_to_order_v2,
    refresh_dynamic_market_matrix_v2,
)


class DynamicReplayTeacherBatchV2(NamedTuple):
    market_trace: DynamicMarketTraceV2
    target_candidate_slot: jax.Array
    target_quantity: jax.Array
    selection_weight: jax.Array
    unit_global_features: jax.Array
    unit_candidate_features: jax.Array
    unit_candidate_task_type: jax.Array
    unit_anchor_mask: jax.Array


def actor_visible_arrays_to_state_v2(arrays: Mapping[str, jax.Array]) -> State:
    """Invert the frozen MLBC actor-visible contract into canonical player 0."""

    step = jnp.asarray(arrays["obs_step"], dtype=jnp.int16)
    batch_size = step.shape[0]
    own_inventory = jnp.asarray(arrays["obs_own_unit_inventory"], dtype=jnp.int16)
    positive = own_inventory > 0
    order = jnp.cumsum(positive.astype(jnp.int8), axis=-1) - 1
    own_order = jnp.where(positive, order, -1).astype(jnp.int8)
    own_next = jnp.sum(positive.astype(jnp.int8), axis=-1).astype(jnp.int8)
    unit_inventory = jnp.zeros(
        (batch_size, NUM_PLAYERS, MAX_UNITS, NUM_SHED_ITEMS), dtype=jnp.int16
    ).at[:, 0].set(own_inventory)
    unit_inventory_order = jnp.full(
        (batch_size, NUM_PLAYERS, MAX_UNITS, NUM_SHED_ITEMS), -1, dtype=jnp.int8
    ).at[:, 0].set(own_order)
    unit_inventory_next_order = jnp.zeros(
        (batch_size, NUM_PLAYERS, MAX_UNITS), dtype=jnp.int8
    ).at[:, 0].set(own_next)
    shed = jnp.zeros(
        (batch_size, NUM_PLAYERS, NUM_SHED_ITEMS), dtype=jnp.int16
    ).at[:, 0].set(jnp.asarray(arrays["obs_own_shed"], dtype=jnp.int16))
    seeds = jnp.zeros(
        (batch_size, NUM_PLAYERS, NUM_CROPS), dtype=jnp.int16
    ).at[:, 0].set(jnp.asarray(arrays["obs_own_seeds"], dtype=jnp.int16))
    zeros_players = jnp.zeros((batch_size, NUM_PLAYERS), dtype=jnp.int32)
    return State(
        step=step,
        episode_seed=jnp.zeros((batch_size,), dtype=jnp.int32),
        money=jnp.asarray(arrays["obs_money"], dtype=jnp.int32),
        tile_kind=jnp.asarray(arrays["obs_tile_kind"], dtype=jnp.int8),
        tile_crop=jnp.asarray(arrays["obs_tile_crop"], dtype=jnp.int8),
        tile_animal=jnp.asarray(arrays["obs_tile_animal"], dtype=jnp.int8),
        tile_origin_day=jnp.asarray(arrays["obs_tile_origin_day"], dtype=jnp.int8),
        tile_yield=jnp.asarray(arrays["obs_tile_yield"], dtype=jnp.int16),
        tile_neglect=jnp.asarray(arrays["obs_tile_neglect"], dtype=jnp.int8),
        tile_max_lifespan=jnp.asarray(
            arrays["obs_tile_max_lifespan"], dtype=jnp.int16
        ),
        tile_fertilized_until=jnp.asarray(
            arrays["obs_tile_fertilized_until"], dtype=jnp.int8
        ),
        tile_pending_care=jnp.asarray(
            arrays["obs_tile_pending_care"], dtype=jnp.int8
        ),
        tile_flags=jnp.asarray(arrays["obs_tile_flags"], dtype=jnp.uint8),
        unit_pos=jnp.asarray(arrays["obs_unit_pos"], dtype=jnp.int8),
        unit_active=jnp.asarray(arrays["obs_unit_active"], dtype=jnp.bool_),
        unit_inventory=unit_inventory,
        unit_inventory_order=unit_inventory_order,
        unit_inventory_next_order=unit_inventory_next_order,
        shed=shed,
        seeds=seeds,
        hires_today=jnp.asarray(arrays["obs_hires_today"], dtype=jnp.int8),
        unlocked_count=jnp.asarray(arrays["obs_unlocked_count"], dtype=jnp.int8),
        market_inventory=jnp.asarray(arrays["obs_market_inventory"], dtype=jnp.int32),
        market_price=jnp.asarray(arrays["obs_market_price"], dtype=jnp.int32),
        town_shops=jnp.asarray(arrays["obs_town_shops"], dtype=jnp.int8),
        town_count=jnp.asarray(arrays["obs_town_count"], dtype=jnp.int8),
        reward=zeros_players,
        done=step >= EPISODE_STEPS - 1,
        hand_cap_hits=zeros_players,
        market_loop_cap_hits=jnp.zeros((batch_size,), dtype=jnp.int32),
        price_lut_oob=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


def replay_unit_action_v2(arrays: Mapping[str, jax.Array]) -> Action:
    """Place the expert unit stage in canonical seat 0 and keep seat 1 PASS."""

    expert_op = jnp.asarray(arrays["target_unit_op"], dtype=jnp.int8)
    batch_size = expert_op.shape[0]
    unit_op = jnp.full(
        (batch_size, NUM_PLAYERS, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8
    ).at[:, 0].set(expert_op)
    unit_item = jnp.full(
        (batch_size, NUM_PLAYERS, MAX_UNITS), -1, dtype=jnp.int8
    ).at[:, 0].set(jnp.asarray(arrays["target_unit_item"], dtype=jnp.int8))
    unit_amount = jnp.ones(
        (batch_size, NUM_PLAYERS, MAX_UNITS), dtype=jnp.int32
    ).at[:, 0].set(jnp.asarray(arrays["target_unit_amount"], dtype=jnp.int32))
    unit_count = jnp.ones((batch_size, NUM_PLAYERS), dtype=jnp.int8).at[:, 0].set(
        jnp.asarray(arrays["target_unit_count"], dtype=jnp.int8)
    )
    return Action(
        unit_op=unit_op,
        unit_item=unit_item,
        unit_amount=unit_amount,
        unit_count=unit_count,
        market_op=jnp.full(
            (batch_size, NUM_PLAYERS, MAX_MARKET_ORDERS), MarketOp.NONE, dtype=jnp.int8
        ),
        market_item=jnp.full(
            (batch_size, NUM_PLAYERS, MAX_MARKET_ORDERS), -1, dtype=jnp.int8
        ),
        market_amount=jnp.zeros(
            (batch_size, NUM_PLAYERS, MAX_MARKET_ORDERS), dtype=jnp.int32
        ),
        market_count=jnp.zeros((batch_size, NUM_PLAYERS), dtype=jnp.int8),
    )


def market_order_slots_v2(op: jax.Array, item: jax.Array) -> jax.Array:
    """Map the official fixed action schema to the 21 Dynamic V2 market slots."""

    op = op.astype(jnp.int32)
    item = item.astype(jnp.int32)
    slot = jnp.full(op.shape, -1, dtype=jnp.int16)
    slot = jnp.where(op == MarketOp.BUY_LAND, 0, slot)
    slot = jnp.where((op == MarketOp.BUY_PRODUCT) & (item == 8), 1, slot)
    slot = jnp.where((op == MarketOp.BUY_PRODUCT) & (item == 0), 2, slot)
    slot = jnp.where(
        (op == MarketOp.BUY_ANIMAL) & (item >= NUM_PRODUCTS) & (item < NUM_SHED_ITEMS),
        3 + item - NUM_PRODUCTS,
        slot,
    )
    slot = jnp.where(
        (op == MarketOp.BUY_SEED) & (item >= 0) & (item < NUM_CROPS),
        6 + item,
        slot,
    )
    slot = jnp.where(op == MarketOp.HIRE, 11, slot)
    slot = jnp.where(
        (op == MarketOp.SELL) & (item >= 0) & (item < NUM_PRODUCTS),
        12 + item,
        slot,
    )
    return slot.astype(jnp.int16)


def build_dynamic_replay_teacher_batch_v2(
    arrays: Mapping[str, jax.Array],
    tables: StaticTables,
    *,
    stop_weight_no_market: float = 0.25,
    stop_weight_after_market: float = 0.5,
) -> DynamicReplayTeacherBatchV2:
    """Build leak-free Dynamic V2 teacher traces before applying each label."""

    states = actor_visible_arrays_to_state_v2(arrays)
    batch_size = states.step.shape[0]
    controller_one = reset_controller_state_v1()
    controller = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape),
        controller_one,
    )
    program = broad_replay_bc_candidate_program_v2()
    candidates = build_full_core_candidates_v1(states, controller, tables, 0, program)
    candidates = candidates._replace(replay_priority=jnp.zeros_like(candidates.replay_priority))
    feasibility = evaluate_full_core_feasibility_v1(states, candidates, tables, 0)
    template_feasibility = feasibility._replace(
        legal_now=feasibility.legal_now.at[:, :MARKET_CANDIDATE_COUNT_V2].set(True),
        bankable_before_terminal=feasibility.bankable_before_terminal.at[
            :, :MARKET_CANDIDATE_COUNT_V2
        ].set(True),
    )
    econ = build_full_econ_features_v1(states, candidates, template_feasibility, tables, 0)
    template_score = full_econ_score_v1(candidates, template_feasibility, econ)
    unit_global = build_global_features_v2(
        states, 0, None, include_opponent=True, include_history=False
    )
    unit_features = build_candidate_features_v2(
        states, candidates, feasibility, econ, 0, tables, include_opponent=True
    )
    ledger, _ = close_unit_phase_v2(states, replay_unit_action_v2(arrays), 0)

    target_op = jnp.asarray(arrays["target_market_op"], dtype=jnp.int8)
    target_item = jnp.asarray(arrays["target_market_item"], dtype=jnp.int8)
    target_amount = jnp.asarray(arrays["target_market_amount"], dtype=jnp.int32)
    target_count = jnp.asarray(arrays["target_market_count"], dtype=jnp.int32)
    target_slots = market_order_slots_v2(target_op, target_item)
    batch = jnp.arange(batch_size, dtype=jnp.int32)

    global_rows = []
    feature_rows = []
    task_rows = []
    mask_rows = []
    selected_rows = []
    quantity_rows = []
    filled_rows = []
    weight_rows = []
    for ordinal in range(MAX_MARKET_ORDERS):
        dynamic_market = refresh_dynamic_market_matrix_v2(candidates.quantity, ledger, tables)
        market_features, _ = build_lightweight_dynamic_market_features_v2(
            unit_features,
            candidates.mandatory,
            dynamic_market,
            econ,
            template_score,
            ledger,
            tables,
        )
        mask = jnp.concatenate(
            (dynamic_market.present, jnp.ones((batch_size, 1), dtype=jnp.bool_)),
            axis=-1,
        )
        is_order = ordinal < target_count
        is_stop = ordinal == target_count
        raw_slot = target_slots[:, ordinal]
        target_slot = jnp.where(is_order, raw_slot, MARKET_STOP_INDEX_V2).astype(jnp.int16)
        amount = jnp.where(is_order, target_amount[:, ordinal], 0).astype(jnp.int16)
        safe_slot = jnp.clip(raw_slot.astype(jnp.int32), 0, MARKET_CANDIDATE_COUNT_V2 - 1)
        present = dynamic_market.present[batch, safe_slot] & (raw_slot >= 0)
        valid_order, market_op, market_item, market_quantity = ordered_market_label_to_order_v2(
            raw_slot, target_amount[:, ordinal]
        )
        updated = apply_turn_market_order_v2(
            ledger, market_op, market_item, market_quantity, tables
        )
        safe_count = jnp.clip(ledger.market_count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1)
        filled = updated.market_filled_amount[batch, safe_count]
        apply_label = is_order & valid_order
        ledger = jax.tree.map(
            lambda new, old: jnp.where(
                apply_label.reshape((batch_size,) + (1,) * (new.ndim - 1)), new, old
            ),
            updated,
            ledger,
        )
        stop_weight = jnp.where(target_count == 0, stop_weight_no_market, stop_weight_after_market)
        weight = jnp.where(
            is_order & present & (filled > 0),
            1.0,
            jnp.where(is_stop, stop_weight, 0.0),
        ).astype(jnp.float32)
        global_rows.append(unit_global)
        feature_rows.append(market_features)
        task_rows.append(candidates.task_type[:, :MARKET_CANDIDATE_COUNT_V2])
        mask_rows.append(mask)
        selected_rows.append(target_slot)
        quantity_rows.append(amount)
        filled_rows.append(jnp.where(is_order, filled, 0).astype(jnp.int16))
        weight_rows.append(weight)

    def stack(rows):
        return jnp.stack(rows, axis=1)

    trace = DynamicMarketTraceV2(
        global_features=stack(global_rows).astype(jnp.float32),
        candidate_features=stack(feature_rows).astype(jnp.float16),
        candidate_task_type=stack(task_rows).astype(jnp.int8),
        masks=stack(mask_rows),
        selected_indices=stack(selected_rows),
        requested_quantity=stack(quantity_rows),
        filled_quantity=stack(filled_rows),
        joint_logprob=jnp.zeros((batch_size,), dtype=jnp.float32),
    )
    unit_slot = jnp.arange(candidates.present.shape[-1])[None, :] >= MARKET_CANDIDATE_COUNT_V2
    return DynamicReplayTeacherBatchV2(
        market_trace=trace,
        target_candidate_slot=trace.selected_indices,
        target_quantity=trace.requested_quantity,
        selection_weight=stack(weight_rows),
        unit_global_features=unit_global,
        unit_candidate_features=unit_features,
        unit_candidate_task_type=candidates.task_type,
        unit_anchor_mask=candidates.present & unit_slot,
    )
