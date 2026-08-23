"""Teacher-forced Replay audit for the dynamic Full-core turn ledger.

The current market label is inspected before it is applied.  Candidate inputs
therefore contain only the public observation, the already-selected unit stage,
and earlier market labels.  Current/future labels are never actor inputs.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import (
    ANIMAL_COST,
    CROP_SEED_COST,
    HIRE_COST,
    LAND_PRICES,
    MARKET_LUT_SIZE,
    MARKET_MIN_INVENTORY,
    MAX_HANDS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_PRODUCTS,
    SHED_CAPACITY,
    MarketOp,
    UnitOp,
)
from kaggriculture_jax.types import Action, State, StaticTables

from .e4_core import build_full_core_candidates_v1
from .lifecycle import reset_controller_state_v1
from .replay_bc_v2 import (
    ReplayBCSampleV2,
    _stack_states,
    broad_replay_bc_candidate_program_v2,
)
from .turn_ledger_v2 import (
    TurnLedgerV2,
    apply_turn_market_order_v2,
    close_unit_phase_v2,
    ordered_market_label_to_order_v2,
    refresh_dynamic_market_candidates_v2,
)


DYNAMIC_REPLAY_AUDIT_SCHEMA_V2 = "dynamic_full_core_replay_audit_v2"


class MissingReasonV2:
    PRESENT = 0
    INSUFFICIENT_CASH = 1
    SHED_FULL = 2
    SHED_EMPTY = 3
    HIRE_LIMIT = 4
    LAND_LIMIT = 5
    PHASE_OR_ORDER_LIMIT = 6
    INVALID_QUANTITY = 7
    UNKNOWN = 8


MISSING_REASON_NAMES_V2 = (
    "present",
    "insufficient_cash",
    "shed_full",
    "shed_empty",
    "hire_limit",
    "land_limit",
    "phase_or_order_limit",
    "invalid_quantity",
    "unknown",
)


class DynamicReplayBatchAuditV2(NamedTuple):
    label_valid: jax.Array
    label_slot: jax.Array
    requested_amount: jax.Array
    static_present: jax.Array
    after_unit_present: jax.Array
    dynamic_present: jax.Array
    missing_reason: jax.Array
    filled_amount: jax.Array
    money_before: jax.Array
    money_after: jax.Array
    shed_total_before: jax.Array
    shed_total_after: jax.Array
    final_market_count: jax.Array
    final_invalid_order_count: jax.Array
    final_overflow_order_count: jax.Array


class DynamicReplayAuditInputV2(NamedTuple):
    states: State
    unit_op: jax.Array
    unit_item: jax.Array
    unit_quantity: jax.Array
    unit_count: jax.Array
    market_slot: jax.Array
    market_quantity: jax.Array
    market_valid: jax.Array


def _controller_batch_v2(batch_size: int):
    one = reset_controller_state_v1()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape), one
    )


def _ordered_unit_action_v2(data: DynamicReplayAuditInputV2) -> Action:
    batch_size = data.states.step.shape[0]
    unit_op = jnp.full(
        (batch_size, 2, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8
    ).at[:, 0].set(data.unit_op.astype(jnp.int8))
    unit_item = jnp.full(
        (batch_size, 2, MAX_UNITS), -1, dtype=jnp.int8
    ).at[:, 0].set(data.unit_item.astype(jnp.int8))
    unit_amount = jnp.ones(
        (batch_size, 2, MAX_UNITS), dtype=jnp.int32
    ).at[:, 0].set(data.unit_quantity.astype(jnp.int32))
    unit_count = jnp.ones((batch_size, 2), dtype=jnp.int8).at[:, 0].set(
        data.unit_count.astype(jnp.int8)
    )
    return Action(
        unit_op=unit_op,
        unit_item=unit_item,
        unit_amount=unit_amount,
        unit_count=unit_count,
        market_op=jnp.full(
            (batch_size, 2, MAX_MARKET_ORDERS), MarketOp.NONE, dtype=jnp.int8
        ),
        market_item=jnp.full(
            (batch_size, 2, MAX_MARKET_ORDERS), -1, dtype=jnp.int8
        ),
        market_amount=jnp.zeros(
            (batch_size, 2, MAX_MARKET_ORDERS), dtype=jnp.int32
        ),
        market_count=jnp.zeros((batch_size, 2), dtype=jnp.int8),
    )


def _gather_candidate(mask: jax.Array, selected: jax.Array) -> jax.Array:
    batch = jnp.arange(mask.shape[0], dtype=jnp.int32)
    safe = jnp.clip(selected.astype(jnp.int32), 0, mask.shape[1] - 1)
    return mask[batch, safe]


def _masked_ledger(
    updated: TurnLedgerV2, previous: TurnLedgerV2, use_updated: jax.Array
) -> TurnLedgerV2:
    def choose(new, old):
        shape = (use_updated.shape[0],) + (1,) * (new.ndim - 1)
        return jnp.where(use_updated.reshape(shape), new, old)

    return jax.tree.map(choose, updated, previous)


def _one_unit_cost_v2(
    ledger: TurnLedgerV2, selected: jax.Array, tables: StaticTables
) -> jax.Array:
    safe = jnp.clip(selected.astype(jnp.int32), 0, 20)
    land_index = jnp.clip(
        ledger.unlocked_count.astype(jnp.int32) - 1, 0, len(LAND_PRICES) - 1
    )
    hire_index = jnp.clip(
        ledger.hires_today.astype(jnp.int32), 0, MAX_HANDS - 1
    )
    product_item = jnp.where(safe == 1, 8, 0)
    batch = jnp.arange(safe.shape[0], dtype=jnp.int32)
    product_inventory = ledger.market_inventory[batch, product_item] - 1
    lut_index = jnp.clip(
        product_inventory - MARKET_MIN_INVENTORY, 0, MARKET_LUT_SIZE - 1
    )
    product_cost = tables.market_price[product_item, lut_index].astype(jnp.int32)
    animal = jnp.clip(safe - 3, 0, len(ANIMAL_COST) - 1)
    crop = jnp.clip(safe - 6, 0, len(CROP_SEED_COST) - 1)
    return jnp.where(
        safe == 0,
        jnp.asarray(LAND_PRICES, dtype=jnp.int32)[land_index],
        jnp.where(
            (safe == 1) | (safe == 2),
            product_cost,
            jnp.where(
                (safe >= 3) & (safe <= 5),
                jnp.asarray(ANIMAL_COST, dtype=jnp.int32)[animal],
                jnp.where(
                    (safe >= 6) & (safe <= 10),
                    jnp.asarray(CROP_SEED_COST, dtype=jnp.int32)[crop],
                    jnp.where(
                        safe == 11,
                        jnp.asarray(HIRE_COST, dtype=jnp.int32)[hire_index],
                        0,
                    ),
                ),
            ),
        ),
    )


def _missing_reason_v2(
    ledger: TurnLedgerV2,
    selected: jax.Array,
    amount: jax.Array,
    present: jax.Array,
    tables: StaticTables,
) -> jax.Array:
    safe = jnp.clip(selected.astype(jnp.int32), 0, 20)
    batch = jnp.arange(safe.shape[0], dtype=jnp.int32)
    shed_total = jnp.sum(ledger.shed.astype(jnp.int32), axis=-1)
    sell_item = jnp.clip(safe - 12, 0, NUM_PRODUCTS - 1)
    sell_empty = (safe >= 12) & (ledger.shed[batch, sell_item] <= 0)
    needs_room = (safe >= 1) & (safe <= 5)
    shed_full = needs_room & (shed_total >= SHED_CAPACITY)
    hire_limit = (safe == 11) & (
        ledger.hires_today.astype(jnp.int32) >= MAX_HANDS
    )
    land_index = ledger.unlocked_count.astype(jnp.int32) - 1
    land_limit = (safe == 0) & (
        (land_index < 0) | (land_index >= len(LAND_PRICES))
    )
    phase_limit = (
        (ledger.phase != 1)
        | (ledger.market_count.astype(jnp.int32) >= MAX_MARKET_ORDERS)
    )
    cost = _one_unit_cost_v2(ledger, safe, tables)
    insufficient_cash = (safe < 12) & (ledger.money_nominal < cost)
    reason = jnp.full_like(safe, MissingReasonV2.UNKNOWN, dtype=jnp.int8)
    reason = jnp.where(amount <= 0, MissingReasonV2.INVALID_QUANTITY, reason)
    reason = jnp.where(insufficient_cash, MissingReasonV2.INSUFFICIENT_CASH, reason)
    reason = jnp.where(land_limit, MissingReasonV2.LAND_LIMIT, reason)
    reason = jnp.where(hire_limit, MissingReasonV2.HIRE_LIMIT, reason)
    reason = jnp.where(sell_empty, MissingReasonV2.SHED_EMPTY, reason)
    reason = jnp.where(shed_full, MissingReasonV2.SHED_FULL, reason)
    reason = jnp.where(phase_limit, MissingReasonV2.PHASE_OR_ORDER_LIMIT, reason)
    return jnp.where(present, MissingReasonV2.PRESENT, reason).astype(jnp.int8)


def audit_dynamic_replay_batch_v2(
    data: DynamicReplayAuditInputV2, tables: StaticTables
) -> DynamicReplayBatchAuditV2:
    """Evaluate each market target before teacher-forcing it into the prefix."""

    batch_size = data.states.step.shape[0]
    candidates = build_full_core_candidates_v1(
        data.states,
        _controller_batch_v2(batch_size),
        tables,
        0,
        broad_replay_bc_candidate_program_v2(),
    )
    static_mask = candidates.present
    ledger, _ = close_unit_phase_v2(data.states, _ordered_unit_action_v2(data), 0)
    after_unit = refresh_dynamic_market_candidates_v2(candidates, ledger, tables)

    static_rows = []
    unit_rows = []
    dynamic_rows = []
    reason_rows = []
    filled_rows = []
    money_before_rows = []
    money_after_rows = []
    shed_before_rows = []
    shed_after_rows = []
    label_rows = []
    slot_rows = []
    amount_rows = []
    batch = jnp.arange(batch_size, dtype=jnp.int32)

    for ordinal in range(MAX_MARKET_ORDERS):
        selected = data.market_slot[:, ordinal].astype(jnp.int32)
        amount = data.market_quantity[:, ordinal].astype(jnp.int32)
        label_valid = (
            data.market_valid[:, ordinal]
            & (selected >= 0)
            & (selected < 21)
        )
        refreshed = refresh_dynamic_market_candidates_v2(candidates, ledger, tables)
        static_present = label_valid & _gather_candidate(static_mask, selected)
        unit_present = label_valid & _gather_candidate(after_unit.present, selected)
        dynamic_present = label_valid & _gather_candidate(refreshed.present, selected)
        reason = jnp.where(
            label_valid,
            _missing_reason_v2(ledger, selected, amount, dynamic_present, tables),
            MissingReasonV2.PRESENT,
        ).astype(jnp.int8)

        before_count = ledger.market_count.astype(jnp.int32)
        valid_order, op, item, quantity = ordered_market_label_to_order_v2(
            selected, amount
        )
        consume = label_valid & valid_order
        updated = apply_turn_market_order_v2(ledger, op, item, quantity, tables)
        safe_count = jnp.clip(before_count, 0, MAX_MARKET_ORDERS - 1)
        filled = jnp.where(
            consume,
            updated.market_filled_amount[batch, safe_count],
            0,
        ).astype(jnp.int32)
        next_ledger = _masked_ledger(updated, ledger, consume)

        label_rows.append(label_valid)
        slot_rows.append(selected.astype(jnp.int16))
        amount_rows.append(amount)
        static_rows.append(static_present)
        unit_rows.append(unit_present)
        dynamic_rows.append(dynamic_present)
        reason_rows.append(reason)
        filled_rows.append(filled)
        money_before_rows.append(ledger.money_nominal)
        money_after_rows.append(next_ledger.money_nominal)
        shed_before_rows.append(jnp.sum(ledger.shed.astype(jnp.int32), axis=-1))
        shed_after_rows.append(
            jnp.sum(next_ledger.shed.astype(jnp.int32), axis=-1)
        )
        ledger = next_ledger

    def columns(rows):
        return jnp.stack(rows, axis=1)

    return DynamicReplayBatchAuditV2(
        label_valid=columns(label_rows),
        label_slot=columns(slot_rows),
        requested_amount=columns(amount_rows),
        static_present=columns(static_rows),
        after_unit_present=columns(unit_rows),
        dynamic_present=columns(dynamic_rows),
        missing_reason=columns(reason_rows),
        filled_amount=columns(filled_rows),
        money_before=columns(money_before_rows),
        money_after=columns(money_after_rows),
        shed_total_before=columns(shed_before_rows),
        shed_total_after=columns(shed_after_rows),
        final_market_count=ledger.market_count,
        final_invalid_order_count=ledger.invalid_order_count,
        final_overflow_order_count=ledger.overflow_order_count,
    )


def replay_samples_to_audit_input_v2(
    samples: Iterable[ReplayBCSampleV2],
) -> DynamicReplayAuditInputV2:
    rows = list(samples)
    if not rows:
        raise ValueError("at least one Replay sample is required")
    return DynamicReplayAuditInputV2(
        states=_stack_states([row.state for row in rows]),
        unit_op=jnp.asarray(np.stack([row.ordered_turn.unit_op for row in rows])),
        unit_item=jnp.asarray(np.stack([row.ordered_turn.unit_item for row in rows])),
        unit_quantity=jnp.asarray(
            np.stack([row.ordered_turn.unit_quantity for row in rows])
        ),
        unit_count=jnp.asarray(
            [row.ordered_turn.unit_count for row in rows], dtype=jnp.int8
        ),
        market_slot=jnp.asarray(
            np.stack([row.ordered_turn.market_candidate_slot[:MAX_MARKET_ORDERS] for row in rows])
        ),
        market_quantity=jnp.asarray(
            np.stack([row.ordered_turn.market_quantity[:MAX_MARKET_ORDERS] for row in rows])
        ),
        market_valid=jnp.asarray(
            np.stack([row.ordered_turn.market_valid[:MAX_MARKET_ORDERS] for row in rows])
        ),
    )


def summarize_dynamic_replay_audit_v2(
    audit: DynamicReplayBatchAuditV2,
) -> dict[str, object]:
    host = jax.device_get(audit)
    valid = np.asarray(host.label_valid, dtype=np.bool_)
    static = np.asarray(host.static_present, dtype=np.bool_) & valid
    unit = np.asarray(host.after_unit_present, dtype=np.bool_) & valid
    dynamic = np.asarray(host.dynamic_present, dtype=np.bool_) & valid
    slot = np.asarray(host.label_slot, dtype=np.int16)
    reason = np.asarray(host.missing_reason, dtype=np.int8)
    requested = np.asarray(host.requested_amount, dtype=np.int32)
    filled = np.asarray(host.filled_amount, dtype=np.int32)
    executable = valid & (filled > 0)
    no_op = valid & (filled == 0)
    has_market = np.any(valid, axis=1)
    full_static = np.all((~valid) | static, axis=1) & has_market
    full_dynamic = np.all((~valid) | dynamic, axis=1) & has_market
    missing_reason_counts = Counter(
        MISSING_REASON_NAMES_V2[int(value)]
        for value in reason[valid & (~dynamic)]
    )
    missing_slot_counts = Counter(
        str(int(value)) for value in slot[valid & (~dynamic)]
    )
    order_slot_counts = Counter(str(int(value)) for value in slot[valid])
    return {
        "schema_version": DYNAMIC_REPLAY_AUDIT_SCHEMA_V2,
        "causality_contract": {
            "unit_substep_inputs": ["public_observation"],
            "market_substep_inputs": [
                "public_observation",
                "teacher_forced_unit_prefix",
                "teacher_forced_prior_market_prefix_only",
            ],
            "current_or_future_market_label_is_actor_input": False,
            "expert_quantity_is_applied_only_after_current_presence_check": True,
        },
        "coverage": {
            "samples": int(valid.shape[0]),
            "market_samples": int(np.sum(has_market)),
            "market_orders": int(np.sum(valid)),
            "nominal_executable_orders": int(np.sum(executable)),
            "expert_no_op_orders": int(np.sum(no_op)),
            "static_present_orders": int(np.sum(static)),
            "dynamic_present_orders": int(np.sum(dynamic)),
            "opened_after_unit_phase": int(np.sum(valid & (~static) & unit)),
            "opened_after_prior_market_prefix": int(
                np.sum(valid & (~unit) & dynamic)
            ),
            "still_not_present_orders": int(np.sum(valid & (~dynamic))),
            "dynamic_core_gap_executable_orders": int(
                np.sum(executable & (~dynamic))
            ),
            "dynamic_representation_rate_executable": float(
                np.sum(executable & dynamic) / max(np.sum(executable), 1)
            ),
            "fully_static_market_samples": int(np.sum(full_static)),
            "fully_dynamic_market_samples": int(np.sum(full_dynamic)),
            "samples_promoted_to_full": int(np.sum((~full_static) & full_dynamic)),
            "nominal_zero_fill_orders": int(np.sum(valid & (filled == 0))),
            "partially_filled_orders": int(
                np.sum(valid & (filled > 0) & (filled < requested))
            ),
            "requested_units": int(np.sum(requested[valid])),
            "nominal_filled_units": int(np.sum(filled[valid])),
        },
        "market_order_slot_counts": dict(sorted(order_slot_counts.items())),
        "remaining_missing_reason_counts": dict(
            sorted(missing_reason_counts.items())
        ),
        "remaining_missing_slot_counts": dict(sorted(missing_slot_counts.items())),
        "diagnostics": {
            "final_invalid_order_count": int(
                np.sum(np.asarray(host.final_invalid_order_count))
            ),
            "final_overflow_order_count": int(
                np.sum(np.asarray(host.final_overflow_order_count))
            ),
        },
    }
