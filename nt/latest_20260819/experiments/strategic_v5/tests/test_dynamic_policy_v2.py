from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import load_tables, reset
from kaggriculture_jax.constants import MAX_MARKET_ORDERS
from strategic_v5.dynamic_policy_v2 import (
    _exact_dynamic_sell_revenue_v2,
    attach_dynamic_market_sequence_v2,
    recompute_dynamic_market_logprob_v2,
    select_dynamic_full_core_v2,
)
from strategic_v5.constants import TaskStatusV1
from strategic_v5.e2_core import attach_e2_candidate_batched_v1
from strategic_v5.e4_core import (
    build_full_core_candidates_v1,
    evaluate_full_core_feasibility_v1,
)
from strategic_v5.bc_v2 import (
    FullBCConfigV2,
    dynamic_market_bc_batch_from_trace_v2,
    dynamic_market_bc_loss_v2,
    initialize_full_bc_state_v2,
    make_dynamic_market_bc_update_v2,
)
from strategic_v5.turn_ledger_v2 import (
    apply_turn_market_order_v2,
    close_unit_phase_v2,
)
from kaggriculture_jax import empty_action
from kaggriculture_jax.constants import MarketOp
from strategic_v5.learned_v2 import (
    MODEL_FULL_V2,
    apply_full_learned_candidates_from_context_v2,
    encode_full_learned_context_v2,
    initialize_full_learned_params_v2,
)
from strategic_v5.lifecycle import reset_controller_state_v1
from strategic_v5.opponent_v2 import initialize_opponent_history_v2
from strategic_v5.replay_bc_v2 import broad_replay_bc_candidate_program_v2
from strategic_v5.replay_bc_v2 import ORDERED_STOP_SLOT_V3


def _controllers(batch_size: int):
    one = reset_controller_state_v1()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape), one
    )


_PLAYER_FIELDS = {
    "money",
    "tile_kind",
    "tile_crop",
    "tile_animal",
    "tile_origin_day",
    "tile_yield",
    "tile_neglect",
    "tile_max_lifespan",
    "tile_fertilized_until",
    "tile_pending_care",
    "tile_flags",
    "unit_pos",
    "unit_active",
    "unit_inventory",
    "unit_inventory_order",
    "unit_inventory_next_order",
    "shed",
    "seeds",
    "hires_today",
    "unlocked_count",
    "reward",
    "hand_cap_hits",
}


def _swap_players(states):
    replacements = {
        name: jnp.flip(value, axis=1)
        for name, value in zip(states._fields, states, strict=True)
        if name in _PLAYER_FIELDS
    }
    return states._replace(**replacements)


def test_cached_policy_context_matches_original_flax_model() -> None:
    params = initialize_full_learned_params_v2(jax.random.key(700))
    global_features = jax.random.normal(jax.random.key(701), (5, 232))
    candidate_features = jax.random.normal(jax.random.key(702), (5, 21, 24)).astype(
        jnp.float16
    )
    task_type = jax.random.randint(jax.random.key(703), (5, 21), 0, 21).astype(
        jnp.int8
    )
    expected = MODEL_FULL_V2.apply(
        {"params": params}, global_features, candidate_features, task_type
    )
    context = encode_full_learned_context_v2(params, global_features)
    actual = apply_full_learned_candidates_from_context_v2(
        params, context, candidate_features, task_type
    )
    for expected_value, actual_value in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(expected_value), np.asarray(actual_value), rtol=0.0, atol=1e-6
        )


def test_matrix_market_task_attach_matches_ordered_reference_scans() -> None:
    batch_size = 3
    states = jax.vmap(reset)(jnp.arange(9700, 9700 + batch_size, dtype=jnp.int32))
    controller = _controllers(batch_size)
    controller = controller._replace(
        market_tasks=controller.market_tasks._replace(
            status=controller.market_tasks.status.at[:, 0].set(
                TaskStatusV1.ACTIVE
            )
        )
    )
    tables = load_tables()
    candidates = build_full_core_candidates_v1(states, controller, tables, 0)
    feasibility = evaluate_full_core_feasibility_v1(
        states, candidates, tables, 0
    )
    selected = jnp.asarray(
        (
            (6, 6, 11, 0, 12, -1, -1, -1, -1, -1),
            (3, 4, 5, 1, 2, 7, 8, 9, 10, 20),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
        ),
        dtype=jnp.int16,
    )
    requested = jnp.asarray(
        (
            (2, 3, 1, 1, 7, 0, 0, 0, 0, 0),
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ),
        dtype=jnp.int16,
    )
    actual = attach_dynamic_market_sequence_v2(
        controller, candidates, feasibility, selected, requested, states.step
    )

    expected = controller
    batch = jnp.arange(batch_size, dtype=jnp.int32)
    for ordinal in range(MAX_MARKET_ORDERS):
        choice = selected[:, ordinal]
        safe = jnp.clip(choice.astype(jnp.int32), 0, 20)
        one_candidates = candidates._replace(
            quantity=candidates.quantity.at[batch, safe].set(
                requested[:, ordinal]
            )
        )
        one_feasibility = feasibility._replace(
            market_slots_required=feasibility.market_slots_required.at[
                batch, safe
            ].set(jnp.where(choice >= 0, 1, 0).astype(jnp.int8))
        )
        expected = attach_e2_candidate_batched_v1(
            expected, one_candidates, one_feasibility, choice, states.step
        )
    for expected_value, actual_value in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(
            np.asarray(expected_value), np.asarray(actual_value)
        )


def test_dynamic_policy_smoke_recompute_and_ordered_action_are_finite() -> None:
    batch_size = 3
    states = jax.vmap(reset)(jnp.arange(9300, 9300 + batch_size, dtype=jnp.int32))
    params = initialize_full_learned_params_v2(jax.random.key(1))
    decision = select_dynamic_full_core_v2(
        states,
        _controllers(batch_size),
        initialize_opponent_history_v2(states),
        load_tables(),
        params,
        0,
        jax.random.key(2),
        deterministic=False,
        task_card_program=broad_replay_bc_candidate_program_v2(),
        allow_nonpositive_econ=True,
    )
    assert decision.market_trace.selected_indices.shape == (
        batch_size,
        MAX_MARKET_ORDERS,
    )
    assert decision.market_trace.masks.shape == (
        batch_size,
        MAX_MARKET_ORDERS,
        22,
    )
    assert decision.action.market_op.shape == (batch_size, 2, MAX_MARKET_ORDERS)
    np.testing.assert_array_equal(
        np.asarray(decision.action.market_count[:, 0]),
        np.asarray(decision.ledger.market_count),
    )
    recomputed = recompute_dynamic_market_logprob_v2(
        params, decision.market_trace
    )
    np.testing.assert_allclose(
        np.asarray(recomputed),
        np.asarray(decision.market_trace.joint_logprob),
        rtol=2e-6,
        atol=2e-6,
    )
    assert bool(jnp.all(jnp.isfinite(recomputed)))
    assert int(jnp.max(decision.ledger.market_count)) <= MAX_MARKET_ORDERS


def test_dynamic_policy_market_prefix_never_selects_nonmarket_slots() -> None:
    states = jax.vmap(reset)(jnp.asarray((9401, 9402), dtype=jnp.int32))
    decision = select_dynamic_full_core_v2(
        states,
        _controllers(2),
        initialize_opponent_history_v2(states),
        load_tables(),
        initialize_full_learned_params_v2(jax.random.key(3)),
        1,
        jax.random.key(4),
        deterministic=False,
        task_card_program=broad_replay_bc_candidate_program_v2(),
        allow_nonpositive_econ=True,
    )
    selected = np.asarray(decision.market_trace.selected_indices)
    assert np.all((selected == -1) | ((selected >= 0) & (selected < 21)))
    unit_selected = np.asarray(
        decision.unit_selection.selected_candidate_indices
    )
    assert np.all((unit_selected == -1) | (unit_selected >= 21))


def test_lightweight_sell_quote_matches_exact_ledger_cash_delta() -> None:
    states = jax.vmap(reset)(jnp.asarray((9501,), dtype=jnp.int32))
    states = states._replace(
        shed=states.shed.at[0, 0, 7].set(13).at[0, 0, 8].set(9)
    )
    action = empty_action()
    action = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (1,) + value.shape), action
    )
    tables = load_tables()
    ledger, _ = close_unit_phase_v2(states, action, 0)
    quote = _exact_dynamic_sell_revenue_v2(ledger, tables)
    for item in (7, 8):
        before = ledger.money_nominal
        sold = apply_turn_market_order_v2(
            ledger, MarketOp.SELL, item, int(ledger.shed[0, item]), tables
        )
        assert int(quote[0, item]) == int(sold.money_nominal[0] - before[0])


def test_ordered_dynamic_market_bc_interface_runs_one_gpu_update() -> None:
    batch_size = 3
    states = jax.vmap(reset)(
        jnp.arange(9600, 9600 + batch_size, dtype=jnp.int32)
    )
    params = initialize_full_learned_params_v2(jax.random.key(11))
    decision = select_dynamic_full_core_v2(
        states,
        _controllers(batch_size),
        initialize_opponent_history_v2(states),
        load_tables(),
        params,
        0,
        jax.random.key(12),
        deterministic=True,
        task_card_program=broad_replay_bc_candidate_program_v2(),
        allow_nonpositive_econ=True,
    )
    replay_slots = jnp.where(
        decision.market_trace.selected_indices < 0,
        ORDERED_STOP_SLOT_V3,
        decision.market_trace.selected_indices,
    ).astype(jnp.int16)
    valid = jnp.ones_like(replay_slots, dtype=jnp.bool_)
    batch = dynamic_market_bc_batch_from_trace_v2(
        decision.market_trace,
        replay_slots,
        decision.market_trace.requested_quantity,
        valid,
    )
    loss, metrics = dynamic_market_bc_loss_v2(params, batch)
    assert bool(jnp.isfinite(loss))
    assert int(metrics.illegal_target_count) == 0
    assert int(metrics.supervised_substeps) == batch_size * MAX_MARKET_ORDERS

    state = initialize_full_bc_state_v2(params, FullBCConfigV2())
    update = jax.jit(make_dynamic_market_bc_update_v2())
    next_state, updated_metrics = update(state, batch)
    jax.block_until_ready((next_state, updated_metrics))
    assert bool(jnp.isfinite(updated_metrics.loss))
    assert bool(jnp.isfinite(updated_metrics.grad_norm))
    assert int(next_state.step) == 1
    changed = any(
        not np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree.leaves(state.params),
            jax.tree.leaves(next_state.params),
            strict=True,
        )
    )
    assert changed


def test_dynamic_policy_is_exactly_seat_equivariant_on_swapped_state() -> None:
    states = jax.vmap(reset)(jnp.asarray((9701, 9702), dtype=jnp.int32))
    states = states._replace(
        money=states.money.at[:, 0].set(jnp.asarray((4321, 5432))),
        shed=states.shed.at[:, 0, 7].set(jnp.asarray((3, 5))),
        seeds=states.seeds.at[:, 0, 2].set(jnp.asarray((2, 4))),
    )
    swapped = _swap_players(states)
    params = initialize_full_learned_params_v2(jax.random.key(21))
    program = broad_replay_bc_candidate_program_v2()
    original = select_dynamic_full_core_v2(
        states,
        _controllers(2),
        initialize_opponent_history_v2(states),
        load_tables(),
        params,
        0,
        jax.random.key(22),
        deterministic=True,
        task_card_program=program,
        allow_nonpositive_econ=True,
    )
    mirrored = select_dynamic_full_core_v2(
        swapped,
        _controllers(2),
        initialize_opponent_history_v2(swapped),
        load_tables(),
        params,
        1,
        jax.random.key(22),
        deterministic=True,
        task_card_program=program,
        allow_nonpositive_econ=True,
    )
    for name in ("unit_op", "unit_item", "unit_amount", "unit_count"):
        left = getattr(original.action, name)[:, 0]
        right = getattr(mirrored.action, name)[:, 1]
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    for name in ("market_op", "market_item", "market_amount", "market_count"):
        left = getattr(original.action, name)[:, 0]
        right = getattr(mirrored.action, name)[:, 1]
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    np.testing.assert_array_equal(
        np.asarray(original.unit_selection.selected_candidate_indices),
        np.asarray(mirrored.unit_selection.selected_candidate_indices),
    )
    np.testing.assert_array_equal(
        np.asarray(original.market_trace.selected_indices),
        np.asarray(mirrored.market_trace.selected_indices),
    )
    np.testing.assert_allclose(
        np.asarray(original.value), np.asarray(mirrored.value), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(original.market_trace.joint_logprob),
        np.asarray(mirrored.market_trace.joint_logprob),
        rtol=0,
        atol=0,
    )
