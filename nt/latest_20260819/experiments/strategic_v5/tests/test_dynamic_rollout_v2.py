from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import Events, load_event_bank, load_tables
from kaggriculture_jax.simulator import batched_project_unit_phase
from strategic_v5.dynamic_policy_v2 import (
    combine_dynamic_actions_v2,
    select_dynamic_full_core_v2,
    select_dynamic_unit_phase_v2,
)
from strategic_v5.dynamic_rollout_v2 import (
    make_dynamic_full_collector_v2,
    select_dynamic_full_core_pair_v2,
)
from strategic_v5.learned_v2 import initialize_full_learned_params_v2
from strategic_v5.replay_bc_v2 import broad_replay_bc_candidate_program_v2
from strategic_v5.rollout_v2 import initialize_full_learned_carry_v2


def test_dynamic_rollout_executes_ordered_actions_for_both_players() -> None:
    seed_values, bank = load_event_bank()
    batch_size = 2
    seeds = jnp.asarray(seed_values[:batch_size], dtype=jnp.int32)
    events = Events(bank.weed_spawn[:batch_size], bank.shop_choice[:batch_size])
    carry = initialize_full_learned_carry_v2(seeds, events, jax.random.key(7))
    collector = jax.jit(
        make_dynamic_full_collector_v2(
            rollout_steps=2,
            deterministic=True,
            task_card_program=broad_replay_bc_candidate_program_v2(),
            allow_nonpositive_econ=True,
        )
    )
    rollout = collector(
        carry, load_tables(), initialize_full_learned_params_v2(jax.random.key(8))
    )
    jax.block_until_ready(rollout)
    np.testing.assert_array_equal(
        np.asarray(rollout.final_carry.environment_state.step),
        np.full((batch_size,), 2),
    )
    assert rollout.transitions.market_count.shape == (2, batch_size, 2)
    assert bool(jnp.all(rollout.transitions.invalid_market_orders == 0))
    assert bool(jnp.all(rollout.transitions.overflow_market_orders == 0))
    assert bool(jnp.all(jnp.isfinite(rollout.transitions.joint_logprob)))


def test_paired_market_decode_matches_two_deterministic_seat_decodes() -> None:
    seed_values, bank = load_event_bank()
    batch_size = 2
    seeds = jnp.asarray(seed_values[:batch_size], dtype=jnp.int32)
    events = Events(bank.weed_spawn[:batch_size], bank.shop_choice[:batch_size])
    carry = initialize_full_learned_carry_v2(seeds, events, jax.random.key(70))
    tables = load_tables()
    params = initialize_full_learned_params_v2(jax.random.key(71))
    key0, key1, unit_key0, unit_key1 = jax.random.split(jax.random.key(72), 4)
    kwargs = dict(
        deterministic=True,
        task_card_program=broad_replay_bc_candidate_program_v2(),
        allow_nonpositive_econ=True,
    )
    unit0 = select_dynamic_unit_phase_v2(
        carry.environment_state,
        carry.player0_controller,
        carry.opponent_history,
        tables,
        params,
        0,
        unit_key0,
        **kwargs,
    )
    unit1 = select_dynamic_unit_phase_v2(
        carry.environment_state,
        carry.player1_controller,
        carry.opponent_history,
        tables,
        params,
        1,
        unit_key1,
        **kwargs,
    )
    unit_action = combine_dynamic_actions_v2(unit0.action, unit1.action)
    projected = batched_project_unit_phase(carry.environment_state, unit_action)
    separate0 = select_dynamic_full_core_v2(
        carry.environment_state,
        carry.player0_controller,
        carry.opponent_history,
        tables,
        params,
        0,
        key0,
        unit_decision=unit0,
        projected_unit_state=projected,
        **kwargs,
    )
    separate1 = select_dynamic_full_core_v2(
        carry.environment_state,
        carry.player1_controller,
        carry.opponent_history,
        tables,
        params,
        1,
        key1,
        unit_decision=unit1,
        projected_unit_state=projected,
        **kwargs,
    )
    paired0, paired1 = select_dynamic_full_core_pair_v2(
        carry.environment_state,
        carry.player0_controller,
        carry.player1_controller,
        carry.opponent_history,
        tables,
        params,
        key0,
        key1,
        unit0,
        unit1,
        projected,
        include_opponent=True,
        include_history=False,
        **kwargs,
    )
    for separate, paired in ((separate0, paired0), (separate1, paired1)):
        np.testing.assert_array_equal(
            np.asarray(separate.action.market_op), np.asarray(paired.action.market_op)
        )
        np.testing.assert_array_equal(
            np.asarray(separate.action.market_item),
            np.asarray(paired.action.market_item),
        )
        np.testing.assert_array_equal(
            np.asarray(separate.action.market_amount),
            np.asarray(paired.action.market_amount),
        )
        np.testing.assert_array_equal(
            np.asarray(separate.ledger.money_nominal),
            np.asarray(paired.ledger.money_nominal),
        )
        np.testing.assert_array_equal(
            np.asarray(separate.market_trace.selected_indices),
            np.asarray(paired.market_trace.selected_indices),
        )
