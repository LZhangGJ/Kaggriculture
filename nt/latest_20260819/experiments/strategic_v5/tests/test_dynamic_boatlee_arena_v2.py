from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax import load_event_bank, load_tables
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1
from strategic_v5.dynamic_boatlee_arena_v2 import (
    initialize_dynamic_boatlee_arena_carry_v2,
    make_dynamic_boatlee_arena_rollout_v2,
)
from strategic_v5.learned_v1 import initialize_full_learned_params_v1
from strategic_v5.learned_v2 import migrate_full_learned_params_v1_to_v2


def test_dynamic_boatlee_arena_smoke_both_seats():
    tables = load_tables()
    seed_bank, event_bank = load_event_bank()
    seeds = seed_bank[:2]
    events = jax.tree.map(lambda value: value[:2], event_bank)
    carry = initialize_dynamic_boatlee_arena_carry_v2(
        seeds,
        events,
        jnp.asarray([0, 1], dtype=jnp.int8),
        jax.random.key(7302),
    )
    params = migrate_full_learned_params_v1_to_v2(
        initialize_full_learned_params_v1(jax.random.key(7303)),
        jax.random.key(7304),
    )
    output = make_dynamic_boatlee_arena_rollout_v2(rollout_steps=2)(
        carry, tables, load_boatlee_trace_v1(), params
    )
    assert output.learner_margin.shape == (2,)
    assert output.learner_outcome.shape == (2,)
    assert output.invalid_market_orders.shape == (2,)
    assert output.overflow_market_orders.shape == (2,)
    assert bool(jnp.all(output.final_carry.environment_state.step == 2))
