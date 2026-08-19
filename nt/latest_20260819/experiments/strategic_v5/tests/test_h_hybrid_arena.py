from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax import load_event_bank, load_tables
from strategic_v5 import initialize_full_learned_params_v1
from strategic_v5.h_arena import (
    initialize_h_hybrid_carry_v1,
    make_h_hybrid_arena_rollout_v1,
)


TABLES = load_tables()


def test_hybrid_arena_runs_learned_in_both_seats_without_hard_errors() -> None:
    event_ids, bank = load_event_bank()
    events = jax.tree.map(lambda value: value[:2], bank)
    params = initialize_full_learned_params_v1(jax.random.key(70))
    for learned_player in (0, 1):
        carry = initialize_h_hybrid_carry_v1(
            jnp.asarray(event_ids[:2]), events, jax.random.key(71 + learned_player)
        )
        rollout = jax.jit(
            make_h_hybrid_arena_rollout_v1(
                learned_player=learned_player,
                rollout_steps=2,
                decision_interval=8,
            )
        )(carry, TABLES, params)
        assert bool(jnp.all(rollout.final_carry.environment_state.step == 2))
        diagnostics = rollout.final_carry.diagnostics
        hard = (
            diagnostics.invalid_raw_action_count
            + diagnostics.internal_resource_conflict_count
            + diagnostics.compiler_overlap_count
            + diagnostics.effect_mismatch_count
            + diagnostics.owner_inactive_count
            + diagnostics.deadline_missed_count
            + diagnostics.resource_unavailable_count
        )
        assert int(jnp.sum(hard)) == 0
