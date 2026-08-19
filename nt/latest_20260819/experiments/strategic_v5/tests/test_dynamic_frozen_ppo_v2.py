from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax import Events, load_tables

from strategic_v5.dynamic_frozen_ppo_v2 import dynamic_frozen_sample_step_v2
from strategic_v5.dynamic_ppo_v2 import DynamicPPOCollectorConfigV2
from strategic_v5.dynamic_rollout_v2 import dynamic_full_step_v2
from strategic_v5.learned_v2 import initialize_full_learned_params_v2
from strategic_v5.rollout_v2 import initialize_full_learned_carry_v2


def _carry():
    seeds = jnp.asarray([11, 12], dtype=jnp.int32)
    events = Events(
        weed_spawn=jnp.zeros((2, 30, 200), dtype=jnp.bool_),
        shop_choice=jnp.zeros((2, 30, 201), dtype=jnp.int8),
    )
    return initialize_full_learned_carry_v2(seeds, events, jax.random.key(5))


def test_frozen_sample_matches_identical_policy_step_for_each_learner_seat():
    carry = _carry()
    params = initialize_full_learned_params_v2(jax.random.key(6))
    tables = load_tables()
    config = DynamicPPOCollectorConfigV2(deterministic=True)
    expected, expected_trace = jax.jit(
        lambda one_carry, one_tables, one_params: dynamic_full_step_v2(
            one_carry, one_tables, one_params, deterministic=True
        )
    )(carry, tables, params)
    for learner_player in (0, 1):
        actual, transition, audit = jax.jit(
            lambda one_carry, one_tables, learner, frozen: dynamic_frozen_sample_step_v2(
                one_carry,
                one_tables,
                learner,
                frozen,
                learner_player=learner_player,
                config=config,
            )
        )(carry, tables, params, params)
        jax.block_until_ready(actual.environment_state.money)
        assert all(
            bool(value)
            for value in jax.tree.leaves(
                jax.tree.map(
                    jnp.array_equal,
                    expected.environment_state,
                    actual.environment_state,
                )
            )
        )
        assert jnp.allclose(
            transition.old_logprob[:, 0], expected_trace.joint_logprob[:, learner_player]
        )
        assert jnp.allclose(
            transition.old_value[:, 0], expected_trace.value[:, learner_player]
        )
        assert jnp.array_equal(
            transition.reward[:, 0], expected_trace.reward[:, learner_player]
        )
        assert transition.unit_global_features.shape[:2] == (2, 1)
        assert int(jnp.sum(audit.invalid_market_orders)) == 0
        assert int(jnp.sum(audit.overflow_market_orders)) == 0
