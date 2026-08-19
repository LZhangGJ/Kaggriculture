from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import load_tables, reset
from kaggriculture_jax.types import Events

from strategic_v5.candidateformer_v3 import (
    CANDIDATE_FEATURE_DIM_V3,
    CANDIDATEFORMER_420K_V3,
    PLAN_FEATURE_DIM_V3,
    PLAN_TOKEN_COUNT_V3,
    _recover_semantic_ids,
    apply_candidateformer_market_v3,
    build_candidate_features_v3,
    build_plan_ledger_features_v3,
    candidateformer_parameter_count_v3,
    candidateformer_trainable_parameter_count_v3,
    encode_candidateformer_v3,
    initialize_candidateformer_v3,
)
from strategic_v5.e4_core import (
    build_full_core_candidates_v1,
    evaluate_full_core_feasibility_v1,
)
from strategic_v5.e5_econ import build_full_econ_features_v1
from strategic_v5.dynamic_rollout_v2 import make_dynamic_full_collector_v2
from strategic_v5.learned_v2 import apply_full_learned_model_v2
from strategic_v5.lifecycle import reset_controller_state_v1
from strategic_v5.opponent_v2 import (
    build_candidate_features_v2,
    build_global_features_v2,
)
from strategic_v5.replay_bc_v2 import broad_replay_bc_candidate_program_v2
from strategic_v5.rollout_v2 import initialize_full_learned_carry_v2


def _batch(batch_size: int = 2):
    states = jax.vmap(reset)(jnp.arange(batch_size, dtype=jnp.int32) + 91000)
    one = reset_controller_state_v1()
    controller = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape), one
    )
    tables = load_tables()
    candidates = build_full_core_candidates_v1(
        states,
        controller,
        tables,
        0,
        broad_replay_bc_candidate_program_v2(),
    )
    feasibility = evaluate_full_core_feasibility_v1(
        states, candidates, tables, 0
    )
    econ = build_full_econ_features_v1(
        states, candidates, feasibility, tables, 0
    )
    global_features = build_global_features_v2(
        states, 0, None, include_opponent=True, include_history=False
    )
    base = build_candidate_features_v2(
        states,
        candidates,
        feasibility,
        econ,
        0,
        tables,
        include_opponent=True,
    )
    features = build_candidate_features_v3(
        states, candidates, feasibility, econ, base
    )
    plan = build_plan_ledger_features_v3(
        states, controller, candidates, feasibility, econ, 0
    )
    return states, controller, candidates, feasibility, econ, global_features, features, plan


def _enable_delta_heads(params):
    params = copy.deepcopy(params)
    width = params["transformer"]["candidate_delta"]["kernel"].shape[0]
    params["transformer"]["candidate_delta"]["kernel"] = jnp.linspace(
        -0.03, 0.03, width, dtype=jnp.float32
    )[:, None]
    params["transformer"]["market_delta"]["kernel"] = jnp.linspace(
        0.02, -0.02, width, dtype=jnp.float32
    )[:, None]
    return params


def test_v3_relation_and_plan_features_are_exact_fixed_shape_and_finite():
    _, _, candidates, _, _, _, features, plan = _batch()
    assert features.shape == (2, 96, CANDIDATE_FEATURE_DIM_V3)
    assert plan.shape == (2, PLAN_TOKEN_COUNT_V3, PLAN_FEATURE_DIM_V3)
    assert np.all(np.isfinite(np.asarray(features)))
    assert np.all(np.isfinite(np.asarray(plan)))
    assert np.array_equal(
        np.asarray(features[..., 24] > 0.5), np.asarray(candidates.present)
    )
    owner, item, source = _recover_semantic_ids(features)
    assert np.array_equal(
        np.asarray(owner),
        np.asarray(jnp.clip(candidates.owner_unit + 1, 0, 33)),
    )
    assert np.array_equal(
        np.asarray(item),
        np.asarray(jnp.clip(candidates.item_id + 1, 0, 12)),
    )
    assert np.array_equal(
        np.asarray(source),
        np.asarray(jnp.clip(candidates.source + 1, 0, 9)),
    )


def test_v3_parameter_budget_is_small_and_frozen():
    params = initialize_candidateformer_v3(jax.random.key(3001))
    assert candidateformer_parameter_count_v3(params) == 408_989
    assert candidateformer_trainable_parameter_count_v3(params) == 358_809


def test_v3_zero_initialization_is_exact_old_50k_policy():
    _, _, candidates, _, _, global_features, features, plan = _batch()
    params = initialize_candidateformer_v3(jax.random.key(3002))
    _, output = encode_candidateformer_v3(
        params, global_features, plan, features, candidates.task_type
    )
    old = apply_full_learned_model_v2(
        params["baseline"], global_features, features[..., :24], candidates.task_type
    )
    np.testing.assert_array_equal(np.asarray(output.candidate_logits), np.asarray(old.candidate_logits))
    np.testing.assert_array_equal(np.asarray(output.stop_logit), np.asarray(old.stop_logit))
    np.testing.assert_array_equal(np.asarray(output.value), np.asarray(old.value))
    np.testing.assert_array_equal(
        np.asarray(output.opponent_task_logits), np.asarray(old.opponent_task_logits)
    )


def test_v3_candidate_order_is_equivariant_without_slot_embeddings():
    _, _, candidates, _, _, global_features, features, plan = _batch()
    params = _enable_delta_heads(
        initialize_candidateformer_v3(jax.random.key(3003))
    )
    _, original = encode_candidateformer_v3(
        params, global_features, plan, features, candidates.task_type
    )
    permutation = jnp.asarray(np.random.default_rng(3003).permutation(96))
    _, shuffled = encode_candidateformer_v3(
        params,
        global_features,
        plan,
        features[:, permutation],
        candidates.task_type[:, permutation],
    )
    np.testing.assert_allclose(
        np.asarray(shuffled.candidate_logits),
        np.asarray(original.candidate_logits[:, permutation]),
        # GPU attention reductions may accumulate in a different order after
        # candidate permutation.  The observed CUDA/JAX drift is ~4.3e-5;
        # this still enforces numerical equivariance without requiring a
        # bitwise-stable reduction order.
        rtol=5e-5,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(shuffled.stop_logit), np.asarray(original.stop_logit), atol=2e-6
    )
    np.testing.assert_allclose(
        np.asarray(shuffled.value), np.asarray(original.value), atol=2e-6
    )


def test_v3_attention_fuses_relationship_between_distinct_cards():
    _, _, candidates, _, _, global_features, features, plan = _batch(1)
    params = _enable_delta_heads(
        initialize_candidateformer_v3(jax.random.key(3004))
    )
    present = np.flatnonzero(np.asarray(candidates.present[0]))
    assert len(present) >= 2
    first, second = int(present[0]), int(present[1])
    _, before = encode_candidateformer_v3(
        params, global_features, plan, features, candidates.task_type
    )
    changed = features.at[0, second, 25].set(
        jnp.clip(features[0, second, 25] + 0.25, 0.0, 1.0)
    )
    _, after = encode_candidateformer_v3(
        params, global_features, plan, changed, candidates.task_type
    )
    cross_card_delta = float(
        jnp.abs(after.candidate_logits[0, first] - before.candidate_logits[0, first])
    )
    assert cross_card_delta > 1.0e-8


def test_v3_cached_market_head_uses_dynamic_features_and_keeps_old_fixed_point():
    _, _, candidates, _, _, global_features, features, plan = _batch()
    initial = initialize_candidateformer_v3(jax.random.key(3005))
    context, _ = encode_candidateformer_v3(
        initial, global_features, plan, features, candidates.task_type
    )
    market = features[:, :21]
    output = apply_candidateformer_market_v3(
        initial, context, market, candidates.task_type[:, :21]
    )
    old = apply_full_learned_model_v2(
        initial["baseline"], global_features, market[..., :24], candidates.task_type[:, :21]
    )
    np.testing.assert_array_equal(np.asarray(output.candidate_logits), np.asarray(old.candidate_logits))

    trained = _enable_delta_heads(initial)
    trained_context, _ = encode_candidateformer_v3(
        trained, global_features, plan, features, candidates.task_type
    )
    before = apply_candidateformer_market_v3(
        trained, trained_context, market, candidates.task_type[:, :21]
    )
    changed = market.at[:, 0, 32].add(0.5)
    after = apply_candidateformer_market_v3(
        trained, trained_context, changed, candidates.task_type[:, :21]
    )
    assert np.any(
        np.abs(np.asarray(after.candidate_logits[:, 0] - before.candidate_logits[:, 0]))
        > 1.0e-8
    )


def test_v3_full_and_cached_market_paths_jit():
    _, _, candidates, _, _, global_features, features, plan = _batch()
    params = initialize_candidateformer_v3(jax.random.key(3006))
    full = jax.jit(
        lambda p, g, q, c, t: encode_candidateformer_v3(p, g, q, c, t)
    )
    context, output = full(
        params, global_features, plan, features, candidates.task_type
    )
    jax.block_until_ready(output.candidate_logits)
    market = jax.jit(apply_candidateformer_market_v3)(
        params, context, features[:, :21], candidates.task_type[:, :21]
    )
    jax.block_until_ready(market.candidate_logits)
    assert market.candidate_logits.shape == (2, 21)


def test_v3_is_wired_into_dynamic_full_core_without_hard_errors():
    batch_size = 2
    seeds = jnp.arange(batch_size, dtype=jnp.int32) + 93000
    events = Events(
        weed_spawn=jnp.zeros((batch_size, 30, 200), dtype=jnp.bool_),
        shop_choice=jnp.zeros((batch_size, 30, 201), dtype=jnp.int8),
    )
    carry = initialize_full_learned_carry_v2(
        seeds, events, jax.random.key(3007)
    )
    params = initialize_candidateformer_v3(jax.random.key(3008))
    collector = jax.jit(
        make_dynamic_full_collector_v2(
            rollout_steps=4,
            deterministic=True,
            task_card_program=broad_replay_bc_candidate_program_v2(),
            # CandidateFormer ignores hand-written ROI as a hard gate.
            allow_nonpositive_econ=False,
        )
    )
    rollout = collector(carry, load_tables(), params)
    jax.block_until_ready(rollout.final_carry.environment_state.money)
    assert int(jnp.sum(rollout.transitions.invalid_market_orders)) == 0
    assert int(jnp.sum(rollout.transitions.overflow_market_orders)) == 0
    assert int(jnp.max(rollout.final_carry.environment_state.step)) == 4
