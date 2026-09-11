from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import load_event_bank, load_tables, reset
from strategic_v5 import (
    CANDIDATE_FEATURE_DIM_V1,
    GLOBAL_FEATURE_DIM_V1,
    FullPPOConfigV1,
    FullTimelineV1,
    apply_full_learned_model_v1,
    build_candidate_features_v1,
    build_full_core_candidates_v1,
    build_full_econ_features_v1,
    build_global_features_v1,
    compute_timeline_gae_v1,
    evaluate_full_core_feasibility_v1,
    full_parameter_count_v1,
    initialize_full_learned_carry_v1,
    initialize_full_learned_params_v1,
    initialize_full_ppo_state_v1,
    make_full_learned_collector_v1,
    make_full_strided_collector_v1,
    make_full_ppo_update_v1,
    recompute_full_logprob_v1,
    reset_controller_state_v1,
)


TABLES = load_tables()


def _batch_states(count: int = 2):
    return jax.vmap(reset)(jnp.arange(51, 51 + count, dtype=jnp.int32))


def test_compact_features_cover_full_shapes_and_are_finite() -> None:
    states = _batch_states()
    controller = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (2,) + value.shape),
        reset_controller_state_v1(),
    )
    candidates = build_full_core_candidates_v1(states, controller, TABLES, 0)
    feasibility = evaluate_full_core_feasibility_v1(states, candidates, TABLES, 0)
    econ = build_full_econ_features_v1(states, candidates, feasibility, TABLES, 0)
    global_features = build_global_features_v1(states, 0)
    candidate_features = build_candidate_features_v1(
        states, candidates, feasibility, econ, 0
    )
    assert global_features.shape == (2, GLOBAL_FEATURE_DIM_V1)
    assert candidate_features.shape == (2, 96, CANDIDATE_FEATURE_DIM_V1)
    assert candidate_features.dtype == jnp.float16
    assert bool(jnp.all(jnp.isfinite(global_features)))
    assert bool(jnp.all(jnp.isfinite(candidate_features)))


def test_global_features_do_not_leak_opponent_private_shed() -> None:
    states = _batch_states(1)
    changed = states._replace(shed=states.shed.at[0, 1, 7].set(99))
    np.testing.assert_array_equal(
        np.asarray(build_global_features_v1(states, 0)),
        np.asarray(build_global_features_v1(changed, 0)),
    )
    assert not np.array_equal(
        np.asarray(build_global_features_v1(states, 1)),
        np.asarray(build_global_features_v1(changed, 1)),
    )


def test_zero_initialized_policy_is_small_and_uses_econ_prior() -> None:
    params = initialize_full_learned_params_v1(jax.random.key(3))
    assert full_parameter_count_v1(params) < 1_200_000
    global_features = jnp.zeros((2, GLOBAL_FEATURE_DIM_V1), dtype=jnp.float32)
    candidate_features = jnp.zeros(
        (2, 96, CANDIDATE_FEATURE_DIM_V1), dtype=jnp.float16
    ).at[:, :, 0].set(jnp.linspace(-1.0, 1.0, 96, dtype=jnp.float16))
    task = jnp.zeros((2, 96), dtype=jnp.int8)
    output = apply_full_learned_model_v1(
        params, global_features, candidate_features, task
    )
    np.testing.assert_allclose(
        np.asarray(output.candidate_logits),
        np.asarray(candidate_features[:, :, 0], dtype=np.float32) * 20.0,
        atol=1e-5,
    )
    np.testing.assert_array_equal(np.asarray(output.stop_logit), np.zeros((2,)))
    np.testing.assert_array_equal(np.asarray(output.value), np.zeros((2,)))


def test_rollout_recomputed_logprob_and_real_ppo_update() -> None:
    event_ids, bank = load_event_bank()
    events = jax.tree.map(lambda value: value[:4], bank)
    carry = initialize_full_learned_carry_v1(
        jnp.asarray(event_ids[:4]), events, jax.random.key(4)
    )
    params = initialize_full_learned_params_v1(jax.random.key(5))
    collector = jax.jit(make_full_learned_collector_v1(rollout_steps=2))
    rollout = collector(carry, TABLES, params)
    transition = jax.tree.map(lambda value: value.reshape((-1,) + value.shape[3:]), rollout.transitions)
    output = apply_full_learned_model_v1(
        params,
        transition.global_features,
        transition.candidate_features,
        transition.candidate_task_type,
    )
    replay = recompute_full_logprob_v1(
        output.candidate_logits,
        output.stop_logit,
        transition.task_masks,
        transition.task_selected_indices,
    )
    np.testing.assert_allclose(
        np.asarray(replay),
        np.asarray(rollout.transitions.old_logprob).reshape(-1),
        atol=2e-5,
    )

    config = FullPPOConfigV1(learning_rate=1e-4, minibatch_size=8)
    state = initialize_full_ppo_state_v1(params, config)
    update = jax.jit(make_full_ppo_update_v1(config, sample_count=16))
    training_transitions = rollout.transitions._replace(
        reward=rollout.transitions.reward.at[-1, :, 0].set(1.0).at[-1, :, 1].set(-1.0),
        done=rollout.transitions.done.at[-1].set(True),
    )
    next_state, metrics, _ = update(
        state,
        training_transitions,
        jnp.zeros_like(rollout.bootstrap_value),
        jax.random.key(6),
    )
    assert int(next_state.step) == 2
    assert all(bool(jnp.isfinite(value)) for value in jax.tree.leaves(metrics))
    changed = any(
        not np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree.leaves(state.params), jax.tree.leaves(next_state.params), strict=True
        )
    )
    assert changed


def test_strided_collector_keeps_full_timeline_and_exact_policy_snapshots() -> None:
    event_ids, bank = load_event_bank()
    events = jax.tree.map(lambda value: value[:4], bank)
    carry = initialize_full_learned_carry_v1(
        jnp.asarray(event_ids[:4]), events, jax.random.key(7)
    )
    params = initialize_full_learned_params_v1(jax.random.key(8))
    collector = jax.jit(
        make_full_strided_collector_v1(
            rollout_steps=17, sample_stride=8, decision_interval=8
        )
    )
    rollout = collector(carry, TABLES, params)
    assert rollout.timeline.old_value.shape == (17, 4, 2)
    assert rollout.transitions.old_logprob.shape == (3, 4, 2)
    assert np.all(np.asarray(rollout.final_carry.environment_state.step) == 17)
    flattened = jax.tree.map(
        lambda value: value.reshape((-1,) + value.shape[3:]), rollout.transitions
    )
    output = apply_full_learned_model_v1(
        params,
        flattened.global_features,
        flattened.candidate_features,
        flattened.candidate_task_type,
    )
    replay = recompute_full_logprob_v1(
        output.candidate_logits,
        output.stop_logit,
        flattened.task_masks,
        flattened.task_selected_indices,
    )
    np.testing.assert_allclose(
        np.asarray(replay),
        np.asarray(rollout.transitions.old_logprob).reshape(-1),
        atol=2e-5,
    )


def test_timeline_gae_propagates_terminal_reward_over_unsampled_steps() -> None:
    old_value = jnp.zeros((9, 2, 2), dtype=jnp.float32)
    reward = jnp.zeros_like(old_value).at[-1, :, 0].set(1.0).at[-1, :, 1].set(-1.0)
    done = jnp.zeros_like(old_value, dtype=jnp.bool_).at[-1].set(True)
    advantage, returns = compute_timeline_gae_v1(
        FullTimelineV1(old_value, reward, done),
        jnp.zeros((2, 2), dtype=jnp.float32),
        gamma=0.9,
        gae_lambda=1.0,
    )
    np.testing.assert_allclose(np.asarray(advantage[0, :, 0]), 0.9**8, atol=1e-6)
    np.testing.assert_allclose(np.asarray(advantage[0, :, 1]), -(0.9**8), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(returns), np.asarray(advantage))


def test_terminal_step_718_forced_decision_snapshot_is_saved() -> None:
    event_ids, bank = load_event_bank()
    events = jax.tree.map(lambda value: value[:2], bank)
    carry = initialize_full_learned_carry_v1(
        jnp.asarray(event_ids[:2]), events, jax.random.key(9)
    )
    carry = carry._replace(
        environment_state=carry.environment_state._replace(
            step=jnp.full(
                (2,), 704, dtype=carry.environment_state.step.dtype
            )
        )
    )
    params = initialize_full_learned_params_v1(jax.random.key(10))
    collector = jax.jit(
        make_full_strided_collector_v1(
            rollout_steps=15,
            sample_stride=8,
            decision_interval=8,
            include_final_sample=True,
        )
    )
    rollout = collector(carry, TABLES, params)
    assert rollout.transitions.old_logprob.shape == (3, 2, 2)
    assert np.all(np.asarray(rollout.final_carry.environment_state.step) == 719)
    assert np.all(np.asarray(rollout.final_carry.environment_state.done))
