from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import load_event_bank, load_tables, reset
from kaggriculture_jax.constants import TileKind
from strategic_v5.e4_core import (
    build_full_core_candidates_v1,
    evaluate_full_core_feasibility_v1,
)
from strategic_v5.e5_econ import build_full_econ_features_v1
from strategic_v5.bc_v2 import (
    FullBCBatchV2,
    FullBCConfigV2,
    FullBCListwiseBatchV2,
    full_bc_loss_v2,
    full_bc_listwise_loss_v2,
    initialize_full_bc_state_v2,
    make_full_bc_listwise_update_v2,
    make_full_bc_update_v2,
)
from strategic_v5.learned_v1 import (
    apply_full_learned_model_v1,
    build_candidate_features_v1,
    build_global_features_v1,
    initialize_full_learned_params_v1,
    initialize_full_learned_carry_v1,
    make_full_learned_collector_v1,
    recompute_full_logprob_v1,
)
from strategic_v5.learned_v2 import (
    apply_full_learned_model_v2,
    full_parameter_count_v2,
    migrate_full_learned_params_v1_to_v2,
)
from strategic_v5.lifecycle import reset_controller_state_v1
from strategic_v5.opponent_v2 import (
    CANDIDATE_FEATURE_DIM_V2,
    CANDIDATE_INTERACTION_FEATURE_DIM_V2,
    GLOBAL_FEATURE_DIM_V2,
    OPPONENT_CURRENT_FEATURE_DIM_V2,
    OPPONENT_HISTORY_FEATURE_DIM_V2,
    PUBLIC_SNAPSHOT_DIM_V2,
    build_candidate_features_v2,
    build_global_features_v2,
    build_opponent_current_features_v2,
    build_opponent_history_features_v2,
    build_public_snapshot_v2,
    initialize_opponent_history_v2,
    update_opponent_history_v2,
)
from strategic_v5.rollout_v2 import (
    initialize_full_learned_carry_v2,
    make_full_learned_collector_v2,
)
from strategic_v5.ppo_v1 import FullPPOConfigV1
from strategic_v5.ppo_v2 import initialize_full_ppo_state_v2, make_full_ppo_update_v2
from strategic_v5.replay_bc_v2 import (
    broad_replay_bc_candidate_program_v2,
    build_replay_bc_feature_batch_v2,
)
from strategic_v5.g_arena_v2 import make_full_paired_arena_rollout_v2


TABLES = load_tables()
PLAYER_FIELDS = {
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


def _states(count: int = 2):
    return jax.vmap(reset)(jnp.arange(800, 800 + count, dtype=jnp.int32))


def _controllers(count: int):
    one = reset_controller_state_v1()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (count,) + value.shape), one
    )


def _swap_players(states):
    replacements = {}
    order = jnp.asarray((1, 0), dtype=jnp.int32)
    for name, value in zip(states._fields, states, strict=True):
        if name in PLAYER_FIELDS:
            replacements[name] = jnp.take(value, order, axis=1)
    return states._replace(**replacements)


def _public_opponent_crop(states):
    return states._replace(
        tile_kind=states.tile_kind.at[0, 1, 0, 0].set(TileKind.PLANT),
        tile_crop=states.tile_crop.at[0, 1, 0, 0].set(0),
        tile_origin_day=states.tile_origin_day.at[0, 1, 0, 0].set(0),
        tile_yield=states.tile_yield.at[0, 1, 0, 0].set(3),
        tile_neglect=states.tile_neglect.at[0, 1, 0, 0].set(1),
        tile_max_lifespan=states.tile_max_lifespan.at[0, 1, 0, 0].set(48),
    )


def test_v2_shapes_prefixes_and_finiteness() -> None:
    states = _states()
    history = initialize_opponent_history_v2(states)
    global_v2 = build_global_features_v2(states, 0, history)
    assert global_v2.shape == (2, GLOBAL_FEATURE_DIM_V2)
    assert history.previous_snapshot.shape == (2, 2, PUBLIC_SNAPSHOT_DIM_V2)
    assert bool(jnp.all(jnp.isfinite(global_v2)))
    np.testing.assert_array_equal(
        np.asarray(global_v2[:, :96]), np.asarray(build_global_features_v1(states, 0))
    )

    controller = _controllers(2)
    candidates = build_full_core_candidates_v1(states, controller, TABLES, 0)
    feasibility = evaluate_full_core_feasibility_v1(states, candidates, TABLES, 0)
    econ = build_full_econ_features_v1(states, candidates, feasibility, TABLES, 0)
    candidate_v1 = build_candidate_features_v1(
        states, candidates, feasibility, econ, 0
    )
    candidate_v2 = build_candidate_features_v2(
        states, candidates, feasibility, econ, 0, TABLES
    )
    assert candidate_v2.shape == (2, 96, CANDIDATE_FEATURE_DIM_V2)
    assert candidate_v2.dtype == jnp.float16
    assert bool(jnp.all(jnp.isfinite(candidate_v2)))
    np.testing.assert_array_equal(
        np.asarray(candidate_v2[:, :, :16]), np.asarray(candidate_v1)
    )
    assert candidate_v2.shape[-1] == 16 + CANDIDATE_INTERACTION_FEATURE_DIM_V2


def test_candidate_uses_official_post_opponent_supply_price_scenario() -> None:
    baseline = _states(1)
    pressure = baseline._replace(
        tile_kind=baseline.tile_kind.at[0, 1, 0, 0].set(TileKind.PLANT),
        tile_crop=baseline.tile_crop.at[0, 1, 0, 0].set(0),
        tile_origin_day=baseline.tile_origin_day.at[0, 1, 0, 0].set(0),
        tile_yield=baseline.tile_yield.at[0, 1, 0, 0].set(100),
        tile_max_lifespan=baseline.tile_max_lifespan.at[0, 1, 0, 0].set(719),
    )

    def features(states):
        controller = _controllers(1)
        candidates = build_full_core_candidates_v1(states, controller, TABLES, 0)
        feasibility = evaluate_full_core_feasibility_v1(
            states, candidates, TABLES, 0
        )
        econ = build_full_econ_features_v1(
            states, candidates, feasibility, TABLES, 0
        )
        return build_candidate_features_v2(
            states, candidates, feasibility, econ, 0, TABLES
        )

    without_pressure = features(baseline)
    with_pressure = features(pressure)
    wheat_seed_slot = 6
    post_price_ratio = 16 + 6
    profit_loss = 16 + 7
    assert float(with_pressure[0, wheat_seed_slot, post_price_ratio]) <= float(
        without_pressure[0, wheat_seed_slot, post_price_ratio]
    )
    assert float(with_pressure[0, wheat_seed_slot, profit_loss]) >= float(
        without_pressure[0, wheat_seed_slot, profit_loss]
    )


def test_v2_never_reads_opponent_private_state() -> None:
    states = _states(1)
    baseline_current = build_opponent_current_features_v2(states, 0)
    baseline_snapshot = build_public_snapshot_v2(states, 0)
    baseline_global = build_global_features_v2(states, 0)
    changed = states._replace(
        shed=states.shed.at[0, 1, :].set(99),
        seeds=states.seeds.at[0, 1, :].set(77),
        unit_inventory=states.unit_inventory.at[0, 1, :, :].set(55),
        unit_inventory_order=states.unit_inventory_order.at[0, 1, :, :].set(3),
        unit_inventory_next_order=states.unit_inventory_next_order.at[0, 1, :].set(7),
    )
    np.testing.assert_array_equal(
        np.asarray(baseline_current),
        np.asarray(build_opponent_current_features_v2(changed, 0)),
    )
    np.testing.assert_array_equal(
        np.asarray(baseline_snapshot),
        np.asarray(build_public_snapshot_v2(changed, 0)),
    )
    np.testing.assert_array_equal(
        np.asarray(baseline_global),
        np.asarray(build_global_features_v2(changed, 0)),
    )


def test_public_opponent_change_is_visible_and_history_is_actor_aligned() -> None:
    states = _states(1)
    history = initialize_opponent_history_v2(states)
    changed = _public_opponent_crop(states)
    current_before = build_opponent_current_features_v2(states, 0)
    current_after = build_opponent_current_features_v2(changed, 0)
    assert not np.array_equal(np.asarray(current_before), np.asarray(current_after))

    next_history = update_opponent_history_v2(history, changed)
    features0 = build_opponent_history_features_v2(
        next_history, batch_size=1, player=0
    )
    features1 = build_opponent_history_features_v2(
        next_history, batch_size=1, player=1
    )
    assert features0.shape == (1, OPPONENT_HISTORY_FEATURE_DIM_V2)
    assert np.count_nonzero(np.asarray(features0)) > 0
    # The changed farm belongs to player 1, so it is the opponent of observer 0.
    # Observer 1 sees its own change only through private/own V1 fields, not the
    # opponent-history branch.
    assert np.count_nonzero(np.asarray(features1[:, :27])) == 0


def test_private_change_does_not_enter_history() -> None:
    states = _states(1)
    history = initialize_opponent_history_v2(states)
    changed = states._replace(
        shed=states.shed.at[0, 1, 0].set(91),
        seeds=states.seeds.at[0, 1, 0].set(83),
        unit_inventory=states.unit_inventory.at[0, 1, 0, 0].set(72),
    )
    next_history = update_opponent_history_v2(history, changed)
    np.testing.assert_array_equal(
        np.asarray(next_history.recent_delta),
        np.zeros_like(np.asarray(next_history.recent_delta)),
    )


def test_self_only_ablation_zeros_every_v2_suffix() -> None:
    states = _public_opponent_crop(_states(1))
    history = initialize_opponent_history_v2(states)
    features = build_global_features_v2(
        states,
        0,
        history,
        include_opponent=False,
        include_history=False,
    )
    np.testing.assert_array_equal(
        np.asarray(features[:, :96]), np.asarray(build_global_features_v1(states, 0))
    )
    np.testing.assert_array_equal(
        np.asarray(features[:, 96:]), np.zeros((1, GLOBAL_FEATURE_DIM_V2 - 96))
    )


def test_deployment_default_hides_history_but_offline_bc_keeps_it() -> None:
    states = _states(2)
    history = initialize_opponent_history_v2(states)
    changed = states._replace(
        tile_kind=states.tile_kind.at[:, 1, 0, 0].set(TileKind.PLANT),
        tile_crop=states.tile_crop.at[:, 1, 0, 0].set(0),
        tile_origin_day=states.tile_origin_day.at[:, 1, 0, 0].set(0),
        tile_yield=states.tile_yield.at[:, 1, 0, 0].set(3),
    )
    changed_history = update_opponent_history_v2(history, changed)

    deployed = build_global_features_v2(changed, 0, changed_history)
    history_start = 96 + OPPONENT_CURRENT_FEATURE_DIM_V2
    np.testing.assert_array_equal(
        np.asarray(deployed[:, history_start:]),
        np.zeros((2, OPPONENT_HISTORY_FEATURE_DIM_V2), dtype=np.float32),
    )

    offline = build_replay_bc_feature_batch_v2(changed, changed_history, TABLES)
    assert np.count_nonzero(np.asarray(offline.global_features[:, history_start:])) > 0


def test_seat_swap_preserves_actor_view() -> None:
    states = _public_opponent_crop(_states(1))
    states = states._replace(
        money=states.money.at[0, 0].set(1234).at[0, 1].set(5678),
        shed=states.shed.at[0, 0, 0].set(4).at[0, 1, 1].set(7),
    )
    swapped = _swap_players(states)
    history = initialize_opponent_history_v2(states)
    swapped_history = initialize_opponent_history_v2(swapped)
    np.testing.assert_allclose(
        np.asarray(build_global_features_v2(states, 0, history)),
        np.asarray(build_global_features_v2(swapped, 1, swapped_history)),
        atol=0.0,
    )


def test_jitted_feature_builders_keep_frozen_shapes() -> None:
    states = _public_opponent_crop(_states(2))
    history = initialize_opponent_history_v2(states)
    build_global = jax.jit(lambda state, hist: build_global_features_v2(state, 0, hist))
    build_current = jax.jit(lambda state: build_opponent_current_features_v2(state, 0))
    global_features = build_global(states, history)
    current_features = build_current(states)
    assert global_features.shape == (2, GLOBAL_FEATURE_DIM_V2)
    assert current_features.shape == (2, OPPONENT_CURRENT_FEATURE_DIM_V2)


def test_v1_checkpoint_migration_is_initially_policy_equivalent() -> None:
    states = _public_opponent_crop(_states(2))
    controller = _controllers(2)
    candidates = build_full_core_candidates_v1(states, controller, TABLES, 0)
    feasibility = evaluate_full_core_feasibility_v1(states, candidates, TABLES, 0)
    econ = build_full_econ_features_v1(states, candidates, feasibility, TABLES, 0)
    global_v1 = build_global_features_v1(states, 0)
    candidate_v1 = build_candidate_features_v1(
        states, candidates, feasibility, econ, 0
    )
    global_v2 = build_global_features_v2(
        states, 0, include_opponent=False, include_history=False
    )
    candidate_v2 = build_candidate_features_v2(
        states,
        candidates,
        feasibility,
        econ,
        0,
        TABLES,
        include_opponent=False,
    )
    params_v1 = initialize_full_learned_params_v1(jax.random.key(91))
    params_v2 = migrate_full_learned_params_v1_to_v2(params_v1, jax.random.key(92))
    output_v1 = apply_full_learned_model_v1(
        params_v1, global_v1, candidate_v1, candidates.task_type
    )
    output_v2 = apply_full_learned_model_v2(
        params_v2, global_v2, candidate_v2, candidates.task_type
    )
    np.testing.assert_allclose(
        np.asarray(output_v2.candidate_logits),
        np.asarray(output_v1.candidate_logits),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(output_v2.stop_logit),
        np.asarray(output_v1.stop_logit),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(output_v2.value), np.asarray(output_v1.value), rtol=0.0, atol=1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(output_v2.opponent_task_logits), np.zeros((2, 21), dtype=np.float32)
    )
    assert full_parameter_count_v2(params_v2) < 1_200_000


def test_v2_gpu_native_collector_and_logprob_recompute() -> None:
    event_ids, bank = load_event_bank()
    events = jax.tree.map(lambda value: value[:2], bank)
    params_v1 = initialize_full_learned_params_v1(jax.random.key(101))
    params_v2 = migrate_full_learned_params_v1_to_v2(params_v1, jax.random.key(102))
    carry = initialize_full_learned_carry_v2(
        jnp.asarray(event_ids[:2]), events, jax.random.key(103)
    )
    collector = jax.jit(
        make_full_learned_collector_v2(
            rollout_steps=2, deterministic=True, decision_interval=1
        )
    )
    rollout = collector(carry, TABLES, params_v2)
    assert rollout.transitions.global_features.shape == (2, 2, 2, GLOBAL_FEATURE_DIM_V2)
    assert rollout.transitions.candidate_features.shape == (
        2,
        2,
        2,
        96,
        CANDIDATE_FEATURE_DIM_V2,
    )
    flat = jax.tree.map(
        lambda value: value.reshape((-1,) + value.shape[3:]), rollout.transitions
    )
    output = apply_full_learned_model_v2(
        params_v2,
        flat.global_features,
        flat.candidate_features,
        flat.candidate_task_type,
    )
    replay = recompute_full_logprob_v1(
        output.candidate_logits,
        output.stop_logit,
        flat.task_masks,
        flat.task_selected_indices,
    )
    np.testing.assert_allclose(
        np.asarray(replay),
        np.asarray(rollout.transitions.old_logprob).reshape(-1),
        atol=2e-5,
    )


def test_v2_self_only_rollout_matches_v1_after_migration() -> None:
    event_ids, bank = load_event_bank()
    events = jax.tree.map(lambda value: value[:2], bank)
    seeds = jnp.asarray(event_ids[:2])
    params_v1 = initialize_full_learned_params_v1(jax.random.key(111))
    params_v2 = migrate_full_learned_params_v1_to_v2(params_v1, jax.random.key(112))
    carry_v1 = initialize_full_learned_carry_v1(seeds, events, jax.random.key(113))
    carry_v2 = initialize_full_learned_carry_v2(seeds, events, jax.random.key(113))
    collect_v1 = jax.jit(
        make_full_learned_collector_v1(
            rollout_steps=3, deterministic=True, decision_interval=1
        )
    )
    collect_v2 = jax.jit(
        make_full_learned_collector_v2(
            rollout_steps=3,
            deterministic=True,
            decision_interval=1,
            include_opponent=False,
            include_history=False,
        )
    )
    rollout_v1 = collect_v1(carry_v1, TABLES, params_v1)
    rollout_v2 = collect_v2(carry_v2, TABLES, params_v2)
    for left, right in zip(
        jax.tree.leaves(rollout_v1.final_carry.environment_state),
        jax.tree.leaves(rollout_v2.final_carry.environment_state),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    np.testing.assert_allclose(
        np.asarray(rollout_v1.transitions.old_logprob),
        np.asarray(rollout_v2.transitions.old_logprob),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(rollout_v1.bootstrap_value),
        np.asarray(rollout_v2.bootstrap_value),
        atol=1e-6,
    )


def test_v2_real_ppo_update_is_finite_and_changes_parameters() -> None:
    event_ids, bank = load_event_bank()
    events = jax.tree.map(lambda value: value[:2], bank)
    params_v1 = initialize_full_learned_params_v1(jax.random.key(121))
    params_v2 = migrate_full_learned_params_v1_to_v2(params_v1, jax.random.key(122))
    carry = initialize_full_learned_carry_v2(
        jnp.asarray(event_ids[:2]), events, jax.random.key(123)
    )
    rollout = jax.jit(
        make_full_learned_collector_v2(
            rollout_steps=2, deterministic=False, decision_interval=1
        )
    )(carry, TABLES, params_v2)
    transitions = rollout.transitions._replace(
        reward=rollout.transitions.reward.at[-1, :, 0].set(1.0).at[-1, :, 1].set(-1.0),
        done=rollout.transitions.done.at[-1].set(True),
    )
    config = FullPPOConfigV1(learning_rate=1e-4, minibatch_size=4)
    state = initialize_full_ppo_state_v2(params_v2, config)
    update = jax.jit(make_full_ppo_update_v2(config, sample_count=8))
    next_state, metrics, _ = update(
        state, transitions, jnp.zeros_like(rollout.bootstrap_value), jax.random.key(124)
    )
    assert int(next_state.step) == 2
    assert all(bool(jnp.isfinite(value)) for value in jax.tree.leaves(metrics))
    assert any(
        not np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree.leaves(state.params), jax.tree.leaves(next_state.params), strict=True
        )
    )


def test_v2_teacher_forced_bc_and_opponent_auxiliary_head_update() -> None:
    event_ids, bank = load_event_bank()
    events = jax.tree.map(lambda value: value[:2], bank)
    params_v1 = initialize_full_learned_params_v1(jax.random.key(131))
    params_v2 = migrate_full_learned_params_v1_to_v2(params_v1, jax.random.key(132))
    carry = initialize_full_learned_carry_v2(
        jnp.asarray(event_ids[:2]), events, jax.random.key(133)
    )
    rollout = jax.jit(
        make_full_learned_collector_v2(
            rollout_steps=2, deterministic=True, decision_interval=1
        )
    )(carry, TABLES, params_v2)
    transition = rollout.transitions
    sample_count = 8
    batch = FullBCBatchV2(
        global_features=transition.global_features.reshape((sample_count, -1)),
        candidate_features=transition.candidate_features.reshape(
            (sample_count, 96, CANDIDATE_FEATURE_DIM_V2)
        ),
        candidate_task_type=transition.candidate_task_type.reshape((sample_count, 96)),
        task_masks=transition.task_masks.reshape((sample_count, 8, 97)),
        task_selected_indices=transition.task_selected_indices.reshape((sample_count, 8)),
        selection_weight=jnp.ones((sample_count, 8), dtype=jnp.float32),
        opponent_task_type=jnp.arange(sample_count, dtype=jnp.int32) % 21,
        sample_weight=jnp.ones((sample_count,), dtype=jnp.float32),
    )
    config = FullBCConfigV2(learning_rate=1e-3, opponent_aux_weight=0.1)
    initial_loss, initial_metrics = full_bc_loss_v2(params_v2, batch, config)
    assert bool(jnp.isfinite(initial_loss))
    assert float(initial_metrics.illegal_target_count) == 0.0
    state = initialize_full_bc_state_v2(params_v2, config)
    state, metrics = jax.jit(make_full_bc_update_v2(config))(state, batch)
    assert all(bool(jnp.isfinite(value)) for value in jax.tree.leaves(metrics))
    assert float(metrics.grad_norm) > 0.0
    opponent_kernel = state.params["opponent_task"]["kernel"]
    assert np.count_nonzero(np.asarray(opponent_kernel)) > 0


def test_broad_bc_enumeration_is_action_agnostic_and_exposes_composite_buys() -> None:
    states = _states(2)
    history = initialize_opponent_history_v2(states)
    program = broad_replay_bc_candidate_program_v2()
    features = build_replay_bc_feature_batch_v2(
        states,
        history,
        TABLES,
        program,
        allow_nonpositive_econ=True,
    )
    # Wheat, all three animals, all five seed families and hire are enumerated
    # from the schema even before an animal structure exists.
    np.testing.assert_array_equal(
        np.asarray(features.candidate_present[:, 2:12]),
        np.ones((2, 10), dtype=np.bool_),
    )
    assert not np.asarray(program.market_priority).any()
    assert not np.asarray(program.support).any()
    assert not np.asarray(program.consensus).any()


def test_v2_listwise_replay_bc_update_is_finite() -> None:
    states = _states(4)
    history = initialize_opponent_history_v2(states)
    features = build_replay_bc_feature_batch_v2(states, history, TABLES)
    positive = jnp.zeros((4, 96), dtype=jnp.bool_).at[:, 6].set(True)
    supervised = features.candidate_eligible
    assert bool(jnp.all(supervised[:, 6]))
    batch = FullBCListwiseBatchV2(
        global_features=features.global_features,
        candidate_features=features.candidate_features,
        candidate_task_type=features.candidate_task_type,
        positive_candidate_mask=positive,
        supervised_candidate_mask=supervised,
        opponent_task_type=jnp.asarray((3, 4, 5, -1), dtype=jnp.int8),
        sample_weight=jnp.ones((4,), dtype=jnp.float32),
    )
    params_v1 = initialize_full_learned_params_v1(jax.random.key(141))
    params_v2 = migrate_full_learned_params_v1_to_v2(params_v1, jax.random.key(142))
    config = FullBCConfigV2(learning_rate=1e-3, opponent_aux_weight=0.1)
    loss, metrics = full_bc_listwise_loss_v2(params_v2, batch, config)
    assert bool(jnp.isfinite(loss))
    assert float(metrics.empty_positive_count) == 0.0
    state = initialize_full_bc_state_v2(params_v2, config)
    state, metrics = jax.jit(make_full_bc_listwise_update_v2(config))(state, batch)
    assert int(state.step) == 1
    assert float(metrics.grad_norm) > 0.0
    assert all(bool(jnp.isfinite(value)) for value in jax.tree.leaves(metrics))


def test_v2_two_checkpoint_arena_smoke() -> None:
    event_ids, bank = load_event_bank()
    events = jax.tree.map(lambda value: value[:2], bank)
    params_v1 = initialize_full_learned_params_v1(jax.random.key(151))
    baseline = migrate_full_learned_params_v1_to_v2(params_v1, jax.random.key(152))
    candidate = migrate_full_learned_params_v1_to_v2(params_v1, jax.random.key(153))
    carry = initialize_full_learned_carry_v2(
        jnp.asarray(event_ids[:2]), events, jax.random.key(154)
    )
    arena = jax.jit(
        make_full_paired_arena_rollout_v2(
            rollout_steps=3,
            deterministic=True,
            decision_interval=1,
            include_opponent=True,
            include_history=True,
            task_card_program=broad_replay_bc_candidate_program_v2(),
            allow_nonpositive_econ=True,
        )
    )
    result = arena(carry, TABLES, candidate, baseline)
    assert result.final_carry.environment_state.step.shape == (2,)
    np.testing.assert_array_equal(
        np.asarray(result.final_carry.environment_state.step),
        np.full((2,), 3, dtype=np.int16),
    )
    assert result.player0_outcome.shape == (2,)
