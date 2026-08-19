from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax import load_event_bank, load_tables
from strategic_v5.dynamic_ppo_v2 import (
    DynamicPPOSamplesV2,
    dynamic_ppo_loss_v2,
    dynamic_ppo_sample_step_v2,
    dynamic_ppo_snapshot_v2,
    dynamic_sample_steps_v2,
)
from strategic_v5.learned_v1 import initialize_full_learned_params_v1
from strategic_v5.learned_v2 import migrate_full_learned_params_v1_to_v2
from strategic_v5.ppo_v1 import FullPPOConfigV1
from strategic_v5.rollout_v2 import initialize_full_learned_carry_v2


def test_dynamic_joint_logprob_recomputes_exactly():
    seed_bank, event_bank = load_event_bank()
    carry = initialize_full_learned_carry_v2(
        seed_bank[:2],
        jax.tree.map(lambda value: value[:2], event_bank),
        jax.random.key(8101),
    )
    params = migrate_full_learned_params_v1_to_v2(
        initialize_full_learned_params_v1(jax.random.key(8102)),
        jax.random.key(8103),
    )
    tables = load_tables()
    snapshot = dynamic_ppo_snapshot_v2(carry, tables, params)
    _, transition = dynamic_ppo_sample_step_v2(carry, tables, params)
    assert bool(jnp.allclose(snapshot.old_logprob, transition.old_logprob, atol=1e-6))
    assert bool(jnp.allclose(snapshot.old_value, transition.old_value, atol=1e-6))
    count = 4
    samples = DynamicPPOSamplesV2(
        unit_global_features=transition.unit_global_features.reshape((count, -1)),
        unit_candidate_features=transition.unit_candidate_features.reshape(
            (count,) + transition.unit_candidate_features.shape[2:]
        ),
        unit_candidate_task_type=transition.unit_candidate_task_type.reshape(
            (count,) + transition.unit_candidate_task_type.shape[2:]
        ),
        unit_task_masks=transition.unit_task_masks.reshape(
            (count,) + transition.unit_task_masks.shape[2:]
        ),
        unit_selected_indices=transition.unit_selected_indices.reshape(
            (count,) + transition.unit_selected_indices.shape[2:]
        ),
        market_candidate_features=transition.market_candidate_features.reshape(
            (count,) + transition.market_candidate_features.shape[2:]
        ),
        market_candidate_task_type=transition.market_candidate_task_type.reshape(
            (count,) + transition.market_candidate_task_type.shape[2:]
        ),
        market_masks=transition.market_masks.reshape(
            (count,) + transition.market_masks.shape[2:]
        ),
        market_selected_indices=transition.market_selected_indices.reshape(
            (count,) + transition.market_selected_indices.shape[2:]
        ),
        old_logprob=transition.old_logprob.reshape((count,)),
        old_value=transition.old_value.reshape((count,)),
        advantage=jnp.asarray([1.0, -1.0, -1.0, 1.0], dtype=jnp.float32),
        returns=transition.old_value.reshape((count,)),
    )
    _, metrics = dynamic_ppo_loss_v2(params, samples, FullPPOConfigV1())
    assert abs(float(metrics.approx_kl)) < 1e-5
    assert abs(float(metrics.mean_ratio) - 1.0) < 1e-5
    assert float(metrics.clip_fraction) == 0.0


def test_dynamic_sample_steps_cover_terminal():
    steps = dynamic_sample_steps_v2(32)
    assert steps[0] == 0
    assert steps[-1] == 718
    assert len(steps) == 24
