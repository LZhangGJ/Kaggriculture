from __future__ import annotations

import jax
import jax.numpy as jnp

from strategic_v5.learned_v1 import initialize_full_learned_params_v1
from strategic_v5.learned_v2 import (
    apply_full_learned_model_v2,
    full_parameter_count_v2,
    migrate_full_learned_params_v1_to_v2,
    migrate_full_learned_params_v2_to_50k_v2,
)
from strategic_v5.opponent_v2 import CANDIDATE_FEATURE_DIM_V2, GLOBAL_FEATURE_DIM_V2
from strategic_v5.constants import MAX_CANDIDATES_V1


def test_50k_migration_preserves_policy_and_adds_live_capacity():
    params_8k = migrate_full_learned_params_v1_to_v2(
        initialize_full_learned_params_v1(jax.random.key(9101)),
        jax.random.key(9102),
    )
    params_50k = migrate_full_learned_params_v2_to_50k_v2(
        params_8k, jax.random.key(9103)
    )
    assert full_parameter_count_v2(params_8k) == 8900
    assert full_parameter_count_v2(params_50k) == 50180

    global_features = jax.random.normal(
        jax.random.key(9104), (8, GLOBAL_FEATURE_DIM_V2)
    )
    candidate_features = jax.random.normal(
        jax.random.key(9105),
        (8, MAX_CANDIDATES_V1, CANDIDATE_FEATURE_DIM_V2),
        dtype=jnp.float32,
    ).astype(jnp.float16)
    task_type = jax.random.randint(
        jax.random.key(9106), (8, MAX_CANDIDATES_V1), 0, 21, dtype=jnp.int8
    )
    old = apply_full_learned_model_v2(
        params_8k, global_features, candidate_features, task_type
    )
    new = apply_full_learned_model_v2(
        params_50k, global_features, candidate_features, task_type
    )
    assert bool(jnp.allclose(old.candidate_logits, new.candidate_logits, atol=1e-6))
    assert bool(jnp.allclose(old.stop_logit, new.stop_logit, atol=1e-6))
    assert bool(jnp.allclose(old.value, new.value, atol=1e-6))
    assert bool(jnp.allclose(old.opponent_task_logits, new.opponent_task_logits, atol=1e-6))

    def loss(params):
        output = apply_full_learned_model_v2(
            params, global_features, candidate_features, task_type
        )
        return jnp.mean(jnp.square(output.candidate_logits - 0.5)) + jnp.mean(
            jnp.square(output.value - 0.25)
        )

    gradient = jax.grad(loss)(params_50k)
    # New outbound rows must receive a gradient immediately; otherwise widening
    # would add dead parameters and would not be a meaningful capacity test.
    assert float(jnp.linalg.norm(gradient["value"]["kernel"][32:])) > 0.0
    assert float(jnp.linalg.norm(gradient["task_embedding"][:, 8:])) > 0.0
