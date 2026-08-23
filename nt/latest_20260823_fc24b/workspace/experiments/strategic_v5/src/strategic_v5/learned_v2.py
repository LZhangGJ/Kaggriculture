"""Opponent-aware V2 candidate scorer with lossless V1 parameter migration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

from flax import linen as nn
from flax.core import FrozenDict, freeze, unfreeze
import jax
import jax.numpy as jnp

from .constants import MAX_CANDIDATES_V1
from .learned_v1 import PRIOR_FEATURE_INDEX_V1, PRIOR_LOGIT_SCALE_V1, TASK_TYPE_COUNT_V1
from .opponent_v2 import CANDIDATE_FEATURE_DIM_V2, GLOBAL_FEATURE_DIM_V2


class FullPolicyOutputV2(NamedTuple):
    candidate_logits: jax.Array
    stop_logit: jax.Array
    value: jax.Array
    opponent_task_logits: jax.Array


class FullPolicyContextV2(NamedTuple):
    """Candidate-independent network state reused by all market ordinals."""

    context: jax.Array
    gate: jax.Array
    task_context: jax.Array
    stop_logit: jax.Array
    value: jax.Array
    opponent_task_logits: jax.Array


class CompactFullCandidateScorerV2(nn.Module):
    """V1-compatible scorer plus an optional opponent-task auxiliary head."""

    context_dim: int = 32
    basis_dim: int = 4
    task_dim: int = 8

    @nn.compact
    def __call__(
        self,
        global_features: jax.Array,
        candidate_features: jax.Array,
        candidate_task_type: jax.Array,
    ) -> FullPolicyOutputV2:
        global_features = global_features.astype(jnp.float32)
        candidate_features = candidate_features.astype(jnp.float32)
        context = nn.silu(
            nn.Dense(self.context_dim, name="global_context", precision=jax.lax.Precision.HIGHEST)(
                global_features
            )
        )
        gate = jax.nn.softmax(
            nn.Dense(self.basis_dim, name="candidate_gate", precision=jax.lax.Precision.HIGHEST)(
                context
            ),
            axis=-1,
        )
        basis = nn.Dense(
            self.basis_dim,
            use_bias=False,
            kernel_init=nn.initializers.zeros_init(),
            name="candidate_basis",
            precision=jax.lax.Precision.HIGHEST,
        )(candidate_features)
        residual = jnp.sum(basis * gate[..., None, :], axis=-1)

        task_embedding = self.param(
            "task_embedding",
            nn.initializers.zeros_init(),
            (TASK_TYPE_COUNT_V1, self.task_dim),
        )
        task_context = nn.Dense(
            self.task_dim, name="task_context", precision=jax.lax.Precision.HIGHEST
        )(context)
        safe_task = jnp.clip(
            candidate_task_type.astype(jnp.int32), 0, TASK_TYPE_COUNT_V1 - 1
        )
        task_residual = jnp.sum(
            task_embedding[safe_task] * task_context[..., None, :], axis=-1
        )
        direct = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="candidate_direct",
            precision=jax.lax.Precision.HIGHEST,
        )(candidate_features)[..., 0]
        prior = candidate_features[..., PRIOR_FEATURE_INDEX_V1] * PRIOR_LOGIT_SCALE_V1
        stop = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="stop",
            precision=jax.lax.Precision.HIGHEST,
        )(context)[..., 0]
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="value",
            precision=jax.lax.Precision.HIGHEST,
        )(context)[..., 0]
        opponent_task = nn.Dense(
            TASK_TYPE_COUNT_V1,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="opponent_task",
            precision=jax.lax.Precision.HIGHEST,
        )(context)
        return FullPolicyOutputV2(
            candidate_logits=(prior + residual + task_residual + direct).astype(
                jnp.float32
            ),
            stop_logit=stop.astype(jnp.float32),
            value=jnp.tanh(value).astype(jnp.float32),
            opponent_task_logits=opponent_task.astype(jnp.float32),
        )


MODEL_FULL_V2 = CompactFullCandidateScorerV2()
MODEL_FULL_50K_V2 = CompactFullCandidateScorerV2(
    context_dim=180,
    basis_dim=4,
    task_dim=16,
)


def initialize_full_learned_params_v2(key: jax.Array) -> object:
    global_features = jnp.zeros((1, GLOBAL_FEATURE_DIM_V2), dtype=jnp.float32)
    candidate_features = jnp.zeros(
        (1, MAX_CANDIDATES_V1, CANDIDATE_FEATURE_DIM_V2), dtype=jnp.float16
    )
    task_type = jnp.zeros((1, MAX_CANDIDATES_V1), dtype=jnp.int8)
    return MODEL_FULL_V2.init(
        key, global_features, candidate_features, task_type
    )["params"]


def initialize_full_learned_params_50k_v2(key: jax.Array) -> object:
    """Initialize the 50,180-parameter capacity-control policy."""

    global_features = jnp.zeros((1, GLOBAL_FEATURE_DIM_V2), dtype=jnp.float32)
    candidate_features = jnp.zeros(
        (1, MAX_CANDIDATES_V1, CANDIDATE_FEATURE_DIM_V2), dtype=jnp.float16
    )
    task_type = jnp.zeros((1, MAX_CANDIDATES_V1), dtype=jnp.int8)
    return MODEL_FULL_50K_V2.init(
        key, global_features, candidate_features, task_type
    )["params"]


def apply_full_learned_model_v2(
    params: object,
    global_features: jax.Array,
    candidate_features: jax.Array,
    candidate_task_type: jax.Array,
) -> FullPolicyOutputV2:
    context = encode_full_learned_context_v2(params, global_features)
    return apply_full_learned_candidates_from_context_v2(
        params, context, candidate_features, candidate_task_type
    )


def _dense_v2(
    inputs: jax.Array, layer: Mapping[str, jax.Array]
) -> jax.Array:
    value = jnp.matmul(
        inputs.astype(jnp.float32),
        layer["kernel"],
        precision=jax.lax.Precision.HIGHEST,
    )
    return value + layer.get("bias", 0)


def encode_full_learned_context_v2(
    params: object, global_features: jax.Array
) -> FullPolicyContextV2:
    """Evaluate the immutable global branch exactly once per environment step."""

    context = jax.nn.silu(_dense_v2(global_features, params["global_context"]))
    gate = jax.nn.softmax(_dense_v2(context, params["candidate_gate"]), axis=-1)
    task_context = _dense_v2(context, params["task_context"])
    return FullPolicyContextV2(
        context=context,
        gate=gate,
        task_context=task_context,
        stop_logit=_dense_v2(context, params["stop"])[..., 0].astype(jnp.float32),
        value=jnp.tanh(_dense_v2(context, params["value"])[..., 0]).astype(
            jnp.float32
        ),
        opponent_task_logits=_dense_v2(context, params["opponent_task"]).astype(
            jnp.float32
        ),
    )


def apply_full_learned_candidates_from_context_v2(
    params: object,
    context: FullPolicyContextV2,
    candidate_features: jax.Array,
    candidate_task_type: jax.Array,
) -> FullPolicyOutputV2:
    """Score a refreshed candidate set without recomputing global context."""

    features = candidate_features.astype(jnp.float32)
    basis = _dense_v2(features, params["candidate_basis"])
    residual = jnp.sum(basis * context.gate[..., None, :], axis=-1)
    safe_task = jnp.clip(
        candidate_task_type.astype(jnp.int32), 0, TASK_TYPE_COUNT_V1 - 1
    )
    task_residual = jnp.sum(
        params["task_embedding"][safe_task]
        * context.task_context[..., None, :],
        axis=-1,
    )
    direct = _dense_v2(features, params["candidate_direct"])[..., 0]
    prior = features[..., PRIOR_FEATURE_INDEX_V1] * PRIOR_LOGIT_SCALE_V1
    return FullPolicyOutputV2(
        candidate_logits=(prior + residual + task_residual + direct).astype(
            jnp.float32
        ),
        stop_logit=context.stop_logit,
        value=context.value,
        opponent_task_logits=context.opponent_task_logits,
    )


def full_parameter_count_v2(params: object) -> int:
    return int(sum(value.size for value in jax.tree.leaves(params)))


def _copy_v1_leaf_into_v2(old: Any, new: Any, path: str) -> Any:
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        result = dict(new)
        for key, old_value in old.items():
            if key not in result:
                raise ValueError(f"V1 parameter missing from V2 tree: {path}/{key}")
            result[key] = _copy_v1_leaf_into_v2(
                old_value, result[key], f"{path}/{key}"
            )
        return result
    old_array = jnp.asarray(old)
    new_array = jnp.asarray(new)
    if old_array.ndim != new_array.ndim:
        raise ValueError(
            f"rank mismatch at {path}: {old_array.shape} -> {new_array.shape}"
        )
    if any(left > right for left, right in zip(old_array.shape, new_array.shape)):
        raise ValueError(
            f"V2 parameter is smaller at {path}: {old_array.shape} -> {new_array.shape}"
        )
    if old_array.shape == new_array.shape:
        return old_array.astype(new_array.dtype)
    slices = tuple(slice(0, size) for size in old_array.shape)
    return jnp.zeros_like(new_array).at[slices].set(old_array.astype(new_array.dtype))


def migrate_full_learned_params_v1_to_v2(
    params_v1: object, key: jax.Array
) -> object:
    """Copy V1 exactly and zero every new V2 input row.

    The auxiliary head is zero-initialized.  Consequently V2 on a zero-padded
    V1 observation produces bitwise-equivalent logits/value up to normal float
    evaluation tolerance, while gradients can immediately learn new rows.
    """

    params_v2 = initialize_full_learned_params_v2(key)
    was_frozen = isinstance(params_v2, FrozenDict)
    migrated = _copy_v1_leaf_into_v2(
        unfreeze(params_v1), unfreeze(params_v2), "params"
    )
    return freeze(migrated) if was_frozen else migrated


def migrate_full_learned_params_v2_to_50k_v2(
    params_v2: object, key: jax.Array
) -> object:
    """Expand 8.9K V2 to 50.2K while preserving its initial policy exactly.

    New global-context columns retain random features, while every outbound row
    from those columns starts at zero.  Therefore logits/value are initially
    unchanged, but the added rows receive gradients on the first update.  The
    new task-context columns are random with zero task embeddings, providing a
    second initially silent but trainable expansion path.
    """

    old = unfreeze(params_v2)
    new_params = initialize_full_learned_params_50k_v2(key)
    was_frozen = isinstance(new_params, FrozenDict)
    new = unfreeze(new_params)

    old_context = old["global_context"]["kernel"].shape[1]
    old_task_dim = old["task_embedding"].shape[1]

    # Preserve random new context columns, replacing only the old prefix.
    new["global_context"]["kernel"] = new["global_context"]["kernel"].at[
        :, :old_context
    ].set(old["global_context"]["kernel"])
    new["global_context"]["bias"] = new["global_context"]["bias"].at[
        :old_context
    ].set(old["global_context"]["bias"])

    # Existing candidate basis and direct branch are shape-identical.
    new["candidate_basis"] = old["candidate_basis"]
    new["candidate_direct"] = old["candidate_direct"]

    # Same four-way gate: copy old rows, silence new context rows initially.
    new["candidate_gate"]["kernel"] = jnp.zeros_like(
        new["candidate_gate"]["kernel"]
    ).at[:old_context].set(old["candidate_gate"]["kernel"])
    new["candidate_gate"]["bias"] = old["candidate_gate"]["bias"]

    # Copy old task branch. New task-context columns remain random, while their
    # embeddings start at zero, so they are policy-neutral but learnable.
    new["task_embedding"] = new["task_embedding"].at[
        :, :old_task_dim
    ].set(old["task_embedding"])
    task_kernel = new["task_context"]["kernel"]
    task_kernel = task_kernel.at[:, :old_task_dim].set(0.0)
    task_kernel = task_kernel.at[:old_context, :old_task_dim].set(
        old["task_context"]["kernel"]
    )
    new["task_context"]["kernel"] = task_kernel
    new["task_context"]["bias"] = new["task_context"]["bias"].at[
        :old_task_dim
    ].set(old["task_context"]["bias"])

    # Silence new context rows for every remaining output head.
    for name in ("stop", "value", "opponent_task"):
        new[name]["kernel"] = jnp.zeros_like(new[name]["kernel"]).at[
            :old_context
        ].set(old[name]["kernel"])
        new[name]["bias"] = old[name]["bias"]

    return freeze(new) if was_frozen else new
