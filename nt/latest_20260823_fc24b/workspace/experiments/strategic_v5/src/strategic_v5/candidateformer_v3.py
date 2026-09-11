"""Small relation-aware Transformer scorer for the accepted Dynamic V2 policy.

The model keeps the frozen 50K scorer as an exact residual baseline.  Its
Transformer branch starts policy-neutral: every output delta head is zero, so
initial logits/value are exactly the old policy while attention is trainable.

There are no candidate-slot embeddings.  Candidate order therefore has no
semantic meaning and permuting cards must permute their logits identically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    BOARD_SIZE,
    EPISODE_STEPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    NUM_TILES,
    SHED_CAPACITY,
)

from .constants import CandidateSourceV1, MAX_CANDIDATES_V1, TaskStatusV1
from .e5_econ import full_econ_score_v1
from .learned_v1 import PRIOR_FEATURE_INDEX_V1, PRIOR_LOGIT_SCALE_V1, TASK_TYPE_COUNT_V1
from .learned_v2 import (
    FullPolicyContextV2,
    FullPolicyOutputV2,
    apply_full_learned_candidates_from_context_v2,
    encode_full_learned_context_v2,
    full_parameter_count_v2,
    initialize_full_learned_params_50k_v2,
)
from .opponent_v2 import GLOBAL_FEATURE_DIM_V2, CANDIDATE_FEATURE_DIM_V2
from .schema import CandidateV1, ControllerStateV1, EconFeaturesV1, FeasibilityV1


CANDIDATE_RELATION_FEATURE_DIM_V3 = 8
CANDIDATE_ECON_DETAIL_DIM_V3 = 8
CANDIDATE_FEATURE_DIM_V3 = (
    CANDIDATE_FEATURE_DIM_V2
    + CANDIDATE_RELATION_FEATURE_DIM_V3
    + CANDIDATE_ECON_DETAIL_DIM_V3
)
PLAN_TOKEN_COUNT_V3 = 4
PLAN_FEATURE_DIM_V3 = 32
GLOBAL_TOKEN_COUNT_V3 = 8
GLOBAL_GROUP_RANGES_V3 = (
    (0, 9),    # time, money, land unlock and labor
    (9, 38),   # own shed, seeds and carried inventory
    (38, 65),  # shared market and town calendar
    (65, 87),  # both visible land kinds and own crops
    (87, 96),  # own animals, workers and storage utilization
    (96, 160), # current public opponent projection
    (160, 196),# most recent public-state delta
    (196, 232),# public-state exponential trend
)


@dataclass(frozen=True)
class CandidateFormerConfigV3:
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    ffn_hidden: int = 256

    def validate(self) -> None:
        if self.d_model <= 0 or self.layers <= 0 or self.heads <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.layers != 2:
            raise ValueError("V3 contract freezes exactly two Transformer blocks")


CANDIDATEFORMER_420K_V3 = CandidateFormerConfigV3()


class CandidateFormerContextV3(NamedTuple):
    baseline: FullPolicyContextV2
    candidate_hidden: jax.Array
    pooled_hidden: jax.Array
    stop_logit: jax.Array
    value: jax.Array
    opponent_task_logits: jax.Array


def config_dict_v3(config: CandidateFormerConfigV3) -> dict[str, int]:
    return {name: int(value) for name, value in asdict(config).items()}


def _dense_init(
    key: jax.Array,
    inputs: int,
    outputs: int,
    *,
    zero: bool = False,
) -> dict[str, jax.Array]:
    if zero:
        kernel = jnp.zeros((inputs, outputs), dtype=jnp.float32)
    else:
        limit = jnp.sqrt(6.0 / float(inputs + outputs))
        kernel = jax.random.uniform(
            key,
            (inputs, outputs),
            minval=-limit,
            maxval=limit,
            dtype=jnp.float32,
        )
    return {
        "kernel": kernel,
        "bias": jnp.zeros((outputs,), dtype=jnp.float32),
    }


def _embedding_init(key: jax.Array, rows: int, width: int) -> jax.Array:
    return 0.02 * jax.random.normal(key, (rows, width), dtype=jnp.float32)


def _norm_init(width: int) -> dict[str, jax.Array]:
    return {
        "scale": jnp.ones((width,), dtype=jnp.float32),
        "bias": jnp.zeros((width,), dtype=jnp.float32),
    }


def initialize_candidateformer_v3(
    key: jax.Array,
    *,
    baseline_params: object | None = None,
    config: CandidateFormerConfigV3 = CANDIDATEFORMER_420K_V3,
) -> dict[str, Any]:
    """Initialize an exact old-policy residual plus the relation Transformer."""

    config.validate()
    keys = iter(jax.random.split(key, 128))
    width = config.d_model
    transformer: dict[str, Any] = {
        "global_in": [
            _dense_init(next(keys), stop - start, width)
            for start, stop in GLOBAL_GROUP_RANGES_V3
        ],
        "global_type": _embedding_init(next(keys), GLOBAL_TOKEN_COUNT_V3, width),
        "plan_in": _dense_init(next(keys), PLAN_FEATURE_DIM_V3, width),
        "plan_type": _embedding_init(next(keys), PLAN_TOKEN_COUNT_V3, width),
        "candidate_in": _dense_init(next(keys), CANDIDATE_FEATURE_DIM_V3, width),
        "token_type": _embedding_init(next(keys), 3, width),
        "task_embedding": _embedding_init(next(keys), TASK_TYPE_COUNT_V1, width),
        "owner_embedding": _embedding_init(next(keys), MAX_UNITS + 1, width),
        "item_embedding": _embedding_init(next(keys), NUM_SHED_ITEMS + 1, width),
        "source_embedding": _embedding_init(
            next(keys), len(CandidateSourceV1) + 1, width
        ),
        "blocks": [],
        "final_norm": _norm_init(width),
        "head_hidden": _dense_init(next(keys), width, width),
        "candidate_delta": _dense_init(next(keys), width, 1, zero=True),
        "stop_delta": _dense_init(next(keys), width, 1, zero=True),
        "value_delta": _dense_init(next(keys), width, 1, zero=True),
        "opponent_delta": _dense_init(
            next(keys), width, TASK_TYPE_COUNT_V1, zero=True
        ),
        "market_dynamic": _dense_init(
            next(keys), CANDIDATE_FEATURE_DIM_V3, width
        ),
        "market_fuse": _dense_init(next(keys), width, width),
        "market_delta": _dense_init(next(keys), width, 1, zero=True),
    }
    for _ in range(config.layers):
        transformer["blocks"].append(
            {
                "norm_attention": _norm_init(width),
                "qkv": _dense_init(next(keys), width, 3 * width),
                "attention_out": _dense_init(next(keys), width, width),
                "norm_ffn": _norm_init(width),
                "ffn_in": _dense_init(next(keys), width, config.ffn_hidden),
                "ffn_out": _dense_init(next(keys), config.ffn_hidden, width),
            }
        )
    baseline = (
        initialize_full_learned_params_50k_v2(next(keys))
        if baseline_params is None
        else baseline_params
    )
    return {"baseline": baseline, "transformer": transformer}


def candidateformer_parameter_count_v3(params: object) -> int:
    return int(sum(value.size for value in jax.tree.leaves(params)))


def is_candidateformer_params_v3(params: object) -> bool:
    """Structure-only dispatch that remains static while JAX traces a policy."""

    return (
        isinstance(params, Mapping)
        and "baseline" in params
        and "transformer" in params
    )


def candidateformer_trainable_parameter_count_v3(params: object) -> int:
    return candidateformer_parameter_count_v3(params) - full_parameter_count_v2(
        params["baseline"]
    )


def _dense(value: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    return value.astype(jnp.float32) @ params["kernel"] + params["bias"]


def _layer_norm(value: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    value = value.astype(jnp.float32)
    mean = jnp.mean(value, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(value - mean), axis=-1, keepdims=True)
    return (value - mean) * jax.lax.rsqrt(variance + 1.0e-5) * params[
        "scale"
    ] + params["bias"]


def _attention(
    value: jax.Array,
    params: dict[str, Any],
    token_mask: jax.Array,
    heads: int,
) -> jax.Array:
    batch_size, token_count, width = value.shape
    head_width = width // heads
    qkv = _dense(value, params["qkv"]).reshape(
        (batch_size, token_count, 3, heads, head_width)
    )
    query, key, payload = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    logits = jnp.einsum("bthd,bshd->bhts", query, key)
    logits = logits * jnp.asarray(head_width**-0.5, dtype=jnp.float32)
    logits = jnp.where(token_mask[:, None, None, :], logits, -1.0e30)
    weights = jax.nn.softmax(logits, axis=-1)
    attended = jnp.einsum("bhts,bshd->bthd", weights, payload).reshape(
        (batch_size, token_count, width)
    )
    return _dense(attended, params["attention_out"])


def _transformer(
    tokens: jax.Array,
    token_mask: jax.Array,
    params: dict[str, Any],
    config: CandidateFormerConfigV3,
) -> jax.Array:
    value = tokens
    for block in params["blocks"]:
        normalized = _layer_norm(value, block["norm_attention"])
        value = value + _attention(normalized, block, token_mask, config.heads)
        normalized = _layer_norm(value, block["norm_ffn"])
        value = value + _dense(
            jax.nn.gelu(_dense(normalized, block["ffn_in"])),
            block["ffn_out"],
        )
    return _layer_norm(value, params["final_norm"])


def build_candidate_features_v3(
    states,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    econ: EconFeaturesV1,
    base_v2_features: jax.Array,
) -> jax.Array:
    """Append identity/relationship and detailed economic fields to V2."""

    relation = jnp.stack(
        (
            candidates.present.astype(jnp.float32),
            (candidates.target_id.astype(jnp.float32) + 1.0) / (NUM_TILES + 1),
            (candidates.target_x.astype(jnp.float32) + 1.0) / BOARD_SIZE,
            (candidates.target_y.astype(jnp.float32) + 1.0) / BOARD_SIZE,
            (candidates.source.astype(jnp.float32) + 1.0)
            / (len(CandidateSourceV1) + 1),
            (candidates.source_slot.astype(jnp.float32) + 1.0)
            / (MAX_CANDIDATES_V1 + 1),
            (feasibility.unit_required.astype(jnp.float32) + 1.0)
            / (MAX_UNITS + 1),
            (feasibility.plot_required.astype(jnp.float32) + 1.0)
            / (NUM_TILES + 1),
        ),
        axis=-1,
    )
    detail = jnp.stack(
        (
            econ.cash_flow_before_revenue.astype(jnp.float32) / 10_000.0,
            econ.steps_to_first_revenue.astype(jnp.float32) / EPISODE_STEPS,
            econ.maintenance_actions.astype(jnp.float32) / EPISODE_STEPS,
            econ.plot_occupancy_duration.astype(jnp.float32) / EPISODE_STEPS,
            econ.storage_required.astype(jnp.float32) / SHED_CAPACITY,
            econ.cycles_before_terminal.astype(jnp.float32) / 30.0,
            econ.known_market_impact.astype(jnp.float32) / 10_000.0,
            econ.terminal_salvage_value.astype(jnp.float32) / 10_000.0,
        ),
        axis=-1,
    )
    result = jnp.concatenate(
        (base_v2_features.astype(jnp.float32), relation, detail), axis=-1
    )
    if result.shape[-1] != CANDIDATE_FEATURE_DIM_V3:
        raise ValueError("candidate V3 feature width drifted")
    return result.astype(jnp.float16)


def _normalized_histogram(ids, mask, classes: int, denominator: float) -> jax.Array:
    safe = jnp.clip(ids.astype(jnp.int32), 0, classes - 1)
    return jnp.sum(
        jax.nn.one_hot(safe, classes, dtype=jnp.float32)
        * mask.astype(jnp.float32)[..., None],
        axis=1,
    ) / denominator


def build_plan_ledger_features_v3(
    states,
    controller: ControllerStateV1,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    econ: EconFeaturesV1,
    player: int,
) -> jax.Array:
    """Four fixed summaries of active plans, commitments and current options."""

    unit = controller.unit_tasks
    unit_active = unit.status == TaskStatusV1.ACTIVE
    unit_task_hist = _normalized_histogram(
        unit.task_type, unit_active, TASK_TYPE_COUNT_V1, float(MAX_UNITS)
    )
    phase_hist = _normalized_histogram(unit.phase, unit_active, 8, float(MAX_UNITS))
    unit_denominator = jnp.maximum(jnp.sum(unit_active, axis=1), 1).astype(jnp.float32)
    remaining_deadline = jnp.where(
        unit_active,
        jnp.maximum(unit.deadline_step - states.step[:, None], 0),
        0,
    )
    remaining_finish = jnp.where(
        unit_active,
        jnp.maximum(unit.expected_finish_step - states.step[:, None], 0),
        0,
    )
    token0 = jnp.concatenate(
        (
            unit_task_hist,
            (jnp.sum(unit_active, axis=1) / float(MAX_UNITS))[:, None],
            phase_hist,
            (jnp.sum(remaining_deadline, axis=1) / unit_denominator / EPISODE_STEPS)[:, None],
            (jnp.sum(remaining_finish, axis=1) / unit_denominator / EPISODE_STEPS)[:, None],
        ),
        axis=-1,
    )

    safe_item = jnp.clip(unit.item_id.astype(jnp.int32), 0, NUM_SHED_ITEMS - 1)
    item_one_hot = jax.nn.one_hot(safe_item, NUM_SHED_ITEMS, dtype=jnp.float32)
    item_count = jnp.sum(item_one_hot * unit_active[..., None], axis=1) / float(MAX_UNITS)
    item_quantity = jnp.sum(
        item_one_hot
        * unit_active[..., None]
        * unit.quantity.astype(jnp.float32)[..., None],
        axis=1,
    ) / SHED_CAPACITY
    source_hist = _normalized_histogram(
        jnp.clip(unit.task_type, 0, 7), unit_active, 8, float(MAX_UNITS)
    )
    token1 = jnp.concatenate((item_count, item_quantity, source_hist), axis=-1)

    market = controller.market_tasks
    market_active = market.status == TaskStatusV1.ACTIVE
    market_task_hist = _normalized_histogram(
        market.task_type, market_active, TASK_TYPE_COUNT_V1, float(MAX_MARKET_ORDERS)
    )
    market_product = jnp.clip(market.item_id.astype(jnp.int32), 0, NUM_PRODUCTS - 1)
    market_item_hist = _normalized_histogram(
        market_product, market_active, NUM_PRODUCTS, float(MAX_MARKET_ORDERS)
    )
    token2 = jnp.concatenate(
        (
            market_task_hist,
            market_item_hist,
            (jnp.sum(market_active, axis=1) / float(MAX_MARKET_ORDERS))[:, None],
            (jnp.sum(
                jnp.where(market_active, market.quantity, 0), axis=1
            ).astype(jnp.float32) / SHED_CAPACITY)[:, None],
        ),
        axis=-1,
    )

    present = candidates.present
    present_count = jnp.maximum(jnp.sum(present, axis=1), 1).astype(jnp.float32)
    task_hist = _normalized_histogram(
        candidates.task_type, present, TASK_TYPE_COUNT_V1, float(MAX_CANDIDATES_V1)
    )
    score = full_econ_score_v1(candidates, feasibility, econ)
    scalar = jnp.stack(
        (
            jnp.sum(present, axis=1) / float(MAX_CANDIDATES_V1),
            jnp.sum(present & feasibility.legal_now, axis=1) / float(MAX_CANDIDATES_V1),
            jnp.sum(present & candidates.mandatory, axis=1) / float(MAX_CANDIDATES_V1),
            jnp.sum(present & econ.bankable_before_terminal, axis=1) / float(MAX_CANDIDATES_V1),
            jnp.sum(present & (score > 0), axis=1) / float(MAX_CANDIDATES_V1),
            jnp.sum(jnp.where(present, feasibility.path_steps, 0), axis=1)
            / present_count / 20.0,
            jnp.sum(jnp.where(present, econ.maintenance_actions, 0), axis=1)
            / present_count / EPISODE_STEPS,
            jnp.sum(jnp.where(present, econ.storage_required, 0), axis=1)
            / present_count / SHED_CAPACITY,
            jnp.min(
                jnp.where(
                    present,
                    jnp.maximum(feasibility.deadline_step - states.step[:, None], 0),
                    EPISODE_STEPS,
                ),
                axis=1,
            ) / EPISODE_STEPS,
            states.step.astype(jnp.float32) / (EPISODE_STEPS - 1),
            states.money[:, player].astype(jnp.float32) / 100_000.0,
        ),
        axis=-1,
    )
    token3 = jnp.concatenate((task_hist, scalar), axis=-1)
    result = jnp.stack((token0, token1, token2, token3), axis=1)
    if result.shape[-2:] != (PLAN_TOKEN_COUNT_V3, PLAN_FEATURE_DIM_V3):
        raise ValueError("plan/ledger feature shape drifted")
    return result.astype(jnp.float32)


def _recover_semantic_ids(candidate_features: jax.Array):
    features = candidate_features.astype(jnp.float32)
    owner = jnp.rint(features[..., 2] * MAX_UNITS - 1.0).astype(jnp.int32)
    item = jnp.rint(features[..., 3] * (NUM_SHED_ITEMS + 1) - 1.0).astype(jnp.int32)
    source_index = CANDIDATE_FEATURE_DIM_V2 + 4
    source = jnp.rint(
        features[..., source_index] * (len(CandidateSourceV1) + 1) - 1.0
    ).astype(jnp.int32)
    return (
        jnp.clip(owner + 1, 0, MAX_UNITS),
        jnp.clip(item + 1, 0, NUM_SHED_ITEMS),
        jnp.clip(source + 1, 0, len(CandidateSourceV1)),
    )


def encode_candidateformer_v3(
    params: object,
    global_features: jax.Array,
    plan_features: jax.Array,
    candidate_features: jax.Array,
    candidate_task_type: jax.Array,
    config: CandidateFormerConfigV3 = CANDIDATEFORMER_420K_V3,
) -> tuple[CandidateFormerContextV3, FullPolicyOutputV2]:
    """Run full relation attention once for one official environment step."""

    transformer = params["transformer"]
    global_tokens = []
    for index, ((start, stop), projection) in enumerate(
        zip(GLOBAL_GROUP_RANGES_V3, transformer["global_in"], strict=True)
    ):
        global_tokens.append(
            _dense(global_features[:, start:stop], projection)
            + transformer["global_type"][index]
            + transformer["token_type"][0]
        )
    global_tokens = jnp.stack(global_tokens, axis=1)
    plan_tokens = (
        _dense(plan_features, transformer["plan_in"])
        + transformer["plan_type"][None, :, :]
        + transformer["token_type"][1]
    )
    owner, item, source = _recover_semantic_ids(candidate_features)
    safe_task = jnp.clip(
        candidate_task_type.astype(jnp.int32), 0, TASK_TYPE_COUNT_V1 - 1
    )
    candidate_tokens = (
        _dense(candidate_features, transformer["candidate_in"])
        + transformer["task_embedding"][safe_task]
        + transformer["owner_embedding"][owner]
        + transformer["item_embedding"][item]
        + transformer["source_embedding"][source]
        + transformer["token_type"][2]
    )
    tokens = jnp.concatenate((global_tokens, plan_tokens, candidate_tokens), axis=1)
    candidate_present = candidate_features[
        ..., CANDIDATE_FEATURE_DIM_V2
    ].astype(jnp.float32) > 0.5
    prefix_mask = jnp.ones(
        (tokens.shape[0], GLOBAL_TOKEN_COUNT_V3 + PLAN_TOKEN_COUNT_V3),
        dtype=jnp.bool_,
    )
    token_mask = jnp.concatenate((prefix_mask, candidate_present), axis=1)
    encoded = _transformer(tokens, token_mask, transformer, config)
    candidate_start = GLOBAL_TOKEN_COUNT_V3 + PLAN_TOKEN_COUNT_V3
    candidate_hidden = encoded[:, candidate_start:]
    pooled = jnp.mean(encoded[:, :GLOBAL_TOKEN_COUNT_V3], axis=1)
    head_hidden = jax.nn.gelu(_dense(candidate_hidden, transformer["head_hidden"]))
    delta = _dense(head_hidden, transformer["candidate_delta"])[..., 0]

    baseline_context = encode_full_learned_context_v2(
        params["baseline"], global_features
    )
    baseline = apply_full_learned_candidates_from_context_v2(
        params["baseline"],
        baseline_context,
        candidate_features[..., :CANDIDATE_FEATURE_DIM_V2],
        candidate_task_type,
    )
    stop = baseline.stop_logit + _dense(pooled, transformer["stop_delta"])[..., 0]
    value = jnp.clip(
        baseline.value + _dense(pooled, transformer["value_delta"])[..., 0],
        -1.0,
        1.0,
    )
    opponent = baseline.opponent_task_logits + _dense(
        pooled, transformer["opponent_delta"]
    )
    output = FullPolicyOutputV2(
        candidate_logits=(baseline.candidate_logits + delta).astype(jnp.float32),
        stop_logit=stop.astype(jnp.float32),
        value=value.astype(jnp.float32),
        opponent_task_logits=opponent.astype(jnp.float32),
    )
    return (
        CandidateFormerContextV3(
            baseline=baseline_context,
            candidate_hidden=candidate_hidden,
            pooled_hidden=pooled,
            stop_logit=output.stop_logit,
            value=output.value,
            opponent_task_logits=output.opponent_task_logits,
        ),
        output,
    )


def apply_candidateformer_market_v3(
    params: object,
    context: CandidateFormerContextV3,
    market_features: jax.Array,
    market_task_type: jax.Array,
) -> FullPolicyOutputV2:
    """Refresh 21 market scores without rerunning 108-token attention."""

    transformer = params["transformer"]
    baseline = apply_full_learned_candidates_from_context_v2(
        params["baseline"],
        context.baseline,
        market_features[..., :CANDIDATE_FEATURE_DIM_V2],
        market_task_type,
    )
    cached = context.candidate_hidden[:, : market_features.shape[1]]
    dynamic = _dense(market_features, transformer["market_dynamic"])
    fused = jax.nn.gelu(_dense(jnp.tanh(cached + dynamic), transformer["market_fuse"]))
    delta = _dense(fused, transformer["market_delta"])[..., 0]
    return FullPolicyOutputV2(
        candidate_logits=(baseline.candidate_logits + delta).astype(jnp.float32),
        stop_logit=context.stop_logit,
        value=context.value,
        opponent_task_logits=context.opponent_task_logits,
    )


def old_prior_logits_v3(candidate_features: jax.Array) -> jax.Array:
    """Small test helper documenting the inherited score-prior contract."""

    return (
        candidate_features[..., PRIOR_FEATURE_INDEX_V1].astype(jnp.float32)
        * PRIOR_LOGIT_SCALE_V1
    )
