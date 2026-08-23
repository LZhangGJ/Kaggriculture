"""Compact Full-core learned policy and GPU-native rollout for V5 F/G.

The policy scores the same 96 candidates used by the accepted Full RULE_ONLY
controller.  Exact core/econ features remain outside the network and act as a
zero-initialized prior; the small network learns residual ranking changes.
"""

from __future__ import annotations

from typing import NamedTuple

from flax import linen as nn
import jax
import jax.numpy as jnp

from kaggriculture_jax import reset
from kaggriculture_jax.constants import (
    BOARD_SIZE,
    EPISODE_STEPS,
    FLAG_CARED,
    FLAG_FED,
    FLAG_FERTILIZER_AVAILABLE,
    FLAG_WATERED,
    MAX_SHOPS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    NUM_TILES,
    SHED_CAPACITY,
    TURNS_PER_DAY,
    TileKind,
)
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.types import Events, State, StaticTables

from .constants import MAX_CANDIDATES_V1, MAX_SELECTIONS_V1, TaskStatusV1
from .e2_core import (
    attach_e2_candidate_batched_v1,
    candidate_mask_from_e2_ledger_v1,
    initialize_e2_ledger_v1,
    reserve_e2_candidate_batched_v1,
)
from .e4_core import build_full_core_candidates_v1, evaluate_full_core_feasibility_v1
from .e4_executor import (
    compile_full_core_action_bundle_v1,
    update_full_core_controller_from_effects_v1,
)
from .e5_econ import (
    SIMPLE_CASH_BUFFER,
    build_full_econ_features_v1,
    full_econ_score_v1,
)
from .e5_rollout import cleanup_full_controller_day_end_v1
from .lifecycle import (
    clear_invalidated_full_core_tasks_v1,
    reset_controller_state_v1,
)
from .schema import CandidateV1, ControllerStateV1, EconFeaturesV1, FeasibilityV1


GLOBAL_FEATURE_DIM_V1 = 96
CANDIDATE_FEATURE_DIM_V1 = 16
TASK_TYPE_COUNT_V1 = 21
PRIOR_FEATURE_INDEX_V1 = 0
PRIOR_LOGIT_SCALE_V1 = 20.0


class FullPolicyOutputV1(NamedTuple):
    candidate_logits: jax.Array
    stop_logit: jax.Array
    value: jax.Array


class FullLearnedSelectionV1(NamedTuple):
    controller: ControllerStateV1
    selected_candidate_indices: jax.Array
    masks: jax.Array
    joint_logprob: jax.Array
    internal_resource_conflict: jax.Array


class FullLearnedCarryV1(NamedTuple):
    environment_state: State
    player0_controller: ControllerStateV1
    player1_controller: ControllerStateV1
    current_events: Events
    rng: jax.Array


class FullLearnedTransitionV1(NamedTuple):
    global_features: jax.Array
    candidate_features: jax.Array
    candidate_task_type: jax.Array
    task_masks: jax.Array
    task_selected_indices: jax.Array
    old_logprob: jax.Array
    old_value: jax.Array
    reward: jax.Array
    done: jax.Array


class FullLearnedRolloutV1(NamedTuple):
    final_carry: FullLearnedCarryV1
    transitions: FullLearnedTransitionV1
    bootstrap_value: jax.Array


class FullTimelineV1(NamedTuple):
    old_value: jax.Array
    reward: jax.Array
    done: jax.Array


class FullStridedRolloutV1(NamedTuple):
    final_carry: FullLearnedCarryV1
    transitions: FullLearnedTransitionV1
    timeline: FullTimelineV1
    bootstrap_value: jax.Array


class CompactFullCandidateScorerV1(nn.Module):
    """A deployable mixture of four linear candidate scorers."""

    @nn.compact
    def __call__(
        self,
        global_features: jax.Array,
        candidate_features: jax.Array,
        candidate_task_type: jax.Array,
    ) -> FullPolicyOutputV1:
        global_features = global_features.astype(jnp.float32)
        candidate_features = candidate_features.astype(jnp.float32)
        context = nn.silu(
            nn.Dense(32, name="global_context", precision=jax.lax.Precision.HIGHEST)(
                global_features
            )
        )
        gate = jax.nn.softmax(
            nn.Dense(4, name="candidate_gate", precision=jax.lax.Precision.HIGHEST)(
                context
            ),
            axis=-1,
        )
        basis = nn.Dense(
            4,
            use_bias=False,
            kernel_init=nn.initializers.zeros_init(),
            name="candidate_basis",
            precision=jax.lax.Precision.HIGHEST,
        )(candidate_features)
        residual = jnp.sum(basis * gate[..., None, :], axis=-1)

        task_embedding = self.param(
            "task_embedding",
            nn.initializers.zeros_init(),
            (TASK_TYPE_COUNT_V1, 8),
        )
        task_context = nn.Dense(
            8, name="task_context", precision=jax.lax.Precision.HIGHEST
        )(context)
        safe_task = jnp.clip(candidate_task_type.astype(jnp.int32), 0, TASK_TYPE_COUNT_V1 - 1)
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
        prior = (
            candidate_features[..., PRIOR_FEATURE_INDEX_V1]
            * PRIOR_LOGIT_SCALE_V1
        )
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
        return FullPolicyOutputV1(
            candidate_logits=(prior + residual + task_residual + direct).astype(jnp.float32),
            stop_logit=stop.astype(jnp.float32),
            value=jnp.tanh(value).astype(jnp.float32),
        )


MODEL_FULL_V1 = CompactFullCandidateScorerV1()


def initialize_full_learned_params_v1(key: jax.Array) -> object:
    global_features = jnp.zeros((1, GLOBAL_FEATURE_DIM_V1), dtype=jnp.float32)
    candidate_features = jnp.zeros(
        (1, MAX_CANDIDATES_V1, CANDIDATE_FEATURE_DIM_V1), dtype=jnp.float16
    )
    task_type = jnp.zeros((1, MAX_CANDIDATES_V1), dtype=jnp.int8)
    return MODEL_FULL_V1.init(
        key, global_features, candidate_features, task_type
    )["params"]


def apply_full_learned_model_v1(
    params: object,
    global_features: jax.Array,
    candidate_features: jax.Array,
    candidate_task_type: jax.Array,
) -> FullPolicyOutputV1:
    return MODEL_FULL_V1.apply(
        {"params": params}, global_features, candidate_features, candidate_task_type
    )


def full_parameter_count_v1(params: object) -> int:
    return int(sum(value.size for value in jax.tree.leaves(params)))


def build_global_features_v1(states: State, player: int) -> jax.Array:
    """Compact actor-visible summary; private opponent inventories are excluded."""

    opponent = 1 - player
    batch_size = states.step.shape[0]
    day = states.step.astype(jnp.float32) / float(EPISODE_STEPS - 1)
    turn = (states.step % TURNS_PER_DAY).astype(jnp.float32) / float(TURNS_PER_DAY - 1)
    kind = jax.nn.one_hot(
        jnp.clip(states.tile_kind.astype(jnp.int32), 0, len(TileKind) - 1),
        len(TileKind),
        dtype=jnp.float32,
    ).sum(axis=(2, 3)) / float(NUM_TILES)
    own_crop = states.tile_crop[:, player]
    own_animal = states.tile_animal[:, player]
    crop_valid = own_crop >= 0
    animal_valid = own_animal >= 0
    crop_one_hot = jax.nn.one_hot(
        jnp.clip(own_crop.astype(jnp.int32), 0, NUM_CROPS - 1), NUM_CROPS
    ) * crop_valid[..., None]
    animal_one_hot = jax.nn.one_hot(
        jnp.clip(own_animal.astype(jnp.int32), 0, NUM_ANIMALS - 1), NUM_ANIMALS
    ) * animal_valid[..., None]
    crop_counts = crop_one_hot.sum(axis=(1, 2)) / float(NUM_TILES)
    crop_yield = (
        crop_one_hot * states.tile_yield[:, player, :, :, None].astype(jnp.float32)
    ).sum(axis=(1, 2)) / 100.0
    animal_counts = animal_one_hot.sum(axis=(1, 2)) / float(NUM_TILES)
    animal_yield = (
        animal_one_hot * states.tile_yield[:, player, :, :, None].astype(jnp.float32)
    ).sum(axis=(1, 2)) / 100.0
    town = jax.nn.one_hot(
        jnp.clip(states.town_shops.astype(jnp.int32), 0, MAX_SHOPS - 1), MAX_SHOPS
    )
    town = (
        town
        * (jnp.arange(MAX_SHOPS)[None, :] < states.town_count[:, None])[..., None]
    ).sum(axis=1) / float(MAX_SHOPS)
    own_inventory = jnp.sum(
        states.unit_inventory[:, player].astype(jnp.float32), axis=1
    ) / float(SHED_CAPACITY)
    pieces = (
        day[:, None],
        (states.step // TURNS_PER_DAY).astype(jnp.float32)[:, None] / 29.0,
        turn[:, None],
        states.money[:, (player, opponent)].astype(jnp.float32) / 100_000.0,
        states.unlocked_count[:, (player, opponent)].astype(jnp.float32) / 4.0,
        states.hires_today[:, (player, opponent)].astype(jnp.float32) / MAX_UNITS,
        states.shed[:, player].astype(jnp.float32) / SHED_CAPACITY,
        states.seeds[:, player].astype(jnp.float32) / SHED_CAPACITY,
        own_inventory,
        (states.market_inventory.astype(jnp.float32) - 10_000.0) / 2_000.0,
        states.market_price.astype(jnp.float32) / 500.0,
        town,
        states.town_count.astype(jnp.float32)[:, None] / MAX_SHOPS,
        kind[:, player],
        kind[:, opponent],
        crop_counts,
        crop_yield,
        animal_counts,
        animal_yield,
        jnp.stack(
            (
                jnp.sum(states.unit_active[:, player], axis=-1),
                jnp.sum(states.unit_active[:, opponent], axis=-1),
            ),
            axis=-1,
        ).astype(jnp.float32)
        / MAX_UNITS,
        (jnp.sum(states.shed[:, player], axis=-1).astype(jnp.float32) / SHED_CAPACITY)[:, None],
    )
    result = jnp.concatenate(pieces, axis=-1)
    pad = GLOBAL_FEATURE_DIM_V1 - result.shape[-1]
    if pad < 0:
        raise ValueError("global feature schema exceeds frozen width")
    return jnp.pad(result, ((0, 0), (0, pad))).astype(jnp.float32)


def build_candidate_features_v1(
    states: State,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    econ: EconFeaturesV1,
    player: int,
) -> jax.Array:
    """Compact strategic summary for all 96 candidates.

    Full-core and every Full-econ field are still computed before this point.
    The hot-path transition stores only the quantities needed to learn ranking;
    exact execution remains in the controller and is never delegated to the net.
    """

    del player
    score = full_econ_score_v1(candidates, feasibility, econ)
    fields = [
        jnp.clip(score / 1_000.0, -1.0, 1.0),
        candidates.task_type.astype(jnp.float32) / 20.0,
        (candidates.owner_unit.astype(jnp.float32) + 1.0) / MAX_UNITS,
        (candidates.item_id.astype(jnp.float32) + 1.0) / (NUM_SHED_ITEMS + 1),
        candidates.quantity.astype(jnp.float32) / SHED_CAPACITY,
        candidates.mandatory.astype(jnp.float32),
        feasibility.legal_now.astype(jnp.float32),
        feasibility.path_steps.astype(jnp.float32) / 20.0,
        feasibility.operation_steps.astype(jnp.float32) / 4.0,
        (feasibility.deadline_step.astype(jnp.float32) - states.step[:, None]) / 100.0,
        feasibility.bankable_before_terminal.astype(jnp.float32),
        feasibility.cash_required.astype(jnp.float32) / 10_000.0,
        econ.expected_bank_delta_current_price / 10_000.0,
        econ.minimum_cash_during_task.astype(jnp.float32) / 10_000.0,
        econ.total_required_actions.astype(jnp.float32) / EPISODE_STEPS,
        (
            econ.action_opportunity_cost
            + econ.input_opportunity_cost
            + econ.expected_overflow_loss
            + econ.expected_decay_or_capacity_loss
            + econ.stochastic_event_risk
        )
        / 10_000.0,
    ]
    result = jnp.stack(fields, axis=-1)
    pad = CANDIDATE_FEATURE_DIM_V1 - result.shape[-1]
    if pad < 0:
        raise ValueError("candidate feature schema exceeds frozen width")
    return jnp.pad(result, ((0, 0), (0, 0), (0, pad))).astype(jnp.float16)


def _choose_v1(
    logits: jax.Array, key: jax.Array, deterministic: bool
) -> tuple[jax.Array, jax.Array]:
    choice = (
        jnp.argmax(logits, axis=-1).astype(jnp.int32)
        if deterministic
        else jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)
    )
    logprob = jnp.take_along_axis(
        jax.nn.log_softmax(logits, axis=-1), choice[:, None], axis=-1
    )[:, 0]
    saved = jnp.where(choice < MAX_CANDIDATES_V1, choice, -1).astype(jnp.int16)
    return saved, logprob.astype(jnp.float32)


def select_full_learned_candidates_v1(
    states: State,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    econ: EconFeaturesV1,
    candidate_logits: jax.Array,
    stop_logit: jax.Array,
    controller: ControllerStateV1,
    player: int,
    key: jax.Array,
    *,
    deterministic: bool,
    max_selections: int = MAX_SELECTIONS_V1,
    allow_nonpositive_econ: bool = False,
    candidate_eligibility_override: jax.Array | None = None,
    official_legality_only: bool = False,
) -> FullLearnedSelectionV1:
    batch_size = states.step.shape[0]
    selected = jnp.full((batch_size, max_selections), -1, dtype=jnp.int16)
    masks = jnp.zeros(
        (batch_size, max_selections, MAX_CANDIDATES_V1 + 1), dtype=jnp.bool_
    )
    logprob = jnp.zeros((batch_size,), dtype=jnp.float32)
    conflict = jnp.zeros((batch_size,), dtype=jnp.int32)
    alive = jnp.ones((batch_size,), dtype=jnp.bool_)
    score = full_econ_score_v1(candidates, feasibility, econ)
    positive_or_required = (
        (score > 0.0)
        | candidates.mandatory
        | (candidates.task_type == 20)
        | (candidates.task_type == 2)
    )
    if official_legality_only:
        # The learned policy may deliberately take a short-term loss or an
        # investment that the hand-written ROI model dislikes.  Exact current
        # legality/resource accounting remains enforced below by Full-core.
        eligible = candidates.present & feasibility.legal_now
    else:
        eligible = (
            candidates.present
            & feasibility.legal_now
            & econ.bankable_before_terminal
            & (positive_or_required | allow_nonpositive_econ)
        )
    if candidate_eligibility_override is not None:
        eligible = eligible & candidate_eligibility_override
    selection_feasibility = feasibility._replace(
        cash_required=jnp.where(
            candidates.mandatory,
            0,
            (
                feasibility.cash_required
                if official_legality_only
                else jnp.maximum(feasibility.cash_required, econ.cash_required)
            ),
        ).astype(jnp.int32)
    )
    ledger = initialize_e2_ledger_v1(states, controller, player)
    operating_feed_reserve = jnp.max(
        econ.simple_cash_buffer_after_commit - econ.minimum_cash_during_task,
        axis=-1,
    ).astype(jnp.int32)
    mandatory_cash = jnp.sum(
        jnp.where(
            candidates.mandatory & feasibility.legal_now,
            feasibility.cash_required,
            0,
        ),
        axis=-1,
        dtype=jnp.int32,
    )
    protected_cash = (
        jnp.zeros_like(ledger.cash_available)
        if official_legality_only
        else jnp.minimum(
            operating_feed_reserve,
            jnp.maximum(ledger.cash_available - mandatory_cash, 0),
        )
    )
    ledger = ledger._replace(cash_reserved=ledger.cash_reserved + protected_cash)
    candidate_index = jnp.arange(MAX_CANDIDATES_V1, dtype=jnp.int16)

    def body(index, carry):
        current_controller, current_ledger, choices, saved_masks, joint, conflicts, active, current_key = carry
        current_key, choice_key = jax.random.split(current_key)
        mask = candidate_mask_from_e2_ledger_v1(
            states, candidates, selection_feasibility, current_ledger, player
        ) & eligible
        already = jnp.any(
            choices[:, :, None] == candidate_index[None, None, :], axis=1
        )
        mask = mask & (~already) & active[:, None]
        mandatory = mask & candidates.mandatory
        has_mandatory = jnp.any(mandatory, axis=-1)
        mask = jnp.where(has_mandatory[:, None], mandatory, mask)
        stop_allowed = (~has_mandatory) | (~active)
        extended_mask = jnp.concatenate((mask, stop_allowed[:, None]), axis=-1)
        extended_logits = jnp.concatenate(
            (candidate_logits, stop_logit[:, None]), axis=-1
        )
        extended_logits = jnp.where(
            extended_mask, extended_logits, jnp.finfo(jnp.float32).min
        )
        choice, one_logprob = _choose_v1(extended_logits, choice_key, deterministic)
        repeated = jnp.any(choices == choice[:, None], axis=-1) & (choice >= 0)
        choices = choices.at[:, index].set(choice)
        saved_masks = saved_masks.at[:, index].set(extended_mask)
        current_controller = attach_e2_candidate_batched_v1(
            current_controller,
            candidates,
            selection_feasibility,
            choice,
            states.step,
        )
        current_ledger = reserve_e2_candidate_batched_v1(
            current_ledger, candidates, selection_feasibility, choice
        )
        return (
            current_controller,
            current_ledger,
            choices,
            saved_masks,
            joint + one_logprob,
            conflicts + repeated.astype(jnp.int32),
            active & (choice >= 0),
            current_key,
        )

    controller, _, selected, masks, logprob, conflict, _, _ = jax.lax.fori_loop(
        0,
        max_selections,
        body,
        (controller, ledger, selected, masks, logprob, conflict, alive, key),
    )
    return FullLearnedSelectionV1(controller, selected, masks, logprob, conflict)


def recompute_full_logprob_v1(
    candidate_logits: jax.Array,
    stop_logit: jax.Array,
    saved_masks: jax.Array,
    saved_indices: jax.Array,
) -> jax.Array:
    extended = jnp.concatenate((candidate_logits, stop_logit[:, None]), axis=-1)
    selected = jnp.where(
        saved_indices >= 0, saved_indices, MAX_CANDIDATES_V1
    ).astype(jnp.int32)

    def one(mask, choice):
        masked = jnp.where(mask, extended, jnp.finfo(jnp.float32).min)
        return jnp.take_along_axis(
            jax.nn.log_softmax(masked, axis=-1), choice[:, None], axis=-1
        )[:, 0]

    return jnp.sum(jax.vmap(one, in_axes=(1, 1), out_axes=1)(saved_masks, selected), axis=-1)


def _controller_batch_v1(batch_size: int) -> ControllerStateV1:
    one = reset_controller_state_v1()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape), one
    )


def initialize_full_learned_carry_v1(
    seeds: jax.Array, events: Events, key: jax.Array
) -> FullLearnedCarryV1:
    seeds = jnp.asarray(seeds, dtype=jnp.int32)
    batch_size = seeds.shape[0]
    return FullLearnedCarryV1(
        environment_state=jax.vmap(reset)(seeds),
        player0_controller=_controller_batch_v1(batch_size),
        player1_controller=_controller_batch_v1(batch_size),
        current_events=events,
        rng=key,
    )


def _player_decision_v1(
    states: State,
    controller: ControllerStateV1,
    tables: StaticTables,
    params: object,
    player: int,
    key: jax.Array,
    deterministic: bool,
):
    candidates = build_full_core_candidates_v1(states, controller, tables, player)
    feasibility = evaluate_full_core_feasibility_v1(states, candidates, tables, player)
    econ = build_full_econ_features_v1(states, candidates, feasibility, tables, player)
    global_features = build_global_features_v1(states, player)
    candidate_features = build_candidate_features_v1(
        states, candidates, feasibility, econ, player
    )
    output = apply_full_learned_model_v1(
        params, global_features, candidate_features, candidates.task_type
    )
    selection = select_full_learned_candidates_v1(
        states,
        candidates,
        feasibility,
        econ,
        output.candidate_logits,
        output.stop_logit,
        controller,
        player,
        key,
        deterministic=deterministic,
    )
    return (
        selection,
        global_features,
        candidate_features,
        candidates.task_type,
        output.value,
    )


def full_learned_step_v1(
    carry: FullLearnedCarryV1,
    tables: StaticTables,
    params: object,
    *,
    deterministic: bool = False,
    decision_interval: int = 1,
) -> tuple[FullLearnedCarryV1, FullLearnedTransitionV1]:
    states = carry.environment_state
    next_key, key0, key1 = jax.random.split(carry.rng, 3)
    controller0 = clear_invalidated_full_core_tasks_v1(
        states, carry.player0_controller, 0
    )
    controller1 = clear_invalidated_full_core_tasks_v1(
        states, carry.player1_controller, 1
    )
    should_decide = (
        (states.step[0] % decision_interval) == 0
    ) | (states.step[0] >= EPISODE_STEPS - 2)
    batch_size = states.step.shape[0]

    def decide(_):
        selection0, global0, features0, task0, value0 = _player_decision_v1(
            states, controller0, tables, params, 0, key0, deterministic
        )
        selection1, global1, features1, task1, value1 = _player_decision_v1(
            states, controller1, tables, params, 1, key1, deterministic
        )
        return (
            selection0.controller,
            selection1.controller,
            jnp.stack((global0, global1), axis=1),
            jnp.stack((features0, features1), axis=1),
            jnp.stack((task0, task1), axis=1),
            jnp.stack((selection0.masks, selection1.masks), axis=1),
            jnp.stack(
                (
                    selection0.selected_candidate_indices,
                    selection1.selected_candidate_indices,
                ),
                axis=1,
            ),
            jnp.stack((selection0.joint_logprob, selection1.joint_logprob), axis=1),
            jnp.stack((value0, value1), axis=1),
        )

    def continue_existing(_):
        stop_masks = jnp.zeros(
            (
                batch_size,
                2,
                MAX_SELECTIONS_V1,
                MAX_CANDIDATES_V1 + 1,
            ),
            dtype=jnp.bool_,
        ).at[..., -1].set(True)
        return (
            controller0,
            controller1,
            jnp.zeros((batch_size, 2, GLOBAL_FEATURE_DIM_V1), dtype=jnp.float32),
            jnp.zeros(
                (batch_size, 2, MAX_CANDIDATES_V1, CANDIDATE_FEATURE_DIM_V1),
                dtype=jnp.float16,
            ),
            jnp.zeros((batch_size, 2, MAX_CANDIDATES_V1), dtype=jnp.int8),
            stop_masks,
            jnp.full(
                (batch_size, 2, MAX_SELECTIONS_V1), -1, dtype=jnp.int16
            ),
            jnp.zeros((batch_size, 2), dtype=jnp.float32),
            jnp.zeros((batch_size, 2), dtype=jnp.float32),
        )

    (
        selected_controller0,
        selected_controller1,
        global_features,
        candidate_features,
        candidate_task_type,
        task_masks,
        task_selected_indices,
        old_logprob,
        old_value,
    ) = jax.lax.cond(should_decide, decide, continue_existing, operand=None)
    bundle = compile_full_core_action_bundle_v1(
        states, selected_controller0, selected_controller1
    )
    next_states = batched_step_sync(states, bundle.action, carry.current_events, tables)
    controller0, _ = update_full_core_controller_from_effects_v1(
        states, next_states, selected_controller0, bundle.player0, 0
    )
    controller1, _ = update_full_core_controller_from_effects_v1(
        states, next_states, selected_controller1, bundle.player1, 1
    )
    day_end = ((next_states.step % TURNS_PER_DAY) == 0) & (~next_states.done)
    controller0 = cleanup_full_controller_day_end_v1(
        controller0, next_states.unit_active[:, 0], day_end
    )
    controller1 = cleanup_full_controller_day_end_v1(
        controller1, next_states.unit_active[:, 1], day_end
    )
    outcome = jnp.sign(next_states.money[:, 0] - next_states.money[:, 1]).astype(jnp.float32)
    just_finished = next_states.done & (~states.done)
    reward = jnp.stack((outcome, -outcome), axis=1) * just_finished[:, None]
    transition = FullLearnedTransitionV1(
        global_features=global_features,
        candidate_features=candidate_features,
        candidate_task_type=candidate_task_type,
        task_masks=task_masks,
        task_selected_indices=task_selected_indices,
        old_logprob=old_logprob,
        old_value=old_value,
        reward=reward,
        done=jnp.broadcast_to(next_states.done[:, None], reward.shape),
    )
    return FullLearnedCarryV1(
        next_states, controller0, controller1, carry.current_events, next_key
    ), transition


def _bootstrap_value_v1(
    carry: FullLearnedCarryV1, tables: StaticTables, params: object
) -> jax.Array:
    values = []
    for player, controller in (
        (0, carry.player0_controller),
        (1, carry.player1_controller),
    ):
        controller = clear_invalidated_full_core_tasks_v1(
            carry.environment_state, controller, player
        )
        candidates = build_full_core_candidates_v1(
            carry.environment_state, controller, tables, player
        )
        feasibility = evaluate_full_core_feasibility_v1(
            carry.environment_state, candidates, tables, player
        )
        econ = build_full_econ_features_v1(
            carry.environment_state, candidates, feasibility, tables, player
        )
        global_features = build_global_features_v1(carry.environment_state, player)
        candidate_features = build_candidate_features_v1(
            carry.environment_state, candidates, feasibility, econ, player
        )
        values.append(
            apply_full_learned_model_v1(
                params, global_features, candidate_features, candidates.task_type
            ).value
        )
    result = jnp.stack(values, axis=1)
    return jnp.where(carry.environment_state.done[:, None], 0.0, result)


def make_full_learned_collector_v1(
    *, rollout_steps: int, deterministic: bool = False, decision_interval: int = 1
):
    def collect(
        initial_carry: FullLearnedCarryV1,
        tables: StaticTables,
        params: object,
    ) -> FullLearnedRolloutV1:
        def body(carry, _):
            return full_learned_step_v1(
                carry,
                tables,
                params,
                deterministic=deterministic,
                decision_interval=decision_interval,
            )

        final_carry, transitions = jax.lax.scan(
            body, initial_carry, xs=None, length=rollout_steps
        )
        return FullLearnedRolloutV1(
            final_carry=final_carry,
            transitions=transitions,
            bootstrap_value=_bootstrap_value_v1(final_carry, tables, params),
        )

    return collect


def make_full_strided_collector_v1(
    *,
    rollout_steps: int,
    sample_stride: int,
    deterministic: bool = False,
    decision_interval: int = 1,
    include_final_sample: bool = False,
):
    """Save every environment value/reward but only periodic policy snapshots."""

    if rollout_steps <= 0 or sample_stride <= 0:
        raise ValueError("rollout_steps and sample_stride must be positive")
    groups, remainder = divmod(rollout_steps, sample_stride)

    def collect(
        initial_carry: FullLearnedCarryV1,
        tables: StaticTables,
        params: object,
    ) -> FullStridedRolloutV1:
        def one_step(carry, _):
            return full_learned_step_v1(
                carry,
                tables,
                params,
                deterministic=deterministic,
                decision_interval=decision_interval,
            )

        def one_group(carry, _):
            next_carry, transitions = jax.lax.scan(
                one_step, carry, xs=None, length=sample_stride
            )
            sample = jax.tree.map(lambda value: value[0], transitions)
            timeline = FullTimelineV1(
                old_value=transitions.old_value,
                reward=transitions.reward,
                done=transitions.done,
            )
            return next_carry, (sample, timeline)

        carry = initial_carry
        sample_parts = []
        timeline_parts = []
        if groups:
            carry, (group_samples, group_timeline) = jax.lax.scan(
                one_group, carry, xs=None, length=groups
            )
            sample_parts.append(group_samples)
            timeline_parts.append(
                jax.tree.map(
                    lambda value: value.reshape(
                        (groups * sample_stride,) + value.shape[2:]
                    ),
                    group_timeline,
                )
            )
        if remainder:
            carry, tail = jax.lax.scan(
                one_step, carry, xs=None, length=remainder
            )
            sample_parts.append(jax.tree.map(lambda value: value[0:1], tail))
            timeline_parts.append(
                FullTimelineV1(tail.old_value, tail.reward, tail.done)
            )
            if include_final_sample and remainder > 1:
                sample_parts.append(jax.tree.map(lambda value: value[-1:], tail))
        elif include_final_sample and rollout_steps > 1:
            # This branch is intentionally unavailable for the current 719-step
            # schedule: a full stride ends on a non-decision step.  Keep the
            # contract explicit instead of silently duplicating a snapshot.
            raise ValueError("include_final_sample requires a non-zero remainder")

        transitions = (
            sample_parts[0]
            if len(sample_parts) == 1
            else jax.tree.map(
                lambda *values: jnp.concatenate(values, axis=0), *sample_parts
            )
        )
        timeline = (
            timeline_parts[0]
            if len(timeline_parts) == 1
            else jax.tree.map(
                lambda *values: jnp.concatenate(values, axis=0), *timeline_parts
            )
        )
        return FullStridedRolloutV1(
            final_carry=carry,
            transitions=transitions,
            timeline=timeline,
            bootstrap_value=_bootstrap_value_v1(carry, tables, params),
        )

    return collect
