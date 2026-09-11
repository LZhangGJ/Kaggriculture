"""Deployable two-stage V2 policy using the dynamic Full-core turn ledger."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import (
    EPISODE_STEPS,
    MARKET_LUT_SIZE,
    MARKET_MIN_INVENTORY,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_PRODUCTS,
    SHED_CAPACITY,
    MarketOp,
    UnitOp,
)
from kaggriculture_jax.types import Action, State, StaticTables

from .constants import (
    FailureCodeV1,
    MAX_CANDIDATES_V1,
    TaskStatusV1,
    TaskTypeV1,
)
from .candidateformer_v3 import (
    CANDIDATE_FEATURE_DIM_V3,
    PLAN_FEATURE_DIM_V3,
    PLAN_TOKEN_COUNT_V3,
    apply_candidateformer_market_v3,
    build_candidate_features_v3,
    build_plan_ledger_features_v3,
    encode_candidateformer_v3,
    is_candidateformer_params_v3,
)
from .e4_core import build_full_core_candidates_v1, evaluate_full_core_feasibility_v1
from .e4_executor import E4PlayerActionV1, compile_full_core_player_action_v1
from .e5_econ import build_full_econ_features_v1, full_econ_score_v1
from .learned_v1 import FullLearnedSelectionV1, select_full_learned_candidates_v1
from .learned_v2 import (
    FullPolicyContextV2,
    apply_full_learned_candidates_from_context_v2,
    apply_full_learned_model_v2,
    encode_full_learned_context_v2,
)
from .opponent_v2 import (
    CANDIDATE_FEATURE_DIM_V2,
    GLOBAL_FEATURE_DIM_V2,
    OpponentHistoryV2,
    build_candidate_features_v2,
    build_global_features_v2,
)
from .schema import CandidateV1, ControllerStateV1, EconFeaturesV1, FeasibilityV1
from .task_cards import ReplayTaskCardProgramV1
from .turn_ledger_v2 import (
    DynamicMarketMatrixV2,
    TurnLedgerV2,
    apply_dynamic_market_matrix_slot_v2,
    close_market_phase_v2,
    close_unit_phase_v2,
    initialize_projected_unit_ledger_v2,
    ledger_action_v2,
    refresh_dynamic_market_matrix_v2,
)


MARKET_CANDIDATE_COUNT_V2 = 21
MARKET_STOP_INDEX_V2 = MARKET_CANDIDATE_COUNT_V2


class DynamicMarketTraceV2(NamedTuple):
    global_features: jax.Array
    candidate_features: jax.Array
    candidate_task_type: jax.Array
    masks: jax.Array
    selected_indices: jax.Array
    requested_quantity: jax.Array
    filled_quantity: jax.Array
    joint_logprob: jax.Array


class DynamicFullDecisionV2(NamedTuple):
    controller: ControllerStateV1
    action: Action
    ledger: TurnLedgerV2
    unit_selection: FullLearnedSelectionV1
    unit_global_features: jax.Array
    unit_candidate_features: jax.Array
    unit_candidate_task_type: jax.Array
    market_trace: DynamicMarketTraceV2
    value: jax.Array
    effect_action: E4PlayerActionV1


class DynamicUnitDecisionV2(NamedTuple):
    """Frozen unit-phase choice plus the once-per-step Full-econ template."""

    controller: ControllerStateV1
    action: Action
    selection: FullLearnedSelectionV1
    global_features: jax.Array
    candidate_features: jax.Array
    candidate_task_type: jax.Array
    plan_features: jax.Array
    policy_context: FullPolicyContextV2
    candidates: CandidateV1
    feasibility: FeasibilityV1
    econ: EconFeaturesV1
    market_template_score: jax.Array
    value: jax.Array
    effect_action: E4PlayerActionV1


def _choose_dynamic_v2(
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
    return choice, logprob.astype(jnp.float32)


def _exact_dynamic_sell_revenue_v2(
    ledger: TurnLedgerV2, tables: StaticTables
) -> jax.Array:
    """Quote all current shed sales in O(1) with the official price prefix."""

    quantity = ledger.shed[:, :NUM_PRODUCTS].astype(jnp.int32)
    start = jnp.clip(
        ledger.market_inventory.astype(jnp.int32) - MARKET_MIN_INVENTORY,
        0,
        MARKET_LUT_SIZE - 1,
    )
    available = jnp.minimum(quantity, MARKET_LUT_SIZE - start)
    end = start + available
    product = jnp.arange(NUM_PRODUCTS, dtype=jnp.int32)[None, :]
    prefix = tables.market_price_prefix
    revenue = prefix[product, end] - prefix[product, start]
    tail = quantity - available
    revenue = revenue + tail * tables.market_price[:, -1][None, :].astype(
        jnp.int32
    )
    return revenue.astype(jnp.float32)


def build_lightweight_dynamic_market_features_v2(
    base_features: jax.Array,
    base_mandatory: jax.Array,
    dynamic_market: DynamicMarketMatrixV2,
    template_econ,
    template_score: jax.Array,
    ledger: TurnLedgerV2,
    tables: StaticTables,
) -> tuple[jax.Array, jax.Array]:
    """Refresh the 21 transactional slots without rescanning all 96 routes.

    Long-horizon value/risk comes from the once-per-turn Full-econ template.
    Current legality, quantity, cash, capacity and direct sale revenue come
    from TurnLedgerV2.  Execution is still performed by the exact ledger.
    """

    market = base_features[:, :MARKET_CANDIDATE_COUNT_V2].astype(jnp.float32)
    score = template_score[:, :MARKET_CANDIDATE_COUNT_V2]
    expected_bank = template_econ.expected_bank_delta_current_price[
        :, :MARKET_CANDIDATE_COUNT_V2
    ]
    sell_revenue = _exact_dynamic_sell_revenue_v2(ledger, tables)
    score = score.at[:, 12:21].set(sell_revenue)
    expected_bank = expected_bank.at[:, 12:21].set(sell_revenue)
    cash_required = dynamic_market.cash_required
    minimum_cash = jnp.maximum(
        ledger.money_nominal[:, None] - cash_required, 0
    ).astype(jnp.float32)
    total_actions = dynamic_market.present.astype(jnp.float32)
    risk = (
        template_econ.action_opportunity_cost
        + template_econ.input_opportunity_cost
        + template_econ.expected_overflow_loss
        + template_econ.expected_decay_or_capacity_loss
        + template_econ.stochastic_event_risk
    )[:, :MARKET_CANDIDATE_COUNT_V2]
    market = market.at[:, :, 0].set(jnp.clip(score / 1_000.0, -1.0, 1.0))
    market = market.at[:, :, 4].set(
        dynamic_market.quantity.astype(jnp.float32) / SHED_CAPACITY
    )
    market = market.at[:, :, 5].set(
        base_mandatory[:, :MARKET_CANDIDATE_COUNT_V2].astype(jnp.float32)
    )
    market = market.at[:, :, 6].set(
        dynamic_market.present.astype(jnp.float32)
    )
    market = market.at[:, :, 8].set(total_actions / 4.0)
    market = market.at[:, :, 10].set(
        dynamic_market.present.astype(jnp.float32)
    )
    market = market.at[:, :, 11].set(cash_required.astype(jnp.float32) / 10_000.0)
    market = market.at[:, :, 12].set(expected_bank / 10_000.0)
    market = market.at[:, :, 13].set(minimum_cash / 10_000.0)
    market = market.at[:, :, 14].set(total_actions / EPISODE_STEPS)
    market = market.at[:, :, 15].set(risk / 10_000.0)
    if market.shape[-1] == CANDIDATE_FEATURE_DIM_V3:
        # Refresh detailed fields whose truth changes after each market order.
        market = market.at[:, :, 32].set(
            cash_required.astype(jnp.float32) / 10_000.0
        )
        market = market.at[:, :, 36].set(
            dynamic_market.quantity.astype(jnp.float32) / SHED_CAPACITY
        )
    return market.astype(jnp.float16), score.astype(jnp.float32)


def _actor_unit_action_v2(player_action, player: int) -> Action:
    batch_size = player_action.unit_op.shape[0]
    return Action(
        unit_op=jnp.full(
            (batch_size, 2, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8
        ).at[:, player].set(player_action.unit_op),
        unit_item=jnp.full(
            (batch_size, 2, MAX_UNITS), -1, dtype=jnp.int8
        ).at[:, player].set(player_action.unit_item),
        unit_amount=jnp.ones(
            (batch_size, 2, MAX_UNITS), dtype=jnp.int32
        ).at[:, player].set(player_action.unit_amount),
        unit_count=jnp.ones((batch_size, 2), dtype=jnp.int8).at[:, player].set(
            player_action.unit_count
        ),
        market_op=jnp.full(
            (batch_size, 2, MAX_MARKET_ORDERS), MarketOp.NONE, dtype=jnp.int8
        ),
        market_item=jnp.full(
            (batch_size, 2, MAX_MARKET_ORDERS), -1, dtype=jnp.int8
        ),
        market_amount=jnp.zeros(
            (batch_size, 2, MAX_MARKET_ORDERS), dtype=jnp.int32
        ),
        market_count=jnp.zeros((batch_size, 2), dtype=jnp.int8),
    )


def combine_dynamic_actions_v2(player0: Action, player1: Action) -> Action:
    """Combine independently planned actor-relative actions without reordering."""

    return Action(
        unit_op=jnp.stack((player0.unit_op[:, 0], player1.unit_op[:, 1]), axis=1),
        unit_item=jnp.stack(
            (player0.unit_item[:, 0], player1.unit_item[:, 1]), axis=1
        ),
        unit_amount=jnp.stack(
            (player0.unit_amount[:, 0], player1.unit_amount[:, 1]), axis=1
        ),
        unit_count=jnp.stack(
            (player0.unit_count[:, 0], player1.unit_count[:, 1]), axis=1
        ),
        market_op=jnp.stack(
            (player0.market_op[:, 0], player1.market_op[:, 1]), axis=1
        ),
        market_item=jnp.stack(
            (player0.market_item[:, 0], player1.market_item[:, 1]), axis=1
        ),
        market_amount=jnp.stack(
            (player0.market_amount[:, 0], player1.market_amount[:, 1]), axis=1
        ),
        market_count=jnp.stack(
            (player0.market_count[:, 0], player1.market_count[:, 1]), axis=1
        ),
    )


def attach_dynamic_market_sequence_v2(
    controller: ControllerStateV1,
    candidates: CandidateV1,
    feasibility: FeasibilityV1,
    selected_indices: jax.Array,
    requested_quantity: jax.Array,
    current_step: jax.Array,
) -> ControllerStateV1:
    """Write the ordered dynamic market sequence with one matrix scatter.

    Every dynamic market candidate consumes exactly one persistent market-task
    slot.  The old implementation rescanned all ten slots after every order.
    Here cumulative ranks pair valid orders with initially free slots once;
    this is exactly the same first-free ordering and does not alter settlement.
    """

    batch_size, order_count = selected_indices.shape
    batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    valid = selected_indices >= 0
    safe = jnp.clip(
        selected_indices.astype(jnp.int32), 0, MARKET_CANDIDATE_COUNT_V2 - 1
    )
    task_type = candidates.task_type[batch, safe]
    item_id = candidates.item_id[batch, safe]
    quantity = jnp.where(
        task_type == TaskTypeV1.HIRE_WORKER,
        1,
        requested_quantity.astype(jnp.int16),
    ).astype(jnp.int16)
    deadline = feasibility.deadline_step[batch, safe]

    tasks = controller.market_tasks
    free = tasks.status != TaskStatusV1.ACTIVE
    free_rank = jnp.cumsum(free.astype(jnp.int32), axis=1) - 1
    order_rank = jnp.cumsum(valid.astype(jnp.int32), axis=1) - 1
    write = (
        valid[:, :, None]
        & free[:, None, :]
        & (order_rank[:, :, None] == free_rank[:, None, :])
    )
    has_write = jnp.any(write, axis=1)
    source_order = jnp.argmax(write, axis=1).astype(jnp.int32)
    source_batch = jnp.arange(batch_size, dtype=jnp.int32)[:, None]

    def selected(value):
        return value[source_batch, source_order]

    tasks = tasks._replace(
        task_type=jnp.where(has_write, selected(task_type), tasks.task_type),
        item_id=jnp.where(has_write, selected(item_id), tasks.item_id),
        quantity=jnp.where(has_write, selected(quantity), tasks.quantity),
        start_step=jnp.where(
            has_write, current_step[:, None], tasks.start_step
        ),
        deadline_step=jnp.where(
            has_write, selected(deadline), tasks.deadline_step
        ),
        status=jnp.where(has_write, TaskStatusV1.ACTIVE, tasks.status),
        failure_code=jnp.where(
            has_write, FailureCodeV1.NONE, tasks.failure_code
        ),
    )
    return controller._replace(market_tasks=tasks)


def select_dynamic_unit_phase_v2(
    states: State,
    controller: ControllerStateV1,
    history: OpponentHistoryV2,
    tables: StaticTables,
    params: object,
    player: int,
    key: jax.Array,
    *,
    deterministic: bool,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = False,
) -> DynamicUnitDecisionV2:
    """Choose this operation step's unit work and compute Full-econ once."""

    base_candidates = build_full_core_candidates_v1(
        states, controller, tables, player, task_card_program
    )
    base_feasibility = evaluate_full_core_feasibility_v1(
        states, base_candidates, tables, player
    )
    market_template_legal = base_feasibility.legal_now.at[
        :, :MARKET_CANDIDATE_COUNT_V2
    ].set(True)
    market_template_bankable = base_feasibility.bankable_before_terminal.at[
        :, :MARKET_CANDIDATE_COUNT_V2
    ].set(True)
    template_feasibility = base_feasibility._replace(
        legal_now=market_template_legal,
        bankable_before_terminal=market_template_bankable,
    )
    base_econ = build_full_econ_features_v1(
        states, base_candidates, template_feasibility, tables, player
    )
    market_template_score = full_econ_score_v1(
        base_candidates, template_feasibility, base_econ
    )
    unit_global = build_global_features_v2(
        states,
        player,
        history,
        include_opponent=include_opponent,
        include_history=include_history,
    )
    base_unit_features = build_candidate_features_v2(
        states,
        base_candidates,
        base_feasibility,
        base_econ,
        player,
        tables,
        include_opponent=include_opponent,
    )
    if is_candidateformer_params_v3(params):
        unit_features = build_candidate_features_v3(
            states,
            base_candidates,
            base_feasibility,
            base_econ,
            base_unit_features,
        )
        plan_features = build_plan_ledger_features_v3(
            states,
            controller,
            base_candidates,
            base_feasibility,
            base_econ,
            player,
        )
        policy_context, unit_output = encode_candidateformer_v3(
            params,
            unit_global,
            plan_features,
            unit_features,
            base_candidates.task_type,
        )
    else:
        unit_features = base_unit_features
        plan_features = jnp.zeros(
            (states.step.shape[0], PLAN_TOKEN_COUNT_V3, PLAN_FEATURE_DIM_V3),
            dtype=jnp.float32,
        )
        policy_context = encode_full_learned_context_v2(params, unit_global)
        unit_output = apply_full_learned_candidates_from_context_v2(
            params, policy_context, unit_features, base_candidates.task_type
        )
    unit_only = jnp.broadcast_to(
        (jnp.arange(MAX_CANDIDATES_V1) >= MARKET_CANDIDATE_COUNT_V2)[None, :],
        base_candidates.present.shape,
    )
    unit_selection = select_full_learned_candidates_v1(
        states,
        base_candidates,
        base_feasibility,
        base_econ,
        unit_output.candidate_logits,
        unit_output.stop_logit,
        controller,
        player,
        key,
        deterministic=deterministic,
        allow_nonpositive_econ=allow_nonpositive_econ,
        candidate_eligibility_override=unit_only,
        official_legality_only=is_candidateformer_params_v3(params),
    )

    compiled_unit = compile_full_core_player_action_v1(
        states, unit_selection.controller, player
    )
    return DynamicUnitDecisionV2(
        controller=unit_selection.controller,
        action=_actor_unit_action_v2(compiled_unit, player),
        selection=unit_selection,
        global_features=unit_global,
        candidate_features=unit_features,
        candidate_task_type=base_candidates.task_type,
        plan_features=plan_features,
        policy_context=policy_context,
        candidates=base_candidates,
        feasibility=base_feasibility,
        econ=base_econ,
        market_template_score=market_template_score,
        value=unit_output.value,
        effect_action=compiled_unit,
    )


def select_dynamic_full_core_v2(
    states: State,
    controller: ControllerStateV1,
    history: OpponentHistoryV2,
    tables: StaticTables,
    params: object,
    player: int,
    key: jax.Array,
    *,
    deterministic: bool,
    include_opponent: bool = True,
    include_history: bool = False,
    task_card_program: ReplayTaskCardProgramV1 | None = None,
    allow_nonpositive_econ: bool = False,
    unit_decision: DynamicUnitDecisionV2 | None = None,
    projected_unit_state: State | None = None,
) -> DynamicFullDecisionV2:
    """Select unit work once, then autoregress over ten official market slots.

    A joint two-player rollout supplies ``unit_decision`` and the same
    ``projected_unit_state`` to both actors.  Standalone callers retain the
    original behavior and project the selected actor against opponent PASS.
    """

    batch_size = states.step.shape[0]
    if unit_decision is None:
        key, unit_key = jax.random.split(key)
        unit_decision = select_dynamic_unit_phase_v2(
            states,
            controller,
            history,
            tables,
            params,
            player,
            unit_key,
            deterministic=deterministic,
            include_opponent=include_opponent,
            include_history=include_history,
            task_card_program=task_card_program,
            allow_nonpositive_econ=allow_nonpositive_econ,
        )
    base_candidates = unit_decision.candidates
    base_feasibility = unit_decision.feasibility
    base_econ = unit_decision.econ
    market_template_score = unit_decision.market_template_score
    unit_global = unit_decision.global_features
    unit_features = unit_decision.candidate_features
    unit_selection = unit_decision.selection
    policy_context = unit_decision.policy_context
    if projected_unit_state is None:
        ledger, _ = close_unit_phase_v2(
            states, unit_decision.action, player
        )
    else:
        ledger = initialize_projected_unit_ledger_v2(
            projected_unit_state, unit_decision.action, player
        )
    current_controller = unit_decision.controller
    alive = jnp.ones((batch_size,), dtype=jnp.bool_)
    joint = jnp.zeros((batch_size,), dtype=jnp.float32)
    batch = jnp.arange(batch_size, dtype=jnp.int32)

    def one_market_substep(carry, _):
        current_ledger, one_controller, one_alive, current_key, current_joint = carry
        current_key, choice_key = jax.random.split(current_key)

        def active_branch(values):
            value, controller_value, active, random_key, accumulated = values
            dynamic_market = refresh_dynamic_market_matrix_v2(
                base_candidates.quantity, value, tables
            )
            market_features, score = build_lightweight_dynamic_market_features_v2(
                unit_features,
                base_candidates.mandatory,
                dynamic_market,
                base_econ,
                market_template_score,
                value,
                tables,
            )
            global_features = unit_global
            market_task = base_candidates.task_type[:, :MARKET_CANDIDATE_COUNT_V2]
            output = (
                apply_candidateformer_market_v3(
                    params, policy_context, market_features, market_task
                )
                if is_candidateformer_params_v3(params)
                else apply_full_learned_candidates_from_context_v2(
                    params, policy_context, market_features, market_task
                )
            )
            positive_or_required = (
                (score > 0.0)
                | base_candidates.mandatory[:, :MARKET_CANDIDATE_COUNT_V2]
                | (
                    base_candidates.task_type[:, :MARKET_CANDIDATE_COUNT_V2]
                    == TaskTypeV1.TERMINAL_LIQUIDATION
                )
                | (
                    base_candidates.task_type[:, :MARKET_CANDIDATE_COUNT_V2]
                    == TaskTypeV1.SAFE_RECOVERY
                )
            )
            eligible = (
                dynamic_market.present
                & (
                    positive_or_required
                    | allow_nonpositive_econ
                    | is_candidateformer_params_v3(params)
                )
                & active[:, None]
            )
            extended_mask = jnp.concatenate(
                (eligible, jnp.ones((batch_size, 1), dtype=jnp.bool_)), axis=-1
            )
            extended_mask = jnp.where(
                active[:, None],
                extended_mask,
                jnp.zeros_like(extended_mask).at[:, MARKET_STOP_INDEX_V2].set(True),
            )
            extended_logits = jnp.concatenate(
                (output.candidate_logits, output.stop_logit[:, None]), axis=-1
            )
            masked_logits = jnp.where(
                extended_mask, extended_logits, jnp.finfo(jnp.float32).min
            )
            raw_choice, one_logprob = _choose_dynamic_v2(
                masked_logits, random_key, deterministic
            )
            selected = jnp.where(
                raw_choice < MARKET_CANDIDATE_COUNT_V2, raw_choice, -1
            ).astype(jnp.int16)
            safe = jnp.clip(selected.astype(jnp.int32), 0, 20)
            requested = jnp.where(
                selected >= 0, dynamic_market.quantity[batch, safe], 0
            ).astype(jnp.int16)
            before_count = value.market_count.astype(jnp.int32)
            safe_count = jnp.clip(before_count, 0, MAX_MARKET_ORDERS - 1)

            def commit_selected(args):
                ledger_value, controller_state = args
                next_ledger = apply_dynamic_market_matrix_slot_v2(
                    ledger_value, dynamic_market, selected, tables
                )
                committed = jnp.where(
                    selected >= 0,
                    next_ledger.market_filled_amount[batch, safe_count],
                    0,
                ).astype(jnp.int16)
                return next_ledger, controller_state, committed

            def skip_commit(args):
                ledger_value, controller_state = args
                return (
                    ledger_value,
                    controller_state,
                    jnp.zeros((batch_size,), dtype=jnp.int16),
                )

            next_value, controller_value, filled = jax.lax.cond(
                jnp.any(selected >= 0),
                commit_selected,
                skip_commit,
                (value, controller_value),
            )
            next_alive = active & (selected >= 0)
            return (
                (
                    next_value,
                    controller_value,
                    next_alive,
                    current_key,
                    accumulated + one_logprob,
                ),
                (
                    global_features,
                    market_features,
                    market_task,
                    extended_mask,
                    selected,
                    requested,
                    filled,
                ),
            )

        def inactive_branch(values):
            value, controller_value, active, _, accumulated = values
            stop_mask = jnp.zeros(
                (batch_size, MARKET_CANDIDATE_COUNT_V2 + 1), dtype=jnp.bool_
            ).at[:, MARKET_STOP_INDEX_V2].set(True)
            return (
                (value, controller_value, active, current_key, accumulated),
                (
                    jnp.zeros((batch_size, GLOBAL_FEATURE_DIM_V2), dtype=jnp.float32),
                    jnp.zeros(
                        (
                            batch_size,
                            MARKET_CANDIDATE_COUNT_V2,
                            unit_features.shape[-1],
                        ),
                        dtype=jnp.float16,
                    ),
                    jnp.zeros(
                        (batch_size, MARKET_CANDIDATE_COUNT_V2), dtype=jnp.int8
                    ),
                    stop_mask,
                    jnp.full((batch_size,), -1, dtype=jnp.int16),
                    jnp.zeros((batch_size,), dtype=jnp.int16),
                    jnp.zeros((batch_size,), dtype=jnp.int16),
                ),
            )

        return jax.lax.cond(
            jnp.any(one_alive),
            active_branch,
            inactive_branch,
            (current_ledger, one_controller, one_alive, choice_key, current_joint),
        )

    (ledger, current_controller, _, key, joint), rows = jax.lax.scan(
        one_market_substep,
        (ledger, current_controller, alive, key, joint),
        xs=None,
        length=MAX_MARKET_ORDERS,
    )
    (
        global_rows,
        feature_rows,
        task_rows,
        mask_rows,
        selected_rows,
        requested_rows,
        filled_rows,
    ) = rows
    current_controller = attach_dynamic_market_sequence_v2(
        current_controller,
        base_candidates,
        base_feasibility,
        jnp.swapaxes(selected_rows, 0, 1),
        jnp.swapaxes(requested_rows, 0, 1),
        states.step,
    )
    ledger = close_market_phase_v2(ledger)
    trace = DynamicMarketTraceV2(
        global_features=jnp.swapaxes(global_rows, 0, 1).astype(jnp.float32),
        candidate_features=jnp.swapaxes(feature_rows, 0, 1).astype(jnp.float16),
        candidate_task_type=jnp.swapaxes(task_rows, 0, 1).astype(jnp.int8),
        masks=jnp.swapaxes(mask_rows, 0, 1),
        selected_indices=jnp.swapaxes(selected_rows, 0, 1),
        requested_quantity=jnp.swapaxes(requested_rows, 0, 1),
        filled_quantity=jnp.swapaxes(filled_rows, 0, 1),
        joint_logprob=joint,
    )
    return DynamicFullDecisionV2(
        controller=current_controller,
        action=ledger_action_v2(ledger, player),
        ledger=ledger,
        unit_selection=unit_selection,
        unit_global_features=unit_global,
        unit_candidate_features=unit_features,
        unit_candidate_task_type=base_candidates.task_type,
        market_trace=trace,
        value=unit_decision.value,
        effect_action=unit_decision.effect_action,
    )


def recompute_dynamic_market_logprob_v2(
    params: object,
    trace: DynamicMarketTraceV2,
) -> jax.Array:
    """Recompute the stored teacher-forced market sequence for PPO/BC smoke."""

    rows = []
    for ordinal in range(MAX_MARKET_ORDERS):
        output = apply_full_learned_model_v2(
            params,
            trace.global_features[:, ordinal],
            trace.candidate_features[:, ordinal],
            trace.candidate_task_type[:, ordinal],
        )
        extended = jnp.concatenate(
            (output.candidate_logits, output.stop_logit[:, None]), axis=-1
        )
        masked = jnp.where(
            trace.masks[:, ordinal], extended, jnp.finfo(jnp.float32).min
        )
        selected = jnp.where(
            trace.selected_indices[:, ordinal] >= 0,
            trace.selected_indices[:, ordinal],
            MARKET_STOP_INDEX_V2,
        ).astype(jnp.int32)
        rows.append(
            jnp.take_along_axis(
                jax.nn.log_softmax(masked, axis=-1), selected[:, None], axis=-1
            )[:, 0]
        )
    return jnp.sum(jnp.stack(rows, axis=1), axis=1).astype(jnp.float32)
