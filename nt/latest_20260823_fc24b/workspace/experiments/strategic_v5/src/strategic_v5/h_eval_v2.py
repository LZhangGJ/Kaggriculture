"""Reusable staged H evaluator kernels with exact V1 hybrid semantics.

The monolithic H arena asks XLA to optimize learned selection, rule selection,
the official environment and diagnostics as one large graph.  These V2 kernels
separate those concerns so each fixed-shape executable can be compiled once,
persisted and reused for every checkpoint and rule-risk configuration.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import TURNS_PER_DAY
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.types import StaticTables

from .e4_core import build_full_core_candidates_v1, evaluate_full_core_feasibility_v1
from .e4_executor import (
    compile_full_core_action_bundle_v1,
    update_full_core_controller_from_effects_v1,
)
from .e5_econ import build_full_econ_features_v1, select_full_econ_candidates_v1
from .e5_rollout import cleanup_full_controller_day_end_v1
from .h_arena import HHybridCarryV1, _add_pair
from .learned_v1 import _player_decision_v1
from .lifecycle import clear_invalidated_full_core_tasks_v1
from .schema import ControllerStateV1


class HLearnedDecisionOutputV2(NamedTuple):
    controller: ControllerStateV1
    internal_resource_conflict: jax.Array
    next_key: jax.Array


class HRuleDecisionOutputV2(NamedTuple):
    controller: ControllerStateV1
    internal_resource_conflict: jax.Array


def make_h_learned_decision_kernel_v2(player: int):
    """Return one deterministic learned decision kernel for a fixed seat."""

    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")

    def decide(carry: HHybridCarryV1, tables: StaticTables, learned_params: object):
        next_key, learned_key = jax.random.split(carry.rng)
        controller = (
            carry.player0_controller if player == 0 else carry.player1_controller
        )
        controller = clear_invalidated_full_core_tasks_v1(
            carry.environment_state, controller, player
        )
        selection, _, _, _, _ = _player_decision_v1(
            carry.environment_state,
            controller,
            tables,
            learned_params,
            player,
            learned_key,
            True,
        )
        return HLearnedDecisionOutputV2(
            selection.controller,
            selection.internal_resource_conflict,
            next_key,
        )

    return decide


def make_h_rule_decision_kernel_v2(player: int):
    """Return one rule decision kernel with dynamic expected/scenario flags."""

    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")

    def decide(
        carry: HHybridCarryV1,
        tables: StaticTables,
        risk_flags: jax.Array,
    ):
        controller = (
            carry.player0_controller if player == 0 else carry.player1_controller
        )
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
            carry.environment_state,
            candidates,
            feasibility,
            tables,
            player,
            include_expected_risk=risk_flags[0],
            include_scenario_risk=risk_flags[1],
        )
        selection = select_full_econ_candidates_v1(
            carry.environment_state,
            candidates,
            feasibility,
            econ,
            controller,
            player,
        )
        return HRuleDecisionOutputV2(
            selection.controller,
            selection.internal_resource_conflict,
        )

    return decide


def h_execute_selected_step_v2(
    carry: HHybridCarryV1,
    tables: StaticTables,
    selected0: ControllerStateV1,
    selected1: ControllerStateV1,
    conflict0: jax.Array,
    conflict1: jax.Array,
    next_key: jax.Array,
) -> HHybridCarryV1:
    """Execute already selected controllers and retain exact V1 diagnostics."""

    states = carry.environment_state
    bundle = compile_full_core_action_bundle_v1(states, selected0, selected1)
    next_states = batched_step_sync(
        states, bundle.action, carry.current_events, tables
    )
    updated0, effect0 = update_full_core_controller_from_effects_v1(
        states, next_states, selected0, bundle.player0, 0
    )
    updated1, effect1 = update_full_core_controller_from_effects_v1(
        states, next_states, selected1, bundle.player1, 1
    )
    day_end = ((next_states.step % TURNS_PER_DAY) == 0) & (~next_states.done)
    updated0 = cleanup_full_controller_day_end_v1(
        updated0, next_states.unit_active[:, 0], day_end
    )
    updated1 = cleanup_full_controller_day_end_v1(
        updated1, next_states.unit_active[:, 1], day_end
    )
    diagnostics = carry.diagnostics._replace(
        invalid_raw_action_count=_add_pair(
            carry.diagnostics.invalid_raw_action_count,
            bundle.player0.diagnostics.invalid_raw_action_count,
            bundle.player1.diagnostics.invalid_raw_action_count,
        ),
        internal_resource_conflict_count=_add_pair(
            carry.diagnostics.internal_resource_conflict_count,
            conflict0,
            conflict1,
        ),
        unexpected_pass_count=_add_pair(
            carry.diagnostics.unexpected_pass_count,
            bundle.player0.diagnostics.unexpected_pass_count,
            bundle.player1.diagnostics.unexpected_pass_count,
        ),
        compiler_overlap_count=_add_pair(
            carry.diagnostics.compiler_overlap_count,
            bundle.player0.diagnostics.unit_compiler_overlap_count
            + bundle.player0.diagnostics.market_compiler_overlap_count,
            bundle.player1.diagnostics.unit_compiler_overlap_count
            + bundle.player1.diagnostics.market_compiler_overlap_count,
        ),
        effect_mismatch_count=_add_pair(
            carry.diagnostics.effect_mismatch_count,
            effect0.effect_mismatch_count,
            effect1.effect_mismatch_count,
        ),
        owner_inactive_count=_add_pair(
            carry.diagnostics.owner_inactive_count,
            effect0.owner_inactive_count,
            effect1.owner_inactive_count,
        ),
        deadline_missed_count=_add_pair(
            carry.diagnostics.deadline_missed_count,
            effect0.deadline_missed_count,
            effect1.deadline_missed_count,
        ),
        resource_unavailable_count=_add_pair(
            carry.diagnostics.resource_unavailable_count,
            effect0.resource_unavailable_count,
            effect1.resource_unavailable_count,
        ),
    )
    return HHybridCarryV1(
        next_states,
        updated0,
        updated1,
        carry.current_events,
        next_key,
        diagnostics,
    )


def staged_h_hybrid_step_v2(
    carry: HHybridCarryV1,
    tables: StaticTables,
    learned_params: object,
    risk_flags: jax.Array,
    *,
    learned_player: int,
) -> HHybridCarryV1:
    """Unjitted composition used as the exact reference for staged executables."""

    learned = make_h_learned_decision_kernel_v2(learned_player)(
        carry, tables, learned_params
    )
    rule_player = 1 - learned_player
    rule = make_h_rule_decision_kernel_v2(rule_player)(carry, tables, risk_flags)
    if learned_player == 0:
        selected0, conflict0 = learned.controller, learned.internal_resource_conflict
        selected1, conflict1 = rule.controller, rule.internal_resource_conflict
    else:
        selected0, conflict0 = rule.controller, rule.internal_resource_conflict
        selected1, conflict1 = learned.controller, learned.internal_resource_conflict
    return h_execute_selected_step_v2(
        carry,
        tables,
        selected0,
        selected1,
        conflict0,
        conflict1,
        learned.next_key,
    )
