"""Learned-vs-rule Full-core arena used by V5 H promotion panels."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax import reset
from kaggriculture_jax.constants import EPISODE_STEPS, TURNS_PER_DAY
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.types import Events, State, StaticTables

from .e4_core import build_full_core_candidates_v1, evaluate_full_core_feasibility_v1
from .e4_executor import (
    compile_full_core_action_bundle_v1,
    update_full_core_controller_from_effects_v1,
)
from .e5_econ import (
    build_full_econ_features_v1,
    select_full_econ_candidates_v1,
)
from .e5_rollout import cleanup_full_controller_day_end_v1
from .learned_v1 import _player_decision_v1
from .lifecycle import (
    clear_invalidated_full_core_tasks_v1,
    reset_controller_state_v1,
)
from .schema import ControllerStateV1


class HHybridDiagnosticsV1(NamedTuple):
    invalid_raw_action_count: jax.Array
    internal_resource_conflict_count: jax.Array
    unexpected_pass_count: jax.Array
    compiler_overlap_count: jax.Array
    effect_mismatch_count: jax.Array
    owner_inactive_count: jax.Array
    deadline_missed_count: jax.Array
    resource_unavailable_count: jax.Array


class HHybridCarryV1(NamedTuple):
    environment_state: State
    player0_controller: ControllerStateV1
    player1_controller: ControllerStateV1
    current_events: Events
    rng: jax.Array
    diagnostics: HHybridDiagnosticsV1


class HHybridArenaResultV1(NamedTuple):
    final_carry: HHybridCarryV1
    player0_outcome: jax.Array


def _controller_batch(batch_size: int) -> ControllerStateV1:
    one = reset_controller_state_v1()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape), one
    )


def _empty_diagnostics(batch_size: int) -> HHybridDiagnosticsV1:
    pair = lambda: jnp.zeros((batch_size, 2), dtype=jnp.int32)
    return HHybridDiagnosticsV1(*[pair() for _ in HHybridDiagnosticsV1._fields])


def initialize_h_hybrid_carry_v1(
    seeds: jax.Array, events: Events, key: jax.Array
) -> HHybridCarryV1:
    seeds = jnp.asarray(seeds, dtype=jnp.int32)
    batch_size = seeds.shape[0]
    return HHybridCarryV1(
        environment_state=jax.vmap(reset)(seeds),
        player0_controller=_controller_batch(batch_size),
        player1_controller=_controller_batch(batch_size),
        current_events=events,
        rng=key,
        diagnostics=_empty_diagnostics(batch_size),
    )


def _rule_decision_v1(
    states: State,
    controller: ControllerStateV1,
    tables: StaticTables,
    player: int,
    *,
    include_expected_risk: bool,
    include_scenario_risk: bool,
):
    candidates = build_full_core_candidates_v1(states, controller, tables, player)
    feasibility = evaluate_full_core_feasibility_v1(states, candidates, tables, player)
    econ = build_full_econ_features_v1(
        states,
        candidates,
        feasibility,
        tables,
        player,
        include_expected_risk=include_expected_risk,
        include_scenario_risk=include_scenario_risk,
    )
    return select_full_econ_candidates_v1(
        states, candidates, feasibility, econ, controller, player
    )


def _add_pair(prior: jax.Array, left: jax.Array, right: jax.Array) -> jax.Array:
    return prior + jnp.stack((left, right), axis=1).astype(jnp.int32)


def h_hybrid_step_v1(
    carry: HHybridCarryV1,
    tables: StaticTables,
    learned_params: object,
    *,
    learned_player: int,
    decision_interval: int = 8,
    include_expected_risk: bool = True,
    include_scenario_risk: bool = True,
) -> HHybridCarryV1:
    states = carry.environment_state
    next_key, learned_key = jax.random.split(carry.rng)
    controller0 = clear_invalidated_full_core_tasks_v1(
        states, carry.player0_controller, 0
    )
    controller1 = clear_invalidated_full_core_tasks_v1(
        states, carry.player1_controller, 1
    )
    should_decide = (
        (states.step[0] % decision_interval) == 0
    ) | (states.step[0] >= EPISODE_STEPS - 2)

    def learned_or_continue(controller, player):
        def decide(_):
            selection, _, _, _, _ = _player_decision_v1(
                states,
                controller,
                tables,
                learned_params,
                player,
                learned_key,
                True,
            )
            return selection.controller, selection.internal_resource_conflict

        return jax.lax.cond(
            should_decide,
            decide,
            lambda _: (
                controller,
                jnp.zeros((states.step.shape[0],), dtype=jnp.int32),
            ),
            operand=None,
        )

    if learned_player == 0:
        selected0, conflict0 = learned_or_continue(controller0, 0)
        selection1 = _rule_decision_v1(
            states,
            controller1,
            tables,
            1,
            include_expected_risk=include_expected_risk,
            include_scenario_risk=include_scenario_risk,
        )
        selected1, conflict1 = (
            selection1.controller,
            selection1.internal_resource_conflict,
        )
    elif learned_player == 1:
        selection0 = _rule_decision_v1(
            states,
            controller0,
            tables,
            0,
            include_expected_risk=include_expected_risk,
            include_scenario_risk=include_scenario_risk,
        )
        selected0, conflict0 = (
            selection0.controller,
            selection0.internal_resource_conflict,
        )
        selected1, conflict1 = learned_or_continue(controller1, 1)
    else:
        raise ValueError("learned_player must be 0 or 1")

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


def make_h_hybrid_arena_rollout_v1(
    *,
    learned_player: int,
    rollout_steps: int = EPISODE_STEPS - 1,
    decision_interval: int = 8,
    include_expected_risk: bool = True,
    include_scenario_risk: bool = True,
):
    def rollout(initial_carry, tables, learned_params):
        def body(carry, _):
            return (
                h_hybrid_step_v1(
                    carry,
                    tables,
                    learned_params,
                    learned_player=learned_player,
                    decision_interval=decision_interval,
                    include_expected_risk=include_expected_risk,
                    include_scenario_risk=include_scenario_risk,
                ),
                None,
            )

        final_carry, _ = jax.lax.scan(
            body, initial_carry, xs=None, length=rollout_steps
        )
        money = final_carry.environment_state.money
        return HHybridArenaResultV1(
            final_carry,
            jnp.sign(money[:, 0] - money[:, 1]).astype(jnp.int8),
        )

    return rollout
