"""G0 audit collector for saved tasks, masks and resource ledgers."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture_jax.constants import EPISODE_STEPS
from kaggriculture_jax.types import StaticTables

from .constants import MAX_SELECTIONS_V1
from .e2_core import (
    initialize_e2_ledger_v1,
    reserve_e2_candidate_batched_v1,
)
from .e4_core import build_full_core_candidates_v1, evaluate_full_core_feasibility_v1
from .e5_econ import build_full_econ_features_v1
from .learned_v1 import (
    FullLearnedCarryV1,
    FullLearnedTransitionV1,
    FullTimelineV1,
    full_learned_step_v1,
)
from .lifecycle import clear_invalidated_full_core_tasks_v1
from .schema import ControllerStateV1, LedgerV1


class FullG0AuditSnapshotV1(NamedTuple):
    state_step: jax.Array
    player0_ledger: LedgerV1
    player1_ledger: LedgerV1
    player0_controller: ControllerStateV1
    player1_controller: ControllerStateV1
    player0_unlocked_count: jax.Array
    player1_unlocked_count: jax.Array


class FullG0AuditRolloutV1(NamedTuple):
    final_carry: FullLearnedCarryV1
    transitions: FullLearnedTransitionV1
    timeline: FullTimelineV1
    snapshots: FullG0AuditSnapshotV1


def _reserved_ledger_for_selected_v1(
    carry: FullLearnedCarryV1,
    controller: ControllerStateV1,
    selected: jax.Array,
    tables: StaticTables,
    player: int,
) -> LedgerV1:
    states = carry.environment_state
    controller = clear_invalidated_full_core_tasks_v1(states, controller, player)
    candidates = build_full_core_candidates_v1(states, controller, tables, player)
    feasibility = evaluate_full_core_feasibility_v1(
        states, candidates, tables, player
    )
    econ = build_full_econ_features_v1(
        states, candidates, feasibility, tables, player
    )
    selection_feasibility = feasibility._replace(
        cash_required=jnp.where(
            candidates.mandatory,
            0,
            jnp.maximum(feasibility.cash_required, econ.cash_required),
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
    protected_cash = jnp.minimum(
        operating_feed_reserve,
        jnp.maximum(ledger.cash_available - mandatory_cash, 0),
    )
    ledger = ledger._replace(cash_reserved=ledger.cash_reserved + protected_cash)

    def reserve(index, current):
        return reserve_e2_candidate_batched_v1(
            current, candidates, selection_feasibility, selected[:, index]
        )

    return jax.lax.fori_loop(0, MAX_SELECTIONS_V1, reserve, ledger)


def full_g0_audited_step_v1(
    carry: FullLearnedCarryV1,
    tables: StaticTables,
    params: object,
    *,
    deterministic: bool = False,
    decision_interval: int = 8,
):
    """Advance one step and retain the post-selection resource contract."""

    state_step = carry.environment_state.step
    next_carry, transition = full_learned_step_v1(
        carry,
        tables,
        params,
        deterministic=deterministic,
        decision_interval=decision_interval,
    )
    ledger0 = _reserved_ledger_for_selected_v1(
        carry,
        carry.player0_controller,
        transition.task_selected_indices[:, 0],
        tables,
        0,
    )
    ledger1 = _reserved_ledger_for_selected_v1(
        carry,
        carry.player1_controller,
        transition.task_selected_indices[:, 1],
        tables,
        1,
    )
    snapshot = FullG0AuditSnapshotV1(
        state_step=state_step,
        player0_ledger=ledger0,
        player1_ledger=ledger1,
        player0_controller=next_carry.player0_controller,
        player1_controller=next_carry.player1_controller,
        player0_unlocked_count=carry.environment_state.unlocked_count[:, 0],
        player1_unlocked_count=carry.environment_state.unlocked_count[:, 1],
    )
    return next_carry, (transition, snapshot)


def make_full_g0_audit_collector_v1(
    *,
    rollout_steps: int,
    sample_stride: int = 8,
    deterministic: bool = False,
    decision_interval: int = 8,
    include_final_sample: bool = False,
):
    """Collect all rewards and periodic transition/task/ledger snapshots."""

    if rollout_steps <= 0 or sample_stride <= 0:
        raise ValueError("rollout_steps and sample_stride must be positive")
    groups, remainder = divmod(rollout_steps, sample_stride)

    def collect(initial_carry, tables, params):
        def one_step(carry, _):
            return full_g0_audited_step_v1(
                carry,
                tables,
                params,
                deterministic=deterministic,
                decision_interval=decision_interval,
            )

        def one_group(carry, _):
            next_carry, (transitions, snapshots) = jax.lax.scan(
                one_step, carry, xs=None, length=sample_stride
            )
            sample_transition = jax.tree.map(lambda value: value[0], transitions)
            sample_snapshot = jax.tree.map(lambda value: value[0], snapshots)
            timeline = FullTimelineV1(
                transitions.old_value, transitions.reward, transitions.done
            )
            return next_carry, (sample_transition, sample_snapshot, timeline)

        carry = initial_carry
        transition_parts = []
        snapshot_parts = []
        timeline_parts = []
        if groups:
            carry, (group_transitions, group_snapshots, group_timeline) = jax.lax.scan(
                one_group, carry, xs=None, length=groups
            )
            transition_parts.append(group_transitions)
            snapshot_parts.append(group_snapshots)
            timeline_parts.append(
                jax.tree.map(
                    lambda value: value.reshape(
                        (groups * sample_stride,) + value.shape[2:]
                    ),
                    group_timeline,
                )
            )
        if remainder:
            carry, (tail_transitions, tail_snapshots) = jax.lax.scan(
                one_step, carry, xs=None, length=remainder
            )
            transition_parts.append(
                jax.tree.map(lambda value: value[0:1], tail_transitions)
            )
            snapshot_parts.append(
                jax.tree.map(lambda value: value[0:1], tail_snapshots)
            )
            timeline_parts.append(
                FullTimelineV1(
                    tail_transitions.old_value,
                    tail_transitions.reward,
                    tail_transitions.done,
                )
            )
            if include_final_sample and remainder > 1:
                transition_parts.append(
                    jax.tree.map(lambda value: value[-1:], tail_transitions)
                )
                snapshot_parts.append(
                    jax.tree.map(lambda value: value[-1:], tail_snapshots)
                )
        elif include_final_sample and rollout_steps > 1:
            raise ValueError("include_final_sample requires a non-zero remainder")

        concatenate = lambda parts: (
            parts[0]
            if len(parts) == 1
            else jax.tree.map(
                lambda *values: jnp.concatenate(values, axis=0), *parts
            )
        )
        return FullG0AuditRolloutV1(
            final_carry=carry,
            transitions=concatenate(transition_parts),
            timeline=concatenate(timeline_parts),
            snapshots=concatenate(snapshot_parts),
        )

    return collect
