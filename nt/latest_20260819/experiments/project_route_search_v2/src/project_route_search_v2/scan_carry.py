"""JAX scan carry helpers with lane-wise episode-boundary resets."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax.types import State

from .lifecycle import (
    controller_equal_per_batch_v2,
    initialize_project_controller_v2,
    reconcile_project_controller_v2,
)
from .schema import (
    ControllerCarryDiagnosticsV2,
    ControllerReconcileDiagnosticsV2,
    ControllerScanCarryV2,
    ProjectControllerStateV2,
)


def initialize_controller_scan_carry_v2(
    states: State,
    player: int,
) -> ControllerScanCarryV2:
    return ControllerScanCarryV2(
        environment_state=states,
        controller=initialize_project_controller_v2(states, player),
    )


def _select_lanes(mask: jax.Array, when_true, when_false):
    return jax.tree.map(
        lambda true_value, false_value: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (true_value.ndim - 1)),
            true_value,
            false_value,
        ),
        when_true,
        when_false,
    )


def advance_controller_scan_carry_v2(
    carry: ControllerScanCarryV2,
    next_states: State,
    episode_start: jax.Array,
    player: int,
) -> tuple[ControllerScanCarryV2, ControllerCarryDiagnosticsV2]:
    """Advance state/controller carry and hard-reset selected episode lanes."""

    episode_start = jnp.asarray(episode_start, dtype=jnp.bool_)
    fresh = initialize_project_controller_v2(next_states, player)
    reconciled, reconcile = reconcile_project_controller_v2(
        next_states, carry.controller, player
    )
    selected = _select_lanes(episode_start, fresh, reconciled)
    clean_after_reset = controller_equal_per_batch_v2(selected, fresh)
    contamination = episode_start & (~clean_after_reset)

    zero_reconcile = ControllerReconcileDiagnosticsV2(
        snapshot_was_valid=jnp.zeros_like(reconcile.snapshot_was_valid),
        money_delta=jnp.zeros_like(reconcile.money_delta),
        shed_delta=jnp.zeros_like(reconcile.shed_delta),
        seed_delta=jnp.zeros_like(reconcile.seed_delta),
        tile_kind_changed_count=jnp.zeros_like(reconcile.tile_kind_changed_count),
        tile_yield_changed_count=jnp.zeros_like(reconcile.tile_yield_changed_count),
        cleared_project_count=jnp.zeros_like(reconcile.cleared_project_count),
        cleared_unit_task_count=jnp.zeros_like(reconcile.cleared_unit_task_count),
        cleared_market_task_count=jnp.zeros_like(reconcile.cleared_market_task_count),
        cleared_unit_plan_count=jnp.zeros_like(reconcile.cleared_unit_plan_count),
    )
    reconcile = _select_lanes(episode_start, zero_reconcile, reconcile)
    return (
        ControllerScanCarryV2(
            environment_state=next_states,
            controller=selected,
        ),
        ControllerCarryDiagnosticsV2(
            episode_reset=episode_start,
            cross_episode_contamination=contamination,
            reconcile=reconcile,
        ),
    )


def scan_controller_state_sequence_v2(
    initial_carry: ControllerScanCarryV2,
    state_sequence: State,
    episode_start_sequence: jax.Array,
    player: int,
) -> tuple[ControllerScanCarryV2, ControllerCarryDiagnosticsV2]:
    """Run controller lifecycle through a time-major sequence with ``lax.scan``."""

    def body(carry, inputs):
        states, episode_start = inputs
        return advance_controller_scan_carry_v2(
            carry, states, episode_start, player
        )

    return jax.lax.scan(
        body,
        initial_carry,
        (state_sequence, episode_start_sequence),
    )


def poison_controller_for_reset_test_v2(
    controller: ProjectControllerStateV2,
) -> ProjectControllerStateV2:
    """Make every leaf non-default so reset tests cannot pass accidentally."""

    def poison(value):
        if jnp.issubdtype(value.dtype, jnp.bool_):
            return ~value
        return value + jnp.ones((), dtype=value.dtype)

    return jax.tree.map(poison, controller)


__all__ = [
    "advance_controller_scan_carry_v2",
    "initialize_controller_scan_carry_v2",
    "poison_controller_for_reset_test_v2",
    "scan_controller_state_sequence_v2",
]

