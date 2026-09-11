from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture_jax import load_event_bank, load_tables
from kaggriculture_jax.constants import MAX_UNITS
from strategic_v5 import (
    TaskStatusV1,
    cleanup_full_controller_day_end_v1,
    full_rule_only_step_v1,
    initialize_full_rule_carry_v1,
    make_full_rule_only_arena_rollout_v1,
    reset_controller_state_v1,
)


TABLES = load_tables()


def _events(batch_size: int):
    _, bank = load_event_bank()
    return jax.tree.map(lambda value: value[:batch_size], bank)


def _controller_batch(batch_size: int):
    one = reset_controller_state_v1()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape), one
    )


def test_day_end_cleanup_clears_only_vanished_hands() -> None:
    controller = _controller_batch(2)
    status = controller.unit_tasks.status.at[:, :3].set(TaskStatusV1.ACTIVE)
    controller = controller._replace(
        unit_tasks=controller.unit_tasks._replace(status=status)
    )
    active = jnp.zeros((2, MAX_UNITS), dtype=jnp.bool_).at[:, 0].set(True)
    day_end = jnp.asarray([True, False])
    cleaned = jax.jit(cleanup_full_controller_day_end_v1)(
        controller, active, day_end
    )
    assert int(cleaned.unit_tasks.status[0, 0]) == TaskStatusV1.ACTIVE
    assert bool(jnp.all(cleaned.unit_tasks.status[0, 1:] == TaskStatusV1.EMPTY))
    assert int(cleaned.unit_tasks.status[1, 1]) == TaskStatusV1.ACTIVE


def test_full_econ_rule_step_is_jittable_and_hard_safe() -> None:
    seeds = jnp.asarray([1201, 1202], dtype=jnp.int32)
    carry = initialize_full_rule_carry_v1(seeds, _events(2))
    step = jax.jit(lambda value: full_rule_only_step_v1(value, TABLES)[0])
    following = step(carry)
    jax.block_until_ready(following.environment_state.money)
    assert following.environment_state.step.tolist() == [1, 1]
    diagnostics = following.diagnostics
    hard_fields = (
        diagnostics.invalid_raw_action_count,
        diagnostics.internal_resource_conflict_count,
        diagnostics.unexpected_silent_noop_count,
        diagnostics.effect_mismatch_count,
        diagnostics.owner_inactive_count,
        diagnostics.deadline_missed_count,
        diagnostics.resource_unavailable_count,
        diagnostics.nan_or_inf_count,
    )
    assert all(int(value.sum()) == 0 for value in hard_fields)


def test_full_econ_rule_only_completes_a_full_season() -> None:
    seeds = jnp.asarray([1211, 1212, 1213, 1214], dtype=jnp.int32)
    carry = initialize_full_rule_carry_v1(seeds, _events(4))
    rollout = jax.jit(make_full_rule_only_arena_rollout_v1())
    result = rollout(carry, TABLES)
    jax.block_until_ready(result.final_carry.environment_state.money)
    states = result.final_carry.environment_state
    assert states.step.tolist() == [719, 719, 719, 719]
    assert bool(jnp.all(states.done))
    diagnostics = result.final_carry.diagnostics
    hard_fields = (
        diagnostics.invalid_raw_action_count,
        diagnostics.internal_resource_conflict_count,
        diagnostics.unexpected_silent_noop_count,
        diagnostics.effect_mismatch_count,
        diagnostics.owner_inactive_count,
        diagnostics.deadline_missed_count,
        diagnostics.resource_unavailable_count,
        diagnostics.cross_episode_task_contamination_count,
        diagnostics.nan_or_inf_count,
    )
    assert all(int(value.sum()) == 0 for value in hard_fields)
