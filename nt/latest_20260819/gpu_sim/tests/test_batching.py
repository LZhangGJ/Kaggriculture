from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import (
    Events,
    batched_project_unit_phase,
    batched_step_from_projected_unit_phase_sync,
    batched_step_sync,
    empty_action,
    encode_actions,
    load_event_bank,
    load_tables,
    reset,
    step_env,
)


def _broadcast_action(batch_size: int):
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size, *value.shape)),
        empty_action(),
    )


def _assert_tree_equal(left, right) -> None:
    for actual, expected in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(jax.device_get(actual), jax.device_get(expected))


def test_synchronized_fast_batch_matches_independent_rollouts() -> None:
    seeds, bank = load_event_bank()
    tables = load_tables()
    batch_size = 8
    seed_values = jnp.asarray(seeds[:batch_size], dtype=jnp.int32)
    initial_states = jax.vmap(reset)(seed_values)
    states_sync = initial_states
    actions = _broadcast_action(batch_size)
    events = Events(bank.weed_spawn[:batch_size], bank.shop_choice[:batch_size])
    fast = jax.jit(batched_step_sync)
    for _ in range(50):
        states_sync = fast(states_sync, actions, events, tables)

    @jax.jit
    def independent_rollout(state, event):
        def body(_, carry):
            return step_env(carry, empty_action(), event, tables)

        return jax.lax.fori_loop(0, 50, body, state)

    independent = [
        independent_rollout(
            jax.tree.map(lambda value: value[index], initial_states),
            Events(events.weed_spawn[index], events.shop_choice[index]),
        )
        for index in range(batch_size)
    ]
    expected = jax.tree.map(lambda *values: jnp.stack(values), *independent)
    _assert_tree_equal(states_sync, expected)


def test_batch_members_do_not_crosstalk() -> None:
    seeds, bank = load_event_bank()
    tables = load_tables()
    actions = empty_action()
    independent = []
    event_rows = []
    for index in range(4):
        state = reset(int(seeds[index]))
        events = Events(bank.weed_spawn[index], bank.shop_choice[index])
        independent.append(state)
        event_rows.append(events)
    states = jax.tree.map(lambda *values: jnp.stack(values), *independent)
    batch_actions = _broadcast_action(len(independent))
    batch_events = jax.tree.map(lambda *values: jnp.stack(values), *event_rows)
    batch_result = jax.jit(batched_step_sync)(
        states, batch_actions, batch_events, tables
    )
    expected = jax.tree.map(
        lambda *values: jnp.stack(values),
        *[
            step_env(state, actions, events, tables)
            for state, events in zip(independent, event_rows, strict=True)
        ],
    )
    _assert_tree_equal(batch_result, expected)


def test_projected_unit_continuation_exactly_matches_full_sync_step() -> None:
    """The reusable unit projection must be the exact full-step prefix."""

    seeds, bank = load_event_bank()
    tables = load_tables()
    batch_size = 5
    states = jax.vmap(reset)(jnp.asarray(seeds[:batch_size], dtype=jnp.int32))
    one_action = encode_actions(
        [
            {
                "farmer": ["SOUTH"],
                "hands": [],
                "market": [
                    ["BUY_SEED", "WHEAT", 2],
                    ["HIRE", 1],
                ],
            },
            {
                "farmer": ["EAST"],
                "hands": [],
                "market": [
                    ["BUY_SEED", "CARROT", 1],
                    ["BUY_LAND", 1],
                ],
            },
        ]
    )
    actions = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape),
        one_action,
    )
    events = Events(bank.weed_spawn[:batch_size], bank.shop_choice[:batch_size])
    direct_step = jax.jit(batched_step_sync)
    project = jax.jit(batched_project_unit_phase)
    continue_step = jax.jit(batched_step_from_projected_unit_phase_sync)

    # Cross an end-of-day boundary; repeated invalid actions must also remain
    # exact silent no-ops on both paths.
    for _ in range(26):
        direct = direct_step(states, actions, events, tables)
        projected = project(states, actions)
        continued = continue_step(projected, actions, events, tables)
        _assert_tree_equal(continued, direct)
        states = direct
