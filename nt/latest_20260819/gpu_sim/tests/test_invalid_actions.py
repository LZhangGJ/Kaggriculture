from __future__ import annotations

import jax
import numpy as np

from kaggriculture_jax import (
    empty_action,
    encode_actions,
    events_for_seed,
    load_event_bank,
    load_tables,
    reset,
    step_env,
)


def _assert_tree_equal(left, right) -> None:
    for actual, expected in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(jax.device_get(actual), jax.device_get(expected))


def test_malformed_and_illegal_actions_are_silent_noops() -> None:
    seeds, bank = load_event_bank()
    events = events_for_seed(0, seeds, bank)
    tables = load_tables()
    state = reset(0)
    baseline = jax.jit(step_env)(state, empty_action(), events, tables)
    invalid = encode_actions(
        [
            {
                "farmer": ["TELEPORT", 999],
                "hands": "not-a-list",
                "market": [
                    ["BUY_SEED", "WHEAT", 0],
                    ["SELL", "NOT_AN_ITEM", 3],
                    None,
                ],
            },
            {
                "farmer": ["PLANT", "MELON"],  # no seed
                "hands": [],
                "market": [["BUY_ANIMAL", "DRAGON", 1]],
            },
        ]
    )
    actual = jax.jit(step_env)(state, invalid, events, tables)
    _assert_tree_equal(actual, baseline)


def test_out_of_bounds_move_is_noop() -> None:
    seeds, bank = load_event_bank()
    events = events_for_seed(0, seeds, bank)
    tables = load_tables()
    state = reset(0)
    west = encode_actions(
        [
            {"farmer": ["WEST"], "hands": [], "market": []},
            {"farmer": ["PASS"], "hands": [], "market": []},
        ]
    )
    for _ in range(4):
        state = step_env(state, west, events, tables)
    assert int(state.unit_pos[0, 0, 0]) == 0
    invalid_result = step_env(state, west, events, tables)
    pass_result = step_env(state, empty_action(), events, tables)
    _assert_tree_equal(invalid_result, pass_result)

