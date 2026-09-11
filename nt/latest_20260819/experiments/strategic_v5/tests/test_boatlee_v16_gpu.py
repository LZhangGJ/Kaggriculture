from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import (
    encode_actions,
    events_for_seed,
    load_event_bank,
    load_tables,
    reset,
    stack_actions,
)
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.types import State
from strategic_v5.boatlee_v16_gpu import (
    boatlee_actions_v1,
    initialize_boatlee_carry_v1,
    load_boatlee_trace_v1,
)
from strategic_v5.boatlee_arena import (
    boatlee_learned_step_v1,
    initialize_boatlee_arena_carry_v1,
)
from strategic_v5.constants import MAX_SELECTIONS_V1
from strategic_v5.learned_v1 import initialize_full_learned_params_v1


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "gpu_sim/tests"))
from reference_assertions import assert_state_matches_frame  # noqa: E402


def _frames() -> list[dict]:
    path = REPO / "gpu_sim/reference/traces/v16_v16_seed0.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert rows[0]["v16_main_sha256"] == (
        "3c9b6e75d1bb9cc1f23b6bf5d8821c84193d1306d5bcb74ada1628359e3fb025"
    )
    return rows[1:]


def test_boatlee_v16_seed0_actions_and_states_match_official_every_step() -> None:
    frames = _frames()
    seed_values, bank = load_event_bank()
    events = events_for_seed(0, seed_values, bank)
    events = jax.tree.map(lambda value: value[None], events)
    tables = load_tables()
    trace = load_boatlee_trace_v1()
    initial = jax.tree.map(lambda value: value[None], reset(0))
    carry = initialize_boatlee_carry_v1(1)

    @jax.jit
    def rollout(state, opponent_carry):
        def body(one_carry, _):
            one_state, one_opponent = one_carry
            action, one_opponent = boatlee_actions_v1(
                one_state, tables, trace, one_opponent
            )
            next_state = batched_step_sync(one_state, action, events, tables)
            return (next_state, one_opponent), (action, next_state)

        return jax.lax.scan(body, (state, opponent_carry), xs=None, length=719)

    (_, _), (actions, trajectory) = rollout(initial, carry)
    actions, trajectory = jax.device_get((actions, trajectory))
    official_actions = stack_actions(
        [encode_actions(frame["actions"]) for frame in frames[1:]]
    )
    for name in actions._fields:
        np.testing.assert_array_equal(
            getattr(actions, name)[:, 0], np.asarray(getattr(official_actions, name))
        )

    assert_state_matches_frame(
        State(*(field[0] for field in jax.device_get(initial))), frames[0]
    )
    for index, frame in enumerate(frames[1:]):
        assert_state_matches_frame(
            State(*(field[index, 0] for field in trajectory)), frame
        )


def test_seat_balanced_learned_boatlee_step_has_one_training_seat() -> None:
    seed_values, bank = load_event_bank()
    by_seed = {int(seed): index for index, seed in enumerate(seed_values)}
    indices = jnp.asarray([by_seed[0], by_seed[1]], dtype=jnp.int32)
    events = jax.tree.map(lambda value: value[indices], bank)
    carry = initialize_boatlee_arena_carry_v1(
        jnp.asarray([0, 1]), events, jnp.asarray([0, 1]), jax.random.key(11)
    )
    params = initialize_full_learned_params_v1(jax.random.key(12))
    following, transition = jax.jit(boatlee_learned_step_v1, static_argnames=("deterministic", "decision_interval"))(
        carry,
        load_tables(),
        load_boatlee_trace_v1(),
        params,
        deterministic=False,
        decision_interval=8,
    )
    assert following.environment_state.step.tolist() == [1, 1]
    assert transition.old_logprob.shape == (2, 1)
    assert transition.task_masks.shape[:3] == (2, 1, MAX_SELECTIONS_V1)
