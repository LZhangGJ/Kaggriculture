from __future__ import annotations

import gzip
import json
from pathlib import Path

import jax
from jax import lax
import pytest

from kaggriculture_jax import (
    encode_actions,
    events_for_seed,
    load_event_bank,
    load_tables,
    reset,
    stack_actions,
    step_env,
)
from kaggriculture_jax.types import State

from reference_assertions import assert_state_matches_frame


PROJECT = Path(__file__).resolve().parents[1]
TRACE_DIR = PROJECT / "reference" / "traces"


def load_frames(name: str) -> tuple[dict, list[dict]]:
    with gzip.open(TRACE_DIR / name, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return rows[0], rows[1:]


def to_host_state(state: State) -> State:
    return jax.tree.map(lambda value: jax.device_get(value), state)


@pytest.mark.parametrize(
    ("filename", "seed"),
    [
        ("pass_pass_seed0.jsonl.gz", 0),
        ("starter_starter_seed1.jsonl.gz", 1),
        ("v16_v16_seed0.jsonl.gz", 0),
    ],
)
def test_full_official_trace(filename: str, seed: int) -> None:
    _, frames = load_frames(filename)
    seeds, event_bank = load_event_bank()
    events = events_for_seed(seed, seeds, event_bank)
    tables = load_tables()
    initial = reset(seed)
    assert_state_matches_frame(to_host_state(initial), frames[0])

    actions = stack_actions([encode_actions(frame["actions"]) for frame in frames[1:]])

    @jax.jit
    def rollout(state, action_sequence):
        def body(carry, action):
            next_state = step_env(carry, action, events, tables)
            return next_state, next_state

        return lax.scan(body, state, action_sequence)

    terminal, trajectory = rollout(initial, actions)
    jax.block_until_ready(terminal)
    trajectory = to_host_state(trajectory)
    for index, frame in enumerate(frames[1:]):
        state = State(*(field[index] for field in trajectory))
        assert_state_matches_frame(state, frame)
