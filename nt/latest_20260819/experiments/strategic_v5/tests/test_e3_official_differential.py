from __future__ import annotations

import gzip
import json
from pathlib import Path

import jax
from jax import lax

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
TRACE = PROJECT / "rules" / "official_traces" / "e3_all_animals_seed92.jsonl.gz"


def test_all_animals_official_trace_matches_every_jax_frame() -> None:
    with gzip.open(TRACE, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    header, frames = rows[0], rows[1:]
    assert header["package_version"] == "1.32.7"
    assert len(frames) == 720
    seed = int(header["seed"])
    seeds, event_bank = load_event_bank()
    events = events_for_seed(seed, seeds, event_bank)
    tables = load_tables()
    initial = reset(seed)
    assert_state_matches_frame(jax.device_get(initial), frames[0])
    actions = stack_actions([encode_actions(frame["actions"]) for frame in frames[1:]])

    @jax.jit
    def rollout(state, action_sequence):
        def body(carry, action):
            following = step_env(carry, action, events, tables)
            return following, following

        return lax.scan(body, state, action_sequence)

    terminal, trajectory = rollout(initial, actions)
    jax.block_until_ready(terminal)
    trajectory = jax.device_get(trajectory)
    for index, frame in enumerate(frames[1:]):
        state = State(*(field[index] for field in trajectory))
        assert_state_matches_frame(state, frame)
