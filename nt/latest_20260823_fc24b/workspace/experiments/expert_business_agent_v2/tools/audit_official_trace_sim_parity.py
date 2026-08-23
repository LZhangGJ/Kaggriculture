#!/usr/bin/env python3
"""Replay an official 1.32.7 action tape through JAX and report first state drift."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "gpu_sim/src",
    ROOT / "gpu_sim/tests",
):
    sys.path.insert(0, str(path))

from kaggriculture_jax import encode_actions, stack_actions  # noqa: E402
from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
from reference_assertions import assert_state_matches_frame  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True, nargs="+")
    args = parser.parse_args()
    all_frames = []
    seeds = []
    all_actions = []
    for trace in args.trace:
        with gzip.open(trace.resolve(), "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        frames = [row for row in rows if row.get("record_type") == "frame"]
        header = next(row for row in rows if row.get("record_type") == "header")
        seeds.append(int(header["seed"]))
        all_frames.append(frames)
        all_actions.append(
            stack_actions([encode_actions(frame["actions"]) for frame in frames[1:]])
        )
    tape = Action(
        *(
            jnp.stack([getattr(actions, field) for actions in all_actions], axis=1)
            for field in Action._fields
        )
    )
    weed, shops = build_events_v1(seeds)
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))
    tables = load_tables()

    @jax.jit
    def rollout(state):
        def body(current, action):
            nxt = batched_step_sync(current, action, events, tables)
            return nxt, nxt

        return jax.lax.scan(body, state, tape)

    terminal, trajectory = rollout(initial)
    terminal, trajectory = jax.device_get((terminal, trajectory))
    first = None
    for batch, frames in enumerate(all_frames):
        for step, frame in enumerate(frames[1:]):
            state = type(terminal)(*(field[step, batch] for field in trajectory))
            try:
                assert_state_matches_frame(state, frame)
            except AssertionError as error:
                first = {"batch": batch, "seed": seeds[batch], "step": step, "message": str(error)}
                break
        if first is not None:
            break
    result = {
        "status": "PASS" if first is None else "FAIL",
        "seeds": seeds,
        "backend": jax.default_backend(),
        "first_mismatch": first,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if first is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
