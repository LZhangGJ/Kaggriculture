"""Replay deterministic random/invalid official actions through JAX exactly."""

from __future__ import annotations

import gzip
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp

from kaggriculture_jax import (
    Events,
    batched_step_sync,
    encode_actions,
    load_event_bank,
    load_tables,
    reset,
    stack_actions,
)
from kaggriculture_jax.types import State


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tests"))
from reference_assertions import assert_state_matches_frame  # noqa: E402


TRACE_DIR = PROJECT / "reference" / "random_differential"
RECEIPT = PROJECT / "receipts" / "random_differential_parity.json"


def load_frames(seed: int) -> list[dict]:
    with gzip.open(TRACE_DIR / f"random_seed{seed}.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return rows[1:]


def state_at(trajectory: State, frame_index: int, batch_index: int) -> State:
    return State(*(field[frame_index, batch_index] for field in trajectory))


def main() -> int:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected GPU backend, got {jax.default_backend()}")
    seed_values = list(range(32, 48))
    official = [load_frames(seed) for seed in seed_values]
    event_seeds, bank = load_event_bank()
    index_by_seed = {int(seed): index for index, seed in enumerate(event_seeds)}
    indices = jnp.asarray([index_by_seed[seed] for seed in seed_values])
    events = Events(bank.weed_spawn[indices], bank.shop_choice[indices])
    tables = load_tables()
    initial = jax.vmap(reset)(jnp.asarray(seed_values, dtype=jnp.int32))

    by_seed = [
        stack_actions([encode_actions(frame["actions"]) for frame in frames[1:]])
        for frames in official
    ]
    action_sequence = stack_actions(by_seed)
    action_sequence = jax.tree.map(
        lambda value: jnp.swapaxes(value, 0, 1), action_sequence
    )

    @jax.jit
    def rollout(states, actions):
        def body(carry, action):
            next_states = batched_step_sync(carry, action, events, tables)
            return next_states, next_states

        return jax.lax.scan(body, states, actions)

    started = time.perf_counter()
    terminal, trajectory = rollout(initial, action_sequence)
    jax.block_until_ready(terminal)
    compile_and_execute_s = time.perf_counter() - started
    host_initial, host_trajectory, terminal_host = jax.device_get(
        (initial, trajectory, terminal)
    )
    compare_started = time.perf_counter()
    for batch_index, frames in enumerate(official):
        try:
            assert_state_matches_frame(
                State(*(field[batch_index] for field in host_initial)), frames[0]
            )
            for trajectory_index, frame in enumerate(frames[1:]):
                assert_state_matches_frame(
                    state_at(host_trajectory, trajectory_index, batch_index), frame
                )
        except AssertionError as error:
            raise AssertionError(
                f"random differential seed={seed_values[batch_index]}: {error}"
            ) from error
    comparison_s = time.perf_counter() - compare_started
    receipt = {
        "schema": "kaggriculture_random_differential_parity_v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "official_package_version": "1.32.6",
        "jax": importlib.metadata.version("jax"),
        "backend": jax.default_backend(),
        "scenario": "state-aware random legal actions plus injected invalid actions",
        "seed_first": seed_values[0],
        "seed_last": seed_values[-1],
        "seed_count": len(seed_values),
        "frames_per_seed": 720,
        "frames_compared": len(seed_values) * 720,
        "exact_zero_error": True,
        "compile_and_execute_s": compile_and_execute_s,
        "comparison_s": comparison_s,
        "terminal_steps": sorted(set(int(value) for value in terminal_host.step.tolist())),
        "max_hand_cap_hits": int(jnp.max(terminal_host.hand_cap_hits)),
        "market_loop_cap_hits": int(jnp.sum(terminal_host.market_loop_cap_hits)),
        "price_lut_oob": int(jnp.sum(terminal_host.price_lut_oob)),
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
