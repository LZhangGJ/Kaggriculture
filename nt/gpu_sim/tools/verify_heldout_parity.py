"""Verify all 100 unseen-seed official frames against one batched GPU rollout."""

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
    empty_action,
    load_event_bank,
    load_tables,
    reset,
)
from kaggriculture_jax.types import State


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tests"))
from reference_assertions import assert_state_matches_frame  # noqa: E402


HELDOUT_DIR = PROJECT / "reference" / "heldout"
RECEIPT = PROJECT / "receipts" / "heldout_parity.json"


def load_frames(seed: int) -> list[dict]:
    path = HELDOUT_DIR / f"pass_pass_seed{seed}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if rows[0]["configuration"]["seed"] != seed or len(rows) != 721:
        raise RuntimeError(f"Malformed heldout trace: {path}")
    return rows[1:]


def state_at(trajectory: State, frame_index: int, batch_index: int) -> State:
    return State(*(field[frame_index, batch_index] for field in trajectory))


def main() -> int:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected GPU backend, got {jax.default_backend()}")
    seed_values = list(range(10000, 10100))
    official = [load_frames(seed) for seed in seed_values]
    event_seeds, bank = load_event_bank()
    index_by_seed = {int(seed): index for index, seed in enumerate(event_seeds)}
    indices = jnp.asarray([index_by_seed[seed] for seed in seed_values])
    events = Events(bank.weed_spawn[indices], bank.shop_choice[indices])
    tables = load_tables()
    initial = jax.vmap(reset)(jnp.asarray(seed_values, dtype=jnp.int32))
    action = empty_action()
    actions = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (len(seed_values), *value.shape)),
        action,
    )

    @jax.jit
    def rollout(states):
        def body(carry, _):
            next_states = batched_step_sync(carry, actions, events, tables)
            return next_states, next_states

        return jax.lax.scan(body, states, xs=None, length=719)

    started = time.perf_counter()
    terminal, trajectory = rollout(initial)
    jax.block_until_ready(terminal)
    compile_and_execute_s = time.perf_counter() - started
    transfer_started = time.perf_counter()
    host_initial, host_trajectory = jax.device_get((initial, trajectory))
    transfer_s = time.perf_counter() - transfer_started

    compare_started = time.perf_counter()
    for batch_index, frames in enumerate(official):
        initial_state = State(*(field[batch_index] for field in host_initial))
        assert_state_matches_frame(initial_state, frames[0])
        for trajectory_index, frame in enumerate(frames[1:]):
            assert_state_matches_frame(
                state_at(host_trajectory, trajectory_index, batch_index), frame
            )
    comparison_s = time.perf_counter() - compare_started
    terminal_host = jax.device_get(terminal)
    receipt = {
        "schema": "kaggriculture_heldout_parity_v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "official_package": "kaggle-environments",
        "official_package_version": "1.32.6",
        "jax": importlib.metadata.version("jax"),
        "backend": jax.default_backend(),
        "seed_first": seed_values[0],
        "seed_last": seed_values[-1],
        "seed_count": len(seed_values),
        "frames_per_seed": 720,
        "frames_compared": len(seed_values) * 720,
        "state_scope": (
            "step/day/hour/status/reward; all farm/tile/unit/private inventory; "
            "market inventory/prices; town shops; diagnostic counters"
        ),
        "exact_integer_price_reward_zero_error": True,
        "compile_and_execute_s": compile_and_execute_s,
        "device_to_host_transfer_s": transfer_s,
        "comparison_s": comparison_s,
        "terminal_steps": sorted(set(int(value) for value in terminal_host.step.tolist())),
        "diagnostics": {
            "hand_cap_hits": int(jnp.sum(terminal_host.hand_cap_hits)),
            "market_loop_cap_hits": int(jnp.sum(terminal_host.market_loop_cap_hits)),
            "price_lut_oob": int(jnp.sum(terminal_host.price_lut_oob)),
        },
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
