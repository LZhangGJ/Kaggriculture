#!/usr/bin/env python3
"""Stepwise parity audit for the six weak Route37 opponents in the broad bank."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools"))
sys.path.insert(0, str(ROOT / "gpu_sim/tests"))

from kaggriculture_jax import encode_actions, stack_actions  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events, State  # noqa: E402
from reference_assertions import assert_state_matches_frame  # noqa: E402
from run_jax_dynamic_counterfactual_panel import (  # noqa: E402
    batched_step_sync,
    build_events_v1,
    build_router_arrays,
    full_opponent_action,
    initialize_full_opponent_carry,
    initialize_trace_player_carry_v1,
    load_bank,
    load_boatlee_trace_v1,
    pair,
    skeleton_player_action_v1,
)


BANK = ROOT / "experiments/expert_business_agent_v2/artifacts/jax_route46_dynamic_broad23_bank_v4.npz"
BANK_RECEIPT = ROOT / "experiments/expert_business_agent_v2/receipts/jax_route46_dynamic_broad23_bank_v4.json"
TRACE_RECEIPT = ROOT / "experiments/expert_business_agent_v2/receipts/weak6_official_parity_traces_v1.json"
RECEIPT = ROOT / "experiments/expert_business_agent_v2/receipts/weak6_jax_stepwise_parity_v3.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_frames(path_text: str) -> list[dict]:
    path = ROOT / path_text
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    if len(records) != 721:
        raise RuntimeError(f"malformed trace {path}")
    return records[1:]


def state_at(states: State, step: int, batch: int) -> State:
    return State(*(field[step, batch] for field in states))


def make_rollout(bank, boatlee_trace, tables, router, candidate_seat: int):
    opponent_seat = 1 - candidate_seat

    @jax.jit
    def rollout(initial, events, candidate_tape, opponent_ids):
        batch_size = initial.step.shape[0]

        def body(value, candidate_action):
            states, opponent_carry = value
            opponent_action, opponent_carry = full_opponent_action(
                states, tables, bank, boatlee_trace, opponent_ids,
                opponent_carry, opponent_seat, router,
            )
            actions = pair(candidate_action, opponent_action) if candidate_seat == 0 else pair(opponent_action, candidate_action)
            next_states = batched_step_sync(states, actions, events, tables)
            return (next_states, opponent_carry), (actions, next_states)

        return jax.lax.scan(
            body,
            (
                initial,
                initialize_full_opponent_carry(batch_size, opponent_ids, router),
            ),
            xs=candidate_tape,
        )

    return rollout


def main() -> int:
    started = time.perf_counter()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    cache = Path.home() / ".cache/kaggriculture_jax"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    trace_receipt = json.loads(TRACE_RECEIPT.read_text(encoding="utf-8"))
    bank_receipt = json.loads(BANK_RECEIPT.read_text(encoding="utf-8"))
    if trace_receipt["official_package_version"] != "1.32.7":
        raise RuntimeError("official trace package is not 1.32.7")
    route37_position = bank_receipt["candidate_names"].index("route37_rank16_8c_4s_75l")
    route37_id = int(bank_receipt["candidate_ids"][route37_position])
    source_to_id = {
        str(Path(row["source"]).resolve()).lower(): int(row["opponent_id"])
        for row in bank_receipt["opponents"]
    }
    opponent_ids_by_name = {}
    for row in trace_receipt["opponents"]:
        key = str(Path(row["path"]).resolve()).lower()
        if key not in source_to_id:
            raise KeyError(f"weak6 source missing from JAX bank: {row['name']} {row['path']}")
        opponent_ids_by_name[row["name"]] = source_to_id[key]

    bank = load_bank(BANK)
    router = build_router_arrays(bank_receipt)
    boatlee_trace = load_boatlee_trace_v1()
    tables = load_tables()
    per_opponent = {
        name: {
            "contexts": 0,
            "action_exact_contexts": 0,
            "state_exact_contexts": 0,
            "terminal_reward_exact_contexts": 0,
            "first_action_mismatch": None,
            "first_state_mismatch": None,
        }
        for name in opponent_ids_by_name
    }
    timing = []

    for candidate_seat in (0, 1):
        traces = [
            row for row in trace_receipt["traces"]
            if int(row["candidate_seat"]) == candidate_seat
        ]
        names = [str(row["opponent"]) for row in traces]
        seeds = [int(row["seed"]) for row in traces]
        official = [load_frames(str(row["path"])) for row in traces]
        weed, shops = build_events_v1(seeds)
        events = Events(jnp.asarray(weed), jnp.asarray(shops))
        initial = jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))
        expected_batches = [
            stack_actions([encode_actions(frame["actions"]) for frame in frames[1:]])
            for frames in official
        ]
        candidate_tape = Action(
            *(
                jnp.stack(
                    [getattr(expected, field_name)[:, candidate_seat] for expected in expected_batches],
                    axis=1,
                )
                for field_name in Action._fields
            )
        )
        opponent_ids = jnp.asarray([opponent_ids_by_name[name] for name in names], dtype=jnp.int32)
        rollout = make_rollout(bank, boatlee_trace, tables, router, candidate_seat)
        run_started = time.perf_counter()
        result, scanned = rollout(initial, events, candidate_tape, opponent_ids)
        terminal = result[0]
        actions, trajectory = scanned
        jax.block_until_ready(terminal.money)
        timing.append({"candidate_seat": candidate_seat, "games": len(traces), "seconds": time.perf_counter() - run_started})
        host_actions, host_trajectory, host_terminal, host_initial = jax.device_get((actions, trajectory, terminal, initial))

        for batch, frames in enumerate(official):
            name = names[batch]
            seed = seeds[batch]
            target = per_opponent[name]
            target["contexts"] += 1
            expected = expected_batches[batch]
            action_exact = True
            for field_name in actions._fields:
                actual_field = np.asarray(getattr(host_actions, field_name)[:, batch])
                expected_field = np.asarray(getattr(expected, field_name))
                different = np.argwhere(actual_field != expected_field)
                if different.size:
                    action_exact = False
                    if target["first_action_mismatch"] is None:
                        row = different[0]
                        target["first_action_mismatch"] = {
                            "candidate_seat": candidate_seat,
                            "seed": seed,
                            "field": field_name,
                            "index": row.astype(int).tolist(),
                            "actual": int(actual_field[tuple(row)]),
                            "expected": int(expected_field[tuple(row)]),
                        }
            target["action_exact_contexts"] += int(action_exact)

            state_exact = True
            try:
                assert_state_matches_frame(State(*(field[batch] for field in host_initial)), frames[0])
                for step, frame in enumerate(frames[1:]):
                    assert_state_matches_frame(state_at(host_trajectory, step, batch), frame)
            except AssertionError as exc:
                state_exact = False
                if target["first_state_mismatch"] is None:
                    target["first_state_mismatch"] = {
                        "candidate_seat": candidate_seat,
                        "seed": seed,
                        "message": str(exc)[:2000],
                    }
            target["state_exact_contexts"] += int(state_exact)
            reward_exact = np.array_equal(host_terminal.reward[batch], np.asarray(frames[-1]["reward"]))
            target["terminal_reward_exact_contexts"] += int(reward_exact)

    exact_names = []
    for name, row in per_opponent.items():
        row["action_exact_rate"] = row["action_exact_contexts"] / row["contexts"]
        row["state_exact_rate"] = row["state_exact_contexts"] / row["contexts"]
        row["terminal_reward_exact_rate"] = row["terminal_reward_exact_contexts"] / row["contexts"]
        row["strict_exact"] = (
            row["action_exact_contexts"] == row["contexts"]
            and row["state_exact_contexts"] == row["contexts"]
            and row["terminal_reward_exact_contexts"] == row["contexts"]
        )
        if row["strict_exact"]:
            exact_names.append(name)
    output = {
        "schema": "kaggriculture_weak6_jax_stepwise_parity_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AUDITED",
        "backend": jax.default_backend(),
        "jax": importlib.metadata.version("jax"),
        "official_package_version": "1.32.7",
        "seeds": trace_receipt["seeds"],
        "seat_swapped": True,
        "opponent_count": len(per_opponent),
        "strict_exact_count": len(exact_names),
        "strict_exact_names": exact_names,
        "per_opponent": per_opponent,
        "timing": timing,
        "bank_sha256": sha256(BANK),
        "bank_receipt_sha256": sha256(BANK_RECEIPT),
        "trace_receipt_sha256": sha256(TRACE_RECEIPT),
        "wall_seconds": time.perf_counter() - started,
        "acceptance_boundary": "Only strict_exact=true permits replacement of the official Python opponent.",
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
