#!/usr/bin/env python3
"""Seat-swapped official stepwise acceptance for the JAX PRT V6 port."""

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
from kaggriculture_jax.types import Events, State  # noqa: E402
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


SEEDS = (157001, 157002, 157003, 157004)
BANK = ROOT / "experiments/expert_business_agent_v2/artifacts/jax_prt6_exact_port_probe_bank_v1.npz"
BANK_RECEIPT = ROOT / "experiments/expert_business_agent_v2/receipts/jax_prt6_exact_port_probe_bank_v1.json"
TRACE_RECEIPT = ROOT / "experiments/expert_business_agent_v2/receipts/prt6_official_parity_traces_v1.json"
RECEIPT = ROOT / "experiments/expert_business_agent_v2/receipts/prt6_jax_stepwise_parity_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_frames(seed: int, candidate_seat: int) -> list[dict]:
    path = ROOT / (
        "experiments/expert_business_agent_v2/official_eval/prt6_stepwise_parity/"
        f"route37_prt6_candidate_seat{candidate_seat}_seed{seed}.jsonl.gz"
    )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    header, frames = records[0], records[1:]
    if (
        header["seed"] != seed
        or header["candidate_seat"] != candidate_seat
        or len(frames) != 720
    ):
        raise RuntimeError(f"malformed official trace {path}")
    return frames


def state_at(states: State, step: int, batch: int) -> State:
    return State(*(field[step, batch] for field in states))


def make_rollout(bank, boatlee_trace, tables, router, candidate_seat: int):
    opponent_seat = 1 - candidate_seat

    @jax.jit
    def rollout(initial, events, candidate_ids, opponent_ids):
        batch_size = initial.step.shape[0]

        def body(value, _):
            states, candidate_carry, opponent_carry = value
            candidate_action, candidate_carry = skeleton_player_action_v1(
                states,
                tables,
                bank,
                candidate_ids,
                candidate_carry,
                candidate_seat,
            )
            opponent_action, opponent_carry = full_opponent_action(
                states,
                tables,
                bank,
                boatlee_trace,
                opponent_ids,
                opponent_carry,
                opponent_seat,
                router,
            )
            actions = (
                pair(candidate_action, opponent_action)
                if candidate_seat == 0
                else pair(opponent_action, candidate_action)
            )
            next_states = batched_step_sync(states, actions, events, tables)
            return (next_states, candidate_carry, opponent_carry), (actions, next_states)

        return jax.lax.scan(
            body,
            (
                initial,
                initialize_trace_player_carry_v1(batch_size),
                initialize_full_opponent_carry(batch_size, opponent_ids, router),
            ),
            xs=None,
            length=719,
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
    if trace_receipt["official_package_version"] != "1.32.7":
        raise RuntimeError("official trace package is not 1.32.7")
    bank_receipt = json.loads(BANK_RECEIPT.read_text(encoding="utf-8"))
    opponent_by_name = {row["name"]: row for row in bank_receipt["opponents"]}
    prt6_id = int(opponent_by_name["local_prt_v6"]["opponent_id"])
    route37_id = int(bank_receipt["candidate_ids"][0])

    bank = load_bank(BANK)
    router = build_router_arrays(bank_receipt)
    boatlee_trace = load_boatlee_trace_v1()
    tables = load_tables()
    weed, shops = build_events_v1(list(SEEDS))
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(SEEDS, dtype=jnp.int32))
    candidate_ids = jnp.full((len(SEEDS),), route37_id, dtype=jnp.int32)
    opponent_ids = jnp.full((len(SEEDS),), prt6_id, dtype=jnp.int32)

    action_exact = True
    state_exact = True
    first_action_mismatch = None
    first_state_mismatch = None
    terminal_rewards_exact = True
    all_done = True
    diagnostics = {"hand_cap_hits": 0, "market_loop_cap_hits": 0, "price_lut_oob": 0}
    timing = []

    for candidate_seat in (0, 1):
        official = [load_frames(seed, candidate_seat) for seed in SEEDS]
        rollout = make_rollout(bank, boatlee_trace, tables, router, candidate_seat)
        run_started = time.perf_counter()
        (result, (actions, trajectory)) = rollout(
            initial, events, candidate_ids, opponent_ids
        )
        terminal = result[0]
        jax.block_until_ready(terminal.money)
        timing.append({
            "candidate_seat": candidate_seat,
            "games": len(SEEDS),
            "seconds": time.perf_counter() - run_started,
        })
        host_actions, host_trajectory, host_terminal, host_initial = jax.device_get(
            (actions, trajectory, terminal, initial)
        )

        for batch, frames in enumerate(official):
            expected = stack_actions(
                [encode_actions(frame["actions"]) for frame in frames[1:]]
            )
            for field_name in actions._fields:
                actual_field = np.asarray(getattr(host_actions, field_name)[:, batch])
                expected_field = np.asarray(getattr(expected, field_name))
                different = np.argwhere(actual_field != expected_field)
                if different.size:
                    action_exact = False
                    if first_action_mismatch is None:
                        first_action_mismatch = {
                            "candidate_seat": candidate_seat,
                            "seed": SEEDS[batch],
                            "field": field_name,
                            "index": different[0].astype(int).tolist(),
                            "actual": int(actual_field[tuple(different[0])]),
                            "expected": int(expected_field[tuple(different[0])]),
                        }

            try:
                assert_state_matches_frame(
                    State(*(field[batch] for field in host_initial)), frames[0]
                )
                for step, frame in enumerate(frames[1:]):
                    assert_state_matches_frame(
                        state_at(host_trajectory, step, batch), frame
                    )
            except AssertionError as exc:
                state_exact = False
                if first_state_mismatch is None:
                    first_state_mismatch = {
                        "candidate_seat": candidate_seat,
                        "seed": SEEDS[batch],
                        "message": str(exc)[:2000],
                    }

            terminal_rewards_exact &= np.array_equal(
                host_terminal.reward[batch], np.asarray(frames[-1]["reward"])
            )

        all_done &= bool(np.all(host_terminal.done) and np.all(host_terminal.step == 719))
        for name in diagnostics:
            diagnostics[name] += int(np.sum(getattr(host_terminal, name)))

    gates = {
        "official_trace_package_1_32_7": True,
        "all_11504_actions_exact": action_exact,
        "all_5760_state_frames_exact": state_exact,
        "terminal_rewards_exact": bool(terminal_rewards_exact),
        "all_done_at_719": all_done,
        "hard_errors_zero": sum(diagnostics.values()) == 0,
    }
    result = {
        "schema": "kaggriculture_prt6_jax_stepwise_parity_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "backend": jax.default_backend(),
        "jax": importlib.metadata.version("jax"),
        "official_package_version": "1.32.7",
        "seeds": list(SEEDS),
        "seat_swapped": True,
        "candidate_route": "Route37_8C4S",
        "opponent": "local_prt_v6",
        "action_comparisons": 719 * 2 * len(SEEDS) * 2,
        "state_frames": 720 * len(SEEDS) * 2,
        "gates": gates,
        "first_action_mismatch": first_action_mismatch,
        "first_state_mismatch": first_state_mismatch,
        "diagnostics": diagnostics,
        "timing": timing,
        "bank_sha256": sha256(BANK),
        "bank_receipt_sha256": sha256(BANK_RECEIPT),
        "trace_receipt_sha256": sha256(TRACE_RECEIPT),
        "wall_seconds": time.perf_counter() - started,
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
