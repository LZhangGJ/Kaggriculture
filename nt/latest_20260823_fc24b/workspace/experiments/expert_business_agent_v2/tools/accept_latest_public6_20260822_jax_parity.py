#!/usr/bin/env python3
"""Strict latest-public-six JAX parity against official Python 1.32.7 traces."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
for path in (
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "gpu_sim/src",
    ROOT / "gpu_sim/tests",
):
    sys.path.insert(0, str(path))

from kaggriculture_jax import encode_actions, stack_actions  # noqa: E402
from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events, State  # noqa: E402
from reference_assertions import assert_state_matches_frame  # noqa: E402
from accept_public_g02_jax_stepwise_parity import load_frames, state_at  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.latest_public6_20260822_gpu import (  # noqa: E402
    boatlee_v21_player_action_v1,
    initialize_boatlee_v21_carry_v1,
    initialize_moon_v92_carry_v1,
    initialize_soil_v26h_carry_v1,
    initialize_kaito_v39_carry_v1,
    initialize_latest_e284_carry_v1,
    prvsiyan_moon_v92_player_action_v1,
    prvsiyan_soil_v26h_player_action_v1,
    kaito_v39_history_gate_player_action_v1,
    salem_harvestforge_x_player_action_v1,
    steven_e284_hadouken_player_action_v1,
)
from strategic_v5.latest_public8_gpu import initialize_x562_carry_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import initialize_high_potential_v20_carry_v1  # noqa: E402


BANK = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz"
RUNTIME = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"


POLICIES = {
    "boatlee_v21_latest": (
        boatlee_v21_player_action_v1,
        initialize_boatlee_v21_carry_v1,
    ),
    "kaito_v39_history_gate_latest": (
        kaito_v39_history_gate_player_action_v1,
        initialize_kaito_v39_carry_v1,
    ),
    "prvsiyan_soil_v26h_latest": (
        prvsiyan_soil_v26h_player_action_v1,
        initialize_soil_v26h_carry_v1,
    ),
    "prvsiyan_moon_v92_latest": (
        prvsiyan_moon_v92_player_action_v1,
        initialize_moon_v92_carry_v1,
    ),
    "steven_e284_hadouken_latest": (
        steven_e284_hadouken_player_action_v1,
        initialize_latest_e284_carry_v1,
    ),
    "salem_harvestforge_x_latest": (
        salem_harvestforge_x_player_action_v1,
        initialize_x562_carry_v1,
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _pair(left: Action, right: Action) -> Action:
    return Action(*(jnp.stack((a, b), axis=1) for a, b in zip(left, right, strict=True)))


def _make_rollout(policy, initialize_carry, tables, bank, runtime, candidate_seat: int):
    @jax.jit
    def rollout(initial, events, opponent_tape):
        batch_size = initial.step.shape[0]

        def body(value, opponent_action):
            states, carry = value
            candidate_action, carry = policy(
                states, runtime, bank, carry, candidate_seat
            )
            actions = (
                _pair(candidate_action, opponent_action)
                if candidate_seat == 0
                else _pair(opponent_action, candidate_action)
            )
            next_states = batched_step_sync(states, actions, events, tables)
            return (next_states, carry), (actions, next_states)

        return jax.lax.scan(
            body, (initial, initialize_carry(batch_size)), xs=opponent_tape
        )

    return rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent", choices=tuple(POLICIES), required=True)
    parser.add_argument("--candidate-seat", type=int, choices=(0, 1))
    args = parser.parse_args()
    started = time.perf_counter()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    cache = Path.home() / ".cache" / "kaggriculture_jax"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    trace_path = args.trace_receipt.resolve()
    trace_receipt = json.loads(trace_path.read_text(encoding="utf-8"))
    if trace_receipt["official_package_version"] != "1.32.7":
        raise RuntimeError("trace is not official package 1.32.7")

    tables = load_tables()
    bank = load_bank(BANK)
    runtime = load_high_potential_runtime_tables_v1(RUNTIME)
    policy, initialize_carry = POLICIES[args.agent]
    counters = {
        "contexts": 0,
        "action_exact_contexts": 0,
        "state_exact_contexts": 0,
        "terminal_reward_exact_contexts": 0,
        "first_action_mismatch": None,
        "first_state_mismatch": None,
    }
    timings = []
    tape_diagnostics = []

    selected_seats = (
        (args.candidate_seat,) if args.candidate_seat is not None else (0, 1)
    )
    for candidate_seat in selected_seats:
        traces = [
            row
            for row in trace_receipt["traces"]
            if row["opponent"] == args.agent
            and int(row["candidate_seat"]) == candidate_seat
        ]
        if not traces:
            raise RuntimeError(f"no traces for {args.agent} seat {candidate_seat}")
        seeds = [int(row["seed"]) for row in traces]
        official = [load_frames(str(row["path"])) for row in traces]
        weed, shops = build_events_v1(seeds)
        events = Events(jnp.asarray(weed), jnp.asarray(shops))
        initial = jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))
        expected_batches = [
            stack_actions([encode_actions(frame["actions"]) for frame in frames[1:]])
            for frames in official
        ]
        opponent_seat = 1 - candidate_seat
        opponent_tape = Action(
            *(
                jnp.stack(
                    [
                        getattr(expected, field_name)[:, opponent_seat]
                        for expected in expected_batches
                    ],
                    axis=1,
                )
                for field_name in Action._fields
            )
        )
        rollout = _make_rollout(
            policy, initialize_carry, tables, bank, runtime, candidate_seat
        )
        run_started = time.perf_counter()
        result, scanned = rollout(initial, events, opponent_tape)
        terminal = result[0]
        actions, trajectory = scanned
        jax.block_until_ready(terminal.money)
        timings.append(
            {
                "candidate_seat": candidate_seat,
                "games": len(traces),
                "seconds": time.perf_counter() - run_started,
            }
        )
        host_actions, host_trajectory, host_terminal, host_initial = jax.device_get(
            (actions, trajectory, terminal, initial)
        )

        expected_tape = Action(
            *(
                jnp.stack([getattr(expected, field_name) for expected in expected_batches], axis=1)
                for field_name in Action._fields
            )
        )

        @jax.jit
        def replay_tape(state):
            def replay_body(current, exact_action):
                nxt = batched_step_sync(current, exact_action, events, tables)
                return nxt, nxt

            return jax.lax.scan(replay_body, state, expected_tape)

        exact_terminal, exact_trajectory = replay_tape(initial)
        exact_terminal, exact_trajectory = jax.device_get(
            (exact_terminal, exact_trajectory)
        )
        first_policy_vs_tape_state = None
        for state_field in State._fields:
            left = np.asarray(getattr(host_trajectory, state_field))
            right = np.asarray(getattr(exact_trajectory, state_field))
            difference = np.argwhere(left != right)
            if difference.size:
                row = difference[0]
                first_policy_vs_tape_state = {
                    "field": state_field,
                    "index": row.astype(int).tolist(),
                    "policy": int(left[tuple(row)]),
                    "tape": int(right[tuple(row)]),
                }
                break
        tape_diagnostics.append(
            {
                "candidate_seat": candidate_seat,
                "first_policy_vs_exact_tape_state": first_policy_vs_tape_state,
                "terminal_equal": all(
                    np.array_equal(np.asarray(a), np.asarray(b))
                    for a, b in zip(host_terminal, exact_terminal, strict=True)
                ),
            }
        )

        for batch, frames in enumerate(official):
            counters["contexts"] += 1
            expected = expected_batches[batch]
            action_exact = True
            earliest_action_difference = None
            for field_name in Action._fields:
                actual_field = np.asarray(getattr(host_actions, field_name)[:, batch])
                expected_field = np.asarray(getattr(expected, field_name))
                different = np.argwhere(actual_field != expected_field)
                if different.size:
                    action_exact = False
                    row = different[0]
                    candidate = (int(row[0]), field_name, row, actual_field, expected_field)
                    if (
                        earliest_action_difference is None
                        or candidate[0] < earliest_action_difference[0]
                    ):
                        earliest_action_difference = candidate
            if (
                counters["first_action_mismatch"] is None
                and earliest_action_difference is not None
            ):
                        step, field_name, row, actual_field, expected_field = earliest_action_difference
                        counters["first_action_mismatch"] = {
                            "candidate_seat": candidate_seat,
                            "seed": seeds[batch],
                            "field": field_name,
                            "index": row.astype(int).tolist(),
                            "actual": int(actual_field[tuple(row)]),
                            "expected": int(expected_field[tuple(row)]),
                            "actual_action": {
                                name: np.asarray(getattr(host_actions, name)[step, batch]).tolist()
                                for name in Action._fields
                            },
                            "expected_action": {
                                name: np.asarray(getattr(expected, name)[step]).tolist()
                                for name in Action._fields
                            },
                        }
            counters["action_exact_contexts"] += int(action_exact)

            state_exact = True
            try:
                assert_state_matches_frame(
                    State(*(field[batch] for field in host_initial)), frames[0]
                )
                for step, frame in enumerate(frames[1:]):
                    assert_state_matches_frame(
                        state_at(host_trajectory, step, batch), frame
                    )
            except AssertionError as error:
                state_exact = False
                if counters["first_state_mismatch"] is None:
                    counters["first_state_mismatch"] = {
                        "candidate_seat": candidate_seat,
                        "seed": seeds[batch],
                        "message": str(error)[:2000],
                    }
            counters["state_exact_contexts"] += int(state_exact)
            counters["terminal_reward_exact_contexts"] += int(
                np.array_equal(
                    host_terminal.reward[batch], np.asarray(frames[-1]["reward"])
                )
            )

    for key in ("action", "state", "terminal_reward"):
        counters[f"{key}_exact_rate"] = (
            counters[f"{key}_exact_contexts"] / counters["contexts"]
        )
    strict = all(
        counters[f"{key}_exact_contexts"] == counters["contexts"]
        for key in ("action", "state", "terminal_reward")
    )
    payload = {
        "schema": "kaggriculture.latest_public6_20260822-jax-stepwise-parity.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if strict else "FAIL",
        "strict_exact": strict,
        "agent": args.agent,
        "backend": jax.default_backend(),
        "jax_version": importlib.metadata.version("jax"),
        "official_package_version": "1.32.7",
        "seat_swapped": args.candidate_seat is None,
        "result": counters,
        "timing": timings,
        "tape_diagnostics": tape_diagnostics,
        "inputs": {
            "trace_receipt": {"path": str(trace_path), "sha256": _sha256(trace_path)},
            "bank": {"path": str(BANK), "sha256": _sha256(BANK)},
            "runtime": {"path": str(RUNTIME), "sha256": _sha256(RUNTIME)},
        },
        "wall_seconds": time.perf_counter() - started,
        "acceptance_boundary": (
            "PASS requires JAX candidate actions, every official public/private state "
            "frame, and terminal rewards to equal official Python 1.32.7 in both seats."
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": payload["status"], "result": counters, "output": str(output)},
            ensure_ascii=False,
        )
    )
    return 0 if strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
