#!/usr/bin/env python3
"""Strict official Python 1.32.7 / JAX parity for public Kaito V48."""

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
from strategic_v5.recent_public_20260825_gpu import (  # noqa: E402
    initialize_kaito_v48_carry_v1,
    kaito_v48_player_action_v1,
)


BANK = ROOT / "experiments/public_recent_20260825/artifacts/recent_v48_route_bank_v1.npz"
RUNTIME = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def pair(left: Action, right: Action) -> Action:
    return Action(
        *(jnp.stack((a, b), axis=1) for a, b in zip(left, right, strict=True))
    )


def make_rollout(tables, bank, runtime, candidate_seat: int):
    @jax.jit
    def rollout(initial, events, opponent_tape):
        batch_size = initial.step.shape[0]

        def body(value, opponent_action):
            states, carry = value
            candidate_action, carry = kaito_v48_player_action_v1(
                states, runtime, bank, carry, candidate_seat
            )
            actions = (
                pair(candidate_action, opponent_action)
                if candidate_seat == 0
                else pair(opponent_action, candidate_action)
            )
            next_states = batched_step_sync(states, actions, events, tables)
            return (next_states, carry), (candidate_action, next_states)

        return jax.lax.scan(
            body, (initial, initialize_kaito_v48_carry_v1(batch_size)), xs=opponent_tape
        )

    return rollout


def first_array_difference(actual, expected):
    difference = np.argwhere(np.asarray(actual) != np.asarray(expected))
    if not difference.size:
        return None
    index = tuple(int(value) for value in difference[0])
    return {
        "index": list(index),
        "actual": int(np.asarray(actual)[index]),
        "expected": int(np.asarray(expected)[index]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    cache = Path.home() / ".cache" / "kaggriculture_jax"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    trace_path = args.trace_receipt.resolve()
    receipt = json.loads(trace_path.read_text(encoding="utf-8"))
    if receipt["official_package_version"] != "1.32.7":
        raise RuntimeError("trace is not official package 1.32.7")

    tables = load_tables()
    bank = load_bank(BANK)
    runtime = load_high_potential_runtime_tables_v1(RUNTIME)
    counters = {
        "contexts": 0,
        "action_exact_contexts": 0,
        "state_exact_contexts": 0,
        "terminal_reward_exact_contexts": 0,
        "first_action_mismatch": None,
        "first_state_mismatch": None,
    }
    timings = []

    for candidate_seat in (0, 1):
        traces = [
            row
            for row in receipt["traces"]
            if row["opponent"] == "kaito_v48"
            and int(row["candidate_seat"]) == candidate_seat
        ]
        if not traces:
            raise RuntimeError(f"no traces for seat {candidate_seat}")
        seeds = [int(row["seed"]) for row in traces]
        official = [load_frames(str(row["path"])) for row in traces]
        weed, shops = build_events_v1(seeds)
        events = Events(jnp.asarray(weed), jnp.asarray(shops))
        initial = jax.vmap(reset)(jnp.asarray(seeds, dtype=jnp.int32))
        expected = [
            stack_actions([encode_actions(frame["actions"]) for frame in frames[1:]])
            for frames in official
        ]
        opponent_seat = 1 - candidate_seat
        opponent_tape = Action(
            *(
                jnp.stack(
                    [getattr(item, field)[:, opponent_seat] for item in expected],
                    axis=1,
                )
                for field in Action._fields
            )
        )
        rollout = make_rollout(tables, bank, runtime, candidate_seat)
        run_started = time.perf_counter()
        result, scanned = rollout(initial, events, opponent_tape)
        terminal = result[0]
        candidate_actions, trajectory = scanned
        jax.block_until_ready(terminal.money)
        timings.append(
            {
                "candidate_seat": candidate_seat,
                "games": len(traces),
                "seconds": time.perf_counter() - run_started,
            }
        )
        candidate_actions, trajectory, terminal, initial = jax.device_get(
            (candidate_actions, trajectory, terminal, initial)
        )

        for batch, frames in enumerate(official):
            counters["contexts"] += 1
            expected_candidate = Action(
                *(
                    np.asarray(getattr(expected[batch], field))[:, candidate_seat]
                    for field in Action._fields
                )
            )
            action_exact = True
            earliest = None
            for field in Action._fields:
                actual_field = np.asarray(getattr(candidate_actions, field)[:, batch])
                expected_field = np.asarray(getattr(expected_candidate, field))
                mismatch = first_array_difference(actual_field, expected_field)
                if mismatch is not None:
                    action_exact = False
                    step = int(mismatch["index"][0])
                    if earliest is None or step < earliest["step"]:
                        earliest = {"step": step, "field": field, **mismatch}
            if counters["first_action_mismatch"] is None and earliest is not None:
                mismatch_step = int(earliest["step"])
                counters["first_action_mismatch"] = {
                    "candidate_seat": candidate_seat,
                    "seed": seeds[batch],
                    **earliest,
                    "actual_action": {
                        field: np.asarray(getattr(candidate_actions, field)[mismatch_step, batch]).tolist()
                        for field in Action._fields
                    },
                    "expected_action": {
                        field: np.asarray(getattr(expected_candidate, field)[mismatch_step]).tolist()
                        for field in Action._fields
                    },
                }
            counters["action_exact_contexts"] += int(action_exact)

            state_exact = True
            try:
                assert_state_matches_frame(
                    State(*(field[batch] for field in initial)), frames[0]
                )
                for step, frame in enumerate(frames[1:]):
                    assert_state_matches_frame(state_at(trajectory, step, batch), frame)
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
                np.array_equal(terminal.reward[batch], np.asarray(frames[-1]["reward"]))
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
        "schema": "kaggriculture.public-recent-20260825.kaito-v48-jax-parity.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if strict else "FAIL",
        "strict_exact": strict,
        "backend": jax.default_backend(),
        "jax_version": importlib.metadata.version("jax"),
        "official_package_version": "1.32.7",
        "seat_swapped": True,
        "result": counters,
        "timing": timings,
        "inputs": {
            "trace_receipt": {"path": str(trace_path), "sha256": sha256(trace_path)},
            "bank": {"path": str(BANK.resolve()), "sha256": sha256(BANK)},
            "runtime": {"path": str(RUNTIME.resolve()), "sha256": sha256(RUNTIME)},
        },
        "wall_seconds": time.perf_counter() - started,
        "acceptance_boundary": (
            "PASS requires candidate actions, every official state frame, and terminal "
            "rewards to equal official Python 1.32.7 in both seats."
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
