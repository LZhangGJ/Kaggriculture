#!/usr/bin/env python3
"""Strict FC12G/FC15 JAX policy parity against official Python 1.32.7 traces."""

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
    ROOT / "experiments/fusion_champion_v1/src",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "gpu_sim/src",
    ROOT / "gpu_sim/tests",
):
    sys.path.insert(0, str(path))

from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc12g_fc2b_mass_hire_weed_guard_player_action_v1,
    fc15_fc14_x562_split_weed_hire_guard_player_action_v1,
    fc15_opening_hedge_player_action_v1,
    fc24_terminal_crop_salvage_player_action_v1,
    initialize_fusion_champion_carry_v3,
    initialize_fusion_champion_base_suffix_carry_v1,
    initialize_fusion_champion_terminal_salvage_carry_v1,
)
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


LATEST_BANK = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
OLD_BANK = ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
RUNTIME = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _pair(left: Action, right: Action) -> Action:
    return Action(*(jnp.stack((a, b), axis=1) for a, b in zip(left, right, strict=True)))


def _make_rollout(policy, initialize_carry, tables, latest_bank, old_bank, runtime, candidate_seat: int):
    opponent_seat = 1 - candidate_seat

    @jax.jit
    def rollout(initial, events, opponent_tape):
        batch_size = initial.step.shape[0]

        def body(value, opponent_action):
            states, carry = value
            candidate_action, carry = policy(
                states,
                tables,
                latest_bank,
                old_bank,
                runtime,
                carry,
                candidate_seat,
            )
            actions = (
                _pair(candidate_action, opponent_action)
                if candidate_seat == 0
                else _pair(opponent_action, candidate_action)
            )
            next_states = batched_step_sync(states, actions, events, tables)
            return (next_states, carry), (actions, next_states)

        carry = initialize_carry(batch_size)
        return jax.lax.scan(body, (initial, carry), xs=opponent_tape)

    return rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy", choices=("fc12g", "fc15", "fc15r", "fc24b"), default="fc12g"
    )
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
    latest_bank = load_bank(LATEST_BANK)
    old_bank = load_bank(OLD_BANK)
    runtime = load_high_potential_runtime_tables_v1(RUNTIME)
    policy = {
        "fc12g": fc12g_fc2b_mass_hire_weed_guard_player_action_v1,
        "fc15": fc15_fc14_x562_split_weed_hire_guard_player_action_v1,
        "fc15r": fc15_opening_hedge_player_action_v1,
        "fc24b": fc24_terminal_crop_salvage_player_action_v1,
    }[args.policy]
    initialize_carry = {
        "fc12g": initialize_fusion_champion_carry_v3,
        "fc15": initialize_fusion_champion_carry_v3,
        "fc15r": initialize_fusion_champion_base_suffix_carry_v1,
        "fc24b": initialize_fusion_champion_terminal_salvage_carry_v1,
    }[args.policy]
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
            row for row in trace_receipt["traces"]
            if int(row["candidate_seat"]) == candidate_seat
        ]
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
                    [getattr(expected, field_name)[:, opponent_seat] for expected in expected_batches],
                    axis=1,
                )
                for field_name in Action._fields
            )
        )
        rollout = _make_rollout(
            policy, initialize_carry, tables, latest_bank, old_bank, runtime, candidate_seat
        )
        run_started = time.perf_counter()
        result, scanned = rollout(initial, events, opponent_tape)
        terminal = result[0]
        actions, trajectory = scanned
        jax.block_until_ready(terminal.money)
        timings.append({
            "candidate_seat": candidate_seat,
            "games": len(traces),
            "seconds": time.perf_counter() - run_started,
        })
        host_actions, host_trajectory, host_terminal, host_initial = jax.device_get(
            (actions, trajectory, terminal, initial)
        )

        for batch, frames in enumerate(official):
            counters["contexts"] += 1
            expected = expected_batches[batch]
            action_exact = True
            for field_name in Action._fields:
                actual_field = np.asarray(getattr(host_actions, field_name)[:, batch])
                expected_field = np.asarray(getattr(expected, field_name))
                different = np.argwhere(actual_field != expected_field)
                if different.size:
                    action_exact = False
                    if counters["first_action_mismatch"] is None:
                        row = different[0]
                        counters["first_action_mismatch"] = {
                            "candidate_seat": candidate_seat,
                            "seed": seeds[batch],
                            "field": field_name,
                            "index": row.astype(int).tolist(),
                            "actual": int(actual_field[tuple(row)]),
                            "expected": int(expected_field[tuple(row)]),
                            "actual_action": {
                                name: np.asarray(
                                    getattr(host_actions, name)[int(row[0]), batch]
                                ).tolist()
                                for name in Action._fields
                            },
                            "expected_action": {
                                name: np.asarray(
                                    getattr(expected, name)[int(row[0])]
                                ).tolist()
                                for name in Action._fields
                            },
                        }
            counters["action_exact_contexts"] += int(action_exact)

            state_exact = True
            try:
                assert_state_matches_frame(State(*(field[batch] for field in host_initial)), frames[0])
                for step, frame in enumerate(frames[1:]):
                    assert_state_matches_frame(state_at(host_trajectory, step, batch), frame)
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
                np.array_equal(host_terminal.reward[batch], np.asarray(frames[-1]["reward"]))
            )

    for key in ("action", "state", "terminal_reward"):
        counters[f"{key}_exact_rate"] = counters[f"{key}_exact_contexts"] / counters["contexts"]
    strict = all(
        counters[f"{key}_exact_contexts"] == counters["contexts"]
        for key in ("action", "state", "terminal_reward")
    )
    payload = {
        "schema": f"kaggriculture.fusion_champion.{args.policy}-jax-stepwise-parity.v1",
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
            "trace_receipt": {"path": str(trace_path), "sha256": _sha256(trace_path)},
            "latest_bank": {"path": str(LATEST_BANK), "sha256": _sha256(LATEST_BANK)},
            "old_bank": {"path": str(OLD_BANK), "sha256": _sha256(OLD_BANK)},
            "runtime": {"path": str(RUNTIME), "sha256": _sha256(RUNTIME)},
        },
        "wall_seconds": time.perf_counter() - started,
        "acceptance_boundary": (
            f"PASS requires {args.policy.upper()} JAX actions, every official public/private state frame, "
            "and terminal rewards to equal official Python 1.32.7 in both seats."
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "result": counters, "output": str(output)}, ensure_ascii=False))
    return 0 if strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
