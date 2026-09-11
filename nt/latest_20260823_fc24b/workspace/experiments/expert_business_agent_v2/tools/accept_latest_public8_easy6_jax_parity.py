from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "experiments" / "expert_business_agent_v2" / "tools"
for path in (
    TOOLS,
    ROOT / "experiments" / "kawashigi_counterfactual_ranker_v2" / "tools",
    ROOT / "experiments" / "strategic_v5" / "src",
    ROOT / "gpu_sim" / "src",
    ROOT / "gpu_sim" / "tests",
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
    MODE_BOATLEE,
    MODE_RAY_K320,
    MODE_TETSUTANI_LATEST,
    high_potential_v20_player_action_v1,
    initialize_high_potential_v20_carry_v1,
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.latest_public8_gpu import (  # noqa: E402
    deniz_v111_player_action_v1,
    initialize_kaito_v36_carry_v1,
    initialize_x562_carry_v1,
    kaito_v36_player_action_v1,
    x562_player_action_v1,
)
from strategic_v5.public_g02_gpu import initialize_public_g02_carry_v1  # noqa: E402
from strategic_v5.public_v25_gpu import public_v25_player_action_v1  # noqa: E402


TARGETS = {
    "deniz_v111_8c4s_latest": ("deniz", -1, 10),
    "boatlee_v20_latest": ("shared", MODE_BOATLEE, -1),
    "kunal_2026_v1_latest": ("shared", MODE_BOATLEE, -1),
    "rayk_rank_agent_latest": ("shared", MODE_RAY_K320, -1),
    "tetsutani_adaptive_latest": ("shared", MODE_TETSUTANI_LATEST, -1),
    "flex_multi_route_latest": ("flex", -1, 14),
    "kaito_v36_latest": ("kaito", -1, 11),
    "x562_latest": ("x562", -1, -1),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def pair(left: Action, right: Action) -> Action:
    return Action(*(jnp.stack((a, b), axis=1) for a, b in zip(left, right, strict=True)))


def make_rollout(bank, runtime, tables, candidate_seat, kind, mode, route_id):
    target_seat = 1 - candidate_seat

    @jax.jit
    def rollout(initial, events, candidate_tape):
        batch_size = initial.step.shape[0]
        ids = jnp.full((batch_size,), route_id, dtype=jnp.int32)

        def body(value, candidate_action):
            states, carry = value
            if kind == "shared":
                action, carry = high_potential_v20_player_action_v1(
                    states, tables, bank, runtime, carry, target_seat, mode
                )
            elif kind == "flex":
                action, carry = public_v25_player_action_v1(
                    states, tables, bank, ids, carry, target_seat
                )
            elif kind == "kaito":
                action, carry = kaito_v36_player_action_v1(
                    states, runtime, bank, ids, carry, target_seat
                )
            elif kind == "x562":
                action, carry = x562_player_action_v1(
                    states, runtime, bank, carry, target_seat
                )
            else:
                action, carry = deniz_v111_player_action_v1(
                    states, bank, ids, carry, target_seat
                )
            actions = (
                pair(candidate_action, action)
                if candidate_seat == 0
                else pair(action, candidate_action)
            )
            next_states = batched_step_sync(states, actions, events, tables)
            return (next_states, carry), (actions, next_states)

        if kind == "shared":
            carry = initialize_high_potential_v20_carry_v1(batch_size)
        elif kind == "kaito":
            carry = initialize_kaito_v36_carry_v1(batch_size)
        elif kind == "x562":
            carry = initialize_x562_carry_v1(batch_size)
        else:
            carry = initialize_public_g02_carry_v1(batch_size)
        return jax.lax.scan(body, (initial, carry), xs=candidate_tape)

    return rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--trace-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--agents",
        nargs="*",
        choices=tuple(TARGETS),
        help="Optional subset; omitted means every implemented target.",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    cache = Path.home() / ".cache" / "kaggriculture_jax"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    bank_receipt = json.loads(args.bank_receipt.read_text(encoding="utf-8"))
    trace_receipt = json.loads(args.trace_receipt.read_text(encoding="utf-8"))
    if sha256(args.bank).lower() != bank_receipt["bank_sha256"].lower():
        raise RuntimeError("bank hash mismatch")
    if trace_receipt["official_package_version"] != "1.32.7":
        raise RuntimeError("official trace version mismatch")
    bank = load_bank(args.bank)
    runtime = load_high_potential_runtime_tables_v1(args.runtime)
    tables = load_tables()
    results = []
    selected_targets = args.agents or list(TARGETS)
    for target in selected_targets:
        kind, mode, route_id = TARGETS[target]
        counters = {
            "contexts": 0,
            "action_exact_contexts": 0,
            "state_exact_contexts": 0,
            "terminal_reward_exact_contexts": 0,
            "first_action_mismatch": None,
            "first_state_mismatch": None,
        }
        timing = []
        for candidate_seat in (0, 1):
            traces = [
                row
                for row in trace_receipt["traces"]
                if row["opponent"] == target
                and int(row["candidate_seat"]) == candidate_seat
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
            candidate_tape = Action(
                *(
                    jnp.stack(
                        [
                            getattr(expected, field)[:, candidate_seat]
                            for expected in expected_batches
                        ],
                        axis=1,
                    )
                    for field in Action._fields
                )
            )
            rollout = make_rollout(
                bank, runtime, tables, candidate_seat, kind, mode, route_id
            )
            begin = time.perf_counter()
            result, scanned = rollout(initial, events, candidate_tape)
            terminal = result[0]
            actions, trajectory = scanned
            jax.block_until_ready(terminal.money)
            timing.append(
                {
                    "candidate_seat": candidate_seat,
                    "games": len(traces),
                    "seconds": time.perf_counter() - begin,
                }
            )
            actual_actions, actual_trajectory, actual_terminal, actual_initial = jax.device_get(
                (actions, trajectory, terminal, initial)
            )
            for batch, frames in enumerate(official):
                counters["contexts"] += 1
                expected = expected_batches[batch]
                action_exact = True
                for field in Action._fields:
                    actual_field = np.asarray(getattr(actual_actions, field)[:, batch])
                    expected_field = np.asarray(getattr(expected, field))
                    different = np.argwhere(actual_field != expected_field)
                    if different.size:
                        action_exact = False
                        if counters["first_action_mismatch"] is None:
                            row = different[0]
                            counters["first_action_mismatch"] = {
                                "candidate_seat": candidate_seat,
                                "seed": seeds[batch],
                                "field": field,
                                "index": row.astype(int).tolist(),
                                "actual": int(actual_field[tuple(row)]),
                                "expected": int(expected_field[tuple(row)]),
                            }
                counters["action_exact_contexts"] += int(action_exact)
                state_exact = True
                try:
                    assert_state_matches_frame(
                        State(*(field[batch] for field in actual_initial)), frames[0]
                    )
                    for step, frame in enumerate(frames[1:]):
                        assert_state_matches_frame(
                            state_at(actual_trajectory, step, batch), frame
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
                        actual_terminal.reward[batch], np.asarray(frames[-1]["reward"])
                    )
                )
        strict = all(
            counters[f"{key}_exact_contexts"] == counters["contexts"]
            for key in ("action", "state", "terminal_reward")
        )
        results.append(
            {
                "agent": target,
                "status": "PASS" if strict else "FAIL",
                "strict_exact": strict,
                "controller_kind": kind,
                "result": counters,
                "timing": timing,
            }
        )
        print(json.dumps(results[-1], ensure_ascii=True), flush=True)

    status = "PASS" if all(row["strict_exact"] for row in results) else "FAIL"
    receipt = {
        "schema": "kaggriculture.latest_public8_easy6_jax_parity.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "backend": jax.default_backend(),
        "jax": importlib.metadata.version("jax"),
        "official_package_version": "1.32.7",
        "seeds": trace_receipt["seeds"],
        "seat_swapped": True,
        "selected_targets": selected_targets,
        "results": results,
        "bank_sha256": sha256(args.bank),
        "trace_receipt_sha256": sha256(args.trace_receipt),
        "wall_seconds": time.perf_counter() - started,
        "acceptance_boundary": (
            "PASS requires every action tensor, every official public/private state "
            "frame, and terminal reward to match official Python 1.32.7 in both seats."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "output": str(args.output)}, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
