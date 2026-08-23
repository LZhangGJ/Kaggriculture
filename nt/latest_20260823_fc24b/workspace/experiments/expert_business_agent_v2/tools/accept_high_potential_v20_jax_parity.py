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
    MODE_TETSUTANI,
    high_potential_v20_player_action_v1,
    initialize_high_potential_v20_carry_v1,
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.public_g02_gpu import (  # noqa: E402
    initialize_public_g02_carry_v1,
    public_rc5_weed_player_action_v1,
)
from strategic_v5.public_v25_gpu import public_v27_player_action_exact_v1  # noqa: E402


TARGETS = {
    "boatlee_v20_multi_route": ("v20", MODE_BOATLEE, "boatlee", -1),
    "rayk_k320_adaptive_rank1": ("v20", MODE_RAY_K320, "ray_k320", -1),
    "tetsutani_adaptive_premium_queue": ("v20", MODE_TETSUTANI, "tetsutani", -1),
    "kaito_v27_midgame_reset": ("v27", -1, "kaito_v27", 8),
    "flexonafft_v59_multi_route": ("flex", -1, "flex_v59", 9),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def pair(left: Action, right: Action) -> Action:
    return Action(*(jnp.stack((a, b), axis=1) for a, b in zip(left, right, strict=True)))


def make_rollout(
    bank, runtime, tables, candidate_seat: int, kind: str, mode: int, route_id: int
):
    target_seat = 1 - candidate_seat

    @jax.jit
    def rollout(initial, events, candidate_tape):
        batch_size = initial.step.shape[0]

        def body(value, candidate_action):
            states, carry = value
            if kind == "v20":
                target_action, carry = high_potential_v20_player_action_v1(
                    states, tables, bank, runtime, carry, target_seat, mode
                )
            else:
                route_ids = jnp.full((batch_size,), route_id, dtype=jnp.int32)
                if kind == "v27":
                    target_action, carry = public_v27_player_action_exact_v1(
                        states, runtime, bank, route_ids, carry, target_seat
                    )
                else:
                    target_action, carry = public_rc5_weed_player_action_v1(
                        states, bank, route_ids, carry, target_seat
                    )
            actions = (
                pair(candidate_action, target_action)
                if candidate_seat == 0
                else pair(target_action, candidate_action)
            )
            next_states = batched_step_sync(states, actions, events, tables)
            return (next_states, carry), (actions, next_states)

        initial_carry = (
            initialize_high_potential_v20_carry_v1(batch_size)
            if kind == "v20"
            else initialize_public_g02_carry_v1(batch_size)
        )
        return jax.lax.scan(body, (initial, initial_carry), xs=candidate_tape)

    return rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--trace-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    cache = Path.home() / ".cache" / "kaggriculture_jax"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    bank_path = args.bank.resolve()
    runtime_path = args.runtime.resolve()
    bank_receipt_path = args.bank_receipt.resolve()
    runtime_receipt_path = args.runtime_receipt.resolve()
    trace_receipt_path = args.trace_receipt.resolve()
    bank_receipt = json.loads(bank_receipt_path.read_text(encoding="utf-8"))
    runtime_receipt = json.loads(runtime_receipt_path.read_text(encoding="utf-8"))
    trace_receipt = json.loads(trace_receipt_path.read_text(encoding="utf-8"))
    if trace_receipt["official_package_version"] != "1.32.7":
        raise RuntimeError("official traces were not generated by version 1.32.7")
    if sha256(bank_path).lower() != bank_receipt["bank_sha256"].lower():
        raise RuntimeError("route bank hash mismatch")
    if sha256(runtime_path).lower() != runtime_receipt["output_sha256"].lower():
        raise RuntimeError("runtime table hash mismatch")

    bank = load_bank(bank_path)
    runtime = load_high_potential_runtime_tables_v1(runtime_path)
    tables = load_tables()
    results = []
    for target_name, (kind, mode, mode_name, route_id) in TARGETS.items():
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
                if row["opponent"] == target_name
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
                            getattr(expected, field_name)[:, candidate_seat]
                            for expected in expected_batches
                        ],
                        axis=1,
                    )
                    for field_name in Action._fields
                )
            )
            rollout = make_rollout(
                bank, runtime, tables, candidate_seat, kind, mode, route_id
            )
            run_started = time.perf_counter()
            result, scanned = rollout(initial, events, candidate_tape)
            terminal = result[0]
            actions, trajectory = scanned
            jax.block_until_ready(terminal.money)
            timing.append(
                {
                    "candidate_seat": candidate_seat,
                    "games": len(traces),
                    "seconds": time.perf_counter() - run_started,
                }
            )
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
            counters[f"{key}_exact_rate"] = counters[f"{key}_exact_contexts"] / counters[
                "contexts"
            ]
        strict = all(
            counters[f"{key}_exact_contexts"] == counters["contexts"]
            for key in ("action", "state", "terminal_reward")
        )
        results.append(
            {
                "opponent": target_name,
                "mode": mode_name,
                "status": "PASS" if strict else "FAIL",
                "strict_exact": strict,
                "result": counters,
                "timing": timing,
            }
        )
        print(json.dumps(results[-1], ensure_ascii=True), flush=True)

    status = "PASS" if all(row["strict_exact"] for row in results) else "FAIL"
    output = {
        "schema": "kaggriculture.high_potential_v20_jax_parity.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "backend": jax.default_backend(),
        "jax": importlib.metadata.version("jax"),
        "official_package_version": "1.32.7",
        "seeds": trace_receipt["seeds"],
        "seat_swapped": True,
        "results": results,
        "bank_sha256": sha256(bank_path),
        "runtime_sha256": sha256(runtime_path),
        "trace_receipt_sha256": sha256(trace_receipt_path),
        "wall_seconds": time.perf_counter() - started,
        "acceptance_boundary": (
            "PASS requires every action tensor, every official public/private state frame, "
            "and terminal reward to match official Python 1.32.7 in both seats."
        ),
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "output": str(output_path)}, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
