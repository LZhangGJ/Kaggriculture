#!/usr/bin/env python3
"""Focused official/JAX stepwise parity audit for public G02 V17."""

from __future__ import annotations

import argparse
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
for path in (
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "gpu_sim/tests",
):
    sys.path.insert(0, str(path))

from kaggriculture_jax import encode_actions, stack_actions  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events, State  # noqa: E402
from reference_assertions import assert_state_matches_frame  # noqa: E402
from run_jax_dynamic_counterfactual_panel import (  # noqa: E402
    batched_step_sync,
    build_events_v1,
    build_router_arrays,
    load_bank,
    pair,
)
from strategic_v5.public_g02_gpu import (  # noqa: E402
    initialize_public_g02_carry_v1,
    public_g02_player_action_v1,
    public_rc5_weed_player_action_v1,
)
from strategic_v5.public_v25_gpu import public_v25_player_action_v1  # noqa: E402
from strategic_v5.public_v14_gpu import public_v14_player_action_v1  # noqa: E402
from strategic_v5.public_v21_gpu import public_v21_player_action_v1  # noqa: E402
from strategic_v5.public_v13_r3_gpu import public_v13_r3_player_action_v1  # noqa: E402
from strategic_v5.public_c68_gpu import public_c68_player_action_v1  # noqa: E402
from strategic_v5.public_four_hire_gpu import (  # noqa: E402
    initialize_four_hire_carry_v1,
    public_four_hire_player_action_v1,
)
from strategic_v5.public_tran_cashflow_gpu import (  # noqa: E402
    initialize_tran_cashflow_carry_v1,
    public_tran_cashflow_player_action_v1,
)
from strategic_v5.public_bruce_route1_gpu import (  # noqa: E402
    initialize_bruce_route1_carry_v1,
    public_bruce_route1_player_action_v1,
)
from strategic_v5.public_v19_control_gpu import (  # noqa: E402
    public_v19_control_player_action_v1,
)
from strategic_v5.public_v18_closed_loop_gpu import (  # noqa: E402
    initialize_v18_closed_loop_carry_v1,
    public_v18_closed_loop_player_action_v1,
)


BANK = ROOT / "experiments/expert_business_agent_v2/artifacts/jax_route46_dynamic_broad23_bank_v4.npz"
BANK_RECEIPT = ROOT / "experiments/expert_business_agent_v2/receipts/jax_route46_dynamic_broad23_bank_v4.json"
TRACE_RECEIPT = ROOT / "experiments/expert_business_agent_v2/receipts/weak6_official_parity_traces_v1.json"
RECEIPT_G02 = ROOT / "experiments/expert_business_agent_v2/receipts/public_g02_jax_stepwise_parity_v1.json"
RECEIPT_G04 = ROOT / "experiments/expert_business_agent_v2/receipts/public_g04_jax_stepwise_parity_v1.json"
RECEIPT_V25 = ROOT / "experiments/expert_business_agent_v2/receipts/public_v25_jax_stepwise_parity_v1.json"
RECEIPT_V14 = ROOT / "experiments/expert_business_agent_v2/receipts/public_v14_jax_stepwise_parity_v1.json"
RECEIPT_V21 = ROOT / "experiments/expert_business_agent_v2/receipts/public_v21_jax_stepwise_parity_v1.json"
RECEIPT_V13 = ROOT / "experiments/expert_business_agent_v2/receipts/public_v13_r3_jax_stepwise_parity_v1.json"
RECEIPT_C68 = ROOT / "experiments/expert_business_agent_v2/receipts/public_c68_jax_stepwise_parity_v1.json"
RECEIPT_FOUR_HIRE = ROOT / "experiments/expert_business_agent_v2/receipts/public_four_hire_jax_stepwise_parity_v1.json"
RECEIPT_TRAN = ROOT / "experiments/expert_business_agent_v2/receipts/public_tran_cashflow_jax_stepwise_parity_v1.json"
RECEIPT_BRUCE = ROOT / "experiments/expert_business_agent_v2/receipts/public_bruce_route1_jax_stepwise_parity_v1.json"
RECEIPT_V19 = ROOT / "experiments/expert_business_agent_v2/receipts/public_v19_control_jax_stepwise_parity_v1.json"
RECEIPT_V18 = ROOT / "experiments/expert_business_agent_v2/receipts/public_v18_closed_loop_jax_stepwise_parity_v1.json"


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


def make_rollout(
    bank,
    tables,
    candidate_seat: int,
    skeleton: int,
    rival_schedule,
    mode: str,
    prototype_signature=None,
    prototype_sales=None,
    hazard_enabled=None,
    hazard_cap=None,
    ray_skeleton=None,
    expert_skeleton_ids=None,
    feature_scale=None,
    market_bias_by_seat=None,
    prototypes_by_day=None,
    distance_strength=None,
    stay_bonus=None,
):
    opponent_seat = 1 - candidate_seat

    @jax.jit
    def rollout(initial, events, candidate_tape):
        batch_size = initial.step.shape[0]
        skeleton_ids = jnp.full((batch_size,), skeleton, dtype=jnp.int32)

        def body(value, candidate_action):
            states, g02_carry = value
            if mode == "g02":
                g02_action, g02_carry = public_g02_player_action_v1(
                    states, bank, skeleton_ids, rival_schedule, g02_carry, opponent_seat
                )
            elif mode == "g04":
                g02_action, g02_carry = public_rc5_weed_player_action_v1(
                    states, bank, skeleton_ids, g02_carry, opponent_seat
                )
            elif mode == "v25":
                g02_action, g02_carry = public_v25_player_action_v1(
                    states,
                    tables,
                    bank,
                    skeleton_ids,
                    g02_carry,
                    opponent_seat,
                )
            elif mode == "v14":
                g02_action, g02_carry = public_v14_player_action_v1(
                    states,
                    tables,
                    bank,
                    skeleton_ids,
                    g02_carry,
                    opponent_seat,
                )
            elif mode == "v21":
                g02_action, g02_carry = public_v21_player_action_v1(
                    states,
                    bank,
                    skeleton_ids,
                    prototype_signature,
                    prototype_sales,
                    g02_carry,
                    opponent_seat,
                )
            elif mode == "v13":
                g02_action, g02_carry = public_v13_r3_player_action_v1(
                    states,
                    bank,
                    skeleton_ids,
                    hazard_enabled,
                    hazard_cap,
                    g02_carry,
                    opponent_seat,
                )
            elif mode == "four_hire":
                ray_ids = jnp.full(
                    (batch_size,), ray_skeleton, dtype=jnp.int32
                )
                g02_action, g02_carry = public_four_hire_player_action_v1(
                    states,
                    tables,
                    bank,
                    skeleton_ids,
                    ray_ids,
                    g02_carry,
                    opponent_seat,
                )
            elif mode == "tran":
                g02_action, g02_carry = public_tran_cashflow_player_action_v1(
                    states,
                    bank,
                    skeleton_ids,
                    g02_carry,
                    opponent_seat,
                )
            elif mode == "bruce":
                g02_action, g02_carry = public_bruce_route1_player_action_v1(
                    states,
                    bank,
                    skeleton_ids,
                    g02_carry,
                    opponent_seat,
                )
            elif mode == "v19":
                g02_action = public_v19_control_player_action_v1(
                    states,
                    bank,
                    skeleton_ids,
                    opponent_seat,
                )
            elif mode == "v18":
                g02_action, g02_carry = public_v18_closed_loop_player_action_v1(
                    states,
                    bank,
                    skeleton_ids,
                    expert_skeleton_ids,
                    feature_scale,
                    market_bias_by_seat,
                    prototypes_by_day,
                    distance_strength,
                    stay_bonus,
                    g02_carry,
                    opponent_seat,
                )
            else:
                g02_action, g02_carry = public_c68_player_action_v1(
                    states,
                    tables,
                    bank,
                    skeleton_ids,
                    g02_carry,
                    opponent_seat,
                )
            actions = (
                pair(candidate_action, g02_action)
                if candidate_seat == 0
                else pair(g02_action, candidate_action)
            )
            next_states = batched_step_sync(states, actions, events, tables)
            return (next_states, g02_carry), (actions, next_states)

        if mode == "four_hire":
            initial_carry = initialize_four_hire_carry_v1(batch_size)
        elif mode == "tran":
            initial_carry = initialize_tran_cashflow_carry_v1(batch_size)
        elif mode == "bruce":
            initial_carry = initialize_bruce_route1_carry_v1(batch_size)
        elif mode == "v18":
            initial_carry = initialize_v18_closed_loop_carry_v1(batch_size)
        else:
            initial_carry = initialize_public_g02_carry_v1(batch_size)
        return jax.lax.scan(
            body,
            (
                initial,
                initial_carry,
            ),
            xs=candidate_tape,
        )

    return rollout


def run_four_hire_stepwise(
    initial,
    events,
    candidate_tape,
    bank,
    tables,
    candidate_seat: int,
    kaito_skeleton: int,
    ray_skeleton: int,
):
    """Keep the composite policy and simulator in separate compiled graphs.

    Nesting Four-Hire's V25/C68 graph inside a 719-step ``lax.scan`` causes
    pathological XLA optimization time on the local 3090 stack.  A host loop
    still executes every transition on GPU while compiling each reusable graph
    only once, which is ideal for the small strict-parity batch.
    """

    opponent_seat = 1 - candidate_seat
    batch_size = initial.step.shape[0]
    kaito_ids = jnp.full((batch_size,), kaito_skeleton, dtype=jnp.int32)
    ray_ids = jnp.full((batch_size,), ray_skeleton, dtype=jnp.int32)

    @jax.jit
    def policy(states, carry):
        return public_four_hire_player_action_v1(
            states,
            tables,
            bank,
            kaito_ids,
            ray_ids,
            carry,
            opponent_seat,
        )

    @jax.jit
    def advance(states, candidate_action, opponent_action):
        actions = (
            pair(candidate_action, opponent_action)
            if candidate_seat == 0
            else pair(opponent_action, candidate_action)
        )
        return actions, batched_step_sync(states, actions, events, tables)

    states = initial
    carry = initialize_four_hire_carry_v1(batch_size)
    action_rows = []
    state_rows = []
    for step in range(719):
        opponent_action, carry = policy(states, carry)
        candidate_action = Action(
            *(getattr(candidate_tape, field)[step] for field in Action._fields)
        )
        actions, states = advance(states, candidate_action, opponent_action)
        action_rows.append(actions)
        state_rows.append(states)
    actions = Action(
        *(
            jnp.stack([getattr(row, field) for row in action_rows], axis=0)
            for field in Action._fields
        )
    )
    trajectory = State(
        *(
            jnp.stack([getattr(row, field) for row in state_rows], axis=0)
            for field in State._fields
        )
    )
    return states, actions, trajectory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", choices=("g02", "g04", "v25", "v14", "v21", "v13", "c68", "four_hire", "tran", "bruce", "v19", "v18"), default="g02")
    parser.add_argument("--bank", type=Path, default=BANK)
    parser.add_argument("--bank-receipt", type=Path, default=BANK_RECEIPT)
    parser.add_argument("--trace-receipt", type=Path, default=TRACE_RECEIPT)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    mode = args.opponent
    mode_spec = {
        "g02": ("public_g02_rc5_c166", RECEIPT_G02, "public14_rank_agent_v17", "public_g02_exact"),
        "g04": ("public_g04_soil_rain", RECEIPT_G04, "public06_soil_rain", "public_g04_exact"),
        "v25": ("public_g06_v25", RECEIPT_V25, "public_g06_v25", "public_v25_exact"),
        "v14": ("public_g08_v14", RECEIPT_V14, "public_g08_v14", "public_v14_exact"),
        "v21": ("public_g11_v21", RECEIPT_V21, "public_g11_v21", "public_v21_exact"),
        "v13": ("public_g12_v13_r3", RECEIPT_V13, "public_g12_v13_r3", "public_v13_r3_exact"),
        "c68": ("public_g09_c68_thunder", RECEIPT_C68, "public_g09_c68_thunder", "public_c68_exact"),
        "four_hire": ("public_g10_four_hire", RECEIPT_FOUR_HIRE, "public_g10_four_hire", "public_four_hire_exact"),
        "tran": ("public_g16_tran_cashflow", RECEIPT_TRAN, "public_g16_tran_cashflow", "public_tran_cashflow_exact"),
        "bruce": ("public_g13_bruce_route1", RECEIPT_BRUCE, "public_g13_bruce_route1", "public_bruce_route1_exact"),
        "v19": ("public_g14_v19_control", RECEIPT_V19, "public_g14_v19_control", "public_v19_control_exact"),
        "v18": ("public_g15_v18_closed_loop", RECEIPT_V18, "public_g15_v18_closed_loop", "public_v18_closed_loop_exact"),
    }
    opponent, default_receipt, bank_name, expected_kind = mode_spec[mode]
    receipt_path = args.receipt or default_receipt
    bank_path = args.bank if args.bank.is_absolute() else ROOT / args.bank
    bank_receipt_path = (
        args.bank_receipt if args.bank_receipt.is_absolute() else ROOT / args.bank_receipt
    )
    trace_receipt_path = (
        args.trace_receipt if args.trace_receipt.is_absolute() else ROOT / args.trace_receipt
    )
    receipt_path = receipt_path if receipt_path.is_absolute() else ROOT / receipt_path
    started = time.perf_counter()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    cache = Path.home() / ".cache/kaggriculture_jax"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    trace_receipt = json.loads(trace_receipt_path.read_text(encoding="utf-8"))
    bank_receipt = json.loads(bank_receipt_path.read_text(encoding="utf-8"))
    if trace_receipt["official_package_version"] != "1.32.7":
        raise RuntimeError("official trace package is not 1.32.7")
    spec = next(row for row in bank_receipt["opponents"] if row["name"] == bank_name)
    if spec["kind"] != expected_kind:
        raise RuntimeError(f"{mode.upper()} bank kind is not exact: {spec['kind']}")

    bank = load_bank(bank_path)
    router = build_router_arrays(bank_receipt)
    tables = load_tables()
    target = {
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
            if row["opponent"] == opponent and int(row["candidate_seat"]) == candidate_seat
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
                    [getattr(expected, field_name)[:, candidate_seat] for expected in expected_batches],
                    axis=1,
                )
                for field_name in Action._fields
            )
        )
        run_started = time.perf_counter()
        if mode == "four_hire":
            terminal, actions, trajectory = run_four_hire_stepwise(
                initial,
                events,
                candidate_tape,
                bank,
                tables,
                candidate_seat,
                int(spec["kaito_skeleton_id"]),
                int(spec["ray_skeleton_id"]),
            )
        else:
            rollout = make_rollout(
                bank,
                tables,
                candidate_seat,
                int(spec["default_skeleton_id"]),
                router.public_g02_rival_schedule,
                mode,
                router.public_v21_prototype_signature,
                router.public_v21_prototype_sales,
                router.public_v13_hazard_enabled,
                router.public_v13_hazard_cap,
                int(spec.get("ray_skeleton_id", spec["default_skeleton_id"])),
                router.public_v18_expert_skeleton_ids,
                router.public_v18_feature_scale,
                router.public_v18_market_bias_by_seat,
                router.public_v18_prototypes_by_day,
                router.public_v18_distance_strength,
                router.public_v18_stay_bonus,
            )
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
            seed = seeds[batch]
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
            reward_exact = np.array_equal(
                host_terminal.reward[batch], np.asarray(frames[-1]["reward"])
            )
            target["terminal_reward_exact_contexts"] += int(reward_exact)

    for key in ("action", "state", "terminal_reward"):
        target[f"{key}_exact_rate"] = target[f"{key}_exact_contexts"] / target["contexts"]
    target["strict_exact"] = all(
        target[f"{key}_exact_contexts"] == target["contexts"]
        for key in ("action", "state", "terminal_reward")
    )
    output = {
        "schema": f"kaggriculture_public_{mode}_jax_stepwise_parity_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if target["strict_exact"] else "FAIL",
        "backend": jax.default_backend(),
        "jax": importlib.metadata.version("jax"),
        "official_package_version": "1.32.7",
        "seeds": trace_receipt["seeds"],
        "seat_swapped": True,
        "opponent": opponent,
        "result": target,
        "timing": timing,
        "bank_sha256": sha256(bank_path),
        "bank_receipt_sha256": sha256(bank_receipt_path),
        "trace_receipt_sha256": sha256(trace_receipt_path),
        "wall_seconds": time.perf_counter() - started,
        "acceptance_boundary": "PASS requires every action field, state frame, and terminal reward to match official Python 1.32.7.",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if target["strict_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
