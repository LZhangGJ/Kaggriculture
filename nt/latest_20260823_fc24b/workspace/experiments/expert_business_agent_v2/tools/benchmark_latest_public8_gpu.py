from __future__ import annotations

import argparse
import hashlib
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
for path in (
    ROOT / "experiments" / "expert_business_agent_v2" / "tools",
    ROOT / "experiments" / "kawashigi_counterfactual_ranker_v2" / "tools",
    ROOT / "experiments" / "strategic_v5" / "src",
    ROOT / "gpu_sim" / "src",
):
    sys.path.insert(0, str(path))

from kaggriculture_jax.constants import MAX_MARKET_ORDERS, MAX_UNITS, MarketOp, UnitOp  # noqa: E402
from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
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


FROZEN = ROOT / "references" / "public_latest8_20260820"
TARGETS = {
    "deniz_v111_8c4s_latest": ("deniz", -1, 10),
    "boatlee_v20_latest": ("shared", MODE_BOATLEE, -1),
    # Kunal's frozen main.py is byte-identical to Boatlee V20.  It remains a
    # separate target/receipt but intentionally reuses the exact same graph.
    "kunal_2026_v1_latest": ("shared", MODE_BOATLEE, -1),
    "rayk_rank_agent_latest": ("shared", MODE_RAY_K320, -1),
    "kaito_v36_latest": ("kaito", -1, 11),
    "x562_latest": ("x562", -1, -1),
    "tetsutani_adaptive_latest": ("shared", MODE_TETSUTANI_LATEST, -1),
    "flex_multi_route_latest": ("flex", -1, 14),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def null_action(states, player: int) -> Action:
    batch = states.step.shape[0]
    count = jnp.sum(states.unit_active[:, player], axis=1).astype(jnp.int8)
    return Action(
        jnp.full((batch, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8),
        jnp.full((batch, MAX_UNITS), -1, dtype=jnp.int8),
        jnp.ones((batch, MAX_UNITS), dtype=jnp.int32),
        count,
        jnp.full((batch, MAX_MARKET_ORDERS), MarketOp.NONE, dtype=jnp.int8),
        jnp.full((batch, MAX_MARKET_ORDERS), -1, dtype=jnp.int8),
        jnp.zeros((batch, MAX_MARKET_ORDERS), dtype=jnp.int32),
        jnp.zeros((batch,), dtype=jnp.int8),
    )


def pair(left: Action, right: Action) -> Action:
    return Action(*(jnp.stack((a, b), axis=1) for a, b in zip(left, right, strict=True)))


def make_rollout(bank, runtime, tables, kind: str, mode: int, route_id: int):
    @jax.jit
    def rollout(initial, events):
        batch = initial.step.shape[0]
        route_ids = jnp.full((batch,), route_id, dtype=jnp.int32)

        if kind == "shared":
            initial_carry = initialize_high_potential_v20_carry_v1(batch)
        elif kind == "kaito":
            initial_carry = initialize_kaito_v36_carry_v1(batch)
        elif kind == "x562":
            initial_carry = initialize_x562_carry_v1(batch)
        else:
            initial_carry = initialize_public_g02_carry_v1(batch)

        def body(value, _):
            states, carry = value
            if kind == "shared":
                action, carry = high_potential_v20_player_action_v1(
                    states, tables, bank, runtime, carry, 0, mode
                )
            elif kind == "flex":
                action, carry = public_v25_player_action_v1(
                    states, tables, bank, route_ids, carry, 0
                )
            elif kind == "kaito":
                action, carry = kaito_v36_player_action_v1(
                    states, runtime, bank, route_ids, carry, 0
                )
            elif kind == "x562":
                action, carry = x562_player_action_v1(
                    states, runtime, bank, carry, 0
                )
            else:
                action, carry = deniz_v111_player_action_v1(
                    states, bank, route_ids, carry, 0
                )
            states = batched_step_sync(
                states, pair(action, null_action(states, 1)), events, tables
            )
            return (states, carry), None

        (terminal, carry), _ = jax.lax.scan(
            body, (initial, initial_carry), xs=None, length=719
        )
        return terminal, carry

    return rollout


def trees_equal(left, right) -> bool:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        np.array_equal(a, b) for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def duplicated_lanes_equal(tree, half: int) -> bool:
    return all(
        np.array_equal(leaf[:half], leaf[half:])
        for leaf in jax.tree.leaves(tree)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--seed-start", type=int, default=520001)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agents", nargs="*", choices=tuple(TARGETS))
    args = parser.parse_args()
    if args.batch < 1000 or args.batch % 2:
        raise ValueError("--batch must be even and >=1000 for the pollution gate")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    cache = Path.home() / ".cache" / "kaggriculture_jax"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    bank_path, runtime_path = args.bank.resolve(), args.runtime.resolve()
    bank = load_bank(bank_path)
    runtime = load_high_potential_runtime_tables_v1(runtime_path)
    tables = load_tables()

    half = args.batch // 2
    unique_seeds = np.arange(args.seed_start, args.seed_start + half, dtype=np.int32)
    seeds = np.concatenate((unique_seeds, unique_seeds))
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))
    targets = args.agents or list(TARGETS)
    rows = []
    device = jax.devices()[0]

    for name in targets:
        kind, mode, route_id = TARGETS[name]
        rollout = make_rollout(bank, runtime, tables, kind, mode, route_id)
        memory_before = device.memory_stats() or {}

        started = time.perf_counter()
        warm_terminal, warm_carry = rollout(initial, events)
        jax.block_until_ready(warm_terminal.money)
        compile_first_seconds = time.perf_counter() - started

        steady_seconds = []
        outputs = []
        for _ in range(2):
            # A fresh reset and a carry initialized inside the JIT are required
            # on every invocation; no prior-game state is passed back in.
            fresh_initial = jax.vmap(reset)(jnp.asarray(seeds))
            started = time.perf_counter()
            terminal, carry = rollout(fresh_initial, events)
            jax.block_until_ready(terminal.money)
            steady_seconds.append(time.perf_counter() - started)
            outputs.append(jax.device_get((terminal, carry)))

        warm_host = jax.device_get((warm_terminal, warm_carry))
        seconds = float(np.median(steady_seconds))
        terminal_host, carry_host = outputs[-1]
        lane_exact = duplicated_lanes_equal((terminal_host, carry_host), half)
        rerun_exact = trees_equal(outputs[0], outputs[1]) and trees_equal(
            warm_host, outputs[0]
        )
        all_done = bool(np.all(terminal_host.done))
        memory_after = device.memory_stats() or {}
        source = FROZEN / name / "main.py"
        row = {
            "agent": name,
            "source_sha256": sha256(source),
            "controller_kind": kind,
            "batch": args.batch,
            "unique_events": half,
            "compile_and_first_seconds": compile_first_seconds,
            "steady_seconds_runs": steady_seconds,
            "steady_seconds_median": seconds,
            "steady_games_per_second": args.batch / seconds,
            "steady_transitions_per_second": args.batch * 719 / seconds,
            "all_done": all_done,
            "duplicated_event_lanes_exact": lane_exact,
            "fresh_reset_reruns_exact": rerun_exact,
            "cross_game_state_pollution": not (lane_exact and rerun_exact),
            "cash_mean": float(np.mean(terminal_host.money[:, 0])),
            "cash_min": int(np.min(terminal_host.money[:, 0])),
            "cash_max": int(np.max(terminal_host.money[:, 0])),
            "device_bytes_in_use_before": int(memory_before.get("bytes_in_use", 0)),
            "device_bytes_in_use_after": int(memory_after.get("bytes_in_use", 0)),
            "device_peak_bytes_in_use": int(memory_after.get("peak_bytes_in_use", 0)),
            "device_allocator_limit_bytes": int(memory_after.get("bytes_limit", 0)),
        }
        row["status"] = (
            "PASS"
            if all_done and lane_exact and rerun_exact
            else "FAIL"
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)

    output = {
        "schema": "kaggriculture.latest_public8_gpu_benchmark_and_reset.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "device": str(device),
        "bank": str(bank_path),
        "bank_sha256": sha256(bank_path),
        "runtime": str(runtime_path),
        "runtime_sha256": sha256(runtime_path),
        "batch": args.batch,
        "unique_events": half,
        "steps": 719,
        "pollution_test": (
            "Each of 1024 independent events is duplicated in a separate batch lane; "
            "the complete terminal State and controller carry must match within each "
            "pair and across three fresh-reset invocations in one process."
        ),
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "results": rows,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": output["status"], "output": str(output_path)}))
    return 0 if output["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
