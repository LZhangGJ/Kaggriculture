"""Reproducible simulator-only GPU benchmark (compile time excluded)."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import (
    Events,
    batched_step_sync,
    empty_action,
    load_event_bank,
    load_tables,
    reset,
)


PROJECT = Path(__file__).resolve().parents[1]


def memory_stats() -> dict[str, int]:
    raw = jax.devices()[0].memory_stats() or {}
    return {key: int(value) for key, value in raw.items() if isinstance(value, int)}


def broadcast_action(batch_size: int):
    action = empty_action()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size, *value.shape)), action
    )


def benchmark_one(
    batch_size: int,
    rollout_steps: int,
    repetitions: int,
    event_seed_values: np.ndarray,
    event_bank: Events,
    tables,
) -> dict:
    event_indices = jnp.arange(batch_size) % event_bank.weed_spawn.shape[0]
    seed_values = jnp.asarray(event_seed_values, dtype=jnp.int32)[event_indices]
    states = jax.vmap(reset)(seed_values)
    actions = broadcast_action(batch_size)
    events = Events(
        weed_spawn=event_bank.weed_spawn[event_indices],
        shop_choice=event_bank.shop_choice[event_indices],
    )

    def rollout(initial, fixed_actions, fixed_events):
        def body(_, state):
            return batched_step_sync(state, fixed_actions, fixed_events, tables)

        return jax.lax.fori_loop(0, rollout_steps, body, initial)

    compiled = jax.jit(rollout)
    started = time.perf_counter()
    output = compiled(states, actions, events)
    jax.block_until_ready(output)
    compile_and_first = time.perf_counter() - started

    timings = []
    for _ in range(repetitions):
        started = time.perf_counter()
        output = compiled(states, actions, events)
        jax.block_until_ready(output)
        timings.append(time.perf_counter() - started)

    median = statistics.median(timings)
    transitions = batch_size * rollout_steps
    return {
        "batch_size": batch_size,
        "rollout_steps": rollout_steps,
        "transitions_per_run": transitions,
        "compile_and_first_s": compile_and_first,
        "steady_median_s": median,
        "steady_min_s": min(timings),
        "steady_max_s": max(timings),
        "repetitions": repetitions,
        "env_transitions_per_s": transitions / median,
        "player_samples_per_s": 2 * transitions / median,
        "state_bytes_per_environment": int(
            sum(np.asarray(value).nbytes for value in jax.tree.leaves(reset(0)))
        ),
        "terminal_step": int(jax.device_get(output.step[0])),
        "diagnostics": {
            "hand_cap_hits": int(jax.device_get(jnp.sum(output.hand_cap_hits))),
            "market_loop_cap_hits": int(
                jax.device_get(jnp.sum(output.market_loop_cap_hits))
            ),
            "price_lut_oob": int(jax.device_get(jnp.sum(output.price_lut_oob))),
        },
        "device_memory_stats": memory_stats(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument(
        "--receipt", type=Path, default=PROJECT / "receipts" / "benchmark_development.json"
    )
    args = parser.parse_args()

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU benchmark requested, backend={jax.default_backend()!r}")
    event_seeds, event_bank = load_event_bank()
    tables = load_tables()
    rows = [
        benchmark_one(
            batch_size,
            args.rollout_steps,
            args.repetitions,
            event_seeds,
            event_bank,
            tables,
        )
        for batch_size in args.batch_sizes
    ]
    for row in rows:
        row["minimum_gate_passed"] = row["env_transitions_per_s"] >= 50_000
        row["formal_target_passed"] = row["env_transitions_per_s"] >= 300_000
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    receipt = {
        "schema": "kaggriculture_simulator_benchmark_v1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "jax": importlib.metadata.version("jax"),
        "jaxlib": importlib.metadata.version("jaxlib"),
        "backend": jax.default_backend(),
        "gpu": gpu,
        "transition_definition": "one transition advances both players by one turn",
        "compilation_excluded_from_throughput": True,
        "rows": rows,
        "minimum_gate_env_transitions_per_s": 50_000,
        "formal_target_env_transitions_per_s": 300_000,
        "event_seed_count": int(len(event_seeds)),
        "minimum_gate_passed": all(row["minimum_gate_passed"] for row in rows),
        "formal_target_passed": any(row["formal_target_passed"] for row in rows),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
