"""Benchmark GPU-resident policy+sim rollout and full PPO iterations."""

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

from kaggriculture_jax import (
    Events,
    PPOConfig,
    PolicyValueNet,
    create_train_state,
    encode_observations,
    generalized_advantage_estimate,
    load_event_bank,
    load_tables,
    make_arena_rollout,
    make_ppo_update,
    make_selfplay_collector,
    reset,
)


PROJECT = Path(__file__).resolve().parents[1]


def memory_stats() -> dict[str, int]:
    raw = jax.devices()[0].memory_stats() or {}
    return {key: int(value) for key, value in raw.items() if isinstance(value, int)}


def batch_inputs(batch_size: int, seed_values, bank):
    indices = jnp.arange(batch_size) % len(seed_values)
    seeds = jnp.asarray(seed_values, dtype=jnp.int32)[indices]
    states = jax.vmap(reset)(seeds)
    events = Events(bank.weed_spawn[indices], bank.shop_choice[indices])
    return states, events


def timed(function, arguments, repetitions: int):
    started = time.perf_counter()
    output = function(*arguments)
    jax.block_until_ready(output)
    compile_and_first = time.perf_counter() - started
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter()
        output = function(*arguments)
        jax.block_until_ready(output)
        timings.append(time.perf_counter() - started)
    return output, compile_and_first, timings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-batches", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--ppo-batches", nargs="+", type=int, default=[256, 1024])
    parser.add_argument("--policy-steps", type=int, default=64)
    parser.add_argument("--ppo-steps", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=PROJECT / "receipts" / "benchmark_policy_and_ppo.json",
    )
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected GPU backend, got {jax.default_backend()}")

    seed_values, bank = load_event_bank()
    tables = load_tables()
    model = PolicyValueNet(hidden_sizes=(128, 128))
    sample = encode_observations(reset(0))[0:1]
    params = model.init(jax.random.key(100), sample)
    parameter_count = sum(value.size for value in jax.tree.leaves(params))

    policy_rows = []
    arena_function = make_arena_rollout(
        model.apply,
        model.apply,
        rollout_steps=args.policy_steps,
        deterministic=True,
    )
    for batch_size in args.policy_batches:
        states, events = batch_inputs(batch_size, seed_values, bank)
        compiled = jax.jit(arena_function)
        result, compile_first, timings = timed(
            compiled,
            (states, events, tables, params, params, jax.random.key(101)),
            args.repetitions,
        )
        transitions = batch_size * args.policy_steps
        median = statistics.median(timings)
        policy_rows.append(
            {
                "batch_size": batch_size,
                "rollout_steps": args.policy_steps,
                "compile_and_first_s": compile_first,
                "steady_median_s": median,
                "steady_min_s": min(timings),
                "repetitions": args.repetitions,
                "env_transitions_per_s": transitions / median,
                "player_samples_per_s": 2 * transitions / median,
                "terminal_step": int(jax.device_get(result.final_state.step[0])),
                "device_memory_stats": memory_stats(),
            }
        )

    config = PPOConfig()
    ppo_rows = []
    collector_fn = make_selfplay_collector(
        model.apply, rollout_steps=args.ppo_steps, deterministic=False
    )
    update_fn = make_ppo_update(model.apply, config)
    for batch_size in args.ppo_batches:
        states, events = batch_inputs(batch_size, seed_values, bank)
        training_state = create_train_state(model.apply, params, config)

        def iteration(current_train_state, initial_states, event_rows, key):
            rollout = collector_fn(
                initial_states,
                event_rows,
                tables,
                current_train_state.params,
                key,
            )
            advantages, returns = generalized_advantage_estimate(
                rollout.transitions,
                rollout.bootstrap_value,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
            next_train_state, metrics = update_fn(
                current_train_state,
                rollout.transitions,
                advantages,
                returns,
            )
            return next_train_state, metrics, rollout.final_state

        compiled_iteration = jax.jit(iteration)
        result, compile_first, timings = timed(
            compiled_iteration,
            (training_state, states, events, jax.random.key(200 + batch_size)),
            args.repetitions,
        )
        transitions = batch_size * args.ppo_steps
        median = statistics.median(timings)
        ppo_rows.append(
            {
                "batch_size": batch_size,
                "rollout_steps": args.ppo_steps,
                "compile_and_first_s": compile_first,
                "steady_median_s": median,
                "steady_min_s": min(timings),
                "repetitions": args.repetitions,
                "env_transitions_per_s": transitions / median,
                "player_samples_per_s": 2 * transitions / median,
                "optimizer_step": int(jax.device_get(result[0].step)),
                "loss": float(jax.device_get(result[1].loss)),
                "device_memory_stats": memory_stats(),
            }
        )

    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    receipt = {
        "schema": "kaggriculture_policy_ppo_benchmark_v1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "jax": importlib.metadata.version("jax"),
        "backend": jax.default_backend(),
        "model": {
            "hidden_sizes": [128, 128],
            "parameter_count": parameter_count,
            "observation_size": int(sample.shape[-1]),
            "policy_market_slots": 2,
        },
        "compilation_excluded_from_throughput": True,
        "transition_definition": "one transition advances both players by one turn",
        "policy_plus_sim": policy_rows,
        "full_ppo_iteration": ppo_rows,
    }
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

