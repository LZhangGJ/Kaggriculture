"""Benchmark M2.5 and M2.6 under one frozen full-season GPU contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
for source_dir in (
    PROJECT_DIR / "src",
    REPO_ROOT / "gpu_sim" / "src",
    REPO_ROOT / "experiments" / "strategic_v5" / "src",
):
    sys.path.insert(0, str(source_dir))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kaggriculture_jax.state import load_event_bank, load_tables  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from project_route_search_v2.m25_controller import default_r2_tomato_m25_config_v2  # noqa: E402
from project_route_search_v2.m25_rollout import (  # noqa: E402
    initialize_m25_rollout_carry_v2, make_m25_crop_rollout_v2,
)
from project_route_search_v2.m26_genome import default_m26_crop_genome_v2  # noqa: E402
from project_route_search_v2.m26_rollout import (  # noqa: E402
    initialize_m26_rollout_carry_v2, make_m26_crop_rollout_v2,
)


def _measure(callable_, repetitions: int):
    compile_started = time.perf_counter()
    output = callable_()
    jax.block_until_ready(output)
    compile_seconds = time.perf_counter() - compile_started
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter()
        output = callable_()
        jax.block_until_ready(output)
        timings.append(time.perf_counter() - started)
    return compile_seconds, timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=719)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--event-bank", type=Path,
        default=PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_DIR / "receipts" / "m26_performance_comparison_v1.json",
    )
    args = parser.parse_args()
    if args.steps != 719:
        parser.error("formal M2.6 performance acceptance requires 719 steps")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    source_seeds, bank = load_event_bank(args.event_bank)
    event_indices = np.arange(args.batch, dtype=np.int32) % len(source_seeds)
    seeds = np.asarray(source_seeds, dtype=np.int64)[event_indices]
    events = Events(bank.weed_spawn[event_indices], bank.shop_choice[event_indices])
    tables = load_tables()

    m25_config = default_r2_tomato_m25_config_v2(args.batch)
    m25_carry = initialize_m25_rollout_carry_v2(
        jnp.asarray(seeds, dtype=jnp.int32), m25_config, player=0
    )
    m25_rollout = jax.jit(make_m25_crop_rollout_v2(rollout_steps=args.steps))
    m25_compile, m25_timings = _measure(
        lambda: m25_rollout(m25_carry, events, tables, m25_config), args.repetitions
    )

    m26_genome = default_m26_crop_genome_v2(args.batch)
    m26_carry = initialize_m26_rollout_carry_v2(
        jnp.asarray(seeds, dtype=jnp.int32), m26_genome, player=0
    )
    m26_rollout = jax.jit(make_m26_crop_rollout_v2(rollout_steps=args.steps))
    m26_compile, m26_timings = _measure(
        lambda: m26_rollout(m26_carry, events, tables, m26_genome), args.repetitions
    )

    transitions = args.batch * args.steps
    m25_median = statistics.median(m25_timings)
    m26_median = statistics.median(m26_timings)
    m25_throughput = transitions / m25_median
    m26_throughput = transitions / m26_median
    ratio = m26_throughput / m25_throughput
    checks = {
        "m26_at_least_60k": m26_throughput >= 60_000,
        "m26_at_least_70pct_same_contract_m25": ratio >= 0.70,
    }
    receipt = {
        "receipt_id": "M26_SAME_CONTRACT_M25_PERFORMANCE_COMPARISON_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "contract": {
            "batch": args.batch, "steps": args.steps,
            "transitions_per_run": transitions,
            "repetitions": args.repetitions,
            "same_process": True, "same_events": True, "trace_saved": False,
        },
        "m25": {
            "compile_and_first_seconds": m25_compile,
            "warm_seconds": m25_timings,
            "median_warm_seconds": m25_median,
            "transitions_per_second": m25_throughput,
        },
        "m26": {
            "compile_and_first_seconds": m26_compile,
            "warm_seconds": m26_timings,
            "median_warm_seconds": m26_median,
            "transitions_per_second": m26_throughput,
        },
        "m26_over_m25_ratio": ratio,
        "checks": checks,
        "boundary": "FULL_CONTROLLER_ROLLOUT_INCLUDES_M26_COVERAGE_COUNTERS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
