"""Benchmark M3.5 against frozen M3 under one full-season GPU contract."""

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
from project_route_search_v2.m3_genome import default_m3_animal_genome_v2  # noqa: E402
from project_route_search_v2.m3_rollout import (  # noqa: E402
    initialize_m3_rollout_carry_v2,
    make_m3_animal_rollout_v2,
)
from project_route_search_v2.m35_genome import default_m35_farm_genome_v2  # noqa: E402
from project_route_search_v2.m35_rollout import (  # noqa: E402
    initialize_m35_rollout_carry_v2,
    make_m35_farm_rollout_v2,
)


def _measure(callable_, repetitions: int):
    started = time.perf_counter()
    output = callable_()
    jax.block_until_ready(output)
    compile_seconds = time.perf_counter() - started
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
        "--event-bank",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "receipts" / "m35_performance_comparison_v1.json",
    )
    args = parser.parse_args()
    if args.steps != 719:
        parser.error("formal M3.5 performance acceptance requires 719 steps")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    source_seeds, bank = load_event_bank(args.event_bank)
    event_indices = np.arange(args.batch, dtype=np.int32) % len(source_seeds)
    seeds = np.asarray(source_seeds, dtype=np.int64)[event_indices]
    events = Events(bank.weed_spawn[event_indices], bank.shop_choice[event_indices])
    tables = load_tables()

    m3_genome = default_m3_animal_genome_v2(args.batch)
    m3_carry = initialize_m3_rollout_carry_v2(
        jnp.asarray(seeds, dtype=jnp.int32), m3_genome
    )
    m3_rollout = jax.jit(make_m3_animal_rollout_v2(rollout_steps=args.steps))
    m3_compile, m3_timings = _measure(
        lambda: m3_rollout(m3_carry, events, tables, m3_genome), args.repetitions
    )

    m35_genome = default_m35_farm_genome_v2(args.batch)
    m35_carry = initialize_m35_rollout_carry_v2(
        jnp.asarray(seeds, dtype=jnp.int32), m35_genome
    )
    m35_rollout = jax.jit(make_m35_farm_rollout_v2(rollout_steps=args.steps))
    m35_compile, m35_timings = _measure(
        lambda: m35_rollout(m35_carry, events, tables, m35_genome), args.repetitions
    )

    transitions = args.batch * args.steps
    m3_median = statistics.median(m3_timings)
    m35_median = statistics.median(m35_timings)
    m3_tps = transitions / m3_median
    m35_tps = transitions / m35_median
    ratio = m35_tps / m3_tps
    checks = {
        "m35_at_least_100k": m35_tps >= 100_000,
        "m35_at_least_60pct_same_contract_m3": ratio >= 0.60,
    }
    receipt = {
        "receipt_id": "M35_SAME_CONTRACT_M3_PERFORMANCE_COMPARISON_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "contract": {
            "batch": args.batch,
            "steps": args.steps,
            "transitions_per_run": transitions,
            "repetitions": args.repetitions,
            "same_process": True,
            "same_events": True,
            "trace_saved": False,
        },
        "m3": {
            "compile_and_first_seconds": m3_compile,
            "warm_seconds": m3_timings,
            "median_warm_seconds": m3_median,
            "transitions_per_second": m3_tps,
        },
        "m35": {
            "compile_and_first_seconds": m35_compile,
            "warm_seconds": m35_timings,
            "median_warm_seconds": m35_median,
            "transitions_per_second": m35_tps,
        },
        "m35_over_m3_ratio": ratio,
        "checks": checks,
        "boundary": "FULL_CONTROLLER_ROLLOUT_INCLUDES_M35_COVERAGE_AND_FLOW_COUNTERS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
