"""Run the M2.5 staged R2 crop controller on frozen seeds and benchmark it."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from project_route_search_v2.m25_controller import (  # noqa: E402
    default_r2_tomato_m25_config_v2,
)
from project_route_search_v2.m25_rollout import (  # noqa: E402
    initialize_m25_rollout_carry_v2,
    make_m25_crop_rollout_v2,
    summarize_m25_rollout_v2,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _events(bank: Events, indices: np.ndarray) -> Events:
    return Events(bank.weed_spawn[indices], bank.shop_choice[indices])


def _rows(seeds: np.ndarray, summary) -> list[dict]:
    arrays = {name: np.asarray(value) for name, value in summary._asdict().items()}
    return [
        {
            "seed": int(seed),
            **{
                name: bool(value[index]) if value.dtype == np.bool_ else int(value[index])
                for name, value in arrays.items()
            },
        }
        for index, seed in enumerate(seeds)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-seasons", type=int, default=100)
    parser.add_argument(
        "--semantic-seeds",
        type=str,
        default=None,
        help="Optional comma-separated frozen seeds for diagnostic replay.",
    )
    parser.add_argument("--performance-batch", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=719)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--event-bank",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "receipts" / "m25_acceptance_v1.json",
    )
    args = parser.parse_args()
    if args.steps != 719:
        parser.error("M2.5 acceptance requires the full 719-step season")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    source_seeds, bank = load_event_bank(args.event_bank)
    panel = json.loads(
        (PROJECT_DIR / "configs" / "seed_panels_v1.json").read_text(encoding="utf-8")
    )
    audit = np.asarray(panel["panels"]["E0_AUDIT_128"]["seeds"], dtype=np.int64)
    if args.semantic_seeds:
        semantic_seeds = np.asarray(
            [int(value.strip()) for value in args.semantic_seeds.split(",") if value.strip()],
            dtype=np.int64,
        )
        args.semantic_seasons = len(semantic_seeds)
    else:
        semantic_seeds = audit[: args.semantic_seasons]
    index = {int(seed): i for i, seed in enumerate(source_seeds)}
    semantic_indices = np.asarray([index[int(seed)] for seed in semantic_seeds], dtype=np.int32)
    semantic_events = _events(bank, semantic_indices)
    semantic_config = default_r2_tomato_m25_config_v2(args.semantic_seasons)
    semantic_carry = initialize_m25_rollout_carry_v2(
        jnp.asarray(semantic_seeds, dtype=jnp.int32), semantic_config, player=args.player
    )
    tables = load_tables()
    rollout = jax.jit(make_m25_crop_rollout_v2(player=args.player))
    started = time.perf_counter()
    semantic_final = rollout(semantic_carry, semantic_events, tables, semantic_config)
    jax.block_until_ready(semantic_final)
    semantic_seconds = time.perf_counter() - started
    summary = jax.device_get(
        summarize_m25_rollout_v2(semantic_final, semantic_config, player=args.player)
    )
    rows = _rows(semantic_seeds, summary)
    metric_rows = _rows(semantic_seeds, jax.device_get(semantic_final.metrics))
    hard_checks = {
        "all_done": all(row["done"] for row in rows),
        "unexplained_failure_zero": all(row["unexplained_failure_count"] == 0 for row in rows),
        "same_day_water_zero": all(row["plant_without_same_day_water"] == 0 for row in rows),
        "land_expansion_executed": all(row["final_unlocked_count"] >= 2 for row in rows),
        "multi_unit_executed": all(row["max_hires_observed"] >= 9 for row in rows),
        "cash_cycle_executed": all(
            row["plant_success_count"] > 0
            and row["harvest_success_count"] > 0
            and row["sold_product_units"] > 0
            for row in rows
        ),
    }

    perf_indices = np.arange(args.performance_batch, dtype=np.int32) % len(source_seeds)
    perf_seeds = np.asarray(source_seeds, dtype=np.int64)[perf_indices]
    perf_events = _events(bank, perf_indices)
    perf_config = default_r2_tomato_m25_config_v2(args.performance_batch)
    perf_carry = initialize_m25_rollout_carry_v2(
        jnp.asarray(perf_seeds, dtype=jnp.int32), perf_config, player=args.player
    )
    perf_rollout = jax.jit(make_m25_crop_rollout_v2(player=args.player))
    compile_start = time.perf_counter()
    perf_final = perf_rollout(perf_carry, perf_events, tables, perf_config)
    jax.block_until_ready(perf_final)
    compile_seconds = time.perf_counter() - compile_start
    timings = []
    for _ in range(args.repetitions):
        run_start = time.perf_counter()
        perf_final = perf_rollout(perf_carry, perf_events, tables, perf_config)
        jax.block_until_ready(perf_final)
        timings.append(time.perf_counter() - run_start)
    median_seconds = statistics.median(timings)
    throughput = args.performance_batch * args.steps / median_seconds

    source_paths = [
        PROJECT_DIR / "src" / "project_route_search_v2" / "m25_controller.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "m25_rollout.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "schema.py",
        Path(__file__).resolve(),
        args.event_bank.resolve(),
    ]
    banks = [row["final_bank"] for row in rows]
    receipt = {
        "receipt_id": "M25_STAGED_R2_CROP_ACCEPTANCE_V1",
        "status": "PASS" if all(hard_checks.values()) else "FAIL",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "semantic": {
            "seasons": args.semantic_seasons,
            "compile_and_run_seconds": semantic_seconds,
            "checks": hard_checks,
            "final_bank": {
                "minimum": min(banks),
                "p10": float(np.percentile(banks, 10)),
                "median": statistics.median(banks),
                "mean": statistics.mean(banks),
                "p90": float(np.percentile(banks, 90)),
                "maximum": max(banks),
            },
            "per_seed": rows,
            "per_seed_metrics": metric_rows,
        },
        "performance": {
            "batch": args.performance_batch,
            "steps": args.steps,
            "compile_and_first_seconds": compile_seconds,
            "warm_seconds": timings,
            "median_warm_seconds": median_seconds,
            "transitions_per_second": throughput,
        },
        "truth_boundary": "JAX_NULL_OPPONENT_DIAGNOSTIC_PENDING_OFFICIAL_AGENT_PARITY",
        "files": {
            path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(): {
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
            }
            for path in source_paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "checks": hard_checks,
                "final_bank": receipt["semantic"]["final_bank"],
                "transitions_per_second": throughput,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
