"""Run M2 100-season semantic acceptance and the RTX 3090 throughput gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
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
from project_route_search_v2.crop_project import (  # noqa: E402
    default_crop_project_config_v2,
)
from project_route_search_v2.crop_rollout import (  # noqa: E402
    initialize_crop_rollout_carry_v2,
    make_m2_crop_rollout_v2,
    summarize_m2_crop_rollout_v2,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _events_by_indices(bank: Events, indices: np.ndarray) -> Events:
    return Events(
        weed_spawn=bank.weed_spawn[indices],
        shop_choice=bank.shop_choice[indices],
    )


def _device_memory_stats() -> dict[str, int]:
    raw = jax.devices()[0].memory_stats() or {}
    return {
        str(key): int(value)
        for key, value in raw.items()
        if isinstance(value, (int, np.integer))
    }


def _nvidia_snapshot() -> str | None:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _semantic_rows(seeds: np.ndarray, summary) -> list[dict]:
    arrays = {name: np.asarray(value) for name, value in summary._asdict().items()}
    rows = []
    for index, seed in enumerate(seeds):
        rows.append(
            {
                "seed": int(seed),
                **{
                    name: (
                        bool(value[index])
                        if value.dtype == np.bool_
                        else int(value[index])
                    )
                    for name, value in arrays.items()
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-seasons", type=int, default=100)
    parser.add_argument("--performance-batch", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=719)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--crop-id", type=int, default=0)
    parser.add_argument("--target-tiles", type=int, default=4)
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
        default=PROJECT_DIR / "receipts" / "m2_crop_acceptance_v1.json",
    )
    args = parser.parse_args()
    if args.semantic_seasons < 100:
        parser.error("M2 acceptance requires at least 100 semantic seasons")
    if args.performance_batch != 2048:
        parser.error("M2 performance contract is frozen at batch 2048")
    if args.steps != 719:
        parser.error("M2 performance contract is frozen at 719 steps")
    if args.repetitions < 5:
        parser.error("M2 performance contract requires at least five repetitions")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(
            f"formal M2 performance requires GPU, got {jax.default_backend()}"
        )

    event_bank_path = args.event_bank.resolve()
    source_seeds, bank = load_event_bank(event_bank_path)
    panel = _load_json(PROJECT_DIR / "configs" / "seed_panels_v1.json")
    audit_seeds = np.asarray(
        panel["panels"]["E0_AUDIT_128"]["seeds"], dtype=np.int64
    )
    if args.semantic_seasons > len(audit_seeds):
        raise ValueError("semantic season count exceeds frozen E0_AUDIT_128")
    semantic_seeds = audit_seeds[: args.semantic_seasons]
    source_index = {int(seed): index for index, seed in enumerate(source_seeds)}
    semantic_indices = np.asarray(
        [source_index[int(seed)] for seed in semantic_seeds], dtype=np.int32
    )
    semantic_events = _events_by_indices(bank, semantic_indices)
    semantic_config = default_crop_project_config_v2(
        args.semantic_seasons,
        crop_id=args.crop_id,
        target_tiles=args.target_tiles,
    )
    semantic_carry = initialize_crop_rollout_carry_v2(
        jnp.asarray(semantic_seeds, dtype=jnp.int32),
        semantic_config,
        args.player,
    )
    tables = load_tables()
    rollout = make_m2_crop_rollout_v2(
        rollout_steps=args.steps, player=args.player
    )
    semantic_compiled = jax.jit(rollout)
    semantic_start = time.perf_counter()
    semantic_final = semantic_compiled(
        semantic_carry, semantic_events, tables, semantic_config
    )
    jax.block_until_ready(semantic_final)
    semantic_seconds = time.perf_counter() - semantic_start
    semantic_summary = summarize_m2_crop_rollout_v2(
        semantic_final, semantic_config, args.player
    )
    semantic_summary = jax.device_get(semantic_summary)
    rows = _semantic_rows(semantic_seeds, semantic_summary)
    semantic_checks = {
        "all_done": all(row["done"] for row in rows),
        "terminal_sellable_shed_value_zero": all(
            row["terminal_sellable_shed_value"] == 0 for row in rows
        ),
        "avoidable_liquidation_loss_zero": all(
            row["avoidable_liquidation_loss"] == 0 for row in rows
        ),
        "plant_without_same_day_water_zero": all(
            row["plant_without_same_day_water"] == 0 for row in rows
        ),
        "unexplained_failure_count_zero": all(
            row["unexplained_failure_count"] == 0 for row in rows
        ),
        "all_routes_completed_cash_cycle": all(
            row["plant_success_count"] > 0
            and row["harvest_success_count"] > 0
            and row["deposit_success_count"] > 0
            and row["sold_product_units"] > 0
            and row["final_bank"] > 3000
            for row in rows
        ),
    }
    semantic_pass = all(semantic_checks.values())

    performance_indices = (
        np.arange(args.performance_batch, dtype=np.int32) % len(source_seeds)
    )
    performance_seeds = np.asarray(source_seeds, dtype=np.int64)[performance_indices]
    performance_events = _events_by_indices(bank, performance_indices)
    performance_config = default_crop_project_config_v2(
        args.performance_batch,
        crop_id=args.crop_id,
        target_tiles=args.target_tiles,
    )
    performance_carry = initialize_crop_rollout_carry_v2(
        jnp.asarray(performance_seeds, dtype=jnp.int32),
        performance_config,
        args.player,
    )
    performance_compiled = jax.jit(rollout)
    compile_start = time.perf_counter()
    performance_output = performance_compiled(
        performance_carry, performance_events, tables, performance_config
    )
    jax.block_until_ready(performance_output)
    compile_and_first = time.perf_counter() - compile_start
    timings = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        performance_output = performance_compiled(
            performance_carry, performance_events, tables, performance_config
        )
        jax.block_until_ready(performance_output)
        timings.append(time.perf_counter() - started)
    median_seconds = statistics.median(timings)
    transitions = args.performance_batch * args.steps
    transitions_per_second = transitions / median_seconds
    device_memory_stats = _device_memory_stats()
    peak_device_bytes = int(
        device_memory_stats.get(
            "peak_bytes_in_use", device_memory_stats.get("bytes_in_use", 0)
        )
    )
    peak_device_limit_bytes = 22 * 1024**3
    gates = _load_json(PROJECT_DIR / "configs" / "performance_gates_v1.json")
    baseline = float(gates["comparison_baseline"]["transitions_per_second"])
    stage_gate = gates["milestone_gates"]["M2_CROP"]
    absolute_minimum = float(stage_gate["minimum_transitions_per_second"])
    relative_minimum = float(stage_gate["minimum_baseline_fraction"])
    relative_ratio = transitions_per_second / baseline
    performance_checks = {
        "backend_is_gpu": jax.default_backend() == "gpu",
        "batch_is_2048": args.performance_batch == 2048,
        "steps_are_719": args.steps == 719,
        "five_or_more_warm_repetitions": len(timings) >= 5,
        "absolute_60k_gate": transitions_per_second >= absolute_minimum,
        "relative_70pct_dynamic_v2_gate": relative_ratio >= relative_minimum,
        "peak_device_memory_under_22gb": (
            peak_device_bytes > 0 and peak_device_bytes < peak_device_limit_bytes
        ),
    }
    performance_pass = all(performance_checks.values())

    source_paths = [
        REPO_ROOT
        / "gpt_review"
        / "gpt"
        / "KAGGRICULTURE_E0_THREE_LAYER_PROJECT_SEARCH_DESIGN_V1_1_ZH.md",
        PROJECT_DIR / "configs" / "performance_gates_v1.json",
        PROJECT_DIR / "configs" / "seed_panels_v1.json",
        PROJECT_DIR / "src" / "project_route_search_v2" / "crop_project.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "obligations.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "crop_executor.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "crop_rollout.py",
        REPO_ROOT / "experiments" / "strategic_v5" / "src" / "strategic_v5" / "e4_executor.py",
        REPO_ROOT / "gpu_sim" / "src" / "kaggriculture_jax" / "simulator.py",
        PROJECT_DIR / "tests" / "test_crop_loop.py",
        Path(__file__).resolve(),
        event_bank_path,
    ]
    receipt = {
        "receipt_id": "M2_CROP_ACCEPTANCE_V1",
        "status": "PASS" if semantic_pass and performance_pass else "FAIL",
        "scope": "M2_MINIMAL_CROP_BUSINESS_LOOP",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "nvidia": _nvidia_snapshot(),
        },
        "semantic": {
            "season_count": args.semantic_seasons,
            "panel": "E0_AUDIT_128_FIRST_100",
            "player": args.player,
            "crop_id": args.crop_id,
            "target_tiles": args.target_tiles,
            "compile_and_run_seconds": semantic_seconds,
            "checks": semantic_checks,
            "final_bank": {
                "minimum": min(row["final_bank"] for row in rows),
                "median": statistics.median(row["final_bank"] for row in rows),
                "maximum": max(row["final_bank"] for row in rows),
            },
            "per_seed": rows,
        },
        "performance": {
            "contract": {
                "batch_size": args.performance_batch,
                "steps": args.steps,
                "repetitions": args.repetitions,
                "compilation_excluded": True,
                "absolute_minimum_transitions_per_second": absolute_minimum,
                "comparison_baseline_transitions_per_second": baseline,
                "relative_minimum_fraction": relative_minimum,
                "peak_device_memory_limit_bytes": peak_device_limit_bytes,
            },
            "compile_and_first_seconds": compile_and_first,
            "warm_seconds": timings,
            "median_warm_seconds": median_seconds,
            "transitions_per_second": transitions_per_second,
            "relative_to_dynamic_v2": relative_ratio,
            "checks": performance_checks,
            "device_memory_stats": device_memory_stats,
        },
        "not_claimed": [
            "crop_strategy_is_high_profit",
            "all_five_crop_species_accepted",
            "official_python_step_parity_completed",
            "competitive_strength_validated",
            "route_search_run",
        ],
        "files": {
            _relative(path): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in source_paths
        },
    }
    _write_json(args.output.resolve(), receipt)
    print(
        json.dumps(
            {
                "receipt_id": receipt["receipt_id"],
                "status": receipt["status"],
                "semantic_checks": semantic_checks,
                "performance_checks": performance_checks,
                "transitions_per_second": transitions_per_second,
                "peak_device_bytes": peak_device_bytes,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
