"""Diagnostic GPU sweep of diverse tomato configurations on frozen seeds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
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
from project_route_search_v2.crop_rollout import (  # noqa: E402
    initialize_crop_rollout_carry_v2,
    make_m2_crop_rollout_v2,
    summarize_m2_crop_rollout_v2,
)
from project_route_search_v2.schema import CropProjectConfigV2  # noqa: E402


TOMATO_ID = 2


def _config(
    config_id: str,
    *,
    target_tiles: int = 8,
    seed_batch_size: int = 4,
    cash_floor: int = 500,
    investment_stop_step: int = 600,
    liquidation_start_step: int = 648,
    harvest_age_days: int = 8,
) -> dict:
    return {
        "config_id": config_id,
        "crop_id": TOMATO_ID,
        "target_tiles": target_tiles,
        "seed_batch_size": seed_batch_size,
        "cash_floor": cash_floor,
        "investment_stop_step": investment_stop_step,
        "liquidation_start_step": liquidation_start_step,
        "harvest_age_days": harvest_age_days,
    }


def tomato_configs_v1() -> list[dict]:
    """One-factor probes plus three combined high-capacity candidates."""

    configs = [
        _config("TARGET_02_BASE", target_tiles=2),
        _config("TARGET_04_BASE", target_tiles=4),
        _config("TARGET_08_BASE", target_tiles=8),
        _config("TARGET_12_BASE", target_tiles=12),
        _config("TARGET_16_BASE", target_tiles=16),
        _config("SEED_BATCH_01", seed_batch_size=1),
        _config("SEED_BATCH_02", seed_batch_size=2),
        _config("SEED_BATCH_08", seed_batch_size=8),
        _config("CASH_FLOOR_0000", cash_floor=0),
        _config("CASH_FLOOR_1000", cash_floor=1000),
        _config("CASH_FLOOR_2000", cash_floor=2000),
        _config("CASH_FLOOR_2800", cash_floor=2800),
        _config("CASH_FLOOR_2900", cash_floor=2900),
        _config("HARVEST_AGE_09", harvest_age_days=9),
        _config("HARVEST_AGE_10", harvest_age_days=10),
        _config("HARVEST_AGE_11", harvest_age_days=11),
        _config("HARVEST_AGE_12", harvest_age_days=12),
        _config(
            "TIMING_EARLY_432_552",
            investment_stop_step=432,
            liquidation_start_step=552,
        ),
        _config(
            "TIMING_MID_504_600",
            investment_stop_step=504,
            liquidation_start_step=600,
        ),
        _config(
            "TIMING_LATE_624_672",
            investment_stop_step=624,
            liquidation_start_step=672,
        ),
        _config(
            "TIMING_LATE_648_696",
            investment_stop_step=648,
            liquidation_start_step=696,
        ),
        _config(
            "COMBINED_T12_H11_LATE",
            target_tiles=12,
            seed_batch_size=8,
            harvest_age_days=11,
            investment_stop_step=624,
            liquidation_start_step=696,
        ),
        _config(
            "COMBINED_T16_H11_LATE",
            target_tiles=16,
            seed_batch_size=8,
            harvest_age_days=11,
            investment_stop_step=624,
            liquidation_start_step=696,
        ),
        _config(
            "COMBINED_T32_H11_LATE",
            target_tiles=32,
            seed_batch_size=16,
            harvest_age_days=11,
            investment_stop_step=624,
            liquidation_start_step=696,
        ),
    ]
    timing_grid = (
        (360, 504),
        (408, 528),
        (432, 552),
        (456, 576),
        (480, 600),
        (504, 624),
        (528, 648),
        (552, 672),
        (576, 696),
        (600, 696),
    )
    for target_tiles in (3, 4, 5, 6, 7, 8):
        for investment_stop_step, liquidation_start_step in timing_grid:
            configs.append(
                _config(
                    f"GRID_T{target_tiles:02d}_S{investment_stop_step}_L{liquidation_start_step}",
                    target_tiles=target_tiles,
                    investment_stop_step=investment_stop_step,
                    liquidation_start_step=liquidation_start_step,
                )
            )
    for target_tiles in (3, 4, 5):
        for investment_stop_step, liquidation_start_step in (
            (408, 528),
            (432, 552),
            (456, 576),
        ):
            for harvest_age_days in (8, 9, 10, 11):
                configs.append(
                    _config(
                        f"REFINE_T{target_tiles:02d}_S{investment_stop_step}_L{liquidation_start_step}_H{harvest_age_days:02d}",
                        target_tiles=target_tiles,
                        harvest_age_days=harvest_age_days,
                        investment_stop_step=investment_stop_step,
                        liquidation_start_step=liquidation_start_step,
                    )
                )
    assert len({row["config_id"] for row in configs}) == len(configs)
    return configs


def _events_by_indices(bank: Events, indices: np.ndarray) -> Events:
    return Events(
        weed_spawn=bank.weed_spawn[indices],
        shop_choice=bank.shop_choice[indices],
    )


def _percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.percentile(values, quantile))


def _aggregate(config: dict, arrays: dict[str, np.ndarray]) -> dict:
    bank = arrays["final_bank"].astype(np.int64)
    zero_fields = (
        "terminal_sellable_shed_value",
        "terminal_unit_inventory_value",
        "terminal_harvestable_map_value",
        "avoidable_liquidation_loss",
        "plant_without_same_day_water",
        "unexplained_failure_count",
    )
    checks = {
        "all_done": bool(np.all(arrays["done"])),
        **{
            f"{field}_zero": bool(np.all(arrays[field] == 0))
            for field in zero_fields
        },
        "all_completed_cash_cycle": bool(
            np.all(arrays["plant_success_count"] > 0)
            and np.all(arrays["harvest_success_count"] > 0)
            and np.all(arrays["deposit_success_count"] > 0)
            and np.all(arrays["sold_product_units"] > 0)
        ),
    }
    cash_cycle = (
        (arrays["plant_success_count"] > 0)
        & (arrays["harvest_success_count"] > 0)
        & (arrays["deposit_success_count"] > 0)
        & (arrays["sold_product_units"] > 0)
    )
    clean_terminal = np.logical_and.reduce(
        [arrays[field] == 0 for field in zero_fields]
    )
    detail_fields = (
        "invalid_raw_action_count",
        "unexpected_pass_count",
        "effect_mismatch_count",
        "owner_inactive_count",
        "deadline_missed_count",
        "resource_unavailable_count",
        "project_cap_hits",
        "obligation_cap_hits",
    )
    return {
        **config,
        "season_count": int(bank.size),
        "strict_closure_rate": float(
            np.mean(
                arrays["done"] & clean_terminal & cash_cycle
            )
        ),
        "cash_cycle_rate": float(np.mean(cash_cycle)),
        "checks": checks,
        "final_bank": {
            "minimum": int(np.min(bank)),
            "p10": _percentile(bank, 10),
            "p25": _percentile(bank, 25),
            "median": _percentile(bank, 50),
            "mean": float(np.mean(bank)),
            "p75": _percentile(bank, 75),
            "p90": _percentile(bank, 90),
            "maximum": int(np.max(bank)),
            "population_std": float(np.std(bank)),
        },
        "median_counts": {
            field: float(np.median(arrays[field]))
            for field in (
                "plant_success_count",
                "water_success_count",
                "harvest_success_count",
                "deposit_success_count",
                "sold_product_units",
                "terminal_harvestable_map_units",
            )
        },
        "failure_lane_count": {
            field: int(np.sum(arrays[field] != 0))
            for field in zero_fields
        },
        "detailed_diagnostics": {
            field: {
                "affected_lanes": int(
                    np.sum(arrays[f"metric_{field}"] != 0)
                ),
                "total_count": int(np.sum(arrays[f"metric_{field}"])),
                "maximum_per_lane": int(
                    np.max(arrays[f"metric_{field}"])
                ),
            }
            for field in detail_fields
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-per-config", type=int, default=100)
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
        default=PROJECT_DIR / "receipts" / "m2_tomato_parameter_sweep_v1.json",
    )
    args = parser.parse_args()
    if args.seeds_per_config <= 0 or args.seeds_per_config > 128:
        parser.error("seeds-per-config must be in 1..128")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    configs = tomato_configs_v1()
    event_bank_path = args.event_bank.resolve()
    source_seeds, bank = load_event_bank(event_bank_path)
    panel = json.loads(
        (PROJECT_DIR / "configs" / "seed_panels_v1.json").read_text(
            encoding="utf-8"
        )
    )
    panel_seeds = np.asarray(
        panel["panels"]["E0_AUDIT_128"]["seeds"][: args.seeds_per_config],
        dtype=np.int64,
    )
    source_index = {int(seed): index for index, seed in enumerate(source_seeds)}
    event_indices_one = np.asarray(
        [source_index[int(seed)] for seed in panel_seeds], dtype=np.int32
    )
    event_indices = np.tile(event_indices_one, len(configs))
    seeds = np.tile(panel_seeds, len(configs)).astype(np.int32)
    events = _events_by_indices(bank, event_indices)

    def repeated(field: str, dtype) -> jax.Array:
        values = np.repeat(
            np.asarray([row[field] for row in configs]), args.seeds_per_config
        )
        return jnp.asarray(values, dtype=dtype)

    config_batch = CropProjectConfigV2(
        crop_id=repeated("crop_id", jnp.int8),
        target_tiles=repeated("target_tiles", jnp.int16),
        seed_batch_size=repeated("seed_batch_size", jnp.int16),
        cash_floor=repeated("cash_floor", jnp.int32),
        investment_stop_step=repeated("investment_stop_step", jnp.int16),
        liquidation_start_step=repeated("liquidation_start_step", jnp.int16),
        harvest_age_days=repeated("harvest_age_days", jnp.int8),
    )
    carry = initialize_crop_rollout_carry_v2(
        jnp.asarray(seeds), config_batch, args.player
    )
    tables = load_tables()
    rollout = jax.jit(make_m2_crop_rollout_v2(rollout_steps=719, player=args.player))
    started = time.perf_counter()
    final = rollout(carry, events, tables, config_batch)
    jax.block_until_ready(final)
    elapsed = time.perf_counter() - started
    summary = jax.device_get(
        summarize_m2_crop_rollout_v2(final, config_batch, args.player)
    )
    all_arrays = {
        field: np.asarray(value) for field, value in summary._asdict().items()
    }
    all_arrays.update(
        {
            f"metric_{field}": np.asarray(value)
            for field, value in jax.device_get(final.metrics)._asdict().items()
        }
    )

    results = []
    count = args.seeds_per_config
    for index, config in enumerate(configs):
        start = index * count
        stop = start + count
        results.append(
            _aggregate(
                config,
                {field: value[start:stop] for field, value in all_arrays.items()},
            )
        )
    ranking = sorted(
        results,
        key=lambda row: (
            row["strict_closure_rate"],
            row["cash_cycle_rate"],
            row["final_bank"]["median"],
            row["final_bank"]["mean"],
        ),
        reverse=True,
    )
    ranking_rows = [
        {
            "rank": rank,
            "config_id": row["config_id"],
            "strict_closure_rate": row["strict_closure_rate"],
            "median_final_bank": row["final_bank"]["median"],
            "mean_final_bank": row["final_bank"]["mean"],
            "minimum_final_bank": row["final_bank"]["minimum"],
            "maximum_final_bank": row["final_bank"]["maximum"],
        }
        for rank, row in enumerate(ranking, start=1)
    ]

    source_paths = [
        Path(__file__).resolve(),
        PROJECT_DIR / "src" / "project_route_search_v2" / "crop_project.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "obligations.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "crop_executor.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "crop_rollout.py",
        PROJECT_DIR / "configs" / "seed_panels_v1.json",
        event_bank_path,
    ]
    receipt = {
        "receipt_id": "M2_TOMATO_PARAMETER_SWEEP_V1",
        "status": "DIAGNOSTIC_COMPLETE",
        "truth_boundary": "JAX_NULL_OPPONENT_DIAGNOSTIC_ONLY",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "compile_and_run_seconds": elapsed,
        },
        "experiment": {
            "crop": "TOMATO",
            "config_count": len(configs),
            "seeds_per_config": args.seeds_per_config,
            "total_seasons": len(configs) * args.seeds_per_config,
            "panel": f"E0_AUDIT_128_FIRST_{args.seeds_per_config}",
            "player": args.player,
            "opponent": "NULL_OPPONENT_V1",
        },
        "ranking": ranking_rows,
        "results": results,
        "not_claimed": [
            "official_python_parity",
            "competitive_strength",
            "global_parameter_optimum",
            "tomato_executor_formally_accepted",
        ],
        "files": {
            path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in source_paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "total_seasons": receipt["experiment"]["total_seasons"],
                "compile_and_run_seconds": elapsed,
                "top_10": ranking_rows[:10],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
