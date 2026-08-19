"""Run 256 structured-random M3.5 full-season safety rollouts on GPU."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
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
from project_route_search_v2.m35_genome import validate_m35_farm_genome_v2  # noqa: E402
from project_route_search_v2.m35_rollout import (  # noqa: E402
    initialize_m35_rollout_carry_v2,
    make_m35_farm_rollout_v2,
    summarize_m35_rollout_v2,
)
from project_route_search_v2.m35_search import sample_m35_farm_genomes_v2  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    count = 256
    sampler_seed = 350719
    event_bank = PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz"
    output = PROJECT_DIR / "receipts" / "m35_random_safety_v1.json"
    source_seeds, bank = load_event_bank(event_bank)
    indices = np.arange(count, dtype=np.int32) % len(source_seeds)
    seeds = np.asarray(source_seeds, dtype=np.int64)[indices]
    events = Events(bank.weed_spawn[indices], bank.shop_choice[indices])
    genome = sample_m35_farm_genomes_v2(count, seed=sampler_seed)
    validation_errors = validate_m35_farm_genome_v2(genome)
    carry = initialize_m35_rollout_carry_v2(jnp.asarray(seeds, dtype=jnp.int32), genome)
    rollout = jax.jit(make_m35_farm_rollout_v2())
    started = time.perf_counter()
    final, _ = rollout(carry, events, load_tables(), genome)
    jax.block_until_ready(final)
    elapsed = time.perf_counter() - started
    summary = jax.device_get(summarize_m35_rollout_v2(final))
    values = {
        key: np.asarray(value)
        for key, value in summary._asdict().items()
        if key not in ("flow", "crop_coverage", "animal_coverage")
    }
    flow = {
        key: np.asarray(value)
        for key, value in jax.device_get(summary.flow)._asdict().items()
    }
    crop = {
        key: np.asarray(value)
        for key, value in jax.device_get(summary.crop_coverage)._asdict().items()
    }
    animal = {
        key: np.asarray(value)
        for key, value in jax.device_get(summary.animal_coverage)._asdict().items()
    }
    hard_zero_fields = (
        "plant_without_same_day_water",
        "unexplained_failure_count",
        "unplanned_animal_escape",
        "animal_capacity_loss",
        "feed_hard_deadline_miss",
        "care_bonus_forfeited_unexplained",
        "care_bonus_capacity_clipped_unexplained",
        "animal_bought_without_place_plan",
        "cow_sheep_pasture_conflict",
        "animals_stranded_in_shed_at_terminal",
        "animals_stranded_in_unit_inventory_at_terminal",
    )
    economic_diagnostic_fields = (
        "terminal_sellable_shed_value",
        "terminal_unit_inventory_value",
        "terminal_harvestable_crop_value",
        "terminal_animal_product_value",
        "avoidable_liquidation_loss",
    )
    signature = np.concatenate(
        (
            crop["crop_plant_actions"],
            crop["crop_harvest_actions"],
            animal["build_actions"],
            animal["purchase_orders"],
            animal["feed_actions"],
            np.stack(tuple(flow.values()), axis=-1),
        ),
        axis=-1,
    )
    unique_signatures = int(np.unique(signature, axis=0).shape[0])
    checks = {
        "genome_validation_pass": not validation_errors,
        "all_256_complete": bool(np.all(values["done"])),
        "all_correctness_diagnostics_zero": all(
            int(np.sum(values[field])) == 0 for field in hard_zero_fields
        ),
        "at_least_24_behavior_signatures": unique_signatures >= 24,
        "all_feed_modes_sampled": int(np.unique(np.asarray(genome.animal.feed_source_policy)).size) == 3,
        "both_recycle_modes_sampled": int(
            np.unique(np.asarray(genome.animal.animal_fertilizer_policy)).size
        ) == 2,
        "joint_resource_conflicts_zero": int(
            np.sum(flow["joint_resource_conflict_count"])
        ) == 0,
    }
    failing_lanes = sorted(
        set(
            np.flatnonzero(
                np.logical_or.reduce(
                    tuple(values[field] != 0 for field in hard_zero_fields)
                )
            ).astype(int).tolist()
        )
    )
    economic_residual_lanes = sorted(
        set(
            np.flatnonzero(
                np.logical_or.reduce(
                    tuple(values[field] != 0 for field in economic_diagnostic_fields)
                )
            ).astype(int).tolist()
        )
    )
    failure_rows = [
        {
            "lane": lane,
            "feed_source_policy": int(np.asarray(genome.animal.feed_source_policy)[lane]),
            "fertilizer_policy": int(
                np.asarray(genome.animal.animal_fertilizer_policy)[lane]
            ),
            "crop_unit_share": float(np.asarray(genome.crop_unit_share)[lane]),
            "liquidation_start_step": int(
                np.asarray(genome.animal.liquidation_start_step)[lane]
            ),
            "final_bank": int(values["final_bank"][lane]),
            "nonzero": {
                field: int(values[field][lane])
                for field in hard_zero_fields
                if int(values[field][lane]) != 0
            },
            "terminal_components": {
                field: int(values[field][lane])
                for field in (
                    "terminal_sellable_shed_value",
                    "terminal_unit_inventory_value",
                    "terminal_harvestable_crop_value",
                    "terminal_animal_product_value",
                )
                if int(values[field][lane]) != 0
            },
        }
        for lane in failing_lanes
    ]
    receipt = {
        "receipt_id": "M35_STRUCTURED_RANDOM_256_FULL_SEASON_SAFETY_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "candidate_count": count,
        "steps_per_candidate": 719,
        "transitions": count * 719,
        "compile_and_execute_seconds": elapsed,
        "sampler_seed": sampler_seed,
        "validation_errors": validation_errors,
        "checks": checks,
        "unique_behavior_signatures": unique_signatures,
        "failing_lanes": failing_lanes,
        "failure_rows": failure_rows,
        "hard_metric_totals": {
            field: int(np.sum(values[field])) for field in hard_zero_fields
        },
        "economic_diagnostic_totals": {
            field: int(np.sum(values[field]))
            for field in economic_diagnostic_fields
        },
        "economic_residual_lanes": economic_residual_lanes,
        "final_bank": {
            "min": int(np.min(values["final_bank"])),
            "median": float(np.median(values["final_bank"])),
            "max": int(np.max(values["final_bank"])),
        },
        "source_sha256": {
            path.relative_to(REPO_ROOT).as_posix(): _sha(path)
            for path in (
                PROJECT_DIR / "src" / "project_route_search_v2" / "m35_search.py",
                PROJECT_DIR / "src" / "project_route_search_v2" / "m35_controller.py",
                PROJECT_DIR / "src" / "project_route_search_v2" / "m35_rollout.py",
            )
        },
        "event_bank_sha256": _sha(event_bank),
        "objective_policy": {
            "primary": "MAXIMIZE_FINAL_BANK",
            "hard_failures": "RULE_LEDGER_AND_UNPLANNED_EXECUTION_ERRORS_ONLY",
            "terminal_residuals": "SOFT_ECONOMIC_DIAGNOSTICS_NOT_ACCEPTANCE_FAILURES",
        },
        "boundary": "STRUCTURED_SAFE_DOMAIN_NOT_UNCONSTRAINED_ROUTE_SEARCH_OR_ROUTE_QUALITY",
    }
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
