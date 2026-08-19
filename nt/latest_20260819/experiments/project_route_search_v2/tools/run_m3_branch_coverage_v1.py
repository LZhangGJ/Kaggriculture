"""Run the complete-season M3A/M3B major-branch acceptance panel."""

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
from project_route_search_v2.m3_candidates import m3_branch_coverage_panel_v2  # noqa: E402
from project_route_search_v2.m3_genome import validate_m3_animal_genome_v2  # noqa: E402
from project_route_search_v2.m3_rollout import (  # noqa: E402
    initialize_m3_rollout_carry_v2,
    make_m3_animal_rollout_v2,
    summarize_m3_rollout_v2,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    event_bank = PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz"
    output = PROJECT_DIR / "receipts" / "m3_branch_coverage_v1.json"
    source_seeds, bank = load_event_bank(event_bank)
    genome, names = m3_branch_coverage_panel_v2()
    validation_errors = validate_m3_animal_genome_v2(genome)
    indices = np.arange(len(names), dtype=np.int32)
    seeds = np.asarray(source_seeds, dtype=np.int64)[indices]
    events = Events(bank.weed_spawn[indices], bank.shop_choice[indices])
    carry = initialize_m3_rollout_carry_v2(
        jnp.asarray(seeds, dtype=jnp.int32), genome
    )
    rollout = jax.jit(make_m3_animal_rollout_v2())
    started = time.perf_counter()
    final, _ = rollout(carry, events, load_tables(), genome)
    jax.block_until_ready(final)
    elapsed = time.perf_counter() - started
    summary = jax.device_get(summarize_m3_rollout_v2(final))
    metrics = {
        key: np.asarray(value)
        for key, value in jax.device_get(final.metrics)._asdict().items()
    }
    values = {
        key: np.asarray(value)
        for key, value in summary._asdict().items()
        if key != "coverage"
    }
    coverage = {
        key: np.asarray(value)
        for key, value in summary.coverage._asdict().items()
    }
    hard_zero_fields = (
        "unexplained_failure_count",
        "unplanned_animal_escape",
        "animal_capacity_loss",
        "feed_hard_deadline_miss",
        "care_bonus_forfeited_unexplained",
        "care_bonus_capacity_clipped_unexplained",
        "animal_bought_without_place_plan",
        "cow_sheep_pasture_conflict",
        "terminal_sellable_shed_value",
        "terminal_unit_inventory_value",
        "avoidable_liquidation_loss",
        "animals_stranded_in_shed_at_terminal",
        "animals_stranded_in_unit_inventory_at_terminal",
    )
    checks = {
        "genome_validation_pass": not validation_errors,
        "all_18_candidates_done": bool(np.all(values["done"])),
        "all_hard_diagnostics_zero": all(
            int(np.sum(values[field])) == 0 for field in hard_zero_fields
        ),
        "all_species_built": bool(np.all(np.sum(coverage["build_actions"], axis=0) > 0)),
        "all_species_purchased": bool(
            np.all(np.sum(coverage["purchase_orders"], axis=0) > 0)
        ),
        "all_species_placed": bool(np.all(np.sum(coverage["place_actions"], axis=0) > 0)),
        "all_species_fed": bool(np.all(np.sum(coverage["feed_actions"], axis=0) > 0)),
        "all_species_cared": bool(np.all(np.sum(coverage["care_actions"], axis=0) > 0)),
        "all_species_harvested": bool(
            np.all(np.sum(coverage["harvest_actions"], axis=0) > 0)
        ),
        "all_species_fertilizer_collected": bool(
            np.all(np.sum(coverage["fertilizer_actions"], axis=0) > 0)
        ),
        "product_sell_path_exercised": int(np.sum(coverage["sell_orders_by_product"][:, 5:])) > 0,
        "land_and_hire_paths_exercised": int(np.sum(coverage["land_orders"])) > 0
        and int(np.sum(coverage["hire_orders"])) > 0,
    }
    rows = []
    for lane, name in enumerate(names):
        rows.append(
            {
                "candidate_id": lane,
                "name": name,
                "seed": int(seeds[lane]),
                "final_bank": int(values["final_bank"][lane]),
                "active_animals_by_species": values["active_animals_by_species"][lane]
                .astype(int)
                .tolist(),
                "hard_diagnostics": {
                    field: int(values[field][lane]) for field in hard_zero_fields
                },
            }
        )
    source_files = tuple(
        PROJECT_DIR / "src" / "project_route_search_v2" / name
        for name in (
            "m3_constants.py",
            "m3_schema.py",
            "m3_genome.py",
            "m3_controller.py",
            "m3_rollout.py",
            "m3_candidates.py",
        )
    )
    receipt = {
        "receipt_id": "M3_ANIMAL_BRANCH_COVERAGE_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "candidate_count": len(names),
        "steps_per_candidate": 719,
        "compile_and_execute_seconds": elapsed,
        "validation_errors": validation_errors,
        "hard_zero_fields": list(hard_zero_fields),
        "checks": checks,
        "coverage_totals": {
            key: np.sum(value, axis=0).astype(int).tolist()
            if value.ndim > 1
            else int(np.sum(value))
            for key, value in coverage.items()
        },
        "metric_totals": {
            key: int(np.sum(value)) for key, value in metrics.items()
        },
        "rows": rows,
        "source_sha256": {
            path.relative_to(REPO_ROOT).as_posix(): _sha(path) for path in source_files
        },
        "event_bank_sha256": _sha(event_bank),
        "boundary": "NULL_OPPONENT_BRANCH_CORRECTNESS_NOT_ROUTE_QUALITY",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "checks": checks,
                "final_bank": values["final_bank"].astype(int).tolist(),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
