"""Run the M3.5 crop/animal resource-loop acceptance panel."""

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
from project_route_search_v2.m35_candidates import (  # noqa: E402
    m35_branch_coverage_panel_v2,
)
from project_route_search_v2.m35_genome import (  # noqa: E402
    validate_m35_farm_genome_v2,
)
from project_route_search_v2.m35_rollout import (  # noqa: E402
    initialize_m35_rollout_carry_v2,
    make_m35_farm_rollout_v2,
    summarize_m35_rollout_v2,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _values(record) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value)
        for key, value in jax.device_get(record)._asdict().items()
    }


def main() -> None:
    event_bank = PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz"
    output = PROJECT_DIR / "receipts" / "m35_branch_coverage_v1.json"
    source_seeds, bank = load_event_bank(event_bank)
    genome, names = m35_branch_coverage_panel_v2()
    validation_errors = validate_m35_farm_genome_v2(genome)
    indices = np.arange(len(names), dtype=np.int32)
    seeds = np.asarray(source_seeds, dtype=np.int64)[indices]
    events = Events(bank.weed_spawn[indices], bank.shop_choice[indices])
    carry = initialize_m35_rollout_carry_v2(
        jnp.asarray(seeds, dtype=jnp.int32), genome
    )
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
    flow = _values(summary.flow)
    crop_coverage = _values(summary.crop_coverage)
    animal_coverage = _values(summary.animal_coverage)
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
    grow = np.asarray((1, 4), dtype=np.int32)
    hybrid = np.asarray((2, 5), dtype=np.int32)
    no_recycle = np.asarray((0, 1, 2), dtype=np.int32)
    recycle = np.asarray((3, 4, 5), dtype=np.int32)
    checks = {
        "genome_validation_pass": not validation_errors,
        "all_6_candidates_done": bool(np.all(values["done"])),
        "all_correctness_diagnostics_zero": all(
            int(np.sum(values[field])) == 0 for field in hard_zero_fields
        ),
        "planned_wheat_and_tomato_planted": bool(
            np.all(np.sum(crop_coverage["crop_plant_actions"], axis=0)[[0, 2]] > 0)
        ),
        "planned_wheat_and_tomato_harvested": bool(
            np.all(
                np.sum(crop_coverage["crop_harvest_actions"], axis=0)[[0, 2]] > 0
            )
        ),
        "all_animal_species_built": bool(
            np.all(np.sum(animal_coverage["build_actions"], axis=0) > 0)
        ),
        "all_animal_species_purchased": bool(
            np.all(np.sum(animal_coverage["purchase_orders"], axis=0) > 0)
        ),
        "all_animal_species_placed": bool(
            np.all(np.sum(animal_coverage["place_actions"], axis=0) > 0)
        ),
        "all_animal_species_fed": bool(
            np.all(np.sum(animal_coverage["feed_actions"], axis=0) > 0)
        ),
        "all_animal_species_harvested": bool(
            np.all(np.sum(animal_coverage["harvest_actions"], axis=0) > 0)
        ),
        "grown_wheat_enters_feed_ledger": bool(
            np.all(flow["wheat_harvest_actions"][grow] > 0)
            and np.all(flow["wheat_market_buy_units"][grow] == 0)
            and np.all(flow["animal_feed_actions"][grow] > 0)
        ),
        "hybrid_uses_grown_and_bought_feed": bool(
            np.all(flow["wheat_harvest_actions"][hybrid] > 0)
            and np.all(flow["wheat_market_buy_units"][hybrid] > 0)
            and np.all(flow["animal_feed_actions"][hybrid] > 0)
        ),
        "animal_fertilizer_returns_to_crops": bool(
            np.all(flow["animal_fertilizer_collect_actions"][recycle] > 0)
            and np.all(flow["crop_fertilizer_apply_actions"][recycle] > 0)
            and np.all(flow["fertilizer_market_buy_units"][recycle] == 0)
        ),
        "recycle_switch_changes_behavior": bool(
            np.all(flow["crop_fertilizer_apply_actions"][no_recycle] == 0)
            and np.all(flow["crop_fertilizer_apply_actions"][recycle] > 0)
        ),
        "joint_resource_conflicts_zero": bool(
            np.all(flow["joint_resource_conflict_count"] == 0)
        ),
    }
    rows = []
    for lane, name in enumerate(names):
        rows.append(
            {
                "candidate_id": lane,
                "name": name,
                "seed": int(seeds[lane]),
                "final_bank": int(values["final_bank"][lane]),
                "active_animals_by_species": values[
                    "active_animals_by_species"
                ][lane].astype(int).tolist(),
                "flow": {key: int(value[lane]) for key, value in flow.items()},
                "hard_diagnostics": {
                    field: int(values[field][lane]) for field in hard_zero_fields
                },
                "economic_diagnostics": {
                    field: int(values[field][lane])
                    for field in economic_diagnostic_fields
                },
            }
        )
    source_files = tuple(
        PROJECT_DIR / "src" / "project_route_search_v2" / name
        for name in (
            "m35_schema.py",
            "m35_genome.py",
            "m35_controller.py",
            "m35_rollout.py",
            "m35_candidates.py",
        )
    )
    receipt = {
        "receipt_id": "M35_CROP_ANIMAL_RESOURCE_LOOP_BRANCH_COVERAGE_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "candidate_count": len(names),
        "steps_per_candidate": 719,
        "compile_and_execute_seconds": elapsed,
        "validation_errors": validation_errors,
        "hard_zero_fields": list(hard_zero_fields),
        "economic_diagnostic_fields": list(economic_diagnostic_fields),
        "checks": checks,
        "crop_coverage_totals": {
            key: np.sum(value, axis=0).astype(int).tolist()
            if value.ndim > 1
            else int(np.sum(value))
            for key, value in crop_coverage.items()
        },
        "animal_coverage_totals": {
            key: np.sum(value, axis=0).astype(int).tolist()
            if value.ndim > 1
            else int(np.sum(value))
            for key, value in animal_coverage.items()
        },
        "rows": rows,
        "source_sha256": {
            path.relative_to(REPO_ROOT).as_posix(): _sha(path)
            for path in source_files
        },
        "event_bank_sha256": _sha(event_bank),
        "objective_policy": {
            "primary": "MAXIMIZE_FINAL_BANK",
            "hard_failures": "RULE_LEDGER_AND_UNPLANNED_EXECUTION_ERRORS_ONLY",
            "terminal_residuals": "SOFT_ECONOMIC_DIAGNOSTICS_NOT_ACCEPTANCE_FAILURES",
        },
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
                "backend": receipt["backend"],
                "checks": checks,
                "final_bank": values["final_bank"].astype(int).tolist(),
                "active_animals": values["active_animals_by_species"].astype(int).tolist(),
                "flow": {key: value.astype(int).tolist() for key, value in flow.items()},
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
