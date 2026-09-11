"""Prove that every active M3 search gene changes full-season behavior."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


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
from project_route_search_v2.m3_genome import (  # noqa: E402
    default_m3_animal_genome_v2,
    validate_m3_animal_genome_v2,
)
from project_route_search_v2.m3_rollout import (  # noqa: E402
    initialize_m3_rollout_carry_v2,
    make_m3_animal_rollout_v2,
)


ACTIVE_SEARCH_FIELDS = (
    "phase_count",
    "phase_start_step",
    "animal_target",
    "land_target",
    "hand_target",
    "animal_investment_stop_step",
    "animal_layout_policy",
    "animal_place_wave_size",
    "feed_stock_horizon_days",
    "care_policy",
    "first_cycle_care_bonus_target",
    "steady_cycle_care_bonus_target",
    "animal_harvest_trigger_units",
    "animal_fertilizer_policy",
    "maintenance_utilization_cap",
    "sell_interval",
    "sell_phase",
    "sell_price_floor_ratio",
    "sell_fraction",
    "shed_pressure_trigger",
    "cash_floor",
    "liquidation_start_step",
)

FROZEN_M3_FIELDS = (
    "animal_project_cash_cap",
    "feed_source_policy",
    "deposit_min_value",
)


def _pair(field: str):
    base = default_m3_animal_genome_v2(1)
    mutant = base
    evidence = ""
    if field == "phase_count":
        mutant = base._replace(phase_count=jnp.asarray((2,), dtype=jnp.int8))
        evidence = "third sheep phase is disabled"
    elif field == "phase_start_step":
        mutant = base._replace(
            phase_start_step=base.phase_start_step.at[0, 2].set(14 * 24)
        )
        evidence = "mixed-animal expansion moves from day 10 to day 14"
    elif field == "animal_target":
        base = base._replace(
            phase_count=jnp.asarray((1,), dtype=jnp.int8),
            animal_target=jnp.zeros_like(base.animal_target).at[0, :, 0].set(1),
            hand_target=jnp.full_like(base.hand_target, 4),
            care_policy=jnp.zeros_like(base.care_policy),
            cash_floor=jnp.asarray((0,), dtype=jnp.int32),
        )
        mutant = base._replace(
            animal_target=base.animal_target.at[0, :, 0].set(2)
        )
        evidence = "low-load goose commitment increases from one to two"
    elif field == "land_target":
        mutant = base._replace(land_target=jnp.ones_like(base.land_target))
        evidence = "second-land purchase is disabled"
    elif field == "hand_target":
        mutant = base._replace(hand_target=jnp.full_like(base.hand_target, 4))
        evidence = "daily workforce target is held at four"
    elif field == "animal_investment_stop_step":
        mutant = base._replace(
            animal_investment_stop_step=base.animal_investment_stop_step.at[0, 1].set(0)
        )
        evidence = "cow admission window is closed"
    elif field == "animal_layout_policy":
        mutant = base._replace(
            animal_layout_policy=base.animal_layout_policy.at[0, 0].set(3)
        )
        evidence = "goose layout changes from center to route strip"
    elif field == "animal_place_wave_size":
        base = base._replace(
            animal_target=jnp.zeros_like(base.animal_target).at[0, :, 0].set(8),
            hand_target=jnp.full_like(base.hand_target, 10),
            care_policy=jnp.zeros_like(base.care_policy),
            cash_floor=jnp.asarray((0,), dtype=jnp.int32),
            animal_place_wave_size=base.animal_place_wave_size.at[0, 0].set(8),
        )
        mutant = base._replace(
            animal_place_wave_size=base.animal_place_wave_size.at[0, 0].set(1)
        )
        evidence = "eight-goose route admission wave drops from eight to one"
    elif field == "feed_stock_horizon_days":
        mutant = base._replace(
            feed_stock_horizon_days=jnp.asarray((4,), dtype=jnp.int8)
        )
        evidence = "physical feed horizon increases from two to four days"
    elif field == "care_policy":
        mutant = base._replace(care_policy=jnp.zeros_like(base.care_policy))
        evidence = "all CARE investment is disabled"
    elif field == "first_cycle_care_bonus_target":
        mutant = base._replace(
            first_cycle_care_bonus_target=base.first_cycle_care_bonus_target.at[0, 0].set(0)
        )
        evidence = "first goose cycle CARE target drops from three to zero"
    elif field == "steady_cycle_care_bonus_target":
        mutant = base._replace(
            steady_cycle_care_bonus_target=base.steady_cycle_care_bonus_target.at[0, 1].set(0)
        )
        evidence = "steady cow CARE target drops from two to zero"
    elif field == "animal_harvest_trigger_units":
        mutant = base._replace(
            animal_harvest_trigger_units=base.animal_harvest_trigger_units.at[0, 0].set(1)
        )
        evidence = "goose harvest threshold drops from three to one"
    elif field == "animal_fertilizer_policy":
        mutant = base._replace(
            animal_fertilizer_policy=jnp.asarray((2,), dtype=jnp.int8)
        )
        evidence = "fertilizer switches from passive to active collection"
    elif field == "maintenance_utilization_cap":
        base = base._replace(
            animal_target=jnp.full_like(base.animal_target, 8),
            hand_target=jnp.full_like(base.hand_target, 5),
            care_policy=jnp.full_like(base.care_policy, 2),
            maintenance_utilization_cap=jnp.asarray((0.95,), dtype=jnp.float32),
        )
        mutant = base._replace(
            maintenance_utilization_cap=jnp.asarray((0.35,), dtype=jnp.float32)
        )
        evidence = "low utilization cap admits fewer CARE-heavy animals"
    elif field == "sell_interval":
        mutant = base._replace(sell_interval=base.sell_interval.at[0, 5].set(48))
        evidence = "egg sale cadence changes from daily to every two days"
    elif field == "sell_phase":
        mutant = base._replace(sell_phase=base.sell_phase.at[0, 5].set(1))
        evidence = "egg sale due step shifts by one"
    elif field == "sell_price_floor_ratio":
        mutant = base._replace(
            sell_price_floor_ratio=base.sell_price_floor_ratio.at[0, 5].set(2.0)
        )
        evidence = "egg sales require twice base price"
    elif field == "sell_fraction":
        mutant = base._replace(sell_fraction=base.sell_fraction.at[0, 5].set(0.25))
        evidence = "only one quarter of due eggs is sold"
    elif field == "shed_pressure_trigger":
        mutant = base._replace(
            shed_pressure_trigger=jnp.asarray((20,), dtype=jnp.int16)
        )
        evidence = "shed pressure selling activates at twenty units"
    elif field == "cash_floor":
        mutant = base._replace(cash_floor=jnp.asarray((1500,), dtype=jnp.int32))
        evidence = "protected operating cash increases from 300 to 1500"
    elif field == "liquidation_start_step":
        mutant = base._replace(
            liquidation_start_step=jnp.asarray((25 * 24,), dtype=jnp.int16)
        )
        evidence = "liquidation advances from day 28 to day 25"
    else:
        raise KeyError(field)
    return base, mutant, evidence


def _stack(records):
    return jax.tree.map(lambda *values: jnp.concatenate(values, axis=0), *records)


def _action_lane(action, lane: int):
    return tuple(np.asarray(field[:, lane]) for field in action)


def _hash_lane(arrays) -> str:
    digest = hashlib.sha256()
    for name, value in zip(
        (
            "unit_op",
            "unit_item",
            "unit_amount",
            "unit_count",
            "market_op",
            "market_item",
            "market_amount",
            "market_count",
        ),
        arrays,
        strict=True,
    ):
        digest.update(name.encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest().upper()


def main() -> None:
    records = []
    metadata = []
    for field in ACTIVE_SEARCH_FIELDS:
        baseline, mutant, evidence = _pair(field)
        records.extend((baseline, mutant))
        metadata.append((field, evidence))
    genome = _stack(records)._replace(
        candidate_id=jnp.arange(len(records), dtype=jnp.int32)
    )
    validation_errors = validate_m3_animal_genome_v2(genome)
    source_seeds, bank = load_event_bank(
        PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz"
    )
    seed_ids = np.repeat(np.arange(len(metadata), dtype=np.int32), 2)
    seeds = np.asarray(source_seeds, dtype=np.int64)[seed_ids]
    events = Events(bank.weed_spawn[seed_ids], bank.shop_choice[seed_ids])
    carry = initialize_m3_rollout_carry_v2(
        jnp.asarray(seeds, dtype=jnp.int32), genome
    )
    final, action_trace = jax.jit(
        make_m3_animal_rollout_v2(trace="actions")
    )(carry, events, load_tables(), genome)
    jax.block_until_ready(final)
    action_trace = jax.device_get(action_trace)
    final_bank = np.asarray(jax.device_get(final.environment_state.money[:, 0]))

    results = []
    for case, (field, evidence) in enumerate(metadata):
        left_lane = case * 2
        right_lane = left_lane + 1
        before = _action_lane(action_trace, left_lane)
        after = _action_lane(action_trace, right_lane)
        step_changed = np.zeros((719,), dtype=bool)
        changed_elements = 0
        for left, right in zip(before, after, strict=True):
            unequal = left != right
            changed_elements += int(np.sum(unequal))
            step_changed |= np.any(unequal.reshape(719, -1), axis=1)
        changed_steps = np.flatnonzero(step_changed)
        status = not validation_errors and changed_steps.size > 0
        results.append(
            {
                "field": field,
                "activation_scenario": evidence,
                "behavior_hash_before": _hash_lane(before),
                "behavior_hash_after": _hash_lane(after),
                "first_changed_step": int(changed_steps[0])
                if changed_steps.size
                else None,
                "changed_step_count": int(changed_steps.size),
                "changed_action_element_count": changed_elements,
                "final_bank_before": int(final_bank[left_lane]),
                "final_bank_after": int(final_bank[right_lane]),
                "status": "PASS" if status else "FAIL",
            }
        )

    observed = [row["field"] for row in results]
    checks = {
        "all_active_fields_have_one_case": observed == list(ACTIVE_SEARCH_FIELDS),
        "all_genomes_valid": not validation_errors,
        "all_active_fields_change_actions": all(
            row["status"] == "PASS" for row in results
        ),
        "frozen_fields_not_in_active_search": not (
            set(FROZEN_M3_FIELDS) & set(ACTIVE_SEARCH_FIELDS)
        ),
    }
    receipt = {
        "receipt_id": "M3_PARAMETER_EFFECT_COVERAGE_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "active_search_fields": list(ACTIVE_SEARCH_FIELDS),
        "frozen_m3_fields": list(FROZEN_M3_FIELDS),
        "validation_errors": validation_errors,
        "checks": checks,
        "results": results,
        "boundary": "FULL_SEASON_ACTION_EFFECT_WIRING_NOT_ROUTE_QUALITY",
    }
    output = PROJECT_DIR / "receipts" / "m3_parameter_effect_coverage_v1.json"
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "checks": checks,
                "failed": [
                    row["field"] for row in results if row["status"] != "PASS"
                ],
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
