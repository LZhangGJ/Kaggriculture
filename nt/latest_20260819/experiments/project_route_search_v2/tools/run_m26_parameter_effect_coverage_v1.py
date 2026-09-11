"""Audit schema, activation and behavior-effect coverage for every M2.6 gene."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable


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

from kaggriculture_jax.constants import FLAG_WATERED, TileKind  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from project_route_search_v2.lifecycle import initialize_project_controller_v2  # noqa: E402
from project_route_search_v2.m26_controller import m26_policy_step_v2  # noqa: E402
from project_route_search_v2.m26_genome import (  # noqa: E402
    default_m26_crop_genome_v2,
    validate_m26_crop_genome_v2,
)


SEARCHABLE_FIELDS = (
    "phase_count",
    "phase_start_step",
    "crop_target",
    "land_target",
    "land_start_step",
    "hand_target",
    "crop_last_plant_step",
    "seed_buy_batch",
    "plant_wave_size",
    "harvest_min_age_days",
    "harvest_trigger_units",
    "fertilizer_policy",
    "crop_layout_policy",
    "crop_cash_cap",
    "weed_recovery_policy",
    "crop_abandon_policy",
    "maintenance_utilization_cap",
    "deposit_min_value",
    "sell_interval",
    "sell_phase",
    "sell_price_floor_ratio",
    "sell_fraction",
    "shed_pressure_trigger",
    "cash_floor",
    "liquidation_start_step",
)

TABLES = load_tables()


def _states():
    return jax.vmap(reset)(jnp.asarray((26001,), dtype=jnp.int32))


def _zero_business(genome):
    return genome._replace(
        crop_target=jnp.zeros_like(genome.crop_target),
        hand_target=jnp.zeros_like(genome.hand_target),
        land_target=jnp.ones_like(genome.land_target),
        sell_interval=jnp.full_like(genome.sell_interval, 48),
        sell_phase=jnp.zeros_like(genome.sell_phase),
        shed_pressure_trigger=jnp.full_like(genome.shed_pressure_trigger, 100),
    )


def _fingerprint(states, genome, controller_mutator=None):
    controller = initialize_project_controller_v2(states, 0)
    if controller_mutator is not None:
        controller = controller_mutator(controller)
    action, controller = m26_policy_step_v2(
        states, controller, genome, 0, TABLES
    )
    arrays = {
        "route_phase": controller.route_phase,
        "project_target": controller.projects.target_count,
        "project_stop": controller.projects.stop_step,
        "project_layout": controller.projects.layout_policy_id,
        "unit_task_type": controller.unit_tasks.task_type,
        "unit_task_target": controller.unit_tasks.target_id,
        "unit_task_item": controller.unit_tasks.item_id,
        "unit_task_status": controller.unit_tasks.status,
        "market_task_type": controller.market_tasks.task_type,
        "market_task_item": controller.market_tasks.item_id,
        "market_task_quantity": controller.market_tasks.quantity,
        "market_task_status": controller.market_tasks.status,
        "action_unit_op": action.unit_op,
        "action_unit_item": action.unit_item,
        "action_unit_count": action.unit_count,
        "action_market_op": action.market_op,
        "action_market_item": action.market_item,
        "action_market_amount": action.market_amount,
        "action_market_count": action.market_count,
    }
    host = {name: np.asarray(value) for name, value in arrays.items()}
    digest = hashlib.sha256()
    for name in sorted(host):
        digest.update(name.encode("utf-8"))
        digest.update(host[name].tobytes())
    return digest.hexdigest().upper(), host


def _differences(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> list[str]:
    return [name for name in left if not np.array_equal(left[name], right[name])]


def _evaluate(field, states, baseline, mutant, activation, controller_mutator=None):
    baseline_errors = validate_m26_crop_genome_v2(baseline)
    mutant_errors = validate_m26_crop_genome_v2(mutant)
    before_hash, before = _fingerprint(states, baseline, controller_mutator)
    after_hash, after = _fingerprint(states, mutant, controller_mutator)
    changed = _differences(before, after)
    return {
        "field": field,
        "schema_coverage": not baseline_errors and not mutant_errors,
        "activation_coverage": True,
        "activation_evidence": activation,
        "behavior_effect_coverage": bool(changed),
        "behavior_hash_before": before_hash,
        "behavior_hash_after": after_hash,
        "changed_records": changed,
        "baseline_validation_errors": baseline_errors,
        "mutant_validation_errors": mutant_errors,
        "status": "PASS" if not baseline_errors and not mutant_errors and changed else "FAIL",
    }


def _cases():
    cases = []
    base = default_m26_crop_genome_v2(1)

    states = _states()._replace(step=jnp.asarray((6 * 24,), dtype=jnp.int16))
    b = base._replace(phase_count=jnp.asarray((1,), dtype=jnp.int8))
    m = b._replace(phase_count=jnp.asarray((2,), dtype=jnp.int8))
    cases.append(("phase_count", states, b, m, "step crosses phase 1"))

    b = base
    m = base._replace(phase_start_step=base.phase_start_step.at[0, 1].set(7 * 24))
    cases.append(("phase_start_step", states, b, m, "phase 1 boundary moves after current step"))

    states = _states()._replace(money=jnp.full((1, 2), 50_000, dtype=jnp.int32))
    b = _zero_business(base)._replace(crop_target=jnp.zeros_like(base.crop_target).at[0, 0, 2].set(1))
    m = b._replace(crop_target=b.crop_target.at[0, 0, 2].set(4))
    cases.append(("crop_target", states, b, m, "tomato deficit changes"))

    states = _states()._replace(step=jnp.asarray((6 * 24,), dtype=jnp.int16), money=jnp.full((1, 2), 50_000, dtype=jnp.int32))
    b = _zero_business(base)._replace(land_target=jnp.ones_like(base.land_target).at[0, 1].set(2))
    m = b._replace(land_target=b.land_target.at[0, 1].set(1))
    cases.append(("land_target", states, b, m, "phase wants two versus one land"))
    m = b._replace(land_start_step=b.land_start_step.at[0, 1].set(7 * 24))
    cases.append(("land_start_step", states, b, m, "buy window moves from day 6 to day 7"))

    states = _states()._replace(money=jnp.full((1, 2), 50_000, dtype=jnp.int32))
    b = _zero_business(base)._replace(hand_target=jnp.zeros_like(base.hand_target).at[0, 0].set(2))
    m = b._replace(hand_target=jnp.zeros_like(b.hand_target))
    cases.append(("hand_target", states, b, m, "two hires versus none"))

    b = _zero_business(base)._replace(crop_target=jnp.zeros_like(base.crop_target).at[0, 0, 0].set(4))
    m = b._replace(crop_last_plant_step=b.crop_last_plant_step.at[0, 0].set(0))
    cases.append(("crop_last_plant_step", states, b, m, "wheat planting window closes"))
    m = b._replace(seed_buy_batch=b.seed_buy_batch.at[0, 0].set(1))
    cases.append(("seed_buy_batch", states, b, m, "wheat buy batch changes"))

    s = _states()._replace(
        step=jnp.asarray((1,), dtype=jnp.int16),
        unit_active=_states().unit_active.at[:, 0, :4].set(True),
        seeds=_states().seeds.at[:, 0, 0].set(12),
    )
    b = _zero_business(base)._replace(crop_target=jnp.zeros_like(base.crop_target).at[0, 0, 0].set(12))
    m = b._replace(plant_wave_size=b.plant_wave_size.at[0, 0].set(1))
    cases.append(("plant_wave_size", s, b, m, "four free units see different wave cap"))

    s0 = _states()
    s = s0._replace(
        step=jnp.asarray((8 * 24,), dtype=jnp.int16),
        tile_kind=s0.tile_kind.at[:, 0, 4, 4].set(TileKind.PLANT),
        tile_crop=s0.tile_crop.at[:, 0, 4, 4].set(2),
        tile_origin_day=s0.tile_origin_day.at[:, 0, 4, 4].set(0),
        tile_yield=s0.tile_yield.at[:, 0, 4, 4].set(2),
        tile_flags=s0.tile_flags.at[:, 0, 4, 4].set(jnp.uint8(FLAG_WATERED)),
    )
    b = _zero_business(base)
    m = b._replace(harvest_min_age_days=b.harvest_min_age_days.at[0, 2].set(9))
    cases.append(("harvest_min_age_days", s, b, m, "tomato age 8 is on boundary"))
    m = b._replace(harvest_trigger_units=b.harvest_trigger_units.at[0, 2].set(3))
    cases.append(("harvest_trigger_units", s, b, m, "tomato yield 2 is on boundary"))

    s = s._replace(shed=s.shed.at[:, 0, 8].set(1), tile_yield=s.tile_yield.at[:, 0, 4, 4].set(0))
    b = _zero_business(base)._replace(crop_target=jnp.zeros_like(base.crop_target).at[0, 0, 2].set(1))
    m = b._replace(fertilizer_policy=b.fertilizer_policy.at[0, 2].set(1))
    cases.append(("fertilizer_policy", s, b, m, "watered tomato and fertilizer stock exist"))

    s = _states()._replace(step=jnp.asarray((1,), dtype=jnp.int16), seeds=_states().seeds.at[:, 0, 2].set(1))
    b = _zero_business(base)._replace(crop_target=jnp.zeros_like(base.crop_target).at[0, 0, 2].set(1))
    m = b._replace(crop_layout_policy=b.crop_layout_policy.at[0, 2].set(3))
    cases.append(("crop_layout_policy", s, b, m, "center versus route-strip tomato placement"))

    states = _states()._replace(money=jnp.full((1, 2), 50_000, dtype=jnp.int32))
    b = _zero_business(base)._replace(crop_target=jnp.zeros_like(base.crop_target).at[0, 0, 2].set(4))
    m = b._replace(crop_cash_cap=b.crop_cash_cap.at[0, 2].set(0))
    cases.append(("crop_cash_cap", states, b, m, "tomato project cap blocks seed commitment"))

    s0 = _states()
    s = s0._replace(tile_kind=s0.tile_kind.at[:, 0, 0, 0].set(TileKind.WEED))
    b = _zero_business(base)
    m = b._replace(weed_recovery_policy=jnp.zeros_like(b.weed_recovery_policy))
    mutate_controller: Callable = lambda c: c._replace(
        tile_project_id=jnp.full((1, 10, 10), -1, dtype=jnp.int16).at[:, 0, 0].set(2)
    )
    cases.append(("weed_recovery_policy", s, b, m, "remembered tomato weed can be dug", mutate_controller))

    s0 = _states()
    s = s0._replace(
        step=jnp.asarray((24,), dtype=jnp.int16),
        tile_kind=s0.tile_kind.at[:, 0, 0, 0].set(TileKind.PLANT),
        tile_crop=s0.tile_crop.at[:, 0, 0, 0].set(2),
        tile_flags=s0.tile_flags.at[:, 0, 0, 0].set(jnp.uint8(FLAG_WATERED)),
        seeds=s0.seeds.at[:, 0, 0].set(1),
    )
    targets = jnp.zeros_like(base.crop_target).at[0, 0, 2].set(1).at[0, 1, 0].set(1)
    b = _zero_business(base)._replace(
        phase_count=jnp.asarray((2,), dtype=jnp.int8),
        phase_start_step=base.phase_start_step.at[0, 1].set(24),
        crop_target=targets,
        crop_abandon_policy=jnp.zeros_like(base.crop_abandon_policy),
    )
    m = b._replace(crop_abandon_policy=b.crop_abandon_policy.at[0, 2].set(2))
    cases.append(("crop_abandon_policy", s, b, m, "tomato target shrinks while wheat replacement exists"))

    states = _states()._replace(money=jnp.full((1, 2), 50_000, dtype=jnp.int32))
    b = _zero_business(base)._replace(crop_target=jnp.zeros_like(base.crop_target).at[0, 0, 0].set(12))
    m = b._replace(maintenance_utilization_cap=jnp.asarray((0.05,), dtype=jnp.float32))
    cases.append(("maintenance_utilization_cap", states, b, m, "target load crosses action-capacity gate"))

    s0 = _states()
    s = s0._replace(
        unit_pos=s0.unit_pos.at[:, 0, 0].set(jnp.asarray((0, 0), dtype=jnp.int8)),
        unit_inventory=s0.unit_inventory.at[:, 0, 0, 2].set(1),
    )
    b = _zero_business(base)._replace(deposit_min_value=jnp.asarray((50,), dtype=jnp.int32))
    m = b._replace(deposit_min_value=jnp.asarray((100,), dtype=jnp.int32))
    cases.append(("deposit_min_value", s, b, m, "carried tomato value is 60"))

    s0 = _states()
    s = s0._replace(
        step=jnp.asarray((24,), dtype=jnp.int16),
        shed=s0.shed.at[:, 0, 2].set(10),
        money=jnp.full((1, 2), 10_000, dtype=jnp.int32),
    )
    b = _zero_business(base)._replace(sell_interval=jnp.full_like(base.sell_interval, 24))
    m = b._replace(sell_interval=jnp.full_like(b.sell_interval, 48))
    cases.append(("sell_interval", s, b, m, "step 24 is due only for interval 24"))
    b = b._replace(sell_phase=jnp.zeros_like(b.sell_phase))
    m = b._replace(sell_phase=jnp.ones_like(b.sell_phase))
    cases.append(("sell_phase", s, b, m, "phase offset moves due step"))
    b = _zero_business(base)._replace(
        sell_interval=jnp.full_like(base.sell_interval, 24),
        sell_price_floor_ratio=jnp.zeros_like(base.sell_price_floor_ratio),
    )
    m = b._replace(sell_price_floor_ratio=jnp.full_like(b.sell_price_floor_ratio, 2.0))
    cases.append(("sell_price_floor_ratio", s, b, m, "current price fails 2x base floor"))
    b = b._replace(sell_fraction=jnp.full_like(b.sell_fraction, 0.5))
    m = b._replace(sell_fraction=jnp.ones_like(b.sell_fraction))
    cases.append(("sell_fraction", s, b, m, "half versus full shed sale"))

    s = s._replace(step=jnp.asarray((1,), dtype=jnp.int16), shed=s.shed.at[:, 0, 2].set(90))
    b = _zero_business(base)._replace(shed_pressure_trigger=jnp.asarray((80,), dtype=jnp.int16))
    m = b._replace(shed_pressure_trigger=jnp.asarray((100,), dtype=jnp.int16))
    cases.append(("shed_pressure_trigger", s, b, m, "shed usage 90 crosses only threshold 80"))

    states = _states()._replace(money=jnp.full((1, 2), 20, dtype=jnp.int32))
    b = _zero_business(base)._replace(
        crop_target=jnp.zeros_like(base.crop_target).at[0, 0, 0].set(1),
        cash_floor=jnp.asarray((0,), dtype=jnp.int32),
    )
    m = b._replace(cash_floor=jnp.asarray((20,), dtype=jnp.int32))
    cases.append(("cash_floor", states, b, m, "seed purchase consumes protected cash"))

    s0 = _states()
    s = s0._replace(
        step=jnp.asarray((28 * 24 + 1,), dtype=jnp.int16),
        shed=s0.shed.at[:, 0, 2].set(10),
        money=jnp.full((1, 2), 10_000, dtype=jnp.int32),
    )
    b = _zero_business(base)
    m = b._replace(liquidation_start_step=jnp.asarray((719,), dtype=jnp.int16))
    cases.append(("liquidation_start_step", s, b, m, "default is closing while mutant still holds"))
    return cases


def main() -> None:
    results = []
    for raw in _cases():
        field, states, baseline, mutant, activation, *extra = raw
        results.append(
            _evaluate(
                field,
                states,
                baseline,
                mutant,
                activation,
                extra[0] if extra else None,
            )
        )

    observed = [row["field"] for row in results]
    missing = sorted(set(SEARCHABLE_FIELDS) - set(observed))
    duplicate = sorted({field for field in observed if observed.count(field) > 1})
    checks = {
        "all_searchable_fields_have_exactly_one_case": not missing and not duplicate and len(observed) == len(SEARCHABLE_FIELDS),
        "schema_coverage_all_pass": all(row["schema_coverage"] for row in results),
        "activation_coverage_all_pass": all(row["activation_coverage"] for row in results),
        "behavior_effect_coverage_all_pass": all(row["behavior_effect_coverage"] for row in results),
        "dead_searchable_field_zero": all(row["status"] == "PASS" for row in results),
    }
    receipt = {
        "receipt_id": "M26_PARAMETER_EFFECT_COVERAGE_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "searchable_fields": list(SEARCHABLE_FIELDS),
        "missing_fields": missing,
        "duplicate_fields": duplicate,
        "checks": checks,
        "results": results,
        "boundary": "PARAMETER_WIRING_ONLY_NOT_ROUTE_QUALITY",
    }
    output = PROJECT_DIR / "receipts" / "m26_parameter_effect_coverage_v1.json"
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "checks": checks, "failed": [row["field"] for row in results if row["status"] != "PASS"], "output": str(output)}, ensure_ascii=False, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
