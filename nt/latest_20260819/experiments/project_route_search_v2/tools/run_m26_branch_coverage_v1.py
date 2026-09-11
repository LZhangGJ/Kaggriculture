"""Run the frozen M2.6 complete-season major-branch panel."""

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
from project_route_search_v2.m26_candidates import m26_branch_coverage_panel_v2  # noqa: E402
from project_route_search_v2.m26_rollout import (  # noqa: E402
    initialize_m26_rollout_carry_v2,
    make_m26_crop_rollout_v2,
    summarize_m26_rollout_v2,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    source_seeds, bank = load_event_bank(PROJECT_DIR / "artifacts" / "events" / "e0_seed_panels_v1.npz")
    genome, names = m26_branch_coverage_panel_v2()
    count = len(names)
    panel = json.loads((PROJECT_DIR / "configs" / "seed_panels_v1.json").read_text(encoding="utf-8"))
    audit = np.asarray(panel["panels"]["E0_AUDIT_128"]["seeds"], dtype=np.int64)[:count]
    source_index = {int(seed): index for index, seed in enumerate(source_seeds)}
    indices = np.asarray([source_index[int(seed)] for seed in audit], dtype=np.int32)
    events = Events(bank.weed_spawn[indices], bank.shop_choice[indices])
    carry = initialize_m26_rollout_carry_v2(jnp.asarray(audit, dtype=jnp.int32), genome)
    rollout = jax.jit(make_m26_crop_rollout_v2())
    started = time.perf_counter()
    final, _ = rollout(carry, events, load_tables(), genome)
    jax.block_until_ready(final)
    elapsed = time.perf_counter() - started
    summary = jax.device_get(summarize_m26_rollout_v2(final))
    metrics = {
        name: np.asarray(value)
        for name, value in jax.device_get(final.metrics)._asdict().items()
    }

    arrays = {name: np.asarray(value) for name, value in summary._asdict().items() if name != "coverage"}
    coverage = {name: np.asarray(value) for name, value in summary.coverage._asdict().items()}
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "candidate_id": index,
                "name": name,
                "seed": int(audit[index]),
                "final_bank": int(arrays["final_bank"][index]),
                "done": bool(arrays["done"][index]),
                "unexplained_failure_count": int(arrays["unexplained_failure_count"][index]),
                "plant_without_same_day_water": int(arrays["plant_without_same_day_water"][index]),
                "terminal_sellable_shed_value": int(arrays["terminal_sellable_shed_value"][index]),
                "avoidable_liquidation_loss": int(arrays["avoidable_liquidation_loss"][index]),
                "terminal_unit_inventory": np.asarray(
                    jax.device_get(final.environment_state.unit_inventory[index, 0, :, :9])
                ).astype(int).tolist(),
                "terminal_shed": np.asarray(
                    jax.device_get(final.environment_state.shed[index, 0, :9])
                ).astype(int).tolist(),
                "diagnostics": {
                    field: int(values[index])
                    for field, values in metrics.items()
                    if field.endswith("_count") or field.endswith("_hits")
                },
                "planted_tiles_by_crop": arrays["planted_tiles_by_crop"][index].astype(int).tolist(),
                "crop_plant_actions": coverage["crop_plant_actions"][index].astype(int).tolist(),
                "layout_plant_actions": coverage["layout_plant_actions"][index].astype(int).tolist(),
                "fertilizer_actions": coverage["fertilizer_actions"][index].astype(int).tolist(),
                "expansion_events": coverage["expansion_events"][index].astype(int).tolist(),
                "shrink_events": coverage["shrink_events"][index].astype(int).tolist(),
                "stop_events": coverage["stop_events"][index].astype(int).tolist(),
                "restart_events": coverage["restart_events"][index].astype(int).tolist(),
            }
        )

    plant_total = np.sum(coverage["crop_plant_actions"], axis=0)
    layout_total = np.sum(coverage["layout_plant_actions"], axis=0)
    checks = {
        "all_done": bool(np.all(arrays["done"])),
        "illegal_or_unexplained_failure_zero": bool(np.all(arrays["unexplained_failure_count"] == 0)),
        "same_day_water_zero": bool(np.all(arrays["plant_without_same_day_water"] == 0)),
        "terminal_sellable_shed_zero": bool(np.all(arrays["terminal_sellable_shed_value"] == 0)),
        "avoidable_liquidation_loss_zero": bool(np.all(arrays["avoidable_liquidation_loss"] == 0)),
        "all_five_crops_plant": bool(np.all(plant_total > 0)),
        "all_four_layouts_plant": bool(np.all(layout_total > 0)),
        "one_three_six_phase_present": bool(
            np.any(np.sum(coverage["phase_step_count"] > 0, axis=1) == 1)
            and np.any(np.sum(coverage["phase_step_count"] > 0, axis=1) == 3)
            and np.any(np.sum(coverage["phase_step_count"] > 0, axis=1) == 6)
        ),
        "expand_shrink_stop_restart_active": bool(
            np.any(coverage["expansion_events"] > 0)
            and np.any(coverage["shrink_events"] > 0)
            and np.any(coverage["stop_events"] > 0)
            and np.any(coverage["restart_events"] > 0)
        ),
        "fertilizer_modes_execute": bool(
            np.sum(coverage["fertilizer_actions"][names.index("FERTILIZER_ALWAYS_STRAWBERRY")]) > 0
            and np.sum(coverage["fertilizer_actions"][names.index("FERTILIZER_HIGH_VALUE_MELON")]) > 0
        ),
    }
    source_paths = [
        PROJECT_DIR / "src" / "project_route_search_v2" / "m26_candidates.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "m26_controller.py",
        PROJECT_DIR / "src" / "project_route_search_v2" / "m26_rollout.py",
        Path(__file__).resolve(),
    ]
    receipt = {
        "receipt_id": "M26_MAJOR_BRANCH_COMPLETE_SEASON_COVERAGE_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "compile_and_run_seconds": elapsed,
        "candidate_count": count,
        "checks": checks,
        "aggregate": {
            "crop_plant_actions": plant_total.astype(int).tolist(),
            "layout_plant_actions": layout_total.astype(int).tolist(),
        },
        "per_candidate": rows,
        "files": {
            path.relative_to(REPO_ROOT).as_posix(): {"sha256": _sha(path), "bytes": path.stat().st_size}
            for path in source_paths
        },
        "boundary": "MAJOR_BRANCH_EXECUTION_COVERAGE_NOT_ROUTE_QUALITY",
    }
    output = PROJECT_DIR / "receipts" / "m26_branch_coverage_v1.json"
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "checks": checks, "banks": {row["name"]: row["final_bank"] for row in rows}, "output": str(output)}, ensure_ascii=False, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
