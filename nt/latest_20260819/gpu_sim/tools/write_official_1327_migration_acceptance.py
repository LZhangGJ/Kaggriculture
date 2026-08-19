"""Write the machine-readable Kaggriculture 1.32.7 migration acceptance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GPU = ROOT / "gpu_sim"
V5 = ROOT / "experiments" / "strategic_v5"
MIGRATION = ROOT / "research" / "official_env_update_20260815"
OUTPUT = MIGRATION / "migration_acceptance.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    reference = load(GPU / "receipts" / "reference_verification.json")
    static = load(GPU / "receipts" / "static_tables_v2.json")
    diff = load(MIGRATION / "rules_and_tables_diff.json")
    tests = load(GPU / "receipts" / "test_suite.json")
    targeted = load(MIGRATION / "strategic_targeted_tests.json")
    heldout = load(GPU / "receipts" / "heldout_parity.json")
    randomized = load(GPU / "receipts" / "random_differential_parity.json")
    dynamic = load(V5 / "receipts" / "dl3_dynamic_official_differential_v2.json")
    boatlee = load(V5 / "receipts" / "h1c_boatlee_v16_gpu_parity_v1.json")
    e1 = load(V5 / "receipts" / "e1_official_rule_evidence_v1.json")
    final = load(GPU / "receipts" / "final_acceptance.json")
    core_perf = load(GPU / "receipts" / "benchmark_simulator_only_full_season.json")
    policy_perf = load(GPU / "receipts" / "benchmark_policy_and_ppo.json")
    old_dynamic = load(V5 / "receipts" / "dynamic_full_v2_86k_freeze_benchmark.json")
    new_dynamic = load(MIGRATION / "dynamic_full_v2_1327_b2048_s719_fair.json")

    expected_source = "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e"
    official_source = ROOT / "data" / "official" / "files" / "kaggriculture.py"
    old_median = float(old_dynamic["timing"]["median_transitions_per_second"])
    new_median = float(new_dynamic["timing"]["median_transitions_per_second"])
    ppo_steps = [int(row["optimizer_step"]) for row in policy_perf["full_ppo_iteration"]]

    gates = {
        "official_1_32_7_installed_and_frozen": reference.get("ok") is True
        and reference.get("installed_package_version") == "1.32.7",
        "workspace_official_source_is_exact": sha256(official_source) == expected_source,
        "new_lut_is_exact_int32": static.get("ok") is True
        and static.get("official_package_version") == "1.32.7"
        and static["market"].get("dtype") == "int32",
        "only_expected_market_products_changed": diff["market_lut"].get("changed_products")
        == ["CARROT", "TOMATO", "EGG"],
        "int16_overflow_risk_eliminated": diff["market_lut"].get("int16_would_overflow") is True,
        "event_tables_are_reusable": diff.get("event_tables_reusable") is True,
        "gpu_sim_unit_suite_14_passed": tests.get("ok") is True
        and "14 passed" in tests.get("output", ""),
        "strategic_targeted_tests_passed": targeted.get("ok") is True,
        "route_compiler_1327_compatibility_passed": targeted.get(
            "route_compiler_compatibility", {}
        ).get("ok")
        is True,
        "strategic_int32_price_chain_passed": targeted.get(
            "strategic_int32_price_chain", {}
        ).get("ok")
        is True,
        "dynamic_training_smoke_passed": targeted.get(
            "dynamic_training_smoke", {}
        ).get("ok")
        is True,
        "official_heldout_72000_frames_exact": heldout.get("ok") is True
        and heldout.get("frames_compared") == 72000,
        "random_invalid_11520_frames_exact": randomized.get("ok") is True
        and randomized.get("frames_compared") == 11520,
        "dynamic_full_core_2160_frames_exact": dynamic.get("accepted") is True
        and dynamic.get("frames_compared") == 2160,
        "boatlee_gpu_5750_plus_frames_exact": boatlee.get("ppo_authorized") is True
        and boatlee.get("state_frames", 0) >= 5750,
        "v5_rule_evidence_reissued_for_1327": e1.get("status") == "PASS"
        and e1.get("official_package_version") == "1.32.7",
        "jax_final_acceptance_current": final.get("ok") is True,
        "core_throughput_target_passed": core_perf.get("formal_target_passed") is True,
        "dynamic_full_path_above_80k": new_dynamic.get("accepted") is True
        and new_dynamic["timing"].get("minimum_transitions_per_second", 0) >= 80000,
        "policy_and_ppo_chain_rebenchmarked": policy_perf.get("measured_at", "").startswith(
            "2026-08-15"
        )
        and ppo_steps
        and all(step == 1 for step in ppo_steps),
    }
    payload = {
        "schema": "kaggriculture_official_1327_migration_acceptance_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "official_package_version": "1.32.7",
        "gates": gates,
        "rule_delta": {
            "changed_products": ["CARROT", "TOMATO", "EGG"],
            "changed_scope": "scarcity-side market price curve",
            "unchanged_scope": [
                "action schema and legality",
                "crop and animal lifecycle",
                "town and shop schedule",
                "weed and shop RNG process",
                "719 action transitions and terminal behavior",
            ],
            "event_bank_regeneration_required": False,
        },
        "performance": {
            "core_simulator_env_transitions_per_s": {
                str(row["batch_size"]): row["env_transitions_per_s"]
                for row in core_perf["rows"]
            },
            "dynamic_full_v2_batch_2048_steps_719": {
                "old_1_32_6_median": old_median,
                "new_1_32_7_median": new_median,
                "median_delta_percent": 100.0 * (new_median / old_median - 1.0),
                "new_minimum": new_dynamic["timing"]["minimum_transitions_per_second"],
                "old_peak_device_bytes": old_dynamic["device_memory_stats"]["peak_bytes_in_use"],
                "new_peak_device_bytes": new_dynamic["device_memory_stats"]["peak_bytes_in_use"],
                "attribution_warning": (
                    "The current Strategic V5 source hashes differ from the historical freeze, "
                    "so the delta cannot be attributed only to the price dtype/rule update."
                ),
            },
            "ppo_batch_1024_env_transitions_per_s": next(
                row["env_transitions_per_s"]
                for row in policy_perf["full_ppo_iteration"]
                if row["batch_size"] == 1024
            ),
        },
        "legacy_boundary": {
            "route_playbook_1_32_6_results": "PRESERVED_BUT_NOT_PROMOTABLE",
            "third_party_agents": "NOT_MUTATED",
            "old_checkpoints": "LOADABLE_BUT_REQUIRE_1_32_7_REEVALUATION",
            "route_rebaseline_required": True,
        },
        "artifact_sha256": {
            "official_wheel": sha256(
                MIGRATION / "download" / "kaggle_environments-1.32.7-py3-none-any.whl"
            ),
            "official_source": sha256(official_source),
            "static_tables_v2": sha256(GPU / "reference" / "static_tables_v2.npz"),
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(OUTPUT), "gates": gates}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
