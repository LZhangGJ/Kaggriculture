#!/usr/bin/env python3
"""Freeze FC24B acceptance facts and SHA-256 provenance into one manifest."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/fusion_champion_v1/receipts/fc24b_frozen_acceptance_manifest_v1.json"


FILES = {
    "latest6_freeze_tool": ROOT / "experiments/expert_business_agent_v2/tools/freeze_latest_public6_20260822.py",
    "latest6_route_builder": ROOT / "experiments/expert_business_agent_v2/tools/build_latest_public6_20260822_route_bank.py",
    "latest6_parity_tool": ROOT / "experiments/expert_business_agent_v2/tools/accept_latest_public6_20260822_jax_parity.py",
    "latest6_source_manifest": ROOT / "references/public_latest6_20260822/manifest.json",
    "latest6_route_bank_receipt": ROOT / "experiments/expert_business_agent_v2/receipts/latest_public6_20260822_route_bank_v1.json",
    "latest6_route_bank": ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz",
    "latest6_jax_controller": ROOT / "experiments/strategic_v5/src/strategic_v5/latest_public6_20260822_gpu.py",
    "boatlee_v21_jax_parity": ROOT / "experiments/expert_business_agent_v2/receipts/boatlee_v21_latest_jax_parity_seed822001_n2x2_v3.json",
    "soil_v26h_jax_parity": ROOT / "experiments/expert_business_agent_v2/receipts/prvsiyan_soil_v26h_latest_jax_parity_seed822001_n2x2_v8.json",
    "moon_v92_jax_parity": ROOT / "experiments/expert_business_agent_v2/receipts/prvsiyan_moon_v92_latest_jax_parity_seed822001_n2x2_v8.json",
    "kaito_v39_jax_parity": ROOT / "experiments/expert_business_agent_v2/receipts/kaito_v39_history_gate_latest_jax_parity_seed822001_n2x2_v1.json",
    "steven_e284_jax_parity": ROOT / "experiments/expert_business_agent_v2/receipts/steven_e284_hadouken_latest_jax_parity_seed822001_n2x2_v3.json",
    "salem_latest_jax_parity": ROOT / "experiments/expert_business_agent_v2/receipts/salem_harvestforge_x_latest_jax_parity_seed822001_n2x2_v1.json",
    "jax_policy": ROOT / "experiments/fusion_champion_v1/src/fusion_champion_v1/policy_gpu.py",
    "cpu_builder": ROOT / "experiments/fusion_champion_v1/tools/build_fc24b_cpu_submission.py",
    "cpu_agent": ROOT / "experiments/fusion_champion_v1/artifacts/fc24b_cpu_v1/main.py",
    "official_trace_tool": ROOT / "experiments/expert_business_agent_v2/tools/generate_official_stepwise_parity_traces.py",
    "fc24b_parity_tool": ROOT / "experiments/fusion_champion_v1/tools/accept_fc12g_jax_stepwise_parity.py",
    "cpu_latency_tool": ROOT / "experiments/fusion_champion_v1/tools/benchmark_fc12g_cpu.py",
    "cpu_reentry_tool": ROOT / "experiments/fusion_champion_v1/tools/validate_fc24b_reentry.py",
    "freeze_tool": ROOT / "experiments/fusion_champion_v1/tools/freeze_fc24b_acceptance.py",
    "frozen_config": ROOT / "experiments/fusion_champion_v1/configs/fc24b_frozen_best_v1.json",
    "current_best_pointer": ROOT / "experiments/fusion_champion_v1/configs/CURRENT_BEST_RULE_CONFIG.json",
    "final_report": ROOT / "experiments/fusion_champion_v1/reports/FC24B_FINAL_OFFICIAL_JAX_ACCEPTANCE_20260823_ZH.md",
    "latest6_panel": ROOT / "experiments/fusion_champion_v1/receipts/fc24b_value_guard_vs_latest6_seed839001_n128x2_v1.json",
    "old28_panel": ROOT / "experiments/fusion_champion_v1/receipts/fc24b_value_guard_vs_old_full_roster_seed594001_n128x2_v1.json",
    "standard_official_traces": ROOT / "experiments/fusion_champion_v1/receipts/fc24b_cpu_v1_official_stepwise_traces_seed594001_n8x2_v1.json",
    "standard_jax_parity": ROOT / "experiments/fusion_champion_v1/receipts/fc24b_cpu_v1_jax_stepwise_parity_seed594001_n8x2_v1.json",
    "targeted_official_traces": ROOT / "experiments/fusion_champion_v1/receipts/fc24b_cpu_v1_official_stepwise_traces_seed594122_targeted_v1.json",
    "targeted_jax_parity": ROOT / "experiments/fusion_champion_v1/receipts/fc24b_cpu_v1_jax_stepwise_parity_seed594122_targeted_v1.json",
    "cpu_latency": ROOT / "experiments/fusion_champion_v1/receipts/fc24b_cpu_v1_cpu_latency_and_action_acceptance_v1.json",
    "cpu_reentry": ROOT / "experiments/fusion_champion_v1/receipts/fc24b_cpu_v1_reentry_forward_reverse_v1.json",
}


LATEST6_PARITY_NAMES = (
    "boatlee_v21_jax_parity",
    "soil_v26h_jax_parity",
    "moon_v92_jax_parity",
    "kaito_v39_jax_parity",
    "steven_e284_jax_parity",
    "salem_latest_jax_parity",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(name: str) -> dict:
    return json.loads(FILES[name].read_text(encoding="utf-8"))


def critical_trace() -> dict:
    receipt = load("targeted_official_traces")
    row = next(
        value
        for value in receipt["traces"]
        if value["opponent"] == "local_prt_v6"
        and int(value["candidate_seat"]) == 0
        and int(value["seed"]) == 594122
    )
    with gzip.open(ROOT / row["path"], "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    frames = records[1:]
    action_709 = frames[710]["actions"][0]
    return {
        "seed": 594122,
        "candidate_seat": 0,
        "opponent": "local_prt_v6",
        "candidate_terminal_cash": int(frames[-1]["reward"][0]),
        "opponent_terminal_cash": int(frames[-1]["reward"][1]),
        "action_step_709": action_709,
        "salvage_harvest_present": any(
            order == ["HARVEST"] for order in action_709["hands"]
        ),
        "trace": {"path": row["path"], "sha256": sha256(ROOT / row["path"])},
    }


def latest6_source_integrity(source_manifest: dict) -> tuple[bool, list[dict]]:
    """Verify both downloaded notebooks and statically frozen Python sources."""

    notebook_root = ROOT / "public_notebooks/recent_latest_20260822_scan_v1"
    frozen_root = ROOT / "references/public_latest6_20260822"
    rows = []
    valid = (
        source_manifest.get("schema")
        == "kaggriculture.latest_public6_20260822.v1"
        and len(source_manifest.get("agents", [])) == 6
    )
    for item in source_manifest.get("agents", []):
        notebook_matches = list(notebook_root.rglob(item["notebook_file"]))
        notebook = notebook_matches[0] if len(notebook_matches) == 1 else None
        source = frozen_root / item["slug"] / "main.py"
        notebook_hash = sha256(notebook) if notebook is not None else None
        source_hash = sha256(source) if source.is_file() else None
        row_ok = (
            len(notebook_matches) == 1
            and notebook_hash.lower() == item["notebook_sha256"].lower()
            and source_hash is not None
            and source_hash.lower() == item["main_sha256"].lower()
            and item["main_sha256"].lower()
            == item["expected_main_sha256"].lower()
            and bool(item.get("python_compile_ok"))
            and not bool(item.get("notebook_code_executed"))
        )
        valid = valid and row_ok
        rows.append(
            {
                "slug": item["slug"],
                "kaggle_ref": item["kaggle_ref"],
                "ok": row_ok,
                "notebook": {
                    "path": (
                        str(notebook.relative_to(ROOT)).replace("\\", "/")
                        if notebook is not None
                        else None
                    ),
                    "sha256": notebook_hash,
                },
                "source": {
                    "path": str(source.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": source_hash,
                },
            }
        )
    return valid, rows


def main() -> int:
    latest6_source_manifest = load("latest6_source_manifest")
    latest6_source_ok, latest6_sources = latest6_source_integrity(
        latest6_source_manifest
    )
    latest6_route_bank = load("latest6_route_bank_receipt")
    latest6_parity = {name: load(name) for name in LATEST6_PARITY_NAMES}
    latest = load("latest6_panel")
    old = load("old28_panel")
    standard_trace = load("standard_official_traces")
    targeted_trace = load("targeted_official_traces")
    standard = load("standard_jax_parity")
    targeted = load("targeted_jax_parity")
    latency = load("cpu_latency")
    reentry = load("cpu_reentry")
    critical = critical_trace()
    frozen_config = load("frozen_config")
    current_best = load("current_best_pointer")
    latest6_parity_ok = all(
        receipt.get("status") == "PASS"
        and bool(receipt.get("strict_exact"))
        and receipt.get("backend") == "gpu"
        and receipt.get("official_package_version") == "1.32.7"
        and bool(receipt.get("seat_swapped"))
        and int(receipt.get("result", {}).get("contexts", 0)) == 4
        and int(receipt.get("result", {}).get("action_exact_contexts", 0)) == 4
        and int(receipt.get("result", {}).get("state_exact_contexts", 0)) == 4
        and int(
            receipt.get("result", {}).get("terminal_reward_exact_contexts", 0)
        )
        == 4
        for receipt in latest6_parity.values()
    )
    current_best_path = ROOT / current_best.get("current_best", "")
    checks = {
        "latest6_download_and_static_freeze_integrity": latest6_source_ok,
        "latest6_route_bank_integrity": latest6_route_bank.get("status") == "PASS"
        and sha256(FILES["latest6_route_bank"]).lower()
        == latest6_route_bank.get("bank_sha256", "").lower(),
        "latest6_all_strict_gpu_jax_parity": latest6_parity_ok,
        "latest6_each_at_least_90pct": latest.get("status") == "PASS"
        and float(latest.get("min_score_rate", 0)) >= 0.90,
        "old28_each_at_least_90pct": old.get("status") == "PASS"
        and float(old.get("min_score_rate", 0)) >= 0.90,
        "all_34_agents_256_games_each": len(latest.get("rows", [])) == 6
        and len(old.get("rows", [])) == 28
        and int(latest.get("games_per_opponent", 0)) == 256
        and int(old.get("games_per_opponent", 0)) == 256,
        "official_version_1_32_7": all(
            value.get("official_package_version") == "1.32.7"
            for value in (
                standard_trace,
                targeted_trace,
                standard,
                targeted,
                *latest6_parity.values(),
            )
        ),
        "standard_32_context_strict_parity": standard.get("status") == "PASS"
        and bool(standard.get("strict_exact"))
        and int(standard.get("result", {}).get("contexts", 0)) == 32,
        "targeted_4_context_strict_parity": targeted.get("status") == "PASS"
        and bool(targeted.get("strict_exact"))
        and int(targeted.get("result", {}).get("contexts", 0)) == 4,
        "critical_salvage_exercised": critical["salvage_harvest_present"]
        and critical["candidate_terminal_cash"] == 154776
        and critical["opponent_terminal_cash"] == 154609,
        "cpu_latency_and_replay_pass": latency.get("status") == "PASS"
        and bool(latency.get("sequential_action_exact"))
        and float(latency.get("max_action_ms", 1e9)) < 1000.0,
        "cross_game_reentry_pass": reentry.get("status") == "PASS"
        and bool(reentry.get("strict_exact"))
        and int(reentry.get("games", 0)) == 64,
        "frozen_best_pointer_integrity": frozen_config.get("name")
        == "fc24b_frozen_best_v1"
        and current_best.get("status") == "FROZEN_LOCAL_CHAMPION"
        and current_best_path == FILES["frozen_config"]
        and current_best.get("sha256") == sha256(FILES["frozen_config"]),
        "frozen_config_source_hashes_match": all(
            (ROOT / relative).is_file()
            and expected == sha256(ROOT / relative)
            for relative, expected in frozen_config.get(
                "source_hashes", {}
            ).items()
        ),
    }
    passed = all(checks.values())
    payload = {
        "schema": "kaggriculture.fusion_champion.fc24b-frozen-acceptance.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "candidate": "FC24B",
        "official_package_version": "1.32.7",
        "checks": checks,
        "latest6_migration": {
            "sources": latest6_sources,
            "route_bank": {
                "routes": latest6_route_bank.get("unique_routes"),
                "sha256": latest6_route_bank.get("bank_sha256"),
            },
            "strict_parity": {
                name: {
                    "agent": receipt["agent"],
                    "contexts": receipt["result"]["contexts"],
                    "action_exact_contexts": receipt["result"][
                        "action_exact_contexts"
                    ],
                    "state_exact_contexts": receipt["result"][
                        "state_exact_contexts"
                    ],
                    "terminal_exact_contexts": receipt["result"][
                        "terminal_reward_exact_contexts"
                    ],
                    "backend": receipt["backend"],
                    "jax_version": receipt["jax_version"],
                }
                for name, receipt in latest6_parity.items()
            },
        },
        "arena": {
            "agents": 34,
            "games": 8704,
            "latest6_min_score_rate": latest["min_score_rate"],
            "old28_min_score_rate": old["min_score_rate"],
        },
        "strict_parity": {
            "standard_contexts": standard["result"]["contexts"],
            "targeted_contexts": targeted["result"]["contexts"],
            "action_exact_contexts": standard["result"]["action_exact_contexts"]
            + targeted["result"]["action_exact_contexts"],
            "state_exact_contexts": standard["result"]["state_exact_contexts"]
            + targeted["result"]["state_exact_contexts"],
            "terminal_exact_contexts": standard["result"]["terminal_reward_exact_contexts"]
            + targeted["result"]["terminal_reward_exact_contexts"],
        },
        "critical_prt_rescue": critical,
        "cpu": {
            "action_calls": latency["action_calls"],
            "mean_action_ms": latency["mean_action_ms"],
            "p95_action_ms": latency["p95_action_ms"],
            "max_action_ms": latency["max_action_ms"],
            "reentry_action_calls": reentry["action_calls"],
        },
        "files": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in FILES.items()
        },
        "boundary": (
            "This freezes local official-1.32.7 and JAX acceptance only. "
            "It does not claim future leaderboard performance or submit to Kaggle."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": checks, "output": str(OUT)}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
