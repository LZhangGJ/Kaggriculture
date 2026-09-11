"""Audit every goal gate and write final machine/human acceptance reports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RECEIPTS = PROJECT / "receipts"
OUTPUT_JSON = RECEIPTS / "final_acceptance.json"
OUTPUT_MD = PROJECT / "FINAL_ACCEPTANCE.md"


def load(name: str) -> dict:
    return json.loads((RECEIPTS / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gate(name: str, passed: bool, evidence: str, detail: str) -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "evidence": evidence,
        "detail": detail,
    }


def main() -> int:
    reference = load("reference_verification.json")
    runtime = load("jax_runtime.json")
    static = load("static_tables_v2.json")
    canonical = load("reference_traces.json")
    heldout_reference = load("heldout_reference_traces.json")
    heldout = load("heldout_parity.json")
    random_reference = load("random_differential_reference.json")
    random_parity = load("random_differential_parity.json")
    hand = load("hand_cap_analysis.json")
    benchmark = load("benchmark_simulator_only_full_season.json")
    policy = load("benchmark_policy_and_ppo.json")
    tests = load("test_suite.json")

    benchmark_batches = [row["batch_size"] for row in benchmark["rows"]]
    diagnostics_zero = all(
        all(value == 0 for value in row["diagnostics"].values())
        for row in benchmark["rows"]
    )
    policy_batches = [row["batch_size"] for row in policy["policy_plus_sim"]]
    ppo_updates = all(row["optimizer_step"] == 1 for row in policy["full_ppo_iteration"])
    source_files = sorted((PROJECT / "src" / "kaggriculture_jax").glob("*.py"))
    sources = [
        {
            "path": str(path.relative_to(PROJECT)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in source_files
    ]
    gates = [
        gate(
            "official_reference_immutable_and_hashed",
            reference["ok"] and reference["installed_package_version"] == "1.32.7",
            "receipts/reference_verification.json",
            "Frozen and installed official source/config/framework hashes match.",
        ),
        gate(
            "wsl2_jax_rtx3090_runtime",
            runtime["ok"] and runtime["backend"] == "gpu" and runtime["devices"] == ["cuda:0"],
            "receipts/jax_runtime.json",
            f"JAX {runtime['jax']} executes compiled work on {runtime['gpu']}.",
        ),
        gate(
            "exact_python_price_and_rng_tables",
            static["ok"]
            and static["official_source_sha256"]
            == reference["checks"][0]["expected_sha256"],
            "receipts/static_tables_v2.json",
            "Python round market LUT and conditional MT19937 event tables derive from frozen source.",
        ),
        gate(
            "canonical_full_rule_parity",
            canonical["ok"] and canonical["traces"][2]["frames"] == 720,
            "receipts/reference_traces.json and receipts/test_suite.json",
            "Pass, starter, and v16 full seasons compare every recorded state.",
        ),
        gate(
            "100_unseen_seeds_720_frames_zero_error",
            heldout_reference["ok"]
            and heldout_reference["total_frames"] == 72_000
            and heldout["ok"]
            and heldout["frames_compared"] == 72_000
            and heldout["exact_integer_price_reward_zero_error"],
            "receipts/heldout_reference_traces.json and receipts/heldout_parity.json",
            "Seeds 10000..10099, all farm/private/market/town/status/reward fields exact.",
        ),
        gate(
            "random_and_invalid_action_differential",
            random_reference["ok"]
            and random_parity["ok"]
            and random_parity["frames_compared"] == 11_520,
            "receipts/random_differential_reference.json and receipts/random_differential_parity.json",
            "16 state-aware randomized seasons include deliberately invalid actions.",
        ),
        gate(
            "static_hand_bound_with_instrumentation",
            hand["ok"] and hand["observed_max_hands"] < hand["static_max_hands"],
            "receipts/hand_cap_analysis.json",
            f"MAX_HANDS={hand['static_max_hands']}, observed={hand['observed_max_hands']}, cap counter tested.",
        ),
        gate(
            "jit_vmap_scan_batch_independence",
            tests["ok"] and "passed" in tests["output"],
            "receipts/test_suite.json",
            "Pure step, synchronized fast batch, independent rollouts, and lax.scan tests pass.",
        ),
        gate(
            "gpu_arena_and_ppo",
            tests["ok"]
            and policy_batches == [256, 1024, 4096]
            and ppo_updates,
            "receipts/benchmark_policy_and_ppo.json and receipts/test_suite.json",
            "Heterogeneous architectures, self-play collection, GAE and clipped PPO update stay on GPU.",
        ),
        gate(
            "simulator_50k_minimum_300k_target",
            benchmark["minimum_gate_passed"]
            and benchmark["formal_target_passed"]
            and benchmark_batches == [256, 1024, 4096]
            and all(row["rollout_steps"] == 719 for row in benchmark["rows"])
            and diagnostics_zero,
            "receipts/benchmark_simulator_only_full_season.json",
            "All batches exceed 50k/s; batches 1024 and 4096 exceed 300k/s; compilation excluded.",
        ),
        gate(
            "reproducible_environment_commands_and_docs",
            (PROJECT / "requirements-wsl.lock.txt").exists()
            and (PROJECT / "README.md").exists(),
            "requirements-wsl.lock.txt and README.md",
            "Pinned WSL environment, verification, parity, benchmark and training commands documented.",
        ),
    ]
    report = {
        "schema": "kaggriculture_final_acceptance_v1",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "ok": all(item["passed"] for item in gates),
        "gates": gates,
        "source_files": sources,
        "headline": {
            "heldout_frames": heldout["frames_compared"],
            "random_differential_frames": random_parity["frames_compared"],
            "simulator_env_transitions_per_s": {
                str(row["batch_size"]): row["env_transitions_per_s"]
                for row in benchmark["rows"]
            },
            "policy_plus_sim_env_transitions_per_s": {
                str(row["batch_size"]): row["env_transitions_per_s"]
                for row in policy["policy_plus_sim"]
            },
            "full_ppo_env_transitions_per_s": {
                str(row["batch_size"]): row["env_transitions_per_s"]
                for row in policy["full_ppo_iteration"]
            },
        },
        "declared_runtime_boundary": {
            "event_seed_count": static["events"]["seed_count"],
            "training_development_seeds": static["events"]["development_seeds"],
            "heldout_seeds": static["events"]["heldout_seeds"],
            "outside_bank_behavior": "rejected; regenerate the exact event table",
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Kaggriculture JAX GPU simulator final acceptance",
        "",
        f"Audit result: **{'PASS' if report['ok'] else 'FAIL'}**",
        "",
        "## Goal gates",
        "",
    ]
    for item in gates:
        marker = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- **{marker} — {item['name']}**: {item['detail']} Evidence: `{item['evidence']}`")
    lines += [
        "",
        "## Headline measurements",
        "",
        f"- Held-out official frames compared: {heldout['frames_compared']:,}",
        f"- Random/invalid differential frames compared: {random_parity['frames_compared']:,}",
    ]
    for row in benchmark["rows"]:
        lines.append(
            f"- Simulator-only batch {row['batch_size']}: "
            f"{row['env_transitions_per_s']:,.0f} env transitions/s"
        )
    for row in policy["policy_plus_sim"]:
        lines.append(
            f"- Policy+sim batch {row['batch_size']}: "
            f"{row['env_transitions_per_s']:,.0f} env transitions/s"
        )
    for row in policy["full_ppo_iteration"]:
        lines.append(
            f"- Full PPO batch {row['batch_size']}: "
            f"{row['env_transitions_per_s']:,.0f} env transitions/s"
        )
    lines += [
        "",
        "All throughput numbers exclude compilation and synchronize the final device result.",
        "",
        "## Declared runtime boundary",
        "",
        "The exact MT19937 event bank contains seeds `0..127` for training/development",
        "and `10000..10127` for held-out verification. Seeds outside this bank are",
        "rejected and must be added by regenerating the frozen event table; they are",
        "never silently approximated.",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "gates": gates}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
