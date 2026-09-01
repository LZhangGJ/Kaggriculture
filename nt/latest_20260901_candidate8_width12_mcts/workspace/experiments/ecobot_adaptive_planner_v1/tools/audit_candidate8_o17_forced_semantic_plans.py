#!/usr/bin/env python3
"""Execute O1.7 action-free semantic plans through the current planner."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
META_AGENT = (
    ROOT
    / "research"
    / "team_mate"
    / "Kaggriculture_main_512631c"
    / "agents"
    / "route_clustering_switch_agent"
    / "src"
)
sys.path.insert(0, str(META_AGENT))

from audit_candidate8_multifuture_oracle import load_genome, sha256  # noqa: E402
from meta_agent.src.native_teammate_executor import NativeTeammateBundle  # noqa: E402


PROFILES = ("full", "target", "milestone")
FORBIDDEN_PLAN_TOKENS = ("action", "coordinate", "opponent", "identity", "seed")


def _case_key(row: dict) -> tuple[str, int, int]:
    return str(row["opponent"]), int(row["seed"]), int(row["seat"])


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _classify(profile: dict[str, dict], reference_margin: float, base_margin: float) -> str:
    milestone = profile["milestone"]
    target = profile["target"]
    full = profile["full"]
    if milestone["won"]:
        return "COARSE_MILESTONE_CANDIDATE_GAP"
    if target["won"]:
        return "DAILY_TARGET_CANDIDATE_GAP"
    if full["won"]:
        return "DAILY_SEMANTIC_CANDIDATE_GAP"
    denominator = max(reference_margin - base_margin, 1.0)
    best_recovery = max(
        (value["margin"] - base_margin) / denominator for value in profile.values()
    )
    if best_recovery >= 0.5:
        return "MIXED_CANDIDATE_AND_SOFT_COMPILER_GAP"
    return "SOFT_COMPILER_SCHEDULER_OR_TRANSACTION_PRIMARY"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--genomes", required=True, type=Path)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    attribution = json.loads(args.attribution.read_text(encoding="utf-8"))
    controls = json.loads(args.controls.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["status"] != "PASS" or manifest["cases"] != 36:
        raise RuntimeError("O1.7 semantic-plan compile manifest is not accepted")
    cases = [
        row
        for row in attribution["rows"]
        if row["category"] == "SEMANTIC_CANDIDATE_OR_SOFT_COMPILER_GAP"
    ]
    if args.case_limit is not None:
        cases = cases[: max(args.case_limit, 0)]
    manifest_rows = {int(row["case_id"]): row for row in manifest["rows"]}
    portfolio = {_case_key(row): row for row in controls["portfolio_rows"]}
    genome = load_genome(args.genomes, args.genome_index)

    rows: list[dict] = []
    started = time.perf_counter()
    for case_id, case in enumerate(cases):
        key = _case_key(case)
        reference = portfolio[key]
        compiled = manifest_rows[case_id]
        if _case_key(compiled) != key:
            raise RuntimeError(f"manifest case ordering mismatch for case {case_id}")
        profile_rows: dict[str, dict] = {}
        for profile in PROFILES:
            plan_path = args.plan_dir / f"case_{case_id:03d}_{profile}.npz"
            expected_hash = str(compiled["profiles"][profile]["sha256"])
            if _digest(plan_path) != expected_hash:
                raise RuntimeError(f"semantic plan hash mismatch: {plan_path}")
            with np.load(plan_path, allow_pickle=False) as payload:
                lower_names = [name.lower() for name in payload.files]
                forbidden = [
                    name
                    for name in lower_names
                    if any(token in name for token in FORBIDDEN_PLAN_TOKENS)
                ]
                if forbidden:
                    raise RuntimeError(
                        f"forbidden semantic plan fields in {plan_path}: {forbidden}"
                    )

            bundle = NativeTeammateBundle(
                args.source, args.actions, args.metadata, plan_path
            )
            seat = int(case["seat"])
            result = bundle.adaptive_executor.play_forced_backbone(
                genome,
                bundle.index(str(case["opponent"])),
                int(case["seed"]),
                seat,
                False,
            )
            own = float(result["rewards"][seat])
            rival = float(result["rewards"][1 - seat])
            margin = own - rival
            reference_gap = float(reference["margin"]) - float(
                case["best_search"]["margin"]
            )
            recovery = (
                margin - float(case["best_search"]["margin"])
            ) / max(reference_gap, 1.0)
            hard_issue = bool(
                int(result["end_overflow"]) > 0
                or int(result["avoidable_crop_losses"]) > 0
                or int(result["avoidable_animal_losses"]) > 0
            )
            profile_rows[profile] = {
                "own_cash": own,
                "opponent_cash": rival,
                "margin": margin,
                "won": own > rival,
                "margin_recovery_ratio": recovery,
                "reference_margin_gap": float(reference["margin"]) - margin,
                "end_overflow": int(result["end_overflow"]),
                "avoidable_crop_losses": int(result["avoidable_crop_losses"]),
                "avoidable_animal_losses": int(result["avoidable_animal_losses"]),
                "hard_execution_issue": hard_issue,
                "replans": int(result["replans"]),
                "override_actions": int(result["override_actions"]),
                "max_crop_targets": [int(value) for value in result["max_crop_targets"]],
                "max_animal_targets": [
                    int(value) for value in result["max_animal_targets"]
                ],
                "plan_path": str(plan_path),
                "plan_sha256": expected_hash,
            }
        category = _classify(
            profile_rows,
            float(reference["margin"]),
            float(case["best_search"]["margin"]),
        )
        rows.append(
            {
                "case_id": case_id,
                "opponent": str(case["opponent"]),
                "seed": int(case["seed"]),
                "seat": int(case["seat"]),
                "best_width_margin": float(case["best_search"]["margin"]),
                "reference_route": str(reference["best_route"]),
                "reference_own_cash": float(reference["own_cash"]),
                "reference_opponent_cash": float(reference["opponent_cash"]),
                "reference_margin": float(reference["margin"]),
                "profiles": profile_rows,
                "full_flow_harm": (
                    profile_rows["full"]["margin"]
                    < profile_rows["target"]["margin"]
                ),
                "category": category,
            }
        )
        print(
            json.dumps(
                {
                    "progress": len(rows),
                    "total": len(cases),
                    "case": case_id,
                    "opponent": case["opponent"],
                    "margins": {
                        name: profile_rows[name]["margin"] for name in PROFILES
                    },
                    "category": category,
                }
            ),
            flush=True,
        )

    categories = Counter(row["category"] for row in rows)
    summary_profiles = {}
    for profile in PROFILES:
        values = [row["profiles"][profile] for row in rows]
        summary_profiles[profile] = {
            "wins": sum(int(value["won"]) for value in values),
            "win_rate": sum(int(value["won"]) for value in values) / max(len(values), 1),
            "mean_own_cash": float(np.mean([value["own_cash"] for value in values])),
            "mean_margin": float(np.mean([value["margin"] for value in values])),
            "mean_margin_recovery_ratio": float(
                np.mean([value["margin_recovery_ratio"] for value in values])
            ),
            "hard_execution_issue_cases": sum(
                int(value["hard_execution_issue"]) for value in values
            ),
        }
    payload = {
        "schema": "kaggriculture.candidate8-o17-forced-semantic-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AUDIT_COMPLETE",
        "boundary": (
            "Offline posthoc attribution only. Each plan is compiled from a known "
            "winning same-case route, but the tested executor receives only semantic "
            "daily targets/obligations. It receives no raw action, coordinate, route "
            "identity, opponent identity or future random event."
        ),
        "cases": len(rows),
        "profiles": summary_profiles,
        "any_forced_profile_wins": sum(
            int(any(row["profiles"][name]["won"] for name in PROFILES))
            for row in rows
        ),
        "category_counts": dict(categories),
        "full_flow_harm_cases": sum(int(row["full_flow_harm"]) for row in rows),
        "simulation_seconds": time.perf_counter() - started,
        "inputs": {
            "attribution": {"path": str(args.attribution), "sha256": sha256(args.attribution)},
            "controls": {"path": str(args.controls), "sha256": sha256(args.controls)},
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "genomes": {"path": str(args.genomes), "sha256": sha256(args.genomes)},
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "cases": payload["cases"],
                "profiles": payload["profiles"],
                "category_counts": payload["category_counts"],
                "simulation_seconds": payload["simulation_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
