#!/usr/bin/env python3
"""Freeze the behaviorally distinct local opponent pool for EBA v2.

The pool deliberately excludes historical/debug copies.  It contains one
current-engine representative for every frozen public family, all current
gold-replay imitation agents, and the two local production anchors (PRT/V8).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/configs/local_representative_pool_v1.json",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/receipts/local_representative_pool_freeze_v1.json",
    )
    args = parser.parse_args()

    public_manifest_path = ROOT / "experiments/public_solution_analysis/dedupe_representative_manifest_20260814.json"
    public_manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
    entries: list[dict[str, object]] = []

    for group in public_manifest["groups"]:
        if group.get("current_engine_status") != "eligible":
            continue
        path = ROOT / str(group["executable_agent_path"])
        entries.append({
            "name": f"public_{str(group['group_id']).lower()}",
            "class": "public_deduplicated_representative",
            "family": str(group["name"]),
            "path": rel(path),
            "sha256": sha256(path),
        })

    gold_root = ROOT / "experiments/gold_adaptive_rule_v2/agents/gold_imitations_current_20260816_0955_v2"
    for directory in sorted(path for path in gold_root.iterdir() if path.is_dir()):
        path = directory / "main.py"
        if not path.is_file():
            continue
        entries.append({
            "name": f"gold_proxy_{directory.name}",
            "class": "current_gold_replay_imitation",
            "family": directory.name,
            "path": rel(path),
            "sha256": sha256(path),
        })

    anchors = [
        (
            "local_prt_v6",
            "experiments/gold_adaptive_rule_v2/agents/pure_public_route_tree_v6/main.py",
        ),
        (
            "local_public_opening_router_v8",
            "experiments/gold_adaptive_rule_v2/agents/public_opening_router_v8/main.py",
        ),
    ]
    for name, path_text in anchors:
        path = ROOT / path_text
        entries.append({
            "name": name,
            "class": "local_production_anchor",
            "family": name,
            "path": rel(path),
            "sha256": sha256(path),
        })

    names = [str(row["name"]) for row in entries]
    missing = [str(row["path"]) for row in entries if not (ROOT / str(row["path"])).is_file()]
    if missing:
        raise FileNotFoundError(f"missing opponent files: {missing}")
    if len(names) != len(set(names)):
        raise ValueError("opponent names are not unique")

    exact_sha_groups: dict[str, list[str]] = {}
    for row in entries:
        exact_sha_groups.setdefault(str(row["sha256"]), []).append(str(row["name"]))
    duplicate_sha_groups = [group for group in exact_sha_groups.values() if len(group) > 1]

    pool = {
        "schema": "kaggriculture-eba-v2-local-representative-pool-v1",
        "official_package_version": "1.32.7",
        "scope": {
            "public_behavior_groups": 16,
            "current_gold_replay_imitation_agents": 19,
            "local_production_anchors": 2,
            "historical_debug_or_exact_copy_variants": "excluded",
        },
        "acceptance_target": {
            "metric": "seat-swapped official Python win rate per opponent",
            "target_per_opponent": 0.90,
            "ties_count_as_wins": False,
        },
        "opponents": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    receipt = {
        "schema": "kaggriculture-eba-v2-local-representative-pool-freeze-v1",
        "status": "PASS" if len(entries) == 37 and not missing else "FAIL",
        "pool_path": rel(args.output),
        "pool_sha256": sha256(args.output),
        "source_public_manifest": rel(public_manifest_path),
        "source_public_manifest_sha256": sha256(public_manifest_path),
        "counts": {
            "total": len(entries),
            "public": sum(row["class"] == "public_deduplicated_representative" for row in entries),
            "gold_proxy": sum(row["class"] == "current_gold_replay_imitation" for row in entries),
            "local_anchor": sum(row["class"] == "local_production_anchor" for row in entries),
        },
        "exact_sha_duplicate_groups": duplicate_sha_groups,
        "missing": missing,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
