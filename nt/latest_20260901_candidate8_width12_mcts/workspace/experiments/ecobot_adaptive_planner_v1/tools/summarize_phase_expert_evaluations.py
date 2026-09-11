#!/usr/bin/env python3
"""Combine disjoint early/middle/late frozen evaluations into one receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def combine_pairwise(parts: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for index in range(len(parts[0])):
        rows = [part[index] for part in parts]
        pairs = sum(int(row["pairs"]) for row in rows)
        correct = sum(float(row["accuracy"]) * int(row["pairs"]) for row in rows)
        output.append({
            "minimum_expected_value_gap": float(rows[0]["minimum_expected_value_gap"]),
            "pairs": pairs,
            "accuracy": correct / pairs if pairs else 0.0,
        })
    return output


def combine_policy(parts: list[dict[str, object]]) -> dict[str, object]:
    groups = sum(int(part["groups"]) for part in parts)
    weighted = lambda name: (
        sum(float(part[name]) * int(part["groups"]) for part in parts) / groups
        if groups else 0.0
    )
    keys = sorted({key for part in parts for key in part["topk_oracle_recall_with_keep"]})
    return {
        "groups": groups,
        "mean_realized_delta": weighted("mean_realized_delta"),
        "minimum_realized_delta_across_phases": min(
            float(part["min_realized_delta"]) for part in parts
        ),
        "positive_choice_rate": weighted("positive_choice_rate"),
        "negative_choice_rate": weighted("negative_choice_rate"),
        "switch_rate": weighted("switch_rate"),
        "top1_oracle_recall_with_keep": weighted("top1_oracle_recall_with_keep"),
        "top3_oracle_recall_with_keep": weighted("top3_oracle_recall_with_keep"),
        "topk_oracle_recall_with_keep": {
            key: sum(
                float(part["topk_oracle_recall_with_keep"][key]) * int(part["groups"])
                for part in parts
            ) / groups
            for key in keys
        },
    }


def file_entry(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", type=Path, action="append", required=True)
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument("--global-receipt", type=Path, required=True)
    parser.add_argument("--global-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.expert) != 3 or len(args.raw) != 3:
        raise ValueError("exactly three --expert and three --raw receipts are required")

    experts = [read(path) for path in args.expert]
    raw = [read(path) for path in args.raw]
    global_receipt = read(args.global_receipt)
    global_raw = read(args.global_raw)
    expert_pairwise = [
        part["selected_test_pairwise_not_used_for_selection"] for part in experts
    ]
    expert_policy = [
        part["selected_test_policy_not_used_for_selection"] for part in experts
    ]
    raw_policy = [part["policy"] for part in raw]
    payload = {
        "schema": "kaggriculture.switch-phase-experts-combined-evaluation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "invariants": {
            "phase_corpora_are_disjoint": True,
            "test_not_used_for_model_or_threshold_selection": True,
            "final_blind_opened": False,
        },
        "inputs": {
            "experts": [file_entry(path) for path in args.expert],
            "raw_screens": [file_entry(path) for path in args.raw],
            "global": file_entry(args.global_receipt),
            "global_raw": file_entry(args.global_raw),
        },
        "phase_experts": {
            "pairwise": combine_pairwise(expert_pairwise),
            "selected_policy": combine_policy(expert_policy),
            "raw_screening_policy": combine_policy(raw_policy),
        },
        "global_shared_model": {
            "pairwise": global_receipt["selected_test_pairwise_not_used_for_selection"],
            "selected_policy": global_receipt["selected_test_policy_not_used_for_selection"],
            "raw_screening_policy": global_raw["policy"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
