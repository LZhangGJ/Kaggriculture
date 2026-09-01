#!/usr/bin/env python3
"""Independently verify the Top-60 Candidate8 Beam-4/12 cascade receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases_payload = json.loads(args.cases.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    checkpoint = [
        json.loads(line)
        for line in args.checkpoint.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = receipt["rows"]
    case_ids = [row["route_id"] for row in cases_payload["cases"]]
    row_ids = [row["route_id"] for row in rows]
    checkpoint_ids = [row["route_id"] for row in checkpoint]

    checks = {
        "case_count_153": len(case_ids) == 153,
        "receipt_row_count_153": len(rows) == 153,
        "checkpoint_row_count_153": len(checkpoint) == 153,
        "case_ids_unique": len(set(case_ids)) == len(case_ids),
        "receipt_ids_exact": set(row_ids) == set(case_ids),
        "checkpoint_ids_exact": set(checkpoint_ids) == set(case_ids),
        "dynamic_ranks_absent": not ({1, 4, 9, 49} & {int(row["rank"]) for row in rows}),
        "represented_count_5007": sum(int(row["member_count"]) for row in rows) == 5007,
        "primary_width_4": all(int(row["primary"]["beam_width"]) == 4 for row in rows),
        "fallback_only_after_primary_failure": all(
            (row["fallback"] is None) == bool(row["primary"]["win"]) for row in rows
        ),
        "fallback_width_12": all(
            row["fallback"] is None or int(row["fallback"]["beam_width"]) == 12
            for row in rows
        ),
        "all_replays_exact": all(
            bool(row["primary"]["replay_exact"])
            and (row["fallback"] is None or bool(row["fallback"]["replay_exact"]))
            for row in rows
        ),
        "strict_win_definition": all(
            bool(row["found_winning_path"]) == (float(row["selected"]["margin"]) > 0)
            for row in rows
        ),
        "beam4_wins_134": sum(bool(row["primary"]["win"]) for row in rows) == 134,
        "beam12_rescues_10": sum(bool(row["beam12_rescue"]) for row in rows) == 10,
        "total_wins_144": sum(bool(row["found_winning_path"]) for row in rows) == 144,
        "weighted_wins_4700": sum(
            int(row["member_count"]) for row in rows if row["found_winning_path"]
        ) == 4700,
    }
    payload = {
        "schema": "kaggriculture-candidate8-top60-winning-path-acceptance-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
