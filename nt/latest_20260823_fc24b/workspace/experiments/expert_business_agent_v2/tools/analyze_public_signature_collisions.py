#!/usr/bin/env python3
"""Audit whether a legal public-state decision point separates opponents.

The report intentionally excludes opponent names, seeds, rewards, and private
inventories from the signature itself.  Names are attached only after hashing
so the offline audit can reveal collisions that a deployable Agent cannot
resolve at that decision point.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


def public_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "shops": snapshot.get("shops", []),
        "prices": snapshot.get("prices", {}),
        "inventory": snapshot.get("inventory", {}),
        "own_stats": snapshot.get("own_stats", {}),
        "opponent_stats": snapshot.get("opponent_stats", {}),
    }


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    rows = [row for row in matrix["rows"] if str(row["candidate"]) == args.candidate]
    if not rows:
        raise ValueError(f"candidate not found: {args.candidate}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    payloads: dict[str, dict[str, Any]] = {}
    for row in rows:
        snapshot = row.get("public_state_steps", {}).get(str(args.step))
        if not isinstance(snapshot, dict):
            raise ValueError(f"step {args.step} snapshot missing")
        payload = public_payload(snapshot)
        digest = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest().upper()
        payloads[digest] = payload
        grouped[digest].append(row)

    groups = []
    for digest, members in sorted(grouped.items()):
        opponents = sorted({str(row["opponent"]) for row in members})
        groups.append(
            {
                "signature_sha256": digest,
                "opponent_count": len(opponents),
                "opponents": opponents,
                "row_count": len(members),
                "seeds": sorted({int(row["seed"]) for row in members}),
                "candidate_seats": sorted({int(row["candidate_seat"]) for row in members}),
                "public_payload": payloads[digest],
            }
        )

    result = {
        "schema": "kaggriculture-public-signature-collision-audit-v1",
        "source_matrix": str(args.matrix.resolve()),
        "candidate": args.candidate,
        "decision_step": args.step,
        "runtime_forbidden_inputs": [
            "opponent identity",
            "seed",
            "reward",
            "future events",
            "private opponent state",
        ],
        "row_count": len(rows),
        "opponent_count": len({str(row["opponent"]) for row in rows}),
        "signature_count": len(groups),
        "collision_group_count": sum(group["opponent_count"] > 1 for group in groups),
        "colliding_opponent_count": len(
            {
                opponent
                for group in groups
                if group["opponent_count"] > 1
                for opponent in group["opponents"]
            }
        ),
        "groups": sorted(
            groups,
            key=lambda group: (-group["opponent_count"], group["signature_sha256"]),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in (
                "candidate",
                "decision_step",
                "row_count",
                "opponent_count",
                "signature_count",
                "collision_group_count",
                "colliding_opponent_count",
            )},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
