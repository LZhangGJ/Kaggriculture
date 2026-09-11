#!/usr/bin/env python3
"""Measure route coverage inside indistinguishable public-state collisions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def public_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "shops": snapshot.get("shops", []),
        "prices": snapshot.get("prices", {}),
        "inventory": snapshot.get("inventory", {}),
        "own_stats": snapshot.get("own_stats", {}),
        "opponent_stats": snapshot.get("opponent_stats", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--reference-candidate", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.matrix.read_text(encoding="utf-8"))
    candidates = [value.strip() for value in args.candidates.split(",") if value.strip()]
    by_case: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in payload["rows"]:
        if row["candidate"] in candidates or row["candidate"] == args.reference_candidate:
            key = (str(row["opponent"]), int(row["seed"]), int(row["candidate_seat"]))
            by_case[key][str(row["candidate"])] = row

    signatures: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    public_by_signature = {}
    for key, rows in by_case.items():
        if args.reference_candidate not in rows:
            continue
        missing = set(candidates) - set(rows)
        if missing:
            raise ValueError(f"missing {sorted(missing)} for {key}")
        snapshot = rows[args.reference_candidate]["public_state_steps"][str(args.step)]
        public = public_payload(snapshot)
        digest = hashlib.sha256(canonical(public).encode()).hexdigest().upper()
        signatures[digest].append(key)
        public_by_signature[digest] = public

    groups = []
    for digest, keys in signatures.items():
        coverage = []
        for candidate in candidates:
            rows = [by_case[key][candidate] for key in keys]
            wins = sum(bool(row["win"]) for row in rows)
            coverage.append({
                "candidate": candidate,
                "games": len(rows),
                "wins": wins,
                "win_rate": wins / len(rows),
                "mean_margin": sum(float(row["margin"]) for row in rows) / len(rows),
            })
        coverage.sort(key=lambda row: (row["win_rate"], row["mean_margin"]), reverse=True)
        groups.append({
            "signature_sha256": digest,
            "case_count": len(keys),
            "opponents": sorted({key[0] for key in keys}),
            "seeds": sorted({key[1] for key in keys}),
            "candidate_seats": sorted({key[2] for key in keys}),
            "best_single_route": coverage[0],
            "routes_winning_every_case": [row["candidate"] for row in coverage if row["wins"] == row["games"]],
            "route_coverage": coverage,
            "public_payload": public_by_signature[digest],
        })
    groups.sort(key=lambda group: (-group["case_count"], group["signature_sha256"]))
    output = {
        "schema": "public-route-collision-feasibility-v1",
        "source_matrix": str(args.matrix),
        "reference_candidate": args.reference_candidate,
        "decision_step": args.step,
        "candidates": candidates,
        "runtime_forbidden_inputs": ["opponent identity", "seed", "reward", "future events", "private opponent state"],
        "signature_count": len(groups),
        "collision_group_count": sum(len(group["opponents"]) > 1 for group in groups),
        "collision_groups_without_universal_winner": sum(
            len(group["opponents"]) > 1 and not group["routes_winning_every_case"] for group in groups
        ),
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "signature_count": output["signature_count"],
        "collision_group_count": output["collision_group_count"],
        "collision_groups_without_universal_winner": output["collision_groups_without_universal_winner"],
        "collisions": [
            {
                "opponents": group["opponents"],
                "seeds": group["seeds"],
                "best_single_route": group["best_single_route"],
                "routes_winning_every_case": group["routes_winning_every_case"],
            }
            for group in groups if len(group["opponents"]) > 1
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
