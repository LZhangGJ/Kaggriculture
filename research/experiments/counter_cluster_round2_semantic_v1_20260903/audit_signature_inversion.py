#!/usr/bin/env python3
"""Audit Round1 Candidate8 signatures against Round2 canonical intents.

This is a metadata-only audit.  It reads no evaluation matrix and runs no
simulator.  Candidate fields are inverted from the finite Candidate8 grammar
and the stable FNV-1a signature contract in adaptive_candidates.cpp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROUND1 = HERE.parent / "counter_cluster_round1" / "search_results.json"
MASK64 = (1 << 64) - 1
FNV_BASIS = 1469598103934665603
FNV_PRIME = 1099511628211
FNV_INVERSE = pow(FNV_PRIME, -1, 1 << 64)
CANONICAL_FIELDS = (
    "family_id",
    "target_delta",
    "effective_delay_days",
    "schedule_profile",
    "market_profile",
    "recovery_profile",
    "suffix_project",
    "market_item",
    "recovery_issue",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mix(value_hash: int, value: int) -> int:
    bits = value & MASK64
    for byte in range(8):
        value_hash ^= (bits >> (8 * byte)) & 0xFF
        value_hash = value_hash * FNV_PRIME & MASK64
    return value_hash


def unmix(value_hash: int, value: int) -> int:
    bits = value & MASK64
    for byte in range(7, -1, -1):
        value_hash = value_hash * FNV_INVERSE & MASK64
        value_hash ^= (bits >> (8 * byte)) & 0xFF
    return value_hash


def signature(candidate: dict[str, Any]) -> int:
    value_hash = mix(FNV_BASIS, candidate["family_id"])
    for value in candidate["target_delta"]:
        value_hash = mix(value_hash, value)
    for field in (
        "hand_delta",
        "quadrant_delta",
        "effective_delay_days",
        "schedule_profile",
        "market_profile",
        "recovery_profile",
        "suffix_project",
        "market_item",
        "recovery_issue",
    ):
        value_hash = mix(value_hash, candidate[field])
    return value_hash


def canonical(value: dict[str, Any]) -> dict[str, Any]:
    return {
        field: ([int(item) for item in value[field]] if field == "target_delta"
                else int(value[field]))
        for field in CANONICAL_FIELDS
    }


def candidate(
    family_id: int,
    target_delta: tuple[int, ...],
    hand_delta: int,
    quadrant_delta: int,
    tail: tuple[int, ...],
) -> dict[str, Any]:
    delay, schedule, market, recovery, suffix, item, issue = tail
    return {
        "family_id": family_id,
        "target_delta": list(target_delta),
        "hand_delta": hand_delta,
        "quadrant_delta": quadrant_delta,
        "effective_delay_days": delay,
        "schedule_profile": schedule,
        "market_profile": market,
        "recovery_profile": recovery,
        "suffix_project": suffix,
        "market_item": item,
        "recovery_issue": issue,
    }


def candidate_language() -> tuple[dict[int, set[tuple[int, ...]]],
                                  dict[int, list[tuple[int, ...]]]]:
    """Return a conservative superset of the native raw candidate grammar."""
    zero = (0,) * 8
    targets: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    for family in (0, 1, 6, 8):
        targets[family].add(zero)

    for project in range(8):
        for amount in (1, 2, 3, 4, 6, 8):
            for direction in (-1, 1):
                row = [0] * 8
                row[project] = direction * amount
                targets[2].add(tuple(row))
            row = [0] * 8
            row[project] = amount
            targets[3].add(tuple(row))
        # build_plan caps crops at 100 and animals at <=50; 100 is a safe
        # common upper bound for reversible negative edits.
        for amount in range(1, 101):
            row = [0] * 8
            row[project] = -amount
            targets[3].add(tuple(row))
            targets[5].add(tuple(row))

    for first in range(8):
        for second in range(first + 1, 8):
            for amount in (1, 2, 4, 6):
                for left, right in ((-1, 1), (1, -1), (1, 1)):
                    row = [0] * 8
                    row[first] = left * amount
                    row[second] = right * amount
                    targets[4].add(tuple(row))
    for source in range(8):
        for first in range(8):
            if first == source:
                continue
            for second in range(first + 1, 8):
                if second == source:
                    continue
                for amount in (1, 2):
                    row = [0] * 8
                    row[source] = -amount
                    row[first] = amount
                    row[second] = amount
                    targets[4].add(tuple(row))

    for project in range(5):
        for amount in (2, 4, 6):
            row = [0] * 8
            row[project] = amount
            targets[7].add(tuple(row))
        for amount in range(1, 101):
            row = [0] * 8
            row[project] = -amount
            targets[7].add(tuple(row))

    base = (0, 0, 0, 0, -1, -1, 0)
    tails = {
        0: [base],
        1: [(0, profile, 0, 0, -1, -1, 0) for profile in range(1, 9)],
        2: [base] + [(0, 0, 4, 0, -1, item, 0) for item in range(9)],
        3: [base] + [(0, 0, 4, 0, -1, item, 0) for item in range(9)],
        4: [base] + [(0, 0, 4, 0, -1, item, 0) for item in range(9)],
        5: [(delay, 0, 0, 0, -1, -1, 0) for delay in (1, 2, -1)],
        6: [(0, 0, profile, 0, -1, item, 0)
            for item in range(9) for profile in range(1, 5)],
        7: ([(0, 0, 0, 0, project, -1, 0) for project in range(5)] +
            [(0, 0, 4, 0, project, item, 0)
             for project in range(5) for item in range(9)]),
        8: [(0, 0, 0, profile, -1, -1, issue)
            for issue in (1, 2, 4, 8) for profile in range(1, 4)],
    }
    return targets, tails


def invert(wanted: set[tuple[int, int]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    targets, tails = candidate_language()
    wanted_by_family: dict[int, set[int]] = defaultdict(set)
    for family, value in wanted:
        wanted_by_family[family].add(value)

    # Reverse the known semantic tail of each wanted signature.  Matching the
    # remaining prefix needs only the finite target/hand/quadrant grammar.
    required: dict[int, dict[int, list[tuple[int, tuple[int, ...]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for family, wanted_signatures in wanted_by_family.items():
        for tail in tails[family]:
            for wanted_signature in wanted_signatures:
                prefix_hash = wanted_signature
                for value in reversed(tail):
                    prefix_hash = unmix(prefix_hash, value)
                required[family][prefix_hash].append((wanted_signature, tail))

    found: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for family, family_targets in targets.items():
        if family not in wanted_by_family:
            continue
        for target_delta in family_targets:
            target_hash = mix(FNV_BASIS, family)
            for value in target_delta:
                target_hash = mix(target_hash, value)
            for hand_delta in range(13):  # frozen genome max_hands == 12
                hand_hash = mix(target_hash, hand_delta)
                for quadrant_delta in range(4):
                    prefix_hash = mix(hand_hash, quadrant_delta)
                    for wanted_signature, tail in required[family].get(prefix_hash, ()):
                        # SELL_TO_FINANCE variants are generated only for a
                        # positive project edit (market family is standalone).
                        if (tail[2] == 4 and family != 6 and
                                not any(value > 0 for value in target_delta)):
                            continue
                        row = candidate(
                            family, target_delta, hand_delta, quadrant_delta, tail
                        )
                        if signature(row) == wanted_signature:
                            found[(family, wanted_signature)].append(row)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, default=ROUND1)
    parser.add_argument("--semantic", type=Path, default=HERE / "semantic_fists.json")
    parser.add_argument("--materialization-audit", type=Path,
                        default=HERE / "materialization_audit.json")
    parser.add_argument("--output", type=Path,
                        default=HERE / "signature_inversion_audit.json")
    args = parser.parse_args()

    search = json.loads(args.search.read_text(encoding="utf-8"))
    semantic = json.loads(args.semantic.read_text(encoding="utf-8"))
    materialization = json.loads(
        args.materialization_audit.read_text(encoding="utf-8")
    )
    if search.get("schema") != "kaggriculture.counter-cluster-search-results.v1":
        raise RuntimeError("unexpected Round1 search schema")
    if semantic.get("schema") != "kaggriculture.counter-cluster-semantic-fists.v1":
        raise RuntimeError("unexpected semantic-fist schema")
    if materialization.get("schema") != "kaggriculture.counter-cluster-plan-materialization-audit.v1":
        raise RuntimeError("unexpected materialization-audit schema")

    search_rows = {row["task_id"]: row for row in search["rows"]}
    audit_rows = {row["task_id"]: row for row in materialization["rows"]}
    fists = {row["fist_id"]: row for row in semantic["fists"]}
    wanted = {
        (int(family), int(value))
        for row in search_rows.values()
        for family, value in zip(
            row["selected_family_ids"], row["selected_signatures"], strict=True
        )
    }
    inversions = invert(wanted)

    rows: list[dict[str, Any]] = []
    canonical_mismatches = 0
    representative_full_mismatches = 0
    for task_id in sorted(search_rows):
        source = search_rows[task_id]
        mapping = audit_rows.get(task_id)
        mapped_fist = fists.get(mapping["semantic_fist_id"]) if mapping else None
        stages = []
        for index, (family, value) in enumerate(zip(
            source["selected_family_ids"],
            source["selected_signatures"],
            strict=True,
        )):
            candidates = inversions.get((int(family), int(value)), [])
            recovered = candidates[0] if len(candidates) == 1 else None
            observed = (canonical(mapped_fist["intents"][index])
                        if mapped_fist is not None else None)
            recovered_intent = canonical(recovered) if recovered is not None else None
            canonical_match = recovered_intent == observed
            canonical_mismatches += not canonical_match

            representative = bool(
                mapped_fist is not None and
                mapped_fist["representative_task_id"] == task_id
            )
            representative_full_match = None
            if representative and recovered is not None:
                execution = mapped_fist["execution_deltas"][index]
                representative_full_match = all(
                    int(execution[field]) == int(recovered[field])
                    for field in ("family_id", "hand_delta", "quadrant_delta",
                                  "effective_delay_days", "schedule_profile",
                                  "market_profile", "recovery_profile",
                                  "suffix_project", "market_item", "recovery_issue")
                ) and [int(item) for item in execution["target_delta"]] == recovered["target_delta"] \
                    and int(execution["signature"]) == int(value)
                representative_full_mismatches += not representative_full_match
            stages.append({
                "stage": index,
                "family_id": int(family),
                "signature": int(value),
                "inverse_count": len(candidates),
                "recovered_intent": recovered_intent,
                "semantic_intent": observed,
                "canonical_match": canonical_match,
                "representative_full_execution_match": representative_full_match,
            })
        rows.append({
            "task_id": task_id,
            "semantic_fist_id": mapping["semantic_fist_id"] if mapping else None,
            "mapped": mapping is not None and mapped_fist is not None,
            "passed": bool(mapping is not None and mapped_fist is not None and
                           all(stage["canonical_match"] for stage in stages) and
                           all(stage["representative_full_execution_match"] is not False
                               for stage in stages)),
            "stages": stages,
        })

    missing = sorted(
        ({"family_id": family, "signature": value}
         for family, value in wanted if not inversions.get((family, value))),
        key=lambda row: (row["family_id"], row["signature"]),
    )
    ambiguous = sorted(
        ({"family_id": family, "signature": value,
          "inverse_count": len(inversions[(family, value)])}
         for family, value in wanted if len(inversions.get((family, value), [])) > 1),
        key=lambda row: (row["family_id"], row["signature"]),
    )
    passed = bool(
        not missing and not ambiguous and not canonical_mismatches and
        not representative_full_mismatches and
        len(search_rows) == len(audit_rows) == 48 and all(row["passed"] for row in rows)
    )
    catalog = []
    for family, value in sorted(wanted):
        candidates = inversions.get((family, value), [])
        catalog.append({
            "family_id": family,
            "signature": value,
            "inverse_count": len(candidates),
            "recovered_execution_fields": candidates[0] if len(candidates) == 1 else None,
        })
    payload = {
        "schema": "kaggriculture.counter-cluster-signature-inversion-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "scope": {
            "simulator_runs": 0,
            "search_runs": 0,
            "evaluation_or_outcome_files_read": [],
            "note": "Only Round1 discovery metadata and Round2 materialization metadata were read; estimated_* fields are deliberately outside the signature and canonical identity.",
        },
        "inputs": {
            "round1_search_results": {"path": str(args.search.resolve()),
                                      "sha256": sha256_file(args.search)},
            "semantic_fists": {"path": str(args.semantic.resolve()),
                               "sha256": sha256_file(args.semantic)},
            "materialization_audit": {
                "path": str(args.materialization_audit.resolve()),
                "sha256": sha256_file(args.materialization_audit),
            },
            "old_native_extension_sha256": search["config"]["native_extension_sha256"],
        },
        "signature_contract": {
            "algorithm": "64-bit FNV-1a over signed fields encoded as uint64 little-endian bytes",
            "included_fields": [
                "family_id", "target_delta[8]", "hand_delta", "quadrant_delta",
                "effective_delay_days", "schedule_profile", "market_profile",
                "recovery_profile", "suffix_project", "market_item",
                "recovery_issue",
            ],
            "excluded_fields": [
                "estimated_value", "estimated_cash_cost",
                "estimated_daily_action_load",
            ],
        },
        "summary": {
            "source_tasks": len(search_rows),
            "source_stages": sum(len(row["selected_signatures"])
                                 for row in search_rows.values()),
            "unique_family_signatures": len(wanted),
            "uniquely_inverted": sum(
                len(inversions.get(key, [])) == 1 for key in wanted
            ),
            "missing": len(missing),
            "ambiguous": len(ambiguous),
            "canonical_intent_mismatches": canonical_mismatches,
            "representative_full_execution_mismatches": representative_full_mismatches,
        },
        "missing": missing,
        "ambiguous": ambiguous,
        "signature_catalog": catalog,
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": passed, **payload["summary"],
                      "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
