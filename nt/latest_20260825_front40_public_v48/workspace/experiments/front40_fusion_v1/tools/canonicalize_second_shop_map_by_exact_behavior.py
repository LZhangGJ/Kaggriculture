"""Replace duplicate route IDs in a second-shop map with exact behavior representatives.

This is an evaluation/runtime compaction only.  A route is replaced when all
raw actions plus the expected unit positions/activity used by recovery are
byte-identical.  Source Replay reward, Episode ID and shop history are retained
in the original route bank/receipts but are not runtime behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BEHAVIOR_FIELDS = (
    "unit_op",
    "unit_item",
    "unit_amount",
    "unit_count",
    "market_op",
    "market_item",
    "market_amount",
    "market_count",
    "expected_unit_pos",
    "expected_unit_active",
)


def _hash(bank: np.lib.npyio.NpzFile, route: int) -> str:
    digest = hashlib.sha256()
    for field in BEHAVIOR_FIELDS:
        value = np.ascontiguousarray(bank[field][route])
        digest.update(field.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    with np.load(args.trace_bank.resolve(), allow_pickle=False) as bank:
        route_count = int(bank["source_reward"].shape[0])
        hashes = [_hash(bank, route) for route in range(route_count)]
        grouped: dict[str, list[int]] = {}
        for route, behavior_hash in enumerate(hashes):
            grouped.setdefault(behavior_hash, []).append(route)
        canonical = np.arange(route_count, dtype=np.int32)
        groups = []
        for behavior_hash, routes in grouped.items():
            # Highest source reward gives the compact route a stable and
            # auditable provenance tie-break without changing behavior.
            representative = max(routes, key=lambda route: int(bank["source_reward"][route]))
            for route in routes:
                for field in BEHAVIOR_FIELDS:
                    if not np.array_equal(bank[field][representative], bank[field][route]):
                        raise ValueError(
                            f"behavior mismatch in hash group {representative}/{route}: {field}"
                        )
                canonical[route] = representative
            groups.append(
                {
                    "representative": representative,
                    "members": routes,
                    "behavior_sha256": behavior_hash,
                }
            )

    changed_ids: dict[int, int] = {}

    def replace(value: int) -> int:
        route = int(value)
        if not 0 <= route < route_count:
            raise ValueError(f"route {route} outside trace bank")
        result = int(canonical[route])
        if result != route:
            changed_ids[route] = result
        return result

    def replace_legacy(value: int) -> int:
        """Preserve padded identity rows that exceed the real bank size."""
        route = int(value)
        return replace(route) if 0 <= route < route_count else route

    payload["first_route_ids"] = [replace(value) for value in payload["first_route_ids"]]
    if "second_route_ids_by_first_shop" not in payload:
        raise ValueError("source map lacks first-shop-indexed second route map")
    payload["second_route_ids_by_first_shop"] = [
        [replace(value) for value in row]
        for row in payload["second_route_ids_by_first_shop"]
    ]
    # Keep the legacy matrix internally consistent for old readers as well.
    if "second_route_ids" in payload:
        payload["second_route_ids"] = [
            [replace_legacy(value) for value in row]
            for row in payload["second_route_ids"]
        ]
    for row in payload.get("routes", []):
        if "base_route" in row:
            row["base_route_before_canonicalization"] = int(row["base_route"])
            row["base_route"] = replace(row["base_route"])
        if "selected_route" in row:
            row["selected_route_before_canonicalization"] = int(row["selected_route"])
            row["selected_route"] = replace(row["selected_route"])

    payload["schema"] = "kaggriculture.front40_fusion.canonical-second-shop-map.v1"
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["source_map"] = str(args.source)
    payload["canonicalization"] = {
        "proof": "byte equality over every forced-route behavior field",
        "behavior_fields": list(BEHAVIOR_FIELDS),
        "group_count": len(groups),
        "changed_route_ids": {str(key): value for key, value in sorted(changed_ids.items())},
        "groups": sorted(groups, key=lambda row: int(row["representative"])),
    }
    payload["status"] = "TRAINING_ONLY"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output),
                "changed_route_ids": payload["canonicalization"]["changed_route_ids"],
                "runtime_routes": sorted(
                    {
                        *payload["first_route_ids"],
                        *np.asarray(payload["second_route_ids_by_first_shop"]).reshape(-1).tolist(),
                    }
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
