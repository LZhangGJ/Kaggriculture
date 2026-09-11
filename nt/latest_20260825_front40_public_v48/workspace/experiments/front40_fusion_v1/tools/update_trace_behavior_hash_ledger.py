#!/usr/bin/env python3
"""Incrementally hash complete trace-bank actions for behavior deduplication."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ACTION_KEYS = (
    "unit_op",
    "unit_item",
    "unit_amount",
    "unit_count",
    "market_op",
    "market_item",
    "market_amount",
    "market_count",
)


def bank_hashes(path: Path) -> list[str]:
    with np.load(path) as bank:
        missing = [key for key in ACTION_KEYS if key not in bank]
        if missing:
            raise RuntimeError(f"{path}: missing action arrays {missing}")
        route_count = int(bank[ACTION_KEYS[0]].shape[0])
        return [
            hashlib.sha256(
                b"".join(
                    np.ascontiguousarray(bank[key][route_id]).tobytes()
                    for key in ACTION_KEYS
                )
            ).hexdigest()
            for route_id in range(route_count)
        ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pattern", default="rank*_trace_bank_v1.npz")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--focus", type=Path)
    args = parser.parse_args()

    old = {}
    if args.ledger.exists():
        old = json.loads(args.ledger.read_text(encoding="utf-8")).get("banks", {})

    banks: dict[str, dict] = {}
    cache_hits = 0
    recomputed = 0
    for path in sorted(args.root.glob(args.pattern)):
        resolved = path.resolve()
        key = resolved.relative_to(Path.cwd().resolve()).as_posix()
        stat = path.stat()
        signature = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        cached = old.get(key)
        if cached and cached.get("signature") == signature:
            hashes = cached["route_hashes"]
            cache_hits += 1
        else:
            hashes = bank_hashes(path)
            recomputed += 1
        banks[key] = {
            "signature": signature,
            "route_count": len(hashes),
            "unique_route_count": len(set(hashes)),
            "route_hashes": hashes,
        }

    payload = {
        "schema": "kaggriculture.front40_fusion.trace-behavior-hash-ledger.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "action_keys": list(ACTION_KEYS),
        "banks": banks,
    }
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    result: dict = {
        "status": "PASS",
        "banks": len(banks),
        "cache_hits": cache_hits,
        "recomputed": recomputed,
        "ledger": args.ledger.as_posix(),
    }
    if args.focus:
        focus_key = args.focus.resolve().relative_to(Path.cwd().resolve()).as_posix()
        focus_hashes = set(banks[focus_key]["route_hashes"])
        overlaps = {}
        for key, row in banks.items():
            if key == focus_key:
                continue
            count = len(focus_hashes.intersection(row["route_hashes"]))
            if count:
                overlaps[key] = count
        result["focus"] = focus_key
        result["focus_unique_routes"] = len(focus_hashes)
        result["overlaps"] = overlaps
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
