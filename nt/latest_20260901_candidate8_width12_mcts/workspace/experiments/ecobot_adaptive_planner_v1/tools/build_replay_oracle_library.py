#!/usr/bin/env python3
"""Append one official Replay action tape to a native route library.

This tool is for offline oracle/diagnostic experiments only.  It deliberately
keeps the raw 719-step tape outside NativeAdaptivePlanner and submission code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path

try:
    import orjson
except ImportError:  # pragma: no cover - stdlib fallback for portability
    orjson = None


def loads(data: bytes):
    return orjson.loads(data) if orjson is not None else json.loads(data)


def dumps(value) -> bytes:
    if orjson is not None:
        return orjson.dumps(value)
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-actions", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--player", type=int, choices=(0, 1), required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--output-actions", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    args = parser.parse_args()

    tapes = loads(zlib.decompress(args.base_actions.read_bytes()))
    replay = loads(args.replay.read_bytes())
    episode = int(replay.get("info", {}).get("EpisodeId", 0))
    route_id = f"{episode}:{args.player}"
    tape = []
    for step in range(1, min(720, len(replay["steps"]))):
        action = replay["steps"][step][args.player].get("action") or {}
        tape.append({
            "farmer": list(action.get("farmer") or ["PASS"]),
            "hands": [list(value or ["PASS"]) for value in action.get("hands") or []],
            "market": [list(value) for value in action.get("market") or []],
        })
    if len(tape) != 719:
        raise ValueError(f"expected 719 actions, got {len(tape)}")
    tapes[route_id] = tape

    metadata = json.loads(args.base_metadata.read_text(encoding="utf-8"))
    metadata["opponent_routes"] = [
        row for row in metadata["opponent_routes"]
        if row.get("family") != args.family
    ]
    entry = {
        "family": args.family,
        "alias": args.alias,
        "route_id": route_id,
        "team": replay.get("info", {}).get("TeamNames", ["UNKNOWN"])[args.player],
        "support": 1,
        "selected": True,
        "drop_reason": None,
        "source_execution_hard_failures": None,
        "oracle_only": True,
        "source_submission": 55714246,
    }
    metadata["opponent_routes"].append(entry)
    metadata["selected"] = [
        row for row in metadata["opponent_routes"] if row.get("selected")
    ]
    metadata["taxonomy"] = "base library plus raw Replay oracle; offline only"

    packed = zlib.compress(dumps(tapes), level=9)
    args.output_actions.parent.mkdir(parents=True, exist_ok=True)
    args.output_actions.write_bytes(packed)
    metadata["actions_file"] = str(args.output_actions)
    metadata["actions_sha256"] = hashlib.sha256(packed).hexdigest().upper()
    args.output_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "family": args.family,
        "route_id": route_id,
        "actions": len(tape),
        "library_routes": len(metadata["opponent_routes"]),
        "actions_sha256": metadata["actions_sha256"],
        "oracle_only": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
