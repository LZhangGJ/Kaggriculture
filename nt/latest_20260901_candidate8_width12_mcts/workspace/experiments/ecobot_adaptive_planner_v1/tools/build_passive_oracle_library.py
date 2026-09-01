#!/usr/bin/env python3
"""Append a deterministic all-PASS opponent to a frozen native route library."""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from copy import deepcopy
from pathlib import Path


PASSIVE_FAMILY = "PASSIVE_NOOP"
PASSIVE_ROUTE_ID = "PASSIVE_NOOP:0"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--output-actions", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    actions = json.loads(zlib.decompress(args.actions.read_bytes()))
    if PASSIVE_ROUTE_ID in actions:
        raise ValueError(f"route already exists: {PASSIVE_ROUTE_ID}")

    passive_actions = [
        {"farmer": ["PASS"], "hands": [], "market": []}
        for _ in range(719)
    ]
    actions[PASSIVE_ROUTE_ID] = passive_actions

    route = {
        "family": PASSIVE_FAMILY,
        "alias": "deterministic all-PASS opponent",
        "route_id": PASSIVE_ROUTE_ID,
        "team": "local-fixture",
        "support": 1,
        "selected": True,
        "drop_reason": None,
        "source_execution_hard_failures": 0,
        "oracle_only": True,
    }
    output_metadata = deepcopy(metadata)
    output_metadata["taxonomy"] = (
        str(output_metadata.get("taxonomy", ""))
        + "; deterministic passive fixture"
    ).strip("; ")
    output_metadata["opponent_routes"] = list(output_metadata["opponent_routes"]) + [route]
    output_metadata["selected"] = list(output_metadata.get("selected", [])) + [route]

    raw_actions = json.dumps(
        actions, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    compressed_actions = zlib.compress(raw_actions, level=9)
    output_metadata["actions_file"] = str(args.output_actions)
    output_metadata["actions_sha256"] = sha256_bytes(compressed_actions)

    args.output_actions.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_actions.write_bytes(compressed_actions)
    args.output_metadata.write_text(
        json.dumps(output_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "family": PASSIVE_FAMILY,
        "route_id": PASSIVE_ROUTE_ID,
        "steps": len(passive_actions),
        "routes": len(actions),
        "actions_sha256": output_metadata["actions_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
