#!/usr/bin/env python3
"""Prune a candidate route library to a base library plus explicit additions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "src"))

from meta_agent.src.teammate_expanded_routes import load_action_tapes


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--keep", type=_csv, default=())
    parser.add_argument(
        "--only-explicit", action="store_true",
        help="Deploy only --keep families instead of retaining the complete base library.",
    )
    parser.add_argument("--output-actions", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    base = json.loads(args.base_metadata.read_text(encoding="utf-8"))
    base_families = {str(value["family"]) for value in base["opponent_routes"]}
    requested = set(args.keep)
    available = {str(value["family"]) for value in metadata["opponent_routes"]}
    missing = sorted(requested - available)
    if missing:
        raise KeyError(f"unknown requested family: {missing[0]}")
    keep = requested if args.only_explicit else base_families | requested
    entries = [
        value for value in metadata["opponent_routes"]
        if str(value["family"]) in keep
    ]
    route_ids = {str(value["route_id"]) for value in entries}
    source_tapes = load_action_tapes(args.actions)
    tapes = {key: value for key, value in source_tapes.items() if key in route_ids}
    if len(tapes) != len(route_ids):
        raise ValueError("metadata and action tape keys are not aligned")
    packed = zlib.compress(
        json.dumps(tapes, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        level=9,
    )
    output = dict(metadata)
    output["opponent_routes"] = entries
    output["selected"] = [
        value for value in metadata.get("selected", entries)
        if str(value["family"]) in keep
    ]
    output["actions_file"] = args.output_actions.name
    output["actions_sha256"] = hashlib.sha256(packed).hexdigest()
    output["pruning"] = {
        "schema": "base-plus-explicit-route-pruning-v1",
        "base_metadata": str(args.base_metadata.resolve()),
        "source_route_count": len(metadata["opponent_routes"]),
        "base_route_count": len(base_families),
        "explicit_additions": sorted(requested),
        "only_explicit": bool(args.only_explicit),
        "output_route_count": len(entries),
    }
    for path in (args.output_actions, args.output_metadata, args.output_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_actions.write_bytes(packed)
    args.output_metadata.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        **output["pruning"],
        "output_actions": str(args.output_actions.resolve()),
        "output_metadata": str(args.output_metadata.resolve()),
        "actions_sha256": hashlib.sha256(packed).hexdigest(),
    }
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
