#!/usr/bin/env python3
"""Rebuild selected dynamic-policy GA genomes as action tapes."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

from evolve_native_route_library import _apply_evolution_genes
from meta_agent.src.teammate_expanded_routes import load_action_tapes


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--families", type=_csv, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = json.loads(args.archive.read_text(encoding="utf-8"))
    rows = {str(row["family"]): row for row in archive["ranking"]}
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    route_ids = {str(row["family"]): str(row["route_id"]) for row in metadata["opponent_routes"]}
    source = load_action_tapes(args.actions)
    source_families = (archive["opening"], *archive["donors"])
    parent_tapes = {family: source[route_ids[family]] for family in source_families}
    tapes = {}
    for family in args.families:
        row = rows[family]
        tapes[family] = _apply_evolution_genes(
            parent_tapes[str(row["parent"])], row["genes"], parent_tapes
        )
        checkpoint = int(archive["checkpoint"])
        if tapes[family][:checkpoint] != parent_tapes[archive["opening"]][:checkpoint]:
            raise AssertionError(f"prefix invariant failed for {family}")
    packed = zlib.compress(
        json.dumps(tapes, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        level=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(packed)
    print(json.dumps({
        "output": str(args.output.resolve()), "families": list(tapes),
        "bytes": len(packed),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
