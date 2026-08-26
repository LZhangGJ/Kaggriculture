#!/usr/bin/env python3
"""Append validated evolved tapes to a runnable route library."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

from meta_agent.src.teammate_expanded_routes import load_action_tapes
from search_native_tape_mutations import _apply_genes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-actions", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--family-prefix", default="EV")
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    tapes = load_action_tapes(args.actions)
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    base_entries = list(metadata["opponent_routes"])
    by_family = {str(value["family"]): value for value in base_entries}
    used_families = set(by_family)
    used_route_ids = set(tapes)
    generated_entries = []
    provenance = []
    next_number = 1

    for row in validation["portfolio"]:
        parent = str(row["parent"])
        if parent not in by_family:
            raise KeyError(f"unknown parent family: {parent}")
        parent_entry = by_family[parent]
        parent_tape = tapes[str(parent_entry["route_id"])]
        genes = list(row.get("genes") or [])
        evolved_tape = _apply_genes(parent_tape, genes)
        serialized = json.dumps(
            evolved_tape, sort_keys=True, separators=(",", ":")
        ).encode()
        digest = hashlib.sha256(serialized).hexdigest()
        while f"{args.family_prefix}{next_number:03d}" in used_families:
            next_number += 1
        family = f"{args.family_prefix}{next_number:03d}"
        next_number += 1
        route_id = f"evolved:{digest[:24]}"
        if route_id in used_route_ids:
            raise ValueError(f"duplicate evolved tape: {digest}")
        used_families.add(family)
        used_route_ids.add(route_id)
        tapes[route_id] = evolved_tape
        entry: dict[str, Any] = {
            "family": family,
            "alias": f"遗传路线{family}-{parent}",
            "route_id": route_id,
            "team": "evolved_route_library",
            "support": 0,
            "selected": True,
            "drop_reason": None,
            "source_execution_hard_failures": int(
                parent_entry.get("source_execution_hard_failures", 0) or 0
            ),
            "evolution": {
                "parent_family": parent,
                "parent_route_id": str(parent_entry["route_id"]),
                "source_validation_family": str(row["validation_family"]),
                "source_rank": int(row["rank"]),
                "portfolio_rank": int(row["portfolio_rank"]),
                "tape_sha256": digest,
                "genes": genes,
                "holdout_candidate_score": float(row["holdout_candidate_score"]),
                "holdout_incremental_oracle_score": float(
                    row["holdout_incremental_oracle_score"]
                ),
            },
        }
        generated_entries.append(entry)
        provenance.append({
            "family": family,
            "route_id": route_id,
            **entry["evolution"],
        })

    actions_bytes = zlib.compress(
        json.dumps(tapes, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        level=9,
    )
    actions_sha256 = hashlib.sha256(actions_bytes).hexdigest()
    output_metadata = copy.deepcopy(metadata)
    output_metadata["opponent_routes"] = [*base_entries, *generated_entries]
    output_metadata["selected"] = [*list(metadata.get("selected", base_entries)), *generated_entries]
    output_metadata["actions_file"] = args.output_actions.name
    output_metadata["actions_sha256"] = actions_sha256
    output_metadata["evolution_extension"] = {
        "schema": "validated-evolved-route-extension-v1",
        "source_validation": str(args.validation.resolve()),
        "base_route_count": len(base_entries),
        "generated_route_count": len(generated_entries),
        "total_route_count": len(base_entries) + len(generated_entries),
        "families": [value["family"] for value in generated_entries],
    }

    for path in (args.output_actions, args.output_metadata, args.output_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_actions.write_bytes(actions_bytes)
    args.output_metadata.write_text(
        json.dumps(output_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "materialized-evolved-route-library-v1",
        "source_actions": str(args.actions.resolve()),
        "source_metadata": str(args.metadata.resolve()),
        "source_validation": str(args.validation.resolve()),
        "output_actions": str(args.output_actions.resolve()),
        "output_metadata": str(args.output_metadata.resolve()),
        "actions_sha256": actions_sha256,
        "base_route_count": len(base_entries),
        "generated_route_count": len(generated_entries),
        "total_route_count": len(base_entries) + len(generated_entries),
        "generated_routes": provenance,
    }
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
