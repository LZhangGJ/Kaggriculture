#!/usr/bin/env python3
"""Materialize causal prefix/suffix route splices for staged selectors."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence


def _splice(value: str) -> tuple[str, str, int, str]:
    family, equal, expression = value.partition("=")
    base, at, remainder = expression.partition("@")
    checkpoint, plus, target = remainder.partition("+")
    if not equal or not at or not plus or not family or not base or not target:
        raise argparse.ArgumentTypeError(
            "splice must be FAMILY=PREFIX_FAMILY@CHECKPOINT+SUFFIX_FAMILY"
        )
    try:
        step = int(checkpoint)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("splice checkpoint must be an integer") from exc
    if not 0 <= step <= 719:
        raise argparse.ArgumentTypeError("splice checkpoint must be in [0, 719]")
    return family, base, step, target


def _load_tapes(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        str(key): list(value)
        for key, value in json.loads(zlib.decompress(path.read_bytes())).items()
    }


def splice_tapes(
    prefix: Sequence[Mapping[str, Any]],
    suffix: Sequence[Mapping[str, Any]],
    checkpoint: int,
) -> list[dict[str, Any]]:
    if len(prefix) < 719 or len(suffix) < 719:
        raise ValueError("both route tapes must contain 719 actions")
    return copy.deepcopy([*prefix[:checkpoint], *suffix[checkpoint:719]])


def add_archive_routes(
    tapes: dict[str, list[dict[str, Any]]],
    entries: list[dict[str, Any]],
    route_by_family: dict[str, str],
    archive: Mapping[str, Sequence[Mapping[str, Any]]],
    source: str,
) -> list[dict[str, Any]]:
    imported = []
    for family, raw_tape in archive.items():
        family = str(family)
        if family in route_by_family:
            raise ValueError(f"duplicate archive family: {family}")
        tape = copy.deepcopy(list(raw_tape))
        if len(tape) != 719:
            raise ValueError(f"archive route {family} must contain 719 actions")
        raw = json.dumps(
            tape, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        route_id = f"archive:{digest[:24]}"
        tapes[route_id] = tape
        entry = {
            "family": family,
            "alias": f"archive-{family}",
            "route_id": route_id,
            "team": "route_archive",
            "support": 0,
            "selected": True,
            "drop_reason": None,
            "source_execution_hard_failures": 0,
            "route_archive": {
                "schema": "route-archive-import-v1",
                "source": source,
                "tape_sha256": digest,
            },
        }
        entries.append(entry)
        imported.append(entry)
        route_by_family[family] = route_id
    return imported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--route-archive", type=Path, action="append", default=[])
    parser.add_argument("--splice", type=_splice, action="append", required=True)
    parser.add_argument("--output-actions", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    tapes = _load_tapes(args.actions)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    entries = list(metadata["opponent_routes"])
    route_by_family = {
        str(value["family"]): str(value["route_id"]) for value in entries
    }
    imported = []
    for path in args.route_archive:
        imported.extend(add_archive_routes(
            tapes, entries, route_by_family, _load_tapes(path), str(path.resolve())
        ))
    used_families = set(route_by_family)
    generated = []
    for family, prefix_family, checkpoint, suffix_family in args.splice:
        if family in used_families:
            raise ValueError(f"duplicate output family: {family}")
        if prefix_family not in route_by_family:
            raise KeyError(f"unknown prefix family: {prefix_family}")
        if suffix_family not in route_by_family:
            raise KeyError(f"unknown suffix family: {suffix_family}")
        prefix_id = route_by_family[prefix_family]
        suffix_id = route_by_family[suffix_family]
        tape = splice_tapes(tapes[prefix_id], tapes[suffix_id], checkpoint)
        raw = json.dumps(
            tape, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        route_id = f"causal:{digest[:24]}"
        # Duplicate tapes are intentionally allowed under different families:
        # a route-tree node is an auditable semantic choice even when two
        # source tapes happened to share the same prefix or suffix.
        tapes[route_id] = tape
        entry = {
            "family": family,
            "alias": f"causal-{prefix_family}-at-{checkpoint}-to-{suffix_family}",
            "route_id": route_id,
            "team": "causal_route_splice",
            "support": 0,
            "selected": True,
            "drop_reason": None,
            "source_execution_hard_failures": 0,
            "causal_splice": {
                "schema": "causal-prefix-suffix-splice-v1",
                "prefix_family": prefix_family,
                "prefix_route_id": prefix_id,
                "checkpoint": checkpoint,
                "suffix_family": suffix_family,
                "suffix_route_id": suffix_id,
                "tape_sha256": digest,
                "selection_boundary": (
                    "the suffix may only be selected from observations available "
                    f"at step {checkpoint}"
                ),
            },
        }
        entries.append(entry)
        generated.append(entry)
        route_by_family[family] = route_id
        used_families.add(family)

    packed = zlib.compress(
        json.dumps(tapes, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        level=9,
    )
    actions_sha256 = hashlib.sha256(packed).hexdigest()
    output_metadata = copy.deepcopy(metadata)
    output_metadata["opponent_routes"] = entries
    output_metadata["selected"] = [
        *list(metadata.get("selected", metadata["opponent_routes"])),
        *imported,
        *generated,
    ]
    output_metadata["actions_file"] = args.output_actions.name
    output_metadata["actions_sha256"] = actions_sha256
    output_metadata["causal_splice_extension"] = {
        "schema": "causal-route-splice-extension-v1",
        "source_metadata": str(args.metadata.resolve()),
        "imported_route_count": len(imported),
        "generated_route_count": len(generated),
        "families": [value["family"] for value in generated],
    }
    manifest = {
        "schema": "materialized-causal-route-splices-v1",
        "source_actions": str(args.actions.resolve()),
        "source_metadata": str(args.metadata.resolve()),
        "output_actions": str(args.output_actions.resolve()),
        "output_metadata": str(args.output_metadata.resolve()),
        "actions_sha256": actions_sha256,
        "base_route_count": len(metadata["opponent_routes"]),
        "imported_route_count": len(imported),
        "generated_route_count": len(generated),
        "total_route_count": len(entries),
        "routes": generated,
    }
    for path in (args.output_actions, args.output_metadata, args.output_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_actions.write_bytes(packed)
    args.output_metadata.write_text(
        json.dumps(output_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_actions": str(args.output_actions.resolve()),
        "output_metadata": str(args.output_metadata.resolve()),
        "generated_routes": [value["family"] for value in generated],
        "total_routes": len(entries),
        "actions_bytes": len(packed),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
