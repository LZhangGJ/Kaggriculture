#!/usr/bin/env python3
"""Build route branches that share an exact, irreversible opening prefix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from meta_agent.src.teammate_expanded_routes import load_action_tapes


TRANSFERABLE_MARKET_OPERATIONS = {
    "BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL", "SELL",
}


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def shared_prefix_route(
    base: Sequence[Mapping[str, Any]],
    donor: Sequence[Mapping[str, Any]],
    checkpoint: int,
    mode: str,
) -> list[dict[str, Any]]:
    """Return a branch whose actions before ``checkpoint`` exactly match base."""

    if len(base) != len(donor):
        raise ValueError("base and donor tapes must have equal length")
    if not 0 < checkpoint < len(base):
        raise ValueError("checkpoint must be inside the action tape")
    result = copy.deepcopy(list(base))
    if mode == "full":
        result[checkpoint:] = copy.deepcopy(list(donor[checkpoint:]))
    elif mode == "market":
        for step in range(checkpoint, len(result)):
            structural = [
                list(order)
                for order in result[step].get("market", ()) or ()
                if not order or str(order[0]) not in TRANSFERABLE_MARKET_OPERATIONS
            ]
            transferred = [
                list(order)
                for order in donor[step].get("market", ()) or ()
                if order and str(order[0]) in TRANSFERABLE_MARKET_OPERATIONS
            ]
            result[step]["market"] = (structural + transferred)[:10]
    else:
        raise ValueError(f"unsupported shared-prefix mode: {mode}")
    if result[:checkpoint] != list(base[:checkpoint]):
        raise AssertionError("shared-prefix construction changed the opening")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--extra-actions", type=Path, action="append", default=[])
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--bases", type=_csv, required=True)
    parser.add_argument("--donors", type=_csv, required=True)
    parser.add_argument("--checkpoints", type=_ints, required=True)
    parser.add_argument("--modes", type=_csv, default=("full", "market"))
    parser.add_argument("--family-prefix", default="SP")
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    if not set(args.modes) <= {"full", "market"}:
        parser.error("--modes supports only full and market")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    route_id = {
        str(row["family"]): str(row["route_id"])
        for row in metadata["opponent_routes"]
    }
    tapes = load_action_tapes(args.actions)
    for path in args.extra_actions:
        extra = load_action_tapes(path)
        duplicates = sorted(set(tapes) & set(extra))
        if duplicates:
            raise ValueError(f"duplicate extra route family: {duplicates[0]}")
        tapes.update(extra)
        route_id.update({family: family for family in extra})
    missing = sorted(set((*args.bases, *args.donors)) - set(route_id))
    if missing:
        raise KeyError(f"unknown route family: {missing[0]}")
    archive: dict[str, list[dict[str, Any]]] = {}
    rows = []
    ordinal = 1
    for base in args.bases:
        for checkpoint in args.checkpoints:
            for donor in args.donors:
                if donor == base:
                    continue
                for mode in args.modes:
                    family = f"{args.family_prefix}{ordinal:04d}"
                    route = shared_prefix_route(
                        tapes[route_id[base]], tapes[route_id[donor]], checkpoint, mode
                    )
                    archive[family] = route
                    rows.append({
                        "family": family,
                        "base": base,
                        "donor": donor,
                        "checkpoint": checkpoint,
                        "mode": mode,
                        "tape_sha256": hashlib.sha256(json.dumps(
                            route, ensure_ascii=False, separators=(",", ":")
                        ).encode("utf-8")).hexdigest(),
                    })
                    ordinal += 1

    packed = zlib.compress(json.dumps(
        archive, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8"), level=9)
    for path in (args.output_archive, args.output_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_archive.write_bytes(packed)
    manifest = {
        "schema": "shared-prefix-route-panel-v1",
        "actions": str(args.actions.resolve()),
        "metadata": str(args.metadata.resolve()),
        "route_count": len(rows),
        "prefix_invariant": "branch[:checkpoint] == base[:checkpoint]",
        "archive_sha256": hashlib.sha256(packed).hexdigest(),
        "routes": rows,
    }
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "routes": len(rows),
        "archive": str(args.output_archive.resolve()),
        "manifest": str(args.output_manifest.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
