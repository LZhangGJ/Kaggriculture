#!/usr/bin/env python3
"""Map executable replay-trie branches onto globally deduplicated macro families.

The runtime selector reasons in exact action-prefix hashes, while analysis and
opponent recognition use global ``G#`` macro families.  This script preserves
both identities and produces the join table needed by counterfactual search.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--macro-cache", type=Path, required=True)
    parser.add_argument("--global-clusters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _fallback_family(route_id: str) -> str:
    """Keep mined/synthetic routes visible without pretending they are a G family."""
    if route_id.startswith("counter:"):
        parts = route_id.split(":", 2)
        return f"COUNTER:{parts[1]}" if len(parts) >= 2 else "COUNTER"
    if route_id.startswith("equilibrium:"):
        parts = route_id.split(":", 2)
        return f"EQUILIBRIUM:{parts[1]}" if len(parts) >= 2 else "EQUILIBRIUM"
    return "UNMAPPED"


def _route_family_map(macro_cache: Path, global_clusters: Path) -> dict[str, str]:
    with np.load(macro_cache, allow_pickle=True) as cached:
        rows = list(cached["rows"])
    with np.load(global_clusters) as cached:
        labels = cached["labels"].astype(int)
    if len(rows) != len(labels):
        raise ValueError("global labels do not align with macro rows")
    sizes = Counter(labels.tolist())
    median_rewards = {
        label: float(np.median([
            float(row["reward"]) for row, value in zip(rows, labels)
            if int(value) == int(label)
        ]))
        for label in sizes
    }
    # Keep IDs byte-for-byte consistent with export_macro_route_dashboard.py.
    ordered = sorted(
        sizes, key=lambda label: (-sizes[label], -median_rewards[label])
    )
    name = {label: f"G{index + 1}" for index, label in enumerate(ordered)}
    return {
        f"{int(row['episode_id'])}:{int(row['player_index'])}": name[int(label)]
        for row, label in zip(rows, labels)
    }


def build(
    runtime: Path, macro_cache: Path, global_clusters: Path
) -> dict[str, Any]:
    with (runtime / "manifest.pkl").open("rb") as handle:
        manifest = pickle.load(handle)
    route_ids = {str(route_id) for route_id in manifest["route_offsets"]}
    known = _route_family_map(macro_cache, global_clusters)
    family_by_route = {
        route_id: known.get(route_id, _fallback_family(route_id))
        for route_id in sorted(route_ids)
    }
    checkpoints = [int(value) for value in manifest["checkpoints"]]
    nodes: dict[str, dict[str, Any]] = {}
    checkpoint_summary: dict[str, Any] = {}
    for checkpoint in checkpoints:
        with (runtime / f"checkpoint-{checkpoint:03d}.pkl").open("rb") as handle:
            rows = pickle.load(handle)
        by_prefix: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        for row in rows:
            by_prefix[str(row[2])].append(row)
        branch_nodes = 0
        branch_candidates = 0
        for prefix, prefix_rows in by_prefix.items():
            by_next: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
            for row in prefix_rows:
                by_next[str(row[3])].append(row)
            if len(by_next) <= 1:
                continue
            branch_nodes += 1
            branch_candidates += len(by_next)
            candidates = []
            for next_family, family_rows in sorted(by_next.items()):
                global_counts = Counter(
                    family_by_route[str(row[0])] for row in family_rows
                )
                representative = max(
                    family_rows, key=lambda row: (float(row[1]), str(row[0]))
                )
                candidates.append({
                    "next_family_hash": next_family,
                    "support": len(family_rows),
                    "representative_route_id": str(representative[0]),
                    "median_historical_reward": float(np.median([
                        float(row[1]) for row in family_rows
                    ])),
                    "global_families": dict(sorted(
                        global_counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )),
                    "route_ids": sorted(str(row[0]) for row in family_rows),
                })
            nodes[f"{checkpoint}:{prefix}"] = {
                "checkpoint": checkpoint,
                "prefix_hash": prefix,
                "routes": len(prefix_rows),
                "candidates": candidates,
            }
        checkpoint_summary[str(checkpoint)] = {
            "prefixes": len(by_prefix),
            "branch_nodes": branch_nodes,
            "branch_candidates": branch_candidates,
        }
    mapped = sum(value.startswith("G") for value in family_by_route.values())
    return {
        "schema_version": 1,
        "runtime": str(runtime),
        "checkpoints": checkpoints,
        "family_by_route": family_by_route,
        "nodes": nodes,
        "summary": {
            "routes": len(route_ids),
            "mapped_global_routes": mapped,
            "unmapped_or_synthetic_routes": len(route_ids) - mapped,
            "branch_nodes": len(nodes),
            "checkpoint": checkpoint_summary,
        },
    }


def main() -> None:
    args = arguments()
    payload = build(args.runtime, args.macro_cache, args.global_clusters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".json":
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        args.output.write_bytes(pickle.dumps(payload, protocol=5))
    print(json.dumps({
        "output": str(args.output), **payload["summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
