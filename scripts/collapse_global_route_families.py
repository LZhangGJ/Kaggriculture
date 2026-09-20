#!/usr/bin/env python3
"""Collapse execution-fragment clusters into the major global route families."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-cache", type=Path, required=True)
    parser.add_argument("--global-clusters", type=Path, required=True)
    parser.add_argument("--distance-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--families", type=int, default=56)
    parser.add_argument(
        "--drop-minor",
        action="store_true",
        help="Leave non-major execution fragments unlabeled instead of force-mapping them.",
    )
    args = parser.parse_args()

    with np.load(args.macro_cache, allow_pickle=True) as cached:
        rows = list(cached["rows"])
    with np.load(args.global_clusters) as cached:
        labels = cached["labels"].astype(int)
    sizes = Counter(labels.tolist())
    medians = {
        label: float(np.median([
            float(row["reward"]) for row, value in zip(rows, labels) if int(value) == int(label)
        ]))
        for label in sizes
    }
    ordered = sorted(sizes, key=lambda label: (-sizes[label], -medians[label]))
    major = ordered[: args.families]
    raw_to_name = {int(label): f"G{index + 1}" for index, label in enumerate(major)}
    major_indices = np.flatnonzero(np.isin(labels, major))
    count = len(rows)
    distances = np.memmap(
        args.distance_matrix, mode="r", dtype=np.float32, shape=(count, count)
    )
    names = np.empty(count, dtype=f"U{len(str(args.families)) + 1}")
    collapse_distance = np.zeros(count, dtype=np.float32)
    collapsed = 0
    for index, label in enumerate(labels.tolist()):
        name = raw_to_name.get(int(label))
        if name is None:
            if args.drop_minor:
                names[index] = ""
                collapsed += 1
                continue
            nearest_offset = int(np.argmin(distances[index, major_indices]))
            nearest = int(major_indices[nearest_offset])
            name = raw_to_name[int(labels[nearest])]
            collapse_distance[index] = float(distances[index, nearest])
            collapsed += 1
        names[index] = name
    np.savez_compressed(
        args.output, family_names=names, collapse_distance=collapse_distance
    )
    measured = collapse_distance[collapse_distance > 0]
    payload = {
        "schema_version": 1,
        "families": args.families,
        "direct_major_sides": count - collapsed,
        "collapsed_minor_sides": collapsed,
        "collapsed_distance": {
            "median": float(np.median(measured)) if len(measured) else None,
            "p90": float(np.quantile(measured, 0.9)) if len(measured) else None,
        },
        "family_sizes": dict(Counter(value for value in names.tolist() if value)),
    }
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
