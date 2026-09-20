#!/usr/bin/env python3
"""Cluster every replay side into global macro-route families."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_macro_route_library import _clusters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--cpp-executable", type=Path)
    args = parser.parse_args()

    with np.load(args.macro_cache, allow_pickle=True) as cached:
        rows = list(cached["rows"])
    if args.cpp_executable is None:
        labels, distances = _clusters(rows, args.threshold)
    else:
        from sklearn.cluster import AgglomerativeClustering

        work = args.output.parent
        distance_input = work / "macro-distance-input.bin"
        distance_output = work / "macro-distance-v1.f32"
        production = np.ascontiguousarray(
            np.stack([row["production"] for row in rows]), dtype=np.int16
        )
        layouts = np.ascontiguousarray(
            np.stack([row["layouts"] for row in rows]), dtype=np.int8
        )
        schedule = np.ascontiguousarray(
            np.stack([row["schedule"] for row in rows]), dtype=np.int16
        )
        with distance_input.open("wb") as handle:
            np.asarray([len(rows)], dtype=np.int32).tofile(handle)
            production.tofile(handle)
            layouts.tofile(handle)
            schedule.tofile(handle)
        subprocess.run(
            [str(args.cpp_executable.resolve()), str(distance_input), str(distance_output)],
            check=True,
        )
        distances = np.memmap(
            distance_output, mode="r", dtype=np.float32, shape=(len(rows), len(rows))
        )
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=args.threshold,
        ).fit_predict(distances).astype(np.int32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, labels=labels)

    sizes = Counter(int(value) for value in labels.tolist())
    pairwise = distances[np.triu_indices(len(rows), 1)]
    payload = {
        "schema_version": 1,
        "macro_cache": str(args.macro_cache),
        "threshold": args.threshold,
        "target_sides": len(rows),
        "families": len(sizes),
        "family_sizes_desc": sorted(sizes.values(), reverse=True),
        "pair_distance": {
            "median": float(np.median(pairwise)) if len(pairwise) else 0.0,
            "p10": float(np.quantile(pairwise, 0.1)) if len(pairwise) else 0.0,
            "p90": float(np.quantile(pairwise, 0.9)) if len(pairwise) else 0.0,
        },
    }
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
