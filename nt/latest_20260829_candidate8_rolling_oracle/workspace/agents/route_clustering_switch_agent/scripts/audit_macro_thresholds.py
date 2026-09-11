#!/usr/bin/env python3
"""Report global macro-family sensitivity for a precomputed C++ distance matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument(
        "--thresholds", default="0.03,0.05,0.08,0.10,0.12,0.14,0.16,0.20"
    )
    args = parser.parse_args()
    distances = np.memmap(
        args.distance, mode="r", dtype=np.float32, shape=(args.count, args.count)
    )
    for threshold in (float(value) for value in args.thresholds.split(",")):
        labels = AgglomerativeClustering(
            n_clusters=None, metric="precomputed", linkage="average",
            distance_threshold=threshold,
        ).fit_predict(distances)
        sizes = np.bincount(labels)
        descending = np.sort(sizes)[::-1]
        print(json.dumps({
            "threshold": threshold,
            "families": len(sizes),
            "singletons": int(np.count_nonzero(sizes == 1)),
            "top5_coverage": round(float(descending[:5].sum() / args.count), 4),
            "top56_coverage": round(float(descending[:56].sum() / args.count), 4),
        }), flush=True)


if __name__ == "__main__":
    main()
