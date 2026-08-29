#!/usr/bin/env python3
"""Cluster unified routes by intended macro actions, not realized failures."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cpp-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.12)
    parser.add_argument(
        "--sensitivity", default=".04,.06,.08,.10,.12,.14,.16,.18,.20"
    )
    args = parser.parse_args()

    from sklearn.cluster import AgglomerativeClustering

    with np.load(args.features, allow_pickle=True) as cached:
        rows = list(cached["rows"])
    with np.load(args.audit) as cached:
        keys = cached["keys"]
        planned_layouts = np.ascontiguousarray(cached["planned_layouts"], dtype=np.int8)
    feature_keys = np.asarray(
        [(int(row["episode_id"]), int(row["player_index"])) for row in rows],
        dtype=np.int64,
    )
    if not np.array_equal(keys, feature_keys):
        raise ValueError("execution audit is not aligned with macro features")
    schedules = np.ascontiguousarray(
        np.stack([row["schedule"] for row in rows]), dtype=np.int16
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    binary_input = args.output.parent / "intent-distance-input-v1.bin"
    binary_output = args.output.parent / "intent-distance-v1.f32"
    with binary_input.open("wb") as handle:
        np.asarray([len(rows)], dtype=np.int32).tofile(handle)
        planned_layouts.tofile(handle)
        schedules.tofile(handle)
    subprocess.run(
        [str(args.cpp_executable.resolve()), str(binary_input), str(binary_output)],
        check=True,
    )
    distances = np.memmap(
        binary_output, mode="r", dtype=np.float32, shape=(len(rows), len(rows))
    )

    def cluster(threshold: float) -> np.ndarray:
        return AgglomerativeClustering(
            n_clusters=None, metric="precomputed", linkage="average",
            distance_threshold=threshold,
        ).fit_predict(distances).astype(np.int32)

    labels = cluster(args.threshold)
    np.savez_compressed(args.output, labels=labels, keys=keys)
    sensitivity = {}
    for threshold in (float(value) for value in args.sensitivity.split(",")):
        trial = cluster(threshold)
        sizes = Counter(trial.tolist())
        top = sum(sorted(sizes.values(), reverse=True)[:56])
        sensitivity[str(threshold)] = {
            "families": len(sizes),
            "singletons": sum(size == 1 for size in sizes.values()),
            "top56_coverage": top / len(rows),
        }
    sizes = Counter(labels.tolist())
    pairs = distances[np.triu_indices(len(rows), 1)]
    summary = {
        "schema_version": 1,
        "identity": "intended macro plan; realized failures excluded",
        "distance": "0.45 intended composition + 0.35 intended layout + 0.20 cumulative macro schedule",
        "threshold": args.threshold,
        "rows": len(rows),
        "families": len(sizes),
        "singletons": sum(size == 1 for size in sizes.values()),
        "family_sizes_desc": sorted(sizes.values(), reverse=True),
        "pair_distance": {
            "p10": float(np.quantile(pairs, .1)),
            "median": float(np.median(pairs)),
            "p90": float(np.quantile(pairs, .9)),
        },
        "sensitivity": sensitivity,
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
