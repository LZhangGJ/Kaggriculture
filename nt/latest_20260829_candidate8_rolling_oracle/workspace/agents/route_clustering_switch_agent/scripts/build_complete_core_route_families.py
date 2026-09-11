#!/usr/bin/env python3
"""Build route families around complete executions, then attach failed variants."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--distance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.12)
    args = parser.parse_args()

    from sklearn.cluster import AgglomerativeClustering

    with np.load(args.features, allow_pickle=True) as cached:
        rows = list(cached["rows"])
    with np.load(args.audit) as cached:
        keys = cached["keys"]
        hard_failures = cached["hard_failures"]
        reward_error = cached["reward_abs_error"]
    size = len(rows)
    distances = np.memmap(args.distance, mode="r", dtype=np.float32, shape=(size, size))
    complete = np.flatnonzero((hard_failures == 0) & (reward_error == 0))
    incomplete = np.flatnonzero(~((hard_failures == 0) & (reward_error == 0)))
    labels = np.full(size, -1, dtype=np.int32)
    core_trial = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=args.threshold,
    ).fit_predict(np.asarray(distances[np.ix_(complete, complete)])).astype(np.int32)
    labels[complete] = core_trial
    core_count = int(core_trial.max()) + 1

    nearest_complete_position = np.argmin(distances[np.ix_(incomplete, complete)], axis=1)
    nearest_complete_distance = distances[incomplete, complete[nearest_complete_position]]
    attached_mask = nearest_complete_distance <= args.threshold
    attached = incomplete[attached_mask]
    labels[attached] = labels[complete[nearest_complete_position[attached_mask]]]

    residual = incomplete[~attached_mask]
    residual_family_count = 0
    if len(residual) == 1:
        labels[residual] = core_count
        residual_family_count = 1
    elif len(residual) > 1:
        residual_trial = AgglomerativeClustering(
            n_clusters=None, metric="precomputed", linkage="average",
            distance_threshold=args.threshold,
        ).fit_predict(np.asarray(distances[np.ix_(residual, residual)])).astype(np.int32)
        labels[residual] = residual_trial + core_count
        residual_family_count = int(residual_trial.max()) + 1
    family_count = core_count + residual_family_count
    family_has_complete_core = np.arange(family_count) < core_count

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        labels=labels,
        keys=keys,
        complete_mask=(hard_failures == 0) & (reward_error == 0),
        family_has_complete_core=family_has_complete_core,
        incomplete_indices=incomplete,
        incomplete_nearest_complete_indices=complete[nearest_complete_position],
        incomplete_nearest_complete_distance=nearest_complete_distance,
    )
    sizes = Counter(labels.tolist())
    core_sizes = Counter(labels[labels < core_count].tolist())
    residual_sizes = Counter(labels[labels >= core_count].tolist())
    top56 = sum(sorted(core_sizes.values(), reverse=True)[:56])
    summary = {
        "schema_version": 1,
        "threshold": args.threshold,
        "rows": size,
        "complete_rows": len(complete),
        "incomplete_rows": len(incomplete),
        "complete_core_families": core_count,
        "failed_rows_attached_to_complete_core": int(attached_mask.sum()),
        "incomplete_only_rows": len(residual),
        "incomplete_only_families": residual_family_count,
        "families_total": family_count,
        "top56_complete_core_coverage_all_rows": top56 / size,
        "complete_core_sizes_desc": sorted(core_sizes.values(), reverse=True),
        "incomplete_only_sizes_desc": sorted(residual_sizes.values(), reverse=True),
        "nearest_complete_distance_incomplete": {
            "p10": float(np.quantile(nearest_complete_distance, .1)),
            "median": float(np.median(nearest_complete_distance)),
            "p90": float(np.quantile(nearest_complete_distance, .9)),
            "max": float(nearest_complete_distance.max(initial=0)),
        },
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
