#!/usr/bin/env python3
"""Combine compatible portfolio counterfactual NPZ datasets with provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument(
        "--manifest", type=Path, action="append", default=[],
        help=(
            "Generation manifest containing a datasets[] list. This avoids "
            "Windows command-line limits for large route-by-day corpora."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--prefer-more-future-samples", action="store_true",
        help=(
            "For duplicate (prefix_seed, seat, candidate_rank) rows with "
            "identical features, retain the observation with more future "
            "rollouts.  Refuse feature-mismatched collisions."
        ),
    )
    args = parser.parse_args()

    inputs = list(args.input)
    for manifest_path in args.manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Generation manifests are written by the Windows orchestrator.  The
        # combiner is also used inside WSL, where a backslash is a literal
        # filename character rather than a separator.  Forward slashes work on
        # both platforms, so normalize only manifest-provided relative paths.
        manifest_inputs = [
            Path(str(row["dataset"]).replace("\\", "/"))
            for row in manifest["datasets"]
        ]
        inputs.extend(manifest_inputs)
    if not inputs:
        parser.error("at least one --input or --manifest is required")
    if len(set(inputs)) != len(inputs):
        raise ValueError("duplicate input dataset path")

    datasets = [np.load(path, allow_pickle=False) for path in inputs]
    derived_keys = {
        "source_dataset_index",
        "source_dataset_names",
        "future_sample_count",
    }
    reference_files = set(datasets[0].files) - derived_keys
    reference_names = np.asarray(datasets[0]["feature_names"])
    for path, dataset in zip(inputs, datasets, strict=True):
        if set(dataset.files) - derived_keys != reference_files:
            raise ValueError(f"schema mismatch: {path}")
        if not np.array_equal(dataset["feature_names"], reference_names):
            raise ValueError(f"feature-name mismatch: {path}")

    static_keys = {"feature_names"}
    payload: dict[str, np.ndarray] = {}
    for key in reference_files:
        if key in static_keys:
            payload[key] = np.asarray(datasets[0][key])
        elif key == "future_delta_samples":
            widths = [
                int(np.asarray(dataset[key]).shape[1])
                for dataset in datasets
                if np.asarray(dataset[key]).ndim == 2
            ]
            if not widths:
                raise ValueError("all future-delta sample arrays are schema-less")
            max_width = max(widths)
            padded = []
            for dataset in datasets:
                value = np.asarray(dataset[key], dtype=np.float64)
                if value.size == 0 and value.ndim == 1:
                    value = value.reshape(0, max_width)
                if value.ndim != 2:
                    raise ValueError(
                        f"invalid future-delta sample shape: {value.shape}"
                    )
                if value.shape[1] < max_width:
                    value = np.pad(
                        value,
                        ((0, 0), (0, max_width - value.shape[1])),
                        mode="constant",
                        constant_values=np.nan,
                    )
                padded.append(value)
            payload[key] = np.concatenate(padded, axis=0)
        else:
            reference = np.asarray(datasets[0][key])
            arrays = []
            for dataset in datasets:
                value = np.asarray(dataset[key])
                # A decision point with no feasible SWITCH candidates writes
                # empty row arrays as shape (0,) in NumPy.  Preserve that
                # source in provenance, but restore the trailing schema so it
                # concatenates as zero rows instead of being silently dropped.
                if value.size == 0 and value.ndim != reference.ndim:
                    value = value.reshape((0,) + reference.shape[1:])
                arrays.append(value)
            payload[key] = np.concatenate(
                arrays, axis=0
            )
    # Preserve leaf-dataset provenance when an input is itself a combined
    # corpus.  Collapsing an entire 60-route corpus to one source would make
    # per-route generalisation audits impossible and could merge groups whose
    # seed ranges happen to overlap.
    source_indices: list[np.ndarray] = []
    source_names: list[str] = []
    source_offset = 0
    future_sample_counts: list[np.ndarray] = []
    for path, dataset in zip(inputs, datasets, strict=True):
        rows = len(dataset["features"])
        if "source_dataset_index" in dataset.files:
            local_index = np.asarray(dataset["source_dataset_index"], dtype=np.int64)
            local_names = [str(name) for name in dataset["source_dataset_names"]]
            if len(local_index) != rows:
                raise ValueError(f"source-index row mismatch: {path}")
            if local_index.size and (
                int(local_index.min()) < 0 or int(local_index.max()) >= len(local_names)
            ):
                raise ValueError(f"invalid nested source index: {path}")
            source_indices.append(local_index + source_offset)
            source_names.extend(local_names)
            source_offset += len(local_names)
        else:
            source_indices.append(np.full(rows, source_offset, dtype=np.int64))
            source_names.append(path.name)
            source_offset += 1

        if "future_sample_count" in dataset.files:
            count = np.asarray(dataset["future_sample_count"], dtype=np.int16)
            if count.shape != (rows,):
                raise ValueError(f"future-sample-count row mismatch: {path}")
        else:
            future_samples = np.asarray(dataset["future_delta_samples"])
            inferred_width = (
                int(future_samples.shape[1])
                if future_samples.ndim == 2
                else int(payload["future_delta_samples"].shape[1])
            )
            count = np.full(
                rows,
                inferred_width,
                dtype=np.int16,
            )
        future_sample_counts.append(count)

    payload["source_dataset_index"] = np.concatenate(source_indices).astype(
        np.int32, copy=False
    )
    payload["source_dataset_names"] = np.asarray(source_names)
    payload["future_sample_count"] = np.concatenate(future_sample_counts)

    deduplicated_rows = 0
    if args.prefer_more_future_samples:
        logical_identity = (
            np.asarray(payload["prefix_seed"], dtype=np.int64) * 64
            + np.asarray(payload["seat"], dtype=np.int64) * 32
            + np.asarray(payload["candidate_rank"], dtype=np.int64) + 1
        )
        keep = np.ones(len(logical_identity), dtype=bool)
        order = np.argsort(logical_identity, kind="stable")
        boundaries = np.flatnonzero(
            np.r_[True, logical_identity[order][1:] != logical_identity[order][:-1], True]
        )
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            rows = order[begin:end]
            if len(rows) <= 1:
                continue
            reference_features = payload["features"][rows[0]]
            if any(
                not np.array_equal(payload["features"][row], reference_features)
                for row in rows[1:]
            ):
                raise ValueError(
                    "logical identity collision has different public features: "
                    f"seed={payload['prefix_seed'][rows[0]]}, "
                    f"seat={payload['seat'][rows[0]]}, "
                    f"rank={payload['candidate_rank'][rows[0]]}"
                )
            counts = payload["future_sample_count"][rows]
            selected = int(rows[int(np.argmax(counts))])
            keep[rows] = False
            keep[selected] = True
        deduplicated_rows = int(np.sum(~keep))
        row_count = len(keep)
        for key, value in list(payload.items()):
            if key in {"feature_names", "source_dataset_names"}:
                continue
            if value.ndim >= 1 and value.shape[0] == row_count:
                payload[key] = value[keep]

    # Dataset identity must not accidentally collapse two states from separate
    # source files that reused the same seed range.  Refuse duplicates rather
    # than silently leaking one public state across train/validation folds.
    group = (
        np.asarray(payload["source_dataset_index"], dtype=np.int64) * 10**10
        + np.asarray(payload["prefix_seed"], dtype=np.int64) * 2
        + np.asarray(payload["seat"], dtype=np.int64)
    )
    rank = np.asarray(payload["candidate_rank"], dtype=np.int64)
    identity = group * 32 + rank
    if len(np.unique(identity)) != len(identity):
        raise ValueError("duplicate (prefix_seed, seat, candidate_rank) rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    receipt = {
        "schema": "kaggriculture.portfolio-expected-value-combine.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [
            {
                "path": str(path),
                "sha256": sha256(path),
                "rows": int(len(dataset["features"])),
                "groups": int(
                    len(
                        np.unique(
                            np.asarray(dataset["prefix_seed"], dtype=np.int64) * 2
                            + np.asarray(dataset["seat"], dtype=np.int64)
                        )
                    )
                ),
            }
            for path, dataset in zip(inputs, datasets, strict=True)
        ],
        "manifests": [
            {"path": str(path), "sha256": sha256(path)}
            for path in args.manifest
        ],
        "deduplication": {
            "prefer_more_future_samples": args.prefer_more_future_samples,
            "removed_rows": deduplicated_rows,
            "requires_identical_features": True,
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256(args.output),
            "rows": int(len(payload["features"])),
            "groups": int(len(np.unique(group))),
            "feature_dim": int(payload["features"].shape[1]),
            "future_sample_width": int(payload["future_delta_samples"].shape[1]),
            "future_sample_count_min": int(payload["future_sample_count"].min()),
            "future_sample_count_max": int(payload["future_sample_count"].max()),
            "leaf_sources": int(len(payload["source_dataset_names"])),
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt["output"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
