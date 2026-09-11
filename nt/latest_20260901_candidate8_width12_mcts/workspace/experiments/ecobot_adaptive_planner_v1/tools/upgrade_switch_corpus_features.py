#!/usr/bin/env python3
"""Upgrade SWITCH corpus features while preserving frozen rollout labels.

The old corpus owns every stochastic continuation label.  The new corpus is
generated with the same prefix states and candidate ordering, but can use one
future sample because only its deterministic feature matrix is consumed.  A
row is reusable only when provenance keys and every shared feature match
exactly; otherwise this tool refuses the upgrade.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def identity_rows(dataset: np.lib.npyio.NpzFile) -> list[tuple[int, int, int, int]]:
    required = ("source_dataset_index", "prefix_seed", "seat", "candidate_rank")
    missing = [name for name in required if name not in dataset.files]
    if missing:
        raise ValueError(f"missing row identity arrays: {missing}")
    columns = [np.asarray(dataset[name], dtype=np.int64) for name in required]
    row_count = len(np.asarray(dataset["features"]))
    if any(column.shape != (row_count,) for column in columns):
        raise ValueError("row identity array shape mismatch")
    rows = list(zip(*(column.tolist() for column in columns), strict=True))
    if len(set(rows)) != len(rows):
        raise ValueError("duplicate row identity in corpus")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    labels = np.load(args.labels, allow_pickle=False)
    features = np.load(args.features, allow_pickle=False)
    old_matrix = np.asarray(labels["features"])
    new_matrix = np.asarray(features["features"])
    old_names = np.asarray(labels["feature_names"])
    new_names = np.asarray(features["feature_names"])
    old_dim = int(old_matrix.shape[1])
    new_dim = int(new_matrix.shape[1])
    if new_dim <= old_dim:
        raise ValueError(f"feature dimension did not increase: {old_dim} -> {new_dim}")
    if not np.array_equal(old_names, new_names[:old_dim]):
        raise ValueError("shared feature names or ordering changed")

    old_identity = identity_rows(labels)
    new_identity = identity_rows(features)
    if set(old_identity) != set(new_identity):
        missing = len(set(old_identity) - set(new_identity))
        extra = len(set(new_identity) - set(old_identity))
        raise ValueError(f"row identity mismatch: missing={missing}, extra={extra}")
    new_position = {identity: index for index, identity in enumerate(new_identity)}
    order = np.asarray([new_position[identity] for identity in old_identity], dtype=np.int64)
    aligned_new = new_matrix[order]
    shared_new = aligned_new[:, :old_dim]
    if old_matrix.shape != shared_new.shape:
        raise ValueError("shared feature matrix shape mismatch")
    if not np.array_equal(old_matrix, shared_new):
        difference = np.abs(
            old_matrix.astype(np.float64) - shared_new.astype(np.float64)
        )
        index = np.unravel_index(int(np.nanargmax(difference)), difference.shape)
        raise ValueError(
            "shared deterministic features changed: "
            f"max_abs={difference[index]} at row={index[0]}, col={index[1]}"
        )

    payload = {name: np.asarray(labels[name]) for name in labels.files}
    payload["features"] = aligned_new
    payload["feature_names"] = new_names

    # Labels and stochastic continuations must be byte-identical to the old
    # source.  Record their hashes before and after writing for an auditable
    # proof rather than relying only on aggregate means.
    frozen_label_names = [
        name for name in labels.files
        if name not in {"features", "feature_names"}
    ]
    frozen_hashes = {
        name: array_sha256(np.asarray(labels[name])) for name in frozen_label_names
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    written = np.load(args.output, allow_pickle=False)
    written_hashes = {
        name: array_sha256(np.asarray(written[name])) for name in frozen_label_names
    }
    if frozen_hashes != written_hashes:
        raise RuntimeError("frozen rollout labels changed while writing upgraded corpus")

    future_counts = np.asarray(payload.get("future_sample_count", []), dtype=np.int64)
    receipt = {
        "schema": "kaggriculture.switch-corpus-feature-upgrade.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "contract": {
            "new_corpus_used_only_for_deterministic_features": True,
            "old_corpus_owns_all_stochastic_labels": True,
            "requires_exact_row_identity": True,
            "requires_exact_shared_features": True,
            "frozen_label_arrays_byte_identical": True,
        },
        "labels": {
            "path": str(args.labels),
            "sha256": file_sha256(args.labels),
            "rows": int(len(old_matrix)),
            "feature_dim": old_dim,
        },
        "features": {
            "path": str(args.features),
            "sha256": file_sha256(args.features),
            "rows": int(len(new_matrix)),
            "feature_dim": new_dim,
        },
        "validation": {
            "identity_rows_equal": True,
            "shared_feature_names_equal": True,
            "shared_feature_values_exact": True,
            "shared_feature_max_abs_difference": 0.0,
            "frozen_array_count": len(frozen_label_names),
            "frozen_array_hashes": frozen_hashes,
        },
        "output": {
            "path": str(args.output),
            "sha256": file_sha256(args.output),
            "rows": int(len(payload["features"])),
            "feature_dim": int(payload["features"].shape[1]),
            "future_sample_count_min": (
                int(future_counts.min()) if future_counts.size else None
            ),
            "future_sample_count_max": (
                int(future_counts.max()) if future_counts.size else None
            ),
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
