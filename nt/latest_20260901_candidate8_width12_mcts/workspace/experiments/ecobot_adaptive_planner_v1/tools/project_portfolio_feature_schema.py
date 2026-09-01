#!/usr/bin/env python3
"""Project a portfolio corpus onto a frozen reference feature schema."""

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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    source = np.load(args.input, allow_pickle=False)
    reference = np.load(args.reference, allow_pickle=False)
    source_names = [str(value) for value in source["feature_names"]]
    reference_names = [str(value) for value in reference["feature_names"]]
    source_at = {name: index for index, name in enumerate(source_names)}
    if len(source_at) != len(source_names):
        raise ValueError("source feature names are not unique")
    missing = [name for name in reference_names if name not in source_at]
    if missing:
        raise ValueError(f"reference features missing from source: {missing}")

    indices = np.asarray([source_at[name] for name in reference_names], dtype=np.int64)
    payload = {key: np.asarray(source[key]) for key in source.files}
    payload["features"] = np.asarray(source["features"])[:, indices]
    payload["feature_names"] = np.asarray(reference["feature_names"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    removed = [name for name in source_names if name not in set(reference_names)]
    receipt = {
        "schema": "kaggriculture.portfolio-feature-schema-projection.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(args.input), "sha256": sha256(args.input)},
        "reference": {
            "path": str(args.reference),
            "sha256": sha256(args.reference),
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256(args.output),
            "rows": int(payload["features"].shape[0]),
            "feature_dim": int(payload["features"].shape[1]),
        },
        "contract": {
            "labels_unchanged": True,
            "row_order_unchanged": True,
            "reference_order_preserved": True,
            "source_feature_dim": len(source_names),
            "reference_feature_dim": len(reference_names),
            "removed_feature_count": len(removed),
            "removed_features": removed,
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
