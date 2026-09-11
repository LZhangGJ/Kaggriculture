#!/usr/bin/env python3
"""Split a counterfactual corpus by a state feature without splitting groups."""

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


def group_keys(data: dict[str, np.ndarray]) -> np.ndarray:
    rows = len(data["prefix_seed"])
    source = np.asarray(
        data.get("source_dataset_index", np.zeros(rows)), dtype=np.int64
    )
    seed = np.asarray(data["prefix_seed"], dtype=np.int64)
    seat = np.asarray(data["seat"], dtype=np.int64)
    return source * 10**10 + seed * 2 + seat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--values", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    raw = np.load(args.dataset, allow_pickle=False)
    data = {name: np.asarray(raw[name]) for name in raw.files}
    names = [str(value) for value in data["feature_names"]]
    if args.feature not in names:
        raise ValueError(f"missing feature {args.feature!r}")
    feature_index = names.index(args.feature)
    state_value = np.asarray(data["features"][:, feature_index], dtype=np.int64)
    keys = group_keys(data)
    for key in np.unique(keys):
        values = np.unique(state_value[keys == key])
        if len(values) != 1:
            raise ValueError(
                f"feature {args.feature} is not constant in group {key}: {values}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, object]] = []
    row_count = len(state_value)
    requested = [int(value) for value in args.values.split(",")]
    for value in requested:
        selected = state_value == value
        output = args.output_dir / f"{args.stem}_{args.feature}{value}.npz"
        subset: dict[str, np.ndarray] = {}
        for name, array in data.items():
            subset[name] = array[selected] if len(array) == row_count else array
        np.savez_compressed(output, **subset)
        outputs.append({
            "value": value,
            "path": str(output),
            "sha256": sha256(output),
            "rows": int(np.sum(selected)),
            "groups": int(len(np.unique(keys[selected]))),
        })

    payload = {
        "schema": "kaggriculture.switch-dataset-state-split.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
        "feature": args.feature,
        "group_atomic": True,
        "outputs": outputs,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
