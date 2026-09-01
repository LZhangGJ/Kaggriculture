#!/usr/bin/env python3
"""Split a SWITCH NPZ into group-atomic early/middle/late datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from train_switch_pairwise_policy import group_keys
from train_switch_value_regressor import load


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def subset(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    rows = len(data["expected_delta"])
    return {
        name: value[mask] if value.ndim > 0 and value.shape[0] == rows else value
        for name, value in data.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--phase-ends", default="10,20,30")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    ends = [int(value) for value in args.phase_ends.split(",") if value.strip()]
    if len(ends) != 3 or ends != sorted(ends):
        raise ValueError("phase-ends must contain three increasing integers")
    data = load(args.dataset)
    names = [str(value) for value in data["feature_names"]]
    if "day" not in names:
        raise ValueError("dataset has no day feature")
    day = np.asarray(data["features"][:, names.index("day")], dtype=np.int64)
    phase = np.where(day <= ends[0], 0, np.where(day <= ends[1], 1, 2))
    keys = group_keys(data)
    for key in np.unique(keys):
        values = np.unique(phase[keys == key])
        if len(values) != 1:
            raise ValueError(f"phase is not group atomic for group {key}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = ("early", "middle", "late")
    outputs: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        mask = phase == index
        part = subset(data, mask)
        path = args.output_dir / f"{args.stem}_{label}.npz"
        np.savez_compressed(path, **part)
        outputs.append({
            "phase": label,
            "day_range": (
                [0, ends[0]] if index == 0
                else [ends[0] + 1, ends[1]] if index == 1
                else [ends[1] + 1, ends[2]]
            ),
            "path": str(path),
            "sha256": sha256(path),
            "rows": int(np.sum(mask)),
            "groups": int(len(np.unique(keys[mask]))),
        })
    payload = {
        "schema": "kaggriculture.switch-dataset-phase-split.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
        "phase_ends": ends,
        "group_atomic": True,
        "outputs": outputs,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
