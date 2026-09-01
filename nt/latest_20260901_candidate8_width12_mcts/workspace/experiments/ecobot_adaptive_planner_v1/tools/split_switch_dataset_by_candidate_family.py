#!/usr/bin/env python3
"""Create group-valid KEEP+crop and KEEP+animal SWITCH corpora.

The split is candidate-semantic rather than author- or route-specific.  Every
retained group contains exactly one KEEP row and at least one edit from the
requested public project family.
"""

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


def family_mask(destination: np.ndarray, family: str) -> np.ndarray:
    keep = destination < 0
    if family == "crop":
        return keep | ((destination >= 0) & (destination < 5))
    if family == "animal":
        return keep | (destination >= 5)
    raise ValueError(f"unknown family: {family}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    data = load(args.dataset)
    names = [str(value) for value in data["feature_names"]]
    if "destination_project" not in names:
        raise ValueError("dataset has no destination_project feature")
    destination = np.asarray(
        data["features"][:, names.index("destination_project")], dtype=np.int64
    )
    keys = group_keys(data)
    candidate_rank = np.asarray(data["candidate_rank"], dtype=np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, object]] = []
    for family in ("crop", "animal"):
        requested = family_mask(destination, family)
        retained_groups: list[int] = []
        for key in np.unique(keys):
            rows = np.flatnonzero((keys == key) & requested)
            if len(rows) < 2:
                continue
            if int(np.sum(candidate_rank[rows] < 0)) != 1:
                raise ValueError(f"group {key} does not contain exactly one KEEP row")
            retained_groups.append(int(key))
        mask = requested & np.isin(keys, np.asarray(retained_groups, dtype=np.int64))
        part = subset(data, mask)
        output = args.output_dir / f"{args.stem}_{family}.npz"
        np.savez_compressed(output, **part)
        edit_rows = int(np.sum(mask & (candidate_rank >= 0)))
        outputs.append({
            "family": family,
            "path": str(output),
            "sha256": sha256(output),
            "rows": int(np.sum(mask)),
            "edit_rows": edit_rows,
            "groups": int(len(retained_groups)),
            "exactly_one_keep_per_group": True,
        })

    payload = {
        "schema": "kaggriculture.switch-dataset-candidate-family-split.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
        "project_partition": {"crop": [0, 4], "animal": [5, 7], "keep": -1},
        "outputs": outputs,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
