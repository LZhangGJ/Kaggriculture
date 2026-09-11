#!/usr/bin/env python3
"""Audit O1.1 feature visibility, variance, and semantic neutrality."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FORBIDDEN_NAME_PARTS = (
    "opponent_shed",
    "opponent_seed",
    "opponent_carried",
    "opponent_inventory_private",
    "opponent_route",
    "opponent_identity",
    "future_seed",
    "actual_future",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.baseline, allow_pickle=False) as old_data, np.load(
        args.candidate, allow_pickle=False
    ) as new_data:
        old = {name: np.asarray(old_data[name]) for name in old_data.files}
        new = {name: np.asarray(new_data[name]) for name in new_data.files}

    old_names = [str(value) for value in old["feature_names"]]
    new_names = [str(value) for value in new["feature_names"]]
    old_width = old["features"].shape[1]
    new_width = new["features"].shape[1]
    added_names = new_names[old_width:]
    comparable_keys = sorted(
        (set(old) & set(new)) - {"features", "feature_names"}
    )
    key_checks = {
        name: bool(np.array_equal(old[name], new[name]))
        for name in comparable_keys
    }
    prefix_exact = bool(
        old_names == new_names[:old_width]
        and np.array_equal(old["features"], new["features"][:, :old_width])
    )
    forbidden = [
        name
        for name in new_names
        if any(part in name.lower() for part in FORBIDDEN_NAME_PARTS)
    ]
    feature_stats = []
    for offset, name in enumerate(added_names, start=old_width):
        values = new["features"][:, offset]
        feature_stats.append(
            {
                "name": name,
                "minimum": int(values.min()),
                "maximum": int(values.max()),
                "unique_values": int(np.unique(values).size),
                "nonzero_fraction": float(np.mean(values != 0)),
                "variable": bool(np.unique(values).size > 1),
            }
        )
    passed = (
        old_width == 159
        and new_width == 230
        and len(added_names) == 71
        and len(new_names) == len(set(new_names))
        and prefix_exact
        and all(key_checks.values())
        and not forbidden
        and sum(row["variable"] for row in feature_stats) >= 40
    )
    payload = {
        "schema": "kaggriculture.candidate8-o11-feature-contract.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "boundary": (
            "The baseline 159 values, candidate signatures and all rollout "
            "outcomes must remain exact. Added fields may use only public "
            "state/history and may not encode opponent identity or future RNG."
        ),
        "inputs": {
            "baseline": {"path": str(args.baseline), "sha256": sha256(args.baseline)},
            "candidate": {"path": str(args.candidate), "sha256": sha256(args.candidate)},
        },
        "summary": {
            "rows": int(new["features"].shape[0]),
            "baseline_width": int(old_width),
            "candidate_width": int(new_width),
            "added_features": len(added_names),
            "variable_added_features": sum(
                int(row["variable"]) for row in feature_stats
            ),
            "unique_feature_names": len(new_names) == len(set(new_names)),
            "baseline_prefix_exact": prefix_exact,
            "comparable_arrays_exact": all(key_checks.values()),
            "forbidden_feature_names": forbidden,
        },
        "array_exactness": key_checks,
        "added_feature_stats": feature_stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
