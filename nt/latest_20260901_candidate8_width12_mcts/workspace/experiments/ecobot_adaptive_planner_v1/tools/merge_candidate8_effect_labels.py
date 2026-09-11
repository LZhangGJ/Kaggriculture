#!/usr/bin/env python3
"""Merge one-future execution-effect labels into a frozen outcome dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EFFECT_KEYS = (
    "first_action_change_offset",
    "action_change_count_24",
    "action_change_count_full",
    "first_state_change_offset",
    "state_change_count_24",
    "state_change_count_full",
)
IDENTITY_KEYS = (
    "state_id", "opponent", "prefix_seed", "seat", "decision_day",
    "signature", "family", "features", "feature_names",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-dataset", required=True, type=Path)
    parser.add_argument("--effect-dataset", required=True, type=Path)
    parser.add_argument("--merged-output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    with np.load(args.outcome_dataset, allow_pickle=False) as outcome_data:
        outcome = {key: np.asarray(outcome_data[key]) for key in outcome_data.files}
    with np.load(args.effect_dataset, allow_pickle=False) as effect_data:
        missing = set(EFFECT_KEYS) - set(effect_data.files)
        if missing:
            raise ValueError(f"effect dataset lacks {sorted(missing)}")
        identity_exact = {
            key: bool(np.array_equal(outcome[key], effect_data[key]))
            for key in IDENTITY_KEYS
        }
        if not all(identity_exact.values()):
            raise RuntimeError(f"dataset identity mismatch: {identity_exact}")
        for key in EFFECT_KEYS:
            outcome[key] = np.asarray(effect_data[key])

    args.merged_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.merged_output, **outcome)
    state_effect = outcome["state_change_count_full"] > 0
    action_effect = outcome["action_change_count_full"] > 0
    non_keep = np.ones(len(state_effect), dtype=bool)
    for state in np.unique(outcome["state_id"]):
        non_keep[np.flatnonzero(outcome["state_id"] == state)[0]] = False
    payload = {
        "schema": "kaggriculture.candidate8-effect-label-merge.v1",
        "status": "PASS",
        "inputs": {
            "outcome_dataset": {
                "path": str(args.outcome_dataset),
                "sha256": sha256(args.outcome_dataset),
            },
            "effect_dataset": {
                "path": str(args.effect_dataset),
                "sha256": sha256(args.effect_dataset),
            },
        },
        "identity_arrays_exact": identity_exact,
        "rows": int(len(state_effect)),
        "states": int(len(np.unique(outcome["state_id"]))),
        "non_keep_action_effect_rate": float(np.mean(action_effect[non_keep])),
        "non_keep_state_effect_rate": float(np.mean(state_effect[non_keep])),
        "action_only_no_state_effect_rate": float(np.mean(
            action_effect[non_keep] & ~state_effect[non_keep]
        )),
        "merged_output": {
            "path": str(args.merged_output),
            "sha256": sha256(args.merged_output),
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
