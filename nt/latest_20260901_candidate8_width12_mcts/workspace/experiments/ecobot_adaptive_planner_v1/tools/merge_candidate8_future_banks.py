#!/usr/bin/env python3
"""Merge independent Candidate8 future banks for identical frozen states."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


IDENTITY_KEYS = (
    "state_id", "opponent", "prefix_seed", "seat", "decision_day",
    "signature", "family", "features", "feature_names",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--dataset-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        raise ValueError("at least two future banks are required")

    banks = [load(path) for path in args.inputs]
    reference = banks[0]
    for bank_index, bank in enumerate(banks[1:], start=1):
        for key in IDENTITY_KEYS:
            if not np.array_equal(reference[key], bank[key]):
                raise RuntimeError(f"bank {bank_index} differs at {key}")

    future_own = np.concatenate(
        [bank["future_own_cash"] for bank in banks], axis=1
    )
    future_opponent = np.concatenate(
        [bank["future_opponent_cash"] for bank in banks], axis=1
    )
    margins = future_own - future_opponent
    scores = np.where(margins > 0, 1.0, np.where(margins == 0, 0.5, 0.0))

    output = {
        key: value for key, value in reference.items()
        if key not in {
            "expected_score_rate", "expected_margin", "expected_own_cash",
            "expected_opponent_cash", "own_cash_std", "future_own_cash",
            "future_opponent_cash",
        }
    }
    output.update({
        "expected_score_rate": scores.mean(axis=1),
        "expected_margin": margins.mean(axis=1),
        "expected_own_cash": future_own.mean(axis=1),
        "expected_opponent_cash": future_opponent.mean(axis=1),
        "own_cash_std": future_own.std(axis=1),
        "future_own_cash": future_own,
        "future_opponent_cash": future_opponent,
    })
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.dataset_output, **output)

    payload = {
        "schema": "kaggriculture.candidate8_merged_future_banks.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "banks": len(banks),
            "rows": int(len(reference["state_id"])),
            "states": int(len(np.unique(reference["state_id"]))),
            "future_count": int(future_own.shape[1]),
            "feature_count": int(reference["features"].shape[1]),
        },
        "inputs": [
            {"path": str(path), "sha256": sha256(path)} for path in args.inputs
        ],
        "dataset": {
            "path": str(args.dataset_output),
            "sha256": sha256(args.dataset_output),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "dataset": str(args.dataset_output),
        **payload["summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
