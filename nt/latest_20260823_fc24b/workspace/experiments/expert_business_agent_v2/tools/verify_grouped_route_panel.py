#!/usr/bin/env python3
"""Verify a grouped JAX route panel against a previously accepted superset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    reference_path = resolve(args.reference)
    candidate_path = resolve(args.candidate)
    receipt_path = resolve(args.receipt)

    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        candidate_path, allow_pickle=False
    ) as candidate:
        candidate_ids = np.asarray(candidate["candidate_ids"], dtype=np.int32)
        opponent_ids = np.asarray(candidate["opponent_ids"], dtype=np.int32)
        seeds = np.asarray(candidate["seeds"], dtype=np.int64)
        reference_candidate_ids = np.asarray(reference["candidate_ids"], dtype=np.int32)
        reference_opponent_ids = np.asarray(reference["opponent_ids"], dtype=np.int32)
        reference_seeds = np.asarray(reference["seeds"], dtype=np.int64)
        candidate_positions = [
            int(np.flatnonzero(reference_candidate_ids == value)[0])
            for value in candidate_ids
        ]
        opponent_positions = [
            int(np.flatnonzero(reference_opponent_ids == value)[0])
            for value in opponent_ids
        ]
        seed_positions = [
            int(np.flatnonzero(reference_seeds == value)[0]) for value in seeds
        ]
        expected_wins = np.take(
            np.take(
                np.take(reference["wins"], candidate_positions, axis=0),
                opponent_positions,
                axis=3,
            ),
            seed_positions,
            axis=4,
        )
        expected_margins = np.take(
            np.take(
                np.take(reference["margins"], candidate_positions, axis=0),
                opponent_positions,
                axis=3,
            ),
            seed_positions,
            axis=4,
        )
        actual_wins = np.asarray(candidate["wins"])
        actual_margins = np.asarray(candidate["margins"], dtype=np.int64)

    win_equal = bool(np.array_equal(actual_wins, expected_wins))
    margin_error = int(
        np.max(np.abs(actual_margins - expected_margins.astype(np.int64)))
    )
    result = {
        "schema": "kaggriculture-grouped-route-panel-parity-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if win_equal and margin_error == 0 else "FAIL",
        "candidate_ids": candidate_ids.astype(int).tolist(),
        "opponent_ids": opponent_ids.astype(int).tolist(),
        "seeds": seeds.astype(int).tolist(),
        "shape": list(actual_margins.shape),
        "wins_exactly_equal": win_equal,
        "margins_exactly_equal": margin_error == 0,
        "maximum_terminal_margin_error": margin_error,
        "reference": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "candidate": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
