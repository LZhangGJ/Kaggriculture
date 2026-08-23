#!/usr/bin/env python3
"""Prove that adding late public feature captures does not change old outputs."""

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


def take_panel(reference, candidate, key: str, opponent_axis: int, seed_axis: int):
    opponent_ids = np.asarray(candidate["opponent_ids"], dtype=np.int32)
    seeds = np.asarray(candidate["seeds"], dtype=np.int64)
    reference_opponents = np.asarray(reference["opponent_ids"], dtype=np.int32)
    reference_seeds = np.asarray(reference["seeds"], dtype=np.int64)
    opponent_positions = [
        int(np.flatnonzero(reference_opponents == value)[0]) for value in opponent_ids
    ]
    seed_positions = [
        int(np.flatnonzero(reference_seeds == value)[0]) for value in seeds
    ]
    values = np.take(reference[key], opponent_positions, axis=opponent_axis)
    return np.take(values, seed_positions, axis=seed_axis)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--reference-outcomes", type=Path, required=True)
    parser.add_argument("--candidate-features", type=Path, required=True)
    parser.add_argument("--candidate-outcomes", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: resolve(getattr(args, name)) for name in (
        "reference_features", "reference_outcomes",
        "candidate_features", "candidate_outcomes",
    )}

    with np.load(paths["reference_features"], allow_pickle=False) as ref_f, np.load(
        paths["candidate_features"], allow_pickle=False
    ) as cand_f, np.load(paths["reference_outcomes"], allow_pickle=False) as ref_o, np.load(
        paths["candidate_outcomes"], allow_pickle=False
    ) as cand_o:
        feature_errors = {}
        for key in ("features", "features20", "features120"):
            expected = take_panel(ref_f, cand_f, key, opponent_axis=1, seed_axis=2)
            actual = np.asarray(cand_f[key])
            feature_errors[key] = float(
                np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
            )
        expected_wins = take_panel(ref_o, cand_o, "wins", opponent_axis=3, seed_axis=4)
        expected_margins = take_panel(
            ref_o, cand_o, "margins", opponent_axis=3, seed_axis=4
        )
        wins_equal = bool(np.array_equal(cand_o["wins"], expected_wins))
        margin_error = int(
            np.max(
                np.abs(
                    np.asarray(cand_o["margins"], dtype=np.int64)
                    - expected_margins.astype(np.int64)
                )
            )
        )
        late = {
            key: {
                "shape": list(np.asarray(cand_f[key]).shape),
                "finite": bool(np.all(np.isfinite(cand_f[key]))),
                "nonzero_values": int(np.count_nonzero(cand_f[key])),
            }
            for key in ("features168", "features216")
        }

    passed = (
        wins_equal
        and margin_error == 0
        and max(feature_errors.values()) == 0.0
        and all(row["finite"] and row["nonzero_values"] > 0 for row in late.values())
    )
    result = {
        "schema": "kaggriculture-extended-feature-probe-parity-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "old_feature_max_abs_errors": feature_errors,
        "wins_exactly_equal": wins_equal,
        "maximum_terminal_margin_error": margin_error,
        "late_feature_checks": late,
        "sources": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    receipt_path = resolve(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
