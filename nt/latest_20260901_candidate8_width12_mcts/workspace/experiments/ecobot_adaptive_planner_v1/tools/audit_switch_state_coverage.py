#!/usr/bin/env python3
"""Audit independent decision-state coverage by day, phase and farm scale."""

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


def as_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--phase-ends", default="10,20,30")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    phase_ends = [int(value) for value in args.phase_ends.split(",")]
    if len(phase_ends) < 1 or phase_ends != sorted(phase_ends):
        raise ValueError("phase ends must be a sorted non-empty list")

    reports: list[dict[str, object]] = []
    for raw_path in args.dataset:
        path = Path(raw_path)
        raw = np.load(path, allow_pickle=False)
        names = [str(value) for value in raw["feature_names"]]
        at = {name: index for index, name in enumerate(names)}
        required = {
            "day", "quadrants", "candidate_quadrants",
        }
        missing = sorted(required - set(at))
        if missing:
            raise ValueError(f"{path}: missing features {missing}")

        features = np.asarray(raw["features"])
        rows = len(features)
        if rows == 0:
            reports.append({
                "path": str(path),
                "sha256": sha256(path),
                "rows": 0,
                "independent_decision_groups": 0,
                "day_group_counts": {},
                "phase_group_counts": {},
                "observed_quadrant_group_counts": {},
                "candidate_quadrant_row_counts": {},
                "phase_by_observed_quadrant": {
                    str(phase_id): {str(scale): 0 for scale in range(1, 5)}
                    for phase_id in range(len(phase_ends))
                },
                "day_by_observed_quadrant": {},
                "missing_day_quadrant_cells_within_observed_days": [],
                "has_late_phase_groups": False,
                "has_four_quadrant_groups": False,
            })
            continue
        source = np.asarray(
            raw.get("source_dataset_index", np.zeros(rows)), dtype=np.int64
        )
        prefix = np.asarray(raw["prefix_seed"], dtype=np.int64)
        seat = np.asarray(raw["seat"], dtype=np.int64)
        group_matrix = np.stack([source, prefix, seat], axis=1)
        _, group_index = np.unique(group_matrix, axis=0, return_index=True)

        day = features[group_index, at["day"]].astype(np.int64)
        quadrants = features[group_index, at["quadrants"]].astype(np.int64)
        candidate_quadrants = features[:, at["candidate_quadrants"]].astype(
            np.int64
        )
        phase = np.searchsorted(phase_ends, day, side="left")
        phase = np.minimum(phase, len(phase_ends) - 1)

        cross: dict[str, dict[str, int]] = {}
        for phase_id in range(len(phase_ends)):
            selected = phase == phase_id
            cross[str(phase_id)] = {
                str(scale): int(np.sum(selected & (quadrants == scale)))
                for scale in range(1, 5)
            }

        day_quadrant_cells = {
            f"d{int(day_value)}_q{scale}": int(
                np.sum((day == day_value) & (quadrants == scale))
            )
            for day_value in np.unique(day)
            for scale in range(1, 5)
        }
        missing_cells = [
            cell for cell, count in day_quadrant_cells.items() if count == 0
        ]

        reports.append({
            "path": str(path),
            "sha256": sha256(path),
            "rows": rows,
            "independent_decision_groups": int(len(group_index)),
            "day_group_counts": as_counts(day),
            "phase_group_counts": as_counts(phase),
            "observed_quadrant_group_counts": as_counts(quadrants),
            "candidate_quadrant_row_counts": as_counts(candidate_quadrants),
            "phase_by_observed_quadrant": cross,
            "day_by_observed_quadrant": day_quadrant_cells,
            "missing_day_quadrant_cells_within_observed_days": missing_cells,
            "has_late_phase_groups": bool(np.any(day > phase_ends[-2]))
            if len(phase_ends) > 1 else True,
            "has_four_quadrant_groups": bool(np.any(quadrants == 4)),
        })

    payload = {
        "schema": "kaggriculture.switch-state-coverage-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase_ends": phase_ends,
        "phase_semantics": [
            "early: day <= 10",
            "middle: 11 <= day <= 20",
            "late: 21 <= day <= 30",
        ] if phase_ends == [10, 20, 30] else "custom",
        "group_identity": ["source_dataset_index", "prefix_seed", "seat"],
        "datasets": reports,
        "gate": {
            "all_have_late_phase_groups": all(
                bool(report["has_late_phase_groups"]) for report in reports
            ),
            "all_have_four_quadrant_groups": all(
                bool(report["has_four_quadrant_groups"]) for report in reports
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
