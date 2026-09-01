#!/usr/bin/env python3
"""Repair marginal first-cash features in an existing 346-field corpus.

The original window feature used any already-sellable inventory, which is
constant across counterfactual arms.  The corrected feature measures only the
first return of still-unrealised commitments and is exactly reconstructible
from the stored public/self plan fields and official timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CROP_FIRST = np.asarray([2, 2, 8, 10, 10], dtype=np.int64)
CROP_MAX_DAY = np.asarray([4, 3, 8, 10, 12], dtype=np.int64)
CROP_ONGOING = np.asarray([False, False, True, True, False])
ANIMAL_FIRST = np.asarray([4, 8, 6], dtype=np.int64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def first_cash_lag(
    features: np.ndarray, at: dict[str, int], prefix: str
) -> np.ndarray:
    rows = len(features)
    result = np.full(rows, 30, dtype=np.int32)
    for crop, name in enumerate(("wheat", "carrot", "tomato", "strawberry", "melon")):
        target = features[:, at[f"{prefix}_{name}"]]
        committed = features[:, at[f"committed_{name}"]]
        lag = CROP_FIRST[crop] if CROP_ONGOING[crop] else max(
            int(CROP_FIRST[crop]), int(CROP_MAX_DAY[crop])
        )
        result = np.where(target > committed, np.minimum(result, lag), result)
    setup_turns = features[:, at[f"{prefix}_setup_turns"]]
    hour = features[:, at["hour"]]
    # Match the native scorer exactly: setup delay is charged only while an
    # animal still has to be placed on the map.  Crop setup work and an animal
    # purchase that can remain carried do not delay the first biological cash
    # event.  All terms are already present in the frozen corpus.
    new_placements = np.zeros(rows, dtype=np.int32)
    for name in ("geese", "cows", "sheep"):
        target = features[:, at[f"{prefix}_{name}"]]
        field = features[:, at[f"field_{name}"]]
        new_placements += np.maximum(0, target - field)
    setup_delay = np.where(
        new_placements > 0,
        np.maximum(0, (hour + setup_turns + 2) // 24),
        0,
    )
    for animal, name in enumerate(("geese", "cows", "sheep")):
        target = features[:, at[f"{prefix}_{name}"]]
        owned = features[:, at[f"owned_{name}"]]
        lag = ANIMAL_FIRST[animal] + setup_delay
        result = np.where(target > owned, np.minimum(result, lag), result)
    return result.astype(np.int32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=True) as loaded:
        data = {name: loaded[name].copy() for name in loaded.files}
    names = [str(value) for value in data["feature_names"]]
    at = {name: index for index, name in enumerate(names)}
    if len(names) != 346:
        raise ValueError(f"expected 346 features, found {len(names)}")
    features = np.asarray(data["features"], dtype=np.int32)
    changes: dict[str, object] = {}
    for prefix in ("baseline", "candidate"):
        column = at[f"{prefix}_first_cash_lag"]
        before = features[:, column].copy()
        after = first_cash_lag(features, at, prefix)
        features[:, column] = after
        changes[prefix] = {
            "changed_rows": int(np.sum(before != after)),
            "before_unique": [int(value) for value in np.unique(before)],
            "after_unique": [int(value) for value in np.unique(after)],
        }
    # The KEEP arm is synthesised from the first exposed SWITCH candidate.
    # Corpora generated immediately after the 316 -> 346 schema extension
    # copied the old value components but not the 15 new window components.
    # Restore the invariant candidate == baseline for every KEEP row.
    keep_mask = np.asarray(data["candidate_rank"]) == -1
    window_components = (
        "first_cash_lag", "immediate_commitment_actions_x100",
        "today_action_capacity_x100", "today_deadline_slack_x100",
        "peak_daily_utilization_x100",
        "window_action_demand_lag2_x100",
        "window_action_demand_lag4_x100",
        "window_action_demand_lag8_x100",
        "window_action_capacity_lag2_x100",
        "window_action_capacity_lag4_x100",
        "window_action_capacity_lag8_x100",
        "window_net_cash_lag2", "window_net_cash_lag4",
        "window_net_cash_lag8", "minimum_window_slack_x100",
    )
    repaired_keep_cells = 0
    for component in window_components:
        baseline_column = at[f"baseline_{component}"]
        candidate_column = at[f"candidate_{component}"]
        repaired_keep_cells += int(np.sum(
            keep_mask &
            (features[:, candidate_column] != features[:, baseline_column])
        ))
        features[keep_mask, candidate_column] = features[
            keep_mask, baseline_column
        ]
    changes["keep_window_copy"] = {
        "rows": int(np.sum(keep_mask)),
        "changed_cells": repaired_keep_cells,
        "components": list(window_components),
    }
    data["features"] = features
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    payload = {
        "schema": "kaggriculture.switch-window-first-cash-repair.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(args.input), "sha256": sha256(args.input)},
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
        "rows": int(len(features)),
        "feature_dim": int(features.shape[1]),
        "official_timing_only": True,
        "changes": changes,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
