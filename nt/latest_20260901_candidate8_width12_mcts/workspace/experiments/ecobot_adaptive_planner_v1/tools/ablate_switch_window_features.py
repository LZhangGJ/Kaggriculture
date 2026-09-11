#!/usr/bin/env python3
"""Zero the 30 generic 2/4/8-day project-realisation features.

The labels, groups and all pre-existing state features remain byte-identical.
This creates a controlled ablation corpus for deciding whether the new window
features improve independent ranking rather than merely increasing model size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


COMPONENTS = (
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as loaded:
        data = {name: loaded[name].copy() for name in loaded.files}
    features = np.asarray(data["features"], dtype=np.int32)
    names = [str(value) for value in data["feature_names"]]
    at = {name: index for index, name in enumerate(names)}
    selected = [
        f"{plan}_{component}"
        for plan in ("baseline", "candidate")
        for component in COMPONENTS
    ]
    missing = [name for name in selected if name not in at]
    if missing:
        raise ValueError(f"missing window features: {missing}")
    columns = np.asarray([at[name] for name in selected], dtype=np.int64)
    before = features[:, columns].copy()
    features[:, columns] = 0
    data["features"] = features

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    payload = {
        "schema": "kaggriculture.switch-window-feature-ablation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(args.input), "sha256": sha256(args.input)},
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
        "rows": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "ablated_features": selected,
        "ablated_feature_count": len(selected),
        "nonzero_cells_before": int(np.count_nonzero(before)),
        "nonzero_cells_after": int(np.count_nonzero(features[:, columns])),
        "labels_and_group_keys_unchanged": True,
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
