#!/usr/bin/env python3
"""Append continuation-planner context to an existing SWITCH corpus.

The counterfactual labels do not change.  Every leaf dataset already has a
JSON receipt containing the exact generic genome overrides used for its
rollouts, so the missing policy-state fields can be reconstructed without
resampling any game.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CONTEXT = (
    ("continuation_max_quadrants", "max_quadrants", 1.0),
    ("continuation_stop_new_crops_day", "stop_new_crops_day", 1.0),
    ("continuation_stop_new_animals_day", "stop_new_animals_day", 1.0),
    (
        "continuation_local_edit_hold_days",
        "portfolio_local_edit_hold_days",
        1.0,
    ),
    (
        "continuation_land_capacity_trigger_x100",
        "land_capacity_trigger_fraction",
        100.0,
    ),
    (
        "continuation_proactive_land_investment_x100",
        "proactive_land_investment",
        100.0,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_overrides(receipt: Path) -> dict[str, float]:
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    result: dict[str, float] = {}
    for raw in payload.get("overrides", []):
        name, separator, value = str(raw).partition("=")
        if separator:
            result[name] = float(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--leaf-receipt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}
    names = [str(value) for value in data["feature_names"]]
    context_names = [row[0] for row in CONTEXT]
    if any(name in names for name in context_names):
        raise ValueError("input already contains continuation context")
    if len(names) != 310:
        raise ValueError(f"expected legacy 310 features, found {len(names)}")

    source_names = [str(value) for value in data["source_dataset_names"]]
    source_context = np.zeros((len(source_names), len(CONTEXT)), dtype=np.int32)
    profiles: Counter[tuple[int, ...]] = Counter()
    receipt_hashes: list[dict[str, str]] = []
    for source_index, source_name in enumerate(source_names):
        leaf_receipt = args.leaf_receipt_dir / f"{Path(source_name).stem}.json"
        if not leaf_receipt.is_file():
            raise FileNotFoundError(f"missing leaf receipt: {leaf_receipt}")
        overrides = parse_overrides(leaf_receipt)
        missing = [key for _, key, _ in CONTEXT if key not in overrides]
        if missing:
            raise ValueError(
                f"{leaf_receipt} lacks continuation fields: {missing}"
            )
        values = tuple(
            int(round(overrides[key] * scale)) for _, key, scale in CONTEXT
        )
        source_context[source_index] = values
        profiles[values] += 1
        receipt_hashes.append({
            "path": str(leaf_receipt), "sha256": sha256(leaf_receipt)
        })

    source_index = np.asarray(data["source_dataset_index"], dtype=np.int64)
    if source_index.min(initial=0) < 0 or source_index.max(initial=-1) >= len(source_names):
        raise ValueError("source_dataset_index is outside source metadata")
    row_context = source_context[source_index]
    features = np.asarray(data["features"])
    if features.ndim != 2 or features.shape[1] != len(names):
        raise ValueError(f"invalid feature matrix shape: {features.shape}")
    data["features"] = np.concatenate([features, row_context], axis=1)
    data["feature_names"] = np.asarray(names + context_names)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    payload = {
        "schema": "kaggriculture.switch-continuation-context-augmentation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(args.input), "sha256": sha256(args.input)},
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
        "rows": int(len(features)),
        "old_feature_dim": int(features.shape[1]),
        "new_feature_dim": int(data["features"].shape[1]),
        "context_features": context_names,
        "leaf_sources": len(source_names),
        "all_leaf_receipts_present": True,
        "labels_resampled": False,
        "unique_profiles": [
            {
                "values": dict(zip(context_names, profile, strict=True)),
                "leaf_sources": count,
            }
            for profile, count in sorted(profiles.items())
        ],
        "leaf_receipts": receipt_hashes,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "sha256": payload["output"]["sha256"],
        "rows": payload["rows"],
        "feature_dim": payload["new_feature_dim"],
        "leaf_sources": payload["leaf_sources"],
        "unique_profiles": payload["unique_profiles"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
