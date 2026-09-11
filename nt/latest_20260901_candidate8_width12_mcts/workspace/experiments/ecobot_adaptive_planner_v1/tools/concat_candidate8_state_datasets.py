#!/usr/bin/env python3
"""Concatenate Candidate8 state datasets with unique state ids."""

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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_days(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--days")
    parser.add_argument("--dataset-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    wanted_days = parse_days(args.days)

    parts: dict[str, list[np.ndarray]] = {}
    feature_names = None
    state_offset = 0
    input_rows = []
    for path in args.inputs:
        with np.load(path, allow_pickle=False) as raw:
            data = {name: np.asarray(raw[name]) for name in raw.files}
        if feature_names is None:
            feature_names = data["feature_names"]
        elif not np.array_equal(feature_names, data["feature_names"]):
            raise RuntimeError(f"feature names differ in {path}")
        mask = np.ones(len(data["state_id"]), dtype=bool)
        if wanted_days is not None:
            mask &= np.isin(data["decision_day"], list(wanted_days))
        original_state = data["state_id"][mask]
        unique_states = np.unique(original_state)
        mapping = {
            int(value): state_offset + index
            for index, value in enumerate(unique_states)
        }
        remapped = np.asarray(
            [mapping[int(value)] for value in original_state], dtype=np.int32
        )
        state_offset += len(unique_states)
        for name, values in data.items():
            if name == "feature_names":
                continue
            selected = values[mask]
            if name == "state_id":
                selected = remapped
            parts.setdefault(name, []).append(selected)
        input_rows.append({
            "path": str(path),
            "sha256": sha256(path),
            "input_rows": int(len(data["state_id"])),
            "selected_rows": int(mask.sum()),
            "selected_states": int(len(unique_states)),
        })

    output = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
    output["feature_names"] = feature_names
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.dataset_output, **output)
    payload = {
        "schema": "kaggriculture.candidate8_concatenated_states.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "days": sorted(wanted_days) if wanted_days is not None else None,
        "summary": {
            "rows": int(len(output["state_id"])),
            "states": int(len(np.unique(output["state_id"]))),
            "opponents": int(len(np.unique(output["opponent"]))),
            "feature_count": int(output["features"].shape[1]),
            "future_count": int(output["future_own_cash"].shape[1]),
        },
        "inputs": input_rows,
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
