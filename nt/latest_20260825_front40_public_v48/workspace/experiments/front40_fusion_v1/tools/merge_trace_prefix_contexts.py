"""Merge disjoint trace-prefix context panels along their seed axis."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    panels = [np.load(path, allow_pickle=False) for path in args.contexts]
    fields = panels[0].files
    if any(panel.files != fields for panel in panels[1:]):
        raise RuntimeError("context fields differ")
    decision_step = int(np.asarray(panels[0]["decision_step"]))
    arrays: dict[str, np.ndarray] = {}
    for field in fields:
        values = [np.asarray(panel[field]) for panel in panels]
        if field == "decision_step":
            if any(int(value) != decision_step for value in values):
                raise RuntimeError("decision step differs")
            arrays[field] = np.asarray(decision_step, dtype=values[0].dtype)
        elif field == "seeds":
            arrays[field] = np.concatenate(values, axis=0)
        else:
            if any(value.shape[0] != 2 for value in values):
                raise RuntimeError(f"{field}: expected leading seat axis")
            arrays[field] = np.concatenate(values, axis=1)
    if np.unique(arrays["seeds"]).size != arrays["seeds"].size:
        raise RuntimeError("seed panels overlap")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    payload = {
        "schema": "kaggriculture.front40_fusion.merged-trace-prefix-context.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "sources": [str(path) for path in args.contexts],
        "decision_step": decision_step,
        "seed_count": int(arrays["seeds"].size),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "shapes": {field: list(value.shape) for field, value in arrays.items()},
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "seed_count": payload["seed_count"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
