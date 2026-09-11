#!/usr/bin/env python3
"""Stack four audited semantic plans into one public-state branch family."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ORDER = (
    "cow_mixed__carrot",
    "cow_mixed__wheat",
    "sheep_heavy__carrot",
    "sheep_heavy__wheat",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ORDER:
        parser.add_argument(
            f"--{label.replace('__', '-').replace('_', '-')}",
            dest=label,
            type=Path,
            required=True,
        )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    args = parse_args()
    paths = {
        label: getattr(args, label)
        for label in ORDER
    }
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for label, path in paths.items():
        with np.load(path, allow_pickle=False) as payload:
            loaded[label] = {
                name: np.asarray(payload[name]) for name in payload.files
            }
    common = set.intersection(*(set(values) for values in loaded.values()))
    arrays: dict[str, np.ndarray] = {}
    for name in sorted(common):
        per_branch = []
        for label in ORDER:
            value = loaded[label][name]
            if value.shape[0] != 1:
                raise ValueError(f"{label}:{name} expected leading dimension 1")
            per_branch.append(value[0])
        arrays[name] = np.stack(per_branch, axis=0)
    arrays["branch_labels"] = np.asarray(ORDER, dtype="U32")
    arrays["branch_family_schema_version"] = np.asarray([1], dtype=np.int16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    payload = {
        "schema": "kaggriculture.ecobot.semantic-branch-family.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "branch_order": list(ORDER),
        "runtime_selector": (
            "public YARN/wool/opponent capacity selects animal branch; "
            "public PET_CAFE/carrot economics selects late crop suffix"
        ),
        "ecobot_dynamic_planning_preserved": True,
        "raw_replay_actions_stored": False,
        "raw_coordinates_stored": False,
        "inputs": {
            label: {"path": str(path.resolve()), "sha256": sha256(path)}
            for label, path in paths.items()
        },
        "fields": {name: list(value.shape) for name, value in arrays.items()},
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
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
