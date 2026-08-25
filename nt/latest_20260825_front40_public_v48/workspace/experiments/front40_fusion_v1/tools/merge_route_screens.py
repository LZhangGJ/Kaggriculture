"""Merge disjoint route-screen matrices before selecting a shop map."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PER_SEED_FIELDS = (
    "first_shop", "cash", "opponent_cash", "margin", "invalid", "resync", "hard"
)
SOURCE_FIELDS = (
    "source_episode_id", "source_reward", "source_first_shop", "route_ids"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screens", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrices = []
    payloads = []
    for screen in args.screens:
        payload = json.loads(screen.read_text(encoding="utf-8"))
        payloads.append(payload)
        matrices.append(np.load(screen.with_suffix(".npz"), allow_pickle=False))
    reference = matrices[0]
    for matrix in matrices[1:]:
        for field in SOURCE_FIELDS:
            if not np.array_equal(reference[field], matrix[field]):
                raise RuntimeError(f"source field differs across screens: {field}")
    arrays = {
        "seeds": np.concatenate([matrix["seeds"] for matrix in matrices], axis=0),
    }
    for field in PER_SEED_FIELDS:
        arrays[field] = np.concatenate([matrix[field] for matrix in matrices], axis=1)
    for field in SOURCE_FIELDS:
        arrays[field] = np.asarray(reference[field])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    payload = {
        "schema": "kaggriculture.front40_fusion.merged-route-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [str(path) for path in args.screens],
        "seed_count": int(arrays["seeds"].shape[0]),
        "route_count": int(arrays["source_reward"].shape[0]),
        "games": int(2 * arrays["seeds"].shape[0] * arrays["source_reward"].shape[0]),
        "all_done": all(bool(payload.get("all_done")) for payload in payloads),
        "hard_counter_total": int(np.sum(arrays["hard"])),
        "matrix": str(args.output.with_suffix(".npz")),
        "status": "PASS" if all(bool(payload.get("all_done")) for payload in payloads) and int(np.sum(arrays["hard"])) == 0 else "FAIL",
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
