"""Merge disjoint route columns evaluated on exactly the same event seeds."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PER_ROUTE_FIELDS = (
    "cash",
    "opponent_cash",
    "margin",
    "invalid",
    "resync",
    "hard",
    "observed_shops",
)
SOURCE_FIELDS = (
    "source_episode_id",
    "source_reward",
    "source_first_shop",
    "route_ids",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screens", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.screens
    ]
    matrices = [
        np.load(path.with_suffix(".npz"), allow_pickle=False)
        for path in args.screens
    ]
    reference = matrices[0]
    for matrix in matrices[1:]:
        if not np.array_equal(reference["seeds"], matrix["seeds"]):
            raise RuntimeError("screens use different seeds")
        if not np.array_equal(reference["first_shop"], matrix["first_shop"]):
            raise RuntimeError("screens disagree on first public shop")

    route_ids = np.concatenate(
        [np.asarray(matrix["route_ids"], dtype=np.int32) for matrix in matrices]
    )
    if np.unique(route_ids).size != route_ids.size:
        raise RuntimeError("screen route columns overlap")
    order = np.argsort(route_ids)
    arrays: dict[str, np.ndarray] = {
        "seeds": np.asarray(reference["seeds"]),
        "first_shop": np.asarray(reference["first_shop"]),
        "route_ids": route_ids[order],
    }
    for field in PER_ROUTE_FIELDS:
        if not all(field in matrix.files for matrix in matrices):
            raise RuntimeError(f"field missing from a screen: {field}")
        arrays[field] = np.concatenate(
            [np.asarray(matrix[field]) for matrix in matrices], axis=2
        )[:, :, order]
    for field in SOURCE_FIELDS[:-1]:
        arrays[field] = np.concatenate(
            [np.asarray(matrix[field]) for matrix in matrices], axis=0
        )[order]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    hard_total = int(np.sum(arrays["hard"]))
    all_done = all(bool(payload.get("all_done")) for payload in payloads)
    payload = {
        "schema": "kaggriculture.front40_fusion.merged-route-columns.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [str(path) for path in args.screens],
        "seed_count": int(arrays["seeds"].size),
        "route_count": int(arrays["route_ids"].size),
        "route_ids": arrays["route_ids"].tolist(),
        "games": int(2 * arrays["seeds"].size * arrays["route_ids"].size),
        "all_done": all_done,
        "hard_counter_total": hard_total,
        "matrix": str(args.output.with_suffix(".npz")),
        "status": "PASS" if all_done and hard_total == 0 else "FAIL",
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "routes": payload["route_count"],
                "games": payload["games"],
            }
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
