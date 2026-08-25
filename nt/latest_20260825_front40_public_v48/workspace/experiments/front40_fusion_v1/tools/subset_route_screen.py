"""Create a route-column subset of a saved counterfactual screen."""

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
SOURCE_FIELDS = ("source_episode_id", "source_reward", "source_first_shop")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--route-ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = np.load(args.screen.with_suffix(".npz"), allow_pickle=False)
    global_routes = np.asarray(
        source["route_ids"] if "route_ids" in source else np.arange(source["margin"].shape[2]),
        dtype=np.int32,
    )
    wanted = np.asarray([int(value) for value in args.route_ids.split(",") if value], np.int32)
    lookup = {int(route): index for index, route in enumerate(global_routes)}
    if wanted.size == 0 or np.unique(wanted).size != wanted.size:
        raise ValueError("route IDs must be non-empty and unique")
    missing = [int(route) for route in wanted if int(route) not in lookup]
    if missing:
        raise ValueError(f"routes absent from source: {missing}")
    columns = np.asarray([lookup[int(route)] for route in wanted], np.int32)

    arrays = {
        "seeds": np.asarray(source["seeds"]),
        "first_shop": np.asarray(source["first_shop"]),
        "route_ids": wanted,
    }
    for field in PER_ROUTE_FIELDS:
        if field in source:
            arrays[field] = np.take(np.asarray(source[field]), columns, axis=2)
    for field in SOURCE_FIELDS:
        arrays[field] = np.asarray(source[field])[columns]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    payload = {
        "schema": "kaggriculture.front40_fusion.route-screen-subset.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if int(np.sum(arrays["hard"])) == 0 else "FAIL",
        "source": str(args.screen),
        "seed_count": int(arrays["seeds"].size),
        "route_count": int(wanted.size),
        "route_ids": wanted.tolist(),
        "games": int(2 * arrays["seeds"].size * wanted.size),
        "all_done": True,
        "hard_counter_total": int(np.sum(arrays["hard"])),
        "matrix": str(args.output.with_suffix(".npz")),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "routes": wanted.tolist(), "games": payload["games"]}))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
