"""Blend complete three-shop tree rows by the observed first shop."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=Path, nargs="+", required=True)
    parser.add_argument("--tree-index-by-first", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.trees]
    selected = np.asarray(
        [int(value) for value in args.tree_index_by_first.split(",")], dtype=np.int16
    )
    if selected.shape != (8,):
        raise ValueError("tree-index-by-first must contain eight values")
    if np.any(selected < 0) or np.any(selected >= len(payloads)):
        raise ValueError("tree index is out of range")

    base = np.empty(8, dtype=np.int16)
    second = np.empty((8, 8), dtype=np.int16)
    third = np.empty((8, 8, 8), dtype=np.int16)
    rows = []
    for first_shop in range(8):
        tree_index = int(selected[first_shop])
        payload = payloads[tree_index]
        payload_base = np.asarray(payload["route_ids"], dtype=np.int16)
        payload_second = np.asarray(payload["second_route_ids_by_first_shop"], dtype=np.int16)
        payload_third = np.asarray(payload["third_route_ids_by_shop"], dtype=np.int16)
        if (
            payload_base.shape != (8,)
            or payload_second.shape != (8, 8)
            or payload_third.shape != (8, 8, 8)
        ):
            raise ValueError(f"invalid tree shapes: {args.trees[tree_index]}")
        base[first_shop] = payload_base[first_shop]
        second[first_shop] = payload_second[first_shop]
        third[first_shop] = payload_third[first_shop]
        rows.append(
            {
                "first_shop": first_shop,
                "tree_index": tree_index,
                "source_tree": str(args.trees[tree_index]),
                "base_route": int(base[first_shop]),
                "second_routes": second[first_shop].tolist(),
                "third_routes": third[first_shop].tolist(),
            }
        )

    payload = {
        "schema": "kaggriculture.front40_fusion.first-shop-blended-third-shop-tree.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TRAINING_ONLY",
        "source_trees": [str(path) for path in args.trees],
        "tree_index_by_first_shop": selected.tolist(),
        "route_ids": base.tolist(),
        "second_route_ids_by_first_shop": second.tolist(),
        "third_route_ids_by_shop": third.tolist(),
        "routing_index": "first_shop,second_shop,third_shop",
        "routes": rows,
        "boundary": (
            "Every choice is indexed only by town shops already revealed in the current game. "
            "No identity, Replay ID, seed, or future event is used."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "route_ids": base.tolist(), "tree_index": selected.tolist()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

