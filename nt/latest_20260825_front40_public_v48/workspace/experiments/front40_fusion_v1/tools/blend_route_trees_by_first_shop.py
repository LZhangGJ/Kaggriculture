"""Blend complete causal route-tree rows by observable first-shop condition."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--tree-index-by-first",
        required=True,
        help="Eight comma-separated indices into --trees.",
    )
    parser.add_argument("--route-capacity", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.trees]
    selected_tree = np.asarray(
        [int(value) for value in args.tree_index_by_first.split(",")],
        dtype=np.int16,
    )
    if selected_tree.shape != (8,):
        raise ValueError("tree-index-by-first must contain eight values")
    if np.any(selected_tree < 0) or np.any(selected_tree >= len(payloads)):
        raise ValueError("tree index is out of range")

    base_routes = np.zeros(8, dtype=np.int16)
    second_by_first = np.zeros((8, 8), dtype=np.int16)
    rows: list[dict[str, object]] = []
    for first in range(8):
        tree_index = int(selected_tree[first])
        payload = payloads[tree_index]
        base = int(payload["route_ids"][first])
        if "second_route_ids_by_first_shop" in payload:
            suffixes = np.asarray(
                payload["second_route_ids_by_first_shop"][first], dtype=np.int16
            )
        else:
            suffixes = np.asarray(payload["second_route_ids"][base], dtype=np.int16)
        if suffixes.shape != (8,):
            raise ValueError(f"second-route row is not shape (8,): {args.trees[tree_index]}")
        base_routes[first] = base
        second_by_first[first] = suffixes
        rows.append(
            {
                "first_shop": first,
                "tree_index": tree_index,
                "source_tree": str(args.trees[tree_index]),
                "base_route": base,
                "second_routes": suffixes.tolist(),
            }
        )

    legacy = np.repeat(
        np.arange(args.route_capacity, dtype=np.int16)[:, None], 8, axis=1
    )
    legacy_safe = True
    owner: dict[int, np.ndarray] = {}
    for first, base in enumerate(base_routes.tolist()):
        row = second_by_first[first]
        if base in owner and not np.array_equal(owner[base], row):
            legacy_safe = False
        owner[base] = row
        legacy[base] = row

    result = {
        "schema": "kaggriculture.front40_fusion.first-shop-blended-route-tree.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TRAINING_ONLY",
        "source_trees": [str(path) for path in args.trees],
        "tree_index_by_first_shop": selected_tree.tolist(),
        "route_capacity": args.route_capacity,
        "route_ids": base_routes.tolist(),
        "second_route_ids_by_first_shop": second_by_first.tolist(),
        "second_route_ids": legacy.tolist(),
        "routing_index": "first_shop,second_shop",
        "legacy_route_index_is_lossless": legacy_safe,
        "routes": rows,
        "boundary": (
            "Runtime selection uses only the currently observed first and second shop. "
            "No player identity, Replay ID, event seed, or future event is an input."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "route_ids": result["route_ids"],
        "unique_routes": sorted(set(base_routes.tolist() + second_by_first.ravel().tolist())),
        "legacy_route_index_is_lossless": legacy_safe,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
