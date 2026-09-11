"""Evaluate frozen first-shop route maps on a counterfactual screen matrix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def stats(margin: np.ndarray, hard: np.ndarray) -> dict:
    return {
        "games": int(margin.size),
        "wins": int(np.sum(margin > 0)),
        "win_rate": float(np.mean(margin > 0)),
        "mean_margin": float(np.mean(margin)),
        "median_margin": float(np.median(margin)),
        "hard_counter_total": int(np.sum(hard)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--route-maps", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    screen = np.load(args.matrix)
    margin = np.asarray(screen["margin"])
    hard = np.asarray(screen["hard"])
    first = np.asarray(screen["first_shop"])
    screen_routes = margin.shape[-1]
    global_routes = np.asarray(
        screen["route_ids"] if "route_ids" in screen else np.arange(screen_routes),
        dtype=np.int32,
    )
    lookup = {int(route): index for index, route in enumerate(global_routes)}
    rows = []
    for map_path in args.route_maps:
        mapping = np.asarray(
            json.loads(map_path.read_text(encoding="utf-8"))["route_ids"], dtype=np.int32
        )
        if mapping.shape != (8,):
            raise ValueError(f"route map shape is not (8,): {map_path}")
        if any(int(route) not in lookup for route in mapping):
            raise ValueError(f"route map selects route absent from screen: {map_path}")
        selected_margin = np.zeros(first.shape, dtype=margin.dtype)
        selected_hard = np.zeros(first.shape, dtype=hard.dtype)
        by_shop = []
        for shop in range(8):
            mask = first == shop
            column = lookup[int(mapping[shop])]
            selected_margin[mask] = margin[:, :, column][mask]
            selected_hard[mask] = hard[:, :, column][mask]
            shop_stats = stats(selected_margin[mask], selected_hard[mask])
            shop_stats["shop_id"] = shop
            shop_stats["route_id"] = int(mapping[shop])
            by_shop.append(shop_stats)
        row = {
            "route_map": str(map_path),
            "route_ids": mapping.tolist(),
            "aggregate": stats(selected_margin, selected_hard),
            "by_first_shop": by_shop,
        }
        rows.append(row)

    oracle_margin = np.max(margin, axis=2)
    result = {
        "schema": "kaggriculture.front40_fusion.first-shop-route-map-evaluation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "matrix": str(args.matrix),
        "screen_route_ids": global_routes.tolist(),
        "oracle": stats(oracle_margin, np.zeros_like(oracle_margin)),
        "route_maps": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
