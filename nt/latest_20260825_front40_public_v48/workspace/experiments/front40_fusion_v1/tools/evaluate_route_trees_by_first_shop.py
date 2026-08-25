"""Compare causal two-shop route trees by observable first-shop condition."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def summarize(values: np.ndarray, hard: np.ndarray) -> dict[str, object]:
    return {
        "games": int(values.size),
        "wins": int(np.sum(values > 0)),
        "win_rate": float(np.mean(values > 0)) if values.size else 0.0,
        "mean_margin": float(np.mean(values)) if values.size else 0.0,
        "hard_counter_total": int(np.sum(hard)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--trees", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    screen = np.load(args.matrix, allow_pickle=False)
    margin = np.asarray(screen["margin"])
    hard = np.asarray(screen["hard"])
    observed = np.asarray(screen["observed_shops"])
    route_ids = np.asarray(
        screen["route_ids"]
        if "route_ids" in screen.files
        else np.arange(margin.shape[2]),
        dtype=np.int32,
    )
    lookup = {int(route): index for index, route in enumerate(route_ids)}
    first_by_seat_seed = observed[:, :, 0, 0]

    rows = []
    for tree_path in args.trees:
        payload = json.loads(tree_path.read_text(encoding="utf-8"))
        base_routes = np.asarray(payload["route_ids"], dtype=np.int32)
        if base_routes.shape != (8,):
            raise ValueError(f"tree base map is not shape (8,): {tree_path}")
        by_first = "second_route_ids_by_first_shop" in payload
        second_routes = np.asarray(
            payload[
                "second_route_ids_by_first_shop"
                if by_first
                else "second_route_ids"
            ],
            dtype=np.int32,
        )
        values_by_first: list[list[int]] = [[] for _ in range(8)]
        hard_by_first: list[list[int]] = [[] for _ in range(8)]
        values_by_seat: list[list[int]] = [[], []]
        hard_by_seat: list[list[int]] = [[], []]
        all_values: list[int] = []
        all_hard: list[int] = []
        for seat in range(margin.shape[0]):
            for seed in range(margin.shape[1]):
                first = int(first_by_seat_seed[seat, seed])
                base = int(base_routes[first])
                if base not in lookup:
                    raise ValueError(f"base route {base} absent from {args.matrix}")
                base_column = lookup[base]
                second = int(observed[seat, seed, base_column, 1])
                selected = int(
                    second_routes[first, second]
                    if by_first
                    else second_routes[base, second]
                )
                if selected not in lookup:
                    raise ValueError(
                        f"selected route {selected} absent from {args.matrix}"
                    )
                column = lookup[selected]
                value = int(margin[seat, seed, column])
                hard_value = int(hard[seat, seed, column])
                values_by_first[first].append(value)
                hard_by_first[first].append(hard_value)
                values_by_seat[seat].append(value)
                hard_by_seat[seat].append(hard_value)
                all_values.append(value)
                all_hard.append(hard_value)
        rows.append(
            {
                "tree": str(tree_path),
                "aggregate": summarize(
                    np.asarray(all_values), np.asarray(all_hard)
                ),
                "by_seat": [
                    {"seat": seat, **summarize(
                        np.asarray(values_by_seat[seat]),
                        np.asarray(hard_by_seat[seat]),
                    )}
                    for seat in range(2)
                ],
                "by_first_shop": [
                    {
                        "first_shop": first,
                        **summarize(
                            np.asarray(values_by_first[first]),
                            np.asarray(hard_by_first[first]),
                        ),
                    }
                    for first in range(8)
                ],
            }
        )

    result = {
        "schema": "kaggriculture.front40_fusion.route-tree-first-shop-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "matrix": str(args.matrix),
        "trees": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "trees": [
            {
                "tree": Path(row["tree"]).name,
                "win_rate": row["aggregate"]["win_rate"],
            }
            for row in rows
        ],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
