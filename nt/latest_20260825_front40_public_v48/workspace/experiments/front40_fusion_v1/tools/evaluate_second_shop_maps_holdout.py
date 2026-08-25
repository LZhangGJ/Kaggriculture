"""Evaluate compatible second-shop maps on seed-grouped holdout contexts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def summarize(values: np.ndarray, hard: np.ndarray) -> dict:
    return {
        "games": int(values.size),
        "wins": int(np.sum(values > 0)),
        "win_rate": float(np.mean(values > 0)),
        "mean_margin": float(np.mean(values)),
        "median_margin": float(np.median(values)),
        "hard_counter_total": int(np.sum(hard)),
    }


def summarize_by_seat(
    values: np.ndarray, hard: np.ndarray, seats: np.ndarray
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seat in (0, 1):
        mask = seats == seat
        rows.append({"seat": seat, **summarize(values[mask], hard[mask])})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--base-map", type=Path, required=True)
    parser.add_argument("--second-maps", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-start", type=int, default=192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    screen = np.load(args.matrix)
    margin = np.asarray(screen["margin"])
    hard = np.asarray(screen["hard"])
    observed = np.asarray(screen["observed_shops"])
    global_routes = np.asarray(
        screen["route_ids"] if "route_ids" in screen else np.arange(margin.shape[2]),
        dtype=np.int32,
    )
    route_lookup = {int(route): index for index, route in enumerate(global_routes)}
    base_routes = np.asarray(
        json.loads(args.base_map.read_text(encoding="utf-8"))["route_ids"],
        dtype=np.int32,
    )
    if base_routes.shape != (8,):
        raise ValueError("base route map must have shape (8,)")
    seats, seeds, _ = margin.shape
    if not 0 <= args.holdout_start < seeds:
        raise ValueError("invalid holdout start")

    base_values = []
    base_hard = []
    contexts = []
    context_seats = []
    for seat in range(seats):
        for seed in range(args.holdout_start, seeds):
            first_shop = int(observed[seat, seed, 0, 0])
            base = int(base_routes[first_shop])
            if base not in route_lookup:
                raise ValueError(f"base route {base} absent from screen")
            base_column = route_lookup[base]
            second_shop = int(observed[seat, seed, base_column, 1])
            contexts.append((seat, seed, first_shop, second_shop, base))
            context_seats.append(seat)
            base_values.append(int(margin[seat, seed, base_column]))
            base_hard.append(int(hard[seat, seed, base_column]))
    base_values_array = np.asarray(base_values, dtype=np.int64)
    base_hard_array = np.asarray(base_hard, dtype=np.int64)
    context_seats_array = np.asarray(context_seats, dtype=np.int8)

    rows = []
    for map_path in args.second_maps:
        payload = json.loads(map_path.read_text(encoding="utf-8"))
        by_first = "second_route_ids_by_first_shop" in payload
        second_routes = np.asarray(
            payload[
                "second_route_ids_by_first_shop" if by_first else "second_route_ids"
            ],
            dtype=np.int32,
        )
        values = []
        hard_values = []
        changed = []
        for seat, seed, first_shop, second_shop, base in contexts:
            selected = int(
                second_routes[first_shop, second_shop]
                if by_first
                else second_routes[base, second_shop]
            )
            if selected not in route_lookup:
                raise ValueError(f"selected route {selected} absent from screen")
            selected_column = route_lookup[selected]
            values.append(int(margin[seat, seed, selected_column]))
            hard_values.append(int(hard[seat, seed, selected_column]))
            changed.append(selected != base)
        values_array = np.asarray(values, dtype=np.int64)
        hard_array = np.asarray(hard_values, dtype=np.int64)
        changed_array = np.asarray(changed, dtype=bool)
        row = {
            "map": str(map_path),
            "routing_index": (
                "first_shop,second_shop" if by_first else "legacy_base_route,second_shop"
            ),
            "accepted_pairs": int(sum(bool(item.get("accepted")) for item in payload["routes"])),
            "holdout": summarize(values_array, hard_array),
            "by_seat": summarize_by_seat(
                values_array, hard_array, context_seats_array
            ),
            "changed_games": int(np.sum(changed_array)),
            "net_wins_vs_base": int(np.sum(values_array > 0) - np.sum(base_values_array > 0)),
            "mean_margin_gain_vs_base": float(np.mean(values_array - base_values_array)),
        }
        rows.append(row)

    result = {
        "schema": "kaggriculture.front40_fusion.second-shop-holdout.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "matrix": str(args.matrix),
        "base_map": str(args.base_map),
        "holdout_seed_index_range": [args.holdout_start, seeds - 1],
        "base": {
            **summarize(base_values_array, base_hard_array),
            "by_seat": summarize_by_seat(
                base_values_array, base_hard_array, context_seats_array
            ),
        },
        "second_maps": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
