"""Evaluate a causal three-shop route tree on a saved route screen."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def stats(values: np.ndarray, hard: np.ndarray) -> dict[str, object]:
    return {
        "games": int(values.size),
        "wins": int(np.sum(values > 0)),
        "win_rate": float(np.mean(values > 0)),
        "mean_margin": float(np.mean(values)),
        "median_margin": float(np.median(values)),
        "hard_counter_total": int(np.sum(hard)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    screen = np.load(args.matrix, allow_pickle=False)
    margin = np.asarray(screen["margin"])
    hard = np.asarray(screen["hard"])
    observed = np.asarray(screen["observed_shops"])
    route_ids = np.asarray(
        screen["route_ids"] if "route_ids" in screen.files else np.arange(margin.shape[2]),
        dtype=np.int32,
    )
    lookup = {int(route): index for index, route in enumerate(route_ids.tolist())}
    tree = json.loads(args.tree.read_text(encoding="utf-8"))
    base = np.asarray(tree["route_ids"], dtype=np.int32)
    second = np.asarray(tree["second_route_ids_by_first_shop"], dtype=np.int32)
    third = np.asarray(tree["third_route_ids_by_shop"], dtype=np.int32)
    if base.shape != (8,) or second.shape != (8, 8) or third.shape != (8, 8, 8):
        raise ValueError("invalid three-shop tree shapes")

    two_values: list[int] = []
    two_hard: list[int] = []
    three_values: list[int] = []
    three_hard: list[int] = []
    by_seat_values = [[], []]
    by_seat_hard = [[], []]
    by_first_values = [[], [], [], [], [], [], [], []]
    by_first_hard = [[], [], [], [], [], [], [], []]
    changed = 0
    for seat in range(margin.shape[0]):
        for seed in range(margin.shape[1]):
            first_shop = int(observed[seat, seed, 0, 0])
            base_route = int(base[first_shop])
            base_column = lookup[base_route]
            second_shop = int(observed[seat, seed, base_column, 1])
            two_route = int(second[first_shop, second_shop])
            two_column = lookup[two_route]
            third_shop = int(observed[seat, seed, two_column, 2])
            three_route = int(third[first_shop, second_shop, third_shop])
            three_column = lookup[three_route]
            two_values.append(int(margin[seat, seed, two_column]))
            two_hard.append(int(hard[seat, seed, two_column]))
            value = int(margin[seat, seed, three_column])
            hard_value = int(hard[seat, seed, three_column])
            three_values.append(value)
            three_hard.append(hard_value)
            by_seat_values[seat].append(value)
            by_seat_hard[seat].append(hard_value)
            by_first_values[first_shop].append(value)
            by_first_hard[first_shop].append(hard_value)
            changed += int(three_route != two_route)

    two_values_array = np.asarray(two_values)
    three_values_array = np.asarray(three_values)
    result = {
        "schema": "kaggriculture.front40_fusion.third-shop-route-evaluation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "matrix": str(args.matrix),
        "tree": str(args.tree),
        "two_shop": stats(two_values_array, np.asarray(two_hard)),
        "three_shop": {
            **stats(three_values_array, np.asarray(three_hard)),
            "by_seat": [
                {"seat": seat, **stats(np.asarray(by_seat_values[seat]), np.asarray(by_seat_hard[seat]))}
                for seat in range(2)
            ],
            "by_first_shop": [
                {
                    "first_shop": first_shop,
                    **stats(
                        np.asarray(by_first_values[first_shop]),
                        np.asarray(by_first_hard[first_shop]),
                    ),
                }
                for first_shop in range(8)
            ],
        },
        "changed_games": changed,
        "net_wins_vs_two_shop": int(np.sum(three_values_array > 0) - np.sum(two_values_array > 0)),
        "mean_margin_gain_vs_two_shop": float(np.mean(three_values_array - two_values_array)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
