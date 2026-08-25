"""Evaluate any causal multi-shop route-tree stage against its predecessor."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from route_tree_utils import follow, load_resolver


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

    payload = json.loads(args.tree.read_text(encoding="utf-8"))
    shop_count = int(payload["shop_count"])
    resolver = load_resolver(args.tree)
    screen = np.load(args.matrix, allow_pickle=False)
    margin = np.asarray(screen["margin"])
    hard = np.asarray(screen["hard"])
    observed = np.asarray(screen["observed_shops"])
    route_ids = np.asarray(
        screen["route_ids"] if "route_ids" in screen.files else np.arange(margin.shape[2]),
        dtype=np.int32,
    )
    lookup = {int(route): index for index, route in enumerate(route_ids.tolist())}

    previous_values = []
    current_values = []
    current_hard = []
    by_seat_values = [[], []]
    by_seat_hard = [[], []]
    changed = 0
    fallback = 0
    configured = set(payload["route_by_prefix"])
    for seat in range(margin.shape[0]):
        for seed in range(margin.shape[1]):
            previous_prefix, previous_route = follow(
                resolver, observed, lookup, seat, seed, shop_count - 1
            )
            current_prefix, current_route = follow(
                resolver, observed, lookup, seat, seed, shop_count
            )
            previous_values.append(int(margin[seat, seed, lookup[previous_route]]))
            value = int(margin[seat, seed, lookup[current_route]])
            hard_value = int(hard[seat, seed, lookup[current_route]])
            current_values.append(value)
            current_hard.append(hard_value)
            by_seat_values[seat].append(value)
            by_seat_hard[seat].append(hard_value)
            changed += int(current_route != previous_route)
            fallback += int(",".join(map(str, current_prefix)) not in configured)

    previous_array = np.asarray(previous_values)
    current_array = np.asarray(current_values)
    result = {
        "schema": "kaggriculture.front40_fusion.multi-shop-route-tree-evaluation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "matrix": str(args.matrix),
        "tree": str(args.tree),
        "shop_count": shop_count,
        "previous": stats(previous_array, np.zeros_like(previous_array)),
        "current": {
            **stats(current_array, np.asarray(current_hard)),
            "by_seat": [
                {"seat": seat, **stats(np.asarray(by_seat_values[seat]), np.asarray(by_seat_hard[seat]))}
                for seat in range(2)
            ],
        },
        "changed_games": changed,
        "unseen_prefix_fallback_games": fallback,
        "net_wins_vs_previous": int(np.sum(current_array > 0) - np.sum(previous_array > 0)),
        "mean_margin_gain_vs_previous": float(np.mean(current_array - previous_array)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
