"""Extend a causal multi-shop route tree by one exact-prefix-compatible stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from route_tree_utils import follow, load_resolver


ACTION_FIELDS = (
    "unit_op", "unit_item", "unit_amount", "unit_count",
    "market_op", "market_item", "market_amount", "market_count",
)


def prefix_hash(bank: np.lib.npyio.NpzFile, route: int, start: int, end: int) -> str:
    digest = hashlib.sha256()
    for field in ACTION_FIELDS:
        digest.update(np.ascontiguousarray(bank[field][route, start:end]).tobytes())
    return digest.hexdigest()


def candidate_key(panel_values: list[np.ndarray], mode: str) -> tuple[float, ...]:
    rates = np.asarray([np.mean(values > 0) for values in panel_values])
    margins = np.asarray([np.mean(values) for values in panel_values])
    pooled = np.concatenate(panel_values)
    if mode == "minimax_win":
        return float(np.min(rates)), float(np.mean(rates)), float(np.min(margins)), float(np.mean(pooled))
    if mode == "stable_win":
        return float(np.mean(rates) - 0.5 * np.std(rates)), float(np.mean(rates)), float(np.mean(pooled))
    return float(np.mean(pooled > 0)), float(np.mean(pooled))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--previous-tree", type=Path, required=True)
    parser.add_argument("--screens", type=Path, nargs="+", required=True)
    parser.add_argument("--mode", choices=("minimax_win", "stable_win", "pooled_win"), required=True)
    parser.add_argument("--route-capacity", type=int, default=256)
    parser.add_argument("--prefix-start", type=int, default=72)
    parser.add_argument("--min-games", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    resolver = load_resolver(args.previous_tree)
    previous_count = int(resolver["shop_count"])
    shop_count = previous_count + 1
    switch_step = 72 * shop_count
    if shop_count > 8 or switch_step >= 719:
        raise ValueError("no further public shop stage is available")

    bank = np.load(args.trace_bank, allow_pickle=False)
    route_count = int(bank["source_reward"].shape[0])
    if route_count > args.route_capacity:
        raise ValueError("route bank exceeds route capacity")
    source_shops = np.asarray(bank["source_shop_sequence"], dtype=np.int16)
    hashes = [prefix_hash(bank, route, args.prefix_start, switch_step) for route in range(route_count)]

    panels = []
    common_routes: set[int] | None = None
    for path in args.screens:
        matrix = np.load(path.with_suffix(".npz"), allow_pickle=False)
        route_ids = np.asarray(
            matrix["route_ids"] if "route_ids" in matrix.files else np.arange(matrix["margin"].shape[2]),
            dtype=np.int32,
        )
        lookup = {int(route): index for index, route in enumerate(route_ids.tolist())}
        current = set(lookup)
        common_routes = current if common_routes is None else common_routes & current
        panels.append({
            "path": str(path),
            "matrix": matrix,
            "lookup": lookup,
            "observed": np.asarray(matrix["observed_shops"]),
            "margin": np.asarray(matrix["margin"]),
        })
    assert common_routes is not None

    contexts: dict[tuple[int, ...], list[tuple[int, int, int, int]]] = {}
    current_by_prefix: dict[tuple[int, ...], int] = {}
    for panel_index, panel in enumerate(panels):
        observed = panel["observed"]
        lookup = panel["lookup"]
        for seat in range(observed.shape[0]):
            for seed in range(observed.shape[1]):
                prefix, current_route = follow(
                    resolver, observed, lookup, seat, seed, previous_count
                )
                if len(prefix) != previous_count or current_route not in lookup:
                    continue
                next_shop = int(observed[seat, seed, lookup[current_route], previous_count])
                if not 0 <= next_shop < 8:
                    continue
                full_prefix = tuple((*prefix, next_shop))
                if full_prefix in current_by_prefix and current_by_prefix[full_prefix] != current_route:
                    raise RuntimeError("same public shop prefix resolved to multiple current routes")
                current_by_prefix[full_prefix] = current_route
                contexts.setdefault(full_prefix, []).append((panel_index, seat, seed, current_route))

    route_by_prefix: dict[str, int] = {}
    rows = []
    exact_prefix_checks = 0
    for prefix in sorted(contexts):
        current_route = current_by_prefix[prefix]
        if current_route not in common_routes:
            continue
        compatible = np.asarray(
            [route for route in sorted(common_routes) if hashes[route] == hashes[current_route]],
            dtype=np.int32,
        )
        for route in compatible:
            for field in ACTION_FIELDS:
                if not np.array_equal(
                    bank[field][current_route, args.prefix_start:switch_step],
                    bank[field][route, args.prefix_start:switch_step],
                ):
                    raise RuntimeError("prefix hash collision")
            exact_prefix_checks += 1
        source_matched = compatible[source_shops[compatible, shop_count - 1] == prefix[-1]]
        eligible = np.unique(np.append(source_matched, current_route)).astype(np.int32)

        values_by_route: dict[int, dict[int, list[int]]] = {
            int(route): {} for route in eligible
        }
        for panel_index, seat, seed, _ in contexts[prefix]:
            panel = panels[panel_index]
            for route in eligible:
                value = int(panel["margin"][seat, seed, panel["lookup"][int(route)]])
                values_by_route[int(route)].setdefault(panel_index, []).append(value)
        baseline_panels = [np.asarray(values) for values in values_by_route[current_route].values()]
        selected = current_route
        if len(contexts[prefix]) >= args.min_games:
            scored = []
            for route in eligible:
                panel_values = [np.asarray(values) for values in values_by_route[int(route)].values()]
                if panel_values:
                    scored.append((candidate_key(panel_values, args.mode), int(route)))
            if scored:
                selected = max(scored)[1]
        key = ",".join(map(str, prefix))
        route_by_prefix[key] = selected
        selected_panels = [np.asarray(values) for values in values_by_route[selected].values()]
        rows.append({
            "shop_prefix": list(prefix),
            "current_route": current_route,
            "selected_route": selected,
            "eligible_routes": eligible.tolist(),
            "games": len(contexts[prefix]),
            "baseline_win_rate": float(np.mean(np.concatenate(baseline_panels) > 0)),
            "selected_win_rate": float(np.mean(np.concatenate(selected_panels) > 0)),
            "accepted": selected != current_route,
        })

    result = {
        "schema": "kaggriculture.front40_fusion.compatible-next-shop-route-map.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TRAINING_ONLY",
        "trace_bank": str(args.trace_bank),
        "previous_tree": str(args.previous_tree),
        "source_screens": [str(path) for path in args.screens],
        "mode": args.mode,
        "shop_count": shop_count,
        "route_capacity": args.route_capacity,
        "prefix_range": [args.prefix_start, switch_step],
        "min_games": args.min_games,
        "route_by_prefix": route_by_prefix,
        "routes": rows,
        "exact_prefix_checks": exact_prefix_checks,
        "boundary": (
            "Selection uses only the public unlocked-shop prefix. Unseen prefixes retain the previous route. "
            "Every changed suffix matches the full executed action prefix exactly."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "shop_count": shop_count,
        "prefixes": len(rows),
        "accepted": int(sum(bool(row["accepted"]) for row in rows)),
        "unique_routes": sorted(set(route_by_prefix.values())),
        "exact_prefix_checks": exact_prefix_checks,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
