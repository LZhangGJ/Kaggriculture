"""Build a causal third-shop suffix map with exact executed-prefix compatibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ACTION_FIELDS = (
    "unit_op", "unit_item", "unit_amount", "unit_count",
    "market_op", "market_item", "market_amount", "market_count",
)
SHOP_NAMES = (
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
    "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE",
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
        return (
            float(np.min(rates)), float(np.mean(rates)),
            float(np.min(margins)), float(np.mean(pooled)),
        )
    if mode == "stable_win":
        return (
            float(np.mean(rates) - 0.5 * np.std(rates)),
            float(np.mean(rates)), float(np.mean(pooled)),
        )
    return float(np.mean(pooled > 0)), float(np.mean(pooled))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--two-shop-tree", type=Path, required=True)
    parser.add_argument("--screens", type=Path, nargs="+", required=True)
    parser.add_argument("--mode", choices=("minimax_win", "stable_win", "pooled_win"), required=True)
    parser.add_argument("--route-capacity", type=int, default=256)
    parser.add_argument("--prefix-start", type=int, default=72)
    parser.add_argument("--switch-step", type=int, default=216)
    parser.add_argument("--min-games", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bank = np.load(args.trace_bank, allow_pickle=False)
    route_count = int(bank["source_reward"].shape[0])
    if route_count > args.route_capacity:
        raise ValueError("route bank exceeds route capacity")
    source_shops = np.asarray(bank["source_shop_sequence"], dtype=np.int16)
    hashes = [
        prefix_hash(bank, route, args.prefix_start, args.switch_step)
        for route in range(route_count)
    ]

    tree = json.loads(args.two_shop_tree.read_text(encoding="utf-8"))
    # The compatible second-shop compiler stores the first-stage map as
    # ``first_route_ids``.  Older hand-authored trees used ``route_ids``.
    # Accept both schemas so the output of the second-stage compiler can be
    # consumed directly without a lossy adapter file.
    first_route_key = "first_route_ids" if "first_route_ids" in tree else "route_ids"
    base_routes = np.asarray(tree[first_route_key], dtype=np.int16)
    if base_routes.shape != (8,):
        raise ValueError("two-shop base map must have shape (8,)")
    if "second_route_ids_by_first_shop" in tree:
        second_by_first = np.asarray(tree["second_route_ids_by_first_shop"], dtype=np.int16)
    else:
        legacy = np.asarray(tree["second_route_ids"], dtype=np.int16)
        second_by_first = np.stack(
            [legacy[int(base_routes[first])] for first in range(8)], axis=0
        )
    if second_by_first.shape != (8, 8):
        raise ValueError("two-shop suffix map must have shape (8, 8)")

    matrices = [np.load(path.with_suffix(".npz"), allow_pickle=False) for path in args.screens]
    lookups: list[dict[int, int]] = []
    common_routes: set[int] | None = None
    for matrix in matrices:
        route_ids = np.asarray(
            matrix["route_ids"]
            if "route_ids" in matrix.files
            else np.arange(matrix["margin"].shape[2]),
            dtype=np.int32,
        )
        if not np.array_equal(bank["source_episode_id"][route_ids], matrix["source_episode_id"]):
            raise RuntimeError("screen source episode field differs from route bank")
        lookup = {int(route): index for index, route in enumerate(route_ids.tolist())}
        lookups.append(lookup)
        current = set(lookup)
        common_routes = current if common_routes is None else common_routes & current
    assert common_routes is not None

    third_by_shop = np.repeat(second_by_first[:, :, None], 8, axis=2).astype(np.int16)
    rows: list[dict[str, object]] = []
    exact_prefix_checks = 0
    for first in range(8):
        for second in range(8):
            current_route = int(second_by_first[first, second])
            if current_route not in common_routes:
                raise RuntimeError(f"current route {current_route} is absent from a screen")
            compatible = np.asarray(
                [route for route in sorted(common_routes) if hashes[route] == hashes[current_route]],
                dtype=np.int32,
            )
            for route in compatible:
                for field in ACTION_FIELDS:
                    if not np.array_equal(
                        bank[field][current_route, args.prefix_start:args.switch_step],
                        bank[field][route, args.prefix_start:args.switch_step],
                    ):
                        raise RuntimeError("prefix hash collision")
                exact_prefix_checks += 1

            for third in range(8):
                source_matched = compatible[source_shops[compatible, 2] == third]
                eligible = np.unique(np.append(source_matched, current_route)).astype(np.int32)
                panel_values: dict[int, list[np.ndarray]] = {int(route): [] for route in eligible}
                event_games = 0
                for matrix, lookup in zip(matrices, lookups, strict=True):
                    observed = np.asarray(matrix["observed_shops"])
                    margin = np.asarray(matrix["margin"])
                    current_column = lookup[current_route]
                    for seat in range(margin.shape[0]):
                        mask = (
                            (observed[seat, :, current_column, 0] == first)
                            & (observed[seat, :, current_column, 1] == second)
                            & (observed[seat, :, current_column, 2] == third)
                        )
                        if not np.any(mask):
                            continue
                        event_games += int(np.sum(mask))
                        for route in eligible:
                            panel_values[int(route)].append(
                                margin[seat, mask, lookup[int(route)]].reshape(-1)
                            )

                baseline_values = panel_values[current_route]
                baseline_win = (
                    float(np.mean(np.concatenate(baseline_values) > 0))
                    if baseline_values else 0.0
                )
                accepted = False
                selected = current_route
                best_win = baseline_win
                if event_games >= args.min_games:
                    scored = []
                    for route in eligible:
                        values = panel_values[int(route)]
                        if values:
                            scored.append((candidate_key(values, args.mode), int(route)))
                    if scored:
                        selected = max(scored)[1]
                        selected_values = np.concatenate(panel_values[selected])
                        best_win = float(np.mean(selected_values > 0))
                        accepted = selected != current_route and best_win >= baseline_win
                if accepted:
                    third_by_shop[first, second, third] = selected
                rows.append({
                    "first_shop": SHOP_NAMES[first],
                    "second_shop": SHOP_NAMES[second],
                    "third_shop": SHOP_NAMES[third],
                    "current_route": current_route,
                    "eligible_routes": eligible.tolist(),
                    "selected_route": selected if accepted else current_route,
                    "games": event_games,
                    "baseline_win_rate": baseline_win,
                    "candidate_win_rate": best_win,
                    "accepted": bool(accepted),
                })

    result = {
        "schema": "kaggriculture.front40_fusion.compatible-third-shop-route-map.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TRAINING_ONLY",
        "trace_bank": str(args.trace_bank),
        "two_shop_tree": str(args.two_shop_tree),
        "source_screens": [str(path) for path in args.screens],
        "mode": args.mode,
        "route_capacity": args.route_capacity,
        "prefix_range": [args.prefix_start, args.switch_step],
        "route_ids": base_routes.tolist(),
        "second_route_ids_by_first_shop": second_by_first.tolist(),
        "third_route_ids_by_shop": third_by_shop.tolist(),
        "routing_index": "first_shop,second_shop,third_shop",
        "exact_prefix_checks": exact_prefix_checks,
        "min_games": args.min_games,
        "routes": rows,
        "boundary": (
            "Third-stage selection reads only the three currently unlocked public shops. "
            "Every changed suffix matches the executed route action prefix exactly."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "accepted": int(sum(bool(row["accepted"]) for row in rows)),
        "unique_routes": sorted(set(third_by_shop.ravel().tolist())),
        "exact_prefix_checks": exact_prefix_checks,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
