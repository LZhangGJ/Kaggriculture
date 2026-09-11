"""Compile a safe second-shop suffix map from Hasegawa route screens.

A suffix candidate is eligible only when its day-4-to-day-6 raw unit and
market actions are byte-identical to the first-shop route.  Therefore the
router never splices a candidate with a different planned prefix.  The map is
trained only on publicly observed first/second shop pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ACTION_FIELDS = (
    "unit_op",
    "unit_item",
    "unit_amount",
    "unit_count",
    "market_op",
    "market_item",
    "market_amount",
    "market_count",
)
SHOP_NAMES = (
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
)


def prefix_hash(bank: np.lib.npyio.NpzFile, route: int, start: int, end: int) -> str:
    digest = hashlib.sha256()
    for field in ACTION_FIELDS:
        digest.update(np.ascontiguousarray(bank[field][route, start:end]).tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--base-map", type=Path, required=True)
    parser.add_argument("--screens", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-capacity", type=int, default=256)
    parser.add_argument("--prefix-start", type=int, default=72)
    parser.add_argument("--switch-step", type=int, default=144)
    parser.add_argument("--min-games", type=int, default=8)
    parser.add_argument("--min-win-gain", type=float, default=0.0)
    parser.add_argument("--seed-limit", type=int, default=0)
    parser.add_argument(
        "--allow-cross-first-shop",
        action="store_true",
        help="Allow a base route whose source Replay had a different first shop.",
    )
    parser.add_argument(
        "--allow-cross-second-shop",
        action="store_true",
        help=(
            "Allow every exact-prefix-compatible suffix to compete under the "
            "currently observed second shop, regardless of its source Replay shop."
        ),
    )
    args = parser.parse_args()

    bank = np.load(args.trace_bank, allow_pickle=False)
    route_count = int(bank["source_reward"].shape[0])
    if route_count > args.route_capacity:
        raise ValueError("route bank exceeds route capacity")
    source_shops = np.asarray(bank["source_shop_sequence"])
    source_episode = np.asarray(bank["source_episode_id"])
    source_reward = np.asarray(bank["source_reward"])
    hashes = [
        prefix_hash(bank, route, args.prefix_start, args.switch_step)
        for route in range(route_count)
    ]

    base_payload = json.loads(args.base_map.read_text(encoding="utf-8"))
    base_routes = [int(x) for x in base_payload["route_ids"]]
    if len(base_routes) != 8:
        raise ValueError("base map must contain eight first-shop routes")
    matrices = [np.load(path.with_suffix(".npz"), allow_pickle=False) for path in args.screens]
    matrix_route_ids: list[np.ndarray] = []
    matrix_lookups: list[dict[int, int]] = []
    for matrix in matrices:
        if "observed_shops" not in matrix.files:
            raise ValueError(f"screen lacks observed_shops: {matrix}")
        route_ids = np.asarray(
            matrix["route_ids"]
            if "route_ids" in matrix.files
            else np.arange(matrix["margin"].shape[2]),
            dtype=np.int32,
        )
        if np.unique(route_ids).size != route_ids.size:
            raise RuntimeError("screen route IDs are not unique")
        if np.any(route_ids < 0) or np.any(route_ids >= route_count):
            raise RuntimeError("screen route IDs exceed trace bank")
        for field in ("source_episode_id", "source_reward"):
            if not np.array_equal(bank[field][route_ids], matrix[field]):
                raise RuntimeError(f"screen source field differs: {field}")
        if not np.array_equal(source_shops[route_ids, 0], matrix["source_first_shop"]):
            raise RuntimeError("screen source field differs: source_first_shop")
        matrix_route_ids.append(route_ids)
        matrix_lookups.append(
            {int(route): index for index, route in enumerate(route_ids.tolist())}
        )
    common_routes = set(matrix_route_ids[0].tolist())
    for route_ids in matrix_route_ids[1:]:
        common_routes.intersection_update(route_ids.tolist())

    # Unused padded routes and unsupported pairs retain the current route.
    second_map = np.repeat(
        np.arange(args.route_capacity, dtype=np.int16)[:, None], 8, axis=1
    )
    second_map_by_first_shop = np.repeat(
        np.asarray(base_routes, dtype=np.int16)[:, None], 8, axis=1
    )
    rows: list[dict[str, object]] = []
    exact_prefix_checks = 0

    for first_shop, base_route in enumerate(base_routes):
        if not args.allow_cross_first_shop and int(source_shops[base_route, 0]) != first_shop:
            raise RuntimeError(
                f"base route {base_route} source first shop does not match {first_shop}"
            )
        compatible = np.asarray(
            [
                route
                for route in range(route_count)
                if route in common_routes
                if (args.allow_cross_first_shop or int(source_shops[route, 0]) == first_shop)
                and hashes[route] == hashes[base_route]
            ],
            dtype=np.int32,
        )
        if base_route not in common_routes:
            raise RuntimeError(
                f"base route {base_route} is absent from at least one development screen"
            )
        for route in compatible:
            for field in ACTION_FIELDS:
                if not np.array_equal(
                    bank[field][base_route, args.prefix_start : args.switch_step],
                    bank[field][route, args.prefix_start : args.switch_step],
                ):
                    raise RuntimeError("prefix hash collision")
            exact_prefix_checks += 1

        for second_shop in range(8):
            eligible = (
                compatible
                if args.allow_cross_second_shop
                else compatible[source_shops[compatible, 1] == second_shop]
            )
            candidate_margin: list[np.ndarray] = []
            candidate_invalid: list[np.ndarray] = []
            baseline_margin: list[np.ndarray] = []
            for matrix, lookup in zip(matrices, matrix_lookups, strict=True):
                observed = np.asarray(matrix["observed_shops"])
                margin = np.asarray(matrix["margin"])
                invalid = np.asarray(matrix["invalid"])
                base_column = lookup[base_route]
                eligible_columns = np.asarray(
                    [lookup[int(route)] for route in eligible], dtype=np.int32
                )
                for seat in (0, 1):
                    seed_mask = np.ones(observed.shape[1], dtype=bool)
                    if args.seed_limit > 0:
                        seed_mask[np.arange(observed.shape[1]) >= args.seed_limit] = False
                    event_mask = (
                        seed_mask
                        &
                        (observed[seat, :, base_column, 0] == first_shop)
                        & (observed[seat, :, base_column, 1] == second_shop)
                    )
                    if not np.any(event_mask):
                        continue
                    baseline_margin.append(margin[seat, event_mask, base_column])
                    if eligible.size:
                        candidate_margin.append(
                            margin[seat, event_mask][:, eligible_columns]
                        )
                        candidate_invalid.append(
                            invalid[seat, event_mask][:, eligible_columns]
                        )

            if not baseline_margin:
                rows.append(
                    {
                        "first_shop": SHOP_NAMES[first_shop],
                        "second_shop": SHOP_NAMES[second_shop],
                        "base_route": base_route,
                        "selected_route": base_route,
                        "reason": "no dev events",
                    }
                )
                continue
            baseline = np.concatenate(baseline_margin)
            baseline_win = float(np.mean(baseline > 0))
            if eligible.size == 0 or not candidate_margin:
                rows.append(
                    {
                        "first_shop": SHOP_NAMES[first_shop],
                        "second_shop": SHOP_NAMES[second_shop],
                        "base_route": base_route,
                        "selected_route": base_route,
                        "games": int(baseline.size),
                        "baseline_win_rate": baseline_win,
                        "reason": "no exact-prefix source suffix",
                    }
                )
                continue

            margins = np.concatenate(candidate_margin, axis=0)
            invalids = np.concatenate(candidate_invalid, axis=0)
            games = int(margins.shape[0])
            wins = np.mean(margins > 0, axis=0)
            means = np.mean(margins, axis=0)
            invalid_mean = np.mean(invalids, axis=0)
            order = np.lexsort(
                (source_reward[eligible], -invalid_mean, means, wins)
            )
            best_local = int(order[-1])
            best_route = int(eligible[best_local])
            gain = float(wins[best_local] - baseline_win)
            accepted = games >= args.min_games and gain >= args.min_win_gain
            if accepted:
                second_map[base_route, second_shop] = best_route
                second_map_by_first_shop[first_shop, second_shop] = best_route
            rows.append(
                {
                    "first_shop": SHOP_NAMES[first_shop],
                    "second_shop": SHOP_NAMES[second_shop],
                    "base_route": base_route,
                    "eligible_routes": [int(x) for x in eligible],
                    "selected_route": best_route if accepted else base_route,
                    "best_candidate_route": best_route,
                    "best_source_episode_id": int(source_episode[best_route]),
                    "games": games,
                    "baseline_win_rate": baseline_win,
                    "candidate_win_rate": float(wins[best_local]),
                    "training_win_gain": gain,
                    "candidate_mean_margin": float(means[best_local]),
                    "candidate_invalid_mean": float(invalid_mean[best_local]),
                    "accepted": bool(accepted),
                    "reason": "accepted" if accepted else "insufficient evidence",
                }
            )

    payload = {
        "schema": "kaggriculture.front40_fusion.compatible-second-shop-map.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace_bank": str(args.trace_bank),
        "base_map": str(args.base_map),
        "source_screens": [str(path) for path in args.screens],
        "common_screen_route_count": len(common_routes),
        "prefix_range": [args.prefix_start, args.switch_step],
        "route_capacity": args.route_capacity,
        "first_route_ids": base_routes,
        "second_route_ids": second_map.tolist(),
        "second_route_ids_by_first_shop": second_map_by_first_shop.tolist(),
        "routing_index": "first_shop,second_shop",
        "legacy_second_route_ids_note": (
            "second_route_ids is retained for old readers and is ambiguous when "
            "multiple first shops share one base route; new readers must use "
            "second_route_ids_by_first_shop"
        ),
        "min_games": args.min_games,
        "min_win_gain": args.min_win_gain,
        "seed_limit": args.seed_limit,
        "allow_cross_first_shop": args.allow_cross_first_shop,
        "allow_cross_second_shop": args.allow_cross_second_shop,
        "exact_prefix_checks": exact_prefix_checks,
        "routes": rows,
        "status": "TRAINING_ONLY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    accepted_count = sum(bool(row.get("accepted")) for row in rows)
    print(
        json.dumps(
            {
                "status": "PASS",
                "accepted_pairs": accepted_count,
                "exact_prefix_checks": exact_prefix_checks,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
