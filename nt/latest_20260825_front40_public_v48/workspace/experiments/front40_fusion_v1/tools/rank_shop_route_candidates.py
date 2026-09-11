"""Rank screened Replay routes for selected observed first shops.

This is an analysis-only tool.  It never reads future events when selecting a
route; it groups already completed counterfactual outcomes by the first public
shop and produces a development shortlist for a later independent screen.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--shops", default="0,4,5")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix = np.load(args.screen.with_suffix(".npz"), allow_pickle=False)
    first_shop = np.asarray(matrix["first_shop"])
    margin = np.asarray(matrix["margin"])
    invalid = np.asarray(matrix["invalid"])
    hard = np.asarray(matrix["hard"])
    source_episode = np.asarray(matrix["source_episode_id"])
    source_reward = np.asarray(matrix["source_reward"])
    source_first_shop = np.asarray(matrix["source_first_shop"])
    route_ids = np.asarray(
        matrix["route_ids"] if "route_ids" in matrix else np.arange(source_reward.size),
        dtype=np.int32,
    )
    requested_shops = [int(value) for value in args.shops.split(",") if value]

    results: list[dict[str, object]] = []
    candidate_union: set[int] = set()
    for shop_id in requested_shops:
        if shop_id < 0 or shop_id >= len(SHOP_NAMES):
            raise ValueError(f"invalid shop ID: {shop_id}")
        event_mask = first_shop[0] == shop_id
        if not np.any(event_mask):
            raise RuntimeError(f"screen contains no events for {SHOP_NAMES[shop_id]}")
        group_margin = margin[:, event_mask, :].reshape(-1, route_ids.size)
        group_invalid = invalid[:, event_mask, :].reshape(-1, route_ids.size)
        group_hard = hard[:, event_mask, :].reshape(-1, route_ids.size)
        wins = np.mean(group_margin > 0, axis=0)
        means = np.mean(group_margin, axis=0)
        invalid_means = np.mean(group_invalid, axis=0)
        hard_totals = np.sum(group_hard, axis=0)
        # Win rate is primary.  Margin and fewer repaired intents break ties.
        order = np.lexsort(
            (
                source_reward,
                -invalid_means,
                means,
                wins,
                -hard_totals,
            )
        )[::-1]
        rows: list[dict[str, object]] = []
        for index in order[: args.top_k]:
            route_id = int(route_ids[index])
            candidate_union.add(route_id)
            rows.append(
                {
                    "route_id": route_id,
                    "source_episode_id": int(source_episode[index]),
                    "source_reward": int(source_reward[index]),
                    "source_first_shop_id": int(source_first_shop[index]),
                    "source_first_shop": SHOP_NAMES[int(source_first_shop[index])],
                    "games": int(group_margin.shape[0]),
                    "wins": int(np.sum(group_margin[:, index] > 0)),
                    "win_rate": float(wins[index]),
                    "mean_margin": float(means[index]),
                    "invalid_mean": float(invalid_means[index]),
                    "hard_total": int(hard_totals[index]),
                }
            )
        results.append(
            {
                "shop_id": shop_id,
                "shop": SHOP_NAMES[shop_id],
                "seed_count": int(np.sum(event_mask)),
                "games_per_route": int(group_margin.shape[0]),
                "top_routes": rows,
            }
        )

    payload = {
        "schema": "kaggriculture.front40_fusion.shop-route-candidate-ranking.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_screen": str(args.screen),
        "shops": requested_shops,
        "top_k": args.top_k,
        "candidate_route_ids": sorted(candidate_union),
        "candidate_count": len(candidate_union),
        "ranking_rule": "hard_total ascending, win_rate descending, margin descending, invalid ascending",
        "results": results,
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "candidate_count": len(candidate_union),
                "candidate_route_ids": sorted(candidate_union),
                "leaders": {
                    row["shop"]: row["top_routes"][0] for row in results
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
