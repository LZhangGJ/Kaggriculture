"""Measure each accepted second-shop switch against its base route."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screens", type=Path, nargs="+", required=True)
    parser.add_argument("--base-map", type=Path, required=True)
    parser.add_argument("--second-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_routes = np.asarray(
        json.loads(args.base_map.read_text(encoding="utf-8"))["route_ids"],
        dtype=np.int32,
    )
    second_payload = json.loads(args.second_map.read_text(encoding="utf-8"))
    by_first = "second_route_ids_by_first_shop" in second_payload
    second_routes = np.asarray(
        second_payload[
            "second_route_ids_by_first_shop" if by_first else "second_route_ids"
        ],
        dtype=np.int32,
    )
    accepted = [row for row in second_payload["routes"] if row.get("accepted")]
    rows: list[dict[str, object]] = []

    for source in accepted:
        first_shop = int(next(i for i, name in enumerate(second_payload.get("shop_order", [])) if name == source["first_shop"])) if second_payload.get("shop_order") else None
        if first_shop is None:
            # The compatible-map schema has names but no explicit shop_order.
            names = (
                "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
                "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE",
            )
            first_shop = names.index(source["first_shop"])
            second_shop = names.index(source["second_shop"])
        else:
            second_shop = second_payload["shop_order"].index(source["second_shop"])
        base_route = int(base_routes[first_shop])
        selected_route = int(
            second_routes[first_shop, second_shop]
            if by_first
            else second_routes[base_route, second_shop]
        )
        panels: list[dict[str, object]] = []
        pooled_base: list[np.ndarray] = []
        pooled_selected: list[np.ndarray] = []
        pooled_hard: list[np.ndarray] = []

        for screen_path in args.screens:
            matrix = np.load(screen_path.with_suffix(".npz"), allow_pickle=False)
            route_ids = np.asarray(matrix["route_ids"], dtype=np.int32)
            lookup = {int(route): index for index, route in enumerate(route_ids)}
            if base_route not in lookup or selected_route not in lookup:
                raise ValueError(
                    f"screen lacks pair routes {base_route}/{selected_route}: {screen_path}"
                )
            base_column = lookup[base_route]
            selected_column = lookup[selected_route]
            observed = np.asarray(matrix["observed_shops"])
            margin = np.asarray(matrix["margin"])
            hard = np.asarray(matrix["hard"])
            panel_base: list[np.ndarray] = []
            panel_selected: list[np.ndarray] = []
            panel_hard: list[np.ndarray] = []
            for seat in (0, 1):
                mask = (
                    (observed[seat, :, base_column, 0] == first_shop)
                    & (observed[seat, :, base_column, 1] == second_shop)
                )
                panel_base.append(margin[seat, mask, base_column])
                panel_selected.append(margin[seat, mask, selected_column])
                panel_hard.append(hard[seat, mask, selected_column])
            base_values = np.concatenate(panel_base)
            selected_values = np.concatenate(panel_selected)
            hard_values = np.concatenate(panel_hard)
            pooled_base.append(base_values)
            pooled_selected.append(selected_values)
            pooled_hard.append(hard_values)
            panels.append(
                {
                    "screen": str(screen_path),
                    "games": int(base_values.size),
                    "base_wins": int(np.sum(base_values > 0)),
                    "selected_wins": int(np.sum(selected_values > 0)),
                    "net_wins": int(np.sum(selected_values > 0) - np.sum(base_values > 0)),
                    "base_win_rate": float(np.mean(base_values > 0)) if base_values.size else 0.0,
                    "selected_win_rate": float(np.mean(selected_values > 0)) if selected_values.size else 0.0,
                    "mean_margin_gain": float(np.mean(selected_values - base_values)) if base_values.size else 0.0,
                    "hard_counter_total": int(np.sum(hard_values)),
                }
            )
        base_values = np.concatenate(pooled_base)
        selected_values = np.concatenate(pooled_selected)
        hard_values = np.concatenate(pooled_hard)
        rows.append(
            {
                "first_shop": source["first_shop"],
                "second_shop": source["second_shop"],
                "base_route": base_route,
                "selected_route": selected_route,
                "games": int(base_values.size),
                "base_wins": int(np.sum(base_values > 0)),
                "selected_wins": int(np.sum(selected_values > 0)),
                "net_wins": int(np.sum(selected_values > 0) - np.sum(base_values > 0)),
                "mean_margin_gain": float(np.mean(selected_values - base_values)),
                "hard_counter_total": int(np.sum(hard_values)),
                "panel_net_wins": [int(panel["net_wins"]) for panel in panels],
                "panels": panels,
            }
        )

    payload = {
        "schema": "kaggriculture.front40_fusion.second-shop-pair-effects.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "screens": [str(path) for path in args.screens],
        "base_map": str(args.base_map),
        "second_map": str(args.second_map),
        "routing_index": (
            "first_shop,second_shop" if by_first else "legacy_base_route,second_shop"
        ),
        "pairs": rows,
        "status": "PASS" if sum(row["hard_counter_total"] for row in rows) == 0 else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "pairs": [
                    {
                        "first": row["first_shop"],
                        "second": row["second_shop"],
                        "net_wins": row["net_wins"],
                        "panel_net_wins": row["panel_net_wins"],
                    }
                    for row in rows
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
