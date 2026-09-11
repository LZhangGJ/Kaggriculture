"""Build a public seat-aware first-shop route map from dev screens."""

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
    parser.add_argument("--screens", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("pooled", "stable", "minimax"), default="stable"
    )
    args = parser.parse_args()

    matrices = [np.load(path.with_suffix(".npz"), allow_pickle=False) for path in args.screens]
    reference = matrices[0]
    for matrix in matrices[1:]:
        for field in ("source_episode_id", "source_reward", "source_first_shop"):
            if not np.array_equal(reference[field], matrix[field]):
                raise RuntimeError(f"source field differs across screens: {field}")

    source_episode_id = np.asarray(reference["source_episode_id"])
    source_reward = np.asarray(reference["source_reward"])
    source_first_shop = np.asarray(reference["source_first_shop"])
    route_ids_by_seat: list[list[int]] = [[], []]
    rows: list[dict[str, object]] = []

    for seat in (0, 1):
        for shop_id, shop_name in enumerate(SHOP_NAMES):
            eligible = np.flatnonzero(source_first_shop == shop_id)
            if eligible.size == 0:
                raise RuntimeError(f"no source routes for {shop_name}")
            panel_win: list[np.ndarray] = []
            panel_margin: list[np.ndarray] = []
            panel_invalid: list[np.ndarray] = []
            panel_games: list[int] = []
            for matrix in matrices:
                event_mask = np.asarray(matrix["first_shop"])[seat] == shop_id
                if not np.any(event_mask):
                    continue
                margin = np.asarray(matrix["margin"])[seat, event_mask][:, eligible]
                invalid = np.asarray(matrix["invalid"])[seat, event_mask][:, eligible]
                panel_games.append(int(margin.shape[0]))
                panel_win.append(np.mean(margin > 0, axis=0))
                panel_margin.append(np.mean(margin, axis=0))
                panel_invalid.append(np.mean(invalid, axis=0))
            if not panel_games:
                raise RuntimeError(f"no dev events for seat {seat} {shop_name}")

            win_matrix = np.stack(panel_win)
            margin_matrix = np.stack(panel_margin)
            invalid_matrix = np.stack(panel_invalid)
            weights = np.asarray(panel_games, dtype=np.float64)
            pooled_win = np.average(win_matrix, axis=0, weights=weights)
            pooled_margin = np.average(margin_matrix, axis=0, weights=weights)
            pooled_invalid = np.average(invalid_matrix, axis=0, weights=weights)
            worst_win = np.min(win_matrix, axis=0)
            win_std = np.std(win_matrix, axis=0)

            if args.mode == "pooled":
                primary = pooled_win
                secondary = worst_win
            elif args.mode == "stable":
                primary = pooled_win - 0.5 * win_std
                secondary = worst_win
            else:
                primary = worst_win
                secondary = pooled_win
            order = np.lexsort(
                (
                    source_reward[eligible],
                    -pooled_invalid,
                    pooled_margin,
                    secondary,
                    primary,
                )
            )
            local = int(order[-1])
            route_id = int(eligible[local])
            route_ids_by_seat[seat].append(route_id)
            rows.append(
                {
                    "seat": seat,
                    "shop_id": shop_id,
                    "shop": shop_name,
                    "route_id": route_id,
                    "source_episode_id": int(source_episode_id[route_id]),
                    "source_reward": int(source_reward[route_id]),
                    "training_games": int(sum(panel_games)),
                    "pooled_win_rate": float(pooled_win[local]),
                    "worst_panel_win_rate": float(worst_win[local]),
                    "panel_win_rates": [float(x) for x in win_matrix[:, local]],
                    "pooled_mean_margin": float(pooled_margin[local]),
                    "pooled_invalid_mean": float(pooled_invalid[local]),
                }
            )

    payload = {
        "schema": "kaggriculture.front40_fusion.seat-aware-shop-route-map.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_screens": [str(path) for path in args.screens],
        "shop_order": list(SHOP_NAMES),
        "route_ids": route_ids_by_seat,
        "routes": rows,
        "selection_mode": args.mode,
        "selection_rule": "public seat and first shop only; same-source-shop routes",
        "status": "TRAINING_ONLY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": "PASS", "mode": args.mode, "route_ids": route_ids_by_seat}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
