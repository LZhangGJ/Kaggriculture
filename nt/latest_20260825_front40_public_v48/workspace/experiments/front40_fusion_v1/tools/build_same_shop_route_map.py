"""Build a deployable first-shop map from a Hasegawa route screen.

Only source traces whose own first visible shop matches the runtime shop are
eligible.  This prevents a small training sample from selecting an unrelated
future program solely because it happened to beat one frozen opponent.
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--unrestricted",
        action="store_true",
        help="allow every source route to compete for each observed first shop",
    )
    args = parser.parse_args()

    matrix_path = args.screen.with_suffix(".npz")
    with np.load(matrix_path, allow_pickle=False) as data:
        margin = np.asarray(data["margin"])
        invalid = np.asarray(data["invalid"])
        first_shop = np.asarray(data["first_shop"])
        source_first_shop = np.asarray(data["source_first_shop"])
        source_episode_id = np.asarray(data["source_episode_id"])
        source_reward = np.asarray(data["source_reward"])

    route_ids: list[int] = []
    rows: list[dict[str, object]] = []
    for shop_id, shop_name in enumerate(SHOP_NAMES):
        eligible = np.arange(source_first_shop.shape[0]) if args.unrestricted else np.flatnonzero(source_first_shop == shop_id)
        if eligible.size == 0:
            raise RuntimeError(f"no source routes for {shop_name}")
        event_mask = first_shop[0] == shop_id
        if np.any(event_mask):
            group_margin = margin[:, event_mask, :].reshape(-1, margin.shape[-1])
            group_invalid = invalid[:, event_mask, :].reshape(-1, invalid.shape[-1])
            win_rate = np.mean(group_margin[:, eligible] > 0, axis=0)
            mean_margin = np.mean(group_margin[:, eligible], axis=0)
            invalid_mean = np.mean(group_invalid[:, eligible], axis=0)
            # Prefer wins and cash margin, but use fewer recovery interventions
            # and source reward as stable tie breakers.
            order = np.lexsort(
                (source_reward[eligible], -invalid_mean, mean_margin, win_rate)
            )[::-1]
            selected = int(eligible[order[0]])
            sample_count = int(np.sum(event_mask) * 2)
            selected_win_rate = float(win_rate[order[0]])
            selected_margin = float(mean_margin[order[0]])
            selected_invalid = float(invalid_mean[order[0]])
            selection = "same-shop train result"
        else:
            selected = int(eligible[np.argmax(source_reward[eligible])])
            sample_count = 0
            selected_win_rate = None
            selected_margin = None
            selected_invalid = None
            selection = "same-shop highest-source-reward fallback"
        route_ids.append(selected)
        rows.append(
            {
                "shop_id": shop_id,
                "shop": shop_name,
                "route_id": selected,
                "source_episode_id": int(source_episode_id[selected]),
                "source_reward": int(source_reward[selected]),
                "training_games": sample_count,
                "training_win_rate": selected_win_rate,
                "training_mean_margin": selected_margin,
                "training_invalid_mean": selected_invalid,
                "selection": (
                    "unrestricted " + selection if args.unrestricted else selection
                ),
            }
        )

    payload = {
        "schema": "kaggriculture.front40_fusion.hasegawa_same_shop_route_map.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_screen": str(args.screen),
        "source_matrix": str(matrix_path),
        "shop_order": list(SHOP_NAMES),
        "route_ids": route_ids,
        "routes": rows,
        "selection_rule": (
            ("all source routes eligible" if args.unrestricted else "same source first-shop required")
            + "; maximize training win rate, then mean margin, lower invalid recovery, and source reward"
        ),
        "status": "TRAINING_ONLY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "route_ids": route_ids}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
