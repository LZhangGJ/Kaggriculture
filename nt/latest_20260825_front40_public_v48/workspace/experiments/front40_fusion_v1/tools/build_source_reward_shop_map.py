"""Choose a same-first-shop source-reward baseline from a generic trace bank."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SHOP_NAMES = (
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
    "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.bank, allow_pickle=False) as data:
        shops = np.asarray(data["source_shop_sequence"])
        rewards = np.asarray(data["source_reward"])
        episodes = np.asarray(data["source_episode_id"])
        bootstrap = int(np.asarray(data["bootstrap_route_id"]))
    route_ids: list[int] = []
    rows: list[dict[str, object]] = []
    for shop_id, shop_name in enumerate(SHOP_NAMES):
        eligible = np.flatnonzero(shops[:, 0] == shop_id)
        if eligible.size:
            selected = int(eligible[np.argmax(rewards[eligible])])
            rule = "same-first-shop highest source reward"
        else:
            selected = bootstrap
            rule = "global highest source reward fallback; no same-shop Replay"
        route_ids.append(selected)
        rows.append(
            {
                "shop_id": shop_id,
                "shop": shop_name,
                "route_id": selected,
                "source_episode_id": int(episodes[selected]),
                "source_reward": int(rewards[selected]),
                "same_shop_source_count": int(eligible.size),
                "selection": rule,
            }
        )
    payload = {
        "schema": "kaggriculture.front40_fusion.source_reward_shop_map.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_bank": str(args.bank),
        "shop_order": list(SHOP_NAMES),
        "route_ids": route_ids,
        "routes": rows,
        "status": "BASELINE_ONLY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "route_ids": route_ids}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
