from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Force one revealed first-shop branch to a single trace route."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--first-shop", required=True, type=int, choices=range(8))
    parser.add_argument("--route-id", required=True, type=int)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    out = deepcopy(payload)
    shop = args.first_shop
    route = args.route_id

    out["schema"] = "kaggriculture.front40_fusion.first-shop-blended-third-shop-tree.v1"
    out["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["status"] = "TRAINING_ONLY"
    out["parent_tree"] = str(args.input)
    out["override"] = {
        "first_shop": shop,
        "route_id": route,
        "reason": "Robust fixed-route override selected only after the first shop is public.",
    }
    out["route_ids"][shop] = route
    out["second_route_ids_by_first_shop"][shop] = [route] * 8
    out["third_route_ids_by_shop"][shop] = [[route] * 8 for _ in range(8)]

    if "routes" in out:
        row = out["routes"][shop]
        row["base_route"] = route
        row["second_routes"] = [route] * 8
        row["third_routes"] = [[route] * 8 for _ in range(8)]
        row["override"] = "fixed after public first-shop observation"

    out["boundary"] = (
        "Every choice is indexed only by town shops already revealed in the current game. "
        "No identity, Replay ID, seed, or future event is used."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
