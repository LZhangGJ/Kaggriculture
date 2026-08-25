"""Keep only explicitly named, validated pairs from a second-shop map."""

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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--keep",
        required=True,
        help="Comma-separated FIRST:SECOND shop-name pairs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    capacity = int(payload["route_capacity"])
    keep = {
        tuple(value.split(":", 1))
        for value in args.keep.split(",")
        if value
    }
    unknown = [
        pair for pair in keep if pair[0] not in SHOP_NAMES or pair[1] not in SHOP_NAMES
    ]
    if unknown:
        raise ValueError(f"unknown shop pair: {unknown}")

    base_routes = np.asarray(payload["first_route_ids"], dtype=np.int16)
    second = np.repeat(np.arange(capacity, dtype=np.int16)[:, None], 8, axis=1)
    second_by_first = np.repeat(base_routes[:, None], 8, axis=1)
    accepted_pairs: list[str] = []
    for row in payload["routes"]:
        pair = (row["first_shop"], row["second_shop"])
        keep_row = bool(row.get("accepted")) and pair in keep
        row["accepted_before_prune"] = bool(row.get("accepted"))
        row["accepted"] = keep_row
        if keep_row:
            base = int(row["base_route"])
            selected = int(row["selected_route"])
            second[base, SHOP_NAMES.index(row["second_shop"])] = selected
            second_by_first[
                SHOP_NAMES.index(row["first_shop"]),
                SHOP_NAMES.index(row["second_shop"]),
            ] = selected
            accepted_pairs.append(f"{pair[0]}:{pair[1]}")
        elif row.get("accepted_before_prune"):
            row["selected_route"] = int(row["base_route"])
            row["reason"] = "removed by independent pair-stability prune"

    missing = sorted(f"{a}:{b}" for a, b in keep if f"{a}:{b}" not in accepted_pairs)
    if missing:
        raise RuntimeError(f"requested pair was not accepted in source map: {missing}")
    payload["schema"] = "kaggriculture.front40_fusion.pruned-compatible-second-shop-map.v1"
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["source_map"] = str(args.source)
    payload["second_route_ids"] = second.tolist()
    payload["second_route_ids_by_first_shop"] = second_by_first.tolist()
    payload["routing_index"] = "first_shop,second_shop"
    payload["kept_pairs"] = sorted(accepted_pairs)
    payload["prune_rule"] = "keep only pairs non-negative on every independent validation panel"
    payload["status"] = "TRAINING_ONLY"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "kept_pairs": sorted(accepted_pairs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
