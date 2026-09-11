"""Build a compact Crop-Dusta spine with Hasegawa weak-shop branches."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-bank", type=Path, required=True)
    parser.add_argument("--crop-map", type=Path, required=True)
    parser.add_argument("--hasegawa-bank", type=Path, required=True)
    parser.add_argument("--hasegawa-map", type=Path, required=True)
    parser.add_argument("--hasegawa-second-map", type=Path, required=True)
    parser.add_argument("--weak-shop-ids", default="1,3,6")
    parser.add_argument("--route-capacity", type=int, default=256)
    parser.add_argument("--output-bank", type=Path, required=True)
    parser.add_argument("--output-map", type=Path, required=True)
    parser.add_argument("--output-second-map", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    crop = np.load(args.crop_bank)
    hasegawa = np.load(args.hasegawa_bank)
    crop_map = np.asarray(json.loads(args.crop_map.read_text(encoding="utf-8"))["route_ids"], dtype=np.int32)
    hasegawa_map = np.asarray(
        json.loads(args.hasegawa_map.read_text(encoding="utf-8"))["route_ids"], dtype=np.int32
    )
    hasegawa_second = np.asarray(
        json.loads(args.hasegawa_second_map.read_text(encoding="utf-8"))["second_route_ids"],
        dtype=np.int32,
    )
    weak = sorted({int(value) for value in args.weak_shop_ids.split(",") if value})
    required_hasegawa = {int(hasegawa["bootstrap_route_id"])}
    required_hasegawa.update(int(hasegawa_map[shop]) for shop in weak)
    for shop in weak:
        base = int(hasegawa_map[shop])
        required_hasegawa.update(int(route) for route in hasegawa_second[base])
    required = np.asarray(sorted(required_hasegawa), dtype=np.int32)
    crop_routes = int(crop["source_reward"].shape[0])
    if crop_routes + required.size > args.route_capacity:
        raise ValueError("hybrid route set exceeds capacity")
    h_to_local = {
        int(route): crop_routes + index for index, route in enumerate(required)
    }

    output = {}
    for field in crop.files:
        if field == "bootstrap_route_id":
            output[field] = np.asarray(crop[field], dtype=np.int16)
        else:
            output[field] = np.concatenate((np.asarray(crop[field]), np.asarray(hasegawa[field])[required]), axis=0)
    hybrid_map = crop_map.copy()
    for shop in weak:
        hybrid_map[shop] = h_to_local[int(hasegawa_map[shop])]
    second_map = np.repeat(
        np.arange(args.route_capacity, dtype=np.int16)[:, None], 8, axis=1
    )
    for shop in weak:
        base_h = int(hasegawa_map[shop])
        base_local = h_to_local[base_h]
        for second_shop in range(8):
            target_h = int(hasegawa_second[base_h, second_shop])
            if target_h in h_to_local:
                second_map[base_local, second_shop] = h_to_local[target_h]

    args.output_bank.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_bank, **output)
    args.output_map.write_text(
        json.dumps(
            {"schema": "kaggriculture.front40_fusion.crop-hasegawa-hybrid-map.v1",
             "route_ids": hybrid_map.tolist(), "weak_shop_ids": weak,
             "status": "EXPERIMENTAL"},
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    args.output_second_map.write_text(
        json.dumps(
            {"schema": "kaggriculture.front40_fusion.crop-hasegawa-hybrid-second-map.v1",
             "second_route_ids": second_map.tolist(), "status": "EXPERIMENTAL"},
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "kaggriculture.front40_fusion.crop-hasegawa-hybrid-bank.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "crop_routes": crop_routes,
        "hasegawa_routes": required.tolist(),
        "hasegawa_local_route_ids": h_to_local,
        "weak_shop_ids": weak,
        "hybrid_route_ids": hybrid_map.tolist(),
        "output_routes": int(output["source_reward"].shape[0]),
        "output_bank": str(args.output_bank),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
