#!/usr/bin/env python3
"""Generate the compact, Python-free asset blob used by the native 2965 port.

The authoritative source is imported only here.  Rollout code reads the
resulting binary directly and never starts Python or parses the 7k-line agent.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "work/new_public_opponents/the-2965-master-hybrid-engine/output/main.py"

OPS = [
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP", "PICKUP",
    "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER",
    "CARE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT",
    "BUY_ANIMAL", "SELL",
]
ITEMS = [
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG",
    "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP",
]
SHOPS = [
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
    "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE",
]


def load_source():
    spec = importlib.util.spec_from_file_location("metav4_2965_asset_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    # The final wrapper installs this lazily on the first step-0 callback.  Do
    # it explicitly so the native tape contains the exact final opening.
    module._alt_install(module._ALT_MODE)
    return module


def pack_action(raw) -> bytes:
    raw = list(raw or ["PASS"])
    op = OPS.index(raw[0]) if raw and raw[0] in OPS else 0
    item = ITEMS.index(raw[1]) if len(raw) > 1 and raw[1] in ITEMS else -1
    quantity = int(raw[2]) if len(raw) > 2 else 1
    return struct.pack("<bbi", op, item, quantity)


def pack_player_action(raw: dict) -> bytes:
    units = [raw.get("farmer") or ["PASS"], *(raw.get("hands") or [])]
    market = raw.get("market") or []
    if len(units) > 255 or len(market) > 255:
        raise ValueError("action count exceeds binary schema")
    return (struct.pack("<BB", len(units), len(market)) +
            b"".join(pack_action(value) for value in [*units, *market]))


def route_table(rows: dict, default: int = -1) -> list[int]:
    result = [default] * 64
    for key, route in rows.items():
        left, right = (SHOPS.index(value) for value in key)
        result[left * 8 + right] = int(route)
    return result


def build_blob(module, source_sha: bytes) -> tuple[bytes, dict]:
    routes = module._IMPL.chassis.routes
    route_ids = sorted(int(route) for route in routes)
    if len(route_ids) != 41 or any(len(routes[route]) != 719 for route in route_ids):
        raise ValueError("unexpected canonical route inventory")

    chunks = [b"MV42965\0", struct.pack("<I", 1), source_sha]
    chunks.append(struct.pack("<H", len(route_ids)))
    for route_id in route_ids:
        tape = routes[route_id]
        chunks.append(struct.pack("<hH", route_id, len(tape)))
        chunks.extend(pack_player_action(frame) for frame in tape)

    nonempty_pairs = 0
    for left in SHOPS:
        for right in SHOPS:
            streams = module._v92_p_pair((left, right))
            chunks.append(struct.pack("<H", len(streams)))
            if streams:
                nonempty_pairs += 1
            for _, events in streams:
                ordered = sorted((int(step), int(item), int(quantity))
                                 for (step, item), quantity in events.items())
                chunks.append(struct.pack("<H", len(ordered)))
                chunks.extend(struct.pack("<hbh", *event) for event in ordered)

    tables = {
        "new": route_table(module._R108_SHOP_ROUTES),
        "old": route_table(module._R110_OLD_SHOPS),
        "v92": route_table(module._V92_TABLE),
    }
    for values in tables.values():
        chunks.append(struct.pack("<64h", *values))

    overrides = sorted(module._V93_ROUTE_BY_RIVAL.items())
    chunks.append(struct.pack("<H", len(overrides)))
    for (money, wheat), route in overrides:
        chunks.append(struct.pack("<qih", round(float(money) * 1000),
                                  int(wheat), int(route)))

    blob = b"".join(chunks)
    manifest = {
        "schema": "metav4-2965-native-assets-v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_sha.hex(),
        "asset_sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
        "routes": route_ids,
        "frames_per_route": 719,
        "predictor_pairs": 64,
        "nonempty_predictor_pairs": nonempty_pairs,
        "rival_overrides": len(overrides),
    }
    return blob, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "metav4_2965.assets.bin")
    parser.add_argument("--manifest", type=Path, default=HERE / "assets.manifest.json")
    args = parser.parse_args()

    source_bytes = SOURCE.read_bytes()
    blob, manifest = build_blob(load_source(), hashlib.sha256(source_bytes).digest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(blob)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
