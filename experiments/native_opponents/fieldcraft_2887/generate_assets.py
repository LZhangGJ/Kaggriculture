#!/usr/bin/env python3
"""Pack frozen Fieldcraft routed tape data for a Python-free C++ runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_DIR = (ROOT / "work/new_public_opponents/"
              "kaggriculture-2887-score-fieldcraft-agent/output/extracted")
SOURCE = SOURCE_DIR / "main.py"
MIRROR = SOURCE_DIR / "mirror_plan.py"
RELEASE_MANIFEST = SOURCE_DIR.parent / "release_manifest.json"

EXPECTED_MAIN = "7fab02c889fde933e642ee9ceb35dd56ae9a0a073fffc0f6d7a689128367651d"
EXPECTED_MIRROR = "20d79c8435814979ea6924e09026f3f83c5c646a331e2ce3bd940de426c1dbc4"
EXPECTED_ARCHIVE = "99d277ac82fa43e29adb56db951a914ef20fb39758d245a9410ef563f147e8b8"

OPS = (
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP", "PICKUP",
    "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER",
    "CARE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT",
    "BUY_ANIMAL", "SELL",
)
ITEMS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG",
    "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP",
)
SHOPS = (
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
    "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source():
    if sha(SOURCE) != EXPECTED_MAIN or sha(MIRROR) != EXPECTED_MIRROR:
        raise RuntimeError("Fieldcraft frozen source hash drift")
    old_path = list(sys.path)
    old_mirror = sys.modules.pop("mirror_plan", None)
    try:
        sys.path.insert(0, str(SOURCE_DIR))
        spec = importlib.util.spec_from_file_location(
            "fieldcraft_2887_asset_source", SOURCE)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        if old_mirror is not None:
            sys.modules["mirror_plan"] = old_mirror


def pack_action(raw) -> bytes:
    raw = list(raw or ["PASS"])
    op = OPS.index(raw[0]) if raw and raw[0] in OPS else 0
    item = ITEMS.index(raw[1]) if len(raw) > 1 and raw[1] in ITEMS else -1
    quantity = int(raw[2]) if len(raw) > 2 else 1
    return struct.pack("<bbi", op, item, quantity)


def pack_player_action(raw: dict) -> bytes:
    units = [raw.get("farmer") or ["PASS"], *(raw.get("hands") or [])]
    # The frozen tape uses null rows as sparse placeholders.  Python's plan()
    # drops them before any downstream row-count/capacity checks, so they must
    # not become concrete PASS rows in the native asset.
    market = [value for value in (raw.get("market") or []) if value]
    if not 1 <= len(units) <= 255 or len(market) > 10:
        raise ValueError("Fieldcraft action exceeds frozen ABI bounds")
    return (struct.pack("<BB", len(units), len(market)) +
            b"".join(pack_action(value) for value in (*units, *market)))


def routed_tapes(module):
    tape = module.TAPE
    if len(tape) != 719:
        raise ValueError(f"expected 719 frames, got {len(tape)}")
    tape.reset()
    base = [tape[step] for step in range(len(tape))]
    pair_route_ids = []
    for first in SHOPS:
        for second in SHOPS:
            tape.reset()
            tape.select({"step": 144,
                         "town": {"unlocked_shops": [first, second]}})
            pair_route_ids.append(-1 if tape.route is None else int(tape.route))
    # Route 100 is the source's explicit fallback for an otherwise unknown
    # non-YARN shop pair. All official 7x7 non-YARN pairs are enumerated, so
    # it is not selected by the current table, but retaining it makes the
    # frozen asset faithful to the complete routed source support.
    route_ids = sorted({100, *(set(pair_route_ids) - {-1})})
    routes = []
    for route in route_ids:
        tape.reset()
        tape.route = route
        routes.append([tape[step] for step in range(len(tape))])
    tape.reset()
    return base, route_ids, routes, pair_route_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=HERE / "fieldcraft_2887.assets.bin")
    parser.add_argument("--manifest", type=Path,
                        default=HERE / "assets.manifest.json")
    args = parser.parse_args()

    release = json.loads(RELEASE_MANIFEST.read_text())
    if release.get("release_archive_sha256") != EXPECTED_ARCHIVE:
        raise RuntimeError("Fieldcraft release archive provenance drift")
    module = load_source()
    base, route_ids, routes, pair_route_ids = routed_tapes(module)
    route_to_index = {route: index for index, route in enumerate(route_ids)}
    pair_indices = [route_to_index.get(route, -1) for route in pair_route_ids]

    main_hash = bytes.fromhex(EXPECTED_MAIN)
    mirror_hash = bytes.fromhex(EXPECTED_MIRROR)
    header = (b"FC2887\0\0" + struct.pack("<I", 1) + main_hash + mirror_hash +
              struct.pack("<HH", len(base), len(route_ids)))
    route_header = b"".join(struct.pack("<h", route) for route in route_ids)
    pair_blob = b"".join(struct.pack("<b", index) for index in pair_indices)
    frames = b"".join(pack_player_action(frame) for frame in base)
    frames += b"".join(pack_player_action(frame)
                       for route in routes for frame in route)
    blob = header + route_header + pair_blob + frames

    args.output.write_bytes(blob)
    manifest = {
        "schema": "fieldcraft-2887-native-assets-v1",
        "main_source": str(SOURCE.relative_to(ROOT)),
        "main_source_sha256": EXPECTED_MAIN,
        "mirror_source": str(MIRROR.relative_to(ROOT)),
        "mirror_source_sha256": EXPECTED_MIRROR,
        "release_archive_sha256": EXPECTED_ARCHIVE,
        "license": release.get("license"),
        "public_score": release.get("public_score"),
        "asset_sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
        "frames": len(base),
        "routes": route_ids,
        "route_count": len(route_ids),
        "shop_pair_routes": pair_route_ids,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
