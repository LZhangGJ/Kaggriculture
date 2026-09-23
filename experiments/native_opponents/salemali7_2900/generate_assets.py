#!/usr/bin/env python3
"""Pack the frozen Salemali7 action tape for Python-free native rollout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "opponents/salemali7_2900/main.py"

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


def load_source():
    spec = importlib.util.spec_from_file_location("salemali7_asset_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
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
    if len(units) > 255 or len(market) > 10:
        raise ValueError("action exceeds frozen ABI bounds")
    return (struct.pack("<BB", len(units), len(market)) +
            b"".join(pack_action(action) for action in (*units, *market)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=HERE / "salemali7_2900.assets.bin")
    parser.add_argument("--manifest", type=Path,
                        default=HERE / "assets.manifest.json")
    args = parser.parse_args()

    source_bytes = SOURCE.read_bytes()
    actions = load_source()._ACTIONS
    if len(actions) != 720:
        raise ValueError(f"expected 720 source frames, got {len(actions)}")
    source_hash = hashlib.sha256(source_bytes).digest()
    blob = (b"SL72900\0" + struct.pack("<I", 1) + source_hash +
            struct.pack("<H", len(actions)) +
            b"".join(pack_player_action(action) for action in actions))
    manifest = {
        "schema": "salemali7-2900-native-assets-v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_hash.hex(),
        "asset_sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
        "frames": len(actions),
    }
    args.output.write_bytes(blob)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
