"""Generate exact Python-derived market and stochastic lookup tables.

The official simulator uses Python's binary64 arithmetic, bankers ``round`` and
``random.Random``.  The GPU simulator consumes these frozen tables so its hot
path stays integer/bool-only while preserving those observable results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
REFERENCE = PROJECT / "reference" / "kaggle_environments_1_32_6" / "kaggriculture"
SOURCE = REFERENCE / "kaggriculture.py"
OUTPUT = PROJECT / "reference" / "static_tables_v1.npz"
RECEIPT = PROJECT / "receipts" / "static_tables.json"

MARKET_MIN_INVENTORY = -32768
MARKET_MAX_INVENTORY = 65535
EVENT_DAYS = 30
EVENT_DRAWS = 200  # two 10x10 farms, conditional draw per empty tile
WEED_CHANCE = 0.005
SHOP_NAMES = tuple(
    sorted(
        (
            "BAKERY",
            "PIZZA_SHOP",
            "BRUNCH_SPOT",
            "YARN_STORE",
            "ICE_CREAM_SHOP",
            "PET_CAFE",
            "SMOOTHIE_SHOP",
            "FARMERS_MARKET",
        )
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official():
    spec = importlib.util.spec_from_file_location("frozen_kaggriculture", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event_seeds() -> np.ndarray:
    # 0..127 are development seeds; 10000..10127 are held out for the final
    # 100-seed parity gate.  Keeping both sets in one device bank is cheap.
    return np.asarray([*range(128), *range(10000, 10128)], dtype=np.int64)


def build_events(seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weed = np.zeros((len(seeds), EVENT_DAYS, EVENT_DRAWS), dtype=np.bool_)
    choice = np.zeros((len(seeds), EVENT_DAYS, EVENT_DRAWS + 1), dtype=np.int8)
    for seed_idx, seed_value in enumerate(seeds.tolist()):
        for day in range(EVENT_DAYS):
            rng = random.Random((seed_value * 1_000_003) ^ day)
            for consumed in range(EVENT_DRAWS + 1):
                # random.choice uses getrandbits/_randbelow from the precise MT
                # state after ``consumed`` conditional weed draws.
                clone = random.Random()
                clone.setstate(rng.getstate())
                choice[seed_idx, day, consumed] = SHOP_NAMES.index(
                    clone.choice(SHOP_NAMES)
                )
                if consumed < EVENT_DRAWS:
                    weed[seed_idx, day, consumed] = rng.random() < WEED_CHANCE
    return weed, choice


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()

    official = load_official()
    products = tuple(official.PRODUCTS)
    inventories = range(MARKET_MIN_INVENTORY, MARKET_MAX_INVENTORY + 1)
    price_lut = np.asarray(
        [
            [official.market_price(item, inventory) for inventory in inventories]
            for item in products
        ],
        dtype=np.int16,
    )
    seeds = event_seeds()
    weed_spawn, shop_choice = build_events(seeds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        market_price=price_lut,
        event_seeds=seeds,
        weed_spawn=weed_spawn,
        shop_choice=shop_choice,
        products=np.asarray(products),
        shop_names=np.asarray(SHOP_NAMES),
    )
    receipt = {
        "schema": "kaggriculture_static_tables_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_source_sha256": sha256(SOURCE),
        "output": str(args.output.relative_to(PROJECT)),
        "output_sha256": sha256(args.output),
        "market": {
            "dtype": str(price_lut.dtype),
            "shape": list(price_lut.shape),
            "inventory_min": MARKET_MIN_INVENTORY,
            "inventory_max": MARKET_MAX_INVENTORY,
            "products": list(products),
        },
        "events": {
            "seed_count": int(len(seeds)),
            "development_seeds": [0, 127],
            "heldout_seeds": [10000, 10127],
            "days": EVENT_DAYS,
            "max_conditional_weed_draws": EVENT_DRAWS,
            "weed_chance": WEED_CHANCE,
            "weed_shape": list(weed_spawn.shape),
            "shop_choice_shape": list(shop_choice.shape),
            "shop_names": list(SHOP_NAMES),
        },
        "ok": True,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
