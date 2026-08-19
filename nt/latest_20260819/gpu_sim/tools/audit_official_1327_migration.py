"""Audit the 1.32.6 -> 1.32.7 rule and frozen-table migration."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
OLD_SOURCE = (
    PROJECT
    / "reference"
    / "kaggle_environments_1_32_6"
    / "kaggriculture"
    / "kaggriculture.py"
)
NEW_SOURCE = (
    PROJECT
    / "reference"
    / "kaggle_environments_1_32_7"
    / "kaggriculture"
    / "kaggriculture.py"
)
OLD_TABLES = PROJECT / "reference" / "static_tables_v1.npz"
NEW_TABLES = PROJECT / "reference" / "static_tables_v2.npz"
OUTPUT = (
    WORKSPACE
    / "research"
    / "official_env_update_20260815"
    / "rules_and_tables_diff.json"
)
INVENTORY_MIN = -32768
INVENTORY_MAX = 65535


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    old = load_module(OLD_SOURCE, "kaggriculture_official_1326")
    new = load_module(NEW_SOURCE, "kaggriculture_official_1327")
    inventories = np.arange(INVENTORY_MIN, INVENTORY_MAX + 1, dtype=np.int64)

    with np.load(OLD_TABLES, allow_pickle=False) as old_npz:
        old_price = np.asarray(old_npz["market_price"], dtype=np.int64)
        old_events = {
            key: np.asarray(old_npz[key])
            for key in ("event_seeds", "weed_spawn", "shop_choice")
        }
        old_products = tuple(str(item) for item in old_npz["products"].tolist())
    with np.load(NEW_TABLES, allow_pickle=False) as new_npz:
        new_price = np.asarray(new_npz["market_price"], dtype=np.int64)
        new_events = {
            key: np.asarray(new_npz[key])
            for key in ("event_seeds", "weed_spawn", "shop_choice")
        }
        new_products = tuple(str(item) for item in new_npz["products"].tolist())
        new_price_dtype = str(new_npz["market_price"].dtype)

    old_exact = np.asarray(
        [[old.market_price(item, int(inv)) for inv in inventories] for item in old_products],
        dtype=np.int64,
    )
    new_exact = np.asarray(
        [[new.market_price(item, int(inv)) for inv in inventories] for item in new_products],
        dtype=np.int64,
    )
    if not np.array_equal(old_price, old_exact):
        raise AssertionError("static_tables_v1 does not exactly match official 1.32.6")
    if not np.array_equal(new_price, new_exact):
        raise AssertionError("static_tables_v2 does not exactly match official 1.32.7")

    product_rows = []
    for index, item in enumerate(new_products):
        changed = old_price[index] != new_price[index]
        positions = np.flatnonzero(changed)
        old_params = dict(old.MARKET_PARAMS[item])
        new_params = dict(new.MARKET_PARAMS[item])
        i0 = int(new_params["I0"])
        capacity = int(new_params["T"])
        sample_inventories = sorted(
            {
                i0 + capacity,
                i0,
                i0 - capacity // 2,
                i0 - capacity,
                i0 - 2 * capacity,
            },
            reverse=True,
        )
        samples = [
            {
                "inventory": inv,
                "old_price": int(old.market_price(item, inv)),
                "new_price": int(new.market_price(item, inv)),
            }
            for inv in sample_inventories
        ]
        product_rows.append(
            {
                "product": item,
                "params_changed": old_params != new_params,
                "old_params": old_params,
                "new_params": new_params,
                "lut_changed_cells": int(positions.size),
                "changed_inventory_first": (
                    int(inventories[positions[0]]) if positions.size else None
                ),
                "changed_inventory_last": (
                    int(inventories[positions[-1]]) if positions.size else None
                ),
                "max_absolute_price_delta": int(
                    np.max(np.abs(new_price[index] - old_price[index]))
                ),
                "new_price_min": int(np.min(new_price[index])),
                "new_price_max": int(np.max(new_price[index])),
                "samples": samples,
            }
        )

    event_checks = {
        key: {
            "same": bool(np.array_equal(old_events[key], new_events[key])),
            "shape": list(new_events[key].shape),
        }
        for key in old_events
    }
    changed_products = [
        row["product"] for row in product_rows if row["lut_changed_cells"] > 0
    ]
    # Under the default competition rules these three products cannot be
    # purchased from the market.  Player SELL only raises inventory, so their
    # minimum reachable inventory is bounded exactly by the most adversarial
    # with-replacement shop draw plus the 30 town-center ticks.
    unlock_days = list(range(3, 25, 3))
    last_shop_tick = 716
    shop_ticks_by_unlock_day = {
        str(day): (last_shop_tick - 24 * day) // 4 + 1 for day in unlock_days
    }
    total_shop_instance_ticks = sum(shop_ticks_by_unlock_day.values())
    default_changed_product_bounds = {}
    for item in changed_products:
        best_shop_multiplier = max(
            (
                2 if len(products) == 1 else 1
                for products in new.SHOPS.values()
                if item in products
            ),
            default=0,
        )
        town_center_ticks = 30 if item in new.TOWN_CENTER_PRODUCTS else 0
        maximum_consumption = (
            total_shop_instance_ticks * best_shop_multiplier + town_center_ticks
        )
        minimum_inventory = int(new.MARKET_PARAMS[item]["I0"]) - maximum_consumption
        default_changed_product_bounds[item] = {
            "player_buy_product_allowed": False,
            "maximum_shop_multiplier_per_tick": best_shop_multiplier,
            "maximum_shop_consumption": total_shop_instance_ticks
            * best_shop_multiplier,
            "town_center_consumption": town_center_ticks,
            "minimum_inventory": minimum_inventory,
            "maximum_price": int(new.market_price(item, minimum_inventory)),
        }
    report = {
        "schema": "kaggriculture_official_1_32_7_migration_audit_v1",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "old_version": "1.32.6",
        "new_version": "1.32.7",
        "sources": {
            "old_sha256": sha256(OLD_SOURCE),
            "new_sha256": sha256(NEW_SOURCE),
            "old_static_tables_sha256": sha256(OLD_TABLES),
            "new_static_tables_sha256": sha256(NEW_TABLES),
        },
        "market_lut": {
            "shape": list(new_price.shape),
            "new_dtype": new_price_dtype,
            "inventory_min": INVENTORY_MIN,
            "inventory_max": INVENTORY_MAX,
            "official_1_32_6_exact": True,
            "official_1_32_7_exact": True,
            "changed_products": changed_products,
            "global_new_price_max": int(np.max(new_price)),
            "int16_would_overflow": bool(np.max(new_price) > np.iinfo(np.int16).max),
            "products": product_rows,
        },
        "event_tables": event_checks,
        "event_tables_reusable": all(row["same"] for row in event_checks.values()),
        "default_competition_reachable_bound": {
            "scope": "changed products only; no marketParams/config overrides",
            "last_processed_shop_tick": last_shop_tick,
            "shop_ticks_by_unlock_day": shop_ticks_by_unlock_day,
            "total_shop_instance_ticks": total_shop_instance_ticks,
            "town_center_ticks": 30,
            "products": default_changed_product_bounds,
            "note": (
                "SELL can only raise inventory for these products, and BUY_PRODUCT "
                "does not allow them; therefore town consumption gives their exact "
                "default lower-inventory bound."
            ),
        },
        "ok": (
            changed_products == ["CARROT", "TOMATO", "EGG"]
            and new_price_dtype == "int32"
            and all(row["same"] for row in event_checks.values())
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
