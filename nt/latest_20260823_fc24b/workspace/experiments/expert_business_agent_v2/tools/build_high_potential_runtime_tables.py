from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
GOLD_TOOLS = ROOT / "experiments" / "gold_adaptive_rule_v2" / "tools"
sys.path.insert(0, str(GOLD_TOOLS))

from build_current_gold_gpu_trace_bank import _encode_stream  # noqa: E402


SOURCE = (
    ROOT
    / "references"
    / "public_high_potential_20260819"
    / "boatlee_v20_multi_route"
    / "main.py"
)
ROUTE_BANK = (
    ROOT
    / "experiments"
    / "expert_business_agent_v2"
    / "artifacts"
    / "high_potential_route_bank_v1.npz"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("hp_v20_runtime_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def market_stream(rows: list[list]) -> list[dict]:
    return [
        {"farmer": ["PASS"], "hands": [], "market": list(market or [])}
        for market in rows[:719]
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--route-bank", type=Path, default=ROUTE_BANK)
    args = parser.parse_args()
    module = load_module(SOURCE)
    r5 = _encode_stream(market_stream(module._V17_R5_MARKETS))
    md = _encode_stream(market_stream(module._V17_MD_MARKETS))
    route_bank = args.route_bank.resolve()
    with np.load(route_bank, allow_pickle=False) as bank:
        unit_op = np.asarray(bank["unit_op"])
        unit_item = np.asarray(bank["unit_item"])
        unit_count = np.asarray(bank["unit_count"])
    present = np.arange(unit_op.shape[2])[None, None, :] < unit_count[:, :, None]
    future_plant = np.zeros((unit_op.shape[0], 719, 2), dtype=np.int16)
    for ordinal, crop in enumerate((0, 1)):
        plant = present & (unit_op == 8) & (unit_item == crop)
        by_step = plant.sum(axis=2).astype(np.int16)
        by_step[:, 672:] = 0  # day 28+ cannot finish WHEAT/CARROT before day 30.
        suffix = np.flip(np.cumsum(np.flip(by_step, axis=1), axis=1), axis=1)
        future_plant[:, :-1, ordinal] = suffix[:, 1:]
    market_min_inventory = -32768
    market_max_inventory = 65535
    product_names = (
        "WHEAT",
        "CARROT",
        "TOMATO",
        "STRAWBERRY",
        "MELON",
        "EGG",
        "MILK",
        "WOOL",
        "FERTILIZER",
    )
    inventories = range(market_min_inventory, market_max_inventory + 1)
    agent_market_price_lut = np.asarray(
        [
            [module._market_price(product, inventory) for inventory in inventories]
            for product in product_names
        ],
        dtype=np.int32,
    )
    arrays = {
        "r5_market_op": r5["market_op"],
        "r5_market_item": r5["market_item"],
        "r5_market_amount": r5["market_amount"],
        "r5_market_count": r5["market_count"],
        "md_market_op": md["market_op"],
        "md_market_item": md["market_item"],
        "md_market_amount": md["market_amount"],
        "md_market_count": md["market_count"],
        "future_plant_before_day28": future_plant,
        "agent_market_price_lut": agent_market_price_lut,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    receipt = {
        "schema": "kaggriculture.high_potential_runtime_tables.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE.resolve()),
        "source_sha256": sha256(SOURCE),
        "output": str(output),
        "output_sha256": sha256(output),
        "steps": 719,
        "route_bank": str(route_bank),
        "route_bank_sha256": sha256(route_bank),
        "contents": [
            "V17_R5_MARKETS",
            "V17_MD_MARKETS",
            "future_plant_before_day28",
            "agent_market_price_lut",
        ],
    }
    receipt_path = args.receipt.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
