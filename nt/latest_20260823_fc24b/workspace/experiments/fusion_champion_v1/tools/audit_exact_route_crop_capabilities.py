"""Audit crop capabilities encoded in frozen exact-JAX route banks.

This is a static intent audit.  It does not claim that an intent succeeded in
the simulator; its purpose is to cheaply identify route traces worth a later
state-effect and Arena audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
PRODUCTS = CROPS + ("EGG", "MILK", "WOOL", "FERTILIZER")

UNIT_PASS = 0
UNIT_PLANT = 8
UNIT_WATER = 9
UNIT_HARVEST = 10
UNIT_FERTILIZE = 11
UNIT_FEED = 15
UNIT_CARE = 17

MARKET_BUY_SEED = 3
MARKET_BUY_PRODUCT = 4
MARKET_BUY_ANIMAL = 5
MARKET_SELL = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_last_step(mask: np.ndarray) -> tuple[int | None, int | None]:
    steps = np.flatnonzero(np.any(mask, axis=1))
    if not len(steps):
        return None, None
    return int(steps[0]), int(steps[-1])


def amount_sum(op: np.ndarray, item: np.ndarray, amount: np.ndarray, wanted_op: int, wanted_item: int) -> int:
    mask = (op == wanted_op) & (item == wanted_item)
    return int(np.where(mask, amount, 0).sum())


def count_intents(op: np.ndarray, wanted_op: int) -> int:
    return int((op == wanted_op).sum())


def load_names(receipt: dict, route_count: int) -> list[dict]:
    records = receipt.get("skeletons") or receipt.get("routes") or []
    by_id: dict[int, dict] = {}
    for index, record in enumerate(records):
        route_id = record.get("skeleton_id", record.get("route_id", index))
        by_id[int(route_id)] = record
    result = []
    for route_id in range(route_count):
        record = by_id.get(route_id, {})
        result.append(
            {
                "route_id": route_id,
                "name": record.get("route") or record.get("canonical_name") or f"route_{route_id}",
                "opponent": record.get("opponent"),
                "source": record.get("source"),
                "action_sha256": record.get("action_sha256"),
            }
        )
    return result


def audit_bank(bank_path: Path, receipt_path: Path) -> dict:
    with receipt_path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    with np.load(bank_path) as bank:
        arrays = {key: bank[key] for key in bank.files}

    route_count, episode_steps, max_units = arrays["unit_op"].shape
    _, market_steps, max_market_orders = arrays["market_op"].shape
    if episode_steps != market_steps:
        raise ValueError("unit and market traces have different step counts")

    metadata = load_names(receipt, route_count)
    rows = []
    for route_id in range(route_count):
        unit_op = arrays["unit_op"][route_id]
        unit_item = arrays["unit_item"][route_id]
        market_op = arrays["market_op"][route_id]
        market_item = arrays["market_item"][route_id]
        market_amount = arrays["market_amount"][route_id]

        crop_rows = []
        for crop_id, crop in enumerate(CROPS):
            plant_mask = (unit_op == UNIT_PLANT) & (unit_item == crop_id)
            buy_mask = (market_op == MARKET_BUY_SEED) & (market_item == crop_id)
            sell_mask = (market_op == MARKET_SELL) & (market_item == crop_id)
            first_plant, last_plant = first_last_step(plant_mask)
            first_buy, last_buy = first_last_step(buy_mask)
            first_sell, last_sell = first_last_step(sell_mask)
            crop_rows.append(
                {
                    "crop": crop,
                    "buy_seed_intent_quantity": amount_sum(
                        market_op, market_item, market_amount, MARKET_BUY_SEED, crop_id
                    ),
                    "plant_intent_count": int(plant_mask.sum()),
                    "sell_intent_quantity": amount_sum(
                        market_op, market_item, market_amount, MARKET_SELL, crop_id
                    ),
                    "first_buy_step": first_buy,
                    "last_buy_step": last_buy,
                    "first_plant_step": first_plant,
                    "last_plant_step": last_plant,
                    "first_sell_step": first_sell,
                    "last_sell_step": last_sell,
                }
            )

        carrot = crop_rows[1]
        # HARVEST is item-free, so this is deliberately only a candidate gate.
        # Dynamic effect validation must still prove that carrot was harvested.
        carrot_candidate = bool(
            carrot["buy_seed_intent_quantity"] > 0
            and carrot["plant_intent_count"] > 0
            and count_intents(unit_op, UNIT_HARVEST) > 0
            and carrot["sell_intent_quantity"] > 0
        )
        row = dict(metadata[route_id])
        row.update(
            {
                "crop_intents": crop_rows,
                "unit_intent_counts": {
                    "active": int((unit_op != UNIT_PASS).sum()),
                    "water": count_intents(unit_op, UNIT_WATER),
                    "harvest": count_intents(unit_op, UNIT_HARVEST),
                    "fertilize": count_intents(unit_op, UNIT_FERTILIZE),
                    "feed": count_intents(unit_op, UNIT_FEED),
                    "care": count_intents(unit_op, UNIT_CARE),
                },
                "market_intent_counts": {
                    "buy_product": count_intents(market_op, MARKET_BUY_PRODUCT),
                    "buy_animal": count_intents(market_op, MARKET_BUY_ANIMAL),
                    "sell": count_intents(market_op, MARKET_SELL),
                },
                "carrot_complete_chain_candidate": carrot_candidate,
            }
        )
        rows.append(row)

    candidates = [row["route_id"] for row in rows if row["carrot_complete_chain_candidate"]]
    return {
        "bank": str(bank_path),
        "bank_sha256": sha256(bank_path),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "shape": {
            "routes": route_count,
            "steps": episode_steps,
            "max_units": max_units,
            "max_market_orders": max_market_orders,
        },
        "carrot_complete_chain_candidate_ids": candidates,
        "routes": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", action="append", type=Path, required=True)
    parser.add_argument("--receipt", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.bank) != len(args.receipt):
        raise SystemExit("--bank and --receipt counts must match")
    output = {
        "schema": "fc8.exact_route_crop_capability_audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "truth_boundary": [
            "Static raw-intent audit only; it does not prove official state effects.",
            "HARVEST has no crop item, so dynamic effect validation remains mandatory.",
            "No opponent identity or future state is used.",
        ],
        "banks": [audit_bank(bank, receipt) for bank, receipt in zip(args.bank, args.receipt)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "output": str(args.output),
        "banks": [
            {
                "routes": bank["shape"]["routes"],
                "carrot_candidates": len(bank["carrot_complete_chain_candidate_ids"]),
            }
            for bank in output["banks"]
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
