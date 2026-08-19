"""Compile current-gold imitation action streams into one JAX trace bank.

This is a host-side build step.  The resulting arrays are candidate-major and
can be consumed by ``route_playbook_v1.trace_core.skeleton_player_action_v1``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
MAX_UNITS = 33
MAX_MARKET_ORDERS = 10
UNIT_OPS = {
    name: index
    for index, name in enumerate(
        (
            "PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP", "PICKUP",
            "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
            "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE",
        )
    )
}
MARKET_OPS = {
    name: index
    for index, name in enumerate(
        ("NONE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL")
    )
}
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = PRODUCTS[:5]
ANIMALS = ("GOOSE", "COW", "SHEEP")
SHED_ITEMS = PRODUCTS + ANIMALS
PRODUCT_IDS = {name: index for index, name in enumerate(PRODUCTS)}
CROP_IDS = {name: index for index, name in enumerate(CROPS)}
ANIMAL_IDS = {name: len(PRODUCTS) + index for index, name in enumerate(ANIMALS)}
SHED_ITEM_IDS = {name: index for index, name in enumerate(SHED_ITEMS)}


def _unit(action) -> tuple[int, int, int]:
    if not isinstance(action, list) or not action:
        return UNIT_OPS["PASS"], -1, 1
    op = UNIT_OPS.get(action[0], -1)
    item, amount = -1, 1
    if op == UNIT_OPS["PLANT"] and len(action) >= 2:
        item = CROP_IDS.get(action[1], -1)
    elif op in (UNIT_OPS["PICKUP"], UNIT_OPS["PLACE"]) and len(action) >= 2:
        item = SHED_ITEM_IDS.get(action[1], -1)
        if len(action) >= 3:
            try:
                amount = int(action[2])
            except (TypeError, ValueError, OverflowError):
                return -1, -1, 0
    return int(op), int(item), int(amount)


def _market(order) -> tuple[int, int, int]:
    if not isinstance(order, list) or not order:
        return MARKET_OPS["NONE"], -1, 0
    op = MARKET_OPS.get(order[0], MARKET_OPS["NONE"])
    if op in (MARKET_OPS["HIRE"], MARKET_OPS["BUY_LAND"]):
        return op, -1, 0
    if len(order) < 3:
        return MARKET_OPS["NONE"], -1, 0
    if op == MARKET_OPS["BUY_SEED"]:
        item = CROP_IDS.get(order[1], -1)
    elif op == MARKET_OPS["BUY_ANIMAL"]:
        item = ANIMAL_IDS.get(order[1], -1)
    elif op in (MARKET_OPS["BUY_PRODUCT"], MARKET_OPS["SELL"]):
        item = PRODUCT_IDS.get(order[1], -1)
    else:
        return MARKET_OPS["NONE"], -1, 0
    try:
        amount = int(order[2])
    except (TypeError, ValueError, OverflowError):
        return MARKET_OPS["NONE"], -1, 0
    return int(op), int(item), int(amount)


FIELDS = (
    "unit_op",
    "unit_item",
    "unit_amount",
    "unit_count",
    "market_op",
    "market_item",
    "market_amount",
    "market_count",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path: Path, index: int):
    spec = importlib.util.spec_from_file_location(f"gold_gpu_trace_{index}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encode_stream(stream: list[dict]) -> dict[str, np.ndarray]:
    if len(stream) != 719:
        raise ValueError(f"Expected 719 actions, got {len(stream)}")
    unit_op = np.full((719, MAX_UNITS), UNIT_OPS["PASS"], dtype=np.int8)
    unit_item = np.full((719, MAX_UNITS), -1, dtype=np.int8)
    unit_amount = np.ones((719, MAX_UNITS), dtype=np.int32)
    unit_count = np.ones((719,), dtype=np.int8)
    market_op = np.full(
        (719, MAX_MARKET_ORDERS), MARKET_OPS["NONE"], dtype=np.int8
    )
    market_item = np.full((719, MAX_MARKET_ORDERS), -1, dtype=np.int8)
    market_amount = np.zeros((719, MAX_MARKET_ORDERS), dtype=np.int32)
    market_count = np.zeros((719,), dtype=np.int8)
    for step, raw in enumerate(stream):
        action = raw if isinstance(raw, dict) else {}
        hands = action.get("hands", [])
        hands = hands if isinstance(hands, list) else []
        units = [action.get("farmer", ["PASS"]), *hands]
        unit_count[step] = min(len(units), MAX_UNITS)
        for index, value in enumerate(units[:MAX_UNITS]):
            op, item, amount = _unit(value)
            unit_op[step, index] = op
            unit_item[step, index] = item
            unit_amount[step, index] = amount
        orders = action.get("market", [])
        orders = orders if isinstance(orders, list) else []
        market_count[step] = min(len(orders), MAX_MARKET_ORDERS)
        for index, value in enumerate(orders[:MAX_MARKET_ORDERS]):
            op, item, amount = _market(value)
            market_op[step, index] = op
            market_item[step, index] = item
            market_amount[step, index] = amount
    return {
        "unit_op": unit_op,
        "unit_item": unit_item,
        "unit_amount": unit_amount,
        "unit_count": unit_count,
        "market_op": market_op,
        "market_item": market_item,
        "market_amount": market_amount,
        "market_count": market_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EXPERIMENT / "configs" / "current_gold_imitation_candidates_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT / "artifacts" / "current_gold_gpu_trace_bank_v1.npz",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=EXPERIMENT / "artifacts" / "current_gold_gpu_trace_bank_v1.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    compiled: dict[str, list[np.ndarray]] = {field: [] for field in FIELDS}
    rows: list[dict] = []
    for candidate_index, candidate in enumerate(manifest["candidates"]):
        path = Path(candidate["path"])
        module = _load_module(path, candidate_index)
        streams = getattr(module, "_CGR_STREAMS", None)
        if not isinstance(streams, dict) or not streams:
            actions = getattr(module, "_ACTIONS", None)
            if not isinstance(actions, list):
                raise ValueError(f"No action stream found in {path}")
            streams = {"default": actions}
        for route_name, stream in streams.items():
            encoded = _encode_stream(stream)
            skeleton_id = len(rows)
            for field in FIELDS:
                compiled[field].append(encoded[field])
            canonical = json.dumps(
                stream, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
            rows.append(
                {
                    "skeleton_id": skeleton_id,
                    "candidate": candidate["name"],
                    "route": str(route_name),
                    "source_path": str(path),
                    "source_sha256": _sha256(path),
                    "action_sha256": hashlib.sha256(canonical).hexdigest(),
                }
            )

    arrays = {field: np.stack(values, axis=0) for field, values in compiled.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    receipt = {
        "schema": "kaggriculture-current-gold-gpu-trace-bank-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "truth_boundary": (
            "GPU screening trace bank. Official kaggle-environments 1.32.7 "
            "seat-swapped evaluation remains the referee."
        ),
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": _sha256(args.manifest),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "candidate_count": len(manifest["candidates"]),
        "skeleton_count": len(rows),
        "shapes": {field: list(value.shape) for field, value in arrays.items()},
        "skeletons": rows,
    }
    args.receipt.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "candidate_count": receipt["candidate_count"],
                "skeleton_count": receipt["skeleton_count"],
                "output": str(args.output),
                "receipt": str(args.receipt),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
