"""Compile nine coherent Hasegawa Replay programs for GPU runtime use."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


MAX_UNITS = 33
MAX_MARKET_ORDERS = 10
EPISODE_ACTIONS = 719
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
SHOP_NAMES = tuple(sorted(("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET")))
SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
SHOP_DEMAND = np.asarray(
    [[(2 if len(SHOP_PRODUCTS[s]) == 1 else 1) if p in SHOP_PRODUCTS[s] else 0 for p in PRODUCTS] for s in SHOP_NAMES],
    dtype=np.int16,
)
PRODUCT_ID = {name: index for index, name in enumerate(PRODUCTS)}
CROP_ID = {name: index for index, name in enumerate(CROPS)}
SHED_ID = {name: index for index, name in enumerate(SHED_ITEMS)}
ANIMAL_ID = {name: len(PRODUCTS) + index for index, name in enumerate(ANIMALS)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit(raw) -> tuple[int, int, int]:
    if not isinstance(raw, list) or not raw:
        return UNIT_OPS["PASS"], -1, 1
    op = UNIT_OPS.get(str(raw[0]), UNIT_OPS["PASS"])
    item, amount = -1, 1
    if op == UNIT_OPS["PLANT"] and len(raw) >= 2:
        item = CROP_ID.get(str(raw[1]), -1)
    elif op in (UNIT_OPS["PICKUP"], UNIT_OPS["PLACE"]) and len(raw) >= 2:
        item = SHED_ID.get(str(raw[1]), -1)
        amount = int(raw[2]) if len(raw) >= 3 else 1
    return int(op), int(item), int(amount)


def _market(raw) -> tuple[int, int, int]:
    if not isinstance(raw, list) or not raw:
        return MARKET_OPS["NONE"], -1, 0
    op = MARKET_OPS.get(str(raw[0]), MARKET_OPS["NONE"])
    if op in (MARKET_OPS["HIRE"], MARKET_OPS["BUY_LAND"]):
        return op, -1, 0
    if len(raw) < 3:
        return MARKET_OPS["NONE"], -1, 0
    if op == MARKET_OPS["BUY_SEED"]:
        item = CROP_ID.get(str(raw[1]), -1)
    elif op == MARKET_OPS["BUY_ANIMAL"]:
        item = ANIMAL_ID.get(str(raw[1]), -1)
    else:
        item = PRODUCT_ID.get(str(raw[1]), -1)
    return int(op), int(item), int(raw[2])


def _hasegawa_seat(document: dict) -> int:
    teams = document.get("info", {}).get("TeamNames", [])
    matches = [i for i, name in enumerate(teams) if "Hasegawa" in str(name)]
    if len(matches) != 1:
        raise ValueError(f"cannot identify Hasegawa seat: {teams}")
    return matches[0]


def _first_shop(document: dict, seat: int) -> str:
    for frame in document["steps"]:
        shops = frame[seat]["observation"].get("town", {}).get("unlocked_shops", []) or []
        if shops:
            return str(shops[0])
    raise ValueError("Replay has no unlocked shop")


def _encode(document: dict, seat: int) -> dict[str, np.ndarray]:
    unit_op = np.full((EPISODE_ACTIONS, MAX_UNITS), UNIT_OPS["PASS"], np.int8)
    unit_item = np.full((EPISODE_ACTIONS, MAX_UNITS), -1, np.int8)
    unit_amount = np.ones((EPISODE_ACTIONS, MAX_UNITS), np.int32)
    unit_count = np.ones((EPISODE_ACTIONS,), np.int8)
    market_op = np.full((EPISODE_ACTIONS, MAX_MARKET_ORDERS), MARKET_OPS["NONE"], np.int8)
    market_item = np.full((EPISODE_ACTIONS, MAX_MARKET_ORDERS), -1, np.int8)
    market_amount = np.zeros((EPISODE_ACTIONS, MAX_MARKET_ORDERS), np.int32)
    market_count = np.zeros((EPISODE_ACTIONS,), np.int8)
    expected_pos = np.zeros((EPISODE_ACTIONS, MAX_UNITS, 2), np.int8)
    expected_active = np.zeros((EPISODE_ACTIONS, MAX_UNITS), np.bool_)
    expected_money = np.zeros((EPISODE_ACTIONS,), np.int32)
    for step in range(EPISODE_ACTIONS):
        obs = document["steps"][step][seat]["observation"]
        farm = obs["farms"][seat]
        positions = [farm["farmer"], *(farm.get("hands", []) or [])]
        expected_active[step, : min(len(positions), MAX_UNITS)] = True
        expected_pos[step, : min(len(positions), MAX_UNITS)] = np.asarray(positions[:MAX_UNITS], dtype=np.int8)
        expected_money[step] = int(farm["money"])
        raw = document["steps"][step + 1][seat].get("action") or {}
        units = [raw.get("farmer", ["PASS"]), *(raw.get("hands", []) or [])]
        unit_count[step] = min(len(units), MAX_UNITS)
        for index, action in enumerate(units[:MAX_UNITS]):
            unit_op[step, index], unit_item[step, index], unit_amount[step, index] = _unit(action)
        orders = raw.get("market", []) or []
        market_count[step] = min(len(orders), MAX_MARKET_ORDERS)
        for index, order in enumerate(orders[:MAX_MARKET_ORDERS]):
            market_op[step, index], market_item[step, index], market_amount[step, index] = _market(order)
    return {
        "unit_op": unit_op, "unit_item": unit_item, "unit_amount": unit_amount,
        "unit_count": unit_count, "market_op": market_op, "market_item": market_item,
        "market_amount": market_amount, "market_count": market_count,
        "expected_unit_pos": expected_pos, "expected_unit_active": expected_active,
        "expected_money": expected_money,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    candidates: list[dict] = []
    for path in sorted(args.replay_root.resolve().rglob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if len(document.get("steps", [])) != 720:
            continue
        try:
            seat = _hasegawa_seat(document)
            shop = _first_shop(document, seat)
        except ValueError:
            continue
        reward = int(document.get("rewards", [0, 0])[seat])
        opponent = int(document.get("rewards", [0, 0])[1 - seat])
        if reward <= opponent:
            continue
        candidates.append({"path": path, "document": document, "seat": seat, "shop": shop, "reward": reward})
    if not candidates:
        raise RuntimeError("no winning Hasegawa Replays found")

    groups = [candidates]
    groups.extend([[row for row in candidates if row["shop"] == shop] for shop in SHOP_NAMES])
    selected = [max(group or candidates, key=lambda row: row["reward"]) for group in groups]
    compiled: dict[str, list[np.ndarray]] = {name: [] for name in (
        "unit_op", "unit_item", "unit_amount", "unit_count", "market_op", "market_item",
        "market_amount", "market_count", "expected_unit_pos", "expected_unit_active", "expected_money",
    )}
    sources = []
    for branch, row in enumerate(selected):
        encoded = _encode(row["document"], row["seat"])
        for field, value in encoded.items():
            compiled[field].append(value)
        sources.append({
            "branch": branch,
            "name": "UNKNOWN_GLOBAL" if branch == 0 else SHOP_NAMES[branch - 1],
            "episode_id": int(row["document"].get("info", {}).get("EpisodeId", row["path"].stem)),
            "reward": row["reward"], "seat": row["seat"],
            "path": str(row["path"]), "sha256": _sha256(row["path"]),
        })
    arrays = {field: np.stack(values) for field, values in compiled.items()}
    arrays.update(
        source_episode_id=np.asarray([row["episode_id"] for row in sources], np.int64),
        source_reward=np.asarray([row["reward"] for row in sources], np.int32),
        source_seat=np.asarray([row["seat"] for row in sources], np.int8),
        branch_sample_count=np.asarray([len(group) for group in groups], np.int16),
        branch_shop_demand=np.concatenate((np.zeros((1, len(PRODUCTS)), np.int16), SHOP_DEMAND), axis=0),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    receipt = {
        "schema": "kaggriculture.hasegawa_atomic_trace_bank.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
        "source_replay_root": str(args.replay_root.resolve()),
        "winning_replay_count": len(candidates), "branch_count": 9,
        "branch_selection": "highest-reward coherent full Replay within visible first-shop class",
        "runtime_branch_switches_after_lock": 0,
        "sources": sources, "output": str(args.output.resolve()), "output_sha256": _sha256(args.output),
        "shapes": {field: list(value.shape) for field, value in arrays.items()},
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "winning_replays": len(candidates), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
