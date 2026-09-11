"""Compile every winning Hasegawa Replay, including observable shop prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/hasegawa_jax_v2/tools"))

from build_trace_bank_v2 import (  # noqa: E402
    ANIMALS,
    EPISODE_ACTIONS,
    PRODUCTS,
    SHOP_NAMES,
    _encode,
    _hasegawa_seat,
)


CROPS = PRODUCTS[:5]
CROP_ID = {name: index for index, name in enumerate(CROPS)}
ANIMAL_ID = {name: index for index, name in enumerate(ANIMALS)}
PRODUCT_ID = {name: index for index, name in enumerate(PRODUCTS)}
SHED_ITEMS = PRODUCTS + ANIMALS
SHED_ID = {name: index for index, name in enumerate(SHED_ITEMS)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def shop_sequence(document: dict, seat: int) -> np.ndarray:
    result: list[str] = []
    for frame in document["steps"]:
        shops = frame[seat]["observation"].get("town", {}).get("unlocked_shops", []) or []
        if len(shops) > len(result):
            result = [str(value) for value in shops]
    encoded = np.full((8,), -1, np.int8)
    for index, name in enumerate(result[:8]):
        encoded[index] = SHOP_NAMES.index(name)
    return encoded


def farm_summary(farm: dict) -> np.ndarray:
    value = np.zeros((11,), np.int32)
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if tile != "LOCKED":
                value[8] += 1
            if not isinstance(tile, dict):
                continue
            crop = tile.get("crop")
            animal = tile.get("animal")
            if crop in CROP_ID:
                value[3 + CROP_ID[crop]] += 1
            if animal in ANIMAL_ID:
                value[ANIMAL_ID[animal]] += 1
    value[9] = len(farm.get("hands", []) or [])
    value[10] = int(farm.get("money", 0))
    return value


def public_arrays(document: dict, seat: int):
    own = np.zeros((EPISODE_ACTIONS, 11), np.int32)
    opponent = np.zeros_like(own)
    price = np.zeros((EPISODE_ACTIONS, len(PRODUCTS)), np.int32)
    inventory = np.zeros_like(price)
    shed = np.zeros((EPISODE_ACTIONS, len(SHED_ITEMS)), np.int32)
    seeds = np.zeros((EPISODE_ACTIONS, len(CROPS)), np.int32)
    carried = np.zeros_like(shed)
    for step in range(EPISODE_ACTIONS):
        obs = document["steps"][step][seat]["observation"]
        own[step] = farm_summary(obs["farms"][seat])
        opponent[step] = farm_summary(obs["farms"][1 - seat])
        market = obs["market"]
        private = obs["private"]
        for item, item_id in SHED_ID.items():
            shed[step, item_id] = int(private["shed"].get(item, 0))
        for crop, crop_id in CROP_ID.items():
            seeds[step, crop_id] = int(private["seeds"].get(crop, 0))
        for unit_inventory in private.get("inventories", []) or []:
            for item, quantity in unit_inventory.items():
                if item in SHED_ID:
                    carried[step, SHED_ID[item]] += int(quantity)
        for product, product_id in PRODUCT_ID.items():
            price[step, product_id] = int(market["prices"][product])
            inventory[step, product_id] = int(market["inventory"][product])
    return own, opponent, shed, seeds, carried, price, inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    candidates = []
    for path in sorted(args.replay_root.resolve().rglob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if len(document.get("steps", [])) != 720:
            continue
        try:
            seat = _hasegawa_seat(document)
        except ValueError:
            continue
        rewards = document.get("rewards", [0, 0])
        reward, opponent_reward = int(rewards[seat]), int(rewards[1 - seat])
        if reward <= opponent_reward:
            continue
        candidates.append({
            "path": path,
            "document": document,
            "seat": seat,
            "reward": reward,
            "opponent_reward": opponent_reward,
            "shops": shop_sequence(document, seat),
        })
    if not candidates:
        raise RuntimeError("no winning Hasegawa Replays found")
    # Stable route IDs and a separate explicit bootstrap route avoid hiding
    # strategy constants in runtime conditionals.
    candidates.sort(key=lambda row: (int(row["document"].get("info", {}).get("EpisodeId", 0)), str(row["path"])))
    fields = (
        "unit_op", "unit_item", "unit_amount", "unit_count", "market_op",
        "market_item", "market_amount", "market_count", "expected_unit_pos",
        "expected_unit_active", "expected_money",
    )
    compiled = {field: [] for field in fields}
    self_summary, opponent_summary, sheds, seed_banks, carried_items, prices, inventories = [], [], [], [], [], [], []
    sources = []
    for route_id, row in enumerate(candidates):
        encoded = _encode(row["document"], row["seat"])
        for field in fields:
            compiled[field].append(encoded[field])
        own, opponent, shed, seeds, carried, price, inventory = public_arrays(row["document"], row["seat"])
        self_summary.append(own)
        opponent_summary.append(opponent)
        sheds.append(shed)
        seed_banks.append(seeds)
        carried_items.append(carried)
        prices.append(price)
        inventories.append(inventory)
        episode_id = int(row["document"].get("info", {}).get("EpisodeId", row["path"].stem))
        sources.append({
            "route_id": route_id,
            "episode_id": episode_id,
            "reward": row["reward"],
            "opponent_reward": row["opponent_reward"],
            "margin": row["reward"] - row["opponent_reward"],
            "seat": row["seat"],
            "shops": row["shops"].tolist(),
            "path": str(row["path"]),
            "sha256": sha256(row["path"]),
        })
    arrays = {field: np.stack(value) for field, value in compiled.items()}
    arrays.update(
        expected_self_summary=np.stack(self_summary),
        expected_opponent_summary=np.stack(opponent_summary),
        expected_shed=np.stack(sheds),
        expected_seeds=np.stack(seed_banks),
        expected_carried=np.stack(carried_items),
        expected_market_price=np.stack(prices),
        expected_market_inventory=np.stack(inventories),
        source_shop_sequence=np.stack([row["shops"] for row in candidates]),
        source_episode_id=np.asarray([row["episode_id"] for row in sources], np.int64),
        source_reward=np.asarray([row["reward"] for row in sources], np.int32),
        source_opponent_reward=np.asarray([row["opponent_reward"] for row in sources], np.int32),
        source_margin=np.asarray([row["margin"] for row in sources], np.int32),
        source_seat=np.asarray([row["seat"] for row in sources], np.int8),
        bootstrap_route_id=np.asarray(int(np.argmax([row["reward"] for row in sources])), np.int16),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    receipt = {
        "schema": "kaggriculture.hasegawa_trace_bank.v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source_replay_root": str(args.replay_root.resolve()),
        "winning_replay_count": len(candidates),
        "route_count": len(candidates),
        "bootstrap_route_id": int(arrays["bootstrap_route_id"]),
        "selection_inputs": ["visible_shop_prefix", "public_state_distance", "source_route_quality"],
        "sources": sources,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "shapes": {field: list(value.shape) for field, value in arrays.items()},
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "routes": len(candidates), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
