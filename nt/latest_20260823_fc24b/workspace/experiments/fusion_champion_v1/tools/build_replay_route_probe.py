#!/usr/bin/env python3
"""Compile one player's recorded Replay actions into an isolated JAX route probe.

The artifact is deliberately an experimental potential probe, not a deployable
Agent.  It lets the strict JAX arena answer whether a newly observed production
timeline is economically interesting before any behaviour is promoted into the
rule-based fusion policy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


EPISODE_ACTIONS = 719
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
PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
CROPS = PRODUCTS[:5]
ANIMALS = ("GOOSE", "COW", "SHEEP")
SHED_ITEMS = PRODUCTS + ANIMALS
PRODUCT_ID = {name: index for index, name in enumerate(PRODUCTS)}
CROP_ID = {name: index for index, name in enumerate(CROPS)}
ANIMAL_ID = {name: len(PRODUCTS) + index for index, name in enumerate(ANIMALS)}
SHED_ID = {name: index for index, name in enumerate(SHED_ITEMS)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def encode_unit(raw) -> tuple[int, int, int]:
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


def encode_market(raw) -> tuple[int, int, int]:
    if not isinstance(raw, list) or not raw:
        return MARKET_OPS["NONE"], -1, 0
    op = MARKET_OPS.get(str(raw[0]), MARKET_OPS["NONE"])
    if op in (MARKET_OPS["HIRE"], MARKET_OPS["BUY_LAND"]):
        return int(op), -1, 0
    if len(raw) < 3:
        return MARKET_OPS["NONE"], -1, 0
    if op == MARKET_OPS["BUY_SEED"]:
        item = CROP_ID.get(str(raw[1]), -1)
    elif op == MARKET_OPS["BUY_ANIMAL"]:
        item = ANIMAL_ID.get(str(raw[1]), -1)
    elif op in (MARKET_OPS["BUY_PRODUCT"], MARKET_OPS["SELL"]):
        item = PRODUCT_ID.get(str(raw[1]), -1)
    else:
        return MARKET_OPS["NONE"], -1, 0
    return int(op), int(item), int(raw[2])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--recovery-output",
        type=Path,
        help="Optional nine-branch duplicate bank for Hasegawa V2 state recovery.",
    )
    args = parser.parse_args()

    replay_path = args.replay.resolve()
    document = json.loads(replay_path.read_text(encoding="utf-8"))
    if len(document.get("steps", [])) != EPISODE_ACTIONS + 1:
        raise ValueError("Replay must contain exactly 720 frames")
    teams = [str(value) for value in document.get("info", {}).get("TeamNames", [])]
    matches = [index for index, name in enumerate(teams) if name == args.team]
    if len(matches) != 1:
        raise ValueError(f"team must match exactly once: team={args.team!r}, teams={teams}")
    seat = matches[0]

    unit_op = np.full((1, EPISODE_ACTIONS, MAX_UNITS), UNIT_OPS["PASS"], np.int8)
    unit_item = np.full((1, EPISODE_ACTIONS, MAX_UNITS), -1, np.int8)
    unit_amount = np.ones((1, EPISODE_ACTIONS, MAX_UNITS), np.int32)
    unit_count = np.ones((1, EPISODE_ACTIONS), np.int8)
    market_op = np.full((1, EPISODE_ACTIONS, MAX_MARKET_ORDERS), MARKET_OPS["NONE"], np.int8)
    market_item = np.full((1, EPISODE_ACTIONS, MAX_MARKET_ORDERS), -1, np.int8)
    market_amount = np.zeros((1, EPISODE_ACTIONS, MAX_MARKET_ORDERS), np.int32)
    market_count = np.zeros((1, EPISODE_ACTIONS), np.int8)
    expected_unit_pos = np.zeros((1, EPISODE_ACTIONS, MAX_UNITS, 2), np.int8)
    expected_unit_active = np.zeros((1, EPISODE_ACTIONS, MAX_UNITS), np.bool_)
    expected_money = np.zeros((1, EPISODE_ACTIONS), np.int32)

    for step in range(EPISODE_ACTIONS):
        observation = document["steps"][step][seat]["observation"]
        farm = observation["farms"][seat]
        positions = [farm["farmer"], *(farm.get("hands", []) or [])]
        count = min(len(positions), MAX_UNITS)
        expected_unit_pos[0, step, :count] = np.asarray(
            positions[:MAX_UNITS], dtype=np.int8
        )
        expected_unit_active[0, step, :count] = True
        expected_money[0, step] = int(farm["money"])
        raw = document["steps"][step + 1][seat].get("action") or {}
        units = [raw.get("farmer", ["PASS"]), *(raw.get("hands", []) or [])]
        unit_count[0, step] = min(len(units), MAX_UNITS)
        for index, action in enumerate(units[:MAX_UNITS]):
            (
                unit_op[0, step, index],
                unit_item[0, step, index],
                unit_amount[0, step, index],
            ) = encode_unit(action)
        orders = raw.get("market", []) or []
        market_count[0, step] = min(len(orders), MAX_MARKET_ORDERS)
        for index, order in enumerate(orders[:MAX_MARKET_ORDERS]):
            (
                market_op[0, step, index],
                market_item[0, step, index],
                market_amount[0, step, index],
            ) = encode_market(order)

    arrays = {
        "unit_op": unit_op,
        "unit_item": unit_item,
        "unit_amount": unit_amount,
        "unit_count": unit_count,
        "market_op": market_op,
        "market_item": market_item,
        "market_amount": market_amount,
        "market_count": market_count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    recovery_output = None
    recovery_sha256 = None
    if args.recovery_output is not None:
        args.recovery_output.parent.mkdir(parents=True, exist_ok=True)
        recovery_arrays = {
            name: np.repeat(value, 9, axis=0) for name, value in arrays.items()
        }
        recovery_arrays.update(
            expected_unit_pos=np.repeat(expected_unit_pos, 9, axis=0),
            expected_unit_active=np.repeat(expected_unit_active, 9, axis=0),
            expected_money=np.repeat(expected_money, 9, axis=0),
            source_episode_id=np.full((9,), int(document.get("info", {}).get("EpisodeId", replay_path.stem)), np.int64),
            source_reward=np.full((9,), rewards[seat] if "rewards" in locals() else 0, np.int32),
            source_seat=np.full((9,), seat, np.int8),
            branch_sample_count=np.ones((9,), np.int16),
            branch_shop_demand=np.zeros((9, len(PRODUCTS)), np.int16),
        )
        # rewards is assigned immediately below for the receipt; use the Replay
        # value directly here to keep this build step deterministic.
        recovery_arrays["source_reward"][:] = int(document.get("rewards", [0, 0])[seat])
        np.savez_compressed(args.recovery_output, **recovery_arrays)
        recovery_output = str(args.recovery_output.resolve())
        recovery_sha256 = sha256(args.recovery_output)
    rewards = [int(value) for value in document.get("rewards", [0, 0])]
    receipt = {
        "schema": "kaggriculture.fusion_champion.replay-route-probe.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "truth_boundary": (
            "Experimental Replay route potential probe only. It is not a deployable "
            "Agent and must not be promoted without rule abstraction and independent validation."
        ),
        "source_replay": str(replay_path),
        "source_sha256": sha256(replay_path),
        "episode_id": int(document.get("info", {}).get("EpisodeId", replay_path.stem)),
        "team": args.team,
        "seat": seat,
        "reward": rewards[seat],
        "opponent": teams[1 - seat],
        "opponent_reward": rewards[1 - seat],
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "recovery_output": recovery_output,
        "recovery_output_sha256": recovery_sha256,
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
