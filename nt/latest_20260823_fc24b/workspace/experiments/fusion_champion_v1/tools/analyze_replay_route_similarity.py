#!/usr/bin/env python3
"""Compare a Replay corpus against a frozen raw-action route bank.

This is a provenance/structure audit, not a claim that a recorded player copied
any public route.  Exact action overlap only establishes shared choreography.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


EPISODE_STEPS = 719
MAX_UNITS = 33
MAX_MARKET_ORDERS = 10
UNIT_NAMES = (
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP", "PICKUP",
    "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE",
)
MARKET_NAMES = (
    "NONE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL",
)
PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
CROPS = PRODUCTS[:5]
ANIMALS = ("GOOSE", "COW", "SHEEP")
SHED_ITEMS = PRODUCTS + ANIMALS
UNIT_OPS = {name: index for index, name in enumerate(UNIT_NAMES)}
MARKET_OPS = {name: index for index, name in enumerate(MARKET_NAMES)}
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


def encode_unit(raw: object) -> tuple[int, int, int]:
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


def encode_market(raw: object) -> tuple[int, int, int]:
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


def compile_replay(path: Path, team: str) -> tuple[dict[str, np.ndarray], dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if len(document.get("steps", [])) != EPISODE_STEPS + 1:
        raise ValueError(f"{path}: expected 720 frames")
    teams = [str(value) for value in document.get("info", {}).get("TeamNames", [])]
    matches = [index for index, value in enumerate(teams) if value == team]
    if len(matches) != 1:
        raise ValueError(f"{path}: team={team!r}, teams={teams}")
    seat = matches[0]
    arrays = {
        "unit_op": np.full((EPISODE_STEPS, MAX_UNITS), UNIT_OPS["PASS"], np.int8),
        "unit_item": np.full((EPISODE_STEPS, MAX_UNITS), -1, np.int8),
        "unit_amount": np.ones((EPISODE_STEPS, MAX_UNITS), np.int32),
        "unit_count": np.ones((EPISODE_STEPS,), np.int8),
        "market_op": np.full((EPISODE_STEPS, MAX_MARKET_ORDERS), MARKET_OPS["NONE"], np.int8),
        "market_item": np.full((EPISODE_STEPS, MAX_MARKET_ORDERS), -1, np.int8),
        "market_amount": np.zeros((EPISODE_STEPS, MAX_MARKET_ORDERS), np.int32),
        "market_count": np.zeros((EPISODE_STEPS,), np.int8),
    }
    shops_by_step: list[list[str]] = []
    for step in range(EPISODE_STEPS):
        observation = document["steps"][step][seat]["observation"]
        shops_by_step.append(
            [str(value) for value in observation.get("town", {}).get("unlocked_shops", [])]
        )
        raw = document["steps"][step + 1][seat].get("action") or {}
        units = [raw.get("farmer", ["PASS"]), *(raw.get("hands", []) or [])]
        arrays["unit_count"][step] = min(len(units), MAX_UNITS)
        for index, action in enumerate(units[:MAX_UNITS]):
            (
                arrays["unit_op"][step, index],
                arrays["unit_item"][step, index],
                arrays["unit_amount"][step, index],
            ) = encode_unit(action)
        orders = raw.get("market", []) or []
        arrays["market_count"][step] = min(len(orders), MAX_MARKET_ORDERS)
        for index, order in enumerate(orders[:MAX_MARKET_ORDERS]):
            (
                arrays["market_op"][step, index],
                arrays["market_item"][step, index],
                arrays["market_amount"][step, index],
            ) = encode_market(order)
    rewards = [int(value) for value in document.get("rewards", [0, 0])]
    meta = {
        "episode_id": int(document.get("info", {}).get("EpisodeId", path.stem)),
        "seat": seat,
        "opponent": teams[1 - seat],
        "reward": rewards[seat],
        "opponent_reward": rewards[1 - seat],
        "won": rewards[seat] > rewards[1 - seat],
        "shops": shops_by_step[-1],
    }
    return arrays, meta


def prefix_length(mask: np.ndarray) -> int:
    mismatches = np.flatnonzero(~mask)
    return int(mismatches[0]) if mismatches.size else int(mask.size)


def action_summary(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for step in range(EPISODE_STEPS):
        for index in range(int(arrays["unit_count"][step])):
            op = int(arrays["unit_op"][step, index])
            if op == UNIT_OPS["PASS"]:
                continue
            label = f"UNIT:{UNIT_NAMES[op]}"
            item = int(arrays["unit_item"][step, index])
            if op == UNIT_OPS["PLANT"] and 0 <= item < len(CROPS):
                label += f":{CROPS[item]}"
            elif op in (UNIT_OPS["PICKUP"], UNIT_OPS["PLACE"]) and 0 <= item < len(SHED_ITEMS):
                label += f":{SHED_ITEMS[item]}"
            counts[label] += 1
        for index in range(int(arrays["market_count"][step])):
            op = int(arrays["market_op"][step, index])
            if op == MARKET_OPS["NONE"]:
                continue
            label = f"MARKET:{MARKET_NAMES[op]}"
            item = int(arrays["market_item"][step, index])
            if op == MARKET_OPS["BUY_SEED"] and 0 <= item < len(CROPS):
                label += f":{CROPS[item]}"
            elif op == MARKET_OPS["BUY_ANIMAL"] and len(PRODUCTS) <= item < len(PRODUCTS) + len(ANIMALS):
                label += f":{ANIMALS[item - len(PRODUCTS)]}"
            elif op in (MARKET_OPS["BUY_PRODUCT"], MARKET_OPS["SELL"]) and 0 <= item < len(PRODUCTS):
                label += f":{PRODUCTS[item]}"
            amount = int(arrays["market_amount"][step, index])
            if op in (MARKET_OPS["HIRE"], MARKET_OPS["BUY_LAND"]):
                counts[label] += 1
            else:
                counts[label] += amount
    return dict(sorted(counts.items()))


def summary_delta(recorded: dict[str, int], reference: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(recorded) | set(reference))
    return {
        key: recorded.get(key, 0) - reference.get(key, 0)
        for key in keys
        if recorded.get(key, 0) != reference.get(key, 0)
    }


def compare(arrays: dict[str, np.ndarray], bank: dict[str, np.ndarray]) -> list[dict]:
    unit_match = (
        np.all(bank["unit_op"] == arrays["unit_op"][None, ...], axis=2)
        & np.all(bank["unit_item"] == arrays["unit_item"][None, ...], axis=2)
        & np.all(bank["unit_amount"] == arrays["unit_amount"][None, ...], axis=2)
        & (bank["unit_count"] == arrays["unit_count"][None, ...])
    )
    market_match = (
        np.all(bank["market_op"] == arrays["market_op"][None, ...], axis=2)
        & np.all(bank["market_item"] == arrays["market_item"][None, ...], axis=2)
        & np.all(bank["market_amount"] == arrays["market_amount"][None, ...], axis=2)
        & (bank["market_count"] == arrays["market_count"][None, ...])
    )
    complete = unit_match & market_match
    rows = []
    for route_id in range(complete.shape[0]):
        rows.append(
            {
                "route_id": route_id,
                "complete_steps": int(np.sum(complete[route_id])),
                "unit_steps": int(np.sum(unit_match[route_id])),
                "market_steps": int(np.sum(market_match[route_id])),
                "exact_prefix_steps": prefix_length(complete[route_id]),
                "day_complete_steps": [
                    int(np.sum(complete[route_id, day * 24 : min((day + 1) * 24, EPISODE_STEPS)]))
                    for day in range(30)
                ],
            }
        )
    return sorted(rows, key=lambda row: (-row["complete_steps"], -row["unit_steps"], row["route_id"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    replay_paths = sorted(args.replay_dir.resolve().glob("*.json"))
    if not replay_paths:
        raise ValueError("no replay JSON files found")
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    skeletons = {int(row["skeleton_id"]): row for row in manifest["skeletons"]}
    with np.load(args.bank.resolve(), allow_pickle=False) as handle:
        bank = {name: handle[name] for name in handle.files}
    if bank["unit_op"].shape[0] != len(skeletons):
        raise ValueError("bank/manifest route count mismatch")
    route_summaries = {
        route_id: action_summary({name: value[route_id] for name, value in bank.items()})
        for route_id in skeletons
    }

    episodes = []
    best_counter: Counter[int] = Counter()
    input_hashes = {}
    for replay_path in replay_paths:
        arrays, meta = compile_replay(replay_path, args.team)
        comparisons = compare(arrays, bank)
        for row in comparisons[: args.top_k]:
            route = skeletons[row["route_id"]]
            row.update(
                opponent=route["opponent"],
                route=route["route"],
                action_sha256=route["action_sha256"],
            )
        best = comparisons[0]
        best_counter[int(best["route_id"])] += 1
        recorded_summary = action_summary(arrays)
        reference_summary = route_summaries[int(best["route_id"])]
        meta.update(
            replay=str(replay_path),
            replay_sha256=sha256(replay_path),
            top_matches=comparisons[: args.top_k],
            recorded_action_summary=recorded_summary,
            best_route_action_summary=reference_summary,
            recorded_minus_best_route=summary_delta(recorded_summary, reference_summary),
        )
        episodes.append(meta)
        input_hashes[str(replay_path)] = meta["replay_sha256"]

    payload = {
        "schema": "kaggriculture.fusion_champion.replay-route-similarity.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "truth_boundary": (
            "Raw action overlap establishes shared choreography only; it does not prove copying, "
            "runtime equivalence, causal route selection, or deployable recovery behavior."
        ),
        "team": args.team,
        "replay_count": len(episodes),
        "bank": str(args.bank.resolve()),
        "bank_sha256": sha256(args.bank.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest.resolve()),
        "best_route_histogram": [
            {
                "route_id": route_id,
                "episodes": count,
                "opponent": skeletons[route_id]["opponent"],
                "route": skeletons[route_id]["route"],
            }
            for route_id, count in best_counter.most_common()
        ],
        "episodes": episodes,
        "input_sha256": input_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "replay_count": payload["replay_count"],
        "best_route_histogram": payload["best_route_histogram"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
