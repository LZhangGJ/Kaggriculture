"""Compile Hasegawa winning Replays into a high-level JAX route bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments" / "general_project_planner_v1" / "tools",
    ROOT / "experiments" / "general_project_planner_v1" / "src",
    ROOT / "experiments" / "kaggriculture_execution_core" / "src",
    ROOT / "gpu_sim" / "src",
):
    sys.path.insert(0, str(path))

from compile_reference_obligation_plan import compile_plan  # noqa: E402
from general_project_planner_v1.fulfillment_schema import ObligationPlanV1  # noqa: E402
from kaggriculture_jax.constants import ANIMALS, CROPS, PRODUCTS, SHOP_NAMES  # noqa: E402


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
    [
        [
            (2 if len(SHOP_PRODUCTS[shop]) == 1 else 1)
            if product in SHOP_PRODUCTS[shop]
            else 0
            for product in PRODUCTS
        ]
        for shop in SHOP_NAMES
    ],
    dtype=np.int16,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_daily(path: Path) -> dict[int, dict[int, dict[str, str]]]:
    output: dict[int, dict[int, dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            episode = int(row["episode_id"])
            day = int(row["day"])
            output.setdefault(episode, {})[day] = row
    return output


def numeric(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    return float(value) if value not in (None, "") else default


def reference_arrays(rows: dict[int, dict[int, dict[str, str]]], episodes: list[int]):
    route_count = len(episodes)
    money = np.zeros((route_count, 30), dtype=np.int32)
    hires = np.zeros((route_count, 30), dtype=np.int8)
    unlocked = np.ones((route_count, 30), dtype=np.int8)
    crop = np.zeros((route_count, 30, len(CROPS)), dtype=np.int16)
    animal = np.zeros((route_count, 30, len(ANIMALS)), dtype=np.int16)
    prices = np.zeros((route_count, 30, len(PRODUCTS)), dtype=np.int32)
    demand = np.zeros((route_count, 30, len(PRODUCTS)), dtype=np.int16)

    for route, episode in enumerate(episodes):
        previous: dict[str, str] | None = None
        for day in range(30):
            row = rows.get(episode, {}).get(day, previous)
            if row is None:
                continue
            previous = row
            money[route, day] = int(numeric(row, "money"))
            hires[route, day] = int(numeric(row, "hires_today"))
            unlocked[route, day] = max(1, int(numeric(row, "unlocked_tiles", 25)) // 25)
            crop[route, day] = [int(numeric(row, f"crop_{name}")) for name in CROPS]
            animal[route, day] = [
                int(numeric(row, f"animal_{name}")) for name in ANIMALS
            ]
            prices[route, day] = [
                int(numeric(row, f"market_price_{name}")) for name in PRODUCTS
            ]
            active = np.asarray(
                [int(numeric(row, f"town_shop_{name}")) for name in SHOP_NAMES],
                dtype=np.int16,
            )
            demand[route, day] = active @ SHOP_DEMAND
    return {
        "ref_money": money,
        "ref_hires": hires,
        "ref_unlocked": unlocked,
        "ref_crop": crop,
        "ref_animal": animal,
        "ref_market_price": prices,
        "ref_shop_demand": demand,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    daily_path = args.daily.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    winners = [
        row
        for row in manifest.get("episodes", [])
        if row.get("result") == "WIN" and row.get("download_status") == "downloaded"
    ]
    winners.sort(key=lambda row: int(row["episode_id"]))
    daily = read_daily(daily_path)
    plans: dict[str, list[np.ndarray]] = {name: [] for name in ObligationPlanV1._fields}
    episode_ids: list[int] = []
    rewards: list[int] = []
    source_hashes: list[dict[str, object]] = []
    incomplete: list[int] = []

    for index, row in enumerate(winners, start=1):
        replay_path = Path(row["path"])
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        arrays, observed = compile_plan(replay, int(row["own_seat"]))
        episode = int(row["episode_id"])
        if len(observed) != 30:
            incomplete.append(episode)
            continue
        for field in ObligationPlanV1._fields:
            plans[field].append(np.asarray(arrays[field][0]))
        episode_ids.append(episode)
        rewards.append(int(row["own_reward"]))
        source_hashes.append(
            {
                "episode_id": episode,
                "sha256": str(row.get("sha256") or sha256(replay_path)),
                "seat": int(row["own_seat"]),
                "reward": int(row["own_reward"]),
            }
        )
        if index % 10 == 0 or index == len(winners):
            print(json.dumps({"compiled": index, "total": len(winners)}), flush=True)

    output_arrays = {
        field: np.stack(values, axis=0) for field, values in plans.items()
    }
    output_arrays.update(
        {
            "episode_id": np.asarray(episode_ids, dtype=np.int64),
            "final_reward": np.asarray(rewards, dtype=np.int32),
        }
    )
    output_arrays.update(reference_arrays(daily, episode_ids))
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output_arrays)
    receipt = {
        "schema": "kaggriculture.hasegawa_high_level_plan_bank.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if episode_ids and not incomplete else "FAIL",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "source_daily_states": str(daily_path),
        "source_daily_states_sha256": sha256(daily_path),
        "module_version": "1.32.7",
        "winning_replays_in_manifest": len(winners),
        "compiled_route_count": len(episode_ids),
        "incomplete_episode_ids": incomplete,
        "raw_actions_stored": False,
        "raw_coordinates_stored": False,
        "selection_features": [
            "currently_unlocked_shop_demand",
            "current_market_price",
            "current_money",
            "current_crop_count",
            "current_animal_count",
            "current_hires",
            "current_unlocked_land",
        ],
        "future_event_features_used": False,
        "fields": {name: list(value.shape) for name, value in output_arrays.items()},
        "source_replays": source_hashes,
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }
    receipt_path = args.receipt.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": receipt["status"], "output": str(output_path)}))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
