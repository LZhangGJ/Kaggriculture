#!/usr/bin/env python3
"""Audit Rank-1 replays for generic candidate-generation capabilities.

This tool deliberately does not reconstruct a raw 720-action policy.  It
extracts only observable macro decisions, transaction ordering, concurrent
work, spatial layouts, and weed-recovery behavior.  Any association between a
public checkpoint and a later route family is labelled as association rather
than a claim about the source agent's hidden implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


TEAM = "Crop Dusta"
SUBMISSION_ID = 55829779
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PROJECTS = CROPS + ANIMALS
PRODUCTS = CROPS + ("EGG", "MILK", "WOOL", "FERTILIZER")
SHOPS = (
    "BAKERY",
    "PIZZA_SHOP",
    "BRUNCH_SPOT",
    "YARN_STORE",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "SMOOTHIE_SHOP",
    "FARMERS_MARKET",
)
CHECKPOINT_DAYS = (0, 4, 9, 14, 19, 24, 29)
PHASES = ((0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29))
PRODUCTIVE_OPS = {
    "BUILD_PASTURE",
    "BUILD_COOP",
    "DIG",
    "PLACE",
    "PLANT",
    "WATER",
    "FERTILIZE",
    "FEED",
    "CARE",
    "HARVEST",
    "COLLECT_FERTILIZER",
    "DROP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--team", default=TEAM)
    parser.add_argument("--submission-id", type=int, default=SUBMISSION_ID)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_replay(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def q(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"n": 0, "min": None, "p10": None, "median": None, "mean": None, "p90": None, "max": None}
    return {
        "n": len(materialized),
        "min": min(materialized),
        "p10": q(materialized, 0.10),
        "median": q(materialized, 0.50),
        "mean": float(np.mean(materialized)),
        "p90": q(materialized, 0.90),
        "max": max(materialized),
    }


def tile_summary(farm: Mapping[str, Any]) -> dict[str, Any]:
    crop_count = Counter({crop: 0 for crop in CROPS})
    animal_count = Counter({animal: 0 for animal in ANIMALS})
    crop_yield = Counter({crop: 0 for crop in CROPS})
    animal_yield = Counter({animal: 0 for animal in ANIMALS})
    coordinates: dict[str, list[list[int]]] = {project: [] for project in PROJECTS}
    weeds: list[list[int]] = []
    empty_pastures = 0
    empty_coops = 0
    tiles = farm.get("tiles", ()) or ()
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if not isinstance(tile, Mapping):
                continue
            kind = str(tile.get("kind", ""))
            if kind == "PLANT":
                crop = str(tile.get("crop", ""))
                if crop in crop_count:
                    crop_count[crop] += 1
                    crop_yield[crop] += int(tile.get("yield_units", 0) or 0)
                    coordinates[crop].append([x, y])
            elif kind in ("PASTURE", "COOP"):
                animal = tile.get("animal")
                if animal in animal_count:
                    animal_count[str(animal)] += 1
                    animal_yield[str(animal)] += int(tile.get("yield_units", 0) or 0)
                    coordinates[str(animal)].append([x, y])
                elif kind == "PASTURE":
                    empty_pastures += 1
                else:
                    empty_coops += 1
            elif kind == "WEED":
                weeds.append([x, y])
    return {
        "crop_count": dict(crop_count),
        "animal_count": dict(animal_count),
        "crop_yield": dict(crop_yield),
        "animal_yield": dict(animal_yield),
        "coordinates": coordinates,
        "weeds": weeds,
        "empty_pastures": empty_pastures,
        "empty_coops": empty_coops,
    }


def farm_snapshot(observation: Mapping[str, Any], seat: int) -> dict[str, Any]:
    farm = (observation.get("farms", ()) or ())[seat]
    tiles = tile_summary(farm)
    market = observation.get("market", {}) or {}
    town = observation.get("town", {}) or {}
    return {
        "money": float(farm.get("money", 0) or 0),
        "hands": len(farm.get("hands", ()) or ()),
        "hires_today": int(farm.get("hires_today", 0) or 0),
        "land": len(farm.get("unlocked_quadrants", ()) or ()),
        "farmer": list(farm.get("farmer", ()) or ()),
        "crop_count": tiles["crop_count"],
        "animal_count": tiles["animal_count"],
        "crop_yield": tiles["crop_yield"],
        "animal_yield": tiles["animal_yield"],
        "coordinates": tiles["coordinates"],
        "weeds": tiles["weeds"],
        "empty_pastures": tiles["empty_pastures"],
        "empty_coops": tiles["empty_coops"],
        "prices": {item: float((market.get("prices", {}) or {}).get(item, 0) or 0) for item in PRODUCTS},
        "market_inventory": {
            item: float((market.get("inventory", {}) or {}).get(item, 0) or 0) for item in PRODUCTS
        },
        "shops": list(town.get("unlocked_shops", ()) or ()),
    }


def private_snapshot(observation: Mapping[str, Any]) -> dict[str, Any]:
    private = observation.get("private", {}) or {}
    shed = private.get("shed", {}) or {}
    inventories = private.get("inventories", ()) or ()
    carried = Counter()
    for inventory in inventories:
        if isinstance(inventory, Mapping):
            carried.update({str(key): int(value or 0) for key, value in inventory.items()})
    return {
        "seeds": {crop: int((private.get("seeds", {}) or {}).get(crop, 0) or 0) for crop in CROPS},
        "shed": {item: int(shed.get(item, 0) or 0) for item in tuple(PRODUCTS) + ANIMALS},
        "carried": dict(carried),
    }


def action_token(action: Any) -> str:
    if isinstance(action, list) and action:
        return str(action[0])
    return "MISSING"


def market_quantity(order: Sequence[Any]) -> int:
    if not order:
        return 0
    op = str(order[0])
    if op == "HIRE" or op == "BUY_LAND":
        return 1
    if len(order) >= 3:
        try:
            return int(order[2])
        except (TypeError, ValueError):
            return 0
    return 0


def market_project(order: Sequence[Any]) -> str | None:
    if len(order) < 2:
        return None
    op = str(order[0])
    item = str(order[1])
    if op == "BUY_SEED" and item in CROPS:
        return item
    if op == "BUY_ANIMAL" and item in ANIMALS:
        return item
    return None


def public_environment_features(episode: Mapping[str, Any], day: int) -> tuple[list[str], list[float]]:
    own = episode["day_end"][str(day)]["own"]
    opp = episode["day_end"][str(day)]["opponent"]
    names = ["seat", "own_money", "own_land", "own_weeds", "opp_money", "opp_land", "opp_weeds"]
    values = [
        float(episode["seat"]),
        own["money"],
        own["land"],
        len(own["weeds"]),
        opp["money"],
        opp["land"],
        len(opp["weeds"]),
    ]
    for project in PROJECTS:
        names.append(f"opp_{project}")
        source = opp["crop_count"] if project in CROPS else opp["animal_count"]
        values.append(float(source[project]))
    for item in PRODUCTS:
        names.append(f"price_{item}")
        values.append(float(own["prices"][item]))
        names.append(f"market_inventory_{item}")
        values.append(float(own["market_inventory"][item]))
    shop_counts = Counter(own["shops"])
    for shop in SHOPS:
        names.append(f"shop_{shop}")
        values.append(float(shop_counts[shop]))
    return names, values


def route_vector(episode: Mapping[str, Any]) -> np.ndarray:
    values: list[float] = []
    for start, end in PHASES:
        for project in PROJECTS:
            values.append(sum(episode["daily_project_buys"][str(day)][project] for day in range(start, end + 1)))
        values.append(sum(episode["daily_hires"][str(day)] for day in range(start, end + 1)))
        values.append(sum(episode["daily_land"][str(day)] for day in range(start, end + 1)))
    return np.asarray(values, dtype=np.float64)


def project_total(
    episode: Mapping[str, Any], project: str, start_day: int = 0, end_day: int = 29
) -> int:
    return sum(
        int(episode["daily_project_buys"][str(day)][project])
        for day in range(start_day, end_day + 1)
    )


def choose_clusters(vectors: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    scaled = StandardScaler().fit_transform(vectors)
    candidates: list[dict[str, Any]] = []
    best: tuple[float, np.ndarray, int] | None = None
    for clusters in range(2, min(7, len(vectors) - 1)):
        model = KMeans(n_clusters=clusters, random_state=82929, n_init=30)
        labels = model.fit_predict(scaled)
        counts = Counter(int(value) for value in labels)
        if min(counts.values()) < 4:
            continue
        score = float(silhouette_score(scaled, labels))
        candidates.append({"clusters": clusters, "silhouette": score, "counts": dict(sorted(counts.items()))})
        if best is None or score > best[0]:
            best = (score, labels, clusters)
    if best is None:
        model = KMeans(n_clusters=2, random_state=82929, n_init=30)
        labels = model.fit_predict(scaled)
        best = (float(silhouette_score(scaled, labels)), labels, 2)
    return best[1], {
        "selected_clusters": best[2],
        "selected_silhouette": best[0],
        "candidates": candidates,
        "boundary": "clusters summarize observed macro-action families; they are not hidden source-code modes",
    }


def branch_association(episodes: Sequence[Mapping[str, Any]], labels: np.ndarray, day: int) -> dict[str, Any]:
    names, _ = public_environment_features(episodes[0], day)
    features = np.asarray([public_environment_features(episode, day)[1] for episode in episodes], dtype=np.float64)
    counts = Counter(int(value) for value in labels)
    folds = min(5, min(counts.values()))
    if folds < 2:
        return {"day": day, "status": "INSUFFICIENT_CLUSTER_SUPPORT", "label_counts": dict(counts)}
    model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=82929,
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=82929)
    prediction = cross_val_predict(model, features, labels, cv=cv)
    model.fit(features, labels)
    importance = sorted(
        ((names[index], float(value)) for index, value in enumerate(model.feature_importances_) if value > 0),
        key=lambda pair: -pair[1],
    )
    return {
        "day": day,
        "status": "PASS",
        "balanced_accuracy_cv": float(balanced_accuracy_score(labels, prediction)),
        "label_counts": dict(sorted(counts.items())),
        "top_public_associations": [
            {"feature": name, "importance": value} for name, value in importance[:10]
        ],
        "boundary": "association only; no causal or hidden-policy claim",
    }


def pairwise_jaccard(sets: Sequence[set[tuple[int, int]]]) -> float | None:
    scores: list[float] = []
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            union = sets[left] | sets[right]
            if union:
                scores.append(len(sets[left] & sets[right]) / len(union))
    return float(np.mean(scores)) if scores else None


def inspect_episode(path: Path, team: str) -> dict[str, Any]:
    replay = read_replay(path)
    teams = list((replay.get("info", {}) or {}).get("TeamNames", ()) or ())
    if teams.count(team) != 1:
        raise ValueError(f"{path.name}: expected exactly one {team!r}, found {teams!r}")
    seat = teams.index(team)
    opponent = teams[1 - seat]
    steps = replay.get("steps", ()) or ()
    if len(steps) != 720:
        raise ValueError(f"{path.name}: expected 720 frames, found {len(steps)}")

    daily_project_buys = {str(day): Counter({project: 0 for project in PROJECTS}) for day in range(30)}
    daily_product_buys = {str(day): Counter({item: 0 for item in PRODUCTS}) for day in range(30)}
    daily_sells = {str(day): Counter({item: 0 for item in PRODUCTS}) for day in range(30)}
    daily_hires = {str(day): 0 for day in range(30)}
    daily_land = {str(day): 0 for day in range(30)}
    daily_unit_ops = {str(day): Counter() for day in range(30)}
    daily_market_slots = {str(day): [] for day in range(30)}
    day_end: dict[str, Any] = {}
    market_frames: list[dict[str, Any]] = []
    quantity_events: list[dict[str, Any]] = []
    concurrent_frames: list[dict[str, Any]] = []
    weed_open: dict[tuple[int, int], int] = {}
    weed_latencies: list[int] = []
    last_weed_coordinates: set[tuple[int, int]] = set()
    win_reward = float((replay.get("rewards", ()) or (0, 0))[seat])
    opp_reward = float((replay.get("rewards", ()) or (0, 0))[1 - seat])

    for frame_index, rows in enumerate(steps):
        row = rows[seat]
        observation = row.get("observation", {}) or {}
        if not observation:
            continue
        day = int(observation.get("day", frame_index // 24) or 0)
        hour = int(observation.get("hour", frame_index % 24) or 0)
        action = row.get("action", {}) or {}

        unit_actions = [action.get("farmer")] + list(action.get("hands", ()) or ())
        productive = []
        for unit_index, unit_action in enumerate(unit_actions):
            op = action_token(unit_action)
            daily_unit_ops[str(day)][op] += 1
            if op in PRODUCTIVE_OPS:
                productive.append({"unit": unit_index, "op": op})
        productive_types = sorted({item["op"] for item in productive})
        if len(productive_types) >= 3:
            concurrent_frames.append(
                {"step": frame_index, "day": day, "hour": hour, "types": productive_types, "units": len(productive)}
            )

        orders = list(action.get("market", ()) or ())
        daily_market_slots[str(day)].append(len(orders))
        if orders:
            sequence = []
            has_sell = False
            has_investment = False
            for index, order in enumerate(orders):
                if not isinstance(order, list) or not order:
                    continue
                op = str(order[0])
                quantity = market_quantity(order)
                item = str(order[1]) if len(order) > 1 else None
                sequence.append({"index": index, "op": op, "item": item, "quantity": quantity})
                if quantity not in (0, 2, 4):
                    quantity_events.append(
                        {"step": frame_index, "day": day, "hour": hour, "op": op, "item": item, "quantity": quantity}
                    )
                project = market_project(order)
                if project is not None:
                    daily_project_buys[str(day)][project] += quantity
                    has_investment = True
                if op == "BUY_PRODUCT" and item in PRODUCTS:
                    daily_product_buys[str(day)][item] += quantity
                elif op == "SELL" and item in PRODUCTS:
                    daily_sells[str(day)][item] += quantity
                    has_sell = True
                elif op == "HIRE":
                    daily_hires[str(day)] += 1
                    has_investment = True
                elif op == "BUY_LAND":
                    daily_land[str(day)] += 1
                    has_investment = True
            market_frames.append(
                {
                    "step": frame_index,
                    "day": day,
                    "hour": hour,
                    "slots": len(sequence),
                    "has_sell": has_sell,
                    "has_investment": has_investment,
                    "sequence": sequence,
                }
            )

        own_snapshot = farm_snapshot(observation, seat)
        current_weeds = {tuple(coord) for coord in own_snapshot["weeds"]}
        for coordinate in current_weeds - last_weed_coordinates:
            weed_open[coordinate] = frame_index
        for coordinate in last_weed_coordinates - current_weeds:
            opened = weed_open.pop(coordinate, None)
            if opened is not None:
                weed_latencies.append(frame_index - opened)
        last_weed_coordinates = current_weeds

        if hour == 23 or frame_index == len(steps) - 1:
            day_end[str(day)] = {
                "own": own_snapshot,
                "opponent": farm_snapshot(observation, 1 - seat),
                "private": private_snapshot(observation),
            }

    if sorted(int(day) for day in day_end) != list(range(30)):
        raise ValueError(f"{path.name}: missing day-end snapshots: {sorted(day_end)}")
    for coordinate, opened in weed_open.items():
        weed_latencies.append(720 - opened)

    daily_distinct_projects = {
        str(day): sum(int(quantity > 0) for quantity in daily_project_buys[str(day)].values())
        for day in range(30)
    }
    final = day_end["29"]["own"]
    return {
        "episode_id": int((replay.get("info", {}) or {}).get("EpisodeId", path.stem)),
        "file": str(path.resolve()),
        "sha256": sha256(path),
        "seed": int((replay.get("info", {}) or {}).get("seed", -1) or -1),
        "seat": seat,
        "opponent": opponent,
        "reward": win_reward,
        "opponent_reward": opp_reward,
        "won": bool(win_reward > opp_reward),
        "margin": win_reward - opp_reward,
        "daily_project_buys": {day: dict(values) for day, values in daily_project_buys.items()},
        "daily_product_buys": {day: dict(values) for day, values in daily_product_buys.items()},
        "daily_sells": {day: dict(values) for day, values in daily_sells.items()},
        "daily_hires": daily_hires,
        "daily_land": daily_land,
        "daily_unit_ops": {day: dict(values) for day, values in daily_unit_ops.items()},
        "daily_market_slots": daily_market_slots,
        "daily_distinct_projects": daily_distinct_projects,
        "day_end": day_end,
        "market_frames": market_frames,
        "quantity_events": quantity_events,
        "concurrent_frames": concurrent_frames,
        "weed_latencies": weed_latencies,
        "unresolved_weeds_at_terminal": len(last_weed_coordinates),
        "final": final,
    }


def main() -> int:
    args = parse_args()
    paths = sorted(args.episodes_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"no JSON replays in {args.episodes_dir}")
    episodes = []
    for index, path in enumerate(paths, start=1):
        episodes.append(inspect_episode(path, args.team))
        if index % 10 == 0 or index == len(paths):
            print(f"parsed {index}/{len(paths)}", flush=True)

    vectors = np.stack([route_vector(episode) for episode in episodes])
    labels, clustering = choose_clusters(vectors)
    for episode, label in zip(episodes, labels, strict=True):
        episode["macro_family"] = int(label)

    cluster_summaries = []
    for label in sorted(set(int(value) for value in labels)):
        members = [episode for episode in episodes if episode["macro_family"] == label]
        project_totals = {
            project: describe(
                sum(member["daily_project_buys"][str(day)][project] for day in range(30))
                for member in members
            )
            for project in PROJECTS
        }
        first_investment_day = {}
        for project in PROJECTS:
            days = []
            for member in members:
                active = [day for day in range(30) if member["daily_project_buys"][str(day)][project] > 0]
                if active:
                    days.append(min(active))
            first_investment_day[project] = describe(days)
        cluster_summaries.append(
            {
                "family": label,
                "episodes": len(members),
                "wins": sum(int(member["won"]) for member in members),
                "reward": describe(member["reward"] for member in members),
                "margin": describe(member["margin"] for member in members),
                "project_total_purchases": project_totals,
                "first_investment_day": first_investment_day,
                "hire_total": describe(sum(member["daily_hires"].values()) for member in members),
                "land_total": describe(sum(member["daily_land"].values()) for member in members),
                "example_episode_ids": [member["episode_id"] for member in members[:5]],
            }
        )

    all_quantity_events = [event for episode in episodes for event in episode["quantity_events"]]
    order_quantities = Counter(
        (event["op"], str(event["item"]), int(event["quantity"])) for event in all_quantity_events
    )
    daily_quantity_distribution = Counter()
    daily_project_combinations = Counter()
    single_project_days = 0
    multi_project_days = 0
    pure_expansion_late_days = 0
    coupled_capacity_days = 0
    for episode in episodes:
        for day in range(30):
            buys = episode["daily_project_buys"][str(day)]
            active = [project for project, quantity in buys.items() if quantity > 0]
            if active:
                daily_project_combinations[tuple(active)] += 1
            if len(active) == 1:
                single_project_days += 1
                if day > 0:
                    pure_expansion_late_days += 1
            if len(active) >= 3:
                multi_project_days += 1
            if (episode["daily_hires"][str(day)] or episode["daily_land"][str(day)]) and len(active) >= 2:
                coupled_capacity_days += 1
            for project, quantity in buys.items():
                if quantity > 0:
                    daily_quantity_distribution[(project, int(quantity))] += 1

    finance_frames = [
        frame
        for episode in episodes
        for frame in episode["market_frames"]
        if frame["has_sell"] and frame["has_investment"]
    ]
    frame_project_combinations = Counter()
    capacity_project_frames = 0
    for episode in episodes:
        for frame in episode["market_frames"]:
            projects = tuple(sorted({
                str(order["item"])
                for order in frame["sequence"]
                if (
                    (order["op"] == "BUY_SEED" and order["item"] in CROPS)
                    or (order["op"] == "BUY_ANIMAL" and order["item"] in ANIMALS)
                )
            }))
            if projects:
                frame_project_combinations[projects] += 1
            has_capacity = any(order["op"] in ("HIRE", "BUY_LAND") for order in frame["sequence"])
            if has_capacity and len(projects) >= 2:
                capacity_project_frames += 1
    max_market_slots = max(frame["slots"] for episode in episodes for frame in episode["market_frames"])
    concurrent = [frame for episode in episodes for frame in episode["concurrent_frames"]]
    weed_latencies = [latency for episode in episodes for latency in episode["weed_latencies"]]

    workforce_curve = {}
    land_curve = {}
    for day in range(30):
        workforce_curve[str(day)] = describe(episode["daily_hires"][str(day)] for episode in episodes)
        land_curve[str(day)] = describe(episode["daily_land"][str(day)] for episode in episodes)

    project_statistics = {}
    for project in PROJECTS:
        totals = [project_total(episode, project) for episode in episodes]
        first_days: list[int] = []
        last_days: list[int] = []
        for episode in episodes:
            active_days = [
                day for day in range(30)
                if episode["daily_project_buys"][str(day)][project] > 0
            ]
            if active_days:
                first_days.append(min(active_days))
                last_days.append(max(active_days))
        observed_daily_quantities = sorted({
            quantity
            for (candidate_project, quantity), count in daily_quantity_distribution.items()
            if candidate_project == project and count > 0
        })
        project_statistics[project] = {
            "episodes_active": sum(int(value > 0) for value in totals),
            "total_purchased": describe(totals),
            "first_purchase_day": describe(first_days),
            "last_purchase_day": describe(last_days),
            "observed_daily_quantities": observed_daily_quantities,
        }

    sell_timing = {}
    for item in PRODUCTS:
        totals = []
        first_days = []
        last_days = []
        active_day_counts = []
        inventory_hold_days = 0
        for episode in episodes:
            daily = [int(episode["daily_sells"][str(day)][item]) for day in range(30)]
            active = [day for day, quantity in enumerate(daily) if quantity > 0]
            totals.append(sum(daily))
            active_day_counts.append(len(active))
            if active:
                first_days.append(min(active))
                last_days.append(max(active))
            for day in range(29):
                private = episode["day_end"][str(day)]["private"]
                held = int(private["shed"].get(item, 0)) + int(private["carried"].get(item, 0))
                if held > 0 and daily[day] == 0:
                    inventory_hold_days += 1
        sell_timing[item] = {
            "episodes_selling": sum(int(total > 0) for total in totals),
            "total_sold": describe(totals),
            "first_sell_day": describe(first_days),
            "last_sell_day": describe(last_days),
            "selling_days": describe(active_day_counts),
            "observed_hold_days_with_inventory_and_no_sale": inventory_hold_days,
        }

    branch_specs = {
        "post_day4_sheep_expansion_ge6": (
            np.asarray([int(project_total(episode, "SHEEP", 5, 19) >= 6) for episode in episodes], dtype=np.int32),
            (4, 9),
        ),
        "post_day4_cow_expansion_ge5": (
            np.asarray([int(project_total(episode, "COW", 5, 19) >= 5) for episode in episodes], dtype=np.int32),
            (4, 9),
        ),
        "goose_branch_days5_19": (
            np.asarray([int(project_total(episode, "GOOSE", 5, 19) >= 1) for episode in episodes], dtype=np.int32),
            (4, 9),
        ),
        "tomato_branch_days15_24": (
            np.asarray([int(project_total(episode, "TOMATO", 15, 24) >= 1) for episode in episodes], dtype=np.int32),
            (9, 14),
        ),
        "large_carrot_suffix_days20_29": (
            np.asarray([int(project_total(episode, "CARROT", 20, 29) >= 30) for episode in episodes], dtype=np.int32),
            (14, 19),
        ),
    }
    decision_branch_associations = {}
    for name, (branch_labels, days) in branch_specs.items():
        decision_branch_associations[name] = {
            "positive": int(np.count_nonzero(branch_labels)),
            "negative": int(len(branch_labels) - np.count_nonzero(branch_labels)),
            "checkpoints": [branch_association(episodes, branch_labels, day) for day in days],
        }

    layout = {}
    for day in CHECKPOINT_DAYS:
        layout[str(day)] = {}
        for project in PROJECTS:
            sets = [
                {tuple(coord) for coord in episode["day_end"][str(day)]["own"]["coordinates"][project]}
                for episode in episodes
            ]
            radii = [
                abs(x - 4) + abs(y - 4)
                for coordinates in sets
                for x, y in coordinates
            ]
            layout[str(day)][project] = {
                "episodes_with_project": sum(int(bool(coordinates)) for coordinates in sets),
                "unique_layouts": len({tuple(sorted(coordinates)) for coordinates in sets}),
                "mean_pairwise_jaccard": pairwise_jaccard(sets),
                "distance_from_start_hub": describe(radii),
            }

    payload = {
        "schema": "kaggriculture.rank1-candidate-capability-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "team": args.team,
        "submission_id": args.submission_id,
        "source": str(args.episodes_dir.resolve()),
        "evidence_boundary": {
            "direct": "actions, public/own observations, terminal rewards, transaction order, coordinates",
            "association": "macro-family clustering and public-state branch predictability",
            "not_claimed": "hidden source code, hidden opponent inventory, future knowledge, causal trigger rules",
        },
        "validation": {
            "episodes": len(episodes),
            "frames_per_episode": 720,
            "official_version_expected": "1.32.7",
            "seat_counts": dict(sorted(Counter(episode["seat"] for episode in episodes).items())),
            "wins": sum(int(episode["won"]) for episode in episodes),
            "losses": sum(int(not episode["won"]) for episode in episodes),
            "reward": describe(episode["reward"] for episode in episodes),
            "margin": describe(episode["margin"] for episode in episodes),
        },
        "macro_families": {
            "clustering": clustering,
            "families": cluster_summaries,
            "public_state_association": [branch_association(episodes, labels, day) for day in (0, 4, 9, 14, 19)],
        },
        "candidate_grammar_evidence": {
            "single_project_investment_days": single_project_days,
            "single_project_late_investment_days": pure_expansion_late_days,
            "three_or_more_project_investment_days": multi_project_days,
            "capacity_plus_two_or_more_project_days": coupled_capacity_days,
            "capacity_plus_two_or_more_project_frames": capacity_project_frames,
            "project_statistics": project_statistics,
            "top_daily_project_combinations": [
                {"projects": list(projects), "count": count}
                for projects, count in daily_project_combinations.most_common(20)
            ],
            "top_same_frame_project_combinations": [
                {"projects": list(projects), "count": count}
                for projects, count in frame_project_combinations.most_common(20)
            ],
            "daily_project_quantity_distribution": [
                {"project": project, "quantity": quantity, "count": count}
                for (project, quantity), count in sorted(daily_quantity_distribution.items())
            ],
            "non_2_4_order_quantity_distribution": [
                {"op": op, "item": item, "quantity": quantity, "count": count}
                for (op, item, quantity), count in sorted(order_quantities.items())
            ],
        },
        "transactions": {
            "market_frames": sum(len(episode["market_frames"]) for episode in episodes),
            "max_slots_observed": max_market_slots,
            "sell_and_invest_same_frame": len(finance_frames),
            "sell_and_invest_examples": finance_frames[:20],
            "workforce_hire_by_day": workforce_curve,
            "land_purchase_by_day": land_curve,
            "sell_timing": sell_timing,
        },
        "decision_branch_associations": decision_branch_associations,
        "coordination": {
            "frames_with_three_or_more_productive_action_types": len(concurrent),
            "episodes_with_such_concurrency": sum(int(bool(episode["concurrent_frames"])) for episode in episodes),
            "examples": concurrent[:20],
        },
        "layout": layout,
        "deviation_recovery": {
            "weed_occurrences": len(weed_latencies),
            "weed_visible_duration_frames": describe(weed_latencies),
            "terminal_unresolved_weeds": describe(episode["unresolved_weeds_at_terminal"] for episode in episodes),
            "boundary": "duration measures disappearance from visible board; it does not assume every disappearance was a deliberate DIG",
        },
        "episode_summaries": [
            {
                "episode_id": episode["episode_id"],
                "sha256": episode["sha256"],
                "seed": episode["seed"],
                "seat": episode["seat"],
                "opponent": episode["opponent"],
                "reward": episode["reward"],
                "opponent_reward": episode["opponent_reward"],
                "won": episode["won"],
                "margin": episode["margin"],
                "macro_family": episode["macro_family"],
                "project_total_purchases": {
                    project: sum(episode["daily_project_buys"][str(day)][project] for day in range(30))
                    for project in PROJECTS
                },
                "hire_total": sum(episode["daily_hires"].values()),
                "land_total": sum(episode["daily_land"].values()),
            }
            for episode in episodes
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "episodes": len(episodes), "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
