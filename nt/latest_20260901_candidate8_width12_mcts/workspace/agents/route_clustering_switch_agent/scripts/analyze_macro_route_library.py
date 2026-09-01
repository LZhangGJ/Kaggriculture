#!/usr/bin/env python3
"""Audit replay libraries as a few production/layout macro plans per team."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
CATEGORIES = (*CROPS, *ANIMALS, "COOP", "PASTURE")
CATEGORY_ID = {name: index + 1 for index, name in enumerate(CATEGORIES)}
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
LAND_COST = (1000, 2000, 4000)
ANCHORS = (168, 288, 432, 576, 719)
DAY_STEPS = tuple(range(24, 720, 24)) + (719,)
WINDOW = 72
SCHEDULE_BUCKET = 72


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.12)
    return parser.parse_args()


def _fib(index: int) -> int:
    left = right = 1
    for _ in range(max(0, index)):
        left, right = right, left + right
    return left


def _category(tile: Any) -> int:
    if not isinstance(tile, dict):
        return 0
    if tile.get("crop") in CATEGORY_ID:
        return CATEGORY_ID[str(tile["crop"])]
    if tile.get("animal") in CATEGORY_ID:
        return CATEGORY_ID[str(tile["animal"])]
    return CATEGORY_ID.get(str(tile.get("kind") or ""), 0)


def _snapshot(farm: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    layout = np.zeros(100, dtype=np.int8)
    for y, row in enumerate(list(farm.get("tiles", []) or [])[:10]):
        for x, tile in enumerate(list(row or [])[:10]):
            layout[y * 10 + x] = _category(tile)
    counts = np.asarray(
        [np.count_nonzero(layout == CATEGORY_ID[name]) for name in CATEGORIES]
        + [len(farm.get("unlocked_quadrants", []) or []), len(farm.get("hands", []) or [])],
        dtype=np.int16,
    )
    return counts, layout


def _action_schedule(steps: list[Any], player: int) -> np.ndarray:
    # Ten 72-turn phases: plant crops, build structures, place animals,
    # buy land and hire. Movement and repair details are intentionally omitted.
    result = np.zeros((10, len(CROPS) + 2 + len(ANIMALS) + 2), dtype=np.int16)
    crop_index = {name: index for index, name in enumerate(CROPS)}
    animal_index = {name: len(CROPS) + 2 + index for index, name in enumerate(ANIMALS)}
    for step in range(min(719, len(steps) - 1)):
        action = steps[step + 1][player].get("action") or {}
        phase = min(9, step // SCHEDULE_BUCKET)
        for order in [action.get("farmer"), *(action.get("hands") or [])]:
            row = list(order or [])
            if len(row) >= 2 and row[0] == "PLANT" and row[1] in crop_index:
                result[phase, crop_index[row[1]]] += 1
            elif row and row[0] == "BUILD_COOP":
                result[phase, len(CROPS)] += 1
            elif row and row[0] == "BUILD_PASTURE":
                result[phase, len(CROPS) + 1] += 1
            elif len(row) >= 2 and row[0] == "PLACE" and row[1] in animal_index:
                result[phase, animal_index[row[1]]] += 1
        for order in action.get("market", []) or []:
            row = list(order or [])
            if row and row[0] == "BUY_LAND":
                result[phase, -2] += 1
            elif row and row[0] == "HIRE":
                result[phase, -1] += 1
    return result


def _quoted_flows(steps: list[Any], player: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spend = np.zeros(719, dtype=np.float32)
    revenue = np.zeros(719, dtype=np.float32)
    money = np.zeros(720, dtype=np.float32)
    for step in range(min(720, len(steps))):
        observation = steps[step][player].get("observation") or {}
        farms = list(observation.get("farms", []) or [])
        farm = farms[player] if player < len(farms) else {}
        money[step] = float(farm.get("money", 0.0) or 0.0)
        if step >= 719 or step + 1 >= len(steps):
            continue
        action = steps[step + 1][player].get("action") or {}
        prices = dict((observation.get("market") or {}).get("prices") or {})
        unlocked = max(1, len(farm.get("unlocked_quadrants", []) or []))
        hires = max(0, int(farm.get("hires_today", 0) or 0))
        for raw in action.get("market", []) or []:
            order = list(raw or [])
            if not order:
                continue
            op = str(order[0])
            item = str(order[1]) if len(order) >= 2 else ""
            quantity = max(0, int(order[2] or 0)) if len(order) >= 3 else 1
            if op == "HIRE":
                spend[step] += _fib(hires)
                hires += 1
            elif op == "BUY_LAND" and unlocked - 1 < len(LAND_COST):
                spend[step] += LAND_COST[unlocked - 1]
                unlocked += 1
            elif op == "BUY_SEED" and item in SEED_COST:
                spend[step] += quantity * SEED_COST[item]
            elif op == "BUY_ANIMAL" and item in ANIMAL_COST:
                spend[step] += quantity * ANIMAL_COST[item]
            elif op == "BUY_PRODUCT":
                spend[step] += quantity * float(prices.get(item, 0.0) or 0.0)
            elif op == "SELL":
                revenue[step] += quantity * float(prices.get(item, 0.0) or 0.0)
    return spend, revenue, money


def _capital_profile(spend: np.ndarray, revenue: np.ndarray, money: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    requirement = np.zeros(len(DAY_STEPS), dtype=np.float32)
    slack = np.zeros(len(DAY_STEPS), dtype=np.float32)
    flow = spend - revenue
    for index, start in enumerate((0, *range(24, 696, 24), 696)):
        stop = min(len(flow), start + WINDOW)
        cumulative = np.cumsum(flow[start:stop])
        requirement[index] = max(0.0, float(np.max(cumulative, initial=0.0)))
        slack[index] = float(money[min(start, len(money) - 1)] - requirement[index])
    return requirement, slack


def _extract(task: tuple[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    import orjson

    replay_path, targets = task
    replay = orjson.loads(Path(replay_path).read_bytes())
    steps = replay.get("steps", [])
    output = []
    for target in targets:
        player = int(target["player_index"])
        production = []
        layouts = []
        anchor_set = set(ANCHORS)
        for step in sorted(set(DAY_STEPS) | anchor_set):
            observation = steps[min(step, len(steps) - 1)][player].get("observation") or {}
            farms = list(observation.get("farms", []) or [])
            farm = farms[player] if player < len(farms) else {}
            counts, layout = _snapshot(farm)
            if step in DAY_STEPS:
                production.append(counts)
            if step in anchor_set:
                layouts.append(layout)
        schedule = _action_schedule(steps, player)
        spend, revenue, money = _quoted_flows(steps, player)
        capital, slack = _capital_profile(spend, revenue, money)
        output.append({
            "episode_id": int(replay.get("info", {}).get("EpisodeId") or target["episode_id"]),
            "player_index": player,
            "team": str(target["team_name"]),
            "score": float(target.get("leaderboard_score", 0.0) or 0.0),
            "reward": float(target.get("final_reward", 0.0) or 0.0),
            "production": np.stack(production), "layouts": np.stack(layouts),
            "schedule": schedule, "capital": capital, "slack": slack,
            "min_money": float(np.min(money)),
        })
    return output


def _distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = left["production"].astype(np.float32)
    b = right["production"].astype(np.float32)
    occupied = np.maximum(8.0, np.maximum(a[:, :10].sum(axis=1), b[:, :10].sum(axis=1)))
    production = float(np.mean(np.abs(a[:, :10] - b[:, :10]).sum(axis=1) / occupied))
    layout_parts = []
    for x, y in zip(left["layouts"], right["layouts"]):
        active = (x != 0) | (y != 0)
        layout_parts.append(float(np.mean(x[active] != y[active])) if np.any(active) else 0.0)
    layout = float(np.mean(layout_parts))
    x = left["schedule"].astype(np.float32)
    y = right["schedule"].astype(np.float32)
    scale = np.maximum(6.0, np.maximum(x.sum(axis=1), y.sum(axis=1)))
    schedule = float(np.mean(np.abs(x - y).sum(axis=1) / scale))
    land_labor = float(np.mean(np.abs(a[:, 10:] - b[:, 10:]) / np.asarray([3.0, 12.0])))
    return 0.38 * production + 0.38 * layout + 0.18 * schedule + 0.06 * land_labor


def _clusters(rows: list[dict[str, Any]], threshold: float) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import AgglomerativeClustering

    size = len(rows)
    distances = np.zeros((size, size), dtype=np.float32)
    for left in range(size):
        for right in range(left):
            distances[left, right] = distances[right, left] = _distance(rows[left], rows[right])
    if size == 1:
        return np.zeros(1, dtype=np.int32), distances
    labels = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=threshold,
    ).fit_predict(distances)
    return labels.astype(np.int32), distances


def main() -> None:
    args = _args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.cache.exists():
        with np.load(args.cache, allow_pickle=True) as cached:
            rows = list(cached["rows"])
        print(f"loaded {len(rows)} cached target sides", flush=True)
    else:
        tasks = []
        for replay in manifest.get("replays", []):
            if replay.get("error") is not None or not replay.get("targets"):
                continue
            targets = [{**target, "episode_id": replay["episode_id"]} for target in replay["targets"]]
            tasks.append((str((args.manifest.parent / replay["replay"]).resolve()), targets))
        rows = []
        workers = min(max(1, args.workers), len(tasks))
        with mp.get_context("spawn").Pool(workers) as pool:
            for result in pool.imap_unordered(_extract, tasks, chunksize=1):
                rows.extend(result)
                if len(rows) % 100 < len(result):
                    print(f"extracted {len(rows)} target sides", flush=True)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, rows=np.asarray(rows, dtype=object))
        print(f"cached {len(rows)} target sides in {args.cache}", flush=True)

    teams: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        teams[str(row["team"])].append(row)
    report_rows = []
    thresholds = (0.05, 0.08, 0.12, 0.16, 0.20)
    for team, values in sorted(teams.items()):
        labels, distances = _clusters(values, args.threshold)
        sensitivity = {}
        for threshold in thresholds:
            trial, _ = _clusters(values, threshold)
            sensitivity[str(threshold)] = int(len(set(trial.tolist())))
        clusters = []
        for label in sorted(set(labels.tolist())):
            indices = np.flatnonzero(labels == label)
            subset = [values[index] for index in indices]
            representative = max(subset, key=lambda row: (row["reward"], row["episode_id"]))
            clusters.append({
                "size": len(subset),
                "representative": f"{representative['episode_id']}:{representative['player_index']}",
                "median_reward": float(np.median([row["reward"] for row in subset])),
                "median_min_money": float(np.median([row["min_money"] for row in subset])),
                "median_worst_slack72": float(np.median([np.min(row["slack"]) for row in subset])),
                "episode_ids": [int(row["episode_id"]) for row in subset],
            })
        clusters.sort(key=lambda row: (-row["size"], -row["median_reward"]))
        report_rows.append({
            "team": team, "samples": len(values), "clusters": len(clusters),
            "cluster_sizes": [row["size"] for row in clusters],
            "sensitivity": sensitivity, "plans": clusters,
            "pair_distance": {
                "median": float(np.median(distances[np.triu_indices(len(values), 1)])) if len(values) > 1 else 0.0,
                "p10": float(np.quantile(distances[np.triu_indices(len(values), 1)], 0.1)) if len(values) > 1 else 0.0,
                "p90": float(np.quantile(distances[np.triu_indices(len(values), 1)], 0.9)) if len(values) > 1 else 0.0,
            },
        })
    report = {
        "schema_version": 1, "manifest": str(args.manifest),
        "target_sides": len(rows), "teams": len(teams),
        "distance": "0.38 production + 0.38 layout + 0.18 schedule + 0.06 land/labor",
        "selected_threshold": args.threshold,
        "cluster_count_distribution": dict(Counter(row["clusters"] for row in report_rows)),
        "teams_report": report_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "target_sides": len(rows), "teams": len(teams),
        "cluster_count_distribution": report["cluster_count_distribution"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
