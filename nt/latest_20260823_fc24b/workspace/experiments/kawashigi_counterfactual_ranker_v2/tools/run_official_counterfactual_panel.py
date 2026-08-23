#!/usr/bin/env python3
"""Official 1.32.7 same-state counterfactual route-value panel.

Every candidate must share the exact action prefix through step 144.  For each
(opponent, seed, seat) context this runner verifies that all candidates observe
the exact same public state and 66 actor-visible features at decision step 145,
then compares their terminal win and margin.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DECISION_STEP = 145
ANCHOR_STEPS = (73, 121)
ROUTES = ("10C-4S-75L", "8C-6S-75L", "6C-12S-100L", "6C-8S-75L")
TREND_BASE_NAMES = (
    "price_WHEAT",
    "price_STRAWBERRY",
    "price_MELON",
    "price_MILK",
    "price_WOOL",
    "market_inventory_MILK",
    "market_inventory_WOOL",
    "opp_animal_COW",
    "opp_animal_SHEEP",
)
FEATURE_SOURCE = (
    ROOT
    / "experiments"
    / "gold_adaptive_rule_v2"
    / "agents"
    / "gold_imitations_current_20260816_0955_v2"
    / "rank01_team"
    / "main.py"
)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--opponents", type=Path, required=True)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--seed-values", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_entries(path: Path, key: str) -> list[tuple[str, Path]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: list[tuple[str, Path]] = []
    for item in payload[key]:
        entry_path = resolve(str(item["path"]))
        if not entry_path.is_file():
            raise FileNotFoundError(entry_path)
        entries.append((str(item["name"]), entry_path))
    return entries


@lru_cache(maxsize=1)
def feature_module() -> Any:
    module_name = "kawashigi_counterfactual_feature_source"
    spec = importlib.util.spec_from_file_location(module_name, FEATURE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import feature source: {FEATURE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_feature_names() -> list[str]:
    shops = ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET")
    products = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
    animals = ("GOOSE", "COW", "SHEEP")
    crops = products[:5]
    names = [f"shop_{name}" for name in shops]
    names += [f"price_{name}" for name in products]
    names += [f"market_inventory_{name}" for name in products]
    names += [f"opp_animal_{name}" for name in animals]
    names += [f"opp_crop_{name}" for name in crops]
    names += ["opp_unlocked", "opp_hands", "opp_money"]
    names += [f"own_animal_{name}" for name in animals]
    names += [f"own_crop_{name}" for name in crops]
    names += ["own_unlocked", "own_hands", "own_money"]
    if len(names) != 48:
        raise AssertionError(len(names))
    return names


BASE_NAMES = base_feature_names()
BASE_INDEX = {name: index for index, name in enumerate(BASE_NAMES)}
FEATURE_NAMES = BASE_NAMES + [
    f"delta_from_step{step}_{name}"
    for step in ANCHOR_STEPS
    for name in TREND_BASE_NAMES
]


def extended_features(env: Any, seat: int) -> list[float]:
    module = feature_module()
    current = list(module._cgr_features(env.steps[DECISION_STEP][seat].observation))
    values = list(current)
    for anchor_step in ANCHOR_STEPS:
        anchor = list(module._cgr_features(env.steps[anchor_step][seat].observation))
        values.extend(float(current[BASE_INDEX[name]] - anchor[BASE_INDEX[name]]) for name in TREND_BASE_NAMES)
    if len(values) != 66:
        raise AssertionError(len(values))
    return [float(value) for value in values]


def to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if hasattr(value, "items"):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def observation_hash(observation: Any) -> str:
    payload = json.dumps(to_plain(observation), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def run_one(task: tuple[str, str, str, str, int, int]) -> dict[str, Any]:
    candidate_name, candidate_path, opponent_name, opponent_path, seed, candidate_seat = task
    from kaggle_environments import make

    agents = [candidate_path, opponent_path]
    if candidate_seat == 1:
        agents.reverse()
    started = time.perf_counter()
    try:
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
        env.run(agents)
        final = env.steps[-1]
        candidate_reward = float(final[candidate_seat].reward)
        opponent_reward = float(final[1 - candidate_seat].reward)
        obs145 = env.steps[DECISION_STEP][candidate_seat].observation
        fallback_route = str(feature_module()._cgr_choose_route(obs145))
        return {
            "candidate": candidate_name,
            "opponent": opponent_name,
            "seed": seed,
            "candidate_seat": candidate_seat,
            "frames": len(env.steps),
            "candidate_status": str(final[candidate_seat].status),
            "opponent_status": str(final[1 - candidate_seat].status),
            "candidate_reward": candidate_reward,
            "opponent_reward": opponent_reward,
            "margin": candidate_reward - opponent_reward,
            "win": candidate_reward > opponent_reward,
            "tie": candidate_reward == opponent_reward,
            "utility": (200000.0 if candidate_reward > opponent_reward else 0.0) + max(-50000.0, min(50000.0, candidate_reward - opponent_reward)),
            "fallback_route": fallback_route,
            "features": extended_features(env, candidate_seat),
            "observation_hash_step145": observation_hash(obs145),
            "seconds": time.perf_counter() - started,
            "error": "",
        }
    except Exception as exc:
        return {
            "candidate": candidate_name,
            "opponent": opponent_name,
            "seed": seed,
            "candidate_seat": candidate_seat,
            "frames": 0,
            "candidate_status": "ERROR",
            "opponent_status": "ERROR",
            "candidate_reward": 0.0,
            "opponent_reward": 0.0,
            "margin": 0.0,
            "win": False,
            "tie": False,
            "utility": -1e9,
            "fallback_route": "",
            "features": [],
            "observation_hash_step145": "",
            "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "games": len(rows),
        "wins": sum(bool(row["win"]) for row in rows),
        "win_rate": sum(bool(row["win"]) for row in rows) / len(rows) if rows else 0.0,
        "mean_margin": sum(float(row["margin"]) for row in rows) / len(rows) if rows else 0.0,
        "all_done": all(row["frames"] == 720 and row["candidate_status"] == "DONE" and row["opponent_status"] == "DONE" and not row["error"] for row in rows),
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 18:
        raise ValueError("workers must be between 1 and 18")
    if args.seed_values:
        seed_values = [int(value) for value in args.seed_values]
        if len(seed_values) != len(set(seed_values)):
            raise ValueError("seed-values must be unique")
        if args.seed_start is not None or args.seeds is not None:
            raise ValueError("use either seed-values or seed-start plus seeds")
    else:
        if args.seed_start is None or args.seeds is None or args.seeds <= 0:
            raise ValueError("seed-start and positive seeds are required")
        seed_values = list(range(args.seed_start, args.seed_start + args.seeds))
    candidate_manifest = resolve(str(args.candidates))
    opponent_manifest = resolve(str(args.opponents))
    candidates = load_entries(candidate_manifest, "candidates")
    opponents = load_entries(opponent_manifest, "opponents")
    if tuple(name for name, _ in candidates) != ROUTES:
        raise ValueError(f"candidate order/names must be {ROUTES}")
    tasks = [
        (candidate_name, str(candidate_path), opponent_name, str(opponent_path), seed, seat)
        for opponent_name, opponent_path in opponents
        for seed in seed_values
        for seat in (0, 1)
        for candidate_name, candidate_path in candidates
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        rows = list(executor.map(run_one, tasks, chunksize=1))

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent"]), int(row["seed"]), int(row["candidate_seat"]))].append(row)

    contexts: list[dict[str, Any]] = []
    for (opponent, seed, seat), group in sorted(grouped.items()):
        by_route = {str(row["candidate"]): row for row in group}
        complete = len(by_route) == len(ROUTES) and all(route in by_route for route in ROUTES)
        feature_equal = complete and len({tuple(by_route[route]["features"]) for route in ROUTES}) == 1
        observation_equal = complete and len({by_route[route]["observation_hash_step145"] for route in ROUTES}) == 1
        fallback_equal = complete and len({by_route[route]["fallback_route"] for route in ROUTES}) == 1
        best = max(group, key=lambda row: (float(row["utility"]), float(row["margin"])))
        fallback = str(group[0]["fallback_route"]) if group else ""
        fallback_row = by_route.get(fallback)
        contexts.append({
            "opponent": opponent,
            "seed": seed,
            "candidate_seat": seat,
            "complete_four_routes": complete,
            "features_equal": feature_equal,
            "observation_equal": observation_equal,
            "fallback_equal": fallback_equal,
            "fallback_route": fallback,
            "best_route": str(best["candidate"]),
            "fallback_win": bool(fallback_row["win"]) if fallback_row else False,
            "fallback_margin": float(fallback_row["margin"]) if fallback_row else -1e9,
            "oracle_win": bool(best["win"]),
            "oracle_margin": float(best["margin"]),
            "oracle_utility_gain": float(best["utility"] - fallback_row["utility"]) if fallback_row else 0.0,
            "route_results": {
                route: {
                    "win": bool(by_route[route]["win"]),
                    "margin": float(by_route[route]["margin"]),
                    "candidate_reward": float(by_route[route]["candidate_reward"]),
                }
                for route in ROUTES if route in by_route
            },
        })

    all_done = all(rate([row])["all_done"] for row in rows)
    same_state_pass = all(context["complete_four_routes"] and context["features_equal"] and context["observation_equal"] and context["fallback_equal"] for context in contexts)
    fallback_wins = sum(bool(context["fallback_win"]) for context in contexts)
    oracle_wins = sum(bool(context["oracle_win"]) for context in contexts)
    fallback_margin = sum(float(context["fallback_margin"]) for context in contexts) / len(contexts)
    oracle_margin = sum(float(context["oracle_margin"]) for context in contexts) / len(contexts)
    result = {
        "schema": "kawashigi-official-same-state-counterfactual-panel-v1",
        "official_package_version_required": "1.32.7",
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": sha256(candidate_manifest),
        "opponent_manifest": str(opponent_manifest),
        "opponent_manifest_sha256": sha256(opponent_manifest),
        "feature_source": str(FEATURE_SOURCE),
        "feature_source_sha256": sha256(FEATURE_SOURCE),
        "feature_names": FEATURE_NAMES,
        "decision_step": DECISION_STEP,
        "seed_start": min(seed_values),
        "seed_count": len(seed_values),
        "seed_values": seed_values,
        "seat_swapped": True,
        "workers": args.workers,
        "game_count": len(rows),
        "context_count": len(contexts),
        "all_done": all_done,
        "same_state_pass": same_state_pass,
        "feasibility": {
            "fallback_wins": fallback_wins,
            "fallback_win_rate": fallback_wins / len(contexts),
            "oracle_wins": oracle_wins,
            "oracle_win_rate": oracle_wins / len(contexts),
            "oracle_extra_wins": oracle_wins - fallback_wins,
            "fallback_mean_margin": fallback_margin,
            "oracle_mean_margin": oracle_margin,
            "oracle_mean_margin_gain": oracle_margin - fallback_margin,
            "best_route_counts": dict(Counter(str(context["best_route"]) for context in contexts)),
            "fallback_route_counts": dict(Counter(str(context["fallback_route"]) for context in contexts)),
        },
        "route_metrics": {route: rate([row for row in rows if row["candidate"] == route]) for route in ROUTES},
        "by_opponent": {
            opponent: {
                "fallback_win_rate": sum(bool(context["fallback_win"]) for context in contexts if context["opponent"] == opponent) / sum(1 for context in contexts if context["opponent"] == opponent),
                "oracle_win_rate": sum(bool(context["oracle_win"]) for context in contexts if context["opponent"] == opponent) / sum(1 for context in contexts if context["opponent"] == opponent),
                "oracle_extra_wins": sum(bool(context["oracle_win"]) - bool(context["fallback_win"]) for context in contexts if context["opponent"] == opponent),
            }
            for opponent, _ in opponents
        },
        "candidate_files": [{"name": name, "path": str(path), "sha256": sha256(path)} for name, path in candidates],
        "opponent_files": [{"name": name, "path": str(path), "sha256": sha256(path)} for name, path in opponents],
        "contexts": contexts,
        "rows": rows,
    }
    output = resolve(str(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_done": all_done,
        "same_state_pass": same_state_pass,
        "game_count": len(rows),
        "context_count": len(contexts),
        "feasibility": result["feasibility"],
        "route_metrics": result["route_metrics"],
        "by_opponent": result["by_opponent"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0 if all_done and same_state_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
