#!/usr/bin/env python3
"""Evaluate frozen portable route selectors on official 1.32.7 route outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_models_on_jax_panel import choose_payload  # noqa: E402
from train_hybrid_official_jax_lgbm import ROUTES  # noqa: E402


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def exact_one_sided_sign_p(gained: int, lost: int) -> float:
    total = gained + lost
    if total == 0 or gained <= lost:
        return 1.0
    return float(sum(math.comb(total, value) for value in range(gained, total + 1)) / (2**total))


def metric(
    keys: list[tuple[str, int, int]],
    outcomes: list[dict[str, dict[str, Any]]],
    chosen: np.ndarray,
) -> dict[str, Any]:
    wins: list[bool] = []
    margins: list[float] = []
    by_opponent: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"wins": [], "margins": []})
    by_seat: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"wins": [], "margins": []})
    for index, (opponent, _seed, seat) in enumerate(keys):
        row = outcomes[index][ROUTES[int(chosen[index])]]
        win = bool(row["win"])
        margin = float(row["margin"])
        wins.append(win)
        margins.append(margin)
        by_opponent[opponent]["wins"].append(win)
        by_opponent[opponent]["margins"].append(margin)
        by_seat[seat]["wins"].append(win)
        by_seat[seat]["margins"].append(margin)

    def summarize(values: dict[str, list[float]]) -> dict[str, Any]:
        return {
            "games": len(values["wins"]),
            "wins": int(sum(values["wins"])),
            "win_rate": float(np.mean(values["wins"])),
            "mean_margin": float(np.mean(values["margins"])),
        }

    opponent_metrics = {name: summarize(values) for name, values in by_opponent.items()}
    return {
        "games": len(keys),
        "wins": int(sum(wins)),
        "win_rate": float(np.mean(wins)),
        "mean_margin": float(np.mean(margins)),
        "minimum_opponent_win_rate": min(value["win_rate"] for value in opponent_metrics.values()),
        "route_counts": dict(Counter(ROUTES[int(value)] for value in chosen)),
        "by_opponent": opponent_metrics,
        "by_seat": {str(seat): summarize(values) for seat, values in by_seat.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--official-panels", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = resolve(args.models)
    panel_paths = [resolve(path) for path in args.official_panels]
    output_path = resolve(args.output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    panels = [json.loads(path.read_text(encoding="utf-8")) for path in panel_paths]
    if not config.get("frozen_before_official_evaluation"):
        raise ValueError("candidate config is not marked frozen")
    for panel in panels:
        if panel.get("official_package_version_required") != "1.32.7":
            raise ValueError("official package version boundary mismatch")
        if not panel.get("all_done") or not panel.get("same_state_pass"):
            raise ValueError("official panel failed completion or same-state gate")

    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for panel in panels:
        for row in panel["rows"]:
            key = (str(row["opponent"]), int(row["seed"]), int(row["candidate_seat"]))
            route = str(row["candidate"])
            if route in grouped[key]:
                raise ValueError(f"duplicate official row: {key}, {route}")
            grouped[key][route] = row
    keys = sorted(grouped)
    if any(set(grouped[key]) != set(ROUTES) for key in keys):
        raise ValueError("incomplete four-route official context")
    features = []
    outcomes = []
    for key in keys:
        by_route = grouped[key]
        route_features = [tuple(by_route[route]["features"]) for route in ROUTES]
        if len(set(route_features)) != 1:
            raise ValueError(f"feature mismatch at {key}")
        features.append(route_features[0])
        outcomes.append(by_route)
    x = np.asarray(features, dtype=np.float64)

    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    model_sources: dict[str, Any] = {}
    for spec in config["models"]:
        name = str(spec["name"])
        path = resolve(Path(spec["path"]))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if list(payload["routes"]) != list(ROUTES):
            raise ValueError(f"route order mismatch for {name}")
        if list(payload["feature_names"]) != list(panels[0]["feature_names"]):
            raise ValueError(f"feature names mismatch for {name}")
        chosen = choose_payload(payload, x)
        predictions[name] = chosen
        metrics[name] = metric(keys, outcomes, chosen)
        model_sources[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "schema": payload.get("schema"),
            "threshold": payload.get("threshold"),
        }

    oracle = np.asarray([
        max(
            range(len(ROUTES)),
            key=lambda route_index: (
                float(outcomes[index][ROUTES[route_index]]["utility"]),
                float(outcomes[index][ROUTES[route_index]]["margin"]),
            ),
        )
        for index in range(len(keys))
    ], dtype=np.int16)
    metrics["oracle"] = metric(keys, outcomes, oracle)

    baseline_name = str(config["models"][0]["name"])
    baseline = predictions[baseline_name]
    paired: dict[str, Any] = {}
    for name, chosen in predictions.items():
        gained = lost = 0
        margin_delta = []
        by_opponent: dict[str, dict[str, int]] = defaultdict(lambda: {"gained": 0, "lost": 0, "net": 0})
        for index, (opponent, _seed, _seat) in enumerate(keys):
            base_row = outcomes[index][ROUTES[int(baseline[index])]]
            chosen_row = outcomes[index][ROUTES[int(chosen[index])]]
            base_win = bool(base_row["win"])
            chosen_win = bool(chosen_row["win"])
            if chosen_win and not base_win:
                gained += 1
                by_opponent[opponent]["gained"] += 1
                by_opponent[opponent]["net"] += 1
            elif base_win and not chosen_win:
                lost += 1
                by_opponent[opponent]["lost"] += 1
                by_opponent[opponent]["net"] -= 1
            margin_delta.append(float(chosen_row["margin"] - base_row["margin"]))
        paired[name] = {
            "baseline": baseline_name,
            "different_route_contexts": int(np.sum(chosen != baseline)),
            "gained_wins": gained,
            "lost_wins": lost,
            "net_wins": gained - lost,
            "one_sided_p_value": exact_one_sided_sign_p(gained, lost),
            "mean_margin_delta": float(np.mean(margin_delta)),
            "by_opponent": dict(by_opponent),
        }

    result = {
        "schema": "kawashigi-frozen-models-official-evaluation-v1",
        "status": "OFFICIAL_1_32_7_EVALUATED",
        "truth_boundary": "Frozen models; official Python 1.32.7; fixed seeds; both seats; no post-result tuning.",
        "official_package_version": "1.32.7",
        "contexts": len(keys),
        "games_reused": len(keys) * len(ROUTES),
        "seed_values": sorted({key[1] for key in keys}),
        "seat_swapped": True,
        "opponents": sorted({key[0] for key in keys}),
        "models": model_sources,
        "metrics": metrics,
        "paired_vs_first_model": paired,
        "sources": {
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "panels": [{"path": str(path), "sha256": sha256(path)} for path in panel_paths],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "contexts": result["contexts"],
        "metrics": {
            name: {
                "wins": row["wins"],
                "win_rate": row["win_rate"],
                "mean_margin": row["mean_margin"],
                "minimum_opponent_win_rate": row["minimum_opponent_win_rate"],
            }
            for name, row in metrics.items()
        },
        "paired_vs_first_model": paired,
        "output": str(output_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
