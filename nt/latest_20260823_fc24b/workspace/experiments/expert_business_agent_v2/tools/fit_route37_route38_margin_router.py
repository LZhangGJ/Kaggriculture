#!/usr/bin/env python3
"""Fit an auditable prefix-safe Route37/Route38 terminal-margin router.

The two Rank16 action streams are byte-identical through step 191.  Therefore
the actor-visible observation recorded at frame 192 is a valid same-state
decision point.  This tool never uses opponent names, seeds, seats, private
opponent data, or future events as model inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text


PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
SHOPS = (
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
    "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET",
)
ROUTE37 = "Route37_8C4S"
ROUTE38 = "Route38_6C6S"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def add_numeric(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, bool):
        output[prefix] = float(value)
    elif isinstance(value, (int, float)):
        output[prefix] = float(value)
    elif isinstance(value, dict):
        for key in sorted(value):
            add_numeric(f"{prefix}.{key}" if prefix else str(key), value[key], output)


def features(snapshot: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    shop_counts = Counter(str(value) for value in snapshot.get("shops", []))
    for name in SHOPS:
        result[f"shop.{name}"] = float(shop_counts.get(name, 0))
    for name in PRODUCTS:
        result[f"price.{name}"] = float(snapshot.get("prices", {}).get(name, 0) or 0)
        result[f"market_inventory.{name}"] = float(snapshot.get("inventory", {}).get(name, 0) or 0)
    for side in ("own_stats", "opponent_stats"):
        add_numeric(side, snapshot.get(side, {}), result)
    return result


def summarize(rows: list[dict[str, Any]], choices: np.ndarray) -> dict[str, Any]:
    margins = np.asarray([
        row["margin38"] if choice else row["margin37"]
        for row, choice in zip(rows, choices, strict=True)
    ], dtype=np.float64)
    wins = margins > 0
    ties = margins == 0
    per_opponent: dict[str, list[bool]] = defaultdict(list)
    for row, win in zip(rows, wins, strict=True):
        per_opponent[row["opponent"]].append(bool(win))
    rates = {name: sum(values) / len(values) for name, values in per_opponent.items()}
    return {
        "games": len(rows),
        "wins": int(wins.sum()),
        "ties": int(ties.sum()),
        "win_rate": float(wins.mean()),
        "score_rate": float((wins.sum() + 0.5 * ties.sum()) / len(rows)),
        "mean_margin": float(margins.mean()),
        "route38_rate": float(np.asarray(choices, dtype=np.float64).mean()),
        "minimum_opponent_win_rate": float(min(rates.values())),
        "opponents_at_zero": sorted(name for name, value in rates.items() if value == 0.0),
        "per_opponent_win_rate": rates,
    }


def build_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    by_case: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in payload["rows"]:
        candidate = str(row["candidate"])
        if candidate not in (ROUTE37, ROUTE38):
            continue
        key = (str(row["opponent"]), int(row["seed"]), int(row["candidate_seat"]))
        by_case[key][candidate] = row

    paired: list[dict[str, Any]] = []
    state_mismatches: list[dict[str, Any]] = []
    seat_duplicates_checked = 0
    for (opponent, seed, seat), pair in sorted(by_case.items()):
        if set(pair) != {ROUTE37, ROUTE38}:
            raise ValueError(f"incomplete pair: {(opponent, seed, seat)}")
        left = pair[ROUTE37]
        right = pair[ROUTE38]
        snapshot37 = left["public_state_steps"]["192"]
        snapshot38 = right["public_state_steps"]["192"]
        if canonical(snapshot37) != canonical(snapshot38):
            state_mismatches.append({"opponent": opponent, "seed": seed, "seat": seat})
            continue
        # Deterministic seat swaps are exact duplicates in this environment.
        # Retain seat 0 only so trees do not double-count identical examples.
        if seat != 0:
            seat_duplicates_checked += 1
            continue
        feature_map = features(snapshot37)
        paired.append({
            "opponent": opponent,
            "seed": seed,
            "features": feature_map,
            "margin37": float(left["margin"]),
            "margin38": float(right["margin"]),
            "delta_margin": float(right["margin"] - left["margin"]),
        })
    if state_mismatches:
        raise ValueError(f"step-192 state mismatches: {state_mismatches[:5]}")
    feature_names = sorted(paired[0]["features"])
    if any(sorted(row["features"]) != feature_names for row in paired):
        raise ValueError("inconsistent feature schema")
    audit = {
        "raw_paired_seat_cases": len(by_case),
        "step192_state_mismatches": len(state_mismatches),
        "seat1_duplicates_dropped": seat_duplicates_checked,
        "unique_opponent_seed_cases": len(paired),
    }
    return paired, feature_names, audit


def matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    return np.asarray([[row["features"][name] for name in names] for row in rows], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--train-seed-max", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.matrix.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows, feature_names, audit = build_rows(payload)
    train = [row for row in rows if row["seed"] <= args.train_seed_max]
    holdout = [row for row in rows if row["seed"] > args.train_seed_max]
    if not train or not holdout:
        raise ValueError("train and holdout must both be non-empty")

    train_seeds = sorted({row["seed"] for row in train})
    holdout_seeds = sorted({row["seed"] for row in holdout})
    depths = (1, 2, 3, 4, 5)
    leaves = (4, 8, 12, 16, 24)
    thresholds = (-1000.0, -500.0, 0.0, 500.0, 1000.0, 2000.0, 4000.0)
    cv_rows: list[dict[str, Any]] = []

    for depth in depths:
        for leaf in leaves:
            fold_predictions: dict[tuple[str, int], float] = {}
            for validation_seed in train_seeds:
                fit_rows = [row for row in train if row["seed"] != validation_seed]
                validation = [row for row in train if row["seed"] == validation_seed]
                model = DecisionTreeRegressor(
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    random_state=20260817,
                )
                model.fit(matrix(fit_rows, feature_names), np.asarray([row["delta_margin"] for row in fit_rows]))
                predicted = model.predict(matrix(validation, feature_names))
                for row, value in zip(validation, predicted, strict=True):
                    fold_predictions[(row["opponent"], row["seed"])] = float(value)
            predictions = np.asarray([
                fold_predictions[(row["opponent"], row["seed"])] for row in train
            ])
            for threshold in thresholds:
                choices = predictions > threshold
                stats = summarize(train, choices)
                cv_rows.append({
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "gain_threshold": threshold,
                    **{key: stats[key] for key in (
                        "win_rate", "mean_margin", "route38_rate",
                        "minimum_opponent_win_rate", "opponents_at_zero",
                    )},
                })

    # Select only from cross-validated training outcomes.  Prefer win rate,
    # then terminal margin, then the simpler/less interventionist tree.
    cv_rows.sort(key=lambda row: (
        row["win_rate"], row["mean_margin"],
        -row["max_depth"], row["min_samples_leaf"], -row["route38_rate"],
    ), reverse=True)
    selected = cv_rows[0]
    model = DecisionTreeRegressor(
        max_depth=int(selected["max_depth"]),
        min_samples_leaf=int(selected["min_samples_leaf"]),
        random_state=20260817,
    )
    model.fit(matrix(train, feature_names), np.asarray([row["delta_margin"] for row in train]))
    train_predictions = model.predict(matrix(train, feature_names))
    holdout_predictions = model.predict(matrix(holdout, feature_names))
    threshold = float(selected["gain_threshold"])

    baseline_train = summarize(train, np.zeros(len(train), dtype=bool))
    baseline_holdout = summarize(holdout, np.zeros(len(holdout), dtype=bool))
    route38_train = summarize(train, np.ones(len(train), dtype=bool))
    route38_holdout = summarize(holdout, np.ones(len(holdout), dtype=bool))
    routed_train = summarize(train, train_predictions > threshold)
    routed_holdout = summarize(holdout, holdout_predictions > threshold)
    oracle_holdout = summarize(
        holdout,
        np.asarray([row["delta_margin"] > 0 for row in holdout], dtype=bool),
    )

    tree_payload = {
        "children_left": model.tree_.children_left.tolist(),
        "children_right": model.tree_.children_right.tolist(),
        "feature": model.tree_.feature.tolist(),
        "threshold": model.tree_.threshold.tolist(),
        "value": model.tree_.value[:, 0, 0].tolist(),
        "n_node_samples": model.tree_.n_node_samples.tolist(),
    }
    result = {
        "schema": "route37-route38-prefix-margin-router-fit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FIT_COMPLETE_NOT_OFFICIAL_HOLDOUT_AGENT_ACCEPTED",
        "source_matrix": str(source),
        "source_matrix_sha256": sha256(source),
        "decision_step": 192,
        "truth_boundary": (
            "Model selection used only seeds at or below train_seed_max. "
            "The later seeds are a diagnostic holdout from the same frozen opponent pool, "
            "not final independent Agent acceptance."
        ),
        "input_exclusions": [
            "opponent_name", "seed", "seat", "opponent_private_state",
            "future_shop_events", "terminal_rewards",
        ],
        "audit": audit,
        "train_seeds": train_seeds,
        "holdout_seeds": holdout_seeds,
        "feature_names": feature_names,
        "selected_cv_configuration": selected,
        "top_cv_configurations": cv_rows[:20],
        "tree": tree_payload,
        "tree_text": export_text(model, feature_names=feature_names, decimals=3),
        "train": {
            "route37": baseline_train,
            "route38": route38_train,
            "router": routed_train,
        },
        "holdout": {
            "route37": baseline_holdout,
            "route38": route38_holdout,
            "router": routed_holdout,
            "oracle": oracle_holdout,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "audit": audit,
        "selected": selected,
        "train": result["train"],
        "holdout": result["holdout"],
        "tree_text": result["tree_text"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
