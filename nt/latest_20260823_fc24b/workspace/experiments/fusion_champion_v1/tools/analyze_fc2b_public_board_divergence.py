#!/usr/bin/env python3
"""Find stable public-board phenotypes of FC2B losses.

This is an attribution tool, not a policy trainer.  It only reads state fields
that the acting player can legally observe.  The two seats of one event seed
are always kept in the same split to prevent paired-seed leakage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def feature_cube(pair: dict, products: list[str], crops: list[str], animals: list[str]):
    games = pair["per_game"]
    columns: list[str] = []
    values: list[np.ndarray] = []

    def add(prefix: str, key: str, names: list[str]) -> None:
        array = np.asarray([game[key] for game in games], dtype=np.float64)
        for index, name in enumerate(names):
            columns.append(f"{prefix}_{name.lower()}")
            values.append(array[:, :, index])

    add("candidate_animal_count", "daily_candidate_animal_count", animals)
    add("opponent_animal_count", "daily_opponent_animal_count_public", animals)
    add("candidate_animal_yield", "daily_candidate_animal_yield", animals)
    add("opponent_animal_yield", "daily_opponent_animal_yield_public", animals)
    add("candidate_crop_count", "daily_candidate_crop_count", crops)
    add("opponent_crop_count", "daily_opponent_crop_count_public", crops)
    add("candidate_crop_yield", "daily_candidate_crop_yield", crops)
    add("opponent_crop_yield", "daily_opponent_crop_yield_public", crops)
    add("market_price", "daily_market_price", products)
    add("market_inventory", "daily_market_inventory", products)

    candidate_animals = np.asarray(
        [game["daily_candidate_animal_count"] for game in games], dtype=np.float64
    )
    opponent_animals = np.asarray(
        [game["daily_opponent_animal_count_public"] for game in games], dtype=np.float64
    )
    candidate_animal_yield = np.asarray(
        [game["daily_candidate_animal_yield"] for game in games], dtype=np.float64
    )
    opponent_animal_yield = np.asarray(
        [game["daily_opponent_animal_yield_public"] for game in games], dtype=np.float64
    )
    candidate_crops = np.asarray(
        [game["daily_candidate_crop_count"] for game in games], dtype=np.float64
    )
    opponent_crops = np.asarray(
        [game["daily_opponent_crop_count_public"] for game in games], dtype=np.float64
    )
    candidate_crop_yield = np.asarray(
        [game["daily_candidate_crop_yield"] for game in games], dtype=np.float64
    )
    opponent_crop_yield = np.asarray(
        [game["daily_opponent_crop_yield_public"] for game in games], dtype=np.float64
    )
    for prefix, delta, names in (
        ("animal_count_lead", candidate_animals - opponent_animals, animals),
        ("animal_yield_lead", candidate_animal_yield - opponent_animal_yield, animals),
        ("crop_count_lead", candidate_crops - opponent_crops, crops),
        ("crop_yield_lead", candidate_crop_yield - opponent_crop_yield, crops),
    ):
        for index, name in enumerate(names):
            columns.append(f"{prefix}_{name.lower()}")
            values.append(delta[:, :, index])

    candidate_units = np.asarray(
        [game["daily_candidate_units"] for game in games], dtype=np.float64
    )
    opponent_units = np.asarray(
        [game["daily_opponent_units_public"] for game in games], dtype=np.float64
    )
    candidate_cash = np.asarray(
        [game["daily_candidate_cash"] for game in games], dtype=np.float64
    )
    opponent_cash = np.asarray(
        [game["daily_opponent_cash_public"] for game in games], dtype=np.float64
    )
    for name, array in (
        ("candidate_units", candidate_units),
        ("opponent_units", opponent_units),
        ("unit_lead", candidate_units - opponent_units),
        ("candidate_cash", candidate_cash),
        ("opponent_cash", opponent_cash),
        ("cash_lead", candidate_cash - opponent_cash),
    ):
        columns.append(name)
        values.append(array)

    return columns, np.stack(values, axis=2)


def effect_rows(columns: list[str], cube: np.ndarray, losses: np.ndarray) -> list[dict]:
    rows = []
    wins = ~losses
    for day in range(cube.shape[1]):
        for feature, name in enumerate(columns):
            loss_values = cube[losses, day, feature]
            win_values = cube[wins, day, feature]
            if not len(loss_values) or not len(win_values):
                continue
            loss_mean = float(np.mean(loss_values))
            win_mean = float(np.mean(win_values))
            pooled = float(np.sqrt((np.var(loss_values) + np.var(win_values)) / 2.0))
            standardized = (loss_mean - win_mean) / max(pooled, 1e-6)
            rows.append(
                {
                    "day": day + 1,
                    "feature": name,
                    "loss_mean": loss_mean,
                    "win_mean": win_mean,
                    "loss_minus_win": loss_mean - win_mean,
                    "standardized_effect": standardized,
                }
            )
    return sorted(rows, key=lambda row: abs(row["standardized_effect"]), reverse=True)


def threshold_rows(
    columns: list[str],
    cube: np.ndarray,
    losses: np.ndarray,
    seeds: np.ndarray,
) -> list[dict]:
    unique_seeds = np.unique(seeds)
    split_seed = unique_seeds[len(unique_seeds) // 2]
    train = seeds < split_seed
    holdout = ~train
    rows = []
    for day in range(3, min(25, cube.shape[1])):
        for feature, name in enumerate(columns):
            train_values = cube[train, day, feature]
            unique = np.unique(train_values)
            if len(unique) < 2:
                continue
            thresholds = np.unique(np.quantile(unique, np.linspace(0.1, 0.9, 9)))
            for threshold in thresholds:
                for operator in ("<=", ">="):
                    condition = (
                        cube[:, day, feature] <= threshold
                        if operator == "<="
                        else cube[:, day, feature] >= threshold
                    )
                    def local(mask: np.ndarray) -> dict:
                        selected = condition & mask
                        local_losses = losses & mask
                        selected_count = int(np.sum(selected))
                        loss_count = int(np.sum(local_losses))
                        selected_losses = int(np.sum(selected & losses))
                        base = float(loss_count / np.sum(mask))
                        return {
                            "selected_games": selected_count,
                            "coverage": float(selected_count / np.sum(mask)),
                            "selected_losses": selected_losses,
                            "loss_precision": float(selected_losses / selected_count) if selected_count else 0.0,
                            "loss_recall": float(selected_losses / loss_count) if loss_count else 0.0,
                            "loss_rate_lift": (
                                float(selected_losses / selected_count / base)
                                if selected_count and base
                                else 0.0
                            ),
                        }

                    train_stats = local(train)
                    holdout_stats = local(holdout)
                    if min(train_stats["selected_games"], holdout_stats["selected_games"]) < 8:
                        continue
                    if max(train_stats["coverage"], holdout_stats["coverage"]) > 0.65:
                        continue
                    stable_lift = min(
                        train_stats["loss_rate_lift"], holdout_stats["loss_rate_lift"]
                    )
                    stability_score = stable_lift * np.sqrt(
                        min(train_stats["loss_recall"], holdout_stats["loss_recall"])
                    )
                    rows.append(
                        {
                            "day": day + 1,
                            "feature": name,
                            "operator": operator,
                            "threshold": float(threshold),
                            "train": train_stats,
                            "holdout": holdout_stats,
                            "stable_score": float(stability_score),
                        }
                    )
    return sorted(rows, key=lambda row: row["stable_score"], reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = []
    for pair in payload["pairs"]:
        columns, cube = feature_cube(
            pair, payload["products"], payload["crops"], payload["animals"]
        )
        losses = np.asarray([game["result"] == "loss" for game in pair["per_game"]])
        seeds = np.asarray([game["seed"] for game in pair["per_game"]], dtype=np.int64)
        rows.append(
            {
                "opponent": pair["opponent"],
                "games": len(losses),
                "losses": int(np.sum(losses)),
                "loss_rate": float(np.mean(losses)),
                "top_standardized_effects": effect_rows(columns, cube, losses)[:500],
                "top_stable_public_conditions": threshold_rows(
                    columns, cube, losses, seeds
                )[:1000],
            }
        )
    result = {
        "schema": "kaggriculture.fusion_champion.fc2b-public-board-divergence.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source": str(args.input.resolve()),
        "split_protocol": "event seeds split in half; both seats stay together",
        "interpretation_boundary": (
            "conditions describe loss phenotypes only; no condition is a policy improvement without counterfactual action validation"
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            json.dumps(
                {
                    "opponent": row["opponent"],
                    "games": row["games"],
                    "losses": row["losses"],
                    "top_conditions": row["top_stable_public_conditions"][:5],
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
