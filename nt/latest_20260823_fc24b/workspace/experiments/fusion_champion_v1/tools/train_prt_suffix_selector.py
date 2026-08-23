#!/usr/bin/env python3
"""Train and audit a no-future-leakage baseline-vs-PRT suffix selector."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text


BASE_PRICE = np.asarray((25, 35, 60, 120, 250, 50, 160, 200, 100), dtype=float)
SHED_ACCESS = np.asarray(((4, 4), (5, 4), (4, 5), (5, 5)), dtype=int)


def add(features: dict[str, float], name: str, value) -> None:
    features[name] = float(value)


def distances_to_shed(mask: np.ndarray) -> tuple[float, float]:
    positions = np.argwhere(mask.reshape(10, 10))
    if len(positions) == 0:
        return 0.0, 0.0
    xy = positions[:, ::-1]
    distances = np.min(np.sum(np.abs(xy[:, None, :] - SHED_ACCESS[None, :, :]), axis=2), axis=1)
    return float(np.mean(distances)), float(np.max(distances))


def build_feature_row(row: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    add(out, "candidate_seat", row["candidate_seat"])
    for name in (
        "money_own",
        "money_rival",
        "hires_today_own",
        "hires_today_rival",
        "unlocked_count_own",
        "unlocked_count_rival",
    ):
        add(out, name, row[name])
    add(out, "money_gap", row["money_own"] - row["money_rival"])
    add(out, "hires_gap", row["hires_today_own"] - row["hires_today_rival"])
    add(out, "unlocked_gap", row["unlocked_count_own"] - row["unlocked_count_rival"])

    aggregates = {}
    for side in ("own", "rival"):
        kind = np.asarray(row[f"tile_kind_{side}"], dtype=int)
        crop = np.asarray(row[f"tile_crop_{side}"], dtype=int)
        animal = np.asarray(row[f"tile_animal_{side}"], dtype=int)
        tile_yield = np.asarray(row[f"tile_yield_{side}"], dtype=int)
        neglect = np.asarray(row[f"tile_neglect_{side}"], dtype=int)
        for value in range(6):
            add(out, f"{side}_kind_{value}_count", np.sum(kind == value))
        crop_counts = []
        crop_yields = []
        for value in range(5):
            mask = crop == value
            count = int(np.sum(mask))
            yield_sum = int(np.sum(np.where(mask, tile_yield, 0)))
            crop_counts.append(count)
            crop_yields.append(yield_sum)
            add(out, f"{side}_crop_{value}_count", count)
            add(out, f"{side}_crop_{value}_yield", yield_sum)
            add(out, f"{side}_crop_{value}_neglect", np.sum(np.where(mask, neglect, 0)))
            mean_distance, max_distance = distances_to_shed(mask)
            add(out, f"{side}_crop_{value}_mean_shed_distance", mean_distance)
            add(out, f"{side}_crop_{value}_max_shed_distance", max_distance)
        animal_counts = []
        animal_yields = []
        for value in range(3):
            mask = animal == value
            count = int(np.sum(mask))
            yield_sum = int(np.sum(np.where(mask, tile_yield, 0)))
            animal_counts.append(count)
            animal_yields.append(yield_sum)
            add(out, f"{side}_animal_{value}_count", count)
            add(out, f"{side}_animal_{value}_yield", yield_sum)
            add(out, f"{side}_animal_{value}_neglect", np.sum(np.where(mask, neglect, 0)))
            mean_distance, max_distance = distances_to_shed(mask)
            add(out, f"{side}_animal_{value}_mean_shed_distance", mean_distance)
            add(out, f"{side}_animal_{value}_max_shed_distance", max_distance)
        positions = np.asarray(row[f"unit_pos_{side}"], dtype=float)
        add(out, f"{side}_unit_count", len(positions))
        if len(positions):
            add(out, f"{side}_unit_x_mean", np.mean(positions[:, 0]))
            add(out, f"{side}_unit_y_mean", np.mean(positions[:, 1]))
            unit_distances = np.min(
                np.sum(np.abs(positions[:, None, :] - SHED_ACCESS[None, :, :]), axis=2),
                axis=1,
            )
            add(out, f"{side}_unit_shed_distance_mean", np.mean(unit_distances))
            add(out, f"{side}_unit_shed_distance_max", np.max(unit_distances))
        else:
            for suffix in ("unit_x_mean", "unit_y_mean", "unit_shed_distance_mean", "unit_shed_distance_max"):
                add(out, f"{side}_{suffix}", 0)
        aggregates[side] = (np.asarray(crop_counts), np.asarray(crop_yields), np.asarray(animal_counts), np.asarray(animal_yields))

    own_crop, own_crop_yield, own_animal, own_animal_yield = aggregates["own"]
    rival_crop, rival_crop_yield, rival_animal, rival_animal_yield = aggregates["rival"]
    for value in range(5):
        add(out, f"crop_{value}_count_gap", own_crop[value] - rival_crop[value])
        add(out, f"crop_{value}_yield_gap", own_crop_yield[value] - rival_crop_yield[value])
    for value in range(3):
        add(out, f"animal_{value}_count_gap", own_animal[value] - rival_animal[value])
        add(out, f"animal_{value}_yield_gap", own_animal_yield[value] - rival_animal_yield[value])

    shed = np.asarray(row["shed_own"], dtype=int)
    seeds = np.asarray(row["seeds_own"], dtype=int)
    own_inventory = np.asarray(row["unit_inventory_own"], dtype=int)
    inventory_total = np.sum(own_inventory, axis=0) if own_inventory.size else np.zeros_like(shed)
    for value in range(len(shed)):
        add(out, f"shed_{value}", shed[value])
        add(out, f"unit_inventory_{value}", inventory_total[value])
    for value in range(len(seeds)):
        add(out, f"seed_{value}", seeds[value])

    market_price = np.asarray(row["market_price"], dtype=float)
    market_inventory = np.asarray(row["market_inventory"], dtype=float)
    for value in range(len(market_price)):
        add(out, f"market_price_{value}", market_price[value])
        add(out, f"market_price_ratio_{value}", market_price[value] / BASE_PRICE[value])
        add(out, f"market_inventory_{value}", market_inventory[value])
    shops = [int(value) for value in row["town_shops"]]
    add(out, "town_count", len(shops))
    for shop in range(8):
        add(out, f"shop_{shop}_count", shops.count(shop))
        add(out, f"shop_first_{shop}", int(len(shops) > 0 and shops[0] == shop))
        add(out, f"shop_second_{shop}", int(len(shops) > 1 and shops[1] == shop))
    return out


def load_outcomes(path: Path, suffix_route: int) -> tuple[dict, dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = next(row for row in payload["rows"] if row["suffix_route"] == -1)
    suffix = next(row for row in payload["rows"] if row["suffix_route"] == suffix_route)
    base_lookup = {(int(row["seed"]), int(row["candidate_seat"])): int(row["margin"]) for row in base["per_game"]}
    suffix_lookup = {(int(row["seed"]), int(row["candidate_seat"])): int(row["margin"]) for row in suffix["per_game"]}
    return base_lookup, suffix_lookup, payload.get("decision_features", [])


def assemble(feature_path: Path, outcome_path: Path, suffix_route: int, feature_names=None):
    feature_payload = json.loads(feature_path.read_text(encoding="utf-8"))
    feature_rows = feature_payload.get("decision_features", [])
    if not feature_rows:
        raise AssertionError(f"no decision features in {feature_path}")
    base, suffix, _ = load_outcomes(outcome_path, suffix_route=suffix_route)
    built = []
    keys = []
    for row in feature_rows:
        key = (int(row["seed"]), int(row["candidate_seat"]))
        if key not in base or key not in suffix:
            raise AssertionError(f"missing outcome for {key}")
        built.append(build_feature_row(row))
        keys.append(key)
    if feature_names is None:
        feature_names = sorted(built[0])
    matrix = np.asarray([[row[name] for name in feature_names] for row in built], dtype=np.float32)
    base_margin = np.asarray([base[key] for key in keys], dtype=np.int64)
    suffix_margin = np.asarray([suffix[key] for key in keys], dtype=np.int64)
    groups = np.asarray([key[0] for key in keys], dtype=np.int64)
    return matrix, base_margin, suffix_margin, groups, keys, feature_names


def labels_and_weights(base: np.ndarray, suffix: np.ndarray):
    base_win, suffix_win = base > 0, suffix > 0
    suffix_only = (~base_win) & suffix_win
    base_only = base_win & (~suffix_win)
    same_result = ~(suffix_only | base_only)
    labels = np.where(suffix_only, 1, np.where(base_only, 0, suffix > base)).astype(int)
    weights = np.where(suffix_only | base_only, 12.0, np.where(same_result, 0.25, 1.0))
    return labels, weights


def route_score(base: np.ndarray, suffix: np.ndarray, probability: np.ndarray, threshold: float):
    choose_suffix = probability >= threshold
    selected = np.where(choose_suffix, suffix, base)
    return {
        "wins": int(np.sum(selected > 0)),
        "ties": int(np.sum(selected == 0)),
        "losses": int(np.sum(selected < 0)),
        "score_rate": float(np.mean(selected > 0) + 0.5 * np.mean(selected == 0)),
        "mean_margin": float(np.mean(selected)),
        "switch_rate": float(np.mean(choose_suffix)),
    }


def probability(model, matrix: np.ndarray) -> np.ndarray:
    values = model.predict_proba(matrix)
    return values[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(matrix))


def choose_threshold(base, suffix, values):
    candidates = np.unique(np.concatenate((np.linspace(0.05, 0.95, 91), values)))
    rows = [(float(value), route_score(base, suffix, values, float(value))) for value in candidates]
    return max(rows, key=lambda row: (row[1]["score_rate"], row[1]["mean_margin"], row[0]))


def evaluate_config(factory, x, y, weights, groups, base, suffix):
    oof = np.zeros(len(x), dtype=float)
    for train_index, valid_index in GroupKFold(n_splits=5).split(x, y, groups):
        model = factory()
        model.fit(x[train_index], y[train_index], sample_weight=weights[train_index])
        oof[valid_index] = probability(model, x[valid_index])
    threshold, metrics = choose_threshold(base, suffix, oof)
    return threshold, metrics, oof


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test-a-features", type=Path, required=True)
    parser.add_argument("--test-a-outcomes", type=Path, required=True)
    parser.add_argument("--test-b-features", type=Path, required=True)
    parser.add_argument("--test-b-outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()

    x, base, suffix, groups, _, feature_names = assemble(args.train, args.train, 81)
    train_payload = json.loads(args.train.read_text(encoding="utf-8"))
    y, weights = labels_and_weights(base, suffix)
    test_a = assemble(args.test_a_features, args.test_a_outcomes, 81, feature_names)
    test_b = assemble(args.test_b_features, args.test_b_outcomes, 81, feature_names)

    configurations = []
    for depth in (1, 2, 3, 4, 5):
        for leaf in (8, 16, 32):
            name = f"tree_d{depth}_leaf{leaf}"
            factory = lambda depth=depth, leaf=leaf: DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=leaf,
                class_weight=None,
                random_state=20260821,
            )
            threshold, cv, _ = evaluate_config(factory, x, y, weights, groups, base, suffix)
            configurations.append(("tree", name, factory, threshold, cv))
    for leaves in (3, 5, 7):
        for min_child in (10, 20, 40):
            name = f"lgbm_leaves{leaves}_child{min_child}"
            factory = lambda leaves=leaves, min_child=min_child: lgb.LGBMClassifier(
                objective="binary",
                n_estimators=80,
                learning_rate=0.04,
                num_leaves=leaves,
                max_depth=3,
                min_child_samples=min_child,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=20260821,
                n_jobs=18,
                deterministic=True,
                verbosity=-1,
            )
            threshold, cv, _ = evaluate_config(factory, x, y, weights, groups, base, suffix)
            configurations.append(("lgbm", name, factory, threshold, cv))

    best_by_family = {}
    rows = []
    for family, name, factory, threshold, cv in configurations:
        model = factory()
        model.fit(x, y, sample_weight=weights)
        tests = {}
        for test_name, dataset in (("test_a", test_a), ("test_b", test_b)):
            test_x, test_base, test_suffix = dataset[0], dataset[1], dataset[2]
            tests[test_name] = route_score(
                test_base, test_suffix, probability(model, test_x), threshold
            )
        row = {"family": family, "name": name, "threshold": threshold, "cv": cv, "tests": tests}
        rows.append(row)
        candidate = best_by_family.get(family)
        if candidate is None or (cv["score_rate"], cv["mean_margin"]) > (
            candidate[4]["score_rate"], candidate[4]["mean_margin"]
        ):
            best_by_family[family] = (family, name, factory, threshold, cv)

    best = max(
        best_by_family.values(),
        key=lambda row: (row[4]["score_rate"], row[4]["mean_margin"]),
    )
    family, name, factory, threshold, cv = best
    final_model = factory()
    final_model.fit(x, y, sample_weight=weights)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": final_model, "feature_names": feature_names, "threshold": threshold, "family": family, "name": name},
        args.model_output,
    )

    def baseline_oracle(dataset):
        _, dataset_base, dataset_suffix = dataset[0], dataset[1], dataset[2]
        oracle = np.maximum(dataset_base, dataset_suffix)
        return {
            "baseline": route_score(dataset_base, dataset_suffix, np.zeros(len(dataset_base)), 0.5),
            "always_suffix": route_score(dataset_base, dataset_suffix, np.ones(len(dataset_base)), 0.5),
            "oracle": {
                "wins": int(np.sum(oracle > 0)),
                "ties": int(np.sum(oracle == 0)),
                "losses": int(np.sum(oracle < 0)),
                "score_rate": float(np.mean(oracle > 0) + 0.5 * np.mean(oracle == 0)),
                "mean_margin": float(np.mean(oracle)),
            },
        }

    chosen_row = next(row for row in rows if row["family"] == family and row["name"] == name)
    tree_best = best_by_family["tree"]
    tree_model = tree_best[2]()
    tree_model.fit(x, y, sample_weight=weights)
    payload = {
        "schema": "kaggriculture.fusion_champion.prt_suffix_selector_training.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "decision_step": int(train_payload["switch_step"]),
        "suffix_route_discovery_id": 81,
        "future_leakage": False,
        "group_cv": "5-fold grouped by episode seed",
        "train_samples": len(x),
        "train_unique_seeds": int(len(np.unique(groups))),
        "feature_count": len(feature_names),
        "critical_counts": {
            "base_only": int(np.sum((base > 0) & (suffix <= 0))),
            "suffix_only": int(np.sum((base <= 0) & (suffix > 0))),
            "both_win": int(np.sum((base > 0) & (suffix > 0))),
            "neither": int(np.sum((base <= 0) & (suffix <= 0))),
        },
        "baselines": {
            "train": baseline_oracle((x, base, suffix)),
            "test_a": baseline_oracle(test_a),
            "test_b": baseline_oracle(test_b),
        },
        "best_tree": {
            "name": tree_best[1],
            "threshold": tree_best[3],
            "cv": tree_best[4],
            "rules": export_text(tree_model, feature_names=feature_names),
        },
        "selected": chosen_row,
        "all_configurations": rows,
        "model_output": str(args.model_output.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "best_tree": payload["best_tree"],
        "selected": payload["selected"],
        "baselines": payload["baselines"],
        "output": str(args.output.resolve()),
        "model_output": str(args.model_output.resolve()),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
