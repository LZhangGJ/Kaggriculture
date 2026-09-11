#!/usr/bin/env python3
"""Fit and audit a public-state-only route30 gate from paired outcomes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor, export_text


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "gpu_sim/src"))
from kaggriculture_jax.constants import (  # noqa: E402
    ANIMALS,
    CROPS,
    PRODUCTS,
    SHED_ITEMS,
    SHOP_NAMES,
)


def flatten_feature(feature: dict) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []

    def add(name: str, value) -> None:
        names.append(name)
        values.append(float(value))

    for name in (
        "money_own",
        "money_rival",
        "hires_today_own",
        "hires_today_rival",
        "unlocked_count_own",
        "unlocked_count_rival",
        "active_units_own",
        "active_units_rival",
        "tile_yield_sum_own",
        "tile_yield_sum_rival",
    ):
        add(name, feature[name])
    add("money_gap", feature["money_own"] - feature["money_rival"])
    add("unlocked_gap", feature["unlocked_count_own"] - feature["unlocked_count_rival"])
    add("active_units_gap", feature["active_units_own"] - feature["active_units_rival"])
    add("tile_yield_gap", feature["tile_yield_sum_own"] - feature["tile_yield_sum_rival"])

    for side in ("own", "rival"):
        for index, item in enumerate(ANIMALS):
            add(f"animal_{side}_{item}", feature[f"animal_counts_{side}"][index])
        for index, item in enumerate(CROPS):
            add(f"crop_{side}_{item}", feature[f"crop_counts_{side}"][index])
    add("animal_total_gap", sum(feature["animal_counts_own"][:3]) - sum(feature["animal_counts_rival"][:3]))
    add("crop_total_gap", sum(feature["crop_counts_own"][:5]) - sum(feature["crop_counts_rival"][:5]))

    for index, item in enumerate(SHED_ITEMS):
        add(f"shed_own_{item}", feature["shed_own"][index])
    for index, item in enumerate(CROPS):
        add(f"seeds_own_{item}", feature["seeds_own"][index])
    for index, item in enumerate(PRODUCTS):
        add(f"market_inventory_{item}", feature["market_inventory"][index])
        add(f"market_price_{item}", feature["market_price"][index])
    shops = feature["town_shops"]
    add("town_shop_count", len(shops))
    for index, item in enumerate(SHOP_NAMES):
        add(f"shop_count_{item}", shops.count(index))
    return names, values


def official_replay_feature(path: Path, own_team: str, step: int) -> dict:
    replay = json.loads(path.read_text(encoding="utf-8"))
    names = list((replay.get("info") or {}).get("TeamNames") or [])
    own = names.index(own_team)
    rival = 1 - own
    obs = replay["steps"][step][own]["observation"]

    def tiles(player: int) -> list[dict]:
        return [
            tile
            for row in obs["farms"][player]["tiles"]
            for tile in row
            if isinstance(tile, dict)
        ]

    own_tiles = tiles(own)
    rival_tiles = tiles(rival)
    own_farm = obs["farms"][own]
    rival_farm = obs["farms"][rival]
    private = obs["private"]
    shops = [SHOP_NAMES.index(name) for name in obs["town"]["unlocked_shops"]]
    return {
        "state_step": step,
        "money_own": int(own_farm["money"]),
        "money_rival": int(rival_farm["money"]),
        "hires_today_own": int(own_farm["hires_today"]),
        "hires_today_rival": int(rival_farm["hires_today"]),
        "unlocked_count_own": int(sum(value != "LOCKED" for row in own_farm["unlocked_quadrants"] for value in row)),
        "unlocked_count_rival": int(sum(value != "LOCKED" for row in rival_farm["unlocked_quadrants"] for value in row)),
        "active_units_own": 1 + len(own_farm["hands"]),
        "active_units_rival": 1 + len(rival_farm["hands"]),
        "animal_counts_own": [sum(tile.get("animal") == item for tile in own_tiles) for item in ANIMALS] + [0] * (len(PRODUCTS) - len(ANIMALS)),
        "animal_counts_rival": [sum(tile.get("animal") == item for tile in rival_tiles) for item in ANIMALS] + [0] * (len(PRODUCTS) - len(ANIMALS)),
        "crop_counts_own": [sum(tile.get("crop") == item for tile in own_tiles) for item in PRODUCTS],
        "crop_counts_rival": [sum(tile.get("crop") == item for tile in rival_tiles) for item in PRODUCTS],
        "tile_yield_sum_own": int(sum(tile.get("yield_units", 0) for tile in own_tiles)),
        "tile_yield_sum_rival": int(sum(tile.get("yield_units", 0) for tile in rival_tiles)),
        "shed_own": [int(private["shed"].get(item, 0)) for item in SHED_ITEMS],
        "seeds_own": [int(private["seeds"].get(item, 0)) for item in CROPS],
        "market_inventory": [int(obs["market"]["inventory"].get(item, 0)) for item in PRODUCTS],
        "market_price": [int(obs["market"]["prices"].get(item, 0)) for item in PRODUCTS],
        "town_shops": shops,
    }


def choose_threshold(prediction: np.ndarray, control: np.ndarray, suffix: np.ndarray) -> tuple[float, dict]:
    candidates = np.unique(prediction)
    candidates = np.concatenate(([np.inf], candidates, [-np.inf]))
    best = None
    for threshold in candidates:
        selected = prediction > threshold
        margin = np.where(selected, suffix, control)
        destroyed = int(np.sum(selected & (control > 0) & (suffix <= 0)))
        rescued = int(np.sum(selected & (control <= 0) & (suffix > 0)))
        row = {
            "threshold": float(threshold),
            "wins": int(np.sum(margin > 0)),
            "ties": int(np.sum(margin == 0)),
            "losses": int(np.sum(margin < 0)),
            "score_rate": float(np.mean(margin > 0) + 0.5 * np.mean(margin == 0)),
            "mean_margin": float(np.mean(margin)),
            "selected": int(np.sum(selected)),
            "destroyed_control_wins": destroyed,
            "rescued_control_losses": rescued,
        }
        key = (row["wins"], -destroyed, row["mean_margin"], -row["selected"])
        if best is None or key > best[0]:
            best = (key, row)
    return best[1]["threshold"], best[1]


def metrics(control: np.ndarray, suffix: np.ndarray, selected: np.ndarray) -> dict:
    margin = np.where(selected, suffix, control)
    return {
        "games": int(margin.size),
        "wins": int(np.sum(margin > 0)),
        "ties": int(np.sum(margin == 0)),
        "losses": int(np.sum(margin < 0)),
        "score_rate": float(np.mean(margin > 0) + 0.5 * np.mean(margin == 0)),
        "mean_margin": float(np.mean(margin)),
        "selected": int(np.sum(selected)),
        "destroyed_control_wins": int(np.sum(selected & (control > 0) & (suffix <= 0))),
        "rescued_control_losses": int(np.sum(selected & (control <= 0) & (suffix > 0))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--old-receipts", type=Path, nargs="*", default=[])
    parser.add_argument("--public-episodes", type=Path, nargs="*", default=[])
    parser.add_argument("--own-team", default="QQ Farming")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    labels = list(payload["labels"])
    for receipt_path in args.old_receipts:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        control_row = next(row for row in receipt["rows"] if row["suffix_route"] < 0)
        suffix_row = next(row for row in receipt["rows"] if row["suffix_route"] == payload["suffix_route"])
        control_by_key = {
            (row["seed"], row["candidate_seat"]): row["margin"]
            for row in control_row["per_game"]
        }
        suffix_by_key = {
            (row["seed"], row["candidate_seat"]): row["margin"]
            for row in suffix_row["per_game"]
        }
        for feature in receipt["decision_features"]:
            key = (feature["seed"], feature["candidate_seat"])
            control_margin = int(control_by_key[key])
            suffix_margin = int(suffix_by_key[key])
            labels.append(
                {
                    "opponent": receipt["opponent"],
                    "seed": int(feature["seed"]),
                    "candidate_seat": int(feature["candidate_seat"]),
                    "control_margin": control_margin,
                    "suffix_margin": suffix_margin,
                    "margin_delta": suffix_margin - control_margin,
                    "suffix_better": suffix_margin > control_margin,
                    "control_win": control_margin > 0,
                    "suffix_win": suffix_margin > 0,
                    "decision_features": {
                        key: value
                        for key, value in feature.items()
                        if key not in ("seed", "candidate_seat")
                    },
                }
            )
    feature_names, first = flatten_feature(labels[0]["decision_features"])
    x = np.asarray([flatten_feature(row["decision_features"])[1] for row in labels], dtype=np.float64)
    control = np.asarray([row["control_margin"] for row in labels], dtype=np.float64)
    suffix = np.asarray([row["suffix_margin"] for row in labels], dtype=np.float64)
    target = np.clip(suffix - control, -30000, 30000)
    groups = np.asarray([row["seed"] for row in labels])

    candidates = [
        (depth, leaf)
        for depth in (1, 2, 3, 4)
        for leaf in (8, 16, 32, 64)
    ]
    model_rows = []
    splitter = GroupKFold(n_splits=4)
    for depth, leaf in candidates:
        oof_prediction = np.zeros(len(labels), dtype=np.float64)
        oof_selected = np.zeros(len(labels), dtype=bool)
        fold_rows = []
        for fold, (train, valid) in enumerate(splitter.split(x, target, groups)):
            model = DecisionTreeRegressor(
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=20260823 + fold,
            )
            model.fit(x[train], target[train])
            train_prediction = model.predict(x[train])
            threshold, train_metrics = choose_threshold(
                train_prediction, control[train], suffix[train]
            )
            valid_prediction = model.predict(x[valid])
            selected = valid_prediction > threshold
            oof_prediction[valid] = valid_prediction
            oof_selected[valid] = selected
            fold_rows.append(
                {
                    "fold": fold,
                    "threshold": threshold,
                    "train": train_metrics,
                    "valid": metrics(control[valid], suffix[valid], selected),
                }
            )
        row = {
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "oof": metrics(control, suffix, oof_selected),
            "folds": fold_rows,
        }
        model_rows.append(row)

    lgbm_rows = []
    binary_targets = {
        "flip": ((control <= 0) & (suffix > 0)).astype(np.int8),
        "better": (suffix > control).astype(np.int8),
    }
    for target_name, binary_target in binary_targets.items():
        for leaves in (3, 5, 7):
            for min_child in (8, 16, 32):
                for positive_weight in (1.0, 4.0, 12.0):
                    oof_selected = np.zeros(len(labels), dtype=bool)
                    fold_rows = []
                    for fold, (train, valid) in enumerate(
                        splitter.split(x, binary_target, groups)
                    ):
                        model = LGBMClassifier(
                            objective="binary",
                            n_estimators=40,
                            learning_rate=0.05,
                            num_leaves=leaves,
                            min_child_samples=min_child,
                            reg_lambda=2.0,
                            reg_alpha=0.5,
                            scale_pos_weight=positive_weight,
                            verbosity=-1,
                            deterministic=True,
                            force_col_wise=True,
                            random_state=20260823 + fold,
                            n_jobs=4,
                        )
                        model.fit(x[train], binary_target[train])
                        train_prediction = model.predict_proba(x[train])[:, 1]
                        threshold, train_metrics = choose_threshold(
                            train_prediction, control[train], suffix[train]
                        )
                        valid_prediction = model.predict_proba(x[valid])[:, 1]
                        selected = valid_prediction > threshold
                        oof_selected[valid] = selected
                        fold_rows.append(
                            {
                                "fold": fold,
                                "threshold": threshold,
                                "train": train_metrics,
                                "valid": metrics(
                                    control[valid], suffix[valid], selected
                                ),
                            }
                        )
                    lgbm_rows.append(
                        {
                            "target": target_name,
                            "num_leaves": leaves,
                            "min_child_samples": min_child,
                            "scale_pos_weight": positive_weight,
                            "oof": metrics(control, suffix, oof_selected),
                            "folds": fold_rows,
                        }
                    )

    baseline = metrics(control, suffix, np.zeros(len(labels), dtype=bool))
    oracle = metrics(control, suffix, suffix > control)
    best_tree = max(
        model_rows,
        key=lambda row: (
            row["oof"]["wins"],
            -row["oof"]["destroyed_control_wins"],
            row["oof"]["mean_margin"],
            -row["oof"]["selected"],
        ),
    )
    best_lgbm = max(
        lgbm_rows,
        key=lambda row: (
            row["oof"]["wins"],
            -row["oof"]["destroyed_control_wins"],
            row["oof"]["mean_margin"],
            -row["oof"]["selected"],
        ),
    )
    best_family = "lgbm" if (
        best_lgbm["oof"]["wins"],
        -best_lgbm["oof"]["destroyed_control_wins"],
        best_lgbm["oof"]["mean_margin"],
        -best_lgbm["oof"]["selected"],
    ) > (
        best_tree["oof"]["wins"],
        -best_tree["oof"]["destroyed_control_wins"],
        best_tree["oof"]["mean_margin"],
        -best_tree["oof"]["selected"],
    ) else "tree"

    final_tree = DecisionTreeRegressor(
        max_depth=best_tree["max_depth"],
        min_samples_leaf=best_tree["min_samples_leaf"],
        random_state=20260823,
    ).fit(x, target)
    final_lgbm_target = binary_targets[best_lgbm["target"]]
    final_lgbm = LGBMClassifier(
        objective="binary",
        n_estimators=40,
        learning_rate=0.05,
        num_leaves=best_lgbm["num_leaves"],
        min_child_samples=best_lgbm["min_child_samples"],
        reg_lambda=2.0,
        reg_alpha=0.5,
        scale_pos_weight=best_lgbm["scale_pos_weight"],
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        random_state=20260823,
        n_jobs=4,
    ).fit(x, final_lgbm_target)
    if best_family == "lgbm":
        final_prediction = final_lgbm.predict_proba(x)[:, 1]
    else:
        final_prediction = final_tree.predict(x)
    final_threshold, in_sample = choose_threshold(
        final_prediction, control, suffix
    )

    public_rows = []
    for path in args.public_episodes:
        feature = official_replay_feature(path, args.own_team, payload["switch_step"])
        names, values = flatten_feature(feature)
        if names != feature_names:
            raise AssertionError("public feature schema mismatch")
        if best_family == "lgbm":
            prediction = float(final_lgbm.predict_proba(np.asarray([values]))[0, 1])
        else:
            prediction = float(final_tree.predict(np.asarray([values]))[0])
        public_rows.append(
            {
                "episode": int(json.loads(path.read_text(encoding="utf-8"))["info"]["EpisodeId"]),
                "path": str(path.resolve()),
                "prediction": prediction,
                "selected": prediction > final_threshold,
                "feature": feature,
            }
        )

    result = {
        "schema": "kaggriculture.fc17-ueddy-gate-fit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source_dataset": str(args.dataset.resolve()),
        "old_receipts": [str(path.resolve()) for path in args.old_receipts],
        "feature_names": feature_names,
        "baseline": baseline,
        "oracle": oracle,
        "models": model_rows,
        "lgbm_models": lgbm_rows,
        "best_tree": best_tree,
        "best_lgbm": best_lgbm,
        "best_family": best_family,
        "final": {
            "threshold": final_threshold,
            "in_sample": in_sample,
            "tree_text": export_text(final_tree, feature_names=feature_names),
            "tree_feature": final_tree.tree_.feature.tolist(),
            "tree_threshold": final_tree.tree_.threshold.tolist(),
            "tree_children_left": final_tree.tree_.children_left.tolist(),
            "tree_children_right": final_tree.tree_.children_right.tolist(),
            "tree_value": final_tree.tree_.value.reshape(-1).tolist(),
            "lgbm_dump": final_lgbm.booster_.dump_model(),
        },
        "public_replay_external_rows": public_rows,
        "truth_boundary": (
            "OOF folds are grouped by event seed. Public Replay rows are external feature probes; "
            "their frozen-opponent counterfactual benefit is not mixed into model fitting."
        ),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "baseline": baseline,
                "oracle": oracle,
                "best_family": best_family,
                "best_tree": best_tree["oof"],
                "best_lgbm": best_lgbm["oof"],
                "best_config": best_lgbm if best_family == "lgbm" else {
                    "max_depth": best_tree["max_depth"],
                    "min_samples_leaf": best_tree["min_samples_leaf"],
                },
                "final_threshold": final_threshold,
                "public": [
                    {key: row[key] for key in ("episode", "prediction", "selected")}
                    for row in public_rows
                ],
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
