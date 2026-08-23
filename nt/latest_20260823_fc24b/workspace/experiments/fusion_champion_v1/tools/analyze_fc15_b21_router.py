#!/usr/bin/env python3
"""Leakage-safe OOF routing study for FC15 versus the B21 counter branch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text


ROOT = Path(__file__).resolve().parents[3]


def flatten(record: dict) -> tuple[list[float], list[str]]:
    values: list[float] = []
    names: list[str] = []
    for key in (
        "step",
        "money",
        "hires_today",
        "town_count",
        "town_shops",
        "market_price",
        "market_inventory",
        "own_shed",
        "own_crops",
        "rival_crops",
        "own_animals",
        "rival_animals",
        "unit_count",
        "fc15_route",
        "counter_route",
        "counter_overlay",
        "fc15_sales",
        "counter_sales",
    ):
        value = record[key]
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        values.extend(array.tolist())
        names.extend([key] if array.size == 1 else [f"{key}_{index}" for index in range(array.size)])
    return values, names


def metrics(margins: np.ndarray) -> dict:
    return {
        "wins": int(np.sum(margins > 0)),
        "ties": int(np.sum(margins == 0)),
        "losses": int(np.sum(margins < 0)),
        "win_rate": float(np.mean(margins > 0)),
        "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "experiments/fusion_champion_v1/artifacts/fc15_b21_divergence_features_seed823001_n128x2_v1.json",
    )
    parser.add_argument(
        "--fc15",
        type=Path,
        default=ROOT / "experiments/fusion_champion_v1/receipts/fc15_vs_b21_seed823001_n128x2_diagnostics_v2.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    feature_payload = json.loads(args.features.read_text(encoding="utf-8"))
    fc15_payload = json.loads(args.fc15.read_text(encoding="utf-8"))
    fc15_games = {
        (row["seed"], row["candidate_seat"]): row["margin"]
        for row in fc15_payload["rows"][0]["per_game"]
    }
    counter_games = {}
    counter_paths = sorted(
        glob.glob(
            str(
                ROOT
                / "experiments/fusion_champion_v1/receipts/b21_horizon_counter_seed823*_n16x2_pergame_v1.json"
            )
        )
    )
    for path in counter_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        row = next(item for item in payload["rows"] if item["config_id"] == 3)
        counter_games.update(
            {
                (game["seed"], game["candidate_seat"]): game["margin"]
                for game in row["per_game"]
            }
        )

    records = sorted(
        feature_payload["records"],
        key=lambda row: (row["seed"], row["candidate_seat"]),
    )
    if not records or any(not row["divergence_found"] for row in records):
        raise AssertionError("every game must expose an FC15/counter divergence")
    features, feature_names = zip(*(flatten(row) for row in records), strict=True)
    if any(names != feature_names[0] for names in feature_names):
        raise AssertionError("feature order mismatch")
    names = feature_names[0]
    x = np.asarray(features, dtype=np.float32)
    keys = [(row["seed"], row["candidate_seat"]) for row in records]
    groups = np.asarray([row["seed"] for row in records], dtype=np.int64)
    base = np.asarray([fc15_games[key] for key in keys], dtype=np.int64)
    counter = np.asarray([counter_games[key] for key in keys], dtype=np.int64)
    target = (counter > base).astype(np.int8)
    folds = list(GroupKFold(n_splits=4).split(x, target, groups))

    candidates = []
    for depth in range(1, 7):
        for leaf in (4, 8, 16, 24):
            candidates.append(
                (
                    f"tree_d{depth}_leaf{leaf}",
                    lambda depth=depth, leaf=leaf: DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        class_weight="balanced",
                        random_state=20260822,
                    ),
                    {"kind": "decision_tree", "max_depth": depth, "min_samples_leaf": leaf},
                )
            )
    for leaves in (3, 7, 15):
        for estimators in (20, 40, 80):
            candidates.append(
                (
                    f"lgbm_l{leaves}_n{estimators}",
                    lambda leaves=leaves, estimators=estimators: lgb.LGBMClassifier(
                        objective="binary",
                        n_estimators=estimators,
                        num_leaves=leaves,
                        max_depth=4,
                        learning_rate=0.05,
                        min_child_samples=16,
                        reg_lambda=1.0,
                        verbosity=-1,
                        deterministic=True,
                        force_col_wise=True,
                        random_state=20260822,
                    ),
                    {"kind": "lightgbm", "num_leaves": leaves, "n_estimators": estimators},
                )
            )

    rows = []
    predictions_by_name = {}
    for name, factory, spec in candidates:
        prediction = np.zeros(target.shape, dtype=np.int8)
        for train, valid in folds:
            model = factory()
            model.fit(x[train], target[train])
            prediction[valid] = model.predict(x[valid]).astype(np.int8)
        selected = np.where(prediction > 0, counter, base)
        row = {
            "name": name,
            "spec": spec,
            "label_accuracy": float(np.mean(prediction == target)),
            "counter_selection_rate": float(np.mean(prediction > 0)),
            "selected": metrics(selected),
            "rescued_base_losses": int(np.sum((base < 0) & (selected > 0))),
            "spoiled_base_wins": int(np.sum((base > 0) & (selected < 0))),
        }
        rows.append(row)
        predictions_by_name[name] = prediction

    rows.sort(
        key=lambda row: (
            row["selected"]["wins"],
            row["selected"]["score_rate"],
            row["selected"]["mean_margin"],
            -row["spec"].get("n_estimators", 0),
        ),
        reverse=True,
    )
    best = rows[0]
    best_factory = next(factory for name, factory, _ in candidates if name == best["name"])
    final_model = best_factory()
    final_model.fit(x, target)
    if isinstance(final_model, DecisionTreeClassifier):
        final_description = export_text(final_model, feature_names=list(names))
    else:
        importance = sorted(
            zip(names, final_model.feature_importances_.tolist(), strict=True),
            key=lambda item: item[1],
            reverse=True,
        )[:20]
        final_description = "\n".join(f"{name}: {value}" for name, value in importance)

    payload = {
        "schema": "kaggriculture.fc15-b21-public-router-oof.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "visibility": feature_payload["visibility"],
        "samples": int(x.shape[0]),
        "independent_seed_groups": int(np.unique(groups).size),
        "features": int(x.shape[1]),
        "target_counter_rate": float(np.mean(target)),
        "baseline_fc15": metrics(base),
        "counter_branch": metrics(counter),
        "oracle_two_branch": metrics(np.maximum(base, counter)),
        "best_oof": best,
        "best_full_fit_description": final_description,
        "rows": rows,
        "counter_receipts": counter_paths,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "best": best}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
