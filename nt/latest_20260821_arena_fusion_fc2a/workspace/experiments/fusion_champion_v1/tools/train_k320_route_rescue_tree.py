#!/usr/bin/env python3
"""Fit a conservative, public-state-only K320 route rescue decision tree.

The label is deliberately not the highest terminal margin in every game.  If
adaptive K320 already wins, the target remains adaptive K320.  A route override
is labelled only when the baseline loses and that override turns the game into
a win.  This encodes the fusion goal directly: rescue a shortfall without
breaking games that are already healthy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/fusion_champion_v1/tools"))
from train_prt_suffix_selector import build_feature_row  # noqa: E402


def _load(path: Path, feature_names: list[str] | None = None):
    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_rows = payload.get("decision_features", [])
    if not feature_rows:
        raise AssertionError(f"missing decision features: {path}")
    built = [build_feature_row(row) for row in feature_rows]
    if feature_names is None:
        feature_names = sorted(built[0])
    if any(sorted(row) != feature_names for row in built):
        raise AssertionError("feature schema mismatch")
    keys = [(int(row["seed"]), int(row["candidate_seat"])) for row in feature_rows]
    x = np.asarray([[row[name] for name in feature_names] for row in built], dtype=np.float32)
    rows = sorted(payload["rows"], key=lambda row: int(row["route_id"]))
    baseline = next(row for row in rows if int(row["route_id"]) == -1)
    route_rows = [row for row in rows if int(row["route_id"]) >= 0]
    if [int(row["route_id"]) for row in route_rows] != list(range(5)):
        raise AssertionError("expected route IDs 0..4")

    def margins(row):
        lookup = {
            (int(item["seed"]), int(item["candidate_seat"])): int(item["margin"])
            for item in row["per_game"]
        }
        if any(key not in lookup for key in keys):
            raise AssertionError("feature/outcome key mismatch")
        return np.asarray([lookup[key] for key in keys], dtype=np.int64)

    margin_matrix = np.stack([margins(baseline)] + [margins(row) for row in route_rows])
    groups = np.asarray([key[0] for key in keys], dtype=np.int64)
    return payload, x, margin_matrix, groups, keys, feature_names


def _targets(margins: np.ndarray) -> np.ndarray:
    base = margins[0]
    labels = np.zeros(base.shape, dtype=np.int64)
    for index in range(len(base)):
        if base[index] > 0:
            continue
        rescuers = np.flatnonzero(margins[1:, index] > 0)
        if rescuers.size:
            best = rescuers[np.argmax(margins[1 + rescuers, index])]
            labels[index] = 1 + int(best)
    return labels


def _metrics(margins: np.ndarray, choices: np.ndarray) -> dict:
    selected = margins[choices, np.arange(margins.shape[1])]
    base = margins[0]
    return {
        "games": int(len(selected)),
        "wins": int(np.sum(selected > 0)),
        "ties": int(np.sum(selected == 0)),
        "losses": int(np.sum(selected < 0)),
        "score_rate": float(np.mean(selected > 0) + 0.5 * np.mean(selected == 0)),
        "mean_margin": float(np.mean(selected)),
        "switch_rate": float(np.mean(choices != 0)),
        "rescued_losses": int(np.sum((base <= 0) & (selected > 0))),
        "harmed_wins": int(np.sum((base > 0) & (selected <= 0))),
        "choice_histogram": np.bincount(choices, minlength=6).astype(int).tolist(),
    }


def _baseline_oracle(margins: np.ndarray) -> dict:
    base_choice = np.zeros(margins.shape[1], dtype=np.int64)
    conservative_oracle = _targets(margins)
    unrestricted = np.argmax(margins, axis=0)
    return {
        "baseline": _metrics(margins, base_choice),
        "conservative_rescue_oracle": _metrics(margins, conservative_oracle),
        "unrestricted_margin_oracle": _metrics(margins, unrestricted),
        "rescue_label_histogram": np.bincount(conservative_oracle, minlength=6).astype(int).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()

    train_payload, x, margins, groups, _, feature_names = _load(args.train)
    _, hx, hmargins, _, _, holdout_names = _load(args.holdout, feature_names)
    if holdout_names != feature_names:
        raise AssertionError("holdout feature mismatch")
    y = _targets(margins)

    configurations = []
    for depth in (1, 2, 3, 4, 5, 6):
        for leaf in (2, 4, 8, 12):
            for rescue_weight in (2.0, 4.0, 8.0, 12.0, 20.0):
                oof = np.zeros(len(y), dtype=np.int64)
                weights = np.where(y == 0, 1.0, rescue_weight)
                for fit_index, valid_index in GroupKFold(n_splits=5).split(x, y, groups):
                    model = DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        random_state=20260821,
                    )
                    model.fit(x[fit_index], y[fit_index], sample_weight=weights[fit_index])
                    oof[valid_index] = model.predict(x[valid_index]).astype(np.int64)
                metrics = _metrics(margins, oof)
                configurations.append({
                    "depth": depth,
                    "min_samples_leaf": leaf,
                    "rescue_weight": rescue_weight,
                    "oof": metrics,
                })

    # Win rate is the hard objective.  At equal win rate prefer no harm, fewer
    # switches and finally higher mean margin.
    best = max(
        configurations,
        key=lambda row: (
            row["oof"]["score_rate"],
            -row["oof"]["harmed_wins"],
            -row["oof"]["switch_rate"],
            row["oof"]["mean_margin"],
        ),
    )
    weights = np.where(y == 0, 1.0, best["rescue_weight"])
    model = DecisionTreeClassifier(
        max_depth=best["depth"],
        min_samples_leaf=best["min_samples_leaf"],
        random_state=20260821,
    )
    model.fit(x, y, sample_weight=weights)
    train_fit = _metrics(margins, model.predict(x).astype(np.int64))
    holdout = _metrics(hmargins, model.predict(hx).astype(np.int64))
    rules = export_text(model, feature_names=feature_names, decimals=3)

    payload = {
        "schema": "kaggriculture.fusion_champion.k320-route-rescue-tree.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "official_package_version": "1.32.7",
        "decision_step": int(train_payload["feature_step"]),
        "future_leakage": False,
        "label_semantics": "keep baseline if it wins; otherwise choose highest-margin winning route; otherwise baseline",
        "choice_mapping": {
            "0": "adaptive_source",
            "1": "route0_milk_support",
            "2": "route1_default",
            "3": "route2_three_yarn",
            "4": "route3_first_yarn",
            "5": "route4_two_yarn",
        },
        "feature_count": len(feature_names),
        "train_samples": len(x),
        "train_unique_seeds": int(len(np.unique(groups))),
        "baselines": {
            "train": _baseline_oracle(margins),
            "holdout": _baseline_oracle(hmargins),
        },
        "selected_configuration": best,
        "train_fit": train_fit,
        "holdout": holdout,
        "rules": rules,
        "all_configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        "choice_mapping": payload["choice_mapping"],
        "decision_step": payload["decision_step"],
    }, args.model_output)
    print(json.dumps({
        "status": "PASS",
        "selected_configuration": best,
        "train_fit": train_fit,
        "holdout": holdout,
        "baselines": payload["baselines"],
        "rules": rules,
        "output": str(args.output.resolve()),
        "model_output": str(args.model_output.resolve()),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
