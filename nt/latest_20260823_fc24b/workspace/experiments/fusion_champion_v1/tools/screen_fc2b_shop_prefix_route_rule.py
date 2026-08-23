#!/usr/bin/env python3
"""Fit an interpretable public-shop-prefix route rule for FC2B.

This deliberately does *not* learn the per-game counterfactual oracle label.
Games are grouped by the ordered shops that are already public at the decision
step.  For every prefix, the rule compares the empirical paired score of each
existing route with the adaptive FC2B baseline and switches only when support
and a conservative lower confidence bound clear train-selected thresholds.

The runtime rule therefore needs only the public ordered shop prefix.  Future
shops, terminal cash, opponent identity, event seed and Replay metadata are
never policy inputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/fusion_champion_v1/tools"))
from train_k320_route_rescue_tree import _load, _metrics  # noqa: E402


SHOP_NAMES = (
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
)
ROUTE_NAMES = (
    "adaptive_source",
    "route0_milk_support",
    "route1_default",
    "route2_three_yarn",
    "route3_first_yarn",
    "route4_two_yarn",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load(path: Path):
    payload, _, margins, groups, keys, _ = _load(path)
    feature_by_key = {
        (int(row["seed"]), int(row["candidate_seat"])): row
        for row in payload["decision_features"]
    }
    prefixes = [tuple(int(value) for value in feature_by_key[key]["town_shops"]) for key in keys]
    lengths = {len(prefix) for prefix in prefixes}
    if len(lengths) != 1:
        raise AssertionError(f"mixed shop-prefix lengths in {path}: {sorted(lengths)}")
    return payload, margins, groups, keys, prefixes


def score_values(margins: np.ndarray) -> np.ndarray:
    return (margins > 0).astype(np.float64) + 0.5 * (margins == 0)


def fit_mapping(
    prefixes: list[tuple[int, ...]],
    margins: np.ndarray,
    indices: np.ndarray,
    min_support: int,
    min_gain: float,
    confidence_z: float,
) -> dict[tuple[int, ...], dict]:
    outcomes = score_values(margins)
    mapping: dict[tuple[int, ...], dict] = {}
    for prefix in sorted({prefixes[index] for index in indices}):
        selected = np.asarray(
            [index for index in indices if prefixes[index] == prefix], dtype=np.int64
        )
        if len(selected) < min_support:
            continue
        baseline = outcomes[0, selected]
        candidates = []
        for arm in range(1, margins.shape[0]):
            delta = outcomes[arm, selected] - baseline
            mean = float(np.mean(delta))
            standard_error = float(np.std(delta, ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 1.0
            lower_bound = mean - confidence_z * standard_error
            candidates.append(
                {
                    "arm": arm,
                    "support": int(len(selected)),
                    "mean_score_gain": mean,
                    "standard_error": standard_error,
                    "lower_bound": lower_bound,
                    "baseline_score": float(np.mean(baseline)),
                    "route_score": float(np.mean(outcomes[arm, selected])),
                }
            )
        best = max(candidates, key=lambda row: (row["lower_bound"], row["mean_score_gain"], -row["arm"]))
        if best["mean_score_gain"] >= min_gain and best["lower_bound"] > 0.0:
            mapping[prefix] = best
    return mapping


def choices(prefixes: list[tuple[int, ...]], mapping: dict[tuple[int, ...], dict]) -> np.ndarray:
    return np.asarray([mapping.get(prefix, {"arm": 0})["arm"] for prefix in prefixes], dtype=np.int64)


def evaluate(prefixes: list[tuple[int, ...]], margins: np.ndarray, mapping: dict) -> dict:
    return _metrics(margins, choices(prefixes, mapping))


def serialise_mapping(mapping: dict[tuple[int, ...], dict]) -> list[dict]:
    rows = []
    for prefix, detail in sorted(mapping.items()):
        rows.append(
            {
                "shop_prefix_ids": list(prefix),
                "shop_prefix": [SHOP_NAMES[value] for value in prefix],
                "route_arm": int(detail["arm"]),
                "route": ROUTE_NAMES[int(detail["arm"])],
                **{key: value for key, value in detail.items() if key != "arm"},
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout-a", type=Path, required=True)
    parser.add_argument("--holdout-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_payload, margins, groups, _, prefixes = load(args.train)
    _, amargins, _, _, aprefixes = load(args.holdout_a)
    _, bmargins, _, _, bprefixes = load(args.holdout_b)
    prefix_lengths = {len(prefixes[0]), len(aprefixes[0]), len(bprefixes[0])}
    if len(prefix_lengths) != 1:
        raise AssertionError("train/holdout shop-prefix lengths differ")

    configurations = []
    dummy = np.zeros(len(prefixes), dtype=np.int8)
    for min_support in (4, 8, 12, 16, 24, 32):
        for min_gain in (0.0, 0.02, 0.05, 0.10):
            for confidence_z in (0.0, 0.5, 1.0, 1.645):
                oof_choices = np.zeros(len(prefixes), dtype=np.int64)
                fold_rows = []
                for fold, (fit_index, valid_index) in enumerate(
                    GroupKFold(n_splits=5).split(dummy, groups=groups)
                ):
                    mapping = fit_mapping(
                        prefixes,
                        margins,
                        fit_index,
                        min_support,
                        min_gain,
                        confidence_z,
                    )
                    valid_prefixes = [prefixes[index] for index in valid_index]
                    oof_choices[valid_index] = choices(valid_prefixes, mapping)
                    fold_rows.append(
                        {
                            "fold": fold,
                            "mapping_size": len(mapping),
                            "valid_games": int(len(valid_index)),
                        }
                    )
                metric = _metrics(margins, oof_choices)
                configurations.append(
                    {
                        "min_support": min_support,
                        "min_gain": min_gain,
                        "confidence_z": confidence_z,
                        "oof": metric,
                        "folds": fold_rows,
                    }
                )

    best = max(
        configurations,
        key=lambda row: (
            row["oof"]["score_rate"],
            -row["oof"]["harmed_wins"],
            -row["oof"]["switch_rate"],
            row["oof"]["mean_margin"],
            row["min_support"],
            row["min_gain"],
            row["confidence_z"],
        ),
    )
    mapping = fit_mapping(
        prefixes,
        margins,
        np.arange(len(prefixes), dtype=np.int64),
        int(best["min_support"]),
        float(best["min_gain"]),
        float(best["confidence_z"]),
    )
    payload = {
        "schema": "kaggriculture.fusion_champion.fc2b-shop-prefix-route-rule.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "official_package_version": "1.32.7",
        "decision_step": int(train_payload["feature_step"]),
        "shop_prefix_length": len(prefixes[0]),
        "future_leakage": False,
        "runtime_inputs": "ordered public town shops available at decision step only",
        "selection_protocol": (
            "configuration selected by 5-fold OOF grouped by event seed; "
            "both seats remain together; mapping refit on train only; "
            "holdout A/B untouched during selection"
        ),
        "sources": [
            {"role": role, "path": str(path.resolve()), "sha256": sha256(path)}
            for role, path in (
                ("train", args.train),
                ("holdout_a", args.holdout_a),
                ("holdout_b", args.holdout_b),
            )
        ],
        "train_unique_seeds": int(len(np.unique(groups))),
        "train_games": len(prefixes),
        "selected_configuration": best,
        "mapping": serialise_mapping(mapping),
        "train_fit": evaluate(prefixes, margins, mapping),
        "holdout_a": evaluate(aprefixes, amargins, mapping),
        "holdout_b": evaluate(bprefixes, bmargins, mapping),
        "all_configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_configuration": best,
                "mapping": payload["mapping"],
                "train_fit": payload["train_fit"],
                "holdout_a": payload["holdout_a"],
                "holdout_b": payload["holdout_b"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
