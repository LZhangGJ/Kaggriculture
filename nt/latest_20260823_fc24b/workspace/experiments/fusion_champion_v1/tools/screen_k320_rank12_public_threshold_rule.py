#!/usr/bin/env python3
"""Screen one-condition, public-state K320 route-rescue rules.

The rule family is intentionally small and auditable::

    if PUBLIC_FEATURE <=/> THRESHOLD:
        lock one existing K320 route at step 120
    else:
        keep adaptive K320

Only the training event bank participates in feature/direction/route/threshold
selection.  Two independent event banks are reported exactly once as holdouts.
Both seats of a seed stay in the same GroupKFold fold.  No terminal or future
state is present in the online rule.
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
from train_k320_route_rescue_tree import _baseline_oracle, _load, _metrics  # noqa: E402


# Hypothesis-driven public features only.  This is deliberately not a scan of
# all 209 generated features: each item has a direct market/production meaning.
FEATURES = (
    "market_inventory_6",       # milk supply pressure
    "market_inventory_7",       # wool supply pressure
    "market_price_6",
    "market_price_7",
    "market_price_ratio_6",
    "market_price_ratio_7",
    "shop_3_count",             # ice-cream demand
    "shop_7_count",             # yarn demand
    "shop_first_3",
    "shop_first_7",
    "shop_second_3",
    "shop_second_7",
    "own_animal_1_count",       # own cows
    "own_animal_2_count",       # own sheep
    "rival_animal_1_count",
    "rival_animal_2_count",
    "animal_1_count_gap",
    "animal_2_count_gap",
    "own_animal_1_yield",
    "own_animal_2_yield",
    "rival_animal_1_yield",
    "rival_animal_2_yield",
    "animal_1_yield_gap",
    "animal_2_yield_gap",
    "shed_6",
    "shed_7",
    "unit_inventory_6",
    "unit_inventory_7",
    "money_gap",
)


def all_public_features() -> tuple[str, ...]:
    """Return a systematic, interpretable one-feature screen.

    The original profile above was intentionally limited to the milk/wool
    hypothesis used by the Rank12 audit.  That is too narrow for a general PRT
    audit: every official shop can change demand for a different product.  This
    profile still permits only one public feature, one threshold and one
    existing route arm, but it treats all official products and shops
    symmetrically instead of selecting categories after looking at outcomes.
    """

    features = [
        "candidate_seat",
        "money_own",
        "money_rival",
        "money_gap",
        "hires_today_own",
        "hires_today_rival",
        "hires_gap",
        "unlocked_count_own",
        "unlocked_count_rival",
        "unlocked_gap",
    ]
    for product in range(9):
        features.extend(
            (
                f"market_inventory_{product}",
                f"market_price_{product}",
                f"market_price_ratio_{product}",
                f"shed_{product}",
                f"unit_inventory_{product}",
            )
        )
    for shop in range(8):
        features.extend(
            (
                f"shop_{shop}_count",
                f"shop_first_{shop}",
                f"shop_second_{shop}",
            )
        )
    for side in ("own", "rival"):
        features.extend(
            (
                f"{side}_unit_count",
                f"{side}_unit_shed_distance_mean",
                f"{side}_unit_shed_distance_max",
            )
        )
        for crop in range(5):
            features.extend(
                (
                    f"{side}_crop_{crop}_count",
                    f"{side}_crop_{crop}_yield",
                    f"{side}_crop_{crop}_neglect",
                )
            )
        for animal in range(3):
            features.extend(
                (
                    f"{side}_animal_{animal}_count",
                    f"{side}_animal_{animal}_yield",
                    f"{side}_animal_{animal}_neglect",
                )
            )
    for crop in range(5):
        features.extend((f"crop_{crop}_count_gap", f"crop_{crop}_yield_gap"))
    for animal in range(3):
        features.extend((f"animal_{animal}_count_gap", f"animal_{animal}_yield_gap"))
    return tuple(dict.fromkeys(features))

ROUTE_NAMES = {
    1: "route0_milk_support",
    2: "route1_default",
    3: "route2_three_yarn",
    4: "route3_first_yarn",
    5: "route4_two_yarn",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def thresholds(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values.astype(np.float64))
    if len(unique) == 1:
        return unique
    mids = (unique[:-1] + unique[1:]) / 2.0
    return np.concatenate(([unique[0] - 1.0], mids, [unique[-1] + 1.0]))


def choices(values: np.ndarray, direction: str, threshold: float, arm: int) -> np.ndarray:
    mask = values <= threshold if direction == "le" else values > threshold
    return np.where(mask, arm, 0).astype(np.int64)


def selection_key(metrics: dict, threshold: float) -> tuple:
    # Score rate is the hard target.  Equal-score rules prefer fewer broken
    # baseline wins, fewer switches, more mean margin, then a smaller absolute
    # threshold only as a deterministic final tie break.
    return (
        metrics["score_rate"],
        -metrics["harmed_wins"],
        -metrics["switch_rate"],
        metrics["mean_margin"],
        -abs(float(threshold)),
    )


def fit_threshold(values: np.ndarray, margins: np.ndarray, direction: str, arm: int) -> dict:
    rows = []
    for threshold in thresholds(values):
        metric = _metrics(margins, choices(values, direction, float(threshold), arm))
        rows.append({"threshold": float(threshold), "metrics": metric})
    return max(rows, key=lambda row: selection_key(row["metrics"], row["threshold"]))


def grouped_oof(
    values: np.ndarray,
    margins: np.ndarray,
    groups: np.ndarray,
    direction: str,
    arm: int,
) -> tuple[dict, list[dict]]:
    predicted = np.zeros(len(values), dtype=np.int64)
    folds = []
    dummy = np.zeros(len(values), dtype=np.int8)
    for fold, (fit_index, valid_index) in enumerate(
        GroupKFold(n_splits=5).split(values, dummy, groups)
    ):
        fitted = fit_threshold(values[fit_index], margins[:, fit_index], direction, arm)
        predicted[valid_index] = choices(
            values[valid_index], direction, fitted["threshold"], arm
        )
        folds.append(
            {
                "fold": fold,
                "threshold": fitted["threshold"],
                "fit_metrics": fitted["metrics"],
                "valid_games": int(len(valid_index)),
            }
        )
    return _metrics(margins, predicted), folds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout-a", type=Path, required=True)
    parser.add_argument("--holdout-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-profile",
        choices=("targeted_milk_wool", "all_official"),
        default="targeted_milk_wool",
        help=(
            "Keep the historical milk/wool audit by default; all_official "
            "screens the same one-threshold rule over every official "
            "product/shop and public production aggregate."
        ),
    )
    args = parser.parse_args()

    train_payload, x, margins, groups, _, names = _load(args.train)
    _, ax, amargins, _, _, anames = _load(args.holdout_a, names)
    _, bx, bmargins, _, _, bnames = _load(args.holdout_b, names)
    if anames != names or bnames != names:
        raise AssertionError("holdout feature schema mismatch")
    screened_features = (
        FEATURES if args.feature_profile == "targeted_milk_wool" else all_public_features()
    )
    missing = sorted(set(screened_features) - set(names))
    if missing:
        raise AssertionError(f"missing features: {missing}")

    candidates = []
    for feature in screened_features:
        column = names.index(feature)
        for direction in ("le", "gt"):
            for arm in range(1, 6):
                oof, folds = grouped_oof(x[:, column], margins, groups, direction, arm)
                candidates.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "arm": arm,
                        "route": ROUTE_NAMES[arm],
                        "oof": oof,
                        "folds": folds,
                    }
                )

    selected = max(
        candidates,
        key=lambda row: (
            row["oof"]["score_rate"],
            -row["oof"]["harmed_wins"],
            -row["oof"]["switch_rate"],
            row["oof"]["mean_margin"],
        ),
    )
    column = names.index(selected["feature"])
    final = fit_threshold(x[:, column], margins, selected["direction"], selected["arm"])

    def evaluate(matrix: np.ndarray, outcome: np.ndarray) -> dict:
        return _metrics(
            outcome,
            choices(
                matrix[:, column],
                selected["direction"],
                final["threshold"],
                selected["arm"],
            ),
        )

    payload = {
        "schema": "kaggriculture.fusion_champion.k320-rank12-public-threshold-rule.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "official_package_version": "1.32.7",
        "decision_step": int(train_payload["feature_step"]),
        "future_leakage": False,
        "rule_family": "one public feature + one threshold + one existing route arm",
        "feature_profile": args.feature_profile,
        "selection_protocol": (
            "feature/direction/route selected by 5-fold OOF grouped by seed; "
            "threshold refit on train only; holdout A/B never used for selection"
        ),
        "features_screened": list(screened_features),
        "train_samples": int(len(x)),
        "train_unique_seeds": int(len(np.unique(groups))),
        "sources": [
            {"role": role, "path": str(path.resolve()), "sha256": sha256(path)}
            for role, path in (
                ("train", args.train),
                ("holdout_a", args.holdout_a),
                ("holdout_b", args.holdout_b),
            )
        ],
        "baselines": {
            "train": _baseline_oracle(margins),
            "holdout_a": _baseline_oracle(amargins),
            "holdout_b": _baseline_oracle(bmargins),
        },
        "selected": {
            **selected,
            "threshold": final["threshold"],
            "operator": "<=" if selected["direction"] == "le" else ">",
            "rule": (
                f"if {selected['feature']} "
                f"{'<=' if selected['direction'] == 'le' else '>'} "
                f"{final['threshold']}: choose {selected['route']}; else keep adaptive"
            ),
            "train_fit": evaluate(x, margins),
            "holdout_a": evaluate(ax, amargins),
            "holdout_b": evaluate(bx, bmargins),
        },
        "top_candidates": sorted(
            candidates,
            key=lambda row: (
                row["oof"]["score_rate"],
                -row["oof"]["harmed_wins"],
                -row["oof"]["switch_rate"],
                row["oof"]["mean_margin"],
            ),
            reverse=True,
        )[:30],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected": payload["selected"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
