#!/usr/bin/env python3
"""Freeze a diversified Top-K shortlist on calibration and test it once."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from audit_counterfactual_rollout_budget import metrics
from audit_rollout_model_ensemble import pairwise_borda
from train_switch_pairwise_policy import group_keys, model_features
from train_switch_value_regressor import load


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_scores(
    data: dict[str, np.ndarray],
    pair_model: lgb.Booster,
    list_model: lgb.Booster,
    feature_mode: str,
    search_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples = np.asarray(data["future_delta_samples"], dtype=np.float64)
    rollout = np.mean(samples[:, :search_samples], axis=1)
    truth = np.mean(samples[:, search_samples:], axis=1)
    pair = pairwise_borda(pair_model, data, feature_mode)
    features, _, _ = model_features(
        data["features"], data["feature_names"], feature_mode
    )
    listwise = np.asarray(list_model.predict(features), dtype=np.float64)
    return rollout, pair, listwise, truth


def shortlist_metrics(
    rollout: np.ndarray,
    pair: np.ndarray,
    listwise: np.ndarray,
    truth: np.ndarray,
    keys: np.ndarray,
    allocation: tuple[int, int, int],
    top_k: int,
) -> dict[str, float | int | list[int]]:
    hit = 0
    selected: list[float] = []
    shortlist_sizes: list[int] = []
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        orders = (
            np.argsort(-rollout[rows], kind="stable"),
            np.argsort(-pair[rows], kind="stable"),
            np.argsort(-listwise[rows], kind="stable"),
        )
        chosen: list[int] = []
        for order, count in zip(orders, allocation, strict=True):
            for position in order[:count]:
                local = int(position)
                if local not in chosen:
                    chosen.append(local)
                if len(chosen) == top_k:
                    break
            if len(chosen) == top_k:
                break
        # Overlap between sources must not silently shrink the budget.  Fill
        # unused slots by rollout order, which remains the final value judge.
        for position in orders[0]:
            local = int(position)
            if local not in chosen:
                chosen.append(local)
            if len(chosen) == min(top_k, len(rows)):
                break
        actual = truth[rows]
        oracle = int(np.argmax(actual))
        hit += int(oracle in chosen)
        selected_local = max(chosen, key=lambda local: rollout[rows[local]])
        selected.append(float(actual[selected_local]))
        shortlist_sizes.append(len(chosen))
    selected_value = np.asarray(selected, dtype=np.float64)
    groups = len(selected)
    return {
        "allocation_rollout_pair_listwise": list(allocation),
        "groups": groups,
        "top_k": top_k,
        "oracle_recall": hit / groups if groups else 0.0,
        "selected_heldout_mean": float(np.mean(selected_value)),
        "selected_heldout_p10": float(np.quantile(selected_value, 0.10)),
        "selected_heldout_negative_rate": float(np.mean(selected_value < 0.0)),
        "minimum_shortlist_size": int(min(shortlist_sizes)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--pair-model", type=Path, required=True)
    parser.add_argument("--list-model", type=Path, required=True)
    parser.add_argument(
        "--feature-mode", default="shop_bits_scale_forecast_marginal_catfix"
    )
    parser.add_argument("--search-samples", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration = load(args.calibration)
    test = load(args.test)
    if not np.array_equal(calibration["feature_names"], test["feature_names"]):
        raise ValueError("calibration/test feature schema mismatch")
    pair_model = lgb.Booster(model_file=str(args.pair_model))
    list_model = lgb.Booster(model_file=str(args.list_model))
    allocations = [
        (4, 0, 0),
        (3, 1, 0), (2, 2, 0), (1, 3, 0),
        (3, 0, 1), (2, 0, 2), (1, 0, 3),
        (2, 1, 1), (1, 2, 1), (1, 1, 2),
    ]
    cal_scores = candidate_scores(
        calibration, pair_model, list_model, args.feature_mode, args.search_samples
    )
    cal_keys = group_keys(calibration)
    audits = [
        shortlist_metrics(*cal_scores, cal_keys, allocation, args.top_k)
        for allocation in allocations
    ]
    selected = max(
        audits,
        key=lambda row: (
            float(row["oracle_recall"] >= 0.90),
            float(row["oracle_recall"]),
            float(row["selected_heldout_mean"]),
            float(row["selected_heldout_p10"]),
        ),
    )
    allocation = tuple(
        int(value) for value in selected["allocation_rollout_pair_listwise"]
    )
    test_scores = candidate_scores(
        test, pair_model, list_model, args.feature_mode, args.search_samples
    )
    test_keys = group_keys(test)
    test_shortlist = shortlist_metrics(
        *test_scores, test_keys, allocation, args.top_k
    )
    # Pair accuracy is the independent held-out accuracy of the rollout value
    # judge.  The diversified shortlist changes recall, not the pair labels.
    test_ranking = metrics(test_scores[0], test_scores[3], test_keys)
    payload = {
        "schema": "kaggriculture.diversified-rollout-shortlist.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "invariants": {
            "test_used_for_selection": False,
            "search_and_evaluation_futures_disjoint": True,
            "shortlist_and_final_value_judge_separated": True,
            "opponent_or_route_identity_used": False,
        },
        "inputs": {
            "calibration": {"path": str(args.calibration),
                            "sha256": sha256(args.calibration)},
            "test": {"path": str(args.test), "sha256": sha256(args.test)},
            "pair_model": {"path": str(args.pair_model),
                           "sha256": sha256(args.pair_model)},
            "list_model": {"path": str(args.list_model),
                           "sha256": sha256(args.list_model)},
        },
        "feature_mode": args.feature_mode,
        "search_samples": args.search_samples,
        "top_k": args.top_k,
        "calibration_audits": audits,
        "selected_calibration": selected,
        "selected_test_shortlist_not_used_for_selection": test_shortlist,
        "selected_test_rollout_ranking_not_used_for_selection": test_ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_calibration": selected,
        "test_shortlist": test_shortlist,
        "test_pairwise": test_ranking["pairwise"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
