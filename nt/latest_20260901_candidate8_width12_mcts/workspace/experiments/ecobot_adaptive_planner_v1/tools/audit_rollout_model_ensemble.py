#!/usr/bin/env python3
"""Select a model-prior/rollout rank blend on calibration and freeze on test."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from audit_counterfactual_rollout_budget import metrics
from train_switch_pairwise_policy import (
    group_keys,
    model_features,
    predict_pairs,
)
from train_switch_value_regressor import load


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pairwise_borda(
    model: lgb.Booster,
    data: dict[str, np.ndarray],
    feature_mode: str,
) -> np.ndarray:
    features, _, _ = model_features(
        data["features"], data["feature_names"], feature_mode
    )
    left, right, pair_key, _, probability = predict_pairs(model, data, features)
    keys = group_keys(data)
    score = np.zeros(len(keys), dtype=np.float64)
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        local = {int(row): position for position, row in enumerate(rows)}
        local_score = np.zeros(len(rows), dtype=np.float64)
        for pair_row in np.flatnonzero(pair_key == key):
            left_position = local[int(left[pair_row])]
            right_position = local[int(right[pair_row])]
            chance = float(probability[pair_row])
            local_score[left_position] += chance
            local_score[right_position] += 1.0 - chance
        score[rows] = local_score
    return score


def normalize_within_group(
    score: np.ndarray,
    keys: np.ndarray,
    mode: str,
) -> np.ndarray:
    normalized = np.zeros(len(score), dtype=np.float64)
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        value = score[rows]
        if mode == "rank":
            order = np.argsort(-value, kind="stable")
            rank = np.empty(len(rows), dtype=np.float64)
            rank[order] = np.arange(len(rows), dtype=np.float64)
            normalized[rows] = (
                1.0 - rank / max(1.0, float(len(rows) - 1))
            )
        elif mode == "zscore":
            scale = float(np.std(value))
            normalized[rows] = (
                (value - float(np.mean(value))) / scale if scale > 1e-9 else 0.0
            )
        else:
            raise ValueError(mode)
    return normalized


def split_scores(
    data: dict[str, np.ndarray],
    model: lgb.Booster,
    feature_mode: str,
    search_samples: int,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = np.asarray(data["future_delta_samples"], dtype=np.float64)
    if not 1 <= search_samples < samples.shape[1]:
        raise ValueError("invalid search sample budget")
    keys = group_keys(data)
    rollout = np.mean(samples[:, :search_samples], axis=1)
    truth = np.mean(samples[:, search_samples:], axis=1)
    prior = pairwise_borda(model, data, feature_mode)
    return (
        normalize_within_group(rollout, keys, normalization),
        normalize_within_group(prior, keys, normalization),
        truth,
    )


def selection_key(result: dict[str, object]) -> tuple[float, ...]:
    pair = float(result["pairwise"][0]["accuracy"])
    top4 = float(result["topk_oracle_recall"]["4"])
    top5 = float(result["topk_oracle_recall"]["5"])
    return (
        float(pair >= 0.75 and top4 >= 0.90),
        min(pair / 0.75, top4 / 0.90),
        top4,
        pair,
        top5,
        float(result["selected_heldout_mean"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--feature-mode", default="shop_bits_scale_forecast_marginal_catfix"
    )
    parser.add_argument("--search-samples", type=int, default=16)
    parser.add_argument("--weights", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration = load(args.calibration)
    test = load(args.test)
    if not np.array_equal(calibration["feature_names"], test["feature_names"]):
        raise ValueError("calibration/test feature schema mismatch")
    model = lgb.Booster(model_file=str(args.model))
    weights = [float(value) for value in args.weights.split(",") if value.strip()]
    calibration_audits: list[dict[str, object]] = []
    best: tuple[tuple[float, ...], str, float] | None = None
    for normalization in ("rank", "zscore"):
        rollout, prior, truth = split_scores(
            calibration,
            model,
            args.feature_mode,
            args.search_samples,
            normalization,
        )
        keys = group_keys(calibration)
        for prior_weight in weights:
            score = (1.0 - prior_weight) * rollout + prior_weight * prior
            result = metrics(score, truth, keys)
            audit = {
                "normalization": normalization,
                "prior_weight": prior_weight,
                "metrics": result,
            }
            calibration_audits.append(audit)
            key = selection_key(result)
            if best is None or key > best[0]:
                best = (key, normalization, prior_weight)
    assert best is not None
    _, normalization, prior_weight = best
    rollout, prior, truth = split_scores(
        test,
        model,
        args.feature_mode,
        args.search_samples,
        normalization,
    )
    test_result = metrics(
        (1.0 - prior_weight) * rollout + prior_weight * prior,
        truth,
        group_keys(test),
    )
    selected_calibration = next(
        row for row in calibration_audits
        if row["normalization"] == normalization
        and row["prior_weight"] == prior_weight
    )
    payload = {
        "schema": "kaggriculture.rollout-model-rank-ensemble.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "invariants": {
            "test_used_for_selection": False,
            "search_and_evaluation_futures_disjoint": True,
            "opponent_or_route_identity_used": False,
        },
        "calibration": {"path": str(args.calibration),
                        "sha256": sha256(args.calibration)},
        "test": {"path": str(args.test), "sha256": sha256(args.test)},
        "model": {"path": str(args.model), "sha256": sha256(args.model)},
        "feature_mode": args.feature_mode,
        "search_samples": args.search_samples,
        "calibration_audits": calibration_audits,
        "selected_calibration": selected_calibration,
        "selected_test_not_used_for_selection": test_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_calibration": selected_calibration,
        "test": test_result,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
