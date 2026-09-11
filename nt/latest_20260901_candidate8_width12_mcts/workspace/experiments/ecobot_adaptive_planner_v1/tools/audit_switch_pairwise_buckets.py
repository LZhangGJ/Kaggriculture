#!/usr/bin/env python3
"""Audit frozen project-choice pairwise accuracy by public-state buckets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from train_switch_pairwise_policy import model_features, predict_pairs, sha256
from train_switch_value_regressor import load


PROJECTS = ["wheat", "carrot", "tomato", "strawberry", "melon",
            "geese", "cows", "sheep"]


def project_family(project: np.ndarray) -> np.ndarray:
    """Map project ids to keep/crop/animal without relying on route authors."""
    result = np.full(project.shape, "keep", dtype="<U6")
    result[(project >= 0) & (project < 5)] = "crop"
    result[project >= 5] = "animal"
    return result


def phase_for_day(day: np.ndarray) -> np.ndarray:
    result = np.full(day.shape, "late", dtype="<U6")
    result[day <= 10] = "early"
    result[(day >= 11) & (day <= 20)] = "middle"
    return result


def summarize(mask: np.ndarray, difference: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    selected = mask & (difference != 0.0)
    correct = (probability[selected] >= 0.5) == (difference[selected] > 0.0)
    large = selected & (np.abs(difference) >= 2000.0)
    large_correct = (probability[large] >= 0.5) == (difference[large] > 0.0)
    return {
        "pairs": int(np.sum(selected)),
        "accuracy": float(np.mean(correct)) if len(correct) else 0.0,
        "gap2000_pairs": int(np.sum(large)),
        "gap2000_accuracy": float(np.mean(large_correct)) if len(large_correct) else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--feature-mode",
        choices=(
            "legacy", "shop_bits", "shop_bits_scale",
            "shop_bits_scale_catfix", "shop_bits_scale_marginal",
            "shop_bits_scale_forecast",
            "shop_bits_scale_forecast_catfix",
            "shop_bits_scale_forecast_marginal_catfix",
        ),
        default="legacy",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = load(args.corpus)
    names = [str(value) for value in data["feature_names"]]
    at = {name: index for index, name in enumerate(names)}
    features, _, _ = model_features(data["features"], data["feature_names"], args.feature_mode)
    model = lgb.Booster(model_file=str(args.model))
    left, right, _, difference, probability = predict_pairs(model, data, features)
    future_samples = np.asarray(data["future_delta_samples"], dtype=np.float64)
    paired_future_difference = future_samples[left] - future_samples[right]
    future_count = paired_future_difference.shape[1]
    paired_standard_error = (
        np.std(paired_future_difference, axis=1, ddof=1) / np.sqrt(future_count)
        if future_count > 1 else np.zeros(len(difference), dtype=np.float64)
    )
    expected_sign = np.sign(difference)[:, None]
    future_sign_agreement = np.mean(
        np.sign(paired_future_difference) == expected_sign, axis=1
    )
    ci95_excludes_zero = np.abs(difference) > 1.96 * paired_standard_error
    raw = np.asarray(data["features"], dtype=np.float64)
    day = raw[left, at["day"]].astype(np.int64)
    phase = phase_for_day(day)
    quadrants = raw[left, at["quadrants"]].astype(np.int64)
    seat = np.asarray(data["seat"], dtype=np.int64)[left]
    shop_mask = raw[left, at["shop_mask"]].astype(np.int64)
    shop_count = np.asarray([int(value).bit_count() for value in shop_mask], dtype=np.int64)
    destination = raw[:, at["destination_project"]].astype(np.int64)
    left_destination = destination[left]
    right_destination = destination[right]
    left_family = project_family(left_destination)
    right_family = project_family(right_destination)
    supply_columns = [at[f"opponent_visible_supply_x100_{item}"] for item in range(8)]
    lag_columns = [at[f"opponent_committed_supply_lag8_x100_{item}"] for item in range(8)]
    opponent_pressure_by_row = raw[:, supply_columns].sum(axis=1) + raw[:, lag_columns].sum(axis=1)
    opponent_pressure = opponent_pressure_by_row[left]
    pressure_edges = np.quantile(opponent_pressure, [0.25, 0.50, 0.75])
    pressure_bucket = np.digitize(opponent_pressure, pressure_edges, right=True)

    all_mask = np.ones(len(difference), dtype=bool)
    payload: dict[str, object] = {
        "schema": "kaggriculture.switch-pairwise-public-bucket-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": {"path": str(args.corpus), "sha256": sha256(args.corpus)},
        "model": {"path": str(args.model), "sha256": sha256(args.model)},
        "feature_mode": args.feature_mode,
        "overall": summarize(all_mask, difference, probability),
        "future_samples_per_arm": int(future_count),
        "by_label_reliability": {
            "ci95_excludes_zero": summarize(
                ci95_excludes_zero, difference, probability
            ),
            "future_sign_agreement_at_least_75pct": summarize(
                future_sign_agreement >= 0.75, difference, probability
            ),
            "future_sign_agreement_at_least_90pct": summarize(
                future_sign_agreement >= 0.90, difference, probability
            ),
            "ci95_and_gap2000": summarize(
                ci95_excludes_zero & (np.abs(difference) >= 2000.0),
                difference,
                probability,
            ),
        },
        "by_day": {
            str(value): summarize(day == value, difference, probability)
            for value in np.unique(day)
        },
        "by_phase": {
            value: summarize(phase == value, difference, probability)
            for value in ("early", "middle", "late")
        },
        "by_quadrants": {
            str(value): summarize(quadrants == value, difference, probability)
            for value in np.unique(quadrants)
        },
        "by_phase_and_quadrants": {
            f"{phase_value}_q{quadrant_value}": summarize(
                (phase == phase_value) & (quadrants == quadrant_value),
                difference,
                probability,
            )
            for phase_value in ("early", "middle", "late")
            for quadrant_value in np.unique(quadrants)
        },
        "by_seat": {
            str(value): summarize(seat == value, difference, probability)
            for value in np.unique(seat)
        },
        "by_shop_count": {
            str(value): summarize(shop_count == value, difference, probability)
            for value in np.unique(shop_count)
        },
        "opponent_pressure_quartile_edges_x100": [float(value) for value in pressure_edges],
        "by_opponent_pressure_quartile": {
            str(value): summarize(pressure_bucket == value, difference, probability)
            for value in range(4)
        },
        "by_destination_involved": {
            project: summarize(
                (left_destination == project_id) | (right_destination == project_id),
                difference,
                probability,
            )
            for project_id, project in enumerate(PROJECTS)
        },
        "keep_vs_edit": summarize(
            (left_destination < 0) | (right_destination < 0), difference, probability
        ),
        "edit_vs_edit": summarize(
            (left_destination >= 0) & (right_destination >= 0), difference, probability
        ),
        "same_destination": summarize(
            (left_destination >= 0) & (left_destination == right_destination),
            difference,
            probability,
        ),
        "cross_destination": summarize(
            (left_destination >= 0) & (right_destination >= 0)
            & (left_destination != right_destination),
            difference,
            probability,
        ),
        "by_pair_family": {
            f"{left_value}_vs_{right_value}": summarize(
                ((left_family == left_value) & (right_family == right_value))
                | ((left_family == right_value) & (right_family == left_value)),
                difference,
                probability,
            )
            for left_value, right_value in (
                ("keep", "crop"),
                ("keep", "animal"),
                ("crop", "crop"),
                ("crop", "animal"),
                ("animal", "animal"),
            )
        },
        "by_destination_pair": {
            f"{PROJECTS[left_project]}_vs_{PROJECTS[right_project]}": summarize(
                ((left_destination == left_project) & (right_destination == right_project))
                | ((left_destination == right_project) & (right_destination == left_project)),
                difference,
                probability,
            )
            for left_project in range(len(PROJECTS))
            for right_project in range(left_project, len(PROJECTS))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
