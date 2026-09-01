#!/usr/bin/env python3
"""Audit expected-value portfolio candidates by day and project family."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_NAMES = [
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "GOOSE", "COW", "SHEEP",
]


def distribution(values: np.ndarray) -> dict[str, float]:
    if not values.size:
        return {key: 0.0 for key in ("mean", "p10", "median", "p90")}
    return {
        "mean": float(values.mean()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.dataset)
    features = np.asarray(data["features"], dtype=np.float64)
    expected = np.asarray(data["expected_delta"], dtype=np.float64)
    prefix = np.asarray(data["prefix_seed"], dtype=np.int64)
    seat = np.asarray(data["seat"], dtype=np.int8)
    ranks = np.asarray(data["candidate_rank"], dtype=np.int16)
    names = [str(value) for value in data["feature_names"]]
    name_to_index = {name: index for index, name in enumerate(names)}
    group = prefix * 2 + seat
    groups = np.unique(group)

    day_column = name_to_index["day"]
    source_column = name_to_index["source_project"]
    destination_column = name_to_index["destination_project"]
    baseline_column = name_to_index["baseline_analytic_score"]
    candidate_column = name_to_index["candidate_analytic_score"]

    candidate_mask = ranks >= 0
    analytic_gain = features[:, candidate_column] - features[:, baseline_column]
    family_rows: dict[tuple[int, int], list[int]] = defaultdict(list)
    day_rows: dict[int, list[int]] = defaultdict(list)
    for index in np.flatnonzero(candidate_mask):
        source = int(features[index, source_column])
        destination = int(features[index, destination_column])
        family_rows[(source, destination)].append(int(index))
        day_rows[int(features[index, day_column])].append(int(index))

    def row_summary(indices: list[int]) -> dict[str, object]:
        at = np.asarray(indices, dtype=np.int64)
        return {
            "rows": int(at.size),
            "expected_delta": distribution(expected[at]),
            "positive_rate": float(np.mean(expected[at] > 0)) if at.size else 0.0,
            "analytic_gain": distribution(analytic_gain[at]),
            "analytic_expected_correlation": (
                float(np.corrcoef(analytic_gain[at], expected[at])[0, 1])
                if at.size >= 3 and np.std(analytic_gain[at]) > 0 and
                np.std(expected[at]) > 0 else 0.0
            ),
        }

    selected_delta = []
    oracle_delta = []
    oracle_in_top3 = 0
    analytic_positive_realized = []
    candidate_counts = []
    for key in groups:
        indices = np.flatnonzero(group == key)
        if not indices.size:
            continue
        candidate_counts.append(int(np.count_nonzero(ranks[indices] >= 0)))
        # KEEP uses delta=0 and its reconstructed feature row has equal
        # baseline/candidate analytic score.
        analytic_scores = features[indices, candidate_column]
        order = np.argsort(-analytic_scores, kind="stable")
        best = int(indices[int(order[0])])
        oracle = int(indices[int(np.argmax(expected[indices]))])
        selected_delta.append(float(expected[best]))
        oracle_delta.append(float(expected[oracle]))
        oracle_in_top3 += oracle in indices[order[:3]]
        for index in indices:
            if ranks[index] >= 0 and analytic_gain[index] > 0:
                analytic_positive_realized.append(float(expected[index]))

    selected = np.asarray(selected_delta, dtype=np.float64)
    oracle = np.asarray(oracle_delta, dtype=np.float64)
    positive_realized = np.asarray(analytic_positive_realized, dtype=np.float64)
    family = []
    for (source, destination), indices in family_rows.items():
        row = {
            "source": source,
            "destination": destination,
            "family": (
                f"{PROJECT_NAMES[source]}->{PROJECT_NAMES[destination]}"
                if 0 <= source < len(PROJECT_NAMES) and
                0 <= destination < len(PROJECT_NAMES)
                else f"{source}->{destination}"
            ),
        }
        row.update(row_summary(indices))
        family.append(row)
    family.sort(key=lambda row: (-int(row["rows"]), str(row["family"])))

    by_day = []
    for day, indices in sorted(day_rows.items()):
        row: dict[str, object] = {"day": day}
        row.update(row_summary(indices))
        by_day.append(row)

    example_fields = [
        "day", "hour", "cash", "hands", "quadrants", "shed_units",
        "source_project", "destination_project", "remove_count", "add_count",
        "baseline_analytic_score", "candidate_analytic_score",
        "baseline_estimated_hands", "candidate_estimated_hands",
        "owned_geese", "owned_cows", "owned_sheep",
        "field_geese", "field_cows", "field_sheep",
        "field_wheat", "field_carrot", "field_tomato", "field_strawberry",
        "field_melon", "baseline_wheat", "baseline_carrot",
        "baseline_tomato", "baseline_strawberry", "baseline_melon",
        "baseline_geese", "baseline_cows", "baseline_sheep",
        "candidate_wheat", "candidate_carrot", "candidate_tomato",
        "candidate_strawberry", "candidate_melon", "candidate_geese",
        "candidate_cows", "candidate_sheep", "fertilizer_units",
        "wheat_units", "empty_tiles", "empty_pastures", "empty_coops",
        "baseline_unmet_crops", "candidate_unmet_crops",
        "baseline_unmet_animals", "candidate_unmet_animals",
        "baseline_workload_x100", "candidate_workload_x100",
    ]

    def example(index: int) -> dict[str, object]:
        row: dict[str, object] = {
            "prefix_seed": int(prefix[index]),
            "seat": int(seat[index]),
            "candidate_rank": int(ranks[index]),
            "expected_delta": float(expected[index]),
            "analytic_gain": float(analytic_gain[index]),
        }
        for field in example_fields:
            if field in name_to_index:
                row[field] = int(features[index, name_to_index[field]])
        return row

    candidate_indices = np.flatnonzero(candidate_mask)
    worst_indices = candidate_indices[
        np.argsort(expected[candidate_indices], kind="stable")[:20]
    ]
    best_indices = candidate_indices[
        np.argsort(-expected[candidate_indices], kind="stable")[:20]
    ]

    payload = {
        "schema": "kaggriculture.portfolio-expected-value-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "states": int(len(groups)),
        "candidate_rows": int(np.count_nonzero(candidate_mask)),
        "candidate_count_per_state": distribution(
            np.asarray(candidate_counts, dtype=np.float64)
        ),
        "analytic_selected_expected_delta": distribution(selected),
        "analytic_selected_positive_rate": (
            float(np.mean(selected > 0)) if selected.size else 0.0
        ),
        "oracle_expected_delta": distribution(oracle),
        "oracle_top3_recall": oracle_in_top3 / len(groups) if len(groups) else 0.0,
        "all_analytic_positive_candidates_realized_delta": distribution(
            positive_realized
        ),
        "all_analytic_positive_candidates_positive_rate": (
            float(np.mean(positive_realized > 0))
            if positive_realized.size else 0.0
        ),
        "families": family,
        "days": by_day,
        "worst_candidates": [example(int(index)) for index in worst_indices],
        "best_candidates": [example(int(index)) for index in best_indices],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "states": payload["states"],
        "analytic_selected_expected_delta":
            payload["analytic_selected_expected_delta"],
        "analytic_selected_positive_rate":
            payload["analytic_selected_positive_rate"],
        "oracle_expected_delta": payload["oracle_expected_delta"],
        "top_families": family[:12],
        "days": by_day,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
