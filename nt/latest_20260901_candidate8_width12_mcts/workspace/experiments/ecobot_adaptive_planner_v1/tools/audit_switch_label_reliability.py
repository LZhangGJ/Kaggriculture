#!/usr/bin/env python3
"""Measure whether finite-future SWITCH labels are statistically identifiable."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from train_switch_pairwise_policy import pair_structure
from train_switch_value_regressor import load


PROJECTS = ["wheat", "carrot", "tomato", "strawberry", "melon",
            "geese", "cows", "sheep"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_for_day(day: np.ndarray) -> np.ndarray:
    return np.where(day <= 10, "early", np.where(day <= 20, "middle", "late"))


def summarize(mask: np.ndarray, mean_gap: np.ndarray, sample_gap: np.ndarray) -> dict[str, object]:
    mask = mask & (mean_gap != 0.0)
    values = sample_gap[mask]
    gaps = mean_gap[mask]
    if len(gaps) == 0:
        return {"pairs": 0}
    count = values.shape[1]
    sample_std = np.std(values, axis=1, ddof=1) if count > 1 else np.zeros(len(values))
    standard_error = sample_std / np.sqrt(max(1, count))
    expected_sign = np.sign(gaps)[:, None]
    agreement = np.mean(np.sign(values) == expected_sign, axis=1)
    confidence95 = np.abs(gaps) > 1.96 * standard_error
    strong_majority = agreement >= 0.75
    return {
        "pairs": int(len(gaps)),
        "future_samples_per_arm": int(count),
        "median_absolute_mean_gap": float(np.median(np.abs(gaps))),
        "median_pair_standard_error": float(np.median(standard_error)),
        "mean_future_sign_agreement": float(np.mean(agreement)),
        "fraction_future_sign_agreement_at_least_75pct": float(np.mean(strong_majority)),
        "fraction_95pct_ci_excludes_zero": float(np.mean(confidence95)),
        "fraction_abs_gap_at_least_2000": float(np.mean(np.abs(gaps) >= 2000.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = load(args.corpus)
    names = [str(value) for value in data["feature_names"]]
    at = {name: index for index, name in enumerate(names)}
    left, right, _, mean_gap = pair_structure(data, include_exact_ties=False)
    samples = np.asarray(data["future_delta_samples"], dtype=np.float64)
    sample_gap = samples[left] - samples[right]
    raw = np.asarray(data["features"], dtype=np.float64)
    day = raw[left, at["day"]].astype(np.int64)
    phase = phase_for_day(day)
    destination = raw[:, at["destination_project"]].astype(np.int64)
    left_destination = destination[left]
    right_destination = destination[right]
    left_crop = (left_destination >= 0) & (left_destination < 5)
    right_crop = (right_destination >= 0) & (right_destination < 5)
    left_animal = left_destination >= 5
    right_animal = right_destination >= 5
    keep_pair = (left_destination < 0) | (right_destination < 0)
    all_mask = np.ones(len(mean_gap), dtype=bool)

    payload = {
        "schema": "kaggriculture.switch-label-reliability-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": {"path": str(args.corpus), "sha256": sha256(args.corpus)},
        "overall": summarize(all_mask, mean_gap, sample_gap),
        "by_phase": {
            value: summarize(phase == value, mean_gap, sample_gap)
            for value in ("early", "middle", "late")
        },
        "by_pair_family": {
            "keep_vs_edit": summarize(keep_pair, mean_gap, sample_gap),
            "crop_vs_crop": summarize(left_crop & right_crop, mean_gap, sample_gap),
            "crop_vs_animal": summarize(
                (left_crop & right_animal) | (left_animal & right_crop),
                mean_gap,
                sample_gap,
            ),
            "animal_vs_animal": summarize(left_animal & right_animal, mean_gap, sample_gap),
            "same_destination": summarize(
                (left_destination >= 0) & (left_destination == right_destination),
                mean_gap,
                sample_gap,
            ),
        },
        "by_destination_involved": {
            project: summarize(
                (left_destination == project_id) | (right_destination == project_id),
                mean_gap,
                sample_gap,
            )
            for project_id, project in enumerate(PROJECTS)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
