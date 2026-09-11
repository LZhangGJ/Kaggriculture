#!/usr/bin/env python3
"""Measure split-future reproducibility of counterfactual value labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_subset(
    first: np.ndarray,
    second: np.ndarray,
    full: np.ndarray,
    seed: np.ndarray,
    seat: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    group = seed * 2 + seat
    pair_total = pair_agree = 0
    gap_total = gap_agree = 0
    top1_forward = top1_reverse = 0
    top3_forward = top3_reverse = 0
    groups = np.unique(group[mask])
    for key in groups:
        rows = np.flatnonzero(mask & (group == key))
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                i, j = int(rows[left]), int(rows[right])
                first_difference = first[i] - first[j]
                second_difference = second[i] - second[j]
                if first_difference == 0 or second_difference == 0:
                    continue
                pair_total += 1
                pair_agree += int(
                    np.sign(first_difference) == np.sign(second_difference)
                )
                if abs(full[i] - full[j]) >= 2000.0:
                    gap_total += 1
                    gap_agree += int(
                        np.sign(first_difference) == np.sign(second_difference)
                    )
        first_best = int(np.argmax(first[rows]))
        second_best = int(np.argmax(second[rows]))
        first_order = np.argsort(-first[rows], kind="stable")
        second_order = np.argsort(-second[rows], kind="stable")
        top1_forward += first_best == int(second_order[0])
        top1_reverse += second_best == int(first_order[0])
        top3_forward += first_best in second_order[:3]
        top3_reverse += second_best in first_order[:3]
    return {
        "groups": int(len(groups)),
        "pairwise_split_agreement": pair_agree / pair_total if pair_total else 0.0,
        "pairwise_comparisons": pair_total,
        "pairwise_gap2000_split_agreement": (
            gap_agree / gap_total if gap_total else 0.0
        ),
        "pairwise_gap2000_comparisons": gap_total,
        "top1_split_recall_symmetric": (
            (top1_forward + top1_reverse) / (2 * len(groups)) if len(groups) else 0.0
        ),
        "top3_split_recall_symmetric": (
            (top3_forward + top3_reverse) / (2 * len(groups)) if len(groups) else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    samples = np.asarray(data["future_delta_samples"], dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] < 4 or samples.shape[1] % 2:
        raise ValueError("future samples must be an even number >= 4")
    midpoint = samples.shape[1] // 2
    first = np.mean(samples[:, :midpoint], axis=1)
    second = np.mean(samples[:, midpoint:], axis=1)
    full = np.mean(samples, axis=1)
    seed = np.asarray(data["prefix_seed"], dtype=np.int64)
    seat = np.asarray(data["seat"], dtype=np.int64)
    all_rows = np.ones(len(first), dtype=bool)
    per_source = []
    if "source_dataset_index" in data.files:
        source_index = np.asarray(data["source_dataset_index"], dtype=np.int64)
        source_names = [str(value) for value in data["source_dataset_names"]]
        for index, name in enumerate(source_names):
            metrics = audit_subset(
                first, second, full, seed, seat, source_index == index
            )
            per_source.append({"source": name, **metrics})
    standard_error = np.std(samples, axis=1, ddof=1) / np.sqrt(samples.shape[1])
    payload = {
        "schema": "kaggriculture.counterfactual-label-reliability.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
        "future_samples": int(samples.shape[1]),
        "split_samples": int(midpoint),
        "rows": int(len(first)),
        "median_row_standard_error": float(np.median(standard_error)),
        "p90_row_standard_error": float(np.quantile(standard_error, 0.90)),
        "overall": audit_subset(first, second, full, seed, seat, all_rows),
        "per_source": per_source,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "future_samples": payload["future_samples"],
        "median_row_standard_error": payload["median_row_standard_error"],
        "overall": payload["overall"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
