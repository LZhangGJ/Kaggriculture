#!/usr/bin/env python3
"""Audit Monte-Carlo project ranking with disjoint search/evaluation futures."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from train_switch_pairwise_policy import group_keys
from train_switch_value_regressor import load


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(
    score: np.ndarray,
    truth: np.ndarray,
    keys: np.ndarray,
) -> dict[str, object]:
    pair_correct = {gap: 0 for gap in (0.0, 500.0, 1000.0, 2000.0)}
    pair_total = {gap: 0 for gap in pair_correct}
    top_hits = {k: 0 for k in range(1, 6)}
    selected: list[float] = []
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        actual = truth[rows]
        predicted = score[rows]
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                difference = float(actual[left] - actual[right])
                if difference == 0.0:
                    continue
                correct = np.sign(predicted[left] - predicted[right]) == np.sign(
                    difference
                )
                for gap in pair_total:
                    if abs(difference) >= max(1e-9, gap):
                        pair_total[gap] += 1
                        pair_correct[gap] += int(correct)
        oracle = int(np.argmax(actual))
        order = np.argsort(-predicted, kind="stable")
        for k in top_hits:
            top_hits[k] += int(oracle in order[:k])
        selected.append(float(actual[int(order[0])]))
    selected_value = np.asarray(selected, dtype=np.float64)
    group_count = len(np.unique(keys))
    return {
        "groups": int(group_count),
        "pairwise": [
            {
                "minimum_truth_gap": gap,
                "pairs": pair_total[gap],
                "accuracy": (
                    pair_correct[gap] / pair_total[gap] if pair_total[gap] else 0.0
                ),
            }
            for gap in pair_total
        ],
        "topk_oracle_recall": {
            str(k): top_hits[k] / group_count if group_count else 0.0
            for k in top_hits
        },
        "selected_heldout_mean": float(np.mean(selected_value)),
        "selected_heldout_p10": float(np.quantile(selected_value, 0.10)),
        "selected_heldout_negative_rate": float(np.mean(selected_value < 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--search-samples", default="1,2,4,8,16,24,32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = load(args.corpus)
    samples = np.asarray(data["future_delta_samples"], dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] < 2:
        raise ValueError("future_delta_samples must contain at least two futures")
    budgets = [
        int(value) for value in args.search_samples.split(",") if value.strip()
    ]
    if not budgets or min(budgets) < 1 or max(budgets) >= samples.shape[1]:
        raise ValueError("each search budget must be in [1, future_count - 1]")
    keys = group_keys(data)
    audits = []
    for budget in budgets:
        search_score = np.mean(samples[:, :budget], axis=1)
        heldout_truth = np.mean(samples[:, budget:], axis=1)
        audits.append({
            "search_samples": budget,
            "heldout_samples": int(samples.shape[1] - budget),
            "metrics": metrics(search_score, heldout_truth, keys),
        })
    payload = {
        "schema": "kaggriculture.counterfactual-rollout-budget-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": {"path": str(args.corpus), "sha256": sha256(args.corpus)},
        "contract": {
            "search_and_evaluation_futures_disjoint": True,
            "common_random_numbers_within_candidate_group": True,
            "future_event_count": int(samples.shape[1]),
            "opponent_or_route_identity_used": False,
        },
        "audits": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
