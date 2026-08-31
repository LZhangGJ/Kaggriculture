#!/usr/bin/env python3
"""Greedily select GA routes by joint exact-state cross-validation gain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def exact_state_cv_metrics(
    scores: np.ndarray, state_ids: np.ndarray, seed_count: int, folds: int,
) -> dict[str, float]:
    seed_ids = np.repeat(np.arange(seed_count), 2)
    fold_rates = []
    for valid_seeds in np.array_split(np.arange(seed_count), min(folds, seed_count)):
        valid = np.isin(seed_ids, valid_seeds)
        train = ~valid
        predictions = np.zeros(np.sum(valid), dtype=np.int32)
        valid_states = state_ids[valid]
        for state in np.unique(valid_states):
            train_rows = train & (state_ids == state)
            action = int(np.argmax(np.mean(scores[train_rows], axis=0))) if np.any(
                train_rows
            ) else 0
            predictions[valid_states == state] = action
        fold_scores = scores[valid]
        fold_rates.append(float(np.mean(
            fold_scores[np.arange(len(fold_scores)), predictions]
        )))
    predictions = np.zeros(len(state_ids), dtype=np.int32)
    for state in np.unique(state_ids):
        rows = state_ids == state
        predictions[rows] = int(np.argmax(np.mean(scores[rows], axis=0)))
    training = float(np.mean(scores[np.arange(len(scores)), predictions]))
    return {
        "minimum_fold_raw_win_rate": float(min(fold_rates)),
        "cross_validation_raw_win_rate": float(np.mean(fold_rates)),
        "training_raw_win_rate": training,
    }


def metric_key(metrics: dict[str, float]) -> tuple[float, ...]:
    return (
        metrics["minimum_fold_raw_win_rate"],
        metrics["cross_validation_raw_win_rate"],
        metrics["training_raw_win_rate"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--max-routes", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.template, allow_pickle=False) as base, np.load(
        args.candidates, allow_pickle=False
    ) as candidates:
        seeds = base["seeds"].astype(np.int64)
        if not np.array_equal(seeds, candidates["seeds"]):
            raise ValueError("candidate and template seeds do not match")
        base_scores = (
            base["outcome"][0, 0, :, 0].reshape(len(base["targets"]), -1).T == 2
        ).astype(np.float32)
        candidate_scores = candidates["outcome"].reshape(
            len(candidates["families"]), -1
        ).T.astype(np.float32)
        families = candidates["families"].astype(str)
        states = base["states"][0, 0, 0].reshape(-1, base["states"].shape[-1])
    _, state_ids = np.unique(states, axis=0, return_inverse=True)

    selected: list[int] = []
    available = set(range(len(families)))
    scores = base_scores
    current = exact_state_cv_metrics(scores, state_ids, len(seeds), args.folds)
    baseline = dict(current)
    trace = []
    for rank in range(1, args.max_routes + 1):
        best = None
        for index in available:
            trial_scores = np.column_stack((scores, candidate_scores[:, index]))
            metrics = exact_state_cv_metrics(
                trial_scores, state_ids, len(seeds), args.folds
            )
            candidate = (metric_key(metrics), float(np.mean(candidate_scores[:, index])), -index)
            if best is None or candidate > best[0]:
                best = (candidate, index, metrics)
        assert best is not None
        _, index, metrics = best
        if metric_key(metrics) <= metric_key(current):
            break
        selected.append(index)
        available.remove(index)
        scores = np.column_stack((scores, candidate_scores[:, index]))
        current = metrics
        trace.append({
            "rank": rank,
            "family": str(families[index]),
            "candidate_raw_win_rate": float(np.mean(candidate_scores[:, index])),
            **metrics,
        })

    payload = {
        "schema": "dynamic-ga-joint-exact-state-portfolio-v1",
        "template": str(args.template.resolve()),
        "candidates": str(args.candidates.resolve()),
        "folds": args.folds,
        "baseline": baseline,
        "selected_families": [str(families[index]) for index in selected],
        "trace": trace,
        "final": current,
        "joint_route_oracle_raw_win_rate": float(np.mean(np.max(scores, axis=1))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
