#!/usr/bin/env python3
"""Audit paired Candidate8-vs-KEEP label support by decision day."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def paired_targets(
    state_id: np.ndarray,
    own: np.ndarray,
    rival: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    score_delta = np.zeros(len(state_id), dtype=np.float64)
    margin_delta = np.zeros(len(state_id), dtype=np.float64)
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        state_own = own[indices]
        state_rival = rival[indices]
        score = np.where(
            state_own > state_rival,
            1.0,
            np.where(state_own == state_rival, 0.5, 0.0),
        )
        keep_score = score[0:1]
        margin = state_own - state_rival
        keep_margin = margin[0:1]
        score_delta[indices] = (score - keep_score).mean(axis=1)
        margin_delta[indices] = (margin - keep_margin).mean(axis=1)
    return score_delta, margin_delta


def summarize(
    states: np.ndarray,
    state_id: np.ndarray,
    score: np.ndarray,
    margin: np.ndarray,
) -> dict:
    non_keep = np.concatenate([
        np.flatnonzero(state_id == state)[1:] for state in states
    ])
    best_score = []
    best_margin = []
    any_score_positive = []
    any_margin_positive = []
    for state in states:
        indices = np.flatnonzero(state_id == state)
        candidate_indices = indices[1:]
        any_score_positive.append(np.any(score[candidate_indices] > 0.0))
        any_margin_positive.append(np.any(margin[candidate_indices] > 0.0))
        chosen = max(
            candidate_indices,
            key=lambda index: (float(score[index]), float(margin[index])),
        )
        best_score.append(float(score[chosen]))
        best_margin.append(float(margin[chosen]))
    return {
        "states": int(len(states)),
        "non_keep_candidates": int(len(non_keep)),
        "candidate_score_positive_rate": float(np.mean(score[non_keep] > 0.0)),
        "candidate_score_negative_rate": float(np.mean(score[non_keep] < 0.0)),
        "candidate_score_nonzero_rate": float(np.mean(score[non_keep] != 0.0)),
        "candidate_margin_positive_rate": float(np.mean(margin[non_keep] > 0.0)),
        "states_with_score_improvement": float(np.mean(any_score_positive)),
        "states_with_margin_improvement": float(np.mean(any_margin_positive)),
        "lexicographic_oracle_mean_score_gain": float(np.mean(best_score)),
        "lexicographic_oracle_mean_margin_gain": float(np.mean(best_margin)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"])
        day = np.asarray(data["decision_day"])
        own = np.asarray(data["future_own_cash"], dtype=np.float64)
        rival = np.asarray(data["future_opponent_cash"], dtype=np.float64)
        features = np.asarray(data["features"])
        feature_names = [str(value) for value in data["feature_names"]]

    banks = {
        "all": np.arange(own.shape[1]),
        "bank_a_even": np.arange(0, own.shape[1], 2),
        "bank_b_odd": np.arange(1, own.shape[1], 2),
    }
    summaries = {}
    targets = {}
    for bank_name, columns in banks.items():
        score, margin = paired_targets(
            state_id, own[:, columns], rival[:, columns]
        )
        targets[bank_name] = (score, margin)
        summaries[bank_name] = {}
        for decision_day in sorted(int(value) for value in np.unique(day)):
            states = np.unique(state_id[day == decision_day])
            summaries[bank_name][str(decision_day)] = summarize(
                states, state_id, score, margin
            )
        summaries[bank_name]["all_days"] = summarize(
            np.unique(state_id), state_id, score, margin
        )

    score_a, margin_a = targets["bank_a_even"]
    score_b, margin_b = targets["bank_b_odd"]
    non_keep = np.concatenate([
        np.flatnonzero(state_id == state)[1:] for state in np.unique(state_id)
    ])
    added = features[:, 159:]
    payload = {
        "schema": "kaggriculture.candidate8-o11-multiday-label-audit.v1",
        "dataset": str(args.dataset),
        "rows": int(len(state_id)),
        "states": int(len(np.unique(state_id))),
        "future_columns": int(own.shape[1]),
        "feature_count": int(features.shape[1]),
        "added_feature_count": int(added.shape[1]),
        "variable_added_feature_count": int(np.count_nonzero(
            np.ptp(added, axis=0) != 0
        )),
        "bank_agreement": {
            "score_sign_agreement_rate_non_keep": float(np.mean(
                np.sign(score_a[non_keep]) == np.sign(score_b[non_keep])
            )),
            "margin_sign_agreement_rate_non_keep": float(np.mean(
                np.sign(margin_a[non_keep]) == np.sign(margin_b[non_keep])
            )),
        },
        "summaries": summaries,
        "feature_names": feature_names,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "states": payload["states"],
        "variable_added_feature_count": payload["variable_added_feature_count"],
        "bank_agreement": payload["bank_agreement"],
        "all_future_by_day": summaries["all"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
