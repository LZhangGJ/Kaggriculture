#!/usr/bin/env python3
"""Summarize win/loss and terminal cash-margin distributions for Candidate8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def distribution(own: np.ndarray, rival: np.ndarray) -> dict:
    own = np.asarray(own, dtype=np.float64).reshape(-1)
    rival = np.asarray(rival, dtype=np.float64).reshape(-1)
    margin = own - rival
    win = margin > 0
    tie = margin == 0
    loss = margin < 0

    def stats(values: np.ndarray) -> dict:
        if not len(values):
            return {"count": 0}
        return {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p25": float(np.quantile(values, 0.25)),
            "p75": float(np.quantile(values, 0.75)),
            "p90": float(np.quantile(values, 0.90)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    return {
        "games": int(len(margin)),
        "win_rate": float(np.mean(win)),
        "tie_rate": float(np.mean(tie)),
        "loss_rate": float(np.mean(loss)),
        "non_win_rate": float(np.mean(~win)),
        "overall_margin": stats(margin),
        "winning_margin": stats(margin[win]),
        "losing_margin_absolute": stats(-margin[loss]),
        "own_cash_when_winning": stats(own[win]),
        "own_cash_when_losing": stats(own[loss]),
    }


def expected_oracle_indices(
    state_id: np.ndarray, own: np.ndarray, rival: np.ndarray,
) -> np.ndarray:
    selected = []
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        score = np.where(
            own[indices] > rival[indices],
            1.0,
            np.where(own[indices] == rival[indices], 0.5, 0.0),
        ).mean(axis=1)
        own_mean = own[indices].mean(axis=1)
        margin_mean = own_mean - rival[indices].mean(axis=1)
        local = max(
            range(len(indices)),
            key=lambda index: (
                float(score[index]),
                float(margin_mean[index]),
                float(own_mean[index]),
                -index,
            ),
        )
        selected.append(int(indices[local]))
    return np.asarray(selected, dtype=np.int64)


def clairvoyant_arrays(
    state_id: np.ndarray, own: np.ndarray, rival: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected_own = []
    selected_rival = []
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        for future in range(own.shape[1]):
            local = max(
                range(len(indices)),
                key=lambda index: (
                    float(own[indices[index], future] > rival[indices[index], future]),
                    float(own[indices[index], future] - rival[indices[index], future]),
                    float(own[indices[index], future]),
                    -index,
                ),
            )
            selected_own.append(float(own[indices[local], future]))
            selected_rival.append(float(rival[indices[local], future]))
    return np.asarray(selected_own), np.asarray(selected_rival)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"])
        opponent = np.asarray(data["opponent"])
        own = np.asarray(data["future_own_cash"], dtype=np.float64)
        rival = np.asarray(data["future_opponent_cash"], dtype=np.float64)

    keep_indices = np.asarray([
        int(np.flatnonzero(state_id == state)[0]) for state in np.unique(state_id)
    ])
    expected_indices = expected_oracle_indices(state_id, own, rival)
    clairvoyant_own, clairvoyant_rival = clairvoyant_arrays(state_id, own, rival)

    payload = {
        "dataset": str(args.dataset),
        "states": int(len(np.unique(state_id))),
        "candidates": int(len(state_id)),
        "future_count": int(own.shape[1]),
        "all_candidate_continuations": distribution(own, rival),
        "keep_continuations": distribution(own[keep_indices], rival[keep_indices]),
        "expected_oracle_one_candidate_per_state": distribution(
            own[expected_indices], rival[expected_indices]
        ),
        "clairvoyant_oracle_one_candidate_per_future": distribution(
            clairvoyant_own, clairvoyant_rival
        ),
        "g001": {},
    }
    g001_states = {
        int(state) for state in np.unique(state_id)
        if str(opponent[np.flatnonzero(state_id == state)[0]]) == "G001"
    }
    g001_rows = np.flatnonzero(np.isin(state_id, list(g001_states)))
    g001_keep = np.asarray([
        int(np.flatnonzero(state_id == state)[0]) for state in sorted(g001_states)
    ])
    g001_expected_all = expected_oracle_indices(
        state_id[g001_rows], own[g001_rows], rival[g001_rows]
    )
    g001_expected = g001_rows[g001_expected_all]
    g001_clairvoyant_own, g001_clairvoyant_rival = clairvoyant_arrays(
        state_id[g001_rows], own[g001_rows], rival[g001_rows]
    )
    payload["g001"] = {
        "states": len(g001_states),
        "all_candidate_continuations": distribution(own[g001_rows], rival[g001_rows]),
        "keep_continuations": distribution(own[g001_keep], rival[g001_keep]),
        "expected_oracle_one_candidate_per_state": distribution(
            own[g001_expected], rival[g001_expected]
        ),
        "clairvoyant_oracle_one_candidate_per_future": distribution(
            g001_clairvoyant_own, g001_clairvoyant_rival
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
