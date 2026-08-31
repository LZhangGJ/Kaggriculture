#!/usr/bin/env python3
"""Evaluate a frozen true two-switch selector on an aligned holdout matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from train_compact_route_q_selector import policy_metrics


def _lookup(states: np.ndarray, actions: np.ndarray) -> dict[bytes, int]:
    return {
        np.ascontiguousarray(state, dtype=np.float32).tobytes(): int(action)
        for state, action in zip(states, actions)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.model, allow_pickle=False) as model:
        opening = str(model["opening"])
        early_targets = model["early_targets"].astype(str)
        late_targets = model["late_targets"].astype(str)
        early_lookup = _lookup(model["early_states"], model["early_actions"])
        late_states = model["late_states"]
        late_actions = model["late_actions"]
        offsets = model["late_offsets"].astype(int)
        late_lookups = [
            _lookup(late_states[offsets[i]:offsets[i + 1]],
                    late_actions[offsets[i]:offsets[i + 1]])
            for i in range(len(early_targets))
        ]
    with np.load(args.matrix, allow_pickle=False) as saved:
        if not np.array_equal(early_targets, saved["early_targets"].astype(str)):
            raise ValueError("early targets do not match the model")
        if not np.array_equal(late_targets, saved["late_targets"].astype(str)):
            raise ValueError("late targets do not match the model")
        early_features = saved["early_states"].reshape(
            -1, saved["early_states"].shape[-1]
        )
        per_early_features = saved["late_states"].reshape(
            len(early_targets), -1, saved["late_states"].shape[-1]
        )
        outcome = saved["outcome"].reshape(
            len(early_targets), len(late_targets), -1
        )
        margins = saved["margin"].reshape(outcome.shape).astype(np.float32)
        seeds = saved["seeds"].astype(np.int64)

    opening_action = int(np.flatnonzero(early_targets == opening)[0])
    early_seen = np.asarray([
        row.tobytes() in early_lookup for row in early_features
    ])
    early_predictions = np.asarray([
        early_lookup.get(row.tobytes(), opening_action) for row in early_features
    ], dtype=np.int32)
    late_predictions = np.zeros(len(early_features), dtype=np.int32)
    late_seen = np.zeros(len(early_features), dtype=bool)
    for row, early in enumerate(early_predictions):
        state = np.ascontiguousarray(
            per_early_features[early, row], dtype=np.float32
        ).tobytes()
        late_seen[row] = state in late_lookups[early]
        late_predictions[row] = late_lookups[early].get(state, 0)
    rows = np.arange(len(early_features))
    scores = outcome.astype(np.float32) * 0.5
    selected_scores = scores[early_predictions, late_predictions, rows]
    selected_margins = margins[early_predictions, late_predictions, rows]
    metrics = policy_metrics(
        selected_scores[:, None], selected_margins[:, None],
        np.zeros(len(rows), dtype=np.int32), (1, len(seeds), 2),
    )
    paired = selected_scores.reshape(len(seeds), 2)
    payload = {
        "schema": "true-two-switch-exact-selector-evaluation-v1",
        "model": str(args.model.resolve()),
        "matrix": str(args.matrix.resolve()),
        "seeds": seeds.astype(int).tolist(),
        "games": len(rows),
        "raw_win_rate": metrics["raw_win_rate"],
        "seat_win_rates": [
            float(selected_scores[0::2].mean()),
            float(selected_scores[1::2].mean()),
        ],
        "both_seats_win_rate": float(np.all(paired == 1.0, axis=1).mean()),
        "paired_seed_raw_win_lower_95pct": metrics[
            "paired_seed_raw_win_lower_95pct"
        ],
        "mean_margin": metrics["mean_margin"],
        "minimum_margin": metrics["minimum_margin"],
        "completion_rate": float(np.isfinite(selected_margins).mean()),
        "early_seen_state_rate": float(early_seen.mean()),
        "late_seen_state_rate": float(late_seen.mean()),
        "early_target_counts": dict(Counter(early_targets[early_predictions].tolist())),
        "late_action_counts": dict(Counter(late_targets[late_predictions].tolist())),
        "hidden_pair_oracle_raw_win_rate": float(
            np.max(outcome == 2, axis=(0, 1)).mean()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
