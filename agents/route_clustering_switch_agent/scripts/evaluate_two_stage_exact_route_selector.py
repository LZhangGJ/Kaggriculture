#!/usr/bin/env python3
"""Evaluate a frozen two-stage selector on aligned, optionally pruned matrices."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from train_compact_route_q_selector import load_node, policy_metrics


def _lookup(
    states: np.ndarray, actions: np.ndarray, targets: np.ndarray,
) -> dict[bytes, str]:
    return {
        state.tobytes(): str(targets[int(action)])
        for state, action in zip(states, actions)
    }


def _selected(
    node: dict, families: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = {family: index for index, family in enumerate(node["targets"])}
    missing = sorted(set(families.tolist()) - set(indices))
    if missing:
        raise KeyError(f"evaluation matrix lacks selected target: {missing[0]}")
    chosen = np.asarray([indices[str(value)] for value in families], dtype=np.int32)
    rows = np.arange(len(chosen))
    return node["scores"][rows, chosen], node["margins"][rows, chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--early-search", type=Path, required=True)
    parser.add_argument("--late-search", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.model, allow_pickle=False) as saved:
        opening = str(saved["opening"])
        early_checkpoint = int(saved["early_checkpoint"])
        late_checkpoint = int(saved["late_checkpoint"])
        early_lookup = _lookup(
            saved["early_states"], saved["early_actions"],
            saved["early_targets"].astype(str),
        )
        late_lookup = _lookup(
            saved["late_states"], saved["late_actions"],
            saved["late_targets"].astype(str),
        )
    early = load_node(args.early_search, early_checkpoint)
    late = load_node(args.late_search, late_checkpoint)
    for key in ("opening", "sample_shape"):
        if early[key] != late[key] or (key == "opening" and early[key] != opening):
            raise ValueError(f"unaligned selector matrices: {key}")
    for key in ("opponents", "seeds", "groups"):
        if not np.array_equal(early[key], late[key]):
            raise ValueError(f"unaligned selector matrices: {key}")

    early_families = np.asarray([
        early_lookup.get(row.tobytes(), opening) for row in early["features"]
    ], dtype=object)
    deferred = early_families == opening
    late_families = np.asarray([
        late_lookup.get(row.tobytes(), opening) for row in late["features"]
    ], dtype=object)
    final_families = early_families.copy()
    final_families[deferred] = late_families[deferred]
    early_scores, early_margins = _selected(early, np.where(
        deferred, opening, early_families
    ))
    late_scores, late_margins = _selected(late, late_families)
    scores = np.where(deferred, late_scores, early_scores)
    margins = np.where(deferred, late_margins, early_margins)
    metrics = policy_metrics(
        scores[:, None], margins[:, None], np.zeros(len(scores), dtype=np.int32),
        early["sample_shape"],
    )
    per_opponent = scores.reshape(early["sample_shape"]) == 1.0
    payload = {
        "schema": "two-stage-exact-public-state-route-selector-evaluation-v1",
        "model": str(args.model.resolve()),
        "early_search": str(args.early_search.resolve()),
        "late_search": str(args.late_search.resolve()),
        "checkpoints": [early_checkpoint, late_checkpoint],
        "seed_count": len(early["seeds"]),
        "selector": metrics,
        "opponent_raw_win_rates": {
            str(name): float(value)
            for name, value in zip(
                early["opponents"], per_opponent.mean(axis=(1, 2))
            )
        },
        "early_seen_state_rate": float(np.mean([
            row.tobytes() in early_lookup for row in early["features"]
        ])),
        "late_seen_state_rate_on_deferred": float(np.mean([
            row.tobytes() in late_lookup
            for row in late["features"][deferred]
        ])),
        "early_prediction_counts": dict(Counter(early_families.tolist())),
        "final_prediction_counts": dict(Counter(final_families.tolist())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: payload[key] for key in (
            "schema", "checkpoints", "seed_count", "selector",
            "early_seen_state_rate", "late_seen_state_rate_on_deferred",
            "early_prediction_counts", "final_prediction_counts",
        )
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
